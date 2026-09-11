"""FaustWasm artifact and production renderer contracts."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from hydra import compose, initialize_config_module

from synth_setter.data.vst.core import extract_backend_version
from synth_setter.data.vst.faust_param_spec import resolve_faust_param_spec
from synth_setter.data.vst.faustwasm_contract import faustwasm_parameter_contract
from synth_setter.data.vst.faustwasm_renderer import FaustWasmRenderer
from synth_setter.data.vst.generate_vst_dataset import (
    AudioAmplitudeError,
    _reject_clipped_audio,
)
from synth_setter.data.vst.param_spec import CategoricalParameter, ContinuousParameter
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.renderer_factory import make_audio_renderer
from synth_setter.synth_spec import SYNTHS, SynthName

_NODE_UNAVAILABLE = shutil.which("node") is None
_EXPECTED_PARAMETER_ADDRESSES = {
    "faust_bright_organ": (
        ("/Sequencer/DSP1/brightOrgan/Main/volume", "/brightOrgan/Main/volume"),
        ("/Sequencer/DSP1/brightOrgan/Reverb/Amount", "/brightOrgan/Reverb/Amount"),
        ("/Sequencer/DSP1/brightOrgan/Reverb/Damp", "/brightOrgan/Reverb/Damp"),
        ("/Sequencer/DSP1/brightOrgan/Reverb/Size", "/brightOrgan/Reverb/Size"),
        ("/Sequencer/DSP1/brightOrgan/Stops/Fifteenth_2'", "/brightOrgan/Stops/Fifteenth_2-"),
        ("/Sequencer/DSP1/brightOrgan/Stops/Flute_8'", "/brightOrgan/Stops/Flute_8-"),
        ("/Sequencer/DSP1/brightOrgan/Stops/Foundation_8'", "/brightOrgan/Stops/Foundation_8-"),
        ("/Sequencer/DSP1/brightOrgan/Stops/Nasard_2_2/3'", "/brightOrgan/Stops/Nasard_2_2_3-"),
        ("/Sequencer/DSP1/brightOrgan/Stops/Principal_4'", "/brightOrgan/Stops/Principal_4-"),
        ("/Sequencer/DSP1/brightOrgan/Stops/Tierce_1_3/5'", "/brightOrgan/Stops/Tierce_1_3_5-"),
    ),
    "faust_bubble": (
        ("/bubble/Freeverb/0x00/Damp", "/bubble/Freeverb/0x00/Damp"),
        ("/bubble/Freeverb/0x00/RoomSize", "/bubble/Freeverb/0x00/RoomSize"),
        ("/bubble/Freeverb/0x00/Stereo_Spread", "/bubble/Freeverb/0x00/Stereo_Spread"),
        ("/bubble/Freeverb/Wet", "/bubble/Freeverb/Wet"),
        ("/bubble/bubble/freq", "/bubble/bubble/freq"),
        ("/bubble/drop", "/bubble/drop"),
    ),
    "faust_augmentor": (
        ("/augmentor/envelope/depth", "/augmentor/envelope_depth"),
        ("/augmentor/envelope/rate", "/augmentor/envelope_rate"),
        ("/augmentor/filter/cutoff", "/augmentor/filter_cutoff"),
        ("/augmentor/filter/mix", "/augmentor/filter_mix"),
        ("/augmentor/filter/resonance", "/augmentor/filter_resonance"),
        ("/augmentor/gate", "/augmentor/gate"),
        ("/augmentor/noise/amount", "/augmentor/noise_amount"),
        ("/augmentor/pitch/mix", "/augmentor/pitch_mix"),
        ("/augmentor/pitch/shift", "/augmentor/pitch_shift"),
        ("/augmentor/reverse/mix", "/augmentor/reverse_mix"),
        ("/augmentor/source/freq", "/augmentor/source_freq"),
        ("/augmentor/source/gain", "/augmentor/source_gain"),
    ),
    "faust_church_organ": (
        ("/churchOrgan/Zita_Light/Dry/Wet_Mix", "/churchOrgan/Zita_Light/Wet_Dry_Mix"),
        ("/churchOrgan/Zita_Light/Level", "/churchOrgan/Zita_Light/Level"),
        ("/churchOrgan/freq", "/churchOrgan/freq"),
        ("/churchOrgan/gain", "/churchOrgan/gain"),
        ("/churchOrgan/gain_fundamental", "/churchOrgan/gain_fundamental"),
        ("/churchOrgan/gain_8ve_partial", "/churchOrgan/gain_8ve_partial"),
        ("/churchOrgan/gain_5th_partial", "/churchOrgan/gain_5th_partial"),
        ("/churchOrgan/gain_3d_partial", "/churchOrgan/gain_3d_partial"),
        ("/churchOrgan/gain_other_partials", "/churchOrgan/gain_other_partials"),
        ("/churchOrgan/gain_lower_octave", "/churchOrgan/gain_lower_octave"),
        ("/churchOrgan/noise_gain", "/churchOrgan/noise_gain"),
        ("/churchOrgan/gate", "/churchOrgan/gate"),
    ),
    "faust_filter_osc": (
        ("/SINE_WAVE_OSCILLATOR_oscrs/Amplitude", "/SINE_WAVE_OSCILLATOR_oscrs/Amplitude"),
        ("/SINE_WAVE_OSCILLATOR_oscrs/Frequency", "/SINE_WAVE_OSCILLATOR_oscrs/Frequency"),
        ("/SINE_WAVE_OSCILLATOR_oscrs/Portamento", "/SINE_WAVE_OSCILLATOR_oscrs/Portamento"),
    ),
}


def _configured_backend_version() -> str:
    """Return the authored FaustWasm package pin.

    :returns: Configured backend version.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        return str(compose(config_name="render/faustwasm").render.backend_version)


def _config(identity: str = "faust_bright_organ", channels: int = 2) -> RenderConfig:
    return RenderConfig(
        synth=SYNTHS[SynthName(identity)],
        renderer_backend="faustwasm",
        backend_version=_configured_backend_version(),
        block_size=64,
        render_contract_version=2,
        sample_rate=44_100,
        channels=channels,
        velocity=100,
        signal_duration_seconds=4.0,
        min_loudness=-100.0,
        samples_per_render_batch=1,
        samples_per_shard=1,
        plugin_reload_cadence="render",
        gui_toggle_cadence="never",
    )


def _midpoint_patch(identity: str) -> dict[str, float]:
    spec = resolve_faust_param_spec(ParamSpecName(identity))
    patch: dict[str, float] = {}
    for parameter in spec.synth_params:
        if isinstance(parameter, ContinuousParameter):
            patch[parameter.name] = (parameter.min + parameter.max) / 2.0
        elif isinstance(parameter, CategoricalParameter):
            patch[parameter.name] = float(parameter.raw_values[-1])
        else:
            raise TypeError(type(parameter).__name__)
    return patch


@pytest.mark.parametrize("identity", _EXPECTED_PARAMETER_ADDRESSES)
def test_faustwasm_contract_maps_every_parameter_to_exact_compiler_address(identity: str) -> None:
    """Pin the complete canonical-to-compiler address contract independently.

    :param identity: Checked-in Faust program identity.
    """
    contract = faustwasm_parameter_contract(ParamSpecName(identity))

    assert (
        tuple((item.canonical_address, item.wasm_address) for item in contract)
        == _EXPECTED_PARAMETER_ADDRESSES[identity]
    )


def test_faustwasm_contract_covers_every_canonical_parameter_once() -> None:
    """Every canonical control occurs once in each compiler mapping."""
    for identity in _EXPECTED_PARAMETER_ADDRESSES:
        spec = resolve_faust_param_spec(ParamSpecName(identity))
        contract = faustwasm_parameter_contract(ParamSpecName(identity))
        assert [item.canonical_address for item in contract] == spec.synth_param_names


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
def test_faustwasm_backend_version_reads_packaged_metadata() -> None:
    """Backend provenance matches the bundled runtime metadata."""
    assert extract_backend_version("faustwasm") == _configured_backend_version()


def test_faustwasm_legacy_digest_projection_is_rejected() -> None:
    """FaustWasm rejects the legacy plugin-oriented render contract."""
    values = _config().model_dump()
    values["render_contract_version"] = 1
    with pytest.raises(ValueError, match="render_contract_version=1"):
        RenderConfig.model_validate(values)


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
@pytest.mark.parametrize(
    ("identity", "channels"),
    [
        ("faust_augmentor", 2),
        ("faust_bright_organ", 2),
        ("faust_bubble", 2),
        ("faust_church_organ", 2),
        ("faust_filter_osc", 1),
    ],
)
def test_faustwasm_factory_renders_real_source(identity: str, channels: int) -> None:
    """The factory compiles and renders every checked-in Faust program.

    :param identity: Checked-in Faust program identity.
    :param channels: Expected native output channel count.
    """
    renderer = make_audio_renderer(_config(identity, channels))
    params = _midpoint_patch(identity)
    if identity == "faust_bubble":
        params["/bubble/drop"] = 1.0
    if identity == "faust_church_organ":
        params["/churchOrgan/gate"] = 1.0
        params["/churchOrgan/gain"] = 0.2

    audio = renderer.render(params, 60, 100, (0.05, 0.3))

    assert isinstance(renderer, FaustWasmRenderer)
    assert renderer.block_size == 64
    assert audio.shape == (channels, 176_400)
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()
    assert float(np.max(np.abs(audio))) > 1e-4


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
def test_faustwasm_note_off_changes_audio_at_exact_requested_frame() -> None:
    """A note-off event first affects audio at its requested sample."""
    renderer = make_audio_renderer(_config())
    params = _midpoint_patch("faust_bright_organ")

    released = renderer.render(params, 60, 100, (0.1, 0.25))
    sustained = renderer.render(params, 60, 100, (0.1, 0.35))
    differing_frames = np.flatnonzero(np.any(released != sustained, axis=0))

    assert np.max(np.abs(released[:, :4_410])) == 0.0
    assert differing_frames[0] == 11_025


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
def test_faustwasm_patch_state_is_isolated_across_a_b_a_renders() -> None:
    """An intervening patch cannot contaminate a repeated render."""
    patch_a = _midpoint_patch("faust_bright_organ")
    patch_b = dict(patch_a)
    patch_a["/Sequencer/DSP1/brightOrgan/Main/volume"] = 0.25
    patch_b["/Sequencer/DSP1/brightOrgan/Main/volume"] = 1.0
    renderer = make_audio_renderer(_config())

    first_a = renderer.render(patch_a, 60, 100, (0.1, 0.25))
    rendered_b = renderer.render(patch_b, 60, 100, (0.1, 0.25))
    second_a = renderer.render(patch_a, 60, 100, (0.1, 0.25))

    assert not np.array_equal(first_a, rendered_b)
    assert np.array_equal(first_a, second_a)


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
def test_faustwasm_bright_organ_midi_octave_doubles_dominant_frequency() -> None:
    """Bright organ follows MIDI pitch across one octave."""
    renderer = make_audio_renderer(_config())
    params = _midpoint_patch("faust_bright_organ")

    low = renderer.render(params, 48, 100, (0.1, 0.35))[0, 4_410:15_435]
    high = renderer.render(params, 60, 100, (0.1, 0.35))[0, 4_410:15_435]
    frequencies = np.fft.rfftfreq(low.size, d=1.0 / 44_100)
    window = np.hanning(low.size)
    low_dominant = frequencies[np.argmax(np.abs(np.fft.rfft(low * window)))]
    high_dominant = frequencies[np.argmax(np.abs(np.fft.rfft(high * window)))]

    assert high_dominant / low_dominant == pytest.approx(2.0, rel=0.02)


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
@pytest.mark.parametrize(
    ("identity", "channels"),
    [
        ("faust_augmentor", 2),
        ("faust_bubble", 2),
        ("faust_church_organ", 2),
        ("faust_filter_osc", 1),
    ],
)
def test_faustwasm_mono_source_is_independent_of_midi_pitch(
    identity: str,
    channels: int,
) -> None:
    """Mono programs use canonical controls rather than MIDI pitch.

    :param identity: Checked-in monophonic Faust program identity.
    :param channels: Expected native output channel count.
    """
    renderer = make_audio_renderer(_config(identity, channels))
    params = _midpoint_patch(identity)
    if identity == "faust_bubble":
        params["/bubble/drop"] = 1.0
    if identity == "faust_church_organ":
        params["/churchOrgan/gate"] = 1.0
        params["/churchOrgan/gain"] = 0.2

    low = renderer.render(params, 48, 100, (0.05, 0.3))
    high = renderer.render(params, 72, 100, (0.05, 0.3))

    assert float(np.max(np.abs(low))) > 1e-4
    assert np.array_equal(low, high)


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
def test_faustwasm_filter_osc_preserves_clipping_for_dataset_rejection() -> None:
    """Native clipping remains visible to the dataset amplitude gate."""
    params = _midpoint_patch("faust_filter_osc")
    params["/SINE_WAVE_OSCILLATOR_oscrs/Amplitude"] = 10.0
    renderer = make_audio_renderer(_config("faust_filter_osc", 1))

    audio = renderer.render(params, 60, 100, (0.05, 0.3))

    assert float(np.max(np.abs(audio))) > 1.0
    with pytest.raises(AudioAmplitudeError, match=r"within \[-1, 1\]"):
        _reject_clipped_audio(audio)


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
def test_faustwasm_incomplete_patch_is_rejected() -> None:
    """Rendering rejects a patch missing a canonical control."""
    renderer = make_audio_renderer(_config())
    params = _midpoint_patch("faust_bright_organ")
    params.pop(next(iter(params)))

    with pytest.raises(ValueError, match="missing canonical parameter"):
        renderer.render(params, 60, 100, (0.05, 0.3))


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
def test_faustwasm_out_of_domain_patch_is_rejected() -> None:
    """Rendering rejects canonical values outside their native domain."""
    renderer = make_audio_renderer(_config())
    params = _midpoint_patch("faust_bright_organ")
    params["/Sequencer/DSP1/brightOrgan/Main/volume"] = 2.0

    with pytest.raises(ValueError, match="outside native domain"):
        renderer.render(params, 60, 100, (0.05, 0.3))


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
def test_faustwasm_canonical_volume_has_causal_effect() -> None:
    """The canonical volume control changes real rendered level."""
    quiet = _midpoint_patch("faust_bright_organ")
    loud = dict(quiet)
    quiet["/Sequencer/DSP1/brightOrgan/Main/volume"] = 0.0
    loud["/Sequencer/DSP1/brightOrgan/Main/volume"] = 1.0
    renderer = make_audio_renderer(_config())

    quiet_audio = renderer.render(quiet, 60, 100, (0.05, 0.3))
    loud_audio = renderer.render(loud, 60, 100, (0.05, 0.3))

    assert float(np.max(np.abs(quiet_audio))) < 1e-4
    assert float(np.max(np.abs(loud_audio))) > 1e-4
