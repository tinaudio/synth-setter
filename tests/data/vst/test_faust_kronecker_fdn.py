"""Behavior contracts for the impulse-excited Faust Kronecker FDN."""

from __future__ import annotations

import platform
import sys

import numpy as np
import pytest

from synth_setter.data.pyfdn_param_spec import kronecker_feedback_matrix
from synth_setter.data.vst.faust_param_spec import resolve_faust_param_spec
from synth_setter.data.vst.faust_sources import resolve_faust_dsp
from synth_setter.data.vst.param_spec import CategoricalParameter
from synth_setter.data.vst.renderers import AudioRenderer
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.renderer_factory import make_audio_renderer
from synth_setter.synth_spec import SYNTHS, SynthName

IDENTITY = "faust_kronecker_fdn"
# Faust reports UI controls in traversal order: decay, delays, input gains,
# kernel angles, reflect flags, output gains, dry.
EXPECTED_SYNTH_PARAM_NAMES = (
    ("/kroneckerFDN/Decay/t60",)
    + tuple(f"/kroneckerFDN/Delays/d{i}" for i in range(8))
    + tuple(f"/kroneckerFDN/Input/b{i}" for i in range(8))
    + tuple(f"/kroneckerFDN/Kernel/a{i}" for i in range(3))
    + tuple(f"/kroneckerFDN/Kernel/r{i}" for i in range(3))
    + tuple(f"/kroneckerFDN/Output/c{i}" for i in range(8))
    + ("/kroneckerFDN/Output/dry",)
)
EXPECTED_ENCODED_WIDTH = 38
_REFLECT_OFFSET = 20
_DELAYS = (601, 773, 839, 911, 997, 1063, 1129, 1181)
_RENDER_SECONDS = 0.5
_MIDI_NOTE = 60
_MIDI_VELOCITY = 100
_NOTE_WINDOW = (0.0, 0.25)
_MIN_AUDIBLE_PEAK = 1e-4

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"},
    reason="DawDreamer Faust wheels support Linux x86_64",
)


def _audible_patch() -> dict[str, float]:
    """Return a bounded native patch with an audible impulse tail.

    :returns: Complete exact-address native parameter mapping.
    """
    patch = dict.fromkeys(EXPECTED_SYNTH_PARAM_NAMES, 0.0)
    for index, delay in enumerate(_DELAYS):
        patch[f"/kroneckerFDN/Delays/d{index}"] = float(delay)
        patch[f"/kroneckerFDN/Input/b{index}"] = 0.5
        patch[f"/kroneckerFDN/Output/c{index}"] = 0.125
    for index in range(3):
        patch[f"/kroneckerFDN/Kernel/a{index}"] = np.pi / 4
        patch[f"/kroneckerFDN/Kernel/r{index}"] = 1.0
    patch["/kroneckerFDN/Decay/t60"] = 1.5
    return patch


def _renderer() -> AudioRenderer:
    """Build a short mono production-shaped Kronecker render configuration.

    :returns: Audio renderer for the registered Kronecker identity.
    """
    config = RenderConfig(
        synth=SYNTHS[SynthName(IDENTITY)],
        renderer_backend="dawdreamer",
        backend_version="0.8.3",
        sample_rate=44100,
        channels=1,
        velocity=_MIDI_VELOCITY,
        signal_duration_seconds=_RENDER_SECONDS,
        min_loudness=-55.0,
        samples_per_render_batch=1,
        samples_per_shard=1,
        plugin_reload_cadence="render",
        gui_toggle_cadence="never",
    )
    return make_audio_renderer(config)


def _numpy_impulse_response(frames: int) -> np.ndarray:
    """Simulate the pinned patch through the reference Kronecker loop.

    Faust's `~` contributes one implicit sample of delay on the feedback path,
    so feedback reads the previous delay outputs.

    :param frames: Impulse-response length in samples.
    :returns: Mono reference response shaped ``(frames,)``.
    """
    feedback = kronecker_feedback_matrix(
        np.full(3, np.pi / 4), np.ones(3, dtype=np.int64)
    )
    gains = 0.001 ** (np.asarray(_DELAYS) / (1.5 * 44100))
    excitation = np.zeros(frames)
    excitation[0] = 1.0
    states = np.zeros(8)
    previous = np.zeros(8)
    lines = [np.zeros(delay) for delay in _DELAYS]
    heads = [0] * 8
    response = np.zeros(frames)
    for frame in range(frames):
        recirculated = feedback @ previous
        for index in range(8):
            states[index] = lines[index][heads[index]]
            lines[index][heads[index]] = (
                0.5 * excitation[frame] + gains[index] * recirculated[index]
            )
            heads[index] = (heads[index] + 1) % _DELAYS[index]
        response[frame] = 0.125 * states.sum()
        previous = states.copy()
    return response


def test_kronecker_fdn_identity_is_a_mono_impulse_source() -> None:
    """The Kronecker identity needs no MIDI voices and emits one channel."""
    dsp = resolve_faust_dsp(ParamSpecName(IDENTITY))
    spec = resolve_faust_param_spec(ParamSpecName(IDENTITY))

    assert dsp.num_voices == 0
    assert dsp.outputs == 1
    assert spec.synth_param_names == list(EXPECTED_SYNTH_PARAM_NAMES)
    assert spec.encoded_width == EXPECTED_ENCODED_WIDTH


def test_kronecker_fdn_reflect_flags_use_discrete_onehot_domain() -> None:
    """Each rotate/reflect flag encodes only its two native states."""
    spec = resolve_faust_param_spec(ParamSpecName(IDENTITY))

    for index in range(3):
        flag = spec.synth_params[_REFLECT_OFFSET + index]
        assert flag.name == f"/kroneckerFDN/Kernel/r{index}"
        assert isinstance(flag, CategoricalParameter)
        assert flag.raw_values == [0.0, 1.0]
        assert flag.encoding == "onehot"


def test_kronecker_fdn_renders_audible_bounded_impulse_response() -> None:
    """One impulse render is finite, audible, and dataset-safe."""
    renderer = _renderer()

    audio = renderer.render(_audible_patch(), _MIDI_NOTE, _MIDI_VELOCITY, _NOTE_WINDOW)

    assert audio.shape == (1, int(44100 * _RENDER_SECONDS))
    assert np.isfinite(audio).all()
    peak = float(np.max(np.abs(audio)))
    assert peak > _MIN_AUDIBLE_PEAK
    assert peak <= 1.0


def test_kronecker_fdn_matches_reference_kronecker_loop() -> None:
    """The compiled DSP reproduces the reference feedback matrix exactly."""
    renderer = _renderer()

    audio = renderer.render(_audible_patch(), _MIDI_NOTE, _MIDI_VELOCITY, _NOTE_WINDOW)

    assert np.allclose(audio[0, :4096], _numpy_impulse_response(4096), atol=1e-5)


def test_kronecker_fdn_repeated_renders_are_identical() -> None:
    """Fresh processor state keeps the impulse response deterministic."""
    renderer = _renderer()
    patch = _audible_patch()

    first = renderer.render(patch, _MIDI_NOTE, _MIDI_VELOCITY, _NOTE_WINDOW)
    second = renderer.render(patch, _MIDI_NOTE, _MIDI_VELOCITY, _NOTE_WINDOW)

    assert np.array_equal(first, second)


def test_kronecker_fdn_kernel_angle_is_causal_on_the_tail() -> None:
    """A kernel angle change must reshape the rendered tail."""
    renderer = _renderer()
    baseline = _audible_patch()
    detuned = dict(baseline)
    detuned["/kroneckerFDN/Kernel/a0"] = 0.0

    original = renderer.render(baseline, _MIDI_NOTE, _MIDI_VELOCITY, _NOTE_WINDOW)
    changed = renderer.render(detuned, _MIDI_NOTE, _MIDI_VELOCITY, _NOTE_WINDOW)

    assert not np.allclose(original, changed, rtol=0.0, atol=1e-6)


def test_kronecker_fdn_reflect_flag_is_causal_on_the_tail() -> None:
    """A rotate/reflect change must reshape the rendered tail."""
    renderer = _renderer()
    baseline = _audible_patch()
    reflected = dict(baseline)
    reflected["/kroneckerFDN/Kernel/r1"] = 0.0

    original = renderer.render(baseline, _MIDI_NOTE, _MIDI_VELOCITY, _NOTE_WINDOW)
    changed = renderer.render(reflected, _MIDI_NOTE, _MIDI_VELOCITY, _NOTE_WINDOW)

    assert not np.allclose(original, changed, rtol=0.0, atol=1e-6)


def test_kronecker_fdn_ignores_midi_pitch() -> None:
    """MIDI conditioning is an unused pipeline stub for the impulse source."""
    renderer = _renderer()
    patch = _audible_patch()

    low = renderer.render(patch, 48, _MIDI_VELOCITY, _NOTE_WINDOW)
    high = renderer.render(patch, 72, _MIDI_VELOCITY, _NOTE_WINDOW)

    assert np.array_equal(low, high)
