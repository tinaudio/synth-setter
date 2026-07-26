"""Real DawDreamer Faust renderer contracts."""

from __future__ import annotations

import platform
import sys
from typing import Literal

import numpy as np
import pytest

from synth_setter.data.vst.faust_param_spec import resolve_faust_param_spec
from synth_setter.data.vst.generate_vst_dataset import (
    AudioAmplitudeError,
    _reject_clipped_audio,
)
from synth_setter.data.vst.param_spec import CategoricalParameter, ContinuousParameter
from synth_setter.data.vst.renderers import DawDreamerFaustRenderer
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.renderer_factory import make_audio_renderer
from synth_setter.synth_spec import SynthName, SynthSpec

_SAMPLE_RATE = 44100
_RENDER_SECONDS = 0.5
_MIDI_NOTE = 60
_MIDI_VELOCITY = 100
_NOTE_WINDOW = (0.05, 0.3)
_MIN_AUDIBLE_PEAK = 1e-4

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"},
    reason="DawDreamer Faust wheels support Linux x86_64",
)


def _render_config(
    *,
    param_spec_name: str = "faust_bright_organ",
    channels: int = 2,
    reload_cadence: Literal["once", "render"] = "once",
) -> RenderConfig:
    """Build a production-shaped brightOrgan render configuration.

    :param param_spec_name: Checked-in Faust source/spec identity.
    :param channels: Native output channel count.
    :param reload_cadence: Native processor lifecycle policy.
    :returns: Validated Faust render configuration.
    """
    return RenderConfig(
        synth=SynthSpec(
            name=SynthName(param_spec_name),
            param_spec_name=ParamSpecName(param_spec_name),
            plugin_path="faust",
            plugin_state_path="",
        ),
        renderer_version="0.8.3",
        renderer_backend="dawdreamer_faust",
        sample_rate=_SAMPLE_RATE,
        channels=channels,
        velocity=_MIDI_VELOCITY,
        signal_duration_seconds=_RENDER_SECONDS,
        min_loudness=-55.0,
        samples_per_render_batch=1,
        samples_per_shard=1,
        plugin_reload_cadence=reload_cadence,
        gui_toggle_cadence="never",
    )


def _midpoint_params(param_spec_name: str) -> dict[str, float]:
    """Return an audible native-domain patch for one Faust identity.

    :param param_spec_name: Registered Faust parameter-spec identity.
    :returns: Complete exact-address native parameter mapping.
    :raises TypeError: A registered synth control is not continuous or categorical.
    """
    params: dict[str, float] = {}
    for parameter in resolve_faust_param_spec(
        ParamSpecName(param_spec_name)
    ).synth_params:
        if isinstance(parameter, ContinuousParameter):
            params[parameter.name] = (parameter.min + parameter.max) / 2.0
        elif isinstance(parameter, CategoricalParameter):
            params[parameter.name] = float(parameter.raw_values[-1])
        else:
            raise TypeError(type(parameter).__name__)
    return params


@pytest.mark.parametrize(
    ("param_spec_name", "channels"),
    [
        ("faust_bright_organ", 2),
        ("faust_bubble", 2),
        ("faust_church_organ", 2),
        ("faust_filter_osc", 1),
    ],
)
def test_factory_renders_real_checked_in_faust_source(
    param_spec_name: str,
    channels: int,
) -> None:
    """The shared factory compiles source, applies exact addresses, and renders MIDI.

    :param param_spec_name: Checked-in Faust source/spec identity.
    :param channels: Native output channel count.
    """
    renderer = make_audio_renderer(
        _render_config(param_spec_name=param_spec_name, channels=channels)
    )

    audio = renderer.render(
        _midpoint_params(param_spec_name),
        _MIDI_NOTE,
        _MIDI_VELOCITY,
        _NOTE_WINDOW,
    )

    assert isinstance(renderer, DawDreamerFaustRenderer)
    assert audio.shape == (channels, int(_SAMPLE_RATE * _RENDER_SECONDS))
    assert np.isfinite(audio).all()
    assert float(np.max(np.abs(audio))) > _MIN_AUDIBLE_PEAK


def test_factory_rejects_mismatched_faust_renderer_version() -> None:
    """Faust provenance must match the pinned DawDreamer runtime."""
    config = _render_config().model_copy(update={"renderer_version": "9.9.9"})

    with pytest.raises(ValueError, match="DawDreamer renderer version"):
        make_audio_renderer(config)


def test_bright_organ_volume_changes_real_rendered_level() -> None:
    """An exact-address causal control changes native rendered audio."""
    volume_address = "/Sequencer/DSP1/brightOrgan/Main/volume"
    quiet_params = _midpoint_params("faust_bright_organ")
    loud_params = dict(quiet_params)
    quiet_params[volume_address] = 0.0
    loud_params[volume_address] = 1.0
    quiet_renderer = make_audio_renderer(_render_config())
    loud_renderer = make_audio_renderer(_render_config())

    quiet = quiet_renderer.render(
        quiet_params, _MIDI_NOTE, _MIDI_VELOCITY, _NOTE_WINDOW
    )
    loud = loud_renderer.render(loud_params, _MIDI_NOTE, _MIDI_VELOCITY, _NOTE_WINDOW)

    assert float(np.max(np.abs(quiet))) < _MIN_AUDIBLE_PEAK
    assert float(np.max(np.abs(loud))) > _MIN_AUDIBLE_PEAK


def test_faust_clipping_is_rejected_at_dataset_generation_boundary() -> None:
    """Renderers preserve native output while dataset generation rejects clipping."""
    amplitude_address = "/SINE_WAVE_OSCILLATOR_oscrs/Amplitude"
    params = _midpoint_params("faust_filter_osc")
    params[amplitude_address] = 10.0
    renderer = make_audio_renderer(
        _render_config(param_spec_name="faust_filter_osc", channels=1)
    )

    audio = renderer.render(params, _MIDI_NOTE, _MIDI_VELOCITY, _NOTE_WINDOW)

    assert float(np.max(np.abs(audio))) > 1.0
    with pytest.raises(AudioAmplitudeError, match=r"within \[-1, 1\]"):
        _reject_clipped_audio(audio)


def test_faust_renderer_requires_complete_exact_address_patch() -> None:
    """Missing and unknown parameter addresses fail before native rendering."""
    renderer = make_audio_renderer(_render_config())
    params = _midpoint_params("faust_bright_organ")
    params.pop(next(iter(params)))

    with pytest.raises(ValueError, match="missing Faust parameter address"):
        renderer.render(
            params,
            _MIDI_NOTE,
            _MIDI_VELOCITY,
            _NOTE_WINDOW,
        )

    params = _midpoint_params("faust_bright_organ")
    params["/unregistered/control"] = 0.5
    with pytest.raises(KeyError, match="unknown Faust parameter address"):
        renderer.render(
            params,
            _MIDI_NOTE,
            _MIDI_VELOCITY,
            _NOTE_WINDOW,
        )


@pytest.mark.parametrize(
    ("reload_cadence", "engine_replaced"),
    [("once", False), ("render", True)],
)
def test_faust_renderer_honors_processor_reload_cadence(
    reload_cadence: Literal["once", "render"],
    engine_replaced: bool,
) -> None:
    """The configured lifecycle controls reuse of the compiled native graph.

    :param reload_cadence: Public lifecycle policy under test.
    :param engine_replaced: Whether the second render must compile a fresh graph.
    """
    renderer = make_audio_renderer(_render_config(reload_cadence=reload_cadence))
    assert isinstance(renderer, DawDreamerFaustRenderer)
    params = _midpoint_params("faust_bright_organ")
    renderer.render(params, _MIDI_NOTE, _MIDI_VELOCITY, _NOTE_WINDOW)
    first_engine = renderer.engine

    renderer.render(params, _MIDI_NOTE, _MIDI_VELOCITY, _NOTE_WINDOW)

    assert (renderer.engine is not first_engine) is engine_replaced
