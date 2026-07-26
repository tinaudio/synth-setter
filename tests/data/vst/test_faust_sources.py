"""Real Faust source, introspection, and audio-rendering contracts."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Protocol, TypedDict, cast

import numpy as np
import pytest

from synth_setter.data.vst.faust_sources import FaustDsp, faust_dsps, resolve_faust_dsp
from synth_setter.param_spec_name import ParamSpecName

_SAMPLE_RATE = 44100
_BLOCK_SIZE = 128
_RENDER_DURATION_SECONDS = 0.5
_MIDI_NOTE = 60
_MIDI_VELOCITY = 100
_NOTE_START_SECONDS = 0.05
_NOTE_DURATION_SECONDS = 0.25
_MIN_AUDIBLE_PEAK = 1e-4
_EXPECTED_PARAMETER_ADDRESSES: Mapping[str, list[str]] = {
    "faust_bright_organ": [
        "/Sequencer/DSP1/brightOrgan/Main/volume",
        "/Sequencer/DSP1/brightOrgan/Reverb/Amount",
        "/Sequencer/DSP1/brightOrgan/Reverb/Damp",
        "/Sequencer/DSP1/brightOrgan/Reverb/Size",
        "/Sequencer/DSP1/brightOrgan/Stops/Fifteenth_2'",
        "/Sequencer/DSP1/brightOrgan/Stops/Flute_8'",
        "/Sequencer/DSP1/brightOrgan/Stops/Foundation_8'",
        "/Sequencer/DSP1/brightOrgan/Stops/Nasard_2_2/3'",
        "/Sequencer/DSP1/brightOrgan/Stops/Principal_4'",
        "/Sequencer/DSP1/brightOrgan/Stops/Tierce_1_3/5'",
    ],
    "faust_bubble": [
        "/bubble/Freeverb/0x00/Damp",
        "/bubble/Freeverb/0x00/RoomSize",
        "/bubble/Freeverb/0x00/Stereo_Spread",
        "/bubble/Freeverb/Wet",
        "/bubble/bubble/freq",
        "/bubble/drop",
    ],
    "faust_church_organ": [
        "/churchOrgan/Zita_Light/Dry/Wet_Mix",
        "/churchOrgan/Zita_Light/Level",
        "/churchOrgan/freq",
        "/churchOrgan/gain",
        "/churchOrgan/gain_fundamental",
        "/churchOrgan/gain_8ve_partial",
        "/churchOrgan/gain_5th_partial",
        "/churchOrgan/gain_3d_partial",
        "/churchOrgan/gain_other_partials",
        "/churchOrgan/gain_lower_octave",
        "/churchOrgan/noise_gain",
        "/churchOrgan/gate",
    ],
    "faust_filter_osc": [
        "/SINE_WAVE_OSCILLATOR_oscrs/Amplitude",
        "/SINE_WAVE_OSCILLATOR_oscrs/Frequency",
        "/SINE_WAVE_OSCILLATOR_oscrs/Portamento",
    ],
}
_EXPECTED_OUTPUT_CHANNELS = {
    "faust_bright_organ": 2,
    "faust_bubble": 2,
    "faust_church_organ": 2,
    "faust_filter_osc": 1,
}
_RENDER_PARAMETER_OVERRIDES: Mapping[str, Mapping[str, float]] = {
    "faust_bright_organ": {},
    "faust_bubble": {"/bubble/drop": 1.0},
    "faust_church_organ": {
        "/churchOrgan/gain": 0.2,
        "/churchOrgan/gate": 1.0,
    },
    "faust_filter_osc": {},
}


class _ParameterDescription(TypedDict):
    """Faust UI identity returned by DawDreamer.

    .. attribute :: name

       Exact compiled parameter address.
    """

    name: str


class _FaustProcessor(Protocol):
    """Real DawDreamer Faust processor surface used by this contract test.

    .. attribute :: num_voices

       Compile-time polyphony count.
    """

    num_voices: int

    def set_dsp_string(self, source: str) -> bool:
        """Stage source for compilation.

        :param source: Complete Faust program.
        :returns: Whether DawDreamer accepted the source.
        """
        ...

    def compile(self) -> bool:
        """Compile the staged source.

        :returns: Whether compilation succeeded.
        """
        ...

    def get_parameters_description(self) -> list[_ParameterDescription]:
        """Return compiled Faust UI parameter identities.

        :returns: Compiled parameter descriptions in host order.
        """
        ...

    def set_parameter(self, address: str, value: float) -> bool:
        """Set one control by exact address.

        :param address: Compiled Faust parameter address.
        :param value: Native parameter value.
        :returns: Whether DawDreamer accepted the value.
        """
        ...

    def add_midi_note(
        self,
        note: int,
        velocity: int,
        start_time: float,
        duration: float,
    ) -> bool:
        """Schedule one MIDI note for a polyphonic source.

        :param note: MIDI pitch.
        :param velocity: MIDI velocity.
        :param start_time: Note start in seconds.
        :param duration: Note duration in seconds.
        :returns: Whether DawDreamer accepted the note.
        """
        ...


class _RenderEngine(Protocol):
    """Real DawDreamer render engine surface used by this contract test."""

    def make_faust_processor(self, name: str) -> _FaustProcessor:
        """Create a named Faust processor.

        :param name: Render-graph processor name.
        :returns: New Faust processor.
        """
        ...

    def load_graph(self, graph: list[tuple[_FaustProcessor, list[object]]]) -> None:
        """Load a processor graph.

        :param graph: Processor and input-edge pairs.
        """
        ...

    def render(self, duration: float) -> bool:
        """Render the graph for a duration.

        :param duration: Render duration in seconds.
        :returns: Whether DawDreamer completed the render.
        """
        ...

    def get_audio(self) -> np.ndarray:
        """Return rendered graph output.

        :returns: Channel-leading audio array.
        """
        ...


class _DawDreamerModule(Protocol):
    """Lazily imported DawDreamer module surface used by this contract test."""

    def RenderEngine(self, sample_rate: float, block_size: int) -> _RenderEngine:
        """Create a render engine.

        :param sample_rate: Audio sample rate in Hz.
        :param block_size: Engine processing block size in samples.
        :returns: New render engine.
        """
        ...


def _compile_faust(
    dd: _DawDreamerModule,
    param_spec_name: str,
) -> tuple[FaustDsp, _RenderEngine, _FaustProcessor]:
    """Compile one registered source with the real DawDreamer runtime.

    :param dd: Imported DawDreamer module.
    :param param_spec_name: Faust source registry key.
    :returns: Source metadata, engine, and compiled processor.
    """
    dsp = resolve_faust_dsp(ParamSpecName(param_spec_name))
    engine = dd.RenderEngine(_SAMPLE_RATE, _BLOCK_SIZE)
    processor = engine.make_faust_processor(param_spec_name)
    processor.num_voices = dsp.num_voices

    assert processor.set_dsp_string(dsp.source)
    assert processor.compile()
    return dsp, engine, processor


def test_faust_source_registry_resolves_checked_in_source_strings() -> None:
    """Every v1 registry key resolves to an in-memory Faust program."""
    assert set(faust_dsps) == set(_EXPECTED_PARAMETER_ADDRESSES)

    for name in faust_dsps:
        dsp = resolve_faust_dsp(ParamSpecName(name))
        assert isinstance(dsp, FaustDsp)
        assert 'import("stdfaust.lib");' in dsp.source
        assert not hasattr(dsp, "path")
        assert not hasattr(dsp, "uri")


def test_faust_source_registry_rejects_unknown_param_spec_name() -> None:
    """Unknown registry keys fail rather than resolving external content."""
    with pytest.raises(KeyError, match="missing"):
        resolve_faust_dsp(ParamSpecName("missing"))


def test_bright_organ_source_uses_one_polyphonic_voice_and_stereo_effect() -> None:
    """MIDI owns brightOrgan frequency/gate while its effect keeps stereo output."""
    dsp = resolve_faust_dsp(ParamSpecName("faust_bright_organ"))

    assert dsp.num_voices == 1
    assert "effect = _,_;" in dsp.source
    assert "process = output;" in dsp.source
    assert "process = output <: _,_;" not in dsp.source
    assert not any(
        name.endswith(("/freq", "/gate"))
        for name in _EXPECTED_PARAMETER_ADDRESSES["faust_bright_organ"]
    )


@pytest.mark.parametrize("param_spec_name", _EXPECTED_PARAMETER_ADDRESSES)
def test_faust_source_compiles_with_exact_parameter_addresses(
    param_spec_name: str,
) -> None:
    """DawDreamer compilation preserves one source's pinned UI identity.

    :param param_spec_name: Faust source registry key under test.
    """
    dd = cast(_DawDreamerModule, import_module("dawdreamer"))
    _, _, processor = _compile_faust(dd, param_spec_name)

    descriptions = processor.get_parameters_description()
    assert [item["name"] for item in descriptions] == _EXPECTED_PARAMETER_ADDRESSES[
        param_spec_name
    ]


@pytest.mark.parametrize("param_spec_name", _EXPECTED_PARAMETER_ADDRESSES)
def test_faust_source_renders_real_audio(param_spec_name: str) -> None:
    """One compiled source emits finite, audible, bounded audio of its native shape.

    :param param_spec_name: Faust source registry key under test.
    """
    dd = cast(_DawDreamerModule, import_module("dawdreamer"))
    dsp, engine, processor = _compile_faust(dd, param_spec_name)

    for address, value in _RENDER_PARAMETER_OVERRIDES[param_spec_name].items():
        assert processor.set_parameter(address, value)
    if dsp.num_voices:
        assert processor.add_midi_note(
            _MIDI_NOTE,
            _MIDI_VELOCITY,
            _NOTE_START_SECONDS,
            _NOTE_DURATION_SECONDS,
        )

    engine.load_graph([(processor, [])])
    assert engine.render(_RENDER_DURATION_SECONDS)
    audio = engine.get_audio()

    expected_samples = int(_SAMPLE_RATE * _RENDER_DURATION_SECONDS)
    assert audio.shape == (_EXPECTED_OUTPUT_CHANNELS[param_spec_name], expected_samples)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > _MIN_AUDIBLE_PEAK
    assert np.max(np.abs(audio)) <= 1.0
