"""Real Faust source, introspection, and audio-rendering contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, TypedDict, cast

import numpy as np
import pytest

from synth_setter.data.vst.faust_param_spec import resolve_faust_param_spec
from synth_setter.data.vst.faust_sources import FaustDsp, faust_dsps, resolve_faust_dsp
from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
    NoteDurationParameter,
    Parameter,
    decode_model_output,
)
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

    .. attribute :: min

       Native control minimum.

    .. attribute :: max

       Native control maximum.

    .. attribute :: isDiscrete

       Whether the native control has discrete states.
    """

    name: str
    min: float
    max: float
    isDiscrete: bool


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


def _expected_parameter_domain(parameter: Parameter) -> tuple[float, float, bool]:
    """Return the native domain represented by one Faust parameter.

    :param parameter: Exact-address Faust parameter.
    :returns: Native minimum, maximum, and discreteness.
    :raises TypeError: The parameter type cannot represent a Faust UI control.
    """
    if isinstance(parameter, ContinuousParameter):
        return parameter.min, parameter.max, False
    if isinstance(parameter, CategoricalParameter):
        raw_values = [float(value) for value in parameter.raw_values]
        return min(raw_values), max(raw_values), True
    raise TypeError(f"unsupported Faust parameter type {type(parameter).__name__}")


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


@pytest.mark.parametrize(
    ("param_spec_name", "encoded_width"),
    [
        ("faust_bright_organ", 13),
        ("faust_bubble", 10),
        ("faust_church_organ", 16),
        ("faust_filter_osc", 6),
    ],
)
def test_faust_param_spec_preserves_exact_addresses_and_encoded_width(
    param_spec_name: str,
    encoded_width: int,
) -> None:
    """Each source identity resolves to exact addresses and model width.

    :param param_spec_name: Faust source and parameter-spec registry key.
    :param encoded_width: Expected synth-and-note model width.
    """
    spec = resolve_faust_param_spec(ParamSpecName(param_spec_name))

    assert spec.synth_param_names == _EXPECTED_PARAMETER_ADDRESSES[param_spec_name]
    assert spec.encoded_width == encoded_width


@pytest.mark.parametrize(
    "param_spec_name",
    ["faust_bubble", "faust_church_organ"],
)
def test_faust_trigger_controls_use_discrete_onehot_domain(param_spec_name: str) -> None:
    """Monophonic trigger controls encode only their two native states.

    :param param_spec_name: Faust parameter spec whose final control is a trigger.
    """
    trigger = resolve_faust_param_spec(ParamSpecName(param_spec_name)).synth_params[-1]

    assert isinstance(trigger, CategoricalParameter)
    assert trigger.raw_values == [0.0, 1.0]
    assert trigger.encoding == "onehot"
    assert trigger.encode(0.0) == pytest.approx(np.array([1.0, 0.0]))
    assert trigger.encode(1.0) == pytest.approx(np.array([0.0, 1.0]))
    assert trigger.decode(np.array([1.0, 0.0])) == 0.0
    assert trigger.decode(np.array([0.0, 1.0])) == 1.0


@dataclass(frozen=True)
class _NativeDomainCase:
    """One exact-address native-domain expectation.

    .. attribute :: param_spec_name

       Faust parameter-spec identity.

    .. attribute :: address

       Exact compiled parameter address.

    .. attribute :: minimum

       Native minimum.

    .. attribute :: maximum

       Native maximum.
    """

    param_spec_name: str
    address: str
    minimum: float
    maximum: float


@pytest.mark.parametrize(
    "case",
    [
        _NativeDomainCase("faust_bubble", "/bubble/bubble/freq", 150.0, 2000.0),
        _NativeDomainCase(
            "faust_church_organ", "/churchOrgan/Zita_Light/Dry/Wet_Mix", -1.0, 1.0
        ),
        _NativeDomainCase(
            "faust_church_organ", "/churchOrgan/Zita_Light/Level", -70.0, 40.0
        ),
        _NativeDomainCase("faust_church_organ", "/churchOrgan/freq", 50.0, 1000.0),
        _NativeDomainCase(
            "faust_filter_osc", "/SINE_WAVE_OSCILLATOR_oscrs/Amplitude", -120.0, 10.0
        ),
        _NativeDomainCase(
            "faust_filter_osc", "/SINE_WAVE_OSCILLATOR_oscrs/Frequency", 1.0, 88.0
        ),
        _NativeDomainCase(
            "faust_filter_osc", "/SINE_WAVE_OSCILLATOR_oscrs/Portamento", 0.001, 10.0
        ),
    ],
)
def test_faust_continuous_controls_encode_complete_native_bounds(
    case: _NativeDomainCase,
) -> None:
    """Every non-unit Faust range round-trips through the model domain.

    :param case: Exact identity, address, and native bounds under test.
    """
    spec = resolve_faust_param_spec(ParamSpecName(case.param_spec_name))
    parameters = {parameter.name: parameter for parameter in spec.synth_params}
    parameter = parameters[case.address]

    assert isinstance(parameter, ContinuousParameter)
    assert (parameter.min, parameter.max) == (case.minimum, case.maximum)
    assert parameter.encode(case.minimum) == pytest.approx(np.array([0.0]))
    assert parameter.encode(case.maximum) == pytest.approx(np.array([1.0]))
    assert parameter.decode(np.array([0.0])) == pytest.approx(case.minimum)
    assert parameter.decode(np.array([1.0])) == pytest.approx(case.maximum)


def test_faust_model_output_decodes_exact_native_addresses() -> None:
    """Model-domain midpoints decode under exact Faust renderer addresses."""
    spec = resolve_faust_param_spec(ParamSpecName("faust_filter_osc"))

    synth_params, _ = decode_model_output(
        np.zeros(spec.encoded_width, dtype=np.float32), spec
    )

    assert synth_params == pytest.approx(
        {
            "/SINE_WAVE_OSCILLATOR_oscrs/Amplitude": -55.0,
            "/SINE_WAVE_OSCILLATOR_oscrs/Frequency": 44.5,
            "/SINE_WAVE_OSCILLATOR_oscrs/Portamento": 5.0005,
        }
    )


@pytest.mark.parametrize("param_spec_name", _EXPECTED_PARAMETER_ADDRESSES)
def test_faust_note_conditioning_contract_is_identity_stable(param_spec_name: str) -> None:
    """Every Faust identity pins pitch and note-window label domains.

    :param param_spec_name: Faust parameter-spec identity under test.
    """
    pitch, note_window = resolve_faust_param_spec(
        ParamSpecName(param_spec_name)
    ).note_params

    assert isinstance(pitch, DiscreteLiteralParameter)
    assert (pitch.name, pitch.min, pitch.max) == ("pitch", 48, 72)
    assert pitch.decode(np.array([0.0])) == 48
    assert pitch.decode(np.array([1.0])) == 72
    assert isinstance(note_window, NoteDurationParameter)
    assert note_window.name == "note_start_and_end"
    assert note_window.max_note_duration_seconds == 4.0
    assert note_window.decode(np.array([0.0, 1.0])) == pytest.approx((0.0, 4.0))


def test_faust_param_spec_resolution_returns_fresh_specs() -> None:
    """Caller mutation cannot alter subsequently resolved identity contracts."""
    changed = resolve_faust_param_spec(ParamSpecName("faust_bubble"))
    changed.synth_params[0].name = "changed"

    resolved_again = resolve_faust_param_spec(ParamSpecName("faust_bubble"))

    assert (
        resolved_again.synth_params[0].name
        == _EXPECTED_PARAMETER_ADDRESSES["faust_bubble"][0]
    )


def test_faust_param_spec_resolution_rejects_unknown_identity() -> None:
    """Unknown spec identities fail without synthesizing a fallback."""
    with pytest.raises(KeyError, match="missing"):
        resolve_faust_param_spec(ParamSpecName("missing"))


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
def test_faust_compiled_parameter_domains_match_specs(param_spec_name: str) -> None:
    """Real Faust metadata matches every modeled native domain.

    :param param_spec_name: Faust source and parameter-spec identity.
    """
    dd = cast(_DawDreamerModule, import_module("dawdreamer"))
    _, _, processor = _compile_faust(dd, param_spec_name)
    descriptions = processor.get_parameters_description()
    parameters = resolve_faust_param_spec(ParamSpecName(param_spec_name)).synth_params

    assert len(descriptions) == len(parameters)
    for description, parameter in zip(descriptions, parameters, strict=True):
        minimum, maximum, is_discrete = _expected_parameter_domain(parameter)
        assert description["name"] == parameter.name
        assert description["min"] == pytest.approx(minimum)
        assert description["max"] == pytest.approx(maximum)
        assert description["isDiscrete"] is is_discrete


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
