"""Exact-address parameter specifications for checked-in Faust programs.

Usage::

    from synth_setter.data.vst.faust_param_spec import resolve_faust_param_spec

    spec = resolve_faust_param_spec(ParamSpecName("faust_bright_organ"))
    synth_params, note_params = spec.decode(encoded_row)
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType

from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
    NoteDurationParameter,
    Parameter,
    ParamSpec,
)
from synth_setter.param_spec_name import ParamSpecName

# These conditioning bounds are baked into each identity; changes require a new ParamSpecName.
_FAUST_MAX_NOTE_DURATION_SECONDS = 4.0
_FAUST_MIDI_PITCH_MAX = 72
_FAUST_MIDI_PITCH_MIN = 48


def _note_params() -> list[Parameter]:
    """Build fresh dataset-default note controls for one Faust spec.

    :returns: Pitch and note-window parameters.
    """
    return [
        DiscreteLiteralParameter(
            name="pitch",
            min=_FAUST_MIDI_PITCH_MIN,
            max=_FAUST_MIDI_PITCH_MAX,
        ),
        NoteDurationParameter(
            name="note_start_and_end",
            max_note_duration_seconds=_FAUST_MAX_NOTE_DURATION_SECONDS,
        ),
    ]


def _unit_parameter(name: str) -> ContinuousParameter:
    """Build a continuous unit-range control.

    :param name: Exact Faust parameter address.
    :returns: Unit-range continuous parameter.
    """
    return ContinuousParameter(name=name)


def _trigger_parameter(name: str) -> CategoricalParameter:
    """Build a two-state Faust button control.

    :param name: Exact Faust parameter address.
    :returns: One-hot parameter whose decoded values are native button states.
    """
    return CategoricalParameter(
        name=name,
        values=[False, True],
        raw_values=[0.0, 1.0],
        encoding="onehot",
    )


def _bright_organ_param_spec() -> ParamSpec:
    """Build the brightOrgan specification with MIDI-owned frequency and gate.

    :returns: Fresh exact-address brightOrgan specification.
    """
    return ParamSpec(
        [
            _unit_parameter("/Sequencer/DSP1/brightOrgan/Main/volume"),
            _unit_parameter("/Sequencer/DSP1/brightOrgan/Reverb/Amount"),
            _unit_parameter("/Sequencer/DSP1/brightOrgan/Reverb/Damp"),
            _unit_parameter("/Sequencer/DSP1/brightOrgan/Reverb/Size"),
            _unit_parameter("/Sequencer/DSP1/brightOrgan/Stops/Fifteenth_2'"),
            _unit_parameter("/Sequencer/DSP1/brightOrgan/Stops/Flute_8'"),
            _unit_parameter("/Sequencer/DSP1/brightOrgan/Stops/Foundation_8'"),
            _unit_parameter("/Sequencer/DSP1/brightOrgan/Stops/Nasard_2_2/3'"),
            _unit_parameter("/Sequencer/DSP1/brightOrgan/Stops/Principal_4'"),
            _unit_parameter("/Sequencer/DSP1/brightOrgan/Stops/Tierce_1_3/5'"),
        ],
        _note_params(),
    )


def _bubble_param_spec() -> ParamSpec:
    """Build the bubble specification.

    :returns: Fresh exact-address bubble specification.
    """
    return ParamSpec(
        [
            _unit_parameter("/bubble/Freeverb/0x00/Damp"),
            _unit_parameter("/bubble/Freeverb/0x00/RoomSize"),
            _unit_parameter("/bubble/Freeverb/0x00/Stereo_Spread"),
            _unit_parameter("/bubble/Freeverb/Wet"),
            ContinuousParameter(name="/bubble/bubble/freq", min=150.0, max=2000.0),
            _trigger_parameter("/bubble/drop"),
        ],
        _note_params(),
    )


def _church_organ_param_spec() -> ParamSpec:
    """Build the churchOrgan specification.

    :returns: Fresh exact-address churchOrgan specification.
    """
    return ParamSpec(
        [
            ContinuousParameter(
                name="/churchOrgan/Zita_Light/Dry/Wet_Mix",
                min=-1.0,
                max=1.0,
            ),
            ContinuousParameter(
                name="/churchOrgan/Zita_Light/Level",
                min=-70.0,
                max=40.0,
            ),
            ContinuousParameter(name="/churchOrgan/freq", min=50.0, max=1000.0),
            _unit_parameter("/churchOrgan/gain"),
            _unit_parameter("/churchOrgan/gain_fundamental"),
            _unit_parameter("/churchOrgan/gain_8ve_partial"),
            _unit_parameter("/churchOrgan/gain_5th_partial"),
            _unit_parameter("/churchOrgan/gain_3d_partial"),
            _unit_parameter("/churchOrgan/gain_other_partials"),
            _unit_parameter("/churchOrgan/gain_lower_octave"),
            _unit_parameter("/churchOrgan/noise_gain"),
            _trigger_parameter("/churchOrgan/gate"),
        ],
        _note_params(),
    )


def _filter_osc_param_spec() -> ParamSpec:
    """Build the filterOSC specification.

    :returns: Fresh exact-address filterOSC specification.
    """
    return ParamSpec(
        [
            ContinuousParameter(
                name="/SINE_WAVE_OSCILLATOR_oscrs/Amplitude",
                min=-120.0,
                max=10.0,
            ),
            ContinuousParameter(
                name="/SINE_WAVE_OSCILLATOR_oscrs/Frequency",
                min=1.0,
                max=88.0,
            ),
            ContinuousParameter(
                name="/SINE_WAVE_OSCILLATOR_oscrs/Portamento",
                min=0.001,
                max=10.0,
            ),
        ],
        _note_params(),
    )


_faust_param_spec_builders: Mapping[ParamSpecName, Callable[[], ParamSpec]] = MappingProxyType(
    {
        ParamSpecName("faust_bright_organ"): _bright_organ_param_spec,
        ParamSpecName("faust_bubble"): _bubble_param_spec,
        ParamSpecName("faust_church_organ"): _church_organ_param_spec,
        ParamSpecName("faust_filter_osc"): _filter_osc_param_spec,
    }
)


def resolve_faust_param_spec(param_spec_name: ParamSpecName) -> ParamSpec:
    """Build one checked-in Faust specification without shared mutable state.

    :param param_spec_name: Faust source and parameter-spec identity.
    :returns: Fresh exact-address specification for the requested program.
    :raises KeyError: If the key has no checked-in Faust specification.
    """
    try:
        builder = _faust_param_spec_builders[param_spec_name]
    except KeyError:
        raise KeyError(param_spec_name) from None
    return builder()
