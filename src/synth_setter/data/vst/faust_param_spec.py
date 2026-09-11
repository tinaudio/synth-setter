"""Exact-address parameter specifications for checked-in Faust programs.

Usage::

    from synth_setter.data.vst.faust_param_spec import resolve_faust_param_spec

    spec = resolve_faust_param_spec(ParamSpecName("faust_bright_organ"))
    synth_params, note_params = spec.decode(encoded_row)
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType

import numpy as np

from synth_setter.data.pyfdn_param_spec import build_pyfdn_n8_mono_householder_param_spec
from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
    NoteDurationParameter,
    Parameter,
    ParameterValues,
    ParamSpec,
)
from synth_setter.param_spec_name import ParamSpecName

# These conditioning bounds are baked into each identity; changes require a new ParamSpecName.
FAUST_NOTE_DURATION_SECONDS = 4.0
_FAUST_MIDI_PITCH_MAX = 72
_FAUST_MIDI_PITCH_MIN = 48
_SHIMMER_FDN_NOTE_PARAMS: ParameterValues = {
    "pitch": 60,
    "note_start_and_end": (0.0, FAUST_NOTE_DURATION_SECONDS),
}


class ShimmerFDNParamSpec(ParamSpec):
    """Represent only shimmer controls while supplying renderer-compatible note values."""

    def __init__(self, synth_params: list[Parameter]) -> None:
        """Bind the shimmer control vector without sampled MIDI coordinates.

        :param synth_params: Exact-address controls represented in each encoded row.
        """
        super().__init__(synth_params=synth_params, note_params=[])

    def sample(
        self, rng: np.random.Generator | None = None
    ) -> tuple[ParameterValues, ParameterValues]:
        """Sample DSP controls and return the fixed compatibility note.

        :param rng: Optional caller-owned random generator.
        :returns: Sampled controls and fixed four-second MIDI mapping.
        """
        synth_params, _ = super().sample(rng)
        return synth_params, _SHIMMER_FDN_NOTE_PARAMS.copy()

    def decode(self, params: np.ndarray) -> tuple[ParameterValues, ParameterValues]:
        """Decode DSP controls and return the fixed compatibility note.

        :param params: Encoded DSP-control row shaped ``(self.encoded_width,)``.
        :returns: Decoded controls and fixed four-second MIDI mapping.
        """
        synth_params, _ = super().decode(params)
        return synth_params, _SHIMMER_FDN_NOTE_PARAMS.copy()


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
            max_note_duration_seconds=FAUST_NOTE_DURATION_SECONDS,
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


def _shimmer_fdn_param_spec() -> ParamSpec:
    """Build the fixed-impulse shimmer FDN specification.

    :returns: Fresh exact-address shimmer specification without note coordinates.
    """
    shifted_lines = [
        _trigger_parameter(f"/shimmerFDN/Shimmer/shifted_lines/line__{index}")
        for index in range(8)
    ]
    return ShimmerFDNParamSpec(
        [
            ContinuousParameter(name="/shimmerFDN/FDN/T60_low", min=0.1, max=20.0),
            ContinuousParameter(name="/shimmerFDN/FDN/T60_high", min=0.05, max=20.0),
            ContinuousParameter(name="/shimmerFDN/FDN/crossover", min=200.0, max=16000.0),
            ContinuousParameter(name="/shimmerFDN/Shimmer/transpose", min=-2400.0, max=2400.0),
            ContinuousParameter(name="/shimmerFDN/Shimmer/window", min=64.0, max=8192.0),
            *shifted_lines,
            ContinuousParameter(name="/shimmerFDN/Shimmer/DC_comp_max", min=0.0, max=12.0),
            _unit_parameter("/shimmerFDN/Output/dry/wet"),
            ContinuousParameter(name="/shimmerFDN/Output/level", min=-40.0, max=12.0),
            ContinuousParameter(name="/shimmerFDN/Safety/loop_ceiling", min=-40.0, max=0.0),
            _trigger_parameter("/shimmerFDN/Safety/energy_guard_bypass"),
        ]
    )


def _fdn_householder_param_spec() -> ParamSpec:
    """Build the pyFDN householder specification the Faust FDN renders verbatim.

    :returns: Fresh spec identical to ``pyfdn_n8_mono_householder``, so both backends decode
        identical rows without sharing one mutable registry object.
    """
    return build_pyfdn_n8_mono_householder_param_spec()


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
        ParamSpecName("faust_fdn_n8_mono_householder"): _fdn_householder_param_spec,
        ParamSpecName("faust_filter_osc"): _filter_osc_param_spec,
        ParamSpecName("faust_shimmer_fdn"): _shimmer_fdn_param_spec,
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
