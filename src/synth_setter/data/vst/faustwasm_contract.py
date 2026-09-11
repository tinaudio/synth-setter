"""Exact canonical-to-FaustWasm parameter identities."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from synth_setter.data.vst.faust_param_spec import resolve_faust_param_spec
from synth_setter.data.vst.param_spec import CategoricalParameter, ContinuousParameter
from synth_setter.param_spec_name import ParamSpecName


@dataclass(frozen=True)
class FaustWasmParameter:
    """One canonical parameter and its compiler-specific FaustWasm address.

    .. attribute :: canonical_address
       :type: str

       Stable address stored in dataset patches.
    .. attribute :: wasm_address
       :type: str

       Address emitted by this FaustWasm compiler version.
    .. attribute :: minimum
       :type: float

       Native lower bound.
    .. attribute :: maximum
       :type: float

       Native upper bound.
    .. attribute :: kind
       :type: str

       ``continuous`` or ``discrete`` domain kind.
    .. attribute :: values
       :type: tuple[float, ...] | None

       Exact native values for discrete controls.
    """

    canonical_address: str
    wasm_address: str
    minimum: float
    maximum: float
    kind: str
    values: tuple[float, ...] | None


_RESERVED_WASM_ADDRESSES = MappingProxyType(
    {
        ParamSpecName("faust_augmentor"): (),
        ParamSpecName("faust_bright_organ"): (
            "/brightOrgan/Main/freq",
            "/brightOrgan/Main/gate",
        ),
        ParamSpecName("faust_bubble"): (),
        ParamSpecName("faust_church_organ"): (),
        ParamSpecName("faust_filter_osc"): (),
    }
)

_WASM_ADDRESSES = MappingProxyType(
    {
        ParamSpecName("faust_augmentor"): (
            "/augmentor/envelope_depth",
            "/augmentor/envelope_rate",
            "/augmentor/filter_cutoff",
            "/augmentor/filter_mix",
            "/augmentor/filter_resonance",
            "/augmentor/gate",
            "/augmentor/noise_amount",
            "/augmentor/pitch_mix",
            "/augmentor/pitch_shift",
            "/augmentor/reverse_mix",
            "/augmentor/source_freq",
            "/augmentor/source_gain",
        ),
        ParamSpecName("faust_bright_organ"): (
            "/brightOrgan/Main/volume",
            "/brightOrgan/Reverb/Amount",
            "/brightOrgan/Reverb/Damp",
            "/brightOrgan/Reverb/Size",
            "/brightOrgan/Stops/Fifteenth_2-",
            "/brightOrgan/Stops/Flute_8-",
            "/brightOrgan/Stops/Foundation_8-",
            "/brightOrgan/Stops/Nasard_2_2_3-",
            "/brightOrgan/Stops/Principal_4-",
            "/brightOrgan/Stops/Tierce_1_3_5-",
        ),
        ParamSpecName("faust_bubble"): (
            "/bubble/Freeverb/0x00/Damp",
            "/bubble/Freeverb/0x00/RoomSize",
            "/bubble/Freeverb/0x00/Stereo_Spread",
            "/bubble/Freeverb/Wet",
            "/bubble/bubble/freq",
            "/bubble/drop",
        ),
        ParamSpecName("faust_church_organ"): (
            "/churchOrgan/Zita_Light/Wet_Dry_Mix",
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
        ),
        ParamSpecName("faust_filter_osc"): (
            "/SINE_WAVE_OSCILLATOR_oscrs/Amplitude",
            "/SINE_WAVE_OSCILLATOR_oscrs/Frequency",
            "/SINE_WAVE_OSCILLATOR_oscrs/Portamento",
        ),
    }
)


def faustwasm_reserved_addresses(identity: ParamSpecName) -> tuple[str, ...]:
    """Return compiler-owned controls excluded from the canonical patch.

    :param identity: Faust source and parameter-spec identity.
    :returns: Exact compiled addresses controlled by MIDI polyphony.
    """
    return _RESERVED_WASM_ADDRESSES[identity]


def faustwasm_parameter_contract(identity: ParamSpecName) -> tuple[FaustWasmParameter, ...]:
    """Return the complete explicit mapping for one checked-in source identity.

    :param identity: Faust source and parameter-spec identity.
    :returns: Parameters in canonical specification order.
    :raises TypeError: A canonical parameter has no supported native domain.
    :raises ValueError: The address table does not cover every canonical parameter.
    """
    wasm_addresses = _WASM_ADDRESSES[identity]
    parameters = resolve_faust_param_spec(identity).synth_params
    if len(wasm_addresses) != len(parameters):
        raise ValueError("FaustWasm address mapping is incomplete")
    contract = []
    for parameter, wasm_address in zip(parameters, wasm_addresses, strict=True):
        if isinstance(parameter, ContinuousParameter):
            minimum, maximum, kind = parameter.min, parameter.max, "continuous"
            values = None
        elif isinstance(parameter, CategoricalParameter):
            values = tuple(float(value) for value in parameter.raw_values)
            minimum, maximum, kind = min(values), max(values), "discrete"
        else:
            raise TypeError(f"unsupported FaustWasm parameter {type(parameter).__name__}")
        contract.append(
            FaustWasmParameter(
                canonical_address=parameter.name,
                wasm_address=wasm_address,
                minimum=float(minimum),
                maximum=float(maximum),
                kind=kind,
                values=values,
            )
        )
    return tuple(contract)
