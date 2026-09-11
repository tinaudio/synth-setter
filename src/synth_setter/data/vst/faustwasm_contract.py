"""Exact canonical-to-FaustWasm parameter identities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

import numpy as np

from synth_setter.data.pyfdn_param_spec import PYFDN_ORDER, householder_feedback_matrix
from synth_setter.data.vst.faust_param_spec import resolve_faust_param_spec
from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousArrayParameter,
    ContinuousParameter,
    DiscreteArrayParameter,
    Parameter,
    ParameterValue,
    require_scalar_synth_params,
)
from synth_setter.param_spec_name import ParamSpecName

_FDN_HOUSEHOLDER = ParamSpecName("faust_fdn_n8_mono_householder")
_FDN_WASM_PREFIX = "/fdnHouseholder"
# Spec-derived fields a source renders as constants; a patch must carry exactly these values.
_FIXED_DERIVED_FIELDS: Mapping[ParamSpecName, Mapping[str, np.ndarray]] = MappingProxyType(
    {
        _FDN_HOUSEHOLDER: MappingProxyType(
            {"feedback_matrix": householder_feedback_matrix(np.ones(PYFDN_ORDER))}
        ),
    }
)
_FIXED_DERIVED_ATOL = 1e-9


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
        ParamSpecName("faust_bright_organ"): (
            "/brightOrgan/Main/freq",
            "/brightOrgan/Main/gate",
        ),
        ParamSpecName("faust_bubble"): (),
        ParamSpecName("faust_church_organ"): (),
        _FDN_HOUSEHOLDER: (),
        ParamSpecName("faust_filter_osc"): (),
        ParamSpecName("faust_kronecker_fdn"): (),
    }
)

_WASM_ADDRESSES = MappingProxyType(
    {
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
        _FDN_HOUSEHOLDER: (
            *(f"{_FDN_WASM_PREFIX}/delay_{index}" for index in range(PYFDN_ORDER)),
            *(f"{_FDN_WASM_PREFIX}/input_{index}" for index in range(PYFDN_ORDER)),
            *(f"{_FDN_WASM_PREFIX}/output_{index}" for index in range(PYFDN_ORDER)),
            f"{_FDN_WASM_PREFIX}/direct",
            f"{_FDN_WASM_PREFIX}/rt_dc_seconds",
            f"{_FDN_WASM_PREFIX}/rt_nyquist_seconds",
        ),
        ParamSpecName("faust_filter_osc"): (
            "/SINE_WAVE_OSCILLATOR_oscrs/Amplitude",
            "/SINE_WAVE_OSCILLATOR_oscrs/Frequency",
            "/SINE_WAVE_OSCILLATOR_oscrs/Portamento",
        ),
        ParamSpecName("faust_kronecker_fdn"): (
            "/kroneckerFDN/Decay_t60_dc",
            "/kroneckerFDN/Decay_t60_nyquist",
            "/kroneckerFDN/Delays_d0",
            "/kroneckerFDN/Delays_d1",
            "/kroneckerFDN/Delays_d2",
            "/kroneckerFDN/Delays_d3",
            "/kroneckerFDN/Delays_d4",
            "/kroneckerFDN/Delays_d5",
            "/kroneckerFDN/Delays_d6",
            "/kroneckerFDN/Delays_d7",
            "/kroneckerFDN/Input_b0",
            "/kroneckerFDN/Input_b1",
            "/kroneckerFDN/Input_b2",
            "/kroneckerFDN/Input_b3",
            "/kroneckerFDN/Input_b4",
            "/kroneckerFDN/Input_b5",
            "/kroneckerFDN/Input_b6",
            "/kroneckerFDN/Input_b7",
            "/kroneckerFDN/Kernel_a0",
            "/kroneckerFDN/Kernel_a1",
            "/kroneckerFDN/Kernel_a2",
            "/kroneckerFDN/Kernel_r0",
            "/kroneckerFDN/Kernel_r1",
            "/kroneckerFDN/Kernel_r2",
            "/kroneckerFDN/Output_c0",
            "/kroneckerFDN/Output_c1",
            "/kroneckerFDN/Output_c2",
            "/kroneckerFDN/Output_c3",
            "/kroneckerFDN/Output_c4",
            "/kroneckerFDN/Output_c5",
            "/kroneckerFDN/Output_c6",
            "/kroneckerFDN/Output_c7",
            "/kroneckerFDN/Output_dry",
        ),
    }
)


def faustwasm_reserved_addresses(identity: ParamSpecName) -> tuple[str, ...]:
    """Return compiler-owned controls excluded from the canonical patch.

    :param identity: Faust source and parameter-spec identity.
    :returns: Exact compiled addresses controlled by MIDI polyphony.
    """
    return _RESERVED_WASM_ADDRESSES[identity]


class _SliderDomain(NamedTuple):
    """Native domain of one compiled slider before its compiler address is bound.

    .. attribute :: canonical_address
       :type: str

       Canonical coordinate label stored in dataset patches.
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
    minimum: float
    maximum: float
    kind: str
    values: tuple[float, ...] | None


def _slider_domains(parameter: Parameter) -> list[_SliderDomain]:
    """Expand one canonical parameter into per-slider native domains.

    :param parameter: Canonical parameter definition.
    :returns: One domain per compiled slider, in canonical coordinate order.
    :raises TypeError: The parameter has no supported native domain.
    """
    if isinstance(parameter, ContinuousParameter):
        return [_SliderDomain(parameter.name, parameter.min, parameter.max, "continuous", None)]
    if isinstance(parameter, CategoricalParameter):
        values = tuple(float(value) for value in parameter.raw_values)
        return [_SliderDomain(parameter.name, min(values), max(values), "discrete", values)]
    if isinstance(parameter, ContinuousArrayParameter):
        return [
            _SliderDomain(name, parameter.min, parameter.max, "continuous", None)
            for name in parameter.native_names()
        ]
    raise TypeError(f"unsupported FaustWasm parameter {type(parameter).__name__}")


def faustwasm_parameter_contract(identity: ParamSpecName) -> tuple[FaustWasmParameter, ...]:
    """Return the complete explicit mapping for one checked-in source identity.

    Array parameters expand to one slider per element, named by
    :meth:`Parameter.native_names` in C order.

    :param identity: Faust source and parameter-spec identity.
    :returns: Parameters in canonical specification order.
    :raises ValueError: The address table does not cover every canonical slider.
    """
    wasm_addresses = _WASM_ADDRESSES[identity]
    domains = [
        domain
        for parameter in resolve_faust_param_spec(identity).synth_params
        for domain in _slider_domains(parameter)
    ]
    if len(wasm_addresses) != len(domains):
        raise ValueError("FaustWasm address mapping is incomplete")
    return tuple(
        FaustWasmParameter(
            canonical_address=domain.canonical_address,
            wasm_address=wasm_address,
            minimum=float(domain.minimum),
            maximum=float(domain.maximum),
            kind=domain.kind,
            values=domain.values,
        )
        for domain, wasm_address in zip(domains, wasm_addresses, strict=True)
    )


def flatten_canonical_patch(
    identity: ParamSpecName, params: Mapping[str, ParameterValue]
) -> dict[str, float]:
    """Flatten a decoded native patch into the scalar sliders the compiled source exposes.

    :param identity: Faust source and parameter-spec identity.
    :param params: Renderer-native synth values decoded by the identity's spec.
    :returns: One float per compiled slider, keyed by canonical slider address.
    :raises KeyError: A field is neither a spec parameter nor a fixed derived value.
    :raises ValueError: A field has the wrong shape, a discrete field carries a fractional value,
        or a fixed derived value differs.
    """
    fixed = _FIXED_DERIVED_FIELDS.get(identity, {})
    parameters = {
        parameter.name: parameter
        for parameter in resolve_faust_param_spec(identity).synth_params
    }
    unknown = sorted(params.keys() - parameters.keys() - fixed.keys())
    if unknown:
        raise KeyError(f"unknown canonical parameter(s): {', '.join(unknown)}")
    for name, expected in fixed.items():
        supplied = np.asarray(params[name], dtype=np.float64) if name in params else None
        if supplied is None or supplied.shape != expected.shape or not np.allclose(
            supplied, expected, rtol=0.0, atol=_FIXED_DERIVED_ATOL
        ):
            raise ValueError(f"{name} must equal the value compiled into {identity}")
    flat: dict[str, float] = {}
    for name, parameter in parameters.items():
        if name not in params:
            continue
        value = params[name]
        if isinstance(parameter, ContinuousArrayParameter):
            array = np.asarray(value, dtype=np.float64)
            if array.shape != parameter.shape:
                raise ValueError(f"{name} must have shape {parameter.shape}, got {array.shape}")
            # Integer sliders truncate in the DSP, so a fractional value must fail here, not render.
            if isinstance(parameter, DiscreteArrayParameter) and not np.equal(
                array, np.rint(array)
            ).all():
                raise ValueError(f"{name} must contain only integer values")
            flat.update(zip(parameter.native_names(), array.reshape(-1).tolist(), strict=True))
        else:
            flat.update(require_scalar_synth_params({name: value}))
    return flat
