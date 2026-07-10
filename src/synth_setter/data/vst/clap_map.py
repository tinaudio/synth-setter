"""Pedalboard-pyname → CLAP parameter map for the live sound-match bridge (#1787).

``PluginFormatMap`` is the committed JSON contract translating this repo's
pedalboard-normalized parameter world into CLAP parameter ids/names/ranges;
``synth_params_to_clap_rows`` converts one decoded prediction into rows in the
parameters' native CLAP value domain. ``tools/build_clap_map.py`` builds the
committed map; ``cli/predict_capture.py`` consumes it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from synth_setter.data.vst.param_spec import CategoricalParameter, Parameter, ParamSpec


# DOC601/603: pydoclint can't see :ivar: docs on pydantic/dataclass fields (#1787).
class ClapParamRef(BaseModel):  # noqa: DOC601, DOC603
    """CLAP identity and native value range of one mapped synth parameter."""

    model_config = ConfigDict(strict=True, extra="forbid")

    clap_param_id: int
    clap_name: str
    clap_module_name: str
    min_value: float
    max_value: float
    is_stepped: bool


class PluginFormatMap(BaseModel):  # noqa: DOC601, DOC603
    """CLAP param refs for every model parameter, keyed by pedalboard pyname."""

    model_config = ConfigDict(strict=True, extra="forbid")

    plugin: str
    version: str
    params: dict[str, ClapParamRef]


# DOC502: the documented ValidationError propagates from pydantic.
def load_clap_map(path: Path) -> PluginFormatMap:  # noqa: DOC502
    """Parse a committed CLAP map JSON file.

    :param path: Path to a ``PluginFormatMap`` JSON document.
    :returns: The validated map.
    :raises pydantic.ValidationError: when the document does not match the schema.
    """
    return PluginFormatMap.model_validate_json(path.read_text())


@dataclass(frozen=True)
class ClapCsvRow:  # noqa: DOC601, DOC603
    """One ``params.csv`` row; ``clap_value`` is in the parameter's native CLAP domain."""

    # pb_name is the contract-pinned CSV header (#1787) — a rename breaks the C++ reader.
    pb_name: str
    clap_name: str
    clap_module_name: str
    clap_param_id: int
    clap_value: float


def _lerp_to_native(value: float, ref: ClapParamRef) -> float:
    """Map a ``[0, 1]`` value onto the ref's native ``[min_value, max_value]``.

    :param value: Decoded pedalboard-normalized value in ``[0, 1]``.
    :param ref: CLAP identity carrying the native range.
    :returns: The lerped native value.
    """
    return ref.min_value + value * (ref.max_value - ref.min_value)


def _stepped_clap_value(param: Parameter | None, value: float, ref: ClapParamRef) -> float:
    """Convert a decoded [0, 1] value to a stepped parameter's native CLAP value.

    Categorical params map the nearest ``raw_values`` entry's position to
    ``min_value + index``; other stepped params round the lerped native value.

    :param param: The spec parameter that produced ``value``, when known.
    :param value: Decoded pedalboard-normalized value in ``[0, 1]``.
    :param ref: CLAP identity carrying the native ``[min_value, max_value]``.
    :returns: Native CLAP value on an integer step.
    """
    if isinstance(param, CategoricalParameter):
        index = min(range(len(param.raw_values)), key=lambda i: abs(param.raw_values[i] - value))
        return round(ref.min_value + index)
    return round(_lerp_to_native(value, ref))


def synth_params_to_clap_rows(
    synth_params: Mapping[str, float],
    spec: ParamSpec,
    format_map: PluginFormatMap,
) -> list[ClapCsvRow]:
    """Convert decoded synth params into native-domain CLAP rows, one per parameter.

    :param synth_params: Decoded prediction (pyname → pedalboard-normalized value
        in ``[0, 1]``), e.g. from :meth:`ParamSpec.decode`.
    :param spec: Spec that produced ``synth_params``; its categorical parameters
        define the stepped index mapping.
    :param format_map: Committed pyname → CLAP identity map.
    :returns: Rows in ``synth_params`` iteration order.
    :raises ValueError: when any decoded param is missing from ``format_map``
        (all missing names are listed — a partial CSV must never be written).
    """
    missing = sorted(name for name in synth_params if name not in format_map.params)
    if missing:
        raise ValueError(
            f"decoded params missing from the CLAP map ({len(missing)}): {', '.join(missing)}"
        )

    spec_params = {p.name: p for p in spec.synth_params}
    rows = []
    for name, value in synth_params.items():
        ref = format_map.params[name]
        if ref.is_stepped:
            clap_value = _stepped_clap_value(spec_params.get(name), value, ref)
        else:
            clap_value = _lerp_to_native(value, ref)
        rows.append(
            ClapCsvRow(
                pb_name=name,
                clap_name=ref.clap_name,
                clap_module_name=ref.clap_module_name,
                clap_param_id=ref.clap_param_id,
                clap_value=float(clap_value),
            )
        )
    return rows
