"""Dump host metadata and join it into a committed cross-host parameter map.

Typical usage::

    uv run python -m synth_setter.tools.build_param_map build \
        --pedalboard-dump pedalboard.json --clap-dump clap.json \
        --dawdreamer-dump dawdreamer.json --param-spec-name surge_xt --out joint.json
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import click
from pydantic import BaseModel, ConfigDict

from synth_setter.data.vst.clap_introspect import ClapParamInfo, ClapPluginInfo, dump_clap_plugin
from synth_setter.data.vst.clap_map import ClapParamRef
from synth_setter.data.vst.param_map import (
    BackendSnapshot,
    DawDreamerParamRef,
    ParamIdentity,
    PedalboardParamRef,
    SynthParamMap,
)
from synth_setter.data.vst.param_spec import CategoricalParameter, Parameter
from synth_setter.data.vst.param_spec_registry import param_specs

INTROSPECTION_SAMPLE_RATE = 44_100
INTROSPECTION_BLOCK_SIZE = 2_048
PEDALBOARD_FLUSH_DURATION_SECONDS = 32.0
PEDALBOARD_FLUSH_CHANNELS = 2
_SURGE_CLAP_OSCILLATOR_NAMES = {
    f"a_osc_{oscillator}_{semantic_name}": f"A Osc {oscillator} {host_name}"
    for oscillator in range(1, 4)
    for semantic_name, host_name in (
        ("sawtooth", "Shape"),
        ("width", "Sub Mix"),
        ("pulse", "Width 1"),
        ("triangle", "Width 2"),
    )
}
_SURGE_DAWDREAMER_OSCILLATOR_NAMES = {
    f"a_osc_{oscillator}_{semantic_name}": f"A Osc {oscillator} {host_name}"
    for oscillator in range(1, 4)
    for semantic_name, host_name in (
        ("sawtooth", "Shape"),
        ("width", "Sub Mix"),
        ("pulse", "Width 1"),
        ("triangle", "Width 2"),
    )
}
_SURGE_FX_IDENTITIES = {
    "fx_a1_delay_time": ("FX A1 Param 1", "A1", 1),
    "fx_a1_modulation_rate": ("FX A1 Param 2", "A1", 2),
    "fx_a1_modulation_depth": ("FX A1 Param 3", "A1", 3),
    "fx_a1_delay_feedback": ("FX A1 Param 4", "A1", 4),
    "fx_a1_eq_low_cut": ("FX A1 Param 5", "A1", 5),
    "fx_a1_eq_high_cut": ("FX A1 Param 6", "A1", 6),
    "fx_a1_output_mix": ("FX A1 Param 7", "A1", 7),
    "fx_a1_output_width": ("FX A1 Param 8", "A1", 8),
    "fx_a2_delay_time_left": ("FX A2 Param 1", "A2", 1),
    "fx_a2_delay_time_right": ("FX A2 Param 2", "A2", 2),
    "fx_a2_feedback_eq_feedback": ("FX A2 Param 3", "A2", 3),
    "fx_a2_feedback_eq_crossfeed": ("FX A2 Param 4", "A2", 4),
    "fx_a2_feedback_eq_low_cut": ("FX A2 Param 5", "A2", 5),
    "fx_a2_feedback_eq_high_cut": ("FX A2 Param 6", "A2", 6),
    "fx_a2_modulation_rate": ("FX A2 Param 7", "A2", 7),
    "fx_a2_modulation_depth": ("FX A2 Param 8", "A2", 8),
    "fx_a2_input_channel": ("FX A2 Param 9", "A2", 9),
    "fx_a2_output_mix": ("FX A2 Param 11", "A2", 11),
    "fx_a2_output_width": ("FX A2 Param 12", "A2", 12),
    "fx_a3_pre_delay_pre_delay": ("FX A3 Param 1", "A3", 1),
    "fx_a3_reverb_room_size": ("FX A3 Param 2", "A3", 2),
    "fx_a3_reverb_decay_time": ("FX A3 Param 3", "A3", 3),
    "fx_a3_reverb_diffusion": ("FX A3 Param 4", "A3", 4),
    "fx_a3_reverb_buildup": ("FX A3 Param 5", "A3", 5),
    "fx_a3_reverb_modulation": ("FX A3 Param 6", "A3", 6),
    "fx_a3_eq_lf_damping": ("FX A3 Param 7", "A3", 7),
    "fx_a3_eq_hf_damping": ("FX A3 Param 8", "A3", 8),
    "fx_a3_output_width": ("FX A3 Param 9", "A3", 9),
    "fx_a3_output_mix": ("FX A3 Param 10", "A3", 10),
}
_SURGE_FX_SLOT_COUNT = 12


class HostParam(BaseModel):  # noqa: DOC601, DOC603
    """One indexed parameter emitted by a VST host dump."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    index: int
    key: str | None = None
    name: str


class _PedalboardParameter(Protocol):
    """Metadata read from one dynamically exposed Pedalboard parameter."""

    @property
    def index(self) -> int:
        """Host enumeration index."""
        ...

    @property
    def name(self) -> str:
        """Host display name."""
        ...


class _PedalboardMetadata(Protocol):
    """Dynamic Pedalboard metadata used by the offline dump command."""

    @property
    def name(self) -> str:
        """Plugin display name."""
        ...

    @property
    def version(self) -> str:
        """Plugin version string."""
        ...

    @property
    def parameters(self) -> Mapping[str, _PedalboardParameter]:
        """Repository-keyed host parameter metadata."""
        ...


class HostDump(BaseModel):  # noqa: DOC601, DOC603
    """Offline input captured from one plugin host."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    plugin: str
    plugin_version: str
    preset_resource: str
    preset_sha256: str
    params: list[HostParam]


def _read_host_dump(path: Path) -> HostDump:
    """Load one strict host dump.

    :param path: Host dump path.
    :returns: Validated dump.
    """
    return HostDump.model_validate_json(path.read_text(encoding="utf-8"))


def _normalized_identity(value: str) -> str:
    """Normalize a semantic key or host label for independent name resolution.

    :param value: Repository key or host-native display name.
    :returns: Case-insensitive alphanumeric identity.
    """
    return "".join(character for character in value.casefold() if character.isalnum())


def _expected_clap_name(semantic_key: str) -> str:
    """Return the CLAP label declared for one repository semantic key.

    :param semantic_key: Repository-owned parameter identity.
    :returns: Surge's stable CLAP label.
    """
    if semantic_key in _SURGE_FX_IDENTITIES:
        clap_name, _, _ = _SURGE_FX_IDENTITIES[semantic_key]
        return clap_name
    return _SURGE_CLAP_OSCILLATOR_NAMES.get(semantic_key, semantic_key)


def _expected_dawdreamer_name(semantic_key: str) -> str:
    """Return the DawDreamer label declared for one repository semantic key.

    :param semantic_key: Repository-owned parameter identity.
    :returns: Surge's stable non-FX DawDreamer label.
    """
    return _SURGE_DAWDREAMER_OSCILLATOR_NAMES.get(semantic_key, semantic_key)


def _validate_provenance(
    pedalboard: HostDump, clap: ClapPluginInfo, dawdreamer: HostDump
) -> list[str]:
    """Collect provenance mismatches across the three host snapshots.

    :param pedalboard: Pedalboard host dump.
    :param clap: CLAP plugin dump.
    :param dawdreamer: DawDreamer host dump.
    :returns: All provenance mismatches.
    """
    errors: list[str] = []
    if pedalboard.plugin != dawdreamer.plugin or pedalboard.plugin != clap.plugin_name:
        errors.append("plugin identities disagree")
    if len({pedalboard.plugin_version, clap.version, dawdreamer.plugin_version}) != 1:
        errors.append("host plugin versions disagree")
    if pedalboard.preset_resource != dawdreamer.preset_resource:
        errors.append("preset resources disagree")
    if pedalboard.preset_sha256 != dawdreamer.preset_sha256:
        errors.append("preset hashes disagree")
    return errors


def _index_pedalboard(params: list[HostParam], errors: list[str]) -> dict[str, HostParam]:
    """Index Pedalboard's repository keys and collect duplicate identities.

    :param params: Pedalboard parameter enumeration.
    :param errors: Aggregated diagnostics destination.
    :returns: Repository semantic key to Pedalboard identity.
    """
    keyed = [param for param in params if param.key is not None]
    by_key = {param.key: param for param in keyed if param.key is not None}
    if len(by_key) != len(keyed):
        errors.append("duplicate Pedalboard keys")
    indices = [param.index for param in params]
    if len(indices) != len(set(indices)):
        errors.append("duplicate Pedalboard indices")
    if any(index < 0 for index in indices):
        errors.append("negative Pedalboard indices")
    return by_key


def _index_clap(clap: ClapPluginInfo, errors: list[str]) -> dict[str, list[ClapParamInfo]]:
    """Index CLAP's host-native names without using another backend's indices.

    :param clap: CLAP plugin dump.
    :param errors: Aggregated diagnostics destination.
    :returns: Normalized CLAP name to matching parameter records.
    """
    ids = [param.id for param in clap.params]
    if len(ids) != len(set(ids)):
        errors.append("duplicate CLAP parameter ids")
    by_name: dict[str, list[ClapParamInfo]] = {}
    for param in clap.params:
        by_name.setdefault(_normalized_identity(param.name), []).append(param)
    return by_name


def _index_dawdreamer(
    params: list[HostParam], errors: list[str]
) -> tuple[dict[str, list[HostParam]], dict[int, HostParam]]:
    """Index DawDreamer's own names and indices.

    :param params: DawDreamer parameter enumeration.
    :param errors: Aggregated diagnostics destination.
    :returns: Normalized-name and numeric-index lookups.
    """
    by_name: dict[str, list[HostParam]] = {}
    by_index: dict[int, HostParam] = {}
    for param in params:
        by_name.setdefault(_normalized_identity(param.name), []).append(param)
        if param.index in by_index:
            errors.append(f"duplicate DawDreamer index {param.index}")
        if param.index < 0:
            errors.append(f"negative DawDreamer index {param.index}")
        by_index[param.index] = param
    return by_name, by_index


def _resolve_clap_param(
    semantic_key: str,
    by_name: dict[str, list[ClapParamInfo]],
    errors: list[str],
) -> ClapParamInfo | None:
    """Resolve CLAP directly from a repository semantic key.

    :param semantic_key: Repository-owned parameter identity.
    :param by_name: CLAP normalized-name index.
    :param errors: Aggregated diagnostics destination.
    :returns: Unique CLAP record, or ``None`` after recording a diagnostic.
    """
    expected_name = _expected_clap_name(semantic_key)
    candidates = by_name.get(_normalized_identity(expected_name), [])
    if len(candidates) != 1:
        errors.append(f"{semantic_key}: CLAP name {expected_name!r} is missing or ambiguous")
        return None
    return candidates[0]


def _resolve_dawdreamer_param(
    semantic_key: str,
    by_name: dict[str, list[HostParam]],
    errors: list[str],
) -> HostParam | None:
    """Resolve a non-FX DawDreamer parameter from a repository semantic key.

    :param semantic_key: Repository-owned parameter identity.
    :param by_name: DawDreamer normalized-name index.
    :param errors: Aggregated diagnostics destination.
    :returns: Unique DawDreamer record, or ``None`` after a diagnostic.
    """
    expected_name = _expected_dawdreamer_name(semantic_key)
    candidates = by_name.get(_normalized_identity(expected_name), [])
    if len(candidates) != 1:
        errors.append(f"{semantic_key}: DawDreamer name {expected_name!r} is missing or ambiguous")
        return None
    return candidates[0]


def _resolve_dawdreamer_fx_bank(
    bank: str,
    indexes: tuple[dict[str, list[HostParam]], dict[int, HostParam]],
    errors: list[str],
) -> dict[int, HostParam] | None:
    """Validate one complete host-local DawDreamer FX bank.

    :param bank: Surge FX bank identifier.
    :param indexes: DawDreamer normalized-name and numeric-index lookups.
    :param errors: Aggregated diagnostics destination.
    :returns: Slot number to DawDreamer identity when the bank is valid.
    """
    by_name, by_index = indexes
    anchor_name = f"FX {bank} FX Type"
    anchors = by_name.get(_normalized_identity(anchor_name), [])
    if len(anchors) != 1:
        errors.append(f"DawDreamer FX {bank} anchor is missing or ambiguous")
        return None
    anchor_index = anchors[0].index
    slots: dict[int, HostParam] = {}
    for slot in range(1, _SURGE_FX_SLOT_COUNT + 1):
        parameter = by_index.get(anchor_index + slot)
        expected_name = f"FX {bank} -"
        if parameter is None or parameter.name.casefold() != expected_name.casefold():
            errors.append(f"DawDreamer FX {bank} slot {slot} is missing or invalid")
            continue
        slots[slot] = parameter
    return slots if len(slots) == _SURGE_FX_SLOT_COUNT else None


def _clap_reference(parameter: ClapParamInfo) -> ClapParamRef:
    """Convert one validated CLAP record into the committed map schema.

    :param parameter: CLAP parameter metadata.
    :returns: CLAP identity and range reference.
    """
    return ClapParamRef(
        clap_param_id=parameter.id,
        clap_name=parameter.name,
        clap_module_name=parameter.module,
        min_value=parameter.min_value,
        max_value=parameter.max_value,
        is_stepped=parameter.is_stepped,
    )


def _categorical_grid_matches(spec_param: Parameter, clap_ref: ClapParamRef) -> bool:
    """Check a categorical parameter against CLAP's native stepped range.

    :param spec_param: Repository parameter specification.
    :param clap_ref: Resolved CLAP range metadata.
    :returns: Whether every raw value maps to its declared category index.
    """
    if not clap_ref.is_stepped or not isinstance(spec_param, CategoricalParameter):
        return True
    span = clap_ref.max_value - clap_ref.min_value
    return all(round(raw * span) == index for index, raw in enumerate(spec_param.raw_values))


_DawDreamerIndexes = tuple[dict[str, list[HostParam]], dict[int, HostParam]]
_DawDreamerFxBanks = dict[str, dict[int, HostParam] | None]
_JoinIndexes = tuple[
    dict[str, HostParam],
    dict[str, list[ClapParamInfo]],
    _DawDreamerIndexes,
    _DawDreamerFxBanks,
]


def _resolve_dawdreamer_identity(
    semantic_key: str,
    indexes: _DawDreamerIndexes,
    fx_banks: _DawDreamerFxBanks,
    *,
    errors: list[str],
) -> HostParam | None:
    """Resolve one DawDreamer identity from its backend-specific declaration.

    :param semantic_key: Repository-owned parameter identity.
    :param indexes: DawDreamer normalized-name and numeric-index lookups.
    :param fx_banks: Previously validated DawDreamer FX banks.
    :param errors: Aggregated diagnostics destination.
    :returns: Resolved DawDreamer parameter, or ``None`` after a diagnostic.
    """
    if semantic_key not in _SURGE_FX_IDENTITIES:
        by_name, _ = indexes
        return _resolve_dawdreamer_param(semantic_key, by_name, errors)
    _, bank, slot = _SURGE_FX_IDENTITIES[semantic_key]
    if bank not in fx_banks:
        fx_banks[bank] = _resolve_dawdreamer_fx_bank(bank, indexes, errors)
    bank_params = fx_banks[bank]
    return bank_params.get(slot) if bank_params is not None else None


def _resolve_param_identity(
    spec_param: Parameter,
    indexes: _JoinIndexes,
    *,
    errors: list[str],
) -> ParamIdentity | None:
    """Resolve one semantic parameter independently in all three backends.

    :param spec_param: Repository parameter specification to resolve.
    :param indexes: Backend-native identity lookups and validated FX-bank cache.
    :param errors: Aggregated diagnostics destination.
    :returns: Joint identity, or ``None`` after recording all reachable diagnostics.
    """
    pedalboard_by_key, clap_by_name, dawdreamer_indexes, dawdreamer_fx_banks = indexes
    semantic_key = spec_param.name
    pedalboard_param = pedalboard_by_key.get(semantic_key)
    if pedalboard_param is None:
        errors.append(f"{semantic_key}: missing Pedalboard identity")
        return None
    clap_param = _resolve_clap_param(semantic_key, clap_by_name, errors)
    if clap_param is None:
        return None
    dawdreamer_param = _resolve_dawdreamer_identity(
        semantic_key, dawdreamer_indexes, dawdreamer_fx_banks, errors=errors
    )
    if dawdreamer_param is None:
        return None
    clap_ref = _clap_reference(clap_param)
    if not _categorical_grid_matches(spec_param, clap_ref):
        errors.append(f"{semantic_key}: categorical grid does not match CLAP steps")
        return None
    return ParamIdentity(
        pedalboard=PedalboardParamRef(index=pedalboard_param.index, name=pedalboard_param.name),
        clap=clap_ref,
        dawdreamer=DawDreamerParamRef(index=dawdreamer_param.index, name=dawdreamer_param.name),
    )


def join_param_map(
    param_spec_name: str,
    pedalboard: HostDump,
    clap: ClapPluginInfo,
    dawdreamer: HostDump,
) -> SynthParamMap:
    """Join three offline dumps, failing on ambiguous or drifting identity.

    :param param_spec_name: Registered parameter spec name.
    :param pedalboard: Preset-specific Pedalboard dump.
    :param clap: Full CLAP dump.
    :param dawdreamer: Preset-specific DawDreamer dump.
    :returns: Validated joint map.
    :raises ValueError: If host provenance or any parameter identity is invalid.
    """
    errors = _validate_provenance(pedalboard, clap, dawdreamer)
    pedalboard_by_key = _index_pedalboard(pedalboard.params, errors)
    clap_by_name = _index_clap(clap, errors)
    dawdreamer_indexes = _index_dawdreamer(dawdreamer.params, errors)
    indexes: _JoinIndexes = (pedalboard_by_key, clap_by_name, dawdreamer_indexes, {})
    identities: dict[str, ParamIdentity] = {}
    for spec_param in param_specs[param_spec_name].synth_params:
        identity = _resolve_param_identity(spec_param, indexes, errors=errors)
        if identity is not None:
            identities[spec_param.name] = identity
    if errors:
        raise ValueError("parameter map join failed:\n" + "\n".join(errors))
    return SynthParamMap(
        plugin=pedalboard.plugin,
        param_spec_name=param_spec_name,
        preset_resource=pedalboard.preset_resource,
        preset_sha256=pedalboard.preset_sha256,
        pedalboard=BackendSnapshot(
            plugin_version=pedalboard.plugin_version, parameter_count=len(pedalboard.params)
        ),
        clap=BackendSnapshot(plugin_version=clap.version, parameter_count=len(clap.params)),
        dawdreamer=BackendSnapshot(
            plugin_version=dawdreamer.plugin_version, parameter_count=len(dawdreamer.params)
        ),
        params=identities,
    )


@click.group()
def main() -> None:
    """Build committed parameter maps from separately captured host dumps."""


@main.command("dump-clap")
@click.option("--plugin", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def dump_clap(plugin: Path, out: Path) -> None:
    """Capture the CLAP enumeration.

    :param plugin: CLAP plugin path.
    :param out: Dump destination.
    """
    out.write_text(dump_clap_plugin(plugin).model_dump_json(indent=2) + "\n")


@main.command("dump-pedalboard")
@click.option("--plugin", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--preset", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--preset-resource", required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def dump_pedalboard(plugin: Path, preset: Path, preset_resource: str, out: Path) -> None:
    """Capture Pedalboard's flushed, preset-specific enumeration.

    :param plugin: VST3 plugin path.
    :param preset: Preset path.
    :param preset_resource: Repository-relative preset resource.
    :param out: Dump destination.
    """
    import hashlib

    from synth_setter.data.vst.core import load_plugin, load_preset

    loaded = load_plugin(str(plugin))
    load_preset(loaded, str(preset))
    loaded.process(
        [],
        PEDALBOARD_FLUSH_DURATION_SECONDS,
        INTROSPECTION_SAMPLE_RATE,
        PEDALBOARD_FLUSH_CHANNELS,
        INTROSPECTION_BLOCK_SIZE,
        True,
    )
    loaded.reset()
    metadata = cast(_PedalboardMetadata, loaded)
    dump = HostDump(
        plugin=metadata.name,
        plugin_version=metadata.version,
        preset_resource=preset_resource,
        preset_sha256=hashlib.sha256(preset.read_bytes()).hexdigest(),
        params=[
            HostParam(index=param.index, key=key, name=param.name)
            for key, param in metadata.parameters.items()
        ],
    )
    out.write_text(dump.model_dump_json(indent=2) + "\n", encoding="utf-8")


@main.command("dump-dawdreamer")
@click.option("--plugin", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--plugin-name", required=True)
@click.option("--plugin-version", required=True)
@click.option("--preset", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--preset-resource", required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def dump_dawdreamer(
    plugin: Path,
    plugin_name: str,
    plugin_version: str,
    preset: Path,
    preset_resource: str,
    out: Path,
) -> None:
    """Capture DawDreamer's full preset-specific enumeration.

    :param plugin: VST3 plugin path.
    :param plugin_name: Canonical plugin name.
    :param plugin_version: Plugin version snapshot.
    :param preset: Preset path.
    :param preset_resource: Repository-relative preset resource.
    :param out: Dump destination.
    """
    import hashlib

    dawdreamer = import_module("dawdreamer")
    engine = dawdreamer.RenderEngine(INTROSPECTION_SAMPLE_RATE, INTROSPECTION_BLOCK_SIZE)
    loaded = engine.make_plugin_processor("synth", str(plugin.resolve()))
    loaded.load_vst3_preset(str(preset.resolve()))
    dump = HostDump(
        plugin=plugin_name,
        plugin_version=plugin_version,
        preset_resource=preset_resource,
        preset_sha256=hashlib.sha256(preset.read_bytes()).hexdigest(),
        params=[
            HostParam(index=int(item["index"]), name=str(item["name"]))
            for item in loaded.get_parameters_description()
        ],
    )
    out.write_text(dump.model_dump_json(indent=2) + "\n", encoding="utf-8")


@main.command("build")
@click.option("--pedalboard-dump", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--clap-dump", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--dawdreamer-dump", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--param-spec-name", required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def build(
    pedalboard_dump: Path,
    clap_dump: Path,
    dawdreamer_dump: Path,
    param_spec_name: str,
    out: Path,
) -> None:
    """Join previously captured dumps without loading a plugin runtime.

    :param pedalboard_dump: Pedalboard dump path.
    :param clap_dump: CLAP dump path.
    :param dawdreamer_dump: DawDreamer dump path.
    :param param_spec_name: Registered parameter spec name.
    :param out: Joint map destination.
    """
    result = join_param_map(
        param_spec_name,
        _read_host_dump(pedalboard_dump),
        ClapPluginInfo.model_validate_json(clap_dump.read_text(encoding="utf-8")),
        _read_host_dump(dawdreamer_dump),
    )
    out.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
