"""Dump host metadata and join it into a committed cross-host parameter map.

Typical usage::

    uv run python -m synth_setter.tools.build_param_map build \
        --pedalboard-dump pedalboard.json --clap-dump clap.json \
        --dawdreamer-dump dawdreamer.json --param-spec-name surge_xt --out joint.json
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import click
from pydantic import BaseModel, ConfigDict

from synth_setter.data.vst.clap_introspect import ClapParamInfo, ClapPluginInfo, dump_clap_plugin
from synth_setter.data.vst.clap_map import ClapParamRef
from synth_setter.data.vst.dawdreamer_runtime import settle_dawdreamer_preset
from synth_setter.data.vst.param_map import (
    BackendSnapshot,
    DawDreamerParamRef,
    ParamIdentity,
    PedalboardParamRef,
    SurgePyParamRef,
    SynthParamMap,
)
from synth_setter.data.vst.param_spec import CategoricalParameter, Parameter
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.data.vst.surgepy_runtime import surge_component_state
from synth_setter.param_spec_name import ParamSpecName

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
_SURGE_FX_LABELS = {
    "fx_a1_delay_time": ("FX A1 Param 1", "FX A1 Time"),
    "fx_a1_modulation_rate": ("FX A1 Param 2", "FX A1 Rate"),
    "fx_a1_modulation_depth": ("FX A1 Param 3", "FX A1 Depth"),
    "fx_a1_delay_feedback": ("FX A1 Param 4", "FX A1 Feedback"),
    "fx_a1_eq_low_cut": ("FX A1 Param 5", "FX A1 Low Cut"),
    "fx_a1_eq_high_cut": ("FX A1 Param 6", "FX A1 High Cut"),
    "fx_a1_output_mix": ("FX A1 Param 7", "FX A1 Mix"),
    "fx_a1_output_width": ("FX A1 Param 8", "FX A1 Width"),
    "fx_a2_delay_time_left": ("FX A2 Param 1", "FX A2 Left"),
    "fx_a2_delay_time_right": ("FX A2 Param 2", "FX A2 Right"),
    "fx_a2_feedback_eq_feedback": ("FX A2 Param 3", "FX A2 Feedback"),
    "fx_a2_feedback_eq_crossfeed": ("FX A2 Param 4", "FX A2 Crossfeed"),
    "fx_a2_feedback_eq_low_cut": ("FX A2 Param 5", "FX A2 Low Cut"),
    "fx_a2_feedback_eq_high_cut": ("FX A2 Param 6", "FX A2 High Cut"),
    "fx_a2_modulation_rate": ("FX A2 Param 7", "FX A2 Rate"),
    "fx_a2_modulation_depth": ("FX A2 Param 8", "FX A2 Depth"),
    "fx_a2_input_channel": ("FX A2 Param 9", "FX A2 Channel"),
    "fx_a2_output_mix": ("FX A2 Param 11", "FX A2 Mix"),
    "fx_a2_output_width": ("FX A2 Param 12", "FX A2 Width"),
    "fx_a3_pre_delay_pre_delay": ("FX A3 Param 1", "FX A3 Pre-Delay"),
    "fx_a3_reverb_room_size": ("FX A3 Param 2", "FX A3 Room Size"),
    "fx_a3_reverb_decay_time": ("FX A3 Param 3", "FX A3 Decay Time"),
    "fx_a3_reverb_diffusion": ("FX A3 Param 4", "FX A3 Diffusion"),
    "fx_a3_reverb_buildup": ("FX A3 Param 5", "FX A3 Buildup"),
    "fx_a3_reverb_modulation": ("FX A3 Param 6", "FX A3 Modulation"),
    "fx_a3_eq_lf_damping": ("FX A3 Param 7", "FX A3 LF Damping"),
    "fx_a3_eq_hf_damping": ("FX A3 Param 8", "FX A3 HF Damping"),
    "fx_a3_output_width": ("FX A3 Param 9", "FX A3 Width"),
    "fx_a3_output_mix": ("FX A3 Param 10", "FX A3 Mix"),
}
_SURGE_FX_NAMES = {key: generic_name for key, (generic_name, _) in _SURGE_FX_LABELS.items()}
_SURGEPY_FX_NAMES = {key: surgepy_name for key, (_, surgepy_name) in _SURGE_FX_LABELS.items()}


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


class SurgePyHostParam(BaseModel):  # noqa: DOC601, DOC603
    """One synth-side parameter emitted by a real SurgePy patch."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    synth_side_id: int
    name: str


class SurgePyDump(BaseModel):  # noqa: DOC601, DOC603
    """Preset-specific SurgePy engine and parameter snapshot."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    engine_version: str
    preset_resource: str
    preset_sha256: str
    parameter_count: int
    params: list[SurgePyHostParam]


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
    if semantic_key in _SURGE_FX_NAMES:
        return _SURGE_FX_NAMES[semantic_key]
    return _SURGE_CLAP_OSCILLATOR_NAMES.get(semantic_key, semantic_key)


def _expected_surgepy_name(semantic_key: str, pedalboard_name: str) -> str:
    """Return the live SurgePy label for one repository parameter.

    :param semantic_key: Repository-owned parameter identity.
    :param pedalboard_name: Preset-specific VST label for non-FX parameters.
    :returns: SurgePy patch-tree label.
    """
    return _SURGEPY_FX_NAMES.get(semantic_key, pedalboard_name)


def _validate_surgepy_provenance(
    pedalboard: HostDump,
    surgepy: SurgePyDump,
) -> list[str]:
    """Compare native patch state across VST and SurgePy containers.

    :param pedalboard: VST-host baseline dump.
    :param surgepy: SurgePy FXP baseline dump.
    :returns: Component-state provenance errors.
    """
    try:
        vst_state = surge_component_state(Path(pedalboard.preset_resource))
        surgepy_state = surge_component_state(Path(surgepy.preset_resource))
    except (OSError, ValueError) as exc:
        return [f"Surge component provenance unavailable: {exc}"]
    if vst_state != surgepy_state:
        return ["SurgePy and VST preset component states disagree"]
    return []


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


def _index_dawdreamer(params: list[HostParam], errors: list[str]) -> dict[str, list[HostParam]]:
    """Index DawDreamer's own names and validate indices.

    :param params: DawDreamer parameter enumeration.
    :param errors: Aggregated diagnostics destination.
    :returns: Normalized-name lookup.
    """
    by_name: dict[str, list[HostParam]] = {}
    indices: set[int] = set()
    for param in params:
        by_name.setdefault(_normalized_identity(param.name), []).append(param)
        if param.index in indices:
            errors.append(f"duplicate DawDreamer index {param.index}")
        if param.index < 0:
            errors.append(f"negative DawDreamer index {param.index}")
        indices.add(param.index)
    return by_name


def _index_surgepy(
    params: list[SurgePyHostParam], errors: list[str]
) -> dict[str, list[SurgePyHostParam]]:
    """Index SurgePy's native names and validate stable synth-side IDs.

    :param params: SurgePy patch enumeration.
    :param errors: Aggregated diagnostics destination.
    :returns: Normalized-name lookup.
    """
    by_name: dict[str, list[SurgePyHostParam]] = {}
    ids: set[int] = set()
    for param in params:
        by_name.setdefault(_normalized_identity(param.name), []).append(param)
        if param.synth_side_id in ids:
            errors.append(f"duplicate SurgePy synth-side ID {param.synth_side_id}")
        if param.synth_side_id < 0:
            errors.append(f"negative SurgePy synth-side ID {param.synth_side_id}")
        ids.add(param.synth_side_id)
    return by_name


def _resolve_surgepy_param(
    semantic_key: str,
    *,
    pedalboard_name: str,
    by_name: dict[str, list[SurgePyHostParam]],
    errors: list[str],
) -> SurgePyHostParam | None:
    """Resolve a parameter against SurgePy's preset-specific patch tree.

    :param semantic_key: Repository-owned parameter identity.
    :param pedalboard_name: VST label reused when SurgePy has no contextual FX label.
    :param by_name: SurgePy normalized-name index.
    :param errors: Aggregated diagnostics destination.
    :returns: Unique SurgePy record, or ``None`` after a diagnostic.
    """
    expected_name = _expected_surgepy_name(semantic_key, pedalboard_name)
    candidates = by_name.get(_normalized_identity(expected_name), [])
    if len(candidates) != 1:
        errors.append(f"{semantic_key}: SurgePy name {expected_name!r} is missing or ambiguous")
        return None
    return candidates[0]


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
    pedalboard_name: str,
    by_name: dict[str, list[HostParam]],
    *,
    errors: list[str],
) -> HostParam | None:
    """Resolve one preset-settled DawDreamer parameter identity.

    :param semantic_key: Repository-owned parameter identity.
    :param pedalboard_name: Active preset-specific VST parameter name.
    :param by_name: DawDreamer normalized-name index.
    :param errors: Aggregated diagnostics destination.
    :returns: Unique DawDreamer record, or ``None`` after a diagnostic.
    """
    candidates = by_name.get(_normalized_identity(pedalboard_name), [])
    if len(candidates) != 1:
        errors.append(
            f"{semantic_key}: DawDreamer name {pedalboard_name!r} is missing or ambiguous"
        )
        return None
    return candidates[0]


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


_JoinIndexes = tuple[
    dict[str, HostParam],
    dict[str, list[ClapParamInfo]],
    dict[str, list[HostParam]],
    dict[str, list[SurgePyHostParam]] | None,
]


def _resolve_param_identity(
    spec_param: Parameter,
    indexes: _JoinIndexes,
    *,
    errors: list[str],
) -> ParamIdentity | None:
    """Resolve one semantic parameter independently in all three backends.

    :param spec_param: Repository parameter specification to resolve.
    :param indexes: Backend-native identity lookups.
    :param errors: Aggregated diagnostics destination.
    :returns: Joint identity, or ``None`` after recording all reachable diagnostics.
    """
    pedalboard_by_key, clap_by_name, dawdreamer_by_name, surgepy_by_name = indexes
    semantic_key = spec_param.name
    pedalboard_param = pedalboard_by_key.get(semantic_key)
    if pedalboard_param is None:
        errors.append(f"{semantic_key}: missing Pedalboard identity")
        return None
    clap_param = _resolve_clap_param(semantic_key, clap_by_name, errors)
    if clap_param is None:
        return None
    dawdreamer_param = _resolve_dawdreamer_param(
        semantic_key,
        pedalboard_param.name,
        dawdreamer_by_name,
        errors=errors,
    )
    if dawdreamer_param is None:
        return None
    clap_ref = _clap_reference(clap_param)
    if not _categorical_grid_matches(spec_param, clap_ref):
        errors.append(f"{semantic_key}: categorical grid does not match CLAP steps")
        return None
    surgepy_ref = None
    if surgepy_by_name is not None:
        surgepy_param = _resolve_surgepy_param(
            semantic_key,
            pedalboard_name=pedalboard_param.name,
            by_name=surgepy_by_name,
            errors=errors,
        )
        if surgepy_param is None:
            return None
        surgepy_ref = SurgePyParamRef(
            synth_side_id=surgepy_param.synth_side_id,
            name=surgepy_param.name,
        )
    return ParamIdentity(
        pedalboard=PedalboardParamRef(index=pedalboard_param.index, name=pedalboard_param.name),
        clap=clap_ref,
        dawdreamer=DawDreamerParamRef(index=dawdreamer_param.index, name=dawdreamer_param.name),
        surgepy=surgepy_ref,
    )


def join_param_map(
    param_spec_name: str,
    *,
    pedalboard: HostDump,
    clap: ClapPluginInfo,
    dawdreamer: HostDump,
    surgepy: SurgePyDump | None = None,
) -> SynthParamMap:
    """Join host dumps, failing on ambiguous or drifting identity.

    :param param_spec_name: Registered parameter spec name.
    :param pedalboard: Preset-specific Pedalboard dump.
    :param clap: Full CLAP dump.
    :param dawdreamer: Preset-specific DawDreamer dump.
    :param surgepy: Optional preset-specific SurgePy dump for Surge specs.
    :returns: Validated joint map.
    :raises ValueError: If host provenance or any parameter identity is invalid.
    """
    errors = _validate_provenance(pedalboard, clap, dawdreamer)
    if surgepy:
        errors.extend(_validate_surgepy_provenance(pedalboard, surgepy))
    pedalboard_by_key = _index_pedalboard(pedalboard.params, errors)
    clap_by_name = _index_clap(clap, errors)
    dawdreamer_by_name = _index_dawdreamer(dawdreamer.params, errors)
    surgepy_by_name = _index_surgepy(surgepy.params, errors) if surgepy else None
    if surgepy and surgepy.parameter_count != len(surgepy.params):
        errors.append("SurgePy parameter count does not match its enumeration")
    indexes: _JoinIndexes = (
        pedalboard_by_key,
        clap_by_name,
        dawdreamer_by_name,
        surgepy_by_name,
    )
    identities: dict[str, ParamIdentity] = {}
    for spec_param in param_specs[param_spec_name].synth_params:
        identity = _resolve_param_identity(spec_param, indexes, errors=errors)
        if identity is not None:
            identities[spec_param.name] = identity
    if errors:
        raise ValueError("parameter map join failed:\n" + "\n".join(errors))
    return SynthParamMap(
        plugin=pedalboard.plugin,
        param_spec_name=ParamSpecName(param_spec_name),
        preset_resource=pedalboard.preset_resource,
        preset_sha256=pedalboard.preset_sha256,
        pedalboard=BackendSnapshot(
            plugin_version=pedalboard.plugin_version, parameter_count=len(pedalboard.params)
        ),
        clap=BackendSnapshot(plugin_version=clap.version, parameter_count=len(clap.params)),
        dawdreamer=BackendSnapshot(
            plugin_version=dawdreamer.plugin_version, parameter_count=len(dawdreamer.params)
        ),
        surgepy_preset_resource=surgepy.preset_resource if surgepy else None,
        surgepy_preset_sha256=surgepy.preset_sha256 if surgepy else None,
        surgepy=(
            BackendSnapshot(
                plugin_version=surgepy.engine_version,
                parameter_count=surgepy.parameter_count,
            )
            if surgepy
            else None
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


@main.command("dump-surgepy")
@click.option("--preset", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--preset-resource", required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def dump_surgepy(preset: Path, preset_resource: str, out: Path) -> None:
    """Capture SurgePy's real preset-specific patch enumeration.

    :param preset: Surge ``.fxp`` patch path.
    :param preset_resource: Repository-relative patch resource.
    :param out: Dump destination.
    :raises RuntimeError: If SurgePy rejects the patch.
    """
    from synth_setter.data.vst.surgepy_runtime import (
        import_surgepy,
        iter_surgepy_named_params,
    )

    surgepy = import_surgepy()
    synth = surgepy.createSurge(INTROSPECTION_SAMPLE_RATE)
    if not synth.loadPatch(str(preset.resolve())):
        raise RuntimeError(f"SurgePy could not load patch {preset}")
    by_id = {}
    for parameter in iter_surgepy_named_params(synth.getPatch()):
        synth_side_id = parameter.getId().getSynthSideId()
        by_id.setdefault(
            synth_side_id,
            SurgePyHostParam(synth_side_id=synth_side_id, name=parameter.getName()),
        )
    dump = SurgePyDump(
        engine_version=surgepy.getVersion(),
        preset_resource=preset_resource,
        preset_sha256=hashlib.sha256(preset.read_bytes()).hexdigest(),
        parameter_count=len(by_id),
        params=list(by_id.values()),
    )
    out.write_text(dump.model_dump_json(indent=2) + "\n", encoding="utf-8")


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
    dawdreamer = import_module("dawdreamer")
    engine = dawdreamer.RenderEngine(INTROSPECTION_SAMPLE_RATE, INTROSPECTION_BLOCK_SIZE)
    loaded = engine.make_plugin_processor("synth", str(plugin.resolve()))
    engine.load_graph([(loaded, [])])
    loaded.load_vst3_preset(str(preset.resolve()))
    settle_dawdreamer_preset(
        engine,
        sample_rate=INTROSPECTION_SAMPLE_RATE,
        block_size=INTROSPECTION_BLOCK_SIZE,
    )
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
@click.option("--surgepy-dump", type=click.Path(exists=True, path_type=Path))
@click.option("--param-spec-name", required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def build(
    pedalboard_dump: Path,
    clap_dump: Path,
    dawdreamer_dump: Path,
    surgepy_dump: Path | None,
    param_spec_name: str,
    out: Path,
) -> None:
    """Join previously captured dumps without loading a plugin runtime.

    :param pedalboard_dump: Pedalboard dump path.
    :param clap_dump: CLAP dump path.
    :param dawdreamer_dump: DawDreamer dump path.
    :param surgepy_dump: Optional SurgePy dump path.
    :param param_spec_name: Registered parameter spec name.
    :param out: Joint map destination.
    """
    result = join_param_map(
        param_spec_name,
        pedalboard=_read_host_dump(pedalboard_dump),
        clap=ClapPluginInfo.model_validate_json(clap_dump.read_text(encoding="utf-8")),
        dawdreamer=_read_host_dump(dawdreamer_dump),
        surgepy=(
            SurgePyDump.model_validate_json(surgepy_dump.read_text(encoding="utf-8"))
            if surgepy_dump
            else None
        ),
    )
    out.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
