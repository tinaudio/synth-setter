"""Dump host metadata and join it into a committed cross-host parameter map."""

from __future__ import annotations

import re
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import click
from pydantic import BaseModel, ConfigDict

from synth_setter.data.vst.clap_introspect import ClapPluginInfo, dump_clap_plugin
from synth_setter.data.vst.clap_map import ClapParamRef
from synth_setter.data.vst.param_map import (
    BackendSnapshot,
    DawDreamerParamRef,
    ParamIdentity,
    PedalboardParamRef,
    SynthParamMap,
)
from synth_setter.data.vst.param_spec import CategoricalParameter
from synth_setter.data.vst.param_spec_registry import param_specs


class HostParam(BaseModel):  # noqa: DOC601, DOC603
    """One indexed parameter emitted by a VST host dump."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    index: int
    key: str | None = None
    name: str


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
    errors: list[str] = []
    if pedalboard.plugin != dawdreamer.plugin or pedalboard.plugin != clap.plugin_name:
        errors.append("plugin identities disagree")
    if len({pedalboard.plugin_version, clap.version, dawdreamer.plugin_version}) != 1:
        errors.append("host plugin versions disagree")
    if pedalboard.preset_resource != dawdreamer.preset_resource:
        errors.append("preset resources disagree")
    if pedalboard.preset_sha256 != dawdreamer.preset_sha256:
        errors.append("preset hashes disagree")
    pb_by_key = {param.key: param for param in pedalboard.params if param.key is not None}
    if len(pb_by_key) != len([param for param in pedalboard.params if param.key is not None]):
        errors.append("duplicate Pedalboard keys")
    pb_indices = [param.index for param in pedalboard.params]
    if len(pb_indices) != len(set(pb_indices)):
        errors.append("duplicate Pedalboard indices")
    dd_by_name: dict[str, list[HostParam]] = {}
    dd_by_index: dict[int, HostParam] = {}
    for param in dawdreamer.params:
        dd_by_name.setdefault(param.name, []).append(param)
        if param.index in dd_by_index:
            errors.append(f"duplicate DawDreamer index {param.index}")
        dd_by_index[param.index] = param
    identities: dict[str, ParamIdentity] = {}
    for spec_param in param_specs[param_spec_name].synth_params:
        name = spec_param.name
        pb = pb_by_key.get(name)
        if pb is None:
            errors.append(f"{name}: missing Pedalboard identity")
            continue
        if not 0 <= pb.index < len(clap.params):
            errors.append(f"{name}: Pedalboard index {pb.index} outside CLAP dump")
            continue
        clap_param = clap.params[pb.index]
        match = re.fullmatch(r"FX (A[1-4]) Param (\d+)", clap_param.name)
        candidates: list[HostParam]
        if match:
            bank, slot = match.groups()
            anchors = dd_by_name.get(f"FX {bank} FX Type", [])
            target = anchors[0].index + int(slot) if len(anchors) == 1 else -1
            candidates = [dd_by_index[target]] if target in dd_by_index else []
        else:
            candidates = dd_by_name.get(clap_param.name, [])
        if len(candidates) != 1:
            errors.append(f"{name}: DawDreamer name {clap_param.name!r} is missing or ambiguous")
            continue
        clap_ref = ClapParamRef(
            clap_param_id=clap_param.id,
            clap_name=clap_param.name,
            clap_module_name=clap_param.module,
            min_value=clap_param.min_value,
            max_value=clap_param.max_value,
            is_stepped=clap_param.is_stepped,
        )
        if clap_ref.is_stepped and isinstance(spec_param, CategoricalParameter):
            span = clap_ref.max_value - clap_ref.min_value
            if any(round(raw * span) != index for index, raw in enumerate(spec_param.raw_values)):
                errors.append(f"{name}: categorical grid does not match CLAP steps")
                continue
        identities[name] = ParamIdentity(
            pedalboard=PedalboardParamRef(index=pb.index, name=pb.name),
            clap=clap_ref,
            dawdreamer=DawDreamerParamRef(index=candidates[0].index, name=candidates[0].name),
        )
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

    loaded = cast(Any, load_plugin(str(plugin)))
    load_preset(loaded, str(preset))
    loaded.process([], 32.0, 44100.0, 2, 2048, True)
    loaded.reset()
    dump = HostDump(
        plugin=loaded.name,
        plugin_version=loaded.version,
        preset_resource=preset_resource,
        preset_sha256=hashlib.sha256(preset.read_bytes()).hexdigest(),
        params=[
            HostParam(index=param.index, key=key, name=param.name)
            for key, param in loaded.parameters.items()
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
    engine = dawdreamer.RenderEngine(44100, 2048)
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
