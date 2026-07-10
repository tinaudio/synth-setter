"""Build the committed Surge XT CLAP param map for the sound-match bridge (#1787).

Two commands:

- ``dump`` — enumerate the installed CLAP's params via the first-party ctypes
  host (:mod:`synth_setter.data.vst.clap_introspect`) and write the raw dump
  (``surge_xt_clap_info.json``).
- ``build`` — join that dump with pedalboard's view of the VST3 loaded under
  the base preset and emit ``surge_xt_clap_map.json``.

The join is an exact index bridge, not a name heuristic: pedalboard's
``Parameter.index`` is patch-invariant and Surge enumerates params in the same
order over VST3 and CLAP, so ``pyname → index → CLAP entry`` resolves every
spec param — including the FX/osc params the base preset renames. ``build``
validates that premise on every run (init-state names must match the dump
elementwise, and ``surge_params.csv`` display names must match at the bridged
index) and fails loudly listing every unmapped or mismatched parameter.

``build`` loads the VST3 twice via pedalboard, so on Linux it must run under
``src/synth_setter/scripts/run-linux-vst-headless.sh``.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

import click

from synth_setter.data.vst.clap_introspect import (
    SURGE_XT_CLAP_PATH,
    ClapPluginInfo,
    dump_clap_plugin,
)
from synth_setter.data.vst.clap_map import ClapParamRef, PluginFormatMap
from synth_setter.data.vst.param_spec import CategoricalParameter, ParamSpec
from synth_setter.data.vst.param_spec_registry import (
    default_plugin_path,
    param_specs,
    preset_paths,
)

_VST_DIR = Path(__file__).resolve().parent.parent / "data" / "vst"
_DEFAULT_CLAP_INFO_PATH = _VST_DIR / "surge_xt_clap_info.json"
_DEFAULT_MAP_PATH = _VST_DIR / "surge_xt_clap_map.json"


def _stepped_grid_errors(param: CategoricalParameter, ref: ClapParamRef, pyname: str) -> list[str]:
    """Check that a categorical's raw_values land on consecutive native steps.

    The CLI converts a stepped param by emitting ``min_value + position`` of
    the nearest raw_value, which is only correct when position ``i`` lerps to
    native step ``i`` for every entry.

    :param param: Spec categorical parameter under check.
    :param ref: The CLAP ref the map would commit for it.
    :param pyname: Param name used in error messages.
    :returns: One message per raw_value that violates the grid; empty if none.
    """
    span = ref.max_value - ref.min_value
    return [
        f"{pyname}: raw_value {raw} at position {i} lerps to native step "
        f"{round(raw * span)}, expected {i}"
        for i, raw in enumerate(param.raw_values)
        if round(raw * span) != i
    ]


def build_format_map(
    clap_info: ClapPluginInfo,
    preset_indices: Mapping[str, int],
    spec: ParamSpec,
    display_names: Mapping[str, str],
) -> PluginFormatMap:
    """Assemble the pyname → CLAP map for every synth param in ``spec``.

    :param clap_info: Raw CLAP dump in plugin enumeration order.
    :param preset_indices: Pedalboard pyname → patch-invariant parameter index,
        read from the VST3 with the base preset applied.
    :param spec: Param spec whose ``synth_params`` define the map's keys.
    :param display_names: Optional pyname → init-state display name cross-check
        (from ``surge_params.csv``); only consulted for pynames present.
    :returns: The complete map, ready to serialize.
    :raises ValueError: listing every spec param that is unmapped, out of
        range, display-name-mismatched, or stepped-grid-incompatible.
    """
    errors: list[str] = []
    params: dict[str, ClapParamRef] = {}

    for param in spec.synth_params:
        pyname = param.name
        index = preset_indices.get(pyname)
        if index is None:
            errors.append(f"{pyname}: no pedalboard param of that name under the base preset")
            continue
        if not 0 <= index < len(clap_info.params):
            errors.append(f"{pyname}: index {index} outside dump of {len(clap_info.params)}")
            continue
        entry = clap_info.params[index]

        expected = display_names.get(pyname)
        if expected is not None and expected.lower() != entry.name.lower():
            errors.append(
                f"{pyname}: surge_params.csv says {expected!r} but CLAP index {index} "
                f"is {entry.name!r}"
            )
            continue

        ref = ClapParamRef(
            clap_param_id=entry.id,
            clap_name=entry.name,
            clap_module_name=entry.module,
            min_value=entry.min_value,
            max_value=entry.max_value,
            is_stepped=entry.is_stepped,
        )
        if ref.is_stepped and isinstance(param, CategoricalParameter):
            errors.extend(_stepped_grid_errors(param, ref, pyname))
        params[pyname] = ref

    if errors:
        raise ValueError(
            f"CLAP map build failed for {len(errors)} parameter(s):\n" + "\n".join(errors)
        )
    return PluginFormatMap(plugin=clap_info.plugin_name, version=clap_info.version, params=params)


class _IndexedParameter(Protocol):
    """Structural type for the pedalboard parameter surface this module reads."""

    @property
    def name(self) -> str:
        """Host display name."""
        ...

    @property
    def index(self) -> int:
        """Patch-invariant position in the plugin's parameter enumeration."""
        ...


class _IndexedPlugin(Protocol):
    """Structural type for the pedalboard plugin surface this module reads."""

    @property
    def parameters(self) -> Mapping[str, _IndexedParameter]:
        """Mapping of python-name -> parameter wrapper."""
        ...


def _flushed_plugin(plugin_path: str, preset_path: str | None) -> _IndexedPlugin:
    """Load the VST3 (optionally with a preset) and flush once so names settle.

    Surge only refreshes preset-driven FX/osc param names after an audio pass,
    mirroring ``render_params``'s post-load flush.

    :param plugin_path: ``.vst3`` bundle path.
    :param preset_path: ``.vstpreset`` to apply, or ``None`` for the init state.
    :returns: The flushed pedalboard plugin.
    """
    # Deferred: core imports pedalboard; the dump command must work without it.
    from synth_setter.data.vst.core import load_plugin, load_preset

    plugin = load_plugin(plugin_path)
    if preset_path is not None:
        load_preset(plugin, preset_path)
    plugin.process([], 32.0, 44100.0, 2, 2048, True)
    plugin.reset()
    return cast(_IndexedPlugin, plugin)


def init_order_errors(init_names: Sequence[str], clap_info: ClapPluginInfo) -> list[str]:
    """Compare init-state VST3 names against the dump, elementwise and case-insensitive.

    :param init_names: VST3 display names in pedalboard enumeration order.
    :param clap_info: Raw CLAP dump.
    :returns: One message per divergence (count mismatch or positional name mismatch); empty when
        the index-bridge premise holds.
    """
    if len(init_names) != len(clap_info.params):
        return [
            f"VST3 exposes {len(init_names)} params but the CLAP dump has {len(clap_info.params)}"
        ]
    return [
        f"index {i}: VST3 {a!r} != CLAP {b.name!r}"
        for i, (a, b) in enumerate(zip(init_names, clap_info.params))
        if a.lower() != b.name.lower()
    ]


def _assert_init_order_matches(clap_info: ClapPluginInfo, plugin_path: str) -> None:
    """Verify the index bridge's premise: init-state VST3 names == dump names, elementwise.

    :param clap_info: Raw CLAP dump.
    :param plugin_path: ``.vst3`` bundle path.
    :raises ValueError: on a count mismatch or any positional name mismatch.
    """
    init_names = [p.name for p in _flushed_plugin(plugin_path, None).parameters.values()]
    errors = init_order_errors(init_names, clap_info)
    if errors:
        raise ValueError(
            "VST3/CLAP enumeration orders diverge — index bridging is unsound:\n"
            + "\n".join(errors)
        )


def _preset_param_indices(plugin_path: str, preset_path: str) -> dict[str, int]:
    """Read pyname → patch-invariant parameter index under the base preset.

    :param plugin_path: ``.vst3`` bundle path.
    :param preset_path: The spec's base ``.vstpreset``.
    :returns: One entry per parameter pedalboard exposes in the preset state.
    """
    plugin = _flushed_plugin(plugin_path, preset_path)
    return {pyname: param.index for pyname, param in plugin.parameters.items()}


def _read_display_names(csv_path: Path) -> dict[str, str]:
    """Read the pyname → display-name cross-check table from ``surge_params.csv``.

    :param csv_path: CSV with ``pyname`` and ``name`` columns.
    :returns: pyname → init-state display name.
    """
    with csv_path.open() as f:
        return {row["pyname"]: row["name"] for row in csv.DictReader(f)}


@click.group()
def main() -> None:
    """Build the committed Surge XT CLAP param map (see module docstring)."""


@main.command()
@click.option(
    "--plugin",
    type=click.Path(exists=True, path_type=Path),
    default=SURGE_XT_CLAP_PATH,
    show_default=True,
    help="Path to the installed .clap bundle.",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=_DEFAULT_CLAP_INFO_PATH,
    show_default=True,
    help="Where to write the raw dump JSON.",
)
def dump(plugin: Path, out: Path) -> None:
    """Dump the CLAP plugin's full param set to the committed raw-info JSON.

    :param plugin: The ``.clap`` to introspect.
    :param out: Dump destination.
    """
    info = dump_clap_plugin(plugin)
    out.write_text(info.model_dump_json(indent=2) + "\n")
    click.echo(f"wrote {len(info.params)} params ({info.plugin_name} {info.version}) to {out}")


@main.command()
@click.option(
    "--clap-info",
    type=click.Path(exists=True, path_type=Path),
    default=_DEFAULT_CLAP_INFO_PATH,
    show_default=True,
    help="Raw dump produced by the dump command.",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=_DEFAULT_MAP_PATH,
    show_default=True,
    help="Where to write the map JSON.",
)
@click.option("--param-spec-name", default="surge_xt", show_default=True)
@click.option(
    "--params-csv",
    type=click.Path(exists=True, path_type=Path),
    default=Path("surge_params.csv"),
    show_default=True,
    help="pyname → display-name cross-check table.",
)
def build(clap_info: Path, out: Path, param_spec_name: str, params_csv: Path) -> None:
    """Build and validate the committed pyname → CLAP map.

    :param clap_info: Dump file to join against.
    :param out: Map destination.
    :param param_spec_name: Registry key of the spec to map.
    :param params_csv: Cross-check CSV path.
    """
    info = ClapPluginInfo.model_validate_json(clap_info.read_text())
    plugin_path = default_plugin_path()
    _assert_init_order_matches(info, plugin_path)
    indices = _preset_param_indices(plugin_path, preset_paths[param_spec_name])
    format_map = build_format_map(
        info, indices, param_specs[param_spec_name], _read_display_names(params_csv)
    )
    out.write_text(format_map.model_dump_json(indent=2) + "\n")
    click.echo(f"wrote {len(format_map.params)} mapped params to {out}")


if __name__ == "__main__":
    main()
