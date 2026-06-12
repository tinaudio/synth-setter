"""``synth-setter-introspect-plugin`` — scaffold a draft ``ParamSpec`` from any VST3.

Loads the plugin via pedalboard, optionally applies a starting preset, then
writes an editable draft spec module plus a captured baseline ``.vstpreset``
— the two artifacts a new synth needs before it can be registered in
``synth_setter.data.vst.param_spec_registry`` (issue #1596).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import click

from synth_setter.data.vst.core import extract_renderer_version, load_plugin, load_preset
from synth_setter.data.vst.introspect import (
    IntrospectablePlugin,
    capture_preset,
    draft_synth_params,
    render_param_spec_module,
    render_param_table_csv,
)
from synth_setter.data.vst.param_spec_registry import default_plugin_path


@click.command()
@click.option(
    "--plugin-path",
    "-p",
    type=click.Path(exists=True),
    default=default_plugin_path,
    show_default="$SYNTH_SETTER_PLUGIN_PATH or the bundled Surge XT",
    help="Path to the .vst3 bundle to introspect.",
)
@click.option(
    "--preset-path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional .vstpreset to load before drafting and capture.",
)
@click.option(
    "--spec-name",
    required=True,
    help="Registry key for the synth (a Python identifier, e.g. 'odin2').",
)
@click.option(
    "--out-spec",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    show_default="<spec-name>_param_spec.py",
    help="Where to write the draft spec module.",
)
@click.option(
    "--out-preset",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    show_default="<spec-name>-base.vstpreset",
    help="Where to write the captured baseline preset.",
)
@click.option(
    "--out-csv",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    show_default="<spec-name>_params.csv",
    help="Where to write the per-parameter CSV triage table.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing output files (off by default to protect hand-tuned specs).",
)
def main(
    plugin_path: str,
    preset_path: str | None,
    spec_name: str,
    out_spec: str | None,
    out_preset: str | None,
    out_csv: str | None,
    force: bool,
) -> None:
    """Draft a ParamSpec module + baseline preset + CSV table from a VST3 plugin.

    :param plugin_path: Path to the ``.vst3`` bundle to introspect.
    :param preset_path: Optional starting ``.vstpreset`` applied before capture.
    :param spec_name: Registry key; names the emitted constant and default outputs.
    :param out_spec: Draft module destination; defaults from ``spec_name``.
    :param out_preset: Captured preset destination; defaults from ``spec_name``.
    :param out_csv: Per-parameter CSV table destination; defaults from ``spec_name``.
    :param force: Allow overwriting existing output files.
    :raises click.BadParameter: ``spec_name`` is not a valid Python identifier.
    :raises click.UsageError: An output file exists and ``--force`` was not given.
    """
    if not spec_name.isidentifier():
        raise click.BadParameter(
            f"{spec_name!r} is not a Python identifier", param_hint="--spec-name"
        )
    spec_dest = Path(out_spec or f"{spec_name}_param_spec.py")
    preset_dest = Path(out_preset or f"{spec_name}-base.vstpreset")
    csv_dest = Path(out_csv or f"{spec_name}_params.csv")
    # Fail before the (slow) plugin load: re-running with the same spec-name
    # must not clobber a hand-tuned spec.
    for dest in (spec_dest, preset_dest, csv_dest):
        if dest.exists() and not force:
            raise click.UsageError(f"{dest} already exists; pass --force to overwrite")

    vst_plugin = load_plugin(plugin_path)
    if preset_path is not None:
        load_preset(vst_plugin, preset_path)
    # Cast: pedalboard's plugin surface is dynamic, so VST3Plugin's stubs don't
    # declare the attributes IntrospectablePlugin pins structurally.
    plugin = cast(IntrospectablePlugin, vst_plugin)

    drafted, skipped = draft_synth_params(plugin)
    source = render_param_spec_module(
        spec_name,
        plugin_name=plugin.name,
        params=drafted,
        skipped=skipped,
        provenance=_provenance(plugin_path, preset_path),
    )
    # Capture runs before the spec write: under --force a failed capture must
    # leave the existing hand-tuned spec untouched.
    preset_dest.parent.mkdir(parents=True, exist_ok=True)
    capture_preset(plugin, preset_dest)
    spec_dest.parent.mkdir(parents=True, exist_ok=True)
    spec_dest.write_text(source, encoding="utf-8")
    csv_dest.parent.mkdir(parents=True, exist_ok=True)
    csv_dest.write_text(render_param_table_csv(plugin, drafted, skipped), encoding="utf-8")

    click.echo(f"Drafted {len(drafted)} parameter(s), skipped {len(skipped)}.")
    click.echo(f"Spec module : {spec_dest}")
    click.echo(f"Baseline    : {preset_dest}")
    click.echo(f"Param table : {csv_dest}")
    click.echo(
        "Next: hand-tune the spec, then register it under "
        f"{spec_name!r} in synth_setter.data.vst.param_spec_registry."
    )


def _provenance(plugin_path: str, preset_path: str | None) -> str:
    """Build the one-line source description recorded in the emitted module.

    :param plugin_path: Path of the introspected ``.vst3`` bundle.
    :param preset_path: Starting preset applied before drafting, if any.
    :returns: ``plugin: <path> (version <v>), preset: <path or none>``.
    """
    # The version is informational; degrade to "unknown" on the helper's
    # documented failure modes (missing/odd bundle metadata, scan failure).
    try:
        version = extract_renderer_version(Path(plugin_path))
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        click.echo(f"note: could not extract plugin version: {exc}")
        version = "unknown"
    return f"plugin: {plugin_path} (version {version}), preset: {preset_path or 'none'}"


if __name__ == "__main__":
    main()
