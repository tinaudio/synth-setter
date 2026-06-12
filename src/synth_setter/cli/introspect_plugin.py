"""``synth-setter-introspect-plugin`` — scaffold a draft ``ParamSpec`` from any VST3.

Loads the plugin via pedalboard, optionally applies a starting preset, then
writes an editable draft spec module plus a captured baseline ``.vstpreset``.
By default the artifacts land as loose files; ``--register`` instead wires
them into a synth-setter checkout — spec module, preset, render config, and
``param_spec_registry`` entries — so committing the result makes the synth
renderable via ``generate_dataset`` (issue #1596).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

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
from synth_setter.data.vst.registration import (
    RegistrationPaths,
    checkout_relative_path,
    find_repo_root,
    is_checkout_root,
    registration_paths,
    registry_with_spec,
    render_config_yaml,
)
from synth_setter.data.vst.verification import verify_registration

_PluginT = TypeVar("_PluginT")

# Native VST3 init can block for minutes (Six Sines ~120 s); heartbeats at
# this cadence keep a slow load distinguishable from a hang.
_LOAD_HEARTBEAT_SECONDS = 30.0


@dataclass(frozen=True)
class _RegisterTarget:
    """Checkout wiring computed before the (slow) plugin load.

    .. attribute :: root

       Checkout root the artifacts land in.

    .. attribute :: paths

       Destination of every artifact under ``root``.

    .. attribute :: registry_source

       The registry source with the spec already registered, written last so
       a failure on any earlier artifact leaves the registry untouched.
    """

    root: Path
    paths: RegistrationPaths
    registry_source: str


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
    "--plugin-name",
    default=None,
    help="Factory class to open from a multi-class .vst3 bundle (e.g. 'Six Sines').",
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
    "--register",
    is_flag=True,
    default=False,
    help=(
        "Wire the outputs into a synth-setter checkout: spec module, preset, CSV, and "
        "render config at their conventional paths, plus param_spec_registry entries."
    ),
)
@click.option(
    "--repo-root",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Checkout root for --register; auto-detected from the cwd when omitted.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing output files (off by default to protect hand-tuned specs).",
)
@click.option(
    "--verify",
    is_flag=True,
    default=False,
    help=(
        "After --register, run the post-draft verification battery (pre-commit gates, "
        "registry import + sample, Hydra compose, classifier audit), write "
        "verify-<spec-name>.md at the checkout root, and exit non-zero on any BLOCK."
    ),
)
@click.option(
    "--load-timeout",
    type=float,
    default=600.0,
    show_default=True,
    help=(
        "Seconds to wait for plugin initialization before failing loudly. Multi-minute "
        "loads are normal for some synths; run GUI-heavy plugins under "
        "src/synth_setter/scripts/run-linux-vst-headless.sh."
    ),
)
def main(
    plugin_path: str,
    plugin_name: str | None,
    preset_path: str | None,
    spec_name: str,
    out_spec: str | None,
    out_preset: str | None,
    out_csv: str | None,
    register: bool,
    repo_root: str | None,
    force: bool,
    verify: bool,
    load_timeout: float,
) -> None:
    """Draft a ParamSpec module + baseline preset + CSV table from a VST3 plugin.

    :param plugin_path: Path to the ``.vst3`` bundle to introspect.
    :param plugin_name: Factory class to open from a multi-class bundle; ``None``
        opens the sole class.
    :param preset_path: Optional starting ``.vstpreset`` applied before capture.
    :param spec_name: Registry key; names the emitted constant and default outputs.
    :param out_spec: Draft module destination; defaults from ``spec_name``.
    :param out_preset: Captured preset destination; defaults from ``spec_name``.
    :param out_csv: Per-parameter CSV table destination; defaults from ``spec_name``.
    :param register: Wire the outputs into a synth-setter checkout instead of
        loose files, registering the spec for ``generate_dataset``.
    :param repo_root: Checkout root for ``--register``; auto-detected when omitted.
    :param force: Allow overwriting existing output files.
    :param verify: Run the post-draft verification battery after ``--register``.
    :param load_timeout: Seconds to wait for plugin initialization.
    :raises click.BadParameter: ``spec_name`` is not a valid Python identifier.
    :raises click.UsageError: An output file exists and ``--force`` was not
        given; ``--register`` was combined with ``--out-*``; ``--verify`` was
        given without ``--register``; no checkout was found; ``spec_name``
        conflicts with an existing registry entry; or the plugin failed to
        load (a multi-class bundle needing ``--plugin-name``, or initialization
        outlasting ``--load-timeout``).
    """
    if not spec_name.isidentifier():
        raise click.BadParameter(
            f"{spec_name!r} is not a Python identifier", param_hint="--spec-name"
        )
    if verify and not register:
        raise click.UsageError(
            "--verify checks the registered checkout wiring; combine it with --register."
        )
    if register:
        target = _resolve_register_target(spec_name, repo_root, out_spec, out_preset, out_csv)
        paths = target.paths
        spec_dest, preset_dest, csv_dest = paths.spec_module, paths.preset, paths.csv
        guarded = (spec_dest, preset_dest, csv_dest, paths.render_config)
    else:
        target = None
        spec_dest = Path(out_spec or f"{spec_name}_param_spec.py")
        preset_dest = Path(out_preset or f"{spec_name}-base.vstpreset")
        csv_dest = Path(out_csv or f"{spec_name}_params.csv")
        guarded = (spec_dest, preset_dest, csv_dest)
    # Fail before the (slow) plugin load: re-running with the same spec-name
    # must not clobber a hand-tuned spec.
    for dest in guarded:
        if dest.exists() and not force:
            raise click.UsageError(f"{dest} already exists; pass --force to overwrite")

    vst_plugin = _load_plugin_loudly(
        plugin_path, plugin_name, load_plugin, timeout_seconds=load_timeout
    )
    if preset_path is not None:
        load_preset(vst_plugin, preset_path)
    # Cast: pedalboard's plugin surface is dynamic, so VST3Plugin's stubs don't
    # declare the attributes IntrospectablePlugin pins structurally.
    plugin = cast(IntrospectablePlugin, vst_plugin)

    drafted, skipped = draft_synth_params(plugin)
    version = _plugin_version(plugin_path)
    source = render_param_spec_module(
        spec_name,
        plugin_name=plugin.name,
        params=drafted,
        skipped=skipped,
        provenance=(f"plugin: {plugin_path} (version {version}), preset: {preset_path or 'none'}"),
        registered=register,
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
    if target is not None:
        _write_register_wiring(target, spec_name, plugin_path, version)
        if verify:
            _run_verification(target, spec_name, plugin)
    else:
        click.echo(
            "Next: hand-tune the spec, then register it under "
            f"{spec_name!r} in synth_setter.data.vst.param_spec_registry "
            "(or re-run with --register)."
        )


def _resolve_register_target(
    spec_name: str,
    repo_root: str | None,
    out_spec: str | None,
    out_preset: str | None,
    out_csv: str | None,
) -> _RegisterTarget:
    """Validate the register-mode invocation and pre-compute the checkout wiring.

    Runs before the plugin load so a conflicting ``spec_name`` or a missing
    checkout fails fast.

    :param spec_name: Registry key for the synth.
    :param repo_root: Operator-supplied checkout root, if any.
    :param out_spec: Must be unset — ``--register`` owns the layout.
    :param out_preset: Must be unset — ``--register`` owns the layout.
    :param out_csv: Must be unset — ``--register`` owns the layout.
    :returns: The resolved checkout wiring.
    :raises click.UsageError: ``--out-*`` was supplied, no checkout was found,
        or the registry rejects ``spec_name``.
    """
    if out_spec or out_preset or out_csv:
        raise click.UsageError(
            "--register writes to the checkout's conventional paths; "
            "drop --out-spec/--out-preset/--out-csv."
        )
    if repo_root is not None:
        root = Path(repo_root)
        if not is_checkout_root(root):
            raise click.UsageError(
                f"{root} is not a synth-setter checkout "
                "(src/synth_setter/data/vst/param_spec_registry.py not found)."
            )
    else:
        found = find_repo_root(Path.cwd())
        if found is None:
            raise click.UsageError(
                "not inside a synth-setter checkout; pass --repo-root <checkout>."
            )
        root = found
    paths = registration_paths(root, spec_name)
    try:
        updated = registry_with_spec(paths.registry.read_text(encoding="utf-8"), spec_name)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    return _RegisterTarget(root=root, paths=paths, registry_source=updated)


def _write_register_wiring(
    target: _RegisterTarget, spec_name: str, plugin_path: str, version: str
) -> None:
    """Write the render config + registry entries and echo the run instructions.

    :param target: Pre-computed checkout wiring.
    :param spec_name: Registry key for the synth.
    :param plugin_path: Plugin path as given on the CLI; recorded relative to the checkout when it
        sits inside it.
    :param version: Plugin version pinned in the render config.
    """
    recorded_path = checkout_relative_path(plugin_path, target.root)
    render_config = target.paths.render_config
    render_config.parent.mkdir(parents=True, exist_ok=True)
    render_config.write_text(
        render_config_yaml(spec_name, plugin_path=recorded_path, renderer_version=version),
        encoding="utf-8",
    )
    target.paths.registry.write_text(target.registry_source, encoding="utf-8")
    click.echo(f"Render cfg  : {render_config}")
    click.echo(f"Registered  : {spec_name!r} in {target.paths.registry}")
    if version == "unknown":
        click.echo(
            f"WARNING: renderer_version is 'unknown' — edit {render_config} to pin the "
            "real plugin version before generating; generate_dataset cross-checks it "
            "against the loaded plugin.",
            err=True,
        )
    click.echo(
        "Next: hand-tune the spec, run `make format`, commit, then render with:\n"
        f"  synth-setter-generate-dataset experiment=generate_dataset/smoke-shard "
        f"render={spec_name}"
    )


def _load_plugin_loudly(
    plugin_path: str,
    plugin_name: str | None,
    loader: Callable[[str, str | None], _PluginT],
    *,
    timeout_seconds: float,
    heartbeat_seconds: float = _LOAD_HEARTBEAT_SECONDS,
) -> _PluginT:
    """Run ``loader`` on a daemon thread, echoing elapsed-time heartbeats until it returns.

    Native VST3 init can block for minutes with no output, indistinguishable
    from a hang — the operator's natural move is Ctrl-C (issue #1676). The
    heartbeat reports progress to stderr; on timeout the load fails loudly
    with the elapsed time, and the daemon thread cannot block process exit.

    :param plugin_path: Path of the ``.vst3`` bundle, echoed in heartbeats.
    :param plugin_name: Factory class forwarded to ``loader``.
    :param loader: The blocking load call (``load_plugin``; injectable in tests).
    :param timeout_seconds: Give up after this long.
    :param heartbeat_seconds: Interval between elapsed-time echoes.
    :returns: The loaded plugin.
    :raises click.UsageError: The load outlasted ``timeout_seconds``, or the
        loader rejected the bundle with a ``ValueError`` (pedalboard lists the
        factory classes of a multi-class bundle this way when ``--plugin-name``
        is absent).
    :raises RuntimeError: Any other load failure, chained to the original.
    """
    outcome: dict[str, Any] = {}

    def _run() -> None:
        """Capture the loader's plugin or exception for the main thread."""
        # Broad catch: the exception is re-raised on the main thread below.
        try:
            outcome["plugin"] = loader(plugin_path, plugin_name)
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=_run, name="plugin-load", daemon=True)
    started = time.monotonic()
    thread.start()
    while True:
        thread.join(min(heartbeat_seconds, timeout_seconds))
        if not thread.is_alive():
            break
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            raise click.UsageError(
                f"{plugin_path} did not finish loading within {timeout_seconds:g}s "
                f"(elapsed {elapsed:.0f}s). Multi-minute initialization is normal for "
                "some synths — raise --load-timeout, and run GUI-heavy plugins under "
                "src/synth_setter/scripts/run-linux-vst-headless.sh."
            )
        click.echo(
            f"still loading {plugin_path}… {elapsed:.0f}s elapsed "
            "(some plugins take minutes to initialize)",
            err=True,
        )
    error = outcome.get("error")
    if isinstance(error, ValueError):
        raise click.UsageError(str(error)) from error
    if isinstance(error, BaseException):
        raise RuntimeError(f"loading {plugin_path} failed: {error!r}") from error
    return outcome["plugin"]


def _run_verification(
    target: _RegisterTarget, spec_name: str, plugin: IntrospectablePlugin
) -> None:
    """Run the post-draft battery, write the findings report, and gate the exit code.

    :param target: The registered checkout wiring.
    :param spec_name: Registry key of the registered synth.
    :param plugin: The still-loaded plugin, for the classifier audit.
    """
    paths = target.paths
    report = verify_registration(target.root, spec_name, plugin)
    report_path = target.root / f"verify-{spec_name}.md"
    artifacts = [paths.spec_module, paths.preset, paths.csv, paths.render_config, paths.registry]
    report_path.write_text(report.to_markdown(artifacts), encoding="utf-8")
    click.echo(f"Verify      : {report.verdict()} ({report_path})")
    if report.blocks:
        click.get_current_context().exit(1)


def _plugin_version(plugin_path: str) -> str:
    """Extract the plugin bundle's version, degrading to ``"unknown"``.

    The version is informational in the provenance line but load-bearing in
    the render config (``generate`` cross-checks it); the fallback keeps the
    draft flowing on the helper's documented failure modes (missing/odd bundle
    metadata, scan failure) and is echoed so the operator can pin it by hand.

    :param plugin_path: Path of the introspected ``.vst3`` bundle.
    :returns: The extracted version string, or ``"unknown"``.
    """
    try:
        return extract_renderer_version(Path(plugin_path))
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        # stderr: keep diagnostics out of the parseable summary on stdout.
        click.echo(f"note: could not extract plugin version: {exc}", err=True)
        return "unknown"


if __name__ == "__main__":
    main()
