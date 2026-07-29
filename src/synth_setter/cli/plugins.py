"""Manage synth-setter's VST3 packages through the pinned Studiorack CLI.

Run ``synth-setter-plugins install`` to install every manifest pin and create
stable checkout aliases.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import click

from synth_setter.plugin_manager import (
    PluginManifest,
    adopt_plugin_bundle,
    default_plugins_dir,
    install_plugins,
    link_plugin,
    resolve_plugin_bundle,
)

_DEFAULT_MANIFEST = Path("studiorack.json")
_DEFAULT_EXECUTABLE = Path("node_modules/.bin/studiorack")


@click.group()
@click.option(
    "--manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_MANIFEST,
    show_default=True,
    help="Pinned Studiorack project manifest.",
)
@click.option(
    "--plugins-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=default_plugins_dir,
    show_default="$STUDIORACK_PLUGINS_DIR or the user data directory",
    help="Studiorack pluginsDir managed for synth-setter.",
)
@click.option(
    "--links-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("plugins"),
    show_default=True,
    help="Checkout directory receiving stable VST3 aliases.",
)
@click.option(
    "--studiorack-executable",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_EXECUTABLE,
    show_default=True,
    help="Pinned Studiorack CLI executable.",
)
@click.pass_context
def main(
    ctx: click.Context,
    manifest: Path,
    plugins_dir: Path,
    links_dir: Path,
    studiorack_executable: Path,
) -> None:
    """Install, resolve, and link the project's managed VST3 packages.

    :param ctx: Click context carrying validated command configuration.
    :param manifest: Studiorack project manifest.
    :param plugins_dir: Managed package storage root.
    :param links_dir: Stable alias directory.
    :param studiorack_executable: Pinned Studiorack executable.
    """
    ctx.ensure_object(dict)
    ctx.obj.update(
        manifest=PluginManifest.load(manifest),
        plugins_dir=plugins_dir,
        links_dir=links_dir,
        studiorack_executable=studiorack_executable,
    )


@main.command("install")
@click.option(
    "--plugin",
    "packages",
    multiple=True,
    help="Package slug to install; repeat as needed. Installs the full manifest when omitted.",
)
@click.pass_context
def install_command(ctx: click.Context, packages: tuple[str, ...]) -> None:
    """Install exact package pins and create stable checkout aliases.

    :param ctx: Click context configured by the command group.
    :param packages: Selected package slugs; empty selects every package.
    :raises click.ClickException: Installation, resolution, or alias creation fails.
    """
    manifest: PluginManifest = ctx.obj["manifest"]
    try:
        selected = manifest.selected(packages)
        install_plugins(
            selected,
            plugins_dir=ctx.obj["plugins_dir"],
            studiorack_executable=ctx.obj["studiorack_executable"],
        )
        for plugin in selected:
            alias = link_plugin(
                plugin,
                plugins_dir=ctx.obj["plugins_dir"],
                links_dir=ctx.obj["links_dir"],
            )
            click.echo(f"{plugin.reference} -> {alias}")
    except (FileNotFoundError, FileExistsError, KeyError, subprocess.CalledProcessError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("link")
@click.option(
    "--plugin",
    "packages",
    multiple=True,
    help="Package slug to link; repeat as needed. Links the full manifest when omitted.",
)
@click.pass_context
def link_command(ctx: click.Context, packages: tuple[str, ...]) -> None:
    """Refresh stable aliases for already installed Studiorack packages.

    :param ctx: Click context configured by the command group.
    :param packages: Selected package slugs; empty selects every package.
    :raises click.ClickException: Selection, resolution, or alias creation fails.
    """
    manifest: PluginManifest = ctx.obj["manifest"]
    try:
        selected = manifest.selected(packages)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc
    for plugin in selected:
        try:
            alias = link_plugin(
                plugin,
                plugins_dir=ctx.obj["plugins_dir"],
                links_dir=ctx.obj["links_dir"],
            )
        except FileNotFoundError as exc:
            if packages:
                raise click.ClickException(str(exc)) from exc
            click.echo(str(exc), err=True)
            continue
        except FileExistsError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"{plugin.reference} -> {alias}")


@main.command("adopt")
@click.option("--plugin", "package", required=True, help="Manifest package slug.")
@click.option(
    "--bundle-path",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Source/native fallback bundle matching the manifest pin.",
)
@click.pass_context
def adopt_command(ctx: click.Context, package: str, bundle_path: Path) -> None:
    """Adopt an explicitly built fallback into the exact managed namespace.

    :param ctx: Click context configured by the command group.
    :param package: Studiorack package slug represented by the fallback.
    :param bundle_path: Existing VST3 bundle built from the pinned source.
    :raises click.ClickException: The package or bundle cannot be adopted.
    """
    manifest: PluginManifest = ctx.obj["manifest"]
    try:
        managed = adopt_plugin_bundle(
            manifest.resolve(package),
            plugins_dir=ctx.obj["plugins_dir"],
            bundle=bundle_path,
        )
    except (FileNotFoundError, FileExistsError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(managed)


@main.command("resolve")
@click.argument("package")
@click.pass_context
def resolve_command(ctx: click.Context, package: str) -> None:
    """Print the installed VST3 path for one managed package.

    :param ctx: Click context configured by the command group.
    :param package: Studiorack package slug.
    :raises click.ClickException: The package or installed bundle cannot be resolved.
    """
    manifest: PluginManifest = ctx.obj["manifest"]
    try:
        path = resolve_plugin_bundle(manifest.resolve(package), ctx.obj["plugins_dir"])
    except (FileNotFoundError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(path)


if __name__ == "__main__":
    main()
