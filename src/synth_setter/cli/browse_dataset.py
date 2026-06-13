"""``synth-setter-browse-dataset`` — open Lance shards in the SmooSense browser.

The data pipeline writes single-file Lance shards and splits, which SmooSense's
``sense db`` cannot open directly (it browses Lance *dataset* directories). This
command re-materializes each source as a browsable dataset under one db
directory, then launches ``sense db`` on it. ``r2://`` sources are downloaded
first, so a finalized split in R2 can be inspected in one step.

SmooSense is not a project dependency — it requires Python >=3.11 (above this
project's >=3.10 floor) and is installed as an isolated tool
(``uv tool install -U smoosense``); this command shells out to its ``sense``
binary.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Final

import click

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_browse import build_browse_db, duplicate_stems

_SENSE_INSTALL_HINT: Final = (
    "SmooSense's `sense` command was not found on PATH. It needs Python >=3.11 "
    "(above this project's floor), so install it as an isolated tool:\n"
    "  uv tool install -U smoosense\n"
    "then re-run — or pass --no-launch to only export the browsable dataset."
)


def _assert_distinct_table_names(sources: tuple[str, ...]) -> None:
    """Reject sources that map to the same ``<stem>.lance`` table before any I/O.

    Two ``r2://`` URIs that share a basename also collide on the download path,
    so this check runs first — before any source is fetched.

    :param sources: Raw source paths/URIs as given on the command line.
    :raises click.UsageError: Two or more sources share a filename stem.
    """
    duplicates = duplicate_stems(sources)
    if duplicates:
        raise click.UsageError(
            f"sources collide on table name(s) {duplicates}; rename or browse them separately."
        )


def _resolve_source(source: str, download_dir: Path) -> Path:
    """Return a local shard path, downloading ``r2://`` sources into ``download_dir``.

    :param source: A local ``.lance`` path or an ``r2://`` URI.
    :param download_dir: Scratch dir for downloaded shards (caller-owned).
    :returns: The source path unchanged, or the downloaded copy for ``r2://`` URIs.
    :raises click.UsageError: An ``r2://`` URI has no ``.lance`` filename
        component (e.g. a bare bucket or a path ending in ``..``).
    """
    if source.startswith("r2://"):
        name = Path(source).name
        if not name.endswith(".lance"):
            raise click.UsageError(f"r2:// URI has no .lance filename component: {source!r}")
        r2_io.ensure_r2_env_loaded()
        local = download_dir / name
        r2_io.download_to_path(source, local)
        return local
    return Path(source)


def _launch_sense(db_dir: Path) -> None:
    """Run ``sense db <db_dir>``, failing loudly when the binary is absent.

    :param db_dir: Browse-db root to open.
    :raises click.UsageError: The ``sense`` binary is not on PATH.
    """
    binary = shutil.which("sense")
    if binary is None:
        raise click.UsageError(_SENSE_INSTALL_HINT)
    try:
        # Args are CLI-controlled (resolved binary path + local db dir), not untrusted input.
        subprocess.run([binary, "db", str(db_dir)], check=True)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        raise click.UsageError(
            f"`sense db {db_dir}` exited with status {exc.returncode}."
        ) from exc


@click.command()
@click.argument("sources", nargs=-1, required=True, type=str)
@click.option(
    "--db-dir",
    type=click.Path(file_okay=False, writable=True),
    default=None,
    show_default="a fresh temp directory",
    help="Browse-db root the datasets are written under; created if missing.",
)
@click.option(
    "--launch/--no-launch",
    default=True,
    show_default=True,
    help="Launch `sense db` after exporting (off to only write the datasets).",
)
def main(sources: tuple[str, ...], db_dir: str | None, launch: bool) -> None:
    """Export Lance shards into a browsable db and open it in SmooSense.

    When ``--db-dir`` is omitted the datasets are written to a fresh temp
    directory that is left in place (not cleaned up) so ``sense db`` can read it
    after this command returns; the path is printed.

    :param sources: One or more single-file ``.lance`` shards/splits, given as
        local paths or ``r2://`` URIs; each becomes a table named by its stem.
    :param db_dir: Browse-db root; a fresh temp directory when omitted.
    :param launch: Whether to launch ``sense db`` after exporting (which raises
        ``click.UsageError`` via :func:`_launch_sense` when ``sense`` is absent).
    :raises click.UsageError: Sources collide on a table name, a source is
        missing / is already a Lance dataset directory, or an ``r2://`` download
        fails.
    """
    _assert_distinct_table_names(sources)
    auto_db_root = db_dir is None
    db_root = Path(db_dir) if db_dir else Path(tempfile.mkdtemp(prefix="synth-setter-browse-"))
    with tempfile.TemporaryDirectory(prefix="synth-setter-browse-dl-") as dl_dir:
        try:
            local_sources = [_resolve_source(source, Path(dl_dir)) for source in sources]
            tables = build_browse_db(local_sources, db_root)
        # RuntimeError: r2_io.ensure_r2_env_loaded on absent R2 creds; OSError: an
        # unreadable shard whose metadata fails to open after the is_file() guard.
        except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            # Don't leave the auto-created temp db root behind on a failed export.
            if auto_db_root:
                shutil.rmtree(db_root, ignore_errors=True)
            raise click.UsageError(str(exc)) from exc

    click.echo(f"Exported {len(tables)} table(s) to {db_root}:")
    for table in tables:
        click.echo(f"  {table.name}")

    if launch:
        click.echo(f"Launching SmooSense on {db_root} ...")
        _launch_sense(db_root)
    else:
        click.echo(f"Skipped launch (--no-launch). Browse later with:\n  sense db {db_root}")


if __name__ == "__main__":
    main()
