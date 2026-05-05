"""Materialize a DatasetConfig YAML into a spec and upload it to R2.

Local-dev helper for driving a worker container directly (bypassing the
SkyPilot launcher). Prints the resulting `r2://...` URI on stdout so it can
be piped into `scripts/docker_entrypoint.py generate_dataset --spec ...`.

Example:
    python scripts/materialize_and_upload_spec.py \
        --config-in-path configs/dataset/10-1k-shards.yaml \
        --spec-r2-path skypilot-launcher-specs/manual-10-1k.json
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import click

from pipeline.schemas.config import dataset_config_id_from_path, load_dataset_config
from pipeline.schemas.spec import materialize_spec


def _rclone_copyto(local_path: Path, rclone_dest: str) -> None:
    """Upload `local_path` to `rclone_dest` via `rclone copyto --checksum`.

    `copyto` (vs `copy`) treats the destination as a file path, so the object
    lands at exactly `rclone_dest` rather than `rclone_dest/<basename>`.
    """
    args = [  # noqa: S607 — rclone resolved by host's PATH
        "rclone",
        "copyto",
        "--checksum",
        str(local_path),
        rclone_dest,
    ]
    try:
        subprocess.check_call(args)  # noqa: S603 — args from validated inputs
    except FileNotFoundError as exc:
        raise click.ClickException(
            "rclone not found on PATH; install rclone and ensure RCLONE_CONFIG_R2_* "
            "env vars are set before retrying."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f"rclone copyto failed (exit {exc.returncode}): {' '.join(args)}"
        ) from exc


@click.command()
@click.option(
    "--config-in-path",
    "config_in_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to a DatasetConfig YAML (e.g. configs/dataset/10-1k-shards.yaml).",
)
@click.option(
    "--spec-r2-path",
    "spec_r2_path",
    type=str,
    default="skypilot-launcher-specs/manual-spec.json",
    show_default=True,
    help="Destination path within the spec's r2_bucket where the materialized spec is uploaded.",
)
@click.option(
    "--spec-local-path",
    "spec_local_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional local path to keep the materialized spec JSON (default: tmpdir, deleted).",
)
def main(config_in_path: Path, spec_r2_path: str, spec_local_path: Path | None) -> None:
    """Materialize `config_in_path` into a spec, upload to R2, print the r2:// URI."""
    config = load_dataset_config(config_in_path)
    config_id = dataset_config_id_from_path(config_in_path)
    spec = materialize_spec(config, config_id)
    spec_json = spec.model_dump_json(indent=2)

    if spec_local_path is not None:
        spec_local_path.parent.mkdir(parents=True, exist_ok=True)
        spec_local_path.write_text(spec_json, encoding="utf-8")
        local_path = spec_local_path
        cleanup = False
    else:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        tmp.write(spec_json)
        tmp.close()
        local_path = Path(tmp.name)
        cleanup = True

    click.echo(f"Materialized spec: {local_path}", err=True)

    rclone_dest = f"r2:{spec.r2_bucket}/{spec_r2_path}"
    spec_uri = f"r2://{spec.r2_bucket}/{spec_r2_path}"
    try:
        click.echo(f"Uploading -> {rclone_dest}", err=True)
        _rclone_copyto(local_path, rclone_dest)
    finally:
        if cleanup:
            local_path.unlink(missing_ok=True)

    click.echo(spec_uri)


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue] — click injects options
