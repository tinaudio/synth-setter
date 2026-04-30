"""Launch the smoke `generate_dataset` run on RunPod via SkyPilot.

Materializes a `DatasetPipelineSpec` locally from the smoke config, ships the
frozen spec into the worker via `task.update_file_mounts`, forwards the
worker-side env from a `.env.cloud` file via `task.update_envs`, and launches
an unmanaged SkyPilot task (`sky.launch`) that runs the existing container CLI.

`sky.jobs.launch` (managed jobs) requires a cloud-storage backend for
controller state, which RunPod doesn't provide; cluster-level launch is
sufficient for this single-shard smoke probe.

This is the first scaffolding under the SkyPilot integration epic (#534).
No `compute_config` schema, no `pipeline generate --backend skypilot` CLI —
those land in the Phase A–C PRs.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import click
import sky
from dotenv import dotenv_values

from pipeline.schemas.config import dataset_config_id_from_path, load_dataset_config
from pipeline.schemas.spec import materialize_spec

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dataset" / "ci-smoke-test.yaml"
DEFAULT_TEMPLATE = REPO_ROOT / "configs" / "compute" / "runpod-template.yaml"
DEFAULT_ENV_FILE = REPO_ROOT / ".env.cloud"

# Worker-side mount destination. The image's WORKDIR is /home/build/synth-setter (Dockerfile),
# so the spec lands under <repo_root>/data/ on the worker — gitignored, no clash with repo
# files, and portable if the container layout changes later.
WORKER_REPO_ROOT = "/home/build/synth-setter"
WORKER_SPEC_PATH = f"{WORKER_REPO_ROOT}/data/skypilot-launch-smoke-spec.json"

# Local-source directory for the materialized spec. Lives under the system tempdir
# (not the repo) so it shares a filesystem with SkyPilot's staging dir (also under
# /tmp). Inside the dev-snapshot container the repo workspace is bind-mounted from
# the runner host while /tmp is on the container's overlay — crossing those two
# with `os.rename` raises EXDEV (Errno 18) when SkyPilot stages mounts.
LOCAL_SPEC_DIR = Path(tempfile.gettempdir())


def load_worker_env(path: Path) -> dict[str, str]:
    """Read worker-side env from a dotenv file using python-dotenv.

    `dotenv_values` returns a dict whose values are `Optional[str]` (a key with no `=` becomes
    `None`); coerce to a plain `dict[str, str]` for `task.update_envs(...)` and skip None entries.
    """
    return {k: v for k, v in dotenv_values(path).items() if v is not None}


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_CONFIG,
    show_default=True,
    help="Path to a DatasetConfig YAML.",
)
@click.option(
    "--template",
    "template_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_TEMPLATE,
    show_default=True,
    help="Path to the SkyPilot task YAML template.",
)
@click.option(
    "--env-file",
    "env_file_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_ENV_FILE,
    show_default=True,
    help="Path to a KEY=VALUE env file forwarded to the worker.",
)
@click.option(
    "--cluster-name",
    type=str,
    default=None,
    help="SkyPilot cluster name (default: synth-setter-smoke-<config_id[:8]>).",
)
@click.option(
    "--spec-out",
    "spec_out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Where to write the materialized spec JSON. Default: a per-cluster path under "
        "$TMPDIR (avoids parallel-run collisions on a shared host)."
    ),
)
def main(
    config_path: Path,
    template_path: Path,
    env_file_path: Path,
    cluster_name: str | None,
    spec_out: Path | None,
) -> None:
    """Launch the smoke `generate_dataset` run on RunPod via SkyPilot."""
    if not env_file_path.is_file():
        raise click.ClickException(
            f"Worker env file not found: {env_file_path}. "
            f"Copy .env.cloud.example to .env.cloud and fill in values."
        )

    worker_env = load_worker_env(env_file_path)
    if not worker_env:
        raise click.ClickException(f"No env vars parsed from {env_file_path}.")

    config = load_dataset_config(config_path)
    config_id = dataset_config_id_from_path(config_path)
    spec = materialize_spec(config, config_id)

    resolved_cluster_name = cluster_name or f"synth-setter-smoke-{config_id[:8]}"
    # Per-cluster filename so parallel launches (CI matrix, local dev concurrent with CI on
    # the same host) don't clobber one another's spec.
    local_spec_path = (
        spec_out or LOCAL_SPEC_DIR / f"skypilot-launch-smoke-{resolved_cluster_name}.json"
    )

    local_spec_path.parent.mkdir(parents=True, exist_ok=True)
    # Pin encoding so JSON output is locale-independent (workers/CI run with varied locales).
    local_spec_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    click.echo(f"Materialized spec to {local_spec_path}")

    # SkyPilot stages file_mount sources by moving them into its own staging dir; passing
    # local_spec_path directly would leave nothing behind for downstream consumers (CI
    # artifact upload, validate-spec). Stage a sibling copy and mount that instead.
    mount_source = local_spec_path.with_suffix(".mount.json")
    shutil.copyfile(local_spec_path, mount_source)

    task = sky.Task.from_yaml(str(template_path))
    task.update_envs(worker_env)
    task.update_file_mounts({WORKER_SPEC_PATH: str(mount_source)})

    # SkyPilot 0.12 made sky.launch async (it returns a RequestId backed by the local API
    # server). `stream_and_get` blocks until the *launch request* resolves — i.e., the cluster
    # is provisioned, file_mounts are synced, setup runs, and the job is submitted — and
    # streams provisioning logs to stdout along the way. It does NOT wait for the submitted
    # job to actually run to completion, and it does NOT stream the worker's own stdout/stderr.
    # `down=True` tears the cluster down after the run.
    # Non-managed launch (sky.launch, not sky.jobs.launch) — managed jobs require a separate
    # cloud-storage backend for controller state, which RunPod doesn't provide.
    click.echo(f"Provisioning SkyPilot cluster: {resolved_cluster_name}")
    launch_request_id = sky.launch(task, cluster_name=resolved_cluster_name, down=True)
    job_id, _ = sky.stream_and_get(launch_request_id, follow=True)

    # Block on the worker job and stream its stdout/stderr to ours so a worker traceback
    # surfaces here (instead of as a downstream "shard not in R2" error). tail_logs is
    # synchronous when preload_content=True (the default), returns the worker's exit code,
    # and follows the log to completion.
    click.echo(f"Streaming worker job {job_id} logs from {resolved_cluster_name}")
    exit_code = sky.tail_logs(cluster_name=resolved_cluster_name, job_id=job_id, follow=True)
    if exit_code != 0:
        raise click.ClickException(f"Worker job {job_id} exited with code {exit_code}")


if __name__ == "__main__":
    main()
