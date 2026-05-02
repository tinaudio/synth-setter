"""Launch the smoke `generate_dataset` run on RunPod via SkyPilot.

Materializes a `DatasetPipelineSpec` locally from the smoke config, ships the
frozen spec into the worker via `task.update_file_mounts`, forwards the
worker-side env from a `.env` file via `task.update_envs`, and launches
an unmanaged SkyPilot task (`sky.launch`) that runs the existing container CLI.

`sky.jobs.launch` (managed jobs) requires a cloud-storage backend for
controller state, which RunPod doesn't provide; cluster-level launch is
sufficient for this single-shard smoke probe.

This is the first scaffolding under the SkyPilot integration epic (#534).
No `compute_config` schema, no `pipeline generate --backend skypilot` CLI —
those land in the Phase A–C PRs.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

import click
import sky
from dotenv import dotenv_values
from sky.exceptions import ClusterNotUpError

from pipeline.schemas.config import dataset_config_id_from_path, load_dataset_config
from pipeline.schemas.spec import DatasetPipelineSpec, materialize_spec

# R2 prefix for launcher-uploaded specs. Each cluster gets a per-name key so
# parallel launches don't collide. The worker downloads the spec from this URI
# (see `pipeline.entrypoints.generate_dataset.load_spec_from_uri`) — we ship
# specs through R2 instead of `task.update_file_mounts(...)` because that
# SkyPilot RunPod-backend code path triggers a pubkey-overflow rejection at
# pod-create time (see #749).
_LAUNCHER_SPEC_R2_PREFIX = "skypilot-launcher-specs"
_WORKER_SPEC_URI_ENV = "WORKER_SPEC_URI"

# Worker-side env vars the launcher forwards via `task.update_envs`. Each is
# resolved from the optional .env file first, then from the launcher's process
# env, then skipped if neither has it. Keep in sync with the `envs:` block in
# `configs/compute/runpod-template.yaml` — the template lists the same keys
# (with empty defaults) so the SkyPilot Task validates as fully-specified
# even when the launcher hasn't filled all of them.
#
# `WORKER_SPEC_URI` is also forwarded but isn't resolved from .env / process
# env — the launcher injects the per-cluster R2 URI it uploads the spec to.
_WORKER_ENV_KEYS: tuple[str, ...] = (
    "RCLONE_CONFIG_R2_TYPE",
    "RCLONE_CONFIG_R2_PROVIDER",
    "RCLONE_CONFIG_R2_ACCESS_KEY_ID",
    "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY",
    "RCLONE_CONFIG_R2_ENDPOINT",
    "WANDB_API_KEY",
)

_JOB_POLL_INTERVAL_SECONDS = 15
_JOB_DEADLINE_SECONDS = 25 * 60  # bound the poll loop so a stuck job can't block CI forever

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dataset" / "ci-smoke-test.yaml"
DEFAULT_TEMPLATE = REPO_ROOT / "configs" / "compute" / "runpod-template.yaml"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"

# Local directory for the materialized spec written before R2 upload. Tempdir
# so concurrent launches on the same host don't collide.
LOCAL_SPEC_DIR = Path(tempfile.gettempdir())


def load_worker_env(path: Path) -> dict[str, str]:
    """Read worker-side env from a dotenv file using python-dotenv.

    `dotenv_values` returns a dict whose values are `Optional[str]` (a key with no `=` becomes
    `None`); coerce to a plain `dict[str, str]` for `task.update_envs(...)` and skip None entries.
    """
    return {k: v for k, v in dotenv_values(path).items() if v is not None}


def resolve_worker_env(env_file: Path | None) -> dict[str, str]:
    """Resolve the launcher's `_WORKER_ENV_KEYS` from .env and process env.

    For each key in `_WORKER_ENV_KEYS`, the value is taken from `env_file` if
    that file exists and the key is set there, else from the launcher's
    process env if set, else skipped. Skipped keys keep the template's
    default (typically the empty string) — `task.update_envs` only overrides
    keys that are actually resolved here.

    `.env` is the local-dev source of truth; CI flows pass secrets via
    `docker run -e KEY=VAL` and never touch a .env on disk.
    """
    file_env: dict[str, str] = {}
    if env_file is not None and env_file.is_file():
        file_env = load_worker_env(env_file)

    resolved: dict[str, str] = {}
    for key in _WORKER_ENV_KEYS:
        if key in file_env:
            resolved[key] = file_env[key]
        elif key in os.environ:
            resolved[key] = os.environ[key]
    return resolved


def upload_spec_to_r2(spec: DatasetPipelineSpec, cluster_name: str) -> str:
    """Upload `spec` to R2 under a per-cluster key; return the `r2://bucket/key` URI.

    Uses `rclone copyto` (configured via `RCLONE_CONFIG_R2_*` in process env)
    to put the spec at `r2:{spec.r2_bucket}/skypilot-launcher-specs/{cluster_name}.json`.
    The worker pod's env will get `WORKER_SPEC_URI` pointing at the same URI;
    the worker downloads via `load_spec_from_uri` before parsing.

    Workaround for #749: SkyPilot's RunPod backend rejects programmatic
    `task.update_file_mounts(...)` with a pubkey-overflow at pod-create time,
    so the launcher ships the spec via R2 instead.
    """
    spec_key = f"{_LAUNCHER_SPEC_R2_PREFIX}/{cluster_name}.json"
    rclone_dest = f"r2:{spec.r2_bucket}/{spec_key}"
    spec_uri = f"r2://{spec.r2_bucket}/{spec_key}"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write(spec.model_dump_json(indent=2))
        local_path = f.name
    try:
        # `copyto` (vs `copy`) treats the destination as a file path, not a
        # directory — the source-basename-preservation behavior of `copy` would
        # land us at `r2:bucket/skypilot-launcher-specs/<cluster>.json/<tmpname>`
        # which the worker can't address by URI.
        args = [  # noqa: S607 — rclone resolved by host's PATH
            "rclone",
            "copyto",
            "--checksum",
            local_path,
            rclone_dest,
        ]
        subprocess.check_call(args)  # noqa: S603 — args from validated spec/cluster_name
    finally:
        Path(local_path).unlink(missing_ok=True)
    return spec_uri


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
    help=(
        "Optional path to a KEY=VALUE env file. Values for the keys in "
        "`_WORKER_ENV_KEYS` are read from this file first, then from process env, "
        "then skipped. CI flows pass secrets via `docker run -e KEY=VAL` and don't "
        "need a .env file on disk; the default is convenient for local dev where "
        "writing secrets to a .env once is easier than re-`export`ing them."
    ),
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
@click.option(
    "--job-deadline-seconds",
    type=int,
    default=_JOB_DEADLINE_SECONDS,
    show_default=True,
    help=(
        "Wall-clock cap on the job-status polling loop. The launcher fails fast if the "
        "worker job has not reached a terminal status within this window — useful in "
        "debug workflows where we want a stuck cluster to surface in seconds rather "
        "than minutes."
    ),
)
def main(
    config_path: Path,
    template_path: Path,
    env_file_path: Path,
    cluster_name: str | None,
    spec_out: Path | None,
    job_deadline_seconds: int,
) -> None:
    """Launch the smoke `generate_dataset` run on RunPod via SkyPilot."""
    worker_env = resolve_worker_env(env_file_path)
    if not worker_env:
        raise click.ClickException(
            "No worker env vars resolved. Set the rclone-R2 keys in process env "
            f"(e.g. via `docker run -e RCLONE_CONFIG_R2_*=...`) or populate {env_file_path}. "
            f"Expected at least one of: {', '.join(_WORKER_ENV_KEYS)}."
        )

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

    # Upload the spec to R2 and pass the URI to the worker via env var. Worker
    # downloads via load_spec_from_uri. Avoids `task.update_file_mounts(...)`
    # which triggers SkyPilot's RunPod-backend pubkey-overflow bug (#749).
    spec_uri = upload_spec_to_r2(spec, resolved_cluster_name)
    click.echo(f"Spec uploaded to {spec_uri}")
    worker_env[_WORKER_SPEC_URI_ENV] = spec_uri

    task = sky.Task.from_yaml(str(template_path))
    task.update_envs(worker_env)

    # `sky.launch` is async (returns a RequestId). Launch with both
    # `idle_minutes_to_autostop=0` and `down=True` so the cluster has a predictable 1-min
    # autodown timer (sky internally bumps idle=0 to 1 minute) — `down=True` alone leaves
    # the cluster up if setup errors, by design. `stream_and_get` on the launch request
    # blocks until provisioning + setup + job-submission + run completes, streaming
    # provisioning logs along the way; it returns `(job_id, handle)`. Non-managed launch
    # (sky.launch, not sky.jobs.launch) — managed jobs require a separate cloud-storage
    # backend for controller state, which RunPod doesn't provide.
    #
    # The inner finally drives teardown explicitly so the cluster always goes away on
    # success, exception, or worker error — even if down=True or the autodown timer is
    # silently dropped (observed on RunPod when setup runs cleanly but the job exits via
    # a path that doesn't trigger the autodown scheduler).
    click.echo(f"Provisioning SkyPilot cluster: {resolved_cluster_name}")
    launch_request_id = sky.launch(
        task,
        cluster_name=resolved_cluster_name,
        idle_minutes_to_autostop=0,
        down=True,
    )
    launch_result = sky.stream_and_get(launch_request_id)
    if launch_result is None or launch_result[0] is None:
        raise click.ClickException(f"Launch yielded no job_id for cluster {resolved_cluster_name}")
    job_id: int = launch_result[0]
    try:
        # `sky.tail_logs(..., follow=True)` hangs on RunPod waiting for an SSH-stream EOF
        # that never arrives after the worker exits. Poll `sky.job_status` for a terminal
        # JobStatus instead, then dump the buffered worker log non-following so a traceback
        # still surfaces in CI.
        click.echo(f"Polling job {job_id} on {resolved_cluster_name} for completion")
        final_status = _wait_for_job(
            resolved_cluster_name, job_id, deadline_seconds=job_deadline_seconds
        )
        click.echo(f"Job {job_id} reached terminal status: {final_status.name}")

        if final_status != sky.JobStatus.SUCCEEDED:
            raise click.ClickException(
                f"Worker job {job_id} ended with status {final_status.name}"
            )
    finally:
        # Always dump the worker log before teardown so a worker traceback
        # surfaces in CI even when the launcher exited via the deadline path.
        try:
            click.echo(f"--- Worker log (job {job_id}) ---")
            sky.tail_logs(cluster_name=resolved_cluster_name, job_id=job_id, follow=False)
            click.echo(f"--- End worker log (job {job_id}) ---")
        except Exception as e:  # noqa: BLE001 — best-effort diagnostic
            click.echo(f"Worker log dump failed: {e}")

        click.echo(f"Tearing down cluster: {resolved_cluster_name}")
        down_request_id = sky.down(resolved_cluster_name)
        sky.stream_and_get(down_request_id)


def _wait_for_job(
    cluster_name: str, job_id: int, deadline_seconds: int = _JOB_DEADLINE_SECONDS
) -> sky.JobStatus:
    """Poll `sky.job_status` until the given job reaches a terminal status, then return it.

    Used in place of `sky.tail_logs(follow=True)` because the latter hangs on RunPod
    waiting for an SSH-stream EOF that never arrives after the worker exits. Polls every
    `_JOB_POLL_INTERVAL_SECONDS` seconds and times out after `deadline_seconds` so a
    truly stuck worker can't block CI forever.

    `sky.job_status` raises `sky.exceptions.ClusterNotUpError` when the cluster is still
    in INIT (provisioning slow) or transitioning. That's a "not yet ready, keep polling"
    signal — not a terminal failure — so swallow it as long as we're inside the deadline.
    The caller's deadline still bounds total wait, so a cluster that genuinely never
    transitions to UP fails on the deadline check below, not on the first job_status call.
    """
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        try:
            statuses = sky.stream_and_get(sky.job_status(cluster_name, [job_id])) or {}
        except ClusterNotUpError as exc:
            click.echo(f"  cluster not yet UP ({exc.cluster_status}); retrying")
            time.sleep(_JOB_POLL_INTERVAL_SECONDS)
            continue
        status = statuses.get(job_id)
        if status is None:
            click.echo(f"  job {job_id} not yet visible; retrying")
        else:
            click.echo(f"  status={status.name}")
            if status.is_terminal():
                return status
        time.sleep(_JOB_POLL_INTERVAL_SECONDS)
    raise click.ClickException(
        f"Job {job_id} did not reach a terminal status within {deadline_seconds} seconds"
    )


if __name__ == "__main__":
    main()
