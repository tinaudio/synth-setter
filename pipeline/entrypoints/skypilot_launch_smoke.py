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

import shutil
import tempfile
import time
from pathlib import Path

import click
import sky
from dotenv import dotenv_values

from pipeline.schemas.config import dataset_config_id_from_path, load_dataset_config
from pipeline.schemas.spec import materialize_spec

_JOB_POLL_INTERVAL_SECONDS = 15
_JOB_DEADLINE_SECONDS = 10 * 60  # bound the poll loop so a stuck job can't block CI forever

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dataset" / "ci-smoke-test.yaml"
DEFAULT_TEMPLATE = REPO_ROOT / "configs" / "compute" / "runpod-template.yaml"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"

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
            f"Copy .env.example to .env and fill in values."
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

    # `sky.launch` is async (returns a RequestId). Launch with both
    # `idle_minutes_to_autostop=0` and `down=True` so the cluster has a predictable 1-min
    # autodown timer (sky internally bumps idle=0 to 1 minute) — `down=True` alone leaves
    # the cluster up if setup errors, by design. `stream_and_get` on the launch request
    # blocks until provisioning + setup + job-submission + run completes, streaming
    # provisioning logs along the way; it returns `(job_id, handle)`. Non-managed launch
    # (sky.launch, not sky.jobs.launch) — managed jobs require a separate cloud-storage
    # backend for controller state, which RunPod doesn't provide.
    #
    # SkyPilot normally consumes mount_source by rename during staging, but a failure
    # between the copyfile above and stage would leave one .mount.json per cluster behind
    # on shared CI runners. Unlink in finally with missing_ok=True to keep the success
    # path quiet. The inner finally drives teardown explicitly so the cluster always goes
    # away on success, exception, or worker error — even if down=True or the autodown
    # timer is silently dropped (observed on RunPod when setup runs cleanly but the job
    # exits via a path that doesn't trigger the autodown scheduler).
    try:
        click.echo(f"Provisioning SkyPilot cluster: {resolved_cluster_name}")
        launch_request_id = sky.launch(
            task,
            cluster_name=resolved_cluster_name,
            idle_minutes_to_autostop=0,
            down=True,
        )
        # follow=True belongs on tail_logs (which actually tails an evolving stream); on a
        # launch RequestId it just keeps stream_and_get open longer than needed. The launch
        # request resolves when provisioning + setup + job submission complete; let it return
        # then.
        launch_result = sky.stream_and_get(launch_request_id)
        if launch_result is None or launch_result[0] is None:
            raise click.ClickException(
                f"Launch yielded no job_id for cluster {resolved_cluster_name}"
            )
        job_id: int = launch_result[0]
        try:
            # Originally we used `sky.tail_logs(..., follow=True)` to block on the worker
            # and stream its stdout. On RunPod, log tailing relies on SSH to the pod, and
            # SSH gets flaky after the workload finishes — tail_logs(follow=True) waits
            # for an EOF that never arrives and the launcher hangs even though the worker
            # has terminated and uploaded artifacts to R2 (verified via `rclone ls`). Poll
            # `sky.queue` for a terminal JobStatus instead, then dump the buffered worker
            # log non-following so a traceback still surfaces in CI.
            click.echo(f"Polling job {job_id} on {resolved_cluster_name} for completion")
            final_status = _wait_for_job(resolved_cluster_name, job_id)
            click.echo(f"Job {job_id} reached terminal status: {final_status.name}")

            click.echo(f"--- Worker log (job {job_id}) ---")
            sky.tail_logs(cluster_name=resolved_cluster_name, job_id=job_id, follow=False)
            click.echo(f"--- End worker log (job {job_id}) ---")

            if final_status != sky.JobStatus.SUCCEEDED:
                raise click.ClickException(
                    f"Worker job {job_id} ended with status {final_status.name}"
                )
        finally:
            # Diagnostic snapshot before teardown — paper trail for the next hang. We log
            # both the per-job status (the source of truth used by _wait_for_job) and the
            # full cluster queue (so we can correlate with anything else SkyPilot has
            # scheduled on this cluster). Errors are swallowed so a failed query never
            # blocks teardown.
            try:
                final_statuses = sky.stream_and_get(
                    sky.job_status(resolved_cluster_name, [job_id])
                )
                click.echo(f"Pre-teardown job_status: {final_statuses}")
            except Exception as e:  # noqa: BLE001 — best-effort diagnostic
                click.echo(f"Pre-teardown job_status query failed: {e}")
            try:
                queue = sky.stream_and_get(sky.queue(resolved_cluster_name))
                click.echo(f"Pre-teardown queue: {queue}")
            except Exception as e:  # noqa: BLE001 — best-effort diagnostic
                click.echo(f"Pre-teardown queue query failed: {e}")
            # Last-ditch dump of any worker output that may have arrived between the
            # polling loop's terminal-detect and the imminent teardown. follow=False so
            # this can never hang. Best-effort: errors are logged and swallowed.
            try:
                click.echo(f"--- Pre-teardown worker log (job {job_id}) ---")
                sky.tail_logs(cluster_name=resolved_cluster_name, job_id=job_id, follow=False)
                click.echo(f"--- End pre-teardown worker log (job {job_id}) ---")
            except Exception as e:  # noqa: BLE001 — best-effort diagnostic
                click.echo(f"Pre-teardown tail_logs failed: {e}")

            click.echo(f"Tearing down cluster: {resolved_cluster_name}")
            down_request_id = sky.down(resolved_cluster_name)
            sky.stream_and_get(down_request_id)
    finally:
        mount_source.unlink(missing_ok=True)


def _wait_for_job(cluster_name: str, job_id: int) -> sky.JobStatus:
    """Poll `sky.job_status` until the given job reaches a terminal status, then return it.

    Used in place of `sky.tail_logs(follow=True)` because the latter hangs on RunPod
    waiting for an SSH-stream EOF that never arrives after the worker exits. Polls every
    `_JOB_POLL_INTERVAL_SECONDS` seconds and times out after `_JOB_DEADLINE_SECONDS` so a
    truly stuck worker can't block CI forever.
    """
    deadline = time.monotonic() + _JOB_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        statuses = sky.stream_and_get(sky.job_status(cluster_name, [job_id])) or {}
        status = statuses.get(job_id)
        if status is None:
            click.echo(f"  job {job_id} not yet visible; retrying")
        else:
            click.echo(f"  status={status.name}")
            if status.is_terminal():
                return status
        time.sleep(_JOB_POLL_INTERVAL_SECONDS)
    raise click.ClickException(
        f"Job {job_id} did not reach a terminal status within "
        f"{_JOB_DEADLINE_SECONDS // 60} minutes"
    )


if __name__ == "__main__":
    main()
