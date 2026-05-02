"""Launch the smoke `generate_dataset` run on RunPod via SkyPilot.

Materializes a `DatasetPipelineSpec` locally from the smoke config, ships the
frozen spec to the worker via R2 (per #749 — RunPod backend rejects
`task.update_file_mounts(...)` with a pubkey-overflow at pod-create time),
forwards worker-side env from a `.env` file via `task.update_envs`, and
launches an unmanaged SkyPilot task (`sky.launch`) that runs the existing
container CLI. Logs stream live via `sky.tail_logs(..., follow=True)`.

With `--num-workers N>1` the launcher fans out N independent single-node
SkyPilot clusters in parallel (RunPod's backend doesn't support `num_nodes
> 1`). Each rank gets ``OVERRIDE_SKYPILOT_NODE_RANK`` /
``OVERRIDE_SKYPILOT_NUM_NODES`` injected via ``task.update_envs`` — the
``OVERRIDE_`` prefix is required because SkyPilot reserves the unprefixed
names and clobbers our injection on each pod (every single-node cluster
sees ``SKYPILOT_NODE_RANK=0`` natively). The spec is materialized +
uploaded to R2 once and shared across ranks, so all workers write shards
under the same ``r2_prefix``. ``pipeline.partitioning.get_my_shards``
slices each worker's shard ownership from the synthetic rank.

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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import sky
from dotenv import dotenv_values

from pipeline.partitioning import RANK_ENV_VAR, WORLD_ENV_VAR
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
    # When set, the pod's `run:` block fetches+checks out this git ref before
    # invoking generate_dataset — unblocks PR-CI from the dev-snapshot image
    # bake lag (image source is N PRs stale; runtime sync brings it to head).
    "WORKER_GIT_REF",
)

# `sky.tail_logs(..., follow=True)` returns 0 on success, 100 if the worker
# job ended in a non-SUCCEEDED terminal status (per sky/core.py docstring).
_TAIL_LOGS_RC_SUCCESS = 0

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dataset" / "runpod-smoke-shard.yaml"
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
    "--num-workers",
    type=int,
    default=1,
    show_default=True,
    help=(
        "Number of single-node SkyPilot clusters to fan out in parallel. RunPod's backend "
        "does not support num_nodes>1, so we synthesize multi-worker partitioning by launching "
        "N independent clusters and injecting OVERRIDE_SKYPILOT_NODE_RANK / OVERRIDE_SKYPILOT_NUM_NODES per "
        "rank. Each cluster downloads the same materialized spec and uses "
        "pipeline.partitioning.get_my_shards to slice its share."
    ),
)
def main(
    config_path: Path,
    template_path: Path,
    env_file_path: Path,
    cluster_name: str | None,
    spec_out: Path | None,
    num_workers: int,
) -> None:
    """Launch the smoke `generate_dataset` run on RunPod via SkyPilot."""
    if num_workers < 1:
        raise click.ClickException(f"--num-workers must be >= 1, got {num_workers}")

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

    base_cluster_name = cluster_name or f"synth-setter-smoke-{config_id[:8]}"

    # Per-cluster filename so parallel launches (CI matrix, local dev concurrent with CI on
    # the same host) don't clobber one another's spec.
    local_spec_path = (
        spec_out or LOCAL_SPEC_DIR / f"skypilot-launch-smoke-{base_cluster_name}.json"
    )
    local_spec_path.parent.mkdir(parents=True, exist_ok=True)
    # Pin encoding so JSON output is locale-independent (workers/CI run with varied locales).
    local_spec_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    click.echo(f"Materialized spec to {local_spec_path}")

    # One spec upload, shared across all ranks. Spec is keyed by base cluster name (no -rN
    # suffix) so all workers in a fan-out group download from the same R2 object and see the
    # same r2_prefix — this is what makes the partition cohere as one logical dataset.
    spec_uri = upload_spec_to_r2(spec, base_cluster_name)
    click.echo(f"Spec uploaded to {spec_uri}")
    worker_env[_WORKER_SPEC_URI_ENV] = spec_uri

    # Single-worker keeps the unsuffixed cluster name for backward compatibility with debug
    # workflows / CI dashboards that key off it; multi-worker uses -rN suffixes.
    cluster_names = (
        [base_cluster_name]
        if num_workers == 1
        else [f"{base_cluster_name}-r{i}" for i in range(num_workers)]
    )

    rcs = _run_workers(
        worker_env_base=worker_env,
        template_path=template_path,
        cluster_names=cluster_names,
    )

    failed = [
        (cluster_names[i], rcs[i]) for i in range(num_workers) if rcs[i] != _TAIL_LOGS_RC_SUCCESS
    ]
    if failed:
        raise click.ClickException(
            f"{len(failed)} of {num_workers} worker(s) failed: "
            + ", ".join(f"{name}(rc={rc})" for name, rc in failed)
        )


def _run_workers(
    worker_env_base: dict[str, str],
    template_path: Path,
    cluster_names: list[str],
) -> list[int]:
    """Launch len(cluster_names) single-node clusters in parallel; return tail_logs rc per rank.

    Each rank gets its own ``sky.Task`` with ``OVERRIDE_SKYPILOT_NODE_RANK`` /
    ``OVERRIDE_SKYPILOT_NUM_NODES`` injected via ``update_envs``. Provisioning + log streaming
    run concurrently in a ``ThreadPoolExecutor`` (one thread per rank), and all clusters get
    torn down in parallel in the finally block regardless of which ranks succeeded.

    A rank's slot in the returned list is ``-1`` if launching/streaming raised before
    ``sky.tail_logs`` could return; the caller treats anything != ``_TAIL_LOGS_RC_SUCCESS`` as
    a failure for that rank.
    """
    num_workers = len(cluster_names)
    rcs: list[int] = [-1] * num_workers

    def _launch_and_tail(rank: int) -> int:
        cluster = cluster_names[rank]
        env_for_rank = {
            **worker_env_base,
            RANK_ENV_VAR: str(rank),
            WORLD_ENV_VAR: str(num_workers),
        }
        task = sky.Task.from_yaml(str(template_path))
        task.update_envs(env_for_rank)
        click.echo(f"[{cluster}] provisioning rank={rank}/{num_workers}")
        launch_request_id = sky.launch(
            task,
            cluster_name=cluster,
            idle_minutes_to_autostop=5,
            down=True,
        )
        launch_result = sky.stream_and_get(launch_request_id)
        if launch_result is None or launch_result[0] is None:
            raise click.ClickException(f"[{cluster}] launch yielded no job_id")
        job_id = launch_result[0]
        click.echo(f"[{cluster}] streaming logs for job {job_id}")
        rc = sky.tail_logs(cluster_name=cluster, job_id=job_id, follow=True)
        click.echo(f"[{cluster}] tail_logs rc={rc}")
        return rc

    try:
        # noqa: BLE001 — must catch any rank-thread exception to keep teardown loop reachable.
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            future_to_rank = {ex.submit(_launch_and_tail, i): i for i in range(num_workers)}
            for fut, rank in future_to_rank.items():
                try:
                    rcs[rank] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    click.echo(f"[{cluster_names[rank]}] launch raised: {exc}")
                    rcs[rank] = -1
    finally:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            for cluster in cluster_names:
                ex.submit(_teardown_cluster, cluster)
    return rcs


def _teardown_cluster(cluster: str) -> None:
    """Tear down a single cluster, swallowing exceptions so other teardowns aren't skipped."""
    try:
        click.echo(f"[{cluster}] tearing down")
        down_request_id = sky.down(cluster)
        sky.stream_and_get(down_request_id)
    except Exception as exc:  # noqa: BLE001 — best-effort, every cluster gets its turn
        click.echo(f"[{cluster}] teardown failed: {exc}")


if __name__ == "__main__":
    main()
