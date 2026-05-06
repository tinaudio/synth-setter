"""Launch the smoke `generate_dataset` run on RunPod or OCI via SkyPilot.

Provider-neutral entrypoint: the same binary launches against either
`configs/compute/runpod-template.yaml` or `configs/compute/oci-cpu-template.yaml`.
Materializes a spec, ships it via R2 (file_mounts blocked by #749), forwards
worker env via `task.update_envs`, and `sky.launch`-es an unmanaged task with
live `sky.tail_logs(follow=True)`. Cluster-level launch (not jobs.launch) —
neither RunPod nor OCI has a managed-jobs controller backend wired up here.

Per-backend image handling (driven by `--worker-image-tag`):
- RunPod: `docker:<image>` is set on each Resources entry's `image_id`, so SkyPilot
  pulls the image at provision time.
- OCI: SkyPilot's OCI backend rejects `docker:<image>` for `image_id`, so the
  YAML's `run:` block performs a sub-docker invocation that consumes
  `WORKER_IMAGE` from env. The launcher always injects `WORKER_IMAGE`.

`--num-workers N>1` fans out N single-node clusters in parallel (neither backend
supports num_nodes>1 for this workload). Each rank gets SYNTH_SETTER_WORKER_RANK /
SYNTH_SETTER_NUM_WORKERS injected; one shared spec → one r2_prefix.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import sky
from dotenv import dotenv_values

from pipeline.partitioning import NUM_WORKERS_ENV_VAR, WORKER_RANK_ENV_VAR
from pipeline.schemas.config import dataset_config_id_from_path, load_dataset_config
from pipeline.schemas.spec import DatasetPipelineSpec, materialize_spec

# Per-cluster R2 key for the materialized spec (file_mounts blocked by #749).
_LAUNCHER_SPEC_R2_PREFIX = "skypilot-launcher-specs"
_WORKER_SPEC_URI_ENV = "WORKER_SPEC_URI"
_WORKER_IMAGE_ENV = "WORKER_IMAGE"
_WORKER_IMAGE_REPO = "tinaudio/synth-setter"

# OCI distribution tag grammar: leading alnum/_, then up to 127 of [A-Za-z0-9_.-].
_DOCKER_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")

# Validates WORKER_GIT_REF when set — must be a 7-40 char hex git SHA. Worker
# templates pass this verbatim into `git fetch + checkout` inside the container.
_WORKER_GIT_REF_RE = re.compile(r"^[0-9a-f]{7,40}$")

# Forwarded via task.update_envs; each resolved from .env then process env.
# Keep in sync with the envs: block in configs/compute/runpod-template.yaml.
# WORKER_GIT_REF: pod fetches+checks out this ref before generate_dataset, to
# bypass dev-snapshot image-bake lag in PR CI.
_WORKER_ENV_KEYS: tuple[str, ...] = (
    "RCLONE_CONFIG_R2_TYPE",
    "RCLONE_CONFIG_R2_PROVIDER",
    "RCLONE_CONFIG_R2_ACCESS_KEY_ID",
    "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY",
    "RCLONE_CONFIG_R2_ENDPOINT",
    "WANDB_API_KEY",
    "WORKER_GIT_REF",
)

# Migration-window aliases: bare R2_* names mapped to their rclone-prefixed
# canonical form. `resolve_worker_env` accepts either, with a deprecation
# warning when only the bare form is present. Drop in the follow-up PR per #829.
_BARE_TO_RCLONE_R2: dict[str, str] = {
    "R2_ACCESS_KEY_ID": "RCLONE_CONFIG_R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY": "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY",
    "R2_ENDPOINT": "RCLONE_CONFIG_R2_ENDPOINT",
}
_BARE_R2_KEYS: tuple[str, ...] = tuple(_BARE_TO_RCLONE_R2)

_CRED_BOOTSTRAP_SCRIPT = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "skypilot_write_provider_creds.sh"
)
_RCLONE_ENV_PREFIX = "RCLONE_CONFIG_R2_"

# sky.tail_logs(follow=True) rc: 0 = SUCCEEDED, 100 = non-SUCCEEDED terminal.
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

    Migration window (#829): if a rclone-prefixed R2 key is unresolved but its
    bare alias (`R2_ACCESS_KEY_ID` etc.) is set, the bare value is used and a
    deprecation warning is logged. The bare-form fallback is dropped in the
    follow-up PR after CI/docs settle on the prefixed names.

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

    bare_used = _apply_bare_r2_fallback(resolved, file_env)
    if bare_used:
        click.echo(
            f"WARN: resolved R2 credentials from deprecated bare names {sorted(bare_used)}; "
            "rename to RCLONE_CONFIG_R2_* to silence this warning. The bare-form fallback "
            "will be removed in a follow-up PR (see #829).",
            err=True,
        )

    git_ref = resolved.get("WORKER_GIT_REF", "")
    if git_ref and not _WORKER_GIT_REF_RE.match(git_ref):
        raise click.ClickException(
            f"WORKER_GIT_REF must be a 7-40 char hex git SHA, got {git_ref!r}"
        )
    return resolved


def _apply_bare_r2_fallback(resolved: dict[str, str], file_env: dict[str, str]) -> list[str]:
    """Fill missing rclone-prefixed R2 keys in `resolved` from bare aliases.

    Returns the list of bare keys that were used, so the caller can log a
    deprecation warning when any fallback fired. Mutates `resolved` in place.
    """
    used: list[str] = []
    for bare, prefixed in _BARE_TO_RCLONE_R2.items():
        if prefixed in resolved:
            continue
        bare_value = file_env.get(bare) or os.environ.get(bare)
        if bare_value:
            resolved[prefixed] = bare_value
            used.append(bare)
    return used


def _detect_provider(task: sky.Task) -> str:
    """Auto-detect the cred-bootstrap `--provider` flag from a task's resources.

    SkyPilot's cloud classes are imported lazily so the launcher doesn't import
    every cloud backend at module load. The first Resources entry's `cloud`
    determines the provider — every entry in a task's alt-set is the same cloud
    in our templates.
    """
    from sky.clouds import OCI, RunPod

    cloud = next(iter(task.resources)).cloud
    if isinstance(cloud, RunPod):
        return "runpod"
    if isinstance(cloud, OCI):
        return "oci"
    raise click.ClickException(
        f"Unsupported cloud {type(cloud).__name__} in task.resources; cred bootstrap "
        "supports runpod and oci only"
    )


def _run_cred_bootstrap(*, provider: str) -> str:
    """Invoke `scripts/skypilot_write_provider_creds.sh` and return its stdout.

    Stdout carries `RCLONE_CONFIG_R2_*=<value>` lines that the caller parses and
    merges into the worker env. Stderr is forwarded to the launcher's stderr so
    the operator sees the bootstrap's diagnostic output (`::notice::` skip-existing
    lines, `::error::` validation failures, etc.).
    """
    try:
        result = subprocess.run(  # noqa: S603 — controlled args, in-repo script
            ["bash", str(_CRED_BOOTSTRAP_SCRIPT), "--provider", provider],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f"cred bootstrap failed (rc={exc.returncode}): {exc.stderr.strip()}"
        ) from exc
    if result.stderr:
        click.echo(result.stderr, err=True)
    return result.stdout


def _parse_rclone_env_lines(stdout: str) -> dict[str, str]:
    """Parse the bootstrap's stdout into a dict of `RCLONE_CONFIG_R2_*` env vars.

    Lines that don't start with the rclone prefix or aren't `KEY=VALUE` shaped are skipped —
    defensive against future stdout chatter even though the current script only emits env lines.
    """
    env: dict[str, str] = {}
    for line in stdout.splitlines():
        if not line.startswith(_RCLONE_ENV_PREFIX) or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key] = value
    return env


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
        "N independent clusters and injecting SYNTH_SETTER_WORKER_RANK / SYNTH_SETTER_NUM_WORKERS per rank. Each cluster "
        "downloads the same materialized spec and uses pipeline.partitioning.get_my_shards to "
        "slice its share."
    ),
)
@click.option(
    "--worker-image-tag",
    type=str,
    default="dev-snapshot",
    show_default=True,
    help=(
        "Worker Docker image tag (under tinaudio/synth-setter). Injected as WORKER_IMAGE env "
        "for the OCI sub-docker invocation, and as Resources.image_id for backends that accept "
        "`docker:<image>` (e.g. RunPod). OCI's backend rejects `docker:<image>` so its "
        "image_id is left untouched."
    ),
)
def main(
    config_path: Path,
    template_path: Path,
    env_file_path: Path,
    cluster_name: str | None,
    spec_out: Path | None,
    num_workers: int,
    worker_image_tag: str,
) -> None:
    """Launch the smoke `generate_dataset` run via SkyPilot (RunPod or OCI per `--template`)."""
    if num_workers < 1:
        raise click.ClickException(f"--num-workers must be >= 1, got {num_workers}")

    if not _DOCKER_TAG_RE.fullmatch(worker_image_tag):
        raise click.ClickException(
            f"--worker-image-tag must match OCI tag grammar [A-Za-z0-9_][A-Za-z0-9_.-]{{0,127}}; "
            f"got {worker_image_tag!r}"
        )

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

    # Cred bootstrap is launcher-host scoped (writes ~/.cloudflare/, ~/.runpod/,
    # ~/.oci/) and runs once per launcher invocation regardless of fan-out width.
    # Provider auto-detected from the template's `resources.cloud`. The rclone
    # env vars on stdout are merged into the per-rank env so the spec-upload
    # rclone subprocess sees rclone-style creds without manual bridging.
    provider = _detect_provider(sky.Task.from_yaml(str(template_path)))
    bootstrap_stdout = _run_cred_bootstrap(provider=provider)
    worker_env = {**worker_env, **_parse_rclone_env_lines(bootstrap_stdout)}

    rcs = _run_workers(
        worker_env_base=worker_env,
        template_path=template_path,
        cluster_names=cluster_names,
        worker_image_tag=worker_image_tag,
    )

    failed = [
        (cluster_names[i], rcs[i]) for i in range(num_workers) if rcs[i] != _TAIL_LOGS_RC_SUCCESS
    ]
    if failed:
        raise click.ClickException(
            f"{len(failed)} of {num_workers} worker(s) failed: "
            + ", ".join(f"{name}(rc={rc})" for name, rc in failed)
        )


def _override_image_id(task: sky.Task, worker_image: str) -> None:
    """Pin every Resources entry's image_id to ``docker:<worker_image>`` for backends that take it.

    SkyPilot's OCI backend rejects ``image_id: docker:<image>`` — that path runs the worker via
    a sub-docker invocation inside the YAML's run: block and consumes WORKER_IMAGE from env, so
    OCI Resources are left untouched here.
    """
    from sky.clouds import OCI

    docker_ref = f"docker:{worker_image}"
    new_resources: list[sky.Resources] = []
    mutated = False
    for res in task.resources:
        if isinstance(res.cloud, OCI):
            new_resources.append(res)
            continue
        new_resources.append(res.copy(image_id=docker_ref))
        mutated = True
    if mutated:
        task.set_resources(type(task.resources)(new_resources))


def _run_workers(
    worker_env_base: dict[str, str],
    template_path: Path,
    cluster_names: list[str],
    worker_image_tag: str,
) -> list[int]:
    """Launch len(cluster_names) single-node clusters in parallel; return tail_logs rc per rank.

    Each rank's task gets SYNTH_SETTER_WORKER_RANK / SYNTH_SETTER_NUM_WORKERS injected.
    All clusters tear down in the finally block regardless of which ranks succeeded.
    A rank's slot in the result is ``-1`` if launch/stream raised before tail_logs returned.

    Cred bootstrap runs once before any sky.launch — provider auto-detected from
    the task's resources.cloud — and the rclone-prefixed R2 env vars it emits are
    merged into every rank's update_envs payload.

    Args:
        worker_env_base: Env dict forwarded to every rank (rank/world keys are added per call).
        template_path: SkyPilot Task YAML to instantiate per rank.
        cluster_names: One name per rank; ``len()`` defines the world size.
        worker_image_tag: Docker image tag under tinaudio/synth-setter to inject.

    Returns:
        Per-rank tail_logs return code (``0`` = SUCCEEDED, anything else = failed).
    """
    num_workers = len(cluster_names)
    rcs: list[int] = [-1] * num_workers
    worker_image = f"{_WORKER_IMAGE_REPO}:{worker_image_tag}"

    def _launch_and_tail(rank: int) -> int:
        cluster = cluster_names[rank]
        env_for_rank = {
            **worker_env_base,
            WORKER_RANK_ENV_VAR: str(rank),
            NUM_WORKERS_ENV_VAR: str(num_workers),
            _WORKER_IMAGE_ENV: worker_image,
        }
        task = sky.Task.from_yaml(str(template_path))
        _override_image_id(task, worker_image)
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
        # Iterate via as_completed so a fast-failing rank surfaces immediately
        # instead of being blocked behind a slower-but-eventually-successful one.
        # noqa: BLE001 — must catch any rank-thread exception to keep teardown loop reachable.
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_rank = {executor.submit(_launch_and_tail, i): i for i in range(num_workers)}
            for fut in as_completed(future_to_rank):
                rank = future_to_rank[fut]
                try:
                    rcs[rank] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    click.echo(f"[{cluster_names[rank]}] launch raised: {exc}")
                    rcs[rank] = -1
    finally:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for cluster in cluster_names:
                executor.submit(_teardown_cluster, cluster)
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
