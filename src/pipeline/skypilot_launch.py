"""Launch a `generate_dataset` run on RunPod, OCI, or local kind via SkyPilot managed jobs.

Provider-neutral entrypoint: the same binary launches against
`configs/compute/runpod-template.yaml`, `configs/compute/oci-cpu-template.yaml`,
or `configs/compute/local-template.yaml` (kubernetes-via-`sky local up`).
Materializes a spec, ships it via R2 (file_mounts blocked by #749), forwards
worker env via `task.update_envs`, and submits each rank's task to the SkyPilot
managed-jobs controller via `sky.jobs.launch` (see
https://docs.skypilot.co/en/stable/reference/api.html#sky.jobs.launch).

By default the launcher waits for `sky.jobs.launch` + `sky.stream_and_get` to
return a managed-job id per rank (the controller has accepted the job), prints
the `sky jobs logs` / `sky jobs cancel` commands the operator can run, then
exits — without tailing logs and without cancelling successfully-submitted
jobs. Half-submitted jobs (whose `sky.jobs.launch`/`sky.stream_and_get` raised
or yielded no job_id) are still cancelled so the controller doesn't accumulate
orphan state. Pass `--tail` to opt into live `sky.jobs.tail_logs(follow=True)`
and unconditional `finally`-block cancellation of every rank.

Managed jobs differ from cluster-level launches:
- The controller manages provisioning, retries, and teardown automatically.
  There's no per-job autostop window or explicit `down=True` to set — terminal
  status (success / failure / cancel) releases the underlying compute.
- The user-facing identifier is the managed-job *name* (passed to `sky.jobs.*`
  via `name=`), not a cluster name. The launcher's `--cluster-name` flag is the
  job name in this mode; the same value still keys the per-launch R2 spec
  upload.

Per-backend image handling (driven by `--worker-image-tag`):
- RunPod: each Resources entry's `image_id` is pinned to `docker:<image>` before the
  managed-job submission, so the controller's worker provisions from that image.
- OCI: SkyPilot's OCI backend rejects `docker:<image>` for `image_id`, so the
  YAML's `run:` block performs a sub-docker invocation that consumes
  `WORKER_IMAGE` from env. The launcher always injects `WORKER_IMAGE`.

`--num-workers N>1` fans out N independent managed jobs in parallel (neither backend
supports num_nodes>1 for this workload). Each rank gets SYNTH_SETTER_WORKER_RANK /
SYNTH_SETTER_NUM_WORKERS injected; one shared spec → one r2_prefix.
"""

from __future__ import annotations

import functools
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import sky
import sky.jobs  # managed-jobs SDK: sky.jobs.launch / tail_logs / cancel
import yaml
from dotenv import dotenv_values
from hydra import compose, initialize_config_dir
from hydra.errors import HydraException

from src.pipeline.partitioning import NUM_WORKERS_ENV_VAR, WORKER_RANK_ENV_VAR
from src.pipeline.schemas.spec import DatasetSpec
from synth_setter.cli.generate_dataset import spec_from_cfg

# Per-launch R2 key for the materialized spec (file_mounts blocked by #749).
_LAUNCHER_SPEC_R2_PREFIX = "skypilot-launcher-specs"
_WORKER_SPEC_URI_ENV = "WORKER_SPEC_URI"
_WORKER_IMAGE_ENV = "WORKER_IMAGE"
_WORKER_IMAGE_REPO = "tinaudio/synth-setter"

# OCI distribution tag grammar: leading alnum/_, then up to 127 of [A-Za-z0-9_.-].
_DOCKER_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")

# Validates WORKER_GIT_REF when set — must be a 7-40 char hex git SHA. Worker
# templates pass this verbatim into `git fetch + checkout` inside the container.
_WORKER_GIT_REF_RE = re.compile(r"^[0-9a-f]{7,40}$")

# Validates --job-name (and the derived fallback): k8s-label subset — interpolated into a
# tempfile path and an R2 key, so path-separator-free and ≤63 chars. See #876.
_JOB_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")

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

# rclone needs `type` + `provider` to construct the `r2:` remote, but those
# values are constants for Cloudflare R2 — not secrets — so default them
# rather than burdening every workflow / .env with two extra lines. An
# explicit override (env or .env) wins.
_R2_RCLONE_CONSTANTS: dict[str, str] = {
    "RCLONE_CONFIG_R2_TYPE": "s3",
    "RCLONE_CONFIG_R2_PROVIDER": "Cloudflare",
}

# Residual `_WORKER_ENV_KEYS` that are not defaulted by `_R2_RCLONE_CONSTANTS`.
# Used to detect the unconfigured-creds case: the rclone TYPE/PROVIDER constants
# default in, so an "empty" worker_env still has those keys — only this residual
# subset (R2 access creds, WANDB_API_KEY, WORKER_GIT_REF) signals whether
# anything was actually resolved from .env / process env.
_SECRET_WORKER_ENV_KEYS: tuple[str, ...] = tuple(
    k for k in _WORKER_ENV_KEYS if k not in _R2_RCLONE_CONSTANTS
)

_CRED_BOOTSTRAP_SCRIPT = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "skypilot_write_provider_creds.sh"
)

# sky.jobs.tail_logs(follow=True) rc: 0 = SUCCEEDED, 100 = non-SUCCEEDED terminal.
_TAIL_LOGS_RC_SUCCESS = 0

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_EXPERIMENT = "generate_dataset/runpod-smoke-shard"
DEFAULT_TEMPLATE = REPO_ROOT / "configs" / "compute" / "runpod-template.yaml"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"


def _compose_dataset_spec(experiment: str, overrides: list[str]) -> DatasetSpec:
    """Build DatasetSpec via Hydra compose for the named experiment + ad-hoc overrides.

    Uses programmatic ``initialize_config_dir`` + ``compose`` rather than ``@hydra.main`` so
    the launcher's click CLI keeps owning argv parsing. ``cfg.paths.*`` are pinned to the repo
    root because programmatic ``compose()`` doesn't populate ``hydra.runtime.output_dir`` (only
    ``@hydra.main`` does), and ``paths.output_dir = ${hydra:runtime.output_dir}`` would
    otherwise fail to resolve.
    """
    try:
        with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="dataset",
                overrides=[f"experiment={experiment}", *overrides],
            )
    except HydraException as exc:
        # Unknown experiment or malformed override surfaces here; convert the Hydra
        # traceback into a one-line CLI error so the launcher reads as a normal click failure.
        raise click.ClickException(
            f"Hydra compose failed for experiment {experiment!r}: {exc}"
        ) from exc
    cfg.paths.root_dir = str(REPO_ROOT)
    cfg.paths.output_dir = str(REPO_ROOT)
    cfg.paths.work_dir = str(REPO_ROOT)
    return spec_from_cfg(cfg)


# Local directory for the materialized spec written before R2 upload. Tempdir
# so concurrent launches on the same host don't collide.
LOCAL_SPEC_DIR = Path(tempfile.gettempdir())

# `sky local up` uses the kubernetes backend; map both spellings.
_CLOUD_TO_PROVIDER: dict[str, str] = {
    "runpod": "runpod",
    "oci": "oci",
    "kubernetes": "local",
    "k8s": "local",
}

_SKYPILOT_API_SERVER_ENV = "SKYPILOT_API_SERVER_ENDPOINT"


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

    for key, default in _R2_RCLONE_CONSTANTS.items():
        resolved.setdefault(key, default)

    git_ref = resolved.get("WORKER_GIT_REF", "")
    if git_ref and not _WORKER_GIT_REF_RE.match(git_ref):
        raise click.ClickException(
            f"WORKER_GIT_REF must be a 7-40 char hex git SHA, got {git_ref!r}"
        )
    return resolved


def _detect_provider(template_path: Path) -> str:
    """Return the cred-bootstrap `--provider` flag for a Task YAML's first cloud.

    Reads the YAML directly (rather than going through `sky.Task.from_yaml`) so
    each rank's task instantiation isn't burdened with an extra detection load
    and so test fixtures don't need an extra side_effect slot for a probe Task.
    Handles both the flat `resources: { cloud: X }` shape (RunPod, kubernetes)
    and the `resources: { any_of: [{ cloud: X }, ...] }` shape (OCI Flex).
    """
    with template_path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise click.ClickException(
            f"Could not detect cloud from {template_path}; "
            "expected a YAML mapping with a `resources` key, got empty/non-mapping content."
        )
    resources = doc.get("resources") or {}
    if not isinstance(resources, dict):
        raise click.ClickException(
            f"Could not detect cloud from {template_path}; expected `resources` to be a mapping."
        )
    cloud_value = resources.get("cloud")
    if cloud_value is None:
        any_of = resources.get("any_of") or []
        if not isinstance(any_of, list):
            raise click.ClickException(
                f"Could not detect cloud from {template_path}; "
                "expected `resources.any_of` to be a list."
            )
        if any_of:
            first = any_of[0]
            if not isinstance(first, dict):
                raise click.ClickException(
                    f"Could not detect cloud from {template_path}; "
                    "expected `resources.any_of[0]` to be a mapping."
                )
            cloud_value = first.get("cloud")
    if not isinstance(cloud_value, str):
        raise click.ClickException(
            f"Could not detect cloud from {template_path}; "
            "expected resources.cloud (str) or resources.any_of[0].cloud (str)."
        )
    provider = _CLOUD_TO_PROVIDER.get(cloud_value.strip().lower())
    if provider is None:
        raise click.ClickException(
            f"Unsupported cloud {cloud_value!r} in {template_path}; cred bootstrap "
            "supports runpod, oci, and local (kubernetes) only"
        )
    return provider


def _apply_dispatch_mode(api_server: str | None, local: bool) -> None:
    """Apply the launcher's explicit dispatch-mode selection to ``os.environ``.

    This function is the sole enforcer of the ``--api-server`` / ``--local`` contract —
    Click does not natively gate mutually-exclusive options, so the runtime check below
    is what catches both CLI users and programmatic callers. ``--api-server`` exports
    ``SKYPILOT_API_SERVER_ENDPOINT`` (after stripping surrounding whitespace; blank values
    rejected) so all subsequent ``sky.*`` calls dispatch to the remote server.
    ``--local`` clears that env var so an inherited value can't accidentally route
    remote (the failure mode #841 captures). Neither flag passed → leave the env
    untouched (backward-compat).
    """
    if api_server is not None and local:
        raise click.ClickException("--api-server and --local are mutually exclusive")
    if api_server is not None:
        stripped = api_server.strip()
        if not stripped:
            raise click.ClickException("--api-server must be a non-empty URL")
        os.environ[_SKYPILOT_API_SERVER_ENV] = stripped
    elif local:
        os.environ.pop(_SKYPILOT_API_SERVER_ENV, None)


def _run_cred_bootstrap(*, provider: str, env_file_path: Path | None = None) -> None:
    """Invoke `scripts/skypilot_write_provider_creds.sh` for `provider`.

    The script writes cred files to disk and emits no stdout — captured anyway
    via `subprocess.run(capture_output=True)` so even surprise output cannot
    reach a caller's tee'd workflow log.

    When `SKYPILOT_API_SERVER_ENDPOINT` is set the remote API server holds the
    provider creds; the local cred-write is a no-op and this returns early.

    The subprocess inherits `os.environ` merged with `env_file_path` values
    (when provided) so a local-dev `.env` carrying provider creds bootstraps
    cleanly without manual `export`.
    """
    if os.environ.get(_SKYPILOT_API_SERVER_ENV):
        click.echo(
            f"{_SKYPILOT_API_SERVER_ENV} is set; remote API server holds provider "
            "creds, skipping local cred bootstrap",
            err=True,
        )
        return

    env = {**os.environ}
    if env_file_path is not None and env_file_path.is_file():
        env.update(load_worker_env(env_file_path))

    try:
        result = subprocess.run(  # noqa: S603 — controlled args, in-repo script
            ["bash", str(_CRED_BOOTSTRAP_SCRIPT), "--provider", provider],  # noqa: S607 — bash on PATH
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f"cred bootstrap failed (rc={exc.returncode}): {exc.stderr.strip()}"
        ) from exc
    if result.stderr:
        click.echo(result.stderr, err=True)


def upload_spec_to_r2(spec: DatasetSpec, job_name: str) -> str:
    """Upload `spec` to R2 under a per-job key; return the `r2://bucket/key` URI.

    Uses `rclone copyto` (configured via `RCLONE_CONFIG_R2_*` in process env)
    to put the spec at `r2:{spec.r2_bucket}/skypilot-launcher-specs/{job_name}.json`.
    The worker pod's env will get `WORKER_SPEC_URI` pointing at the same URI;
    the worker downloads via `load_spec_from_uri` before parsing.

    Workaround for #749: SkyPilot's RunPod backend rejects programmatic
    `task.update_file_mounts(...)` with a pubkey-overflow at pod-create time,
    so the launcher ships the spec via R2 instead.
    """
    spec_key = f"{_LAUNCHER_SPEC_R2_PREFIX}/{job_name}.json"
    rclone_dest = f"r2:{spec.r2_bucket}/{spec_key}"
    spec_uri = f"r2://{spec.r2_bucket}/{spec_key}"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write(spec.model_dump_json(indent=2))
        local_path = f.name
    try:
        # `copyto` (vs `copy`) treats the destination as a file path, not a
        # directory — the source-basename-preservation behavior of `copy` would
        # land us at `r2:bucket/skypilot-launcher-specs/<job>.json/<tmpname>`
        # which the worker can't address by URI.
        args = [  # noqa: S607 — rclone resolved by host's PATH
            "rclone",
            "copyto",
            "--checksum",
            local_path,
            rclone_dest,
        ]
        # S603 safe: local tempfile path + R2 dest whose job_name is _JOB_NAME_RE-validated
        # in main() before this function is ever called.
        subprocess.check_call(args)  # noqa: S603
    finally:
        Path(local_path).unlink(missing_ok=True)
    return spec_uri


def _warn_if_deprecated_cluster_name() -> None:
    """Warn to stderr if the deprecated ``--cluster-name`` alias was used in argv.

    Click accepts both ``--job-name`` and ``--cluster-name`` (the latter as a
    legacy alias). The two names share a Python parameter, so Click itself
    can't tell us which spelling the caller used — inspect ``sys.argv`` to
    detect the alias and emit a one-line deprecation notice. Cheap, contained,
    and easy to unit-test by manipulating argv via Click's ``CliRunner``.
    """
    if any(a == "--cluster-name" or a.startswith("--cluster-name=") for a in sys.argv[1:]):
        click.echo(
            "DEPRECATION: --cluster-name is deprecated and will be removed in a "
            "future release; use --job-name instead.",
            err=True,
        )


@click.command()
@click.option(
    "--experiment",
    "experiment",
    type=str,
    default=DEFAULT_EXPERIMENT,
    show_default=True,
    help=(
        "Datagen experiment name (e.g. `generate_dataset/runpod-smoke-shard`). Resolved "
        "as Hydra `compose(config_name='dataset', overrides=[f'experiment={name}'])` "
        "against `configs/dataset.yaml`. Use trailing positional args for ad-hoc Hydra "
        "overrides, e.g. `--experiment generate_dataset/ci-materialize-test "
        "render.plugin_path=/path/to/Plugin.vst3`."
    ),
)
@click.argument("hydra_overrides", nargs=-1, type=click.UNPROCESSED)
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
    "--job-name",
    "--cluster-name",
    "job_name",
    type=str,
    default=None,
    help=(
        "Managed-job name (used both as the SkyPilot job identifier and as the R2 key "
        "prefix for the per-launch spec upload). Default: "
        "synth-setter-smoke-<first 8 chars of spec.task_name>. Multi-worker fan-out appends "
        "`-r{i}` per rank. Must match [A-Za-z0-9][A-Za-z0-9_-]{0,62}. "
        "Alias: --cluster-name (deprecated)."
    ),
)
@click.option(
    "--spec-out",
    "spec_out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Where to write the materialized spec JSON. Default: a per-launch path under "
        "$TMPDIR (avoids parallel-run collisions on a shared host)."
    ),
)
@click.option(
    "--num-workers",
    type=int,
    default=None,
    help=(
        "Number of independent managed jobs to fan out in parallel. Defaults to 1 when "
        "omitted; worker count is a launcher concern and is not read from the dataset spec. "
        "RunPod and OCI don't support num_nodes>1 for this workload, so we synthesize "
        "multi-worker partitioning by launching N independent managed jobs and injecting "
        "SYNTH_SETTER_WORKER_RANK / SYNTH_SETTER_NUM_WORKERS per rank. Each rank downloads "
        "the same materialized spec and uses src.pipeline.partitioning.get_my_shards to slice "
        "its share."
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
@click.option(
    "--tail/--no-tail",
    "tail",
    default=False,
    show_default=True,
    help=(
        "Tail managed-job logs and unconditionally cancel every job in `finally`. Default "
        "`--no-tail` waits for `sky.jobs.launch` + `sky.stream_and_get` to return a job_id per "
        "rank (the controller has accepted the job), prints the `sky jobs logs` / "
        "`sky jobs cancel` commands the operator can run, and exits without tailing logs and "
        "without cancelling successfully-submitted jobs — the controller's terminal-status "
        "lifecycle is the safety net. Half-submitted jobs (whose `sky.jobs.launch`/"
        "`sky.stream_and_get` raised or yielded no job_id) are still cancelled in `--no-tail` "
        "so the controller doesn't accumulate orphan state."
    ),
)
@click.option(
    "--api-server",
    "api_server",
    type=str,
    default=None,
    help=(
        "Dispatch to this remote SkyPilot API server URL. Sets SKYPILOT_API_SERVER_ENDPOINT in "
        "the launcher's process env so all sky.* calls go to the remote server, and skips the "
        "local cred bootstrap (the remote server holds provider creds). Mutually exclusive with "
        "--local. When neither is passed the existing env var (if any) is honored."
    ),
)
@click.option(
    "--local",
    "local",
    is_flag=True,
    default=False,
    help=(
        "Force local SDK dispatch. Clears SKYPILOT_API_SERVER_ENDPOINT from the launcher's "
        "process env so an inherited value can't accidentally route remote (#841), and runs "
        "the local cred bootstrap. Mutually exclusive with --api-server."
    ),
)
def main(
    experiment: str,
    hydra_overrides: tuple[str, ...],
    template_path: Path,
    env_file_path: Path,
    job_name: str | None,
    spec_out: Path | None,
    num_workers: int | None,
    worker_image_tag: str,
    tail: bool,
    api_server: str | None,
    local: bool,
) -> None:
    """Launch the smoke `generate_dataset` run via SkyPilot (RunPod or OCI per `--template`)."""
    _warn_if_deprecated_cluster_name()
    _apply_dispatch_mode(api_server=api_server, local=local)

    if job_name is not None and not _JOB_NAME_RE.fullmatch(job_name):
        raise click.ClickException(
            "--job-name must match [A-Za-z0-9][A-Za-z0-9_-]{0,62} "
            "(alphanumerics, underscore, dash; ≤63 chars; no path separators); "
            f"got {job_name!r}"
        )

    if num_workers is not None and num_workers < 1:
        raise click.ClickException(f"--num-workers must be >= 1, got {num_workers}")

    if not _DOCKER_TAG_RE.fullmatch(worker_image_tag):
        raise click.ClickException(
            f"--worker-image-tag must match OCI tag grammar [A-Za-z0-9_][A-Za-z0-9_.-]{{0,127}}; "
            f"got {worker_image_tag!r}"
        )

    worker_env = resolve_worker_env(env_file_path)
    if not any(k in worker_env for k in _SECRET_WORKER_ENV_KEYS):
        raise click.ClickException(
            "No worker env vars resolved. Set the rclone-R2 keys in process env "
            f"(e.g. via `docker run -e RCLONE_CONFIG_R2_*=...`) or populate {env_file_path}. "
            f"Expected at least one of: {', '.join(_SECRET_WORKER_ENV_KEYS)}."
        )

    # rclone subprocess inherits os.environ; mirror launcher-resolved values so .env wins.
    for key, value in worker_env.items():
        if key.startswith("RCLONE_CONFIG_R2_"):
            os.environ[key] = value

    spec = _compose_dataset_spec(experiment, list(hydra_overrides))

    # `--num-workers` overrides the launcher default of 1. Worker count is a
    # launcher concern, no longer baked into the dataset spec.
    resolved_num_workers = num_workers if num_workers is not None else 1

    base_job_name = job_name or f"synth-setter-smoke-{spec.task_name[:8]}"
    # Re-validate the derived default: spec.task_name is only checked for non-blank, so a
    # value containing `/` or `..` would otherwise propagate into the local tempfile path
    # and the R2 object key (path-traversal hardening).
    if not _JOB_NAME_RE.fullmatch(base_job_name):
        raise click.ClickException(
            f"derived job name {base_job_name!r} must match [A-Za-z0-9][A-Za-z0-9_-]{{0,62}} "
            "(spec.task_name contains characters not allowed in a job name; "
            "pass --job-name explicitly or fix spec.task_name)"
        )

    # Per-job filename so parallel launches (CI matrix, local dev concurrent with CI on
    # the same host) don't clobber one another's spec.
    local_spec_path = spec_out or LOCAL_SPEC_DIR / f"skypilot-launch-smoke-{base_job_name}.json"
    local_spec_path.parent.mkdir(parents=True, exist_ok=True)
    # Pin encoding so JSON output is locale-independent (workers/CI run with varied locales).
    local_spec_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    click.echo(f"Materialized spec to {local_spec_path}")

    # Local provider (kind) needs no compute creds; CI writes the controller-resource shrink. See #876.
    provider = _detect_provider(template_path)
    if provider != "local":
        _run_cred_bootstrap(provider=provider, env_file_path=env_file_path)

    # One spec upload, shared across all ranks. Spec is keyed by base job name (no -rN
    # suffix) so all workers in a fan-out group download from the same R2 object and see the
    # same r2_prefix — this is what makes the partition cohere as one logical dataset.
    spec_uri = upload_spec_to_r2(spec, job_name=base_job_name)
    click.echo(f"Spec uploaded to {spec_uri}")
    worker_env[_WORKER_SPEC_URI_ENV] = spec_uri

    # Kubernetes-specific: SkyPilot 0.12 caches enabled-clouds in-process; a
    # CLI `sky check` doesn't always populate the cache the SDK reads, and
    # `sky.jobs.launch` (like `sky.launch`) raises NoCloudAccessError on a fresh
    # runner. Calling `sky.check.check` in-process before launch is the documented
    # workaround (test-skypilot-local.yml). RunPod/OCI source creds from disk on every launch.
    if provider == "local":
        # Deferred so non-kubernetes runs (RunPod / OCI) skip the sky.check submodule import.
        import sky.check

        sky.check.check(clouds=["kubernetes"], quiet=False)

    # Single-worker keeps the unsuffixed managed-job name (the value passed to
    # `sky.jobs.launch(name=...)`) for backward compatibility with debug workflows / CI
    # dashboards that key off it; multi-worker appends -rN per rank.
    job_names = (
        [base_job_name]
        if resolved_num_workers == 1
        else [f"{base_job_name}-r{i}" for i in range(resolved_num_workers)]
    )

    rcs = _run_workers(
        worker_env_base=worker_env,
        template_path=template_path,
        job_names=job_names,
        worker_image_tag=worker_image_tag,
        tail=tail,
    )

    failed = [
        (job_names[i], rcs[i])
        for i in range(resolved_num_workers)
        if rcs[i] != _TAIL_LOGS_RC_SUCCESS
    ]
    if failed:
        raise click.ClickException(
            f"{len(failed)} of {resolved_num_workers} worker(s) failed: "
            + ", ".join(f"{name}(rc={rc})" for name, rc in failed)
        )


def _override_image_id(task: sky.Task, worker_image: str) -> None:
    """Pin every Resources entry's image_id to ``docker:<worker_image>`` for backends that take it.

    SkyPilot's OCI backend rejects ``image_id: docker:<image>`` — that path runs the worker via
    a sub-docker invocation inside the YAML's run: block and consumes WORKER_IMAGE from env, so
    OCI Resources entries are left unmodified. The function unconditionally rebuilds the Task's
    resources collection via ``task.set_resources(...)`` even when no entry was mutated, so
    callers (and mock-based test readers) should expect that call regardless of provider mix.
    """
    from sky.clouds import OCI

    if not task.resources:
        return

    docker_ref = f"docker:{worker_image}"
    new_resources: list[sky.Resources] = []
    for res in task.resources:
        if isinstance(res.cloud, OCI):
            new_resources.append(res)
            continue
        new_resources.append(res.copy(image_id=docker_ref))
    task.set_resources(type(task.resources)(new_resources))


def _run_workers(
    worker_env_base: dict[str, str],
    template_path: Path,
    job_names: list[str],
    worker_image_tag: str,
    tail: bool,
) -> list[int]:
    """Dispatch to the tail- or detach-mode runner; return one rc per rank.

    :param worker_env_base: Env dict forwarded to every rank (rank/world keys are added per call).
    :param template_path: SkyPilot Task YAML to instantiate per rank.
    :param job_names: One managed-job name per rank; ``len()`` defines the world size.
    :param worker_image_tag: Docker image tag under tinaudio/synth-setter to inject.
    :param tail: If True, tail logs and cancel all jobs. If False, detach after launch.
    :return: List with one entry per rank in ``job_names`` order. ``0`` = success;
        ``-1`` = launch/stream raised before the rank's work finished;
        any other non-zero = job failure (with ``tail=True``, the value comes from
        ``sky.jobs.tail_logs``).
    """
    worker_image = f"{_WORKER_IMAGE_REPO}:{worker_image_tag}"
    launch_get_job_id = functools.partial(
        _launch_one_rank,
        job_names=job_names,
        worker_env_base=worker_env_base,
        worker_image=worker_image,
        template_path=template_path,
    )
    if tail:
        return _run_workers_tail(job_names, launch_get_job_id)
    return _run_workers_detached(job_names, launch_get_job_id)


def _launch_one_rank(
    rank: int,
    *,
    job_names: list[str],
    worker_env_base: dict[str, str],
    worker_image: str,
    template_path: Path,
) -> int:
    """Submit rank ``rank``'s managed job and return its ``job_id``.

    Used by ``_run_workers`` to fan out per-rank launches; lives at module level
    rather than as a closure so it can be tested directly without re-running
    the parent ``_run_workers`` setup.

    Raises ``click.ClickException`` if ``sky.jobs.launch`` / ``sky.stream_and_get``
    yields no ``job_id``.
    """
    num_workers = len(job_names)
    job_name = job_names[rank]
    env_for_rank = {
        **worker_env_base,
        WORKER_RANK_ENV_VAR: str(rank),
        NUM_WORKERS_ENV_VAR: str(num_workers),
        _WORKER_IMAGE_ENV: worker_image,
    }
    task = sky.Task.from_yaml(str(template_path))
    _override_image_id(task, worker_image)
    task.update_envs(env_for_rank)
    click.echo(f"[{job_name}] submitting rank={rank}/{num_workers}")
    launch_request_id = sky.jobs.launch(task, name=job_name)
    launch_result = sky.stream_and_get(launch_request_id)
    if launch_result is None:
        raise click.ClickException(
            f"[{job_name}] sky.jobs.launch returned None (no submission handle)"
        )
    job_ids = launch_result[0]
    if not job_ids or job_ids[0] is None:
        raise click.ClickException(
            f"[{job_name}] sky.jobs.launch returned no job_id (empty/null job_ids list)"
        )
    return job_ids[0]


def _run_workers_tail(job_names: list[str], launch_get_job_id: Callable[[int], int]) -> list[int]:
    """Tail-mode runner: tail logs per rank, cancel every job in finally."""
    num_workers = len(job_names)
    rcs: list[int] = [-1] * num_workers

    def _launch_and_tail(rank: int) -> int:
        job_name = job_names[rank]
        job_id = launch_get_job_id(rank)
        click.echo(f"[{job_name}] streaming logs for job {job_id}")
        rc = sky.jobs.tail_logs(job_id=job_id, follow=True)
        click.echo(f"[{job_name}] tail_logs rc={rc}")
        # SDK contract: tail_logs returns None only for follow=False; we pass follow=True.
        if rc is None:
            raise click.ClickException(
                f"[{job_name}] tail_logs returned None with follow=True; job status unknown"
            )
        return rc

    try:
        # Iterate via as_completed so a fast-failing rank surfaces immediately
        # instead of being blocked behind a slower-but-eventually-successful one.
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_rank = {executor.submit(_launch_and_tail, i): i for i in range(num_workers)}
            for fut in as_completed(future_to_rank):
                rank = future_to_rank[fut]
                try:
                    rcs[rank] = fut.result()
                except Exception as exc:  # noqa: BLE001 — keep cancel reachable for every rank.
                    click.echo(f"[{job_names[rank]}] launch or tail raised: {exc}")
                    rcs[rank] = -1
    finally:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for job_name in job_names:
                executor.submit(_cancel_job, job_name)
    return rcs


def _run_workers_detached(
    job_names: list[str], launch_get_job_id: Callable[[int], int]
) -> list[int]:
    """Detach-mode runner: leave successful jobs running; cancel only half-submitted ones."""
    num_workers = len(job_names)
    rcs: list[int] = [-1] * num_workers
    failed_jobs: list[str] = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_rank = {executor.submit(launch_get_job_id, i): i for i in range(num_workers)}
        for fut in as_completed(future_to_rank):
            rank = future_to_rank[fut]
            job_name = job_names[rank]
            try:
                job_id = fut.result()
            except Exception as exc:  # noqa: BLE001 — half-submitted job still needs cleanup.
                click.echo(f"[{job_name}] launch raised: {exc}")
                failed_jobs.append(job_name)
                continue
            click.echo(f"[{job_name}] launched job {job_id} (detached)")
            click.echo(f"  sky jobs logs --name {job_name}")
            click.echo(f"  sky jobs cancel --name {job_name}")
            rcs[rank] = 0

    if failed_jobs:
        with ThreadPoolExecutor(max_workers=len(failed_jobs)) as executor:
            for job_name in failed_jobs:
                executor.submit(_cancel_job, job_name)
    return rcs


def _cancel_job(job_name: str) -> None:
    """Cancel one managed job by name; swallow exceptions so peer cancels keep running."""
    try:
        click.echo(f"[{job_name}] cancelling")
        cancel_request_id = sky.jobs.cancel(name=job_name)
        sky.stream_and_get(cancel_request_id)
    # Managed-jobs cancel can raise multiple specific types (network, controller-state,
    # SDK-internal); blanket except keeps teardown robust across all of them.
    except Exception as exc:  # noqa: BLE001
        click.echo(f"[{job_name}] cancel failed: {exc}")


if __name__ == "__main__":
    main()
