"""Generic SkyPilot launcher used by ``synth-setter-*`` entrypoints.

``dispatch_via_skypilot(sky_cfg)`` is the programmatic surface. Callers pass a
fully populated ``SkypilotLaunchConfig`` — ``compute_template`` and ``cmd``
required; dataset-specific envs flow through ``sky_cfg.extra_envs``; the
worker job-name stem comes from ``sky_cfg.job_name`` (callers pin a
domain-specific stem) or falls back to ``synth-setter-<uuid8>``. The
``synth-setter-skypilot-launch`` CLI (``main``) wraps it for checked-in launch
configs under ``src/synth_setter/configs/launch/`` (train/eval workflows).

Provider-neutral: the same call launches against
`src/synth_setter/configs/compute/runpod-template.yaml`,
`src/synth_setter/configs/compute/vast-template.yaml`,
`src/synth_setter/configs/compute/oci-cpu-template.yaml`, or
`src/synth_setter/configs/compute/local-template.yaml`
(kubernetes-via-`sky local up`).
Worker env is forwarded via `task.update_envs` (#749 explains why
`task.update_file_mounts` is avoided), and each rank's task is submitted to
the SkyPilot managed-jobs controller — see
https://docs.skypilot.co/en/stable/reference/api.html#sky.jobs.launch

By default the launcher waits for `sky.jobs.launch` + `sky.stream_and_get` to
return a managed-job id per rank (the controller has accepted the job), prints
the `sky jobs logs` / `sky jobs cancel` commands the operator can run, then
exits — without tailing logs and without cancelling successfully-submitted
jobs. Half-submitted jobs (whose `sky.jobs.launch`/`sky.stream_and_get` raised
or yielded no job_id) are still cancelled so the controller doesn't accumulate
orphan state. ``sky_cfg.tail=True`` opts into live ``sky.jobs.tail_logs(follow=True)``
and unconditional ``finally``-block cancellation of every rank.

Managed jobs differ from cluster-level launches:
- The controller manages provisioning, retries, and teardown automatically.
  There's no per-job autostop window or explicit `down=True` to set — terminal
  status (success / failure / cancel) releases the underlying compute.
- The user-facing identifier is the managed-job *name* (passed to `sky.jobs.*`
  via `name=`), not a cluster name.

Per-backend image handling (driven by ``sky_cfg.worker_image_tag``):
- RunPod / Vast: each Resources entry's `image_id` is pinned to `docker:<image>` before
  the managed-job submission, so the controller's worker provisions from that image.
- OCI: SkyPilot's OCI backend rejects `docker:<image>` for `image_id`, so the
  YAML's `run:` block performs a sub-docker invocation that consumes
  `WORKER_IMAGE` from env. The launcher always injects `WORKER_IMAGE`.

``sky_cfg.num_workers > 1`` fans out N independent managed jobs in parallel
(neither backend supports num_nodes>1 for this workload). Each rank gets
SYNTH_SETTER_WORKER_RANK / SYNTH_SETTER_NUM_WORKERS injected.
"""

from __future__ import annotations

import base64
import functools
import os
import re
import subprocess
import tomllib
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import sky
import sky.jobs  # managed-jobs SDK: sky.jobs.launch / tail_logs / cancel
import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ValidationError

from synth_setter.pipeline.partitioning import NUM_WORKERS_ENV_VAR, WORKER_RANK_ENV_VAR
from synth_setter.pipeline.schemas.object_storage import (
    RCLONE_ENV_KEYS,
    RCLONE_REQUIRED_ENV_KEYS,
    RCLONE_STRUCTURAL_DEFAULTS,
    storage_settings_from_sources,
)
from synth_setter.pipeline.schemas.skypilot_launch import (
    ENV_SKYPILOT_API_SERVER_ENDPOINT,
    ENV_SKYPILOT_SERVICE_ACCOUNT_TOKEN,
    SkypilotClientSettings,
    SkypilotLaunchConfig,
    skypilot_client_settings_from_sources,
)
from synth_setter.workspace import operator_workspace

_WORKER_IMAGE_ENV = "WORKER_IMAGE"
_WORKER_IMAGE_REPO = "tinaudio/synth-setter"

# Bare image tag for the worker's wandb provenance — log_wandb_provenance
# reads IMAGE_TAG into wandb.config.image_tag (storage-provenance-spec.md §12).
_IMAGE_TAG_ENV = "IMAGE_TAG"

# OCI distribution tag grammar: leading alnum/_, then up to 127 of [A-Za-z0-9_.-].
_DOCKER_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")

# Validates WORKER_GIT_REF when set — must be a 7-40 char hex git SHA. Worker
# templates pass this verbatim into `git fetch + checkout` inside the container.
_WORKER_GIT_REF_RE = re.compile(r"^[0-9a-f]{7,40}$")

# Validates sky_cfg.job_name: k8s-label subset — interpolated into a tempfile path and
# the SkyPilot managed-job name, so path-separator-free and ≤63 chars. See #876.
_JOB_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")

# Forwarded via task.update_envs; each resolved from .env then process env.
# Keep in sync with the envs: block in
# src/synth_setter/configs/compute/runpod-template.yaml.
_WORKER_ENV_KEYS: tuple[str, ...] = (
    *RCLONE_ENV_KEYS,
    "WANDB_API_KEY",
    # Pod checks out this ref before generate_dataset, bypassing image-bake lag in PR CI.
    "WORKER_GIT_REF",
)

# rclone structural constants (not secrets); produced from storage config projection.
_RCLONE_STRUCTURAL_CONSTANTS: Mapping[str, str] = RCLONE_STRUCTURAL_DEFAULTS

# Secrets-only subset of _WORKER_ENV_KEYS: TYPE/PROVIDER default in, so this
# subset is what signals whether any creds were actually resolved.
_SECRET_WORKER_ENV_KEYS: tuple[str, ...] = RCLONE_REQUIRED_ENV_KEYS


def _env_value_is_set(value: str | None) -> bool:
    return value is not None and value.strip() != ""


# sky.jobs.tail_logs(follow=True) rc: 0 = SUCCEEDED, 100 = non-SUCCEEDED terminal.
_TAIL_LOGS_RC_SUCCESS = 0

_OPERATOR_WORKSPACE = operator_workspace()

# Lives outside the package — packaged installs need $SYNTH_SETTER_WORKSPACE
# to point at a checkout with scripts/skypilot/ present. See #1261.
_CRED_BOOTSTRAP_SCRIPT = _OPERATOR_WORKSPACE / "scripts" / "skypilot" / "write_provider_creds.sh"

DEFAULT_ENV_FILE = _OPERATOR_WORKSPACE / ".env"

# CI-mode gate. Truthy → write the managed-jobs controller shrink so the
# controller pod fits on GHA-kind. Operator local-dev leaves this unset.
_CI_MODE_ENV = "SYNTH_SETTER_CI_MODE"
_CI_MODE_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})

# SkyPilot's default (cpus: 4+, memory: 4x) doesn't fit in GHA-kind's
# ~1950m allocatable CPU after kube-system. See PR #876.
_CI_SKY_CONFIG_YAML = """jobs:
  controller:
    resources:
      cpus: 1+
      memory: 1+
"""


def _ensure_ci_sky_config() -> None:
    """Write ``~/.sky/config.yaml`` with the controller shrink when CI mode is truthy.

    Truthy = ``SYNTH_SETTER_CI_MODE`` ∈ {1, true, yes, on} (case-insensitive).
    Any other value (including ``0``, ``false``, unset) is a no-op, so an
    operator who exports ``SYNTH_SETTER_CI_MODE=0`` doesn't clobber a local
    config.
    """
    if os.environ.get(_CI_MODE_ENV, "").strip().lower() not in _CI_MODE_TRUTHY_VALUES:
        return
    sky_dir = Path.home() / ".sky"
    sky_dir.mkdir(parents=True, exist_ok=True)
    config_path = sky_dir / "config.yaml"
    config_path.write_text(_CI_SKY_CONFIG_YAML, encoding="utf-8")
    config_path.chmod(0o600)


# Floor in USD: below this, RunPod refuses/loses pods mid-run, so launches
# would sit in STARTING indefinitely with no visible cause (#2193).
_RUNPOD_MIN_BALANCE_USD = 5.0

_RUNPOD_BALANCE_QUERY = "query { myself { clientBalance } }"


class _RunpodMyself(BaseModel, strict=True):
    clientBalance: float | int  # noqa: N815 — verbatim RunPod GraphQL field name


class _RunpodBalanceData(BaseModel, strict=True):
    myself: _RunpodMyself


# Strict trust-boundary model for the RunPod balance GraphQL response.
class _RunpodBalanceResponse(BaseModel, strict=True):
    data: _RunpodBalanceData


def _fetch_runpod_balance() -> float | None:
    """Probe the RunPod account balance for the launch preflight.

    Fail-open by contract: missing or unparsable ``~/.runpod/config.toml``, API
    errors, and malformed responses all return ``None`` — a balance probe must
    never block a launch on its own failure. Each failure emits a one-line
    stderr notice (exception class only, never the balance or key).

    :returns: Balance in USD, or ``None`` when undeterminable.
    """
    try:
        import runpod
        from runpod.api.graphql import run_graphql_query

        config_text = (Path.home() / ".runpod" / "config.toml").read_text(encoding="utf-8")
        api_key = tomllib.loads(config_text).get("default", {}).get("api_key")
        if not isinstance(api_key, str) or not api_key:
            click.echo(
                "RunPod balance probe found no api_key in ~/.runpod/config.toml; "
                "continuing fail-open (balance not verified)",
                err=True,
            )
            return None
        runpod.api_key = api_key
        response = _RunpodBalanceResponse(**run_graphql_query(_RUNPOD_BALANCE_QUERY))
        return float(response.data.myself.clientBalance)
    except Exception as exc:  # noqa: BLE001 — fail open, see docstring contract
        click.echo(
            f"RunPod balance probe unavailable ({type(exc).__name__}); "
            "continuing fail-open (balance not verified)",
            err=True,
        )
        return None


def _doc_requests_runpod(doc: Mapping[str, object]) -> bool:
    """Report whether any resources entry of a compute template targets RunPod.

    Scans ``resources.cloud`` and every ``resources.any_of`` entry — provider
    detection keys off ``any_of[0]``, but SkyPilot may satisfy the request with
    any listed alternative, so the balance gate must consider them all.

    :param doc: Parsed top-level YAML mapping for a SkyPilot Task.
    :returns: ``True`` when at least one entry names the ``runpod`` cloud.
    """
    resources = doc.get("resources")
    if not isinstance(resources, dict):
        return False
    clouds: list[object] = [resources.get("cloud")]
    any_of = resources.get("any_of")
    if isinstance(any_of, list):
        clouds.extend(entry.get("cloud") for entry in any_of if isinstance(entry, dict))
    return any(isinstance(c, str) and c.strip().lower() == "runpod" for c in clouds)


def _check_runpod_balance() -> None:
    """Abort a RunPod launch when the account balance is below the preflight floor.

    The error deliberately omits the actual balance — logs are shared and the
    amount is account-sensitive. An unknowable balance fails open.

    :raises RuntimeError: Balance is known and below ``_RUNPOD_MIN_BALANCE_USD``.
    """
    balance = _fetch_runpod_balance()
    if balance is not None and balance < _RUNPOD_MIN_BALANCE_USD:
        raise RuntimeError(
            "insufficient RunPod balance: the account is below the "
            f"${_RUNPOD_MIN_BALANCE_USD:.0f} preflight floor. Top up before "
            "launching (amount withheld from logs)."
        )


# `sky local up` uses the kubernetes backend; map both spellings.
_CLOUD_TO_PROVIDER: dict[str, str] = {
    "runpod": "runpod",
    "oci": "oci",
    "vast": "vast",
    "kubernetes": "local",
    "k8s": "local",
}

_SKYPILOT_API_SERVER_ENV = ENV_SKYPILOT_API_SERVER_ENDPOINT


def load_worker_env(path: Path) -> dict[str, str]:
    """Read worker-side env from a dotenv file using python-dotenv.

    `dotenv_values` returns a dict whose values are `Optional[str]` (a key with no `=` becomes
    `None`); coerce to a plain `dict[str, str]` for `task.update_envs(...)` and skip None entries.
    """
    return {k: v for k, v in dotenv_values(path).items() if v is not None}


def _operator_ssh_dir() -> Path:
    """Return the launching operator's ``~/.ssh`` directory.

    :return: The resolved ``~/.ssh`` path (existence not required).
    """
    return Path.home() / ".ssh"


def _operator_ssh_pubkeys_b64(ssh_dir: Path) -> str:
    """Collect the operator's public keys as a base64 blob for pod authorized_keys.

    Reads ``id_ed25519.pub`` and ``authorized_keys`` under ``ssh_dir`` so the
    launching machine — and every machine already trusted to reach it — can SSH
    into the pods it launches (#2297). Base64 keeps the multiline material safe
    through SkyPilot's env serialization.

    :param ssh_dir: Directory holding the operator's SSH files.
    :return: Base64 of newline-joined unique key lines; ``""`` when none exist.
    """
    lines: list[str] = []
    missing: list[str] = []
    try:
        for name in ("id_ed25519.pub", "authorized_keys"):
            path = ssh_dir / name
            if not path.is_file():
                missing.append(name)
                continue
            # errors="replace" salvages intact key lines from a partially
            # corrupted file; mangled lines can't pass the prefix filter.
            for raw in path.read_bytes().decode("utf-8", errors="replace").splitlines():
                line = raw.strip()
                if line.startswith(("ssh-", "ecdsa-")) and line not in lines:
                    lines.append(line)
    except OSError as exc:
        # Fail open: key forwarding is a convenience and must never block a launch.
        click.echo(f"operator SSH key forwarding skipped: {exc}")
        return ""
    if missing:
        click.echo(f"operator SSH keys: {', '.join(missing)} not found under {ssh_dir}")
    if not lines:
        click.echo("no operator SSH keys collected; pods will not trust this machine")
        return ""
    click.echo(f"forwarding {len(lines)} operator SSH key(s) into pod authorized_keys")
    joined = "\n".join(lines) + "\n"
    return base64.b64encode(joined.encode("utf-8")).decode("ascii")


def resolve_worker_env(env_file: Path | None) -> dict[str, str]:
    """Resolve the launcher's `_WORKER_ENV_KEYS` from .env and process env.

    ``env_file=None`` reads ``DEFAULT_ENV_FILE``. Storage settings are loaded
    from canonical ``SYNTH_SETTER_STORAGE_*`` keys and projected into the
    ``RCLONE_CONFIG_R2_*`` keys the current worker templates consume.
    Blank/whitespace counts as absent, so a `.env` line `KEY=` never forwards
    an empty credential to a worker. Missing storage credentials leave only
    the structural rclone defaults, letting ``dispatch_via_skypilot`` produce
    the user-facing "no object storage settings" error.

    `.env` is the local-dev source of truth; CI flows pass secrets via
    `docker run -e KEY=VAL` and never touch a .env on disk.
    """
    file_env: dict[str, str] = {}
    resolved_env_file = env_file if env_file is not None else DEFAULT_ENV_FILE
    if resolved_env_file.is_file():
        file_env = load_worker_env(resolved_env_file)

    try:
        resolved: dict[str, str] = (
            storage_settings_from_sources(resolved_env_file).to_config().rclone_env()
        )
    except ValidationError:
        resolved = dict(_RCLONE_STRUCTURAL_CONSTANTS)

    for key in ("WANDB_API_KEY", "WORKER_GIT_REF"):
        # First non-blank wins, .env over process env; a blank candidate is
        # skipped (not preferred-then-dropped), so a quoted-whitespace `.env`
        # value can't mask a real process-env fallback.
        for candidate in (file_env.get(key), os.environ.get(key)):
            cleaned = candidate.strip() if candidate else ""
            if cleaned:
                resolved[key] = cleaned
                break

    git_ref = resolved.get("WORKER_GIT_REF", "")
    if git_ref and not _WORKER_GIT_REF_RE.match(git_ref):
        raise click.ClickException(
            f"WORKER_GIT_REF must be a 7-40 char hex git SHA, got {git_ref!r}"
        )
    return resolved


def _reset_skypilot_client_cache() -> None:
    """Clear endpoint values cached by the SkyPilot SDK."""
    from sky.server import common as server_common

    server_common.get_server_url.cache_clear()
    server_common.is_api_server_local.cache_clear()


def _configure_local_skypilot_client() -> None:
    """Drop remote client auth so the SDK targets its default local server.

    Mutates os.environ for the remaining process lifetime by design: the launcher is a one-shot
    CLI, and dispatch owns the process env from here on.
    """
    os.environ.pop(ENV_SKYPILOT_API_SERVER_ENDPOINT, None)
    os.environ.pop(ENV_SKYPILOT_SERVICE_ACCOUNT_TOKEN, None)
    _reset_skypilot_client_cache()


def _configure_remote_skypilot_client(settings: SkypilotClientSettings) -> None:
    """Project validated client auth into the env the SkyPilot SDK reads.

    :param settings: Validated client authentication settings.
    """
    os.environ.update(settings.as_env())
    _reset_skypilot_client_cache()


def _run_cred_bootstrap(*, provider: str, env_file_path: Path | None = None) -> None:
    """Invoke `scripts/skypilot/write_provider_creds.sh` for `provider`.

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


def _detect_provider_from_doc(doc: dict[str, object], source: Path) -> str:
    """Detect the cred-bootstrap provider from an already-parsed YAML mapping.

    :param doc: Parsed top-level YAML mapping for a SkyPilot Task.
    :param source: Path the doc was loaded from; used only in error messages.
    :return: ``--provider`` flag for the cred-bootstrap script.
    :raises ValueError: ``resources`` is missing/malformed or names an
        unsupported cloud.
    """
    resources = doc.get("resources") or {}
    if not isinstance(resources, dict):
        raise ValueError(
            f"Could not detect cloud from {source}; expected `resources` to be a mapping."
        )
    cloud_value = resources.get("cloud")
    if cloud_value is None:
        any_of = resources.get("any_of") or []
        if not isinstance(any_of, list):
            raise ValueError(
                f"Could not detect cloud from {source}; expected `resources.any_of` to be a list."
            )
        if any_of:
            first = any_of[0]
            if not isinstance(first, dict):
                raise ValueError(
                    f"Could not detect cloud from {source}; "
                    "expected `resources.any_of[0]` to be a mapping."
                )
            cloud_value = first.get("cloud")
    if not isinstance(cloud_value, str):
        raise ValueError(
            f"Could not detect cloud from {source}; "
            "expected resources.cloud (str) or resources.any_of[0].cloud (str)."
        )
    provider = _CLOUD_TO_PROVIDER.get(cloud_value.strip().lower())
    if provider is None:
        raise ValueError(
            f"Unsupported cloud {cloud_value!r} in {source}; cred bootstrap "
            "supports runpod, oci, vast, and local (kubernetes) only"
        )
    return provider


_WORKER_CMD_SENTINEL = "${WORKER_CMD}"
_NETWORK_VOLUME_SENTINEL = "${NETWORK_VOLUME}"


def _inject_network_volume(
    task_doc: dict[str, object], network_volume: str | None, *, source: Path
) -> dict[str, object]:
    """Substitute ``${NETWORK_VOLUME}`` in the template's ``volumes:`` values.

    Sentinel and config must agree: a template that mounts the sentinel
    requires ``network_volume``, and a configured name with no sentinel to
    land in is a config error — never a silent no-op against the wrong volume.

    :param task_doc: Parsed compute YAML dict, mutated in place.
    :param network_volume: SkyPilot volume name to mount, or ``None``.
    :param source: Template path, named in error messages only.
    :return: ``task_doc`` with sentinel values substituted.
    :raises ValueError: the sentinel is present without ``network_volume``,
        or ``network_volume`` is set but the template has no sentinel.
    """
    volumes = task_doc.get("volumes")
    sentinel_keys = (
        [k for k, v in volumes.items() if v == _NETWORK_VOLUME_SENTINEL]
        if isinstance(volumes, dict)
        else []
    )
    if sentinel_keys and network_volume is None:
        raise ValueError(
            f"compute template {source} mounts {_NETWORK_VOLUME_SENTINEL} but the "
            "launch config does not set network_volume; set it to the SkyPilot "
            "volume name for the target data center."
        )
    if network_volume is not None and not sentinel_keys:
        raise ValueError(
            f"launch config sets network_volume={network_volume!r} but compute "
            f"template {source} has no {_NETWORK_VOLUME_SENTINEL} in its `volumes:` "
            "mapping to substitute."
        )
    if isinstance(volumes, dict):
        for key in sentinel_keys:
            volumes[key] = network_volume
    return task_doc


def _load_compute_template_with_cmd(template_path: Path, cmd: str) -> dict[str, object]:
    """Load ``template_path`` as YAML and inject ``cmd`` into the ``run:`` block.

    Three branches based on the template's existing ``run:``:

    * **Empty/missing** — set ``run = cmd``.
    * **Contains** ``${WORKER_CMD}`` — substitute ``cmd`` into the sentinel,
      preserving surrounding scaffolding (e.g. OCI's ``sudo docker run …
      bash -c "${WORKER_CMD}"``). Caller must shell-quote the context so the
      substituted string lands as a single argv item.
    * **Non-empty without sentinel** — refuse, rather than silently dropping
      the template's ``run:``. Strip ``run:`` or add the sentinel to opt in.

    :param template_path: Path to a SkyPilot Task YAML.
    :param cmd: Bash command to inject.
    :return: The parsed YAML dict with ``run`` populated.
    :raises FileNotFoundError: ``template_path`` does not point to a file.
    :raises ValueError: top-level YAML is not a mapping, or the template's
        ``run:`` is non-empty and lacks the sentinel.
    """
    if not template_path.is_file():
        raise FileNotFoundError(f"compute template not found: {template_path}")
    with template_path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if doc is None:
        doc = {}
    if not isinstance(doc, dict):
        raise ValueError(
            f"Top-level YAML in {template_path} must be a mapping, got {type(doc).__name__}"
        )
    existing_run = doc.get("run")
    if existing_run in (None, ""):
        doc["run"] = cmd
        return doc
    if not isinstance(existing_run, str):
        raise ValueError(
            f"compute template {template_path} `run:` must be a string, "
            f"got {type(existing_run).__name__}"
        )
    if _WORKER_CMD_SENTINEL in existing_run:
        doc["run"] = existing_run.replace(_WORKER_CMD_SENTINEL, cmd)
        return doc
    raise ValueError(
        f"compute template {template_path} has a non-empty `run:` block, but "
        "skypilot_launch.cmd is also set — cmd cannot be silently dropped. "
        f"Strip the YAML's `run:` section, or substitute {_WORKER_CMD_SENTINEL} "
        "where the worker cmd should land, to opt into the Hydra cmd-injection flow."
    )


def _launch_one_rank_from_doc(
    rank: int,
    *,
    job_names: list[str],
    worker_env_base: dict[str, str],
    worker_image: str,
    task_doc: dict[str, object],
) -> int:
    """Submit one rank, building the ``sky.Task`` from an in-memory YAML dict.

    Uses ``sky.Task.from_yaml_config`` so a cmd-injected dict skips the
    disk roundtrip.

    :param rank: This rank's index into ``job_names``.
    :param job_names: One managed-job name per rank; ``len()`` defines the world size.
    :param worker_env_base: Env dict forwarded to the rank (rank/world keys added here).
    :param worker_image: Resolved ``repo:tag`` Docker image reference.
    :param task_doc: Parsed compute YAML dict (with ``run`` already injected).
    :return: SkyPilot-assigned ``job_id`` for this rank.
    :raises RuntimeError: ``sky.jobs.launch`` / ``sky.stream_and_get`` yielded
        no ``job_id``.
    """
    num_workers = len(job_names)
    job_name = job_names[rank]
    env_for_rank = {
        **worker_env_base,
        WORKER_RANK_ENV_VAR: str(rank),
        NUM_WORKERS_ENV_VAR: str(num_workers),
        _WORKER_IMAGE_ENV: worker_image,
        _IMAGE_TAG_ENV: worker_image.rpartition(":")[2],
    }
    task = sky.Task.from_yaml_config(task_doc)
    _override_image_id(task, worker_image)
    task.update_envs(env_for_rank)
    click.echo(f"[{job_name}] submitting rank={rank}/{num_workers}")
    launch_request_id = sky.jobs.launch(task, name=job_name)
    launch_result = sky.stream_and_get(launch_request_id)
    if launch_result is None:
        raise RuntimeError(f"[{job_name}] sky.jobs.launch returned None (no submission handle)")
    job_ids = launch_result[0]
    if not job_ids or job_ids[0] is None:
        raise RuntimeError(
            f"[{job_name}] sky.jobs.launch returned no job_id (empty/null job_ids list)"
        )
    return job_ids[0]


def _run_workers_from_doc(
    worker_env_base: dict[str, str],
    task_doc: dict[str, object],
    job_names: list[str],
    worker_image_tag: str,
    tail: bool,
) -> list[int]:
    """Fan out one rank per ``job_names`` entry from a pre-built YAML dict.

    :param worker_env_base: Env dict forwarded to every rank (rank/world keys added per call).
    :param task_doc: Parsed compute YAML dict (with ``run`` already injected).
    :param job_names: One managed-job name per rank; ``len()`` defines the world size.
    :param worker_image_tag: Docker image tag under tinaudio/synth-setter to inject.
    :param tail: If True, tail logs and cancel all jobs. If False, detach after launch.
    :return: List with one rc per rank in ``job_names`` order — ``0`` = success,
        non-zero = failure (``-1`` for a half-submitted launch).
    """
    worker_image = f"{_WORKER_IMAGE_REPO}:{worker_image_tag}"
    launch_get_job_id = functools.partial(
        _launch_one_rank_from_doc,
        job_names=job_names,
        worker_env_base=worker_env_base,
        worker_image=worker_image,
        task_doc=task_doc,
    )
    if tail:
        return _run_workers_tail(job_names, launch_get_job_id)
    return _run_workers_detached(job_names, launch_get_job_id)


def dispatch_via_skypilot(sky_cfg: SkypilotLaunchConfig) -> None:
    """Dispatch ``sky_cfg.cmd`` to the SkyPilot template named in ``sky_cfg``.

    ``sky_cfg.compute_template`` and ``sky_cfg.cmd`` must both be non-None.

    :param sky_cfg: Validated launcher config; see ``SkypilotLaunchConfig`` for
        per-field semantics.
    :raises ValueError: degenerate ``sky_cfg``, conflicting ``cmd``/``run:`` pair,
        unresolved worker env vars, or ``extra_envs`` keys colliding with
        ``_WORKER_ENV_KEYS``.
    :raises click.ClickException: SkyPilot client auth is invalid or rejected.
    :raises RuntimeError: one or more ranks did not reach the SUCCEEDED terminal status.
    """
    # Phase 1: pure validation — pinned by test_phase1_failures_skip_phase2_side_effects.
    if not sky_cfg.compute_template:
        raise ValueError("dispatch_via_skypilot requires sky_cfg.compute_template to be set")
    if not sky_cfg.cmd:
        raise ValueError("dispatch_via_skypilot requires sky_cfg.cmd to be set")

    # api_server and local express opposite dispatch modes — accepting both
    # would leave the resolved SKYPILOT_API_SERVER_ENDPOINT non-deterministic.
    if sky_cfg.api_server is not None and sky_cfg.local:
        raise ValueError("api_server and local are mutually exclusive")

    template_path = Path(sky_cfg.compute_template).expanduser().resolve()
    task_doc = _load_compute_template_with_cmd(template_path, sky_cfg.cmd)
    task_doc = _inject_network_volume(task_doc, sky_cfg.network_volume, source=template_path)

    if sky_cfg.job_name is not None and not _JOB_NAME_RE.fullmatch(sky_cfg.job_name):
        raise ValueError(
            f"job_name must match {_JOB_NAME_RE.pattern} (alphanumerics, underscore, dash; "
            f"≤63 chars; no path separators); got {sky_cfg.job_name!r}"
        )

    if not _DOCKER_TAG_RE.fullmatch(sky_cfg.worker_image_tag):
        raise ValueError(
            f"worker_image_tag must match OCI tag grammar {_DOCKER_TAG_RE.pattern}; "
            f"got {sky_cfg.worker_image_tag!r}"
        )

    # Reject extra_envs ↔ resolved-worker-env collisions before merge so a caller
    # can't bypass the .env/process-env resolution path for secrets.
    cred_overlap = sorted(set(sky_cfg.extra_envs) & set(_WORKER_ENV_KEYS))
    if cred_overlap:
        raise ValueError(
            f"extra_envs keys collide with worker-resolved env: {cred_overlap}. "
            "Set these via .env or process env (resolved by resolve_worker_env), "
            "not sky_cfg.extra_envs."
        )

    env_file_path = (
        Path(sky_cfg.env_file).expanduser() if sky_cfg.env_file is not None else DEFAULT_ENV_FILE
    )
    worker_env = resolve_worker_env(env_file_path)
    if not any(_env_value_is_set(worker_env.get(k)) for k in _SECRET_WORKER_ENV_KEYS):
        raise ValueError(
            "No object storage settings resolved. Set SYNTH_SETTER_STORAGE_* in process env "
            f"(e.g. via `docker run -e SYNTH_SETTER_STORAGE_*=...`) or populate "
            f"{env_file_path}. "
            "Expected access key id, secret access key, and endpoint URL."
        )
    worker_env.update(sky_cfg.extra_envs)

    try:
        operator_keys = _operator_ssh_pubkeys_b64(_operator_ssh_dir())
    except (RuntimeError, OSError) as exc:
        # Path.home() raises RuntimeError on hosts with no resolvable home
        # (headless CI containers); a launch must survive that.
        click.echo(f"operator SSH key forwarding skipped: {exc}")
        operator_keys = ""
    if operator_keys:
        worker_env.setdefault("OPERATOR_SSH_PUBKEYS_B64", operator_keys)

    client_settings: SkypilotClientSettings | None = None
    if not sky_cfg.local:
        try:
            client_settings = skypilot_client_settings_from_sources(
                env_file_path,
                api_server_endpoint=sky_cfg.api_server,
            )
        except ValidationError as exc:
            details = "; ".join(str(error["msg"]) for error in exc.errors(include_input=False))
            raise click.ClickException(
                f"Invalid SkyPilot client authentication settings in {env_file_path}: {details}"
            ) from exc

    base_job_name = sky_cfg.job_name or f"synth-setter-{uuid.uuid4().hex[:8]}"

    provider = _detect_provider_from_doc(task_doc, source=template_path)

    # Phase 2: commit — side effects in dependency order.
    _ensure_ci_sky_config()
    if client_settings is None:
        _configure_local_skypilot_client()
    else:
        _configure_remote_skypilot_client(client_settings)

    # Mirror RCLONE_CONFIG_R2_* into os.environ so subprocesses (e.g. SkyPilot's
    # storage backend) inherit creds resolved from .env that were never exported.
    for key, value in worker_env.items():
        if key.startswith("RCLONE_CONFIG_R2_"):
            os.environ[key] = value

    if provider != "local":
        _run_cred_bootstrap(provider=provider, env_file_path=env_file_path)
    # Skip under a remote API server (mirrors _run_cred_bootstrap): the server
    # holds the provider creds, so a local config.toml balance may be stale.
    if _doc_requests_runpod(task_doc) and os.environ.get(_SKYPILOT_API_SERVER_ENV) is None:
        _check_runpod_balance()

    if provider == "local":
        import sky.check

        sky.check.check(clouds=["kubernetes"], quiet=False)

    job_names = (
        [base_job_name]
        if sky_cfg.num_workers == 1
        else [f"{base_job_name}-r{i}" for i in range(sky_cfg.num_workers)]
    )

    rcs = _run_workers_from_doc(
        worker_env_base=worker_env,
        task_doc=task_doc,
        job_names=job_names,
        worker_image_tag=sky_cfg.worker_image_tag,
        tail=sky_cfg.tail,
    )

    failed = [
        (job_names[i], rcs[i])
        for i in range(sky_cfg.num_workers)
        if rcs[i] != _TAIL_LOGS_RC_SUCCESS
    ]
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {sky_cfg.num_workers} worker(s) failed: "
            + ", ".join(f"{name}(rc={rc})" for name, rc in failed)
        )


def load_launch_config(path: Path) -> SkypilotLaunchConfig:
    """Load a checked-in launch-config YAML into a validated ``SkypilotLaunchConfig``.

    The YAML is the full launch description (``cmd`` included) — unlike the
    Hydra ``skypilot_launch`` group, which forbids ``cmd`` because the
    generate-dataset entrypoint builds it from argv. ``extra="forbid"`` on the
    model surfaces config typos instead of silently ignoring them.

    :param path: Path to a YAML file whose top level is a mapping of
        ``SkypilotLaunchConfig`` fields.
    :return: Validated launcher config.
    :raises FileNotFoundError: ``path`` does not point to a file.
    :raises ValueError: top-level YAML is not a mapping, or field validation
        fails (``pydantic.ValidationError`` is a ``ValueError`` subclass).
    """
    if not path.is_file():
        raise FileNotFoundError(f"launch config not found: {path}")
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"launch config {path} must be a YAML mapping, got {type(doc).__name__}")
    return SkypilotLaunchConfig(**doc)


@click.command()
@click.option(
    "--extra-env",
    nargs=2,
    multiple=True,
    metavar="KEY VALUE",
    help="Worker environment entry; repeat to set multiple values.",
)
@click.option(
    "--network-volume",
    default=None,
    metavar="NAME",
    help="SkyPilot volume name overriding the config's network_volume.",
)
@click.argument("launch_config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(
    launch_config: Path,
    extra_env: tuple[tuple[str, str], ...],
    network_volume: str | None,
) -> None:
    """Dispatch the SkyPilot launch config at LAUNCH_CONFIG.

    Relative paths inside the config (``compute_template``, ``env_file``) are
    resolved against the working directory, so run from the repo root.

    :param launch_config: Path to a launch-config YAML (see ``load_launch_config``).
    :param extra_env: Worker environment entries that override config ``extra_envs``.
    :param network_volume: Volume-name override; retargets the launch to another
        data center's volume without editing the config.
    :raises click.ClickException: The config or worker environment fails validation.
    """
    try:
        sky_cfg = load_launch_config(launch_config)
        sky_cfg = SkypilotLaunchConfig.model_validate(
            {
                **sky_cfg.model_dump(),
                "extra_envs": {**sky_cfg.extra_envs, **dict(extra_env)},
                **({"network_volume": network_volume} if network_volume is not None else {}),
            }
        )
    except (ValueError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        dispatch_via_skypilot(sky_cfg)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
