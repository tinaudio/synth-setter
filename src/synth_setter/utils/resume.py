"""Auto-resume checkpoint discovery for ``synth-setter-train`` (#1991).

Discovery logic: given a composed train cfg, find the newest usable
``last.ckpt`` for the run's config_id across local sibling run dirs, the R2
mid-run mirrors, and the train-end W&B model artifact. The winning R2/W&B tier
performs a network fetch; everything else is pure. The imperative wiring
(setting ``cfg.ckpt_path``, pinning the recovered W&B run id) lives in
``synth_setter.cli.train``.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# Recovery namespaces are ``{run_id}-{uuid4().hex}`` (cli.train._make_recovery_namespace).
_RECOVERY_NAMESPACE_RE = re.compile(r"^(?P<run_id>.+)-[0-9a-f]{32}$")
# Canonical run ids are ``{config_id}-<YYYYMMDD>T<HHMMSSmmm>Z`` (run_id.make_wandb_run_id).
_RUN_ID_TIMESTAMP_RE = r"\d{8}T\d{9}Z"

_DISABLED_VALUES = (None, False, "off")
_ACTIVE_MODES = ("auto", "require")

# Failures that degrade a best-effort R2/W&B tier to "not found" instead of
# aborting the launch; anything outside this set is a programming error and raises.
_DEGRADABLE_ERRORS = (RuntimeError, OSError, ValueError, subprocess.CalledProcessError)

# Bounds W&B graphql calls so a hung API cannot block a training launch.
_WANDB_API_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class ResumeDecision:
    """One discovered resume source.

    .. attribute :: ckpt_path

        Local path of the ``last.ckpt`` to hand to ``trainer.fit``.

    .. attribute :: wandb_run_id

        Run id of the launch that wrote the checkpoint, when recoverable —
        reused so the W&B history continues on one run page.

    .. attribute :: source

        Discovery tier the checkpoint came from.
    """

    ckpt_path: Path
    wandb_run_id: str | None
    source: Literal["local", "r2", "wandb-artifact"]


def resolve_resume_mode(cfg: DictConfig) -> Literal["auto", "require"] | None:
    """Validate ``training.resume`` and return the active mode.

    :param cfg: Composed train cfg; reads ``training.resume`` and ``ckpt_path``.
    :returns: ``"auto"`` or ``"require"``, or ``None`` when resume is disabled
        (``null``, ``off``, or YAML-1.1 ``off``-as-``False``).
    :raises ValueError: On an unknown mode, or when an active mode is combined
        with an explicit ``ckpt_path`` (ambiguous intent).
    """
    mode = OmegaConf.select(cfg, "training.resume")
    if mode in _DISABLED_VALUES:
        return None
    if mode not in _ACTIVE_MODES:
        raise ValueError(f"training.resume must be one of null/off/auto/require; got {mode!r}.")
    if OmegaConf.select(cfg, "ckpt_path"):
        raise ValueError(
            f"training.resume={mode} and an explicit ckpt_path are mutually "
            "exclusive; drop one of them."
        )
    return mode


def apply_wandb_resume_continuity(cfg: DictConfig) -> None:
    """Mark the W&B logger to continue an existing run instead of erroring on id reuse.

    No-op when the cfg has no ``logger.wandb`` group (mirrors
    ``pin_wandb_run_id``), so logger-free runs need no special-casing.

    :param cfg: Composed train cfg; ``logger.wandb.resume`` is updated in place.
    """
    if OmegaConf.select(cfg, "logger.wandb") is None:
        return
    OmegaConf.update(cfg, "logger.wandb.resume", "allow")


def _run_id_matches_config(run_id: str, config_id: str) -> bool:
    """Report whether a run id is this config_id's canonical ``{config_id}-{timestamp}``.

    Anchored on the timestamp shape so config_ids that prefix each other (e.g.
    ``flow`` vs ``flow-x``) can never cross-match.

    :param run_id: Candidate W&B run id.
    :param config_id: Run identity the id must belong to.
    :returns: Whether the id matches.
    """
    return re.fullmatch(re.escape(config_id) + "-" + _RUN_ID_TIMESTAMP_RE, run_id) is not None


def _run_id_from_run_dir(run_dir: Path) -> str | None:
    """Recover the W&B run id from a run dir's ``wandb/run-<ts>-<id>`` dir.

    :param run_dir: One Hydra run output dir.
    :returns: The run id, or ``None`` when the dir has no wandb run subdir.
    """
    candidates = sorted(run_dir.glob("wandb/run-*"))
    if not candidates:
        return None
    # ``run-<YYYYMMDD_HHMMSS>-<id>``: the id is everything past the second dash.
    parts = candidates[-1].name.split("-", 2)
    return parts[2] if len(parts) == 3 else None


def discover_local_checkpoint(current_output_dir: Path, config_id: str) -> ResumeDecision | None:
    """Find the newest sibling run dir's ``last.ckpt`` for this config_id.

    Scans the current output dir's siblings (the Hydra run-dir family, e.g.
    ``logs/train/<task>/<name>-<ts>/``). A sibling whose recovered W&B run id
    is not this config_id's canonical ``{config_id}-{timestamp}`` is skipped;
    one with no recoverable id (logger disabled) is accepted with a warning
    naming the unverified-identity assumption, since Lightning still fails
    loudly on an architecture mismatch.

    :param current_output_dir: This launch's Hydra output dir, excluded from
        the scan.
    :param config_id: Run identity (experiment basename) candidates must match.
    :returns: The newest matching checkpoint, or ``None``.
    """
    candidates: list[tuple[float, Path, str | None]] = []
    for ckpt in current_output_dir.parent.glob("*/checkpoints/last.ckpt"):
        run_dir = ckpt.parent.parent
        if run_dir == current_output_dir:
            continue
        run_id = _run_id_from_run_dir(run_dir)
        if run_id is not None and not _run_id_matches_config(run_id, config_id):
            continue
        candidates.append((ckpt.stat().st_mtime, ckpt, run_id))
    if not candidates:
        return None
    _, ckpt_path, run_id = max(candidates, key=lambda item: item[0])
    if run_id is None:
        log.warning(
            f"Auto-resume candidate {ckpt_path} has no recoverable W&B run id; "
            f"assuming it belongs to config_id {config_id!r} (its run-dir family). "
            "A wrong-architecture checkpoint still fails loudly at load."
        )
    return ResumeDecision(ckpt_path=ckpt_path, wandb_run_id=run_id, source="local")


def run_id_from_recovery_namespace(namespace: str) -> str | None:
    """Recover the W&B run id embedded in an R2 recovery namespace.

    :param namespace: Directory name of one mid-run mirror,
        ``{run_id}-{32-hex-uuid4}``.
    :returns: The run id, or ``None`` when the name has no uuid suffix.
    """
    match = _RECOVERY_NAMESPACE_RE.match(namespace)
    return match.group("run_id") if match else None


def discover_r2_checkpoint(bucket: str, config_id: str, dest_dir: Path) -> ResumeDecision | None:
    """Download the newest mid-run mirror ``last.ckpt`` for this config_id.

    Scans ``r2://{bucket}/checkpoints/{config_id}/`` for per-launch recovery
    namespaces (written by ``CheckpointUploader``) and pulls the newest by the
    storage-assigned mtime. Best-effort: missing creds, an unreachable remote,
    or a failed transfer degrade to ``None`` with a warning so local runs never
    hard-depend on R2.

    :param bucket: R2 bucket name (``cfg.r2.bucket``).
    :param config_id: Run identity keying the mirror prefix.
    :param dest_dir: Local directory the checkpoint downloads into.
    :returns: The downloaded checkpoint, or ``None``.
    """
    # Deferred so importing this module never pulls the rclone/env machinery.
    from synth_setter.pipeline import r2_io

    prefix = f"r2://{bucket}/checkpoints/{config_id}"
    try:
        r2_io.ensure_r2_env_loaded()
        entries = r2_io.list_entries(f"{prefix}/", recursive=True)
    except _DEGRADABLE_ERRORS as exc:
        log.warning(f"Skipping R2 resume discovery under {prefix}: {exc}")
        return None
    mirrors = [
        entry
        for entry in entries
        if entry.path.count("/") == 1 and entry.path.endswith("/last.ckpt")
    ]
    if not mirrors:
        return None
    newest = max(mirrors, key=lambda entry: entry.mtime)
    dest = dest_dir / "last.ckpt"
    try:
        r2_io.download_to_path(f"{prefix}/{newest.path}", dest)
    except _DEGRADABLE_ERRORS as exc:
        log.warning(f"Failed to download resume checkpoint {newest.path}: {exc}")
        return None
    namespace = newest.path.split("/", 1)[0]
    return ResumeDecision(
        ckpt_path=dest,
        wandb_run_id=run_id_from_recovery_namespace(namespace),
        source="r2",
    )


def discover_wandb_artifact_checkpoint(config_id: str) -> ResumeDecision | None:
    """Fetch the train-end ``model-{config_id}:latest`` artifact checkpoint.

    Last-resort tier: the artifact only exists after a *completed* run, so its
    checkpoint carries train-end optimizer/scheduler state. Best-effort — a
    missing ``wandb`` install, unknown artifact, or fetch failure degrades to
    ``None`` with a warning.

    :param config_id: Run identity naming the model artifact.
    :returns: The cached checkpoint plus the producing run's id, or ``None``.
    """
    if not find_spec("wandb"):
        return None
    # Deferred: wandb is an optional dependency group and slow to import.
    import wandb
    from wandb.errors import Error as WandbError

    from synth_setter.utils.utils import resolve_wandb_checkpoint

    ref = f"model-{config_id}:latest"
    try:
        artifact = wandb.Api(timeout=_WANDB_API_TIMEOUT_SECONDS).artifact(ref)
        producing_run = artifact.logged_by()
        ckpt_path = Path(resolve_wandb_checkpoint(ref))
    except (*_DEGRADABLE_ERRORS, WandbError, KeyError) as exc:
        log.warning(f"Skipping W&B artifact resume discovery for {ref}: {exc}")
        return None
    return ResumeDecision(
        ckpt_path=ckpt_path,
        wandb_run_id=producing_run.id if producing_run is not None else None,
        source="wandb-artifact",
    )


def discover_resume_checkpoint(cfg: DictConfig, config_id: str) -> ResumeDecision | None:
    """Run the discovery tiers in order and return the first hit.

    Tier order trades freshness for cost: local sibling run dirs (free, newest
    mid-run state), then the R2 mid-run mirrors (survives a lost disk), then
    the train-end W&B model artifact (survives everything, but is train-end
    state only). The R2 tier is skipped when ``r2.bucket`` is unset.

    :param cfg: Composed train cfg; reads ``paths.output_dir`` and ``r2.bucket``.
    :param config_id: Run identity keying every tier.
    :returns: The first tier's decision, or ``None`` when nothing is found.
    """
    output_dir = Path(cfg.paths.output_dir)
    decision = discover_local_checkpoint(output_dir, config_id)
    if decision is None:
        bucket = OmegaConf.select(cfg, "r2.bucket")
        if bucket:
            decision = discover_r2_checkpoint(bucket, config_id, output_dir / "resume")
    if decision is None:
        decision = discover_wandb_artifact_checkpoint(config_id)
    return decision
