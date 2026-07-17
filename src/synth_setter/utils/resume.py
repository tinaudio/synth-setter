"""Auto-resume checkpoint discovery for ``synth-setter-train`` (#1991).

Discovery logic: given a composed train cfg, find the newest usable
``last.ckpt`` for the run's config_id across local sibling run dirs and the R2
mid-run mirrors. A winning R2 tier performs a network fetch; everything else
is pure. Deliberately NOT a tier: the train-end ``model-{config_id}`` W&B
artifact — it only exists after a *completed* run and carries the
monitor-best checkpoint, so "resuming" from it warm-starts rather than
recovers (use an explicit ``ckpt_path='${wandb:...}'`` for that). The
imperative wiring (setting ``cfg.ckpt_path``, pinning the recovered W&B run
id) lives in ``synth_setter.cli.train``.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
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
    source: Literal["local", "r2"]


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
    if OmegaConf.select(cfg, "ckpt_path") is not None:
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
    """Recover the W&B run id from a run dir's ``wandb/[offline-]run-<ts>-<id>`` dir.

    Offline-mode launches (``logger.wandb.offline=true``) write
    ``offline-run-*`` dirs; both spellings carry the same id layout.

    :param run_dir: One Hydra run output dir.
    :returns: The run id, or ``None`` when the dir has no wandb run subdir.
    """
    names: list[str] = []
    for pattern in ("wandb/run-*", "wandb/offline-run-*"):
        names.extend(
            candidate.name.removeprefix("offline-") for candidate in run_dir.glob(pattern)
        )
    # ``run-<YYYYMMDD_HHMMSS>-<id>``: the id is everything past the second dash.
    for name in sorted(names, reverse=True):
        parts = name.split("-", 2)
        if len(parts) != 3:
            continue
        try:
            datetime.strptime(parts[1], "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        return parts[2]
    return None


def _config_id_from_hydra_dir(run_dir: Path) -> str | None:
    """Recover a run dir's config_id from its recorded Hydra state.

    Mirrors ``resolve_run_config_id``: the ``experiment`` choice basename when
    one was composed, else the run's ``task_name``.

    :param run_dir: One Hydra run output dir.
    :returns: The config_id, or ``None`` when no readable ``.hydra`` state exists.
    """
    hydra_dir = run_dir / ".hydra"
    try:
        experiment = OmegaConf.select(
            OmegaConf.load(hydra_dir / "hydra.yaml"), "hydra.runtime.choices.experiment"
        )
        if experiment not in (None, "null"):
            return PurePosixPath(str(experiment)).name
        return OmegaConf.select(OmegaConf.load(hydra_dir / "config.yaml"), "task_name")
    except (OSError, yaml.YAMLError) as exc:
        log.debug("Cannot read Hydra state from %s: %s", hydra_dir, exc)
        return None


def discover_local_checkpoint(current_output_dir: Path, config_id: str) -> ResumeDecision | None:
    """Find the newest sibling run dir's ``last.ckpt`` for this config_id.

    Scans the current output dir's siblings (the Hydra run-dir family, e.g.
    ``logs/train/<task>/<name>-<ts>/``). Every candidate must prove its
    identity: a recovered W&B run id must be this config_id's canonical
    ``{config_id}-{timestamp}``; a sibling without a wandb run dir (logger
    disabled) must instead carry matching recorded Hydra state. A sibling with
    neither is skipped — an unverifiable checkpoint of a different config
    could load silently when the architectures happen to match.

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
        if not (run_dir / ".hydra").is_dir():
            log.debug("Skipping auto-resume candidate %s: not a Hydra run directory.", ckpt)
            continue
        run_id = _run_id_from_run_dir(run_dir)
        if run_id is not None:
            if not _run_id_matches_config(run_id, config_id):
                continue
        else:
            candidate_config_id = _config_id_from_hydra_dir(run_dir)
            if candidate_config_id is None:
                log.warning(
                    "Skipping auto-resume candidate %s: no W&B run id or readable "
                    "Hydra state proves its identity.",
                    ckpt,
                )
                continue
            if candidate_config_id != config_id:
                continue
        try:
            mtime = ckpt.stat().st_mtime
        except OSError as exc:  # a concurrent launch rotated the checkpoint away mid-scan
            log.debug("Skipping auto-resume candidate %s: %s", ckpt, exc)
            continue
        candidates.append((mtime, ckpt, run_id))
    if not candidates:
        return None
    # Path string as tie-break: coarse-mtime filesystems can stamp two
    # checkpoints identically, and glob order is not deterministic.
    _, ckpt_path, run_id = max(candidates, key=lambda item: (item[0], str(item[1])))
    return ResumeDecision(ckpt_path=ckpt_path, wandb_run_id=run_id, source="local")


def run_id_from_recovery_namespace(namespace: str) -> str | None:
    """Recover the W&B run id embedded in an R2 recovery namespace.

    :param namespace: Directory name of one mid-run mirror,
        ``{run_id}-{32-hex-uuid4}``.
    :returns: The run id, or ``None`` when the name has no uuid suffix.
    """
    match = _RECOVERY_NAMESPACE_RE.match(namespace)
    return match.group("run_id") if match else None


def checkpoint_mirror_prefix(cfg: DictConfig, config_id: str) -> str | None:
    """Return the ``r2://`` parent prefix mid-run mirrors upload under.

    Mirrors ``cli.train._checkpoint_prefix_uri`` sans the per-launch recovery
    namespace: the parent of ``training.upload_checkpoints_uri`` when that
    override is set, else the auto-derived
    ``r2://{r2.bucket}/checkpoints/{config_id}``. Tolerant where the uploader
    fails fast — a malformed override or missing bucket yields ``None`` so
    discovery degrades instead of aborting the launch.

    :param cfg: Composed train cfg; reads ``training.upload_checkpoints_uri``
        and ``r2.bucket``.
    :param config_id: Run identity keying the auto-derived prefix.
    :returns: The mirror parent prefix, or ``None`` when R2 is not configured.
    """
    override = OmegaConf.select(cfg, "training.upload_checkpoints_uri")
    if override:
        prefix = str(override).rsplit("/", 1)[0]
        if str(override).endswith("/") or not prefix.startswith("r2://") or prefix == "r2://":
            log.warning("Ignoring malformed training.upload_checkpoints_uri %r.", str(override))
            return None
        return prefix
    bucket = OmegaConf.select(cfg, "r2.bucket")
    if not bucket:
        return None
    return f"r2://{bucket}/checkpoints/{config_id}"


def discover_r2_checkpoint(
    prefix: str,
    config_id: str,
    dest_dir: Path,
    diagnostics: list[str] | None = None,
) -> ResumeDecision | None:
    """Download the newest mid-run mirror ``last.ckpt`` under ``prefix``.

    Scans the mirror prefix (see :func:`checkpoint_mirror_prefix`) for
    per-launch recovery namespaces (written by ``CheckpointUploader``) and
    pulls the newest by the storage-assigned mtime. Best-effort: missing
    creds, an unreachable remote, or a failed transfer degrade to ``None``
    with a warning so local runs never hard-depend on R2.

    :param prefix: ``r2://`` parent prefix the mirrors upload under.
    :param config_id: Run identity a namespace-embedded run id must match to be reused.
    :param dest_dir: Local directory the checkpoint downloads into.
    :param diagnostics: Optional list receiving R2 degradation descriptions.
    :returns: The downloaded checkpoint, or ``None``.
    """
    # Deferred so importing this module never pulls the rclone/env machinery.
    from synth_setter.pipeline import r2_io

    try:
        r2_io.ensure_r2_env_loaded()
        entries = r2_io.list_entries(f"{prefix}/", recursive=True)
    except _DEGRADABLE_ERRORS as exc:
        message = f"R2 resume discovery under {prefix} degraded: {exc}"
        log.warning("%s", message)
        if diagnostics is not None:
            diagnostics.append(message)
        return None
    mirrors = []
    for entry in entries:
        if entry.path.count("/") != 1 or not entry.path.endswith("/last.ckpt"):
            continue
        namespace = entry.path.split("/", 1)[0]
        run_id = run_id_from_recovery_namespace(namespace)
        if run_id is None or not _run_id_matches_config(run_id, config_id):
            continue
        mirrors.append((entry, run_id))
    if not mirrors:
        return None
    newest, run_id = max(mirrors, key=lambda candidate: (candidate[0].mtime, candidate[0].path))
    dest = dest_dir / "last.ckpt"
    try:
        r2_io.download_to_path(f"{prefix}/{newest.path}", dest)
    except _DEGRADABLE_ERRORS as exc:
        message = f"R2 resume checkpoint download {newest.path} degraded: {exc}"
        log.warning("%s", message)
        if diagnostics is not None:
            diagnostics.append(message)
        return None
    return ResumeDecision(ckpt_path=dest, wandb_run_id=run_id, source="r2")


def discover_resume_checkpoint(
    cfg: DictConfig, config_id: str, diagnostics: list[str] | None = None
) -> ResumeDecision | None:
    """Run the discovery tiers in order and return the first hit.

    Tier order trades freshness for cost: local sibling run dirs (free, newest
    mid-run state), then the R2 mid-run mirrors (survives a lost disk). The R2
    tier is skipped at INFO level when no mirror prefix is configured.

    :param cfg: Composed train cfg; reads ``paths.output_dir`` and ``r2.bucket``.
    :param config_id: Run identity keying every tier.
    :param diagnostics: Optional list receiving R2 degradation descriptions.
    :returns: The first tier's decision, or ``None`` when nothing is found.
    """
    output_dir = Path(cfg.paths.output_dir)
    decision = discover_local_checkpoint(output_dir, config_id)
    if decision is None:
        prefix = checkpoint_mirror_prefix(cfg, config_id)
        if prefix is None:
            log.info("Skipping R2 resume discovery: no mirror prefix is configured.")
        else:
            decision = discover_r2_checkpoint(
                prefix, config_id, output_dir / "resume", diagnostics=diagnostics
            )
    return decision
