"""``synth-setter-finalize-dataset`` entrypoint: post-generate finalize stage.

Loads the frozen ``DatasetSpec`` from ``input_spec.json`` under
``cfg.dataset_root_uri`` (the R2 run prefix the upstream generate stage's
``upload_spec`` wrote to) and commits its staged winner fragments into each
``{train,val,test}.lance`` split manifest, reducing the winners' Welford
sidecars into ``stats.npz`` — no shard row is decoded (#1776). The
``dataset.complete`` marker is written last per ``pipeline/CLAUDE.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from traceback import format_tb
from typing import cast

import hydra
import wandb
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.constants import DATASET_COMPLETE_FILENAME
from synth_setter.pipeline.data.finalize_progress import (
    FinalizeProgressCallback,
    FinalizeProgressEvent,
    report_finalize_progress,
)
from synth_setter.pipeline.schemas.prefix import assert_r2_prefix_matches
from synth_setter.pipeline.schemas.spec import DatasetSpec, OutputFormat
from synth_setter.pipeline.spec_io import load_spec_from_root
from synth_setter.utils import pin_wandb_run_id
from synth_setter.utils.instantiators import close_loggers, instantiate_loggers
from synth_setter.workspace import operator_workspace

# Resolve workspace at import so ``${oc.env:PROJECT_ROOT}`` in
# ``configs/paths/default.yaml`` interpolates under any install layout.
operator_workspace()


def _log_finalize_metrics(loggers: Sequence[Logger], metrics: Mapping[str, float]) -> None:
    """Log one finalization history row to W&B without making logging mandatory.

    No explicit step is passed because W&B auto-advances history for each
    ``log`` call, including when finalize resumes the generation run.

    :param loggers: Configured Lightning loggers; only ``WandbLogger`` entries receive metrics.
    :param metrics: Completed-progress values for one W&B history row.
    """
    for wandb_logger in loggers:
        if not isinstance(wandb_logger, WandbLogger):
            continue
        try:
            wandb_logger.log_metrics(dict(metrics))
        except Exception as exc:  # noqa: BLE001 — W&B must remain best-effort
            logger.warning(
                "finalize metrics logging failed on {}: {}", type(wandb_logger).__name__, exc
            )


def _make_finalize_progress_logger(
    loggers: Sequence[Logger], total_shards: int
) -> tuple[FinalizeProgressCallback, Callable[[float], None]]:
    """Build the live progress callback and terminal-summary logger.

    :param loggers: Configured Lightning loggers; W&B entries receive history rows.
    :param total_shards: Number of source shards the finalization run can process.
    :returns: Callback for live events and a callable for the terminal summary row.
    """
    processed_shards = 0
    uploaded_artifacts = 0

    def handle_progress_event(event: FinalizeProgressEvent) -> None:
        nonlocal processed_shards, uploaded_artifacts
        if event == "shard_processed":
            processed_shards += 1
            _log_finalize_metrics(
                loggers,
                {
                    "finalize/shards_processed": float(processed_shards),
                    "finalize/shards_total": float(total_shards),
                },
            )
            return

        uploaded_artifacts += 1
        _log_finalize_metrics(loggers, {"finalize/artifacts_uploaded": float(uploaded_artifacts)})

    def log_summary(elapsed_seconds: float) -> None:
        if processed_shards == 0 and uploaded_artifacts == 0:
            return
        _log_finalize_metrics(
            loggers,
            {
                "finalize/elapsed_seconds": elapsed_seconds,
                "finalize/shards_processed": float(processed_shards),
                "finalize/shards_total": float(total_shards),
                "finalize/artifacts_uploaded": float(uploaded_artifacts),
            },
        )

    return handle_progress_event, log_summary


def _require_nonempty_train(spec: DatasetSpec) -> None:
    """Reject a spec whose train split holds no shards — stats need at least one.

    :param spec: Validated dataset spec.
    :raises ValueError: The train split range is empty.
    """
    train_lo, train_hi = spec.split_shard_ranges["train"]
    if train_lo >= train_hi:
        raise ValueError(
            f"train split is empty (split_shard_ranges['train']="
            f"{spec.split_shard_ranges['train']!r}); cannot compute stats "
            f"without at least one train shard."
        )


# DOC502: the documented ValueError propagates from _require_nonempty_train.
def finalize_lance(  # noqa: DOC502
    spec: DatasetSpec,
    work_dir: Path,
    progress_callback: FinalizeProgressCallback | None = None,
) -> None:
    """Commit staged winner fragments into split datasets — no shard row is decoded.

    Delegates to
    :func:`~synth_setter.pipeline.data.lance_finalize.finalize_lance_fragments`:
    winner selection over the staged attempts, structural checks, one atomic
    ``Overwrite`` commit per split, Welford reduction of the winners'
    ``.shard-stats.npz`` sidecars into ``stats.npz``, and the ``dataset.json``
    audit record. Progress events surface one ``shard_processed`` per selected
    winner and one ``artifact_uploaded`` per committed split, plus the stats
    and card uploads.

    :param spec: Validated dataset spec (``output_format == "lance"``).
    :param work_dir: Scratch directory for the staged ``stats.npz`` / ``dataset.json``.
    :param progress_callback: Optional sink for completed shard and upload events.
    :raises ValueError: The train split is empty, a spec shard has no
        staged-valid attempt, or a winner fails a structural check.
    """
    from synth_setter.pipeline.data.lance_finalize import finalize_lance_fragments

    _require_nonempty_train(spec)
    finalize_lance_fragments(spec, work_dir, progress_callback)


def finalize_from_spec(
    spec: DatasetSpec,
    work_dir: Path,
    progress_callback: FinalizeProgressCallback | None = None,
) -> None:
    """Finalize a dataset given an in-memory spec; idempotent on ``dataset.complete``.

    Returns without work when the marker already exists at the run prefix —
    R2 is the source of truth (per ``pipeline/CLAUDE.md``). Lance is the only
    supported ``spec.output_format``; the marker is uploaded strictly last so
    an interrupted run never advertises artifacts that have not landed. Caller
    is responsible for ensuring R2 creds are loaded.

    :param spec: Validated dataset spec.
    :param work_dir: Writable scratch dir; created if missing; retained
        after the call.
    :param progress_callback: Optional sink for completed shard and upload events.
    :raises ValueError: ``spec.output_format`` is not a supported finalized format.
    """
    marker_uri = spec.r2.dataset_complete_marker_uri()
    if r2_io.object_size(marker_uri) is not None:
        logger.info("skip: {} already exists, run is finalized", marker_uri)
        return

    # A custom ``r2.prefix`` (e.g. the oracle-eval e2e isolating objects under
    # ``test-runs/<test>/<uuid>/``) is legitimate: finalize reads the same prefix
    # generate wrote to, so the spec is self-consistent. Surface a divergence
    # from the canonical ``make_r2_prefix`` shape as a warning, never an abort.
    try:
        assert_r2_prefix_matches(spec.r2.prefix, spec.task_name, spec.run_id, spec.r2.prefix_root)
    except ValueError as exc:
        logger.warning("non-canonical r2 prefix (finalizing anyway): {}", exc)
    work_dir.mkdir(parents=True, exist_ok=True)
    if spec.output_format is OutputFormat.LANCE:
        finalize_lance(spec, work_dir, progress_callback)
    else:
        raise ValueError(f"unsupported output_format: {spec.output_format!r}")

    marker_local = work_dir / DATASET_COMPLETE_FILENAME
    marker_local.touch()
    r2_io.upload(marker_local, marker_uri)
    report_finalize_progress(progress_callback, "artifact_uploaded")
    logger.info("wrote dataset.complete to {}", marker_uri)


def _finalized_reference_uris(spec: DatasetSpec) -> list[str]:
    """Return the R2 URIs of the objects finalize materialized for this run.

    Each non-empty split ``.lance`` dataset plus ``stats.npz`` is referenced;
    empty splits contribute nothing — finalize prunes them.

    :param spec: Validated dataset spec.
    :returns: Canonical ``r2://`` URIs, split datasets first then ``stats.npz``.
    """
    split_uris = [
        spec.r2.split_lance_uri(split)
        for split, (lo, hi) in spec.split_shard_ranges.items()
        if lo < hi
    ]
    return [*split_uris, spec.r2.stats_uri()]


def build_dataset_artifact(spec: DatasetSpec) -> wandb.Artifact:
    """Build the canonical ``dataset`` W&B artifact for a finalized run.

    Names the artifact ``data-{spec.task_name}`` (type ``dataset``) per
    ``storage-provenance-spec.md`` §4, references the finalized R2 objects as
    ``s3://`` URIs (split ``.lance`` datasets plus ``stats.npz``), and records
    ``shard_count`` / ``n_samples`` / ``git_sha``
    in ``artifact.metadata`` per §6. References use ``checksum=False`` because
    R2's custom S3 endpoint is not reachable by W&B's default reference
    handler — the URIs record lineage, not a content hash.

    :param spec: Validated dataset spec; its R2 location and split sizes
        determine the references and metadata.
    :returns: An unlogged ``wandb.Artifact`` ready for ``log_artifact``.
    """
    artifact = wandb.Artifact(
        name=f"data-{spec.task_name}",
        type="dataset",
        metadata={
            "shard_count": spec.num_shards,
            "n_samples": sum(spec.train_val_test_sizes),
            "git_sha": spec.git_sha,
        },
    )
    for r2_uri in _finalized_reference_uris(spec):
        artifact.add_reference(r2_io.to_s3_uri(r2_uri), checksum=False)
    return artifact


def _log_finalize_failure(error: BaseException) -> None:
    """Log a failed finalize traceback without exposing the exception text.

    :param error: Exception raised by the finalize body.
    """
    logger.error(
        "finalize failed ({})\n{}", type(error).__name__, "".join(format_tb(error.__traceback__))
    )


def _log_dataset_artifact(loggers: list[Logger], spec: DatasetSpec) -> None:
    """Log the canonical ``dataset`` artifact to each ``WandbLogger`` in ``loggers``.

    Mirrors ``generate_dataset._log_spec_artifact``: a wandb failure warns and
    is swallowed so artifact logging never aborts a completed finalize — the
    R2 outputs and ``dataset.complete`` marker are already written.
    Non-``WandbLogger`` entries (and an empty list) are a no-op, which is the
    path every wandb-free caller (e.g. the existing finalize tests) takes.

    :param loggers: Lightning loggers; only ``WandbLogger`` entries log.
    :param spec: Validated dataset spec forwarded to :func:`build_dataset_artifact`.
    """
    for lg in loggers:
        if not isinstance(lg, WandbLogger):
            continue
        try:
            lg.experiment.log_artifact(build_dataset_artifact(spec), aliases=[spec.run_id])
        except Exception as exc:  # noqa: BLE001 — wandb artifact failure must not abort finalize
            logger.warning(f"_log_dataset_artifact failed on {type(lg).__name__}: {exc}")


def finalize(cfg: DictConfig) -> None:  # noqa: DOC503
    """Finalize the R2 prefix at ``cfg.dataset_root_uri``; idempotent on ``dataset.complete``.

    Loads R2 creds and the spec from ``input_spec.json`` under
    ``cfg.dataset_root_uri``, delegates to
    :func:`finalize_from_spec` for the marker-probe → dispatch → marker-upload
    body, then logs live progress metrics and the canonical ``dataset``
    artifact to any configured ``WandbLogger`` (resuming the data-generation
    run pinned to ``spec.run_id`` so both land on the producer node of the
    lineage DAG). The wandb run id is pinned and ``resume=allow`` is forced so
    finalize attaches to the generation run rather than minting a new one;
    both are no-ops when ``cfg`` carries no ``logger`` group (the wandb-free
    default). On any failure the traceback and partial progress summary are
    logged before the loggers close with status ``"failed"`` and the exception re-raises.

    :param cfg: Composed cfg with ``dataset_root_uri`` (the run-prefix dir
        accepted by :func:`~synth_setter.pipeline.spec_io.load_spec_from_root`),
        ``paths.output_dir`` (writable scratch dir; created if missing;
        retained after the call), and an optional ``logger`` group instantiated
        for W&B progress and artifact logging.
    :raises ValueError: Propagated from :func:`finalize_from_spec` — a drifted
        ``spec.r2.prefix`` or an unsupported ``spec.output_format``.
    """
    r2_io.ensure_r2_env_loaded()
    spec = load_spec_from_root(cfg.dataset_root_uri)
    pin_wandb_run_id(cfg, spec.run_id, "data-generation")
    if OmegaConf.select(cfg, "logger.wandb") is not None:
        OmegaConf.update(cfg, "logger.wandb.resume", "allow", force_add=True)
    loggers = instantiate_loggers(cfg.get("logger"))
    status = "success"
    started_at = perf_counter()
    log_summary: Callable[[float], None] | None = None
    try:
        report_progress, log_summary = _make_finalize_progress_logger(loggers, spec.num_shards)
        finalize_from_spec(spec, Path(cfg.paths.output_dir), report_progress)
        log_summary(perf_counter() - started_at)
        _log_dataset_artifact(loggers, spec)
    except BaseException as error:
        status = "failed"
        failed_elapsed_seconds = perf_counter() - started_at
        _log_finalize_failure(error)
        if log_summary is not None:
            log_summary(failed_elapsed_seconds)
        raise
    finally:
        close_loggers(loggers, status)


@hydra.main(
    version_base="1.3",
    config_path="pkg://synth_setter.configs",
    config_name="finalize_dataset",
)
def main(cfg: DictConfig) -> None:
    """@hydra.main entrypoint; delegates to :func:`finalize` for the contract.

    :param cfg: Hydra-composed cfg; see :func:`finalize` for the field contract.
    """
    finalize(cfg)


if __name__ == "__main__":
    # hydra.main types its wrapper as Any, so pyright sees the undecorated
    # one-arg signature; the wrapper itself takes no positional args.
    cast("Callable[[], None]", main)()
