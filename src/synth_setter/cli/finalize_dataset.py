"""``synth-setter-finalize-dataset`` entrypoint: post-generate finalize stage.

Loads the frozen ``DatasetSpec`` from ``input_spec.json`` under
``cfg.dataset_root_uri`` (the R2 run prefix the upstream generate stage's
``upload_spec`` wrote to) and dispatches
on ``spec.output_format``. Every branch uploads its derived artifact(s)
and then writes the ``dataset.complete`` marker last per
``pipeline/CLAUDE.md``. The wds branch streams train shards through
Welford row-by-row; the hdf5 branch downloads every shard, reshards into
``{train,val,test}.h5``, and computes ``stats.npz`` over the train split;
the lance branch downloads every shard, streams train shards through
Welford for ``stats.npz``, and concatenates each split's shards into
``{train,val,test}.lance``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import hydra
import numpy as np
import wandb
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.constants import (
    DATASET_COMPLETE_FILENAME,
    INPUT_SPEC_FILENAME,
    STATS_NPZ_FILENAME,
)
from synth_setter.pipeline.data.reshard import reshard_dataset
from synth_setter.pipeline.data.stats import get_stats_hdf5, stream_stats_wds
from synth_setter.pipeline.schemas.prefix import assert_r2_prefix_matches
from synth_setter.pipeline.schemas.spec import DatasetSpec, OutputFormat, ShardSpec
from synth_setter.pipeline.spec_io import load_spec_from_root, write_spec_to_path
from synth_setter.utils import pin_wandb_run_id
from synth_setter.utils.instantiators import close_loggers, instantiate_loggers
from synth_setter.workspace import operator_workspace

if TYPE_CHECKING:
    import pyarrow as pa

    LanceBatchIterator: TypeAlias = Iterator[pa.RecordBatch]
    LanceSplitBatches: TypeAlias = tuple[pa.Schema, LanceBatchIterator]
else:
    LanceSplitBatches: TypeAlias = tuple[object, Iterator[object]]

# Resolve workspace at import so ``${oc.env:PROJECT_ROOT}`` in
# ``configs/paths/default.yaml`` interpolates under any install layout.
operator_workspace()


def _download_train_shards_one_at_a_time(spec: DatasetSpec, work_dir: Path) -> Iterator[Path]:
    """Yield one downloaded train shard at a time, unlinking after the consumer is done.

    Peak local disk stays at one shard regardless of split size; the
    ``finally`` clause runs when ``stream_stats_wds`` advances to the next
    iteration, so the previous shard's bytes are released before the next
    download starts.

    :param spec: Validated dataset spec.
    :param work_dir: Scratch directory; shards land here transiently.
    :yields Path: Local path of the just-downloaded train shard.
    """
    train_lo, train_hi = spec.split_shard_ranges["train"]
    for shard in spec.shards[train_lo:train_hi]:
        local = work_dir / shard.filename
        r2_io.download_to_path(spec.r2.shard_uri(shard), local)
        try:
            yield local
        finally:
            local.unlink(missing_ok=True)


def finalize_wds(spec: DatasetSpec, work_dir: Path) -> None:
    """Stream stats over the train shards and upload ``stats.npz``.

    Per-shard tar files stay in their original R2 location; only the
    derived ``stats.npz`` is materialized. Brace patterns for non-empty
    splits are available via
    ``spec.r2.split_wds_brace_uri(spec.split_shard_ranges[split])``;
    callers must check ``lo < hi`` first because empty splits raise
    ``ValueError`` at that helper. ``spec.mask_degenerate_bins`` is
    forwarded to ``stream_stats_wds``.

    :param spec: Validated dataset spec (``output_format == "wds"``).
    :param work_dir: Scratch directory; one shard at a time + the final
        ``stats.npz`` live here transiently.
    :raises ValueError: The train split is empty
        (``spec.split_shard_ranges["train"]`` has ``lo >= hi``); stats
        cannot be computed without at least one train shard.
    """
    train_lo, train_hi = spec.split_shard_ranges["train"]
    if train_lo >= train_hi:
        raise ValueError(
            f"train split is empty (split_shard_ranges['train']="
            f"{spec.split_shard_ranges['train']!r}); cannot compute stats "
            f"without at least one train shard."
        )
    mean, std = stream_stats_wds(
        _download_train_shards_one_at_a_time(spec, work_dir),
        mask_degenerate=spec.mask_degenerate_bins,
    )
    stats_npz = work_dir / STATS_NPZ_FILENAME
    np.savez(stats_npz, mean=mean, std=std)
    r2_io.upload(stats_npz, spec.r2.stats_uri())
    logger.info("uploaded stats to {}", spec.r2.stats_uri())


def finalize_hdf5(spec: DatasetSpec, work_dir: Path) -> None:
    """Download every shard, reshard into split files, compute stats, upload all artifacts.

    Writes ``work_dir/input_spec.json`` flat (via
    :func:`~synth_setter.pipeline.spec_io.write_spec_to_path`) so
    :func:`~synth_setter.pipeline.data.reshard.reshard_dataset`'s default
    spec discovery picks it up without a ``--spec`` override. The flat
    placement diverges from :func:`~synth_setter.pipeline.spec_io.write_spec_locally`'s
    nested ``<output_dir>/data/<task>/<run>/metadata/`` layout because
    ``work_dir`` is a per-finalize scratch tempdir whose only consumer is
    reshard — re-creating the operator-side ``data/`` hierarchy under it
    would force the reshard adapter to learn that layout for no benefit.
    ``get_stats_hdf5`` then writes ``work_dir / "stats.npz"`` (path derived
    via ``VSTDataset.get_stats_file_path(train.h5)``); the post-call
    existence guard pins that contract so a future drift in the derivation
    surfaces here rather than as a missing upload source. Structural
    validation (per ``pipeline/CLAUDE.md``) is delegated to the h5py opens
    that ``reshard_dataset`` performs while staging each split — finalize
    never re-runs the workers' full four-check pass.
    ``spec.mask_degenerate_bins`` is forwarded to ``get_stats_hdf5``.

    :param spec: Validated dataset spec (``output_format == "hdf5"``).
    :param work_dir: Scratch directory; shards, splits, stats and the spec
        copy live here transiently for the duration of the call.
    :raises ValueError: The train split is empty
        (``spec.split_shard_ranges["train"]`` has ``lo >= hi``); reshard
        would prune ``train.h5`` and stats compute would fail with a
        low-signal HDF5 error.
    :raises FileNotFoundError: ``get_stats_hdf5`` returned without writing
        ``work_dir / "stats.npz"``, breaking the upload-source contract.
    """
    train_lo, train_hi = spec.split_shard_ranges["train"]
    if train_lo >= train_hi:
        raise ValueError(
            f"train split is empty (split_shard_ranges['train']="
            f"{spec.split_shard_ranges['train']!r}); cannot compute stats "
            f"without at least one train shard."
        )
    for shard in spec.shards:
        r2_io.download_to_path(spec.r2.shard_uri(shard), work_dir / shard.filename)
    write_spec_to_path(spec, work_dir / INPUT_SPEC_FILENAME)
    reshard_dataset(work_dir)
    get_stats_hdf5(str(work_dir / "train.h5"), mask_degenerate=spec.mask_degenerate_bins)
    stats_npz = work_dir / STATS_NPZ_FILENAME
    if not stats_npz.is_file():
        raise FileNotFoundError(
            f"get_stats_hdf5 did not write {stats_npz}; check "
            f"VSTDataset.get_stats_file_path derivation."
        )
    # Reshard prunes empty splits — only upload the ones it actually wrote.
    # Iterate ``split_shard_ranges`` (Split-typed keys) so split_h5_uri's
    # Literal narrowing holds without a cast.
    for split in spec.split_shard_ranges:
        split_h5 = work_dir / f"{split}.h5"
        if split_h5.exists():
            split_uri = spec.r2.split_h5_uri(split)
            r2_io.upload(split_h5, split_uri)
            logger.info("uploaded {} to {}", split_h5.name, split_uri)
    r2_io.upload(stats_npz, spec.r2.stats_uri())
    logger.info("uploaded stats to {}", spec.r2.stats_uri())


def _lance_split_batches(
    shard_uris: list[str], storage_options: dict[str, str]
) -> LanceSplitBatches:
    """Return the schema and batch iterator for a finalized Lance split.

    Reads shards directly from R2 (no local download) — one sequential pass per
    shard, which Lance streams natively over object storage.

    :param shard_uris: Non-empty list of ``s3://`` shard dataset URIs in split order.
    :param storage_options: Object-store config for the R2 bucket.
    :returns: ``(schema, batches)`` for :func:`write_lance_dataset`.
    :rtype: LanceSplitBatches
    """
    import lance

    schema = lance.dataset(shard_uris[0], storage_options=storage_options).schema

    def _batches() -> LanceBatchIterator:
        for uri in shard_uris:
            yield from lance.dataset(uri, storage_options=storage_options).to_batches()

    return schema, _batches()


def finalize_lance(spec: DatasetSpec, work_dir: Path) -> None:
    """Stream Lance shards from R2 into split datasets, compute stats, upload artifacts.

    Shards are read directly from R2 and each split dataset is written straight to
    its R2 URI via Lance ``storage_options`` — no shard download or split upload.
    Only ``stats.npz`` (a plain numpy archive) is staged locally and uploaded.

    :param spec: Validated dataset spec (``output_format == "lance"``).
    :param work_dir: Scratch directory for the finalized ``stats.npz``.
    :raises ValueError: The train split is empty.
    """
    from synth_setter.pipeline.data.lance_shard import write_lance_dataset
    from synth_setter.pipeline.data.stats import stream_stats_lance

    train_lo, train_hi = spec.split_shard_ranges["train"]
    if train_lo >= train_hi:
        raise ValueError(
            f"train split is empty (split_shard_ranges['train']="
            f"{spec.split_shard_ranges['train']!r}); cannot compute stats "
            f"without at least one train shard."
        )
    storage_options = r2_io.r2_storage_options()

    def _shard_s3_uri(shard: ShardSpec) -> str:
        return r2_io.to_s3_uri(spec.r2.shard_uri(shard))

    train_uris = [_shard_s3_uri(shard) for shard in spec.shards[train_lo:train_hi]]
    mean, std = stream_stats_lance(
        train_uris, mask_degenerate=spec.mask_degenerate_bins, storage_options=storage_options
    )
    stats_npz = work_dir / STATS_NPZ_FILENAME
    np.savez(stats_npz, mean=mean, std=std)

    for split, (lo, hi) in spec.split_shard_ranges.items():
        if lo >= hi:
            continue
        shard_uris = [_shard_s3_uri(shard) for shard in spec.shards[lo:hi]]
        schema, batches = _lance_split_batches(shard_uris, storage_options)
        split_uri = spec.r2.split_lance_uri(split)
        write_lance_dataset(
            r2_io.to_s3_uri(split_uri), schema, batches, storage_options=storage_options
        )
        logger.info("wrote {} split to {}", split, split_uri)
    r2_io.upload(stats_npz, spec.r2.stats_uri())
    logger.info("uploaded stats to {}", spec.r2.stats_uri())


def finalize_from_spec(spec: DatasetSpec, work_dir: Path) -> None:
    """Finalize a dataset given an in-memory spec; idempotent on ``dataset.complete``.

    Returns without work when the marker already exists at the run prefix —
    R2 is the source of truth (per ``pipeline/CLAUDE.md``). The branch on
    ``spec.output_format`` writes the derived artifacts; the marker is
    uploaded strictly last so an interrupted run never advertises artifacts
    that have not landed. Caller is responsible for ensuring R2 creds are
    loaded.

    :param spec: Validated dataset spec.
    :param work_dir: Writable scratch dir; created if missing; retained
        after the call (multi-GB on the hdf5 branch).
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
    if spec.output_format is OutputFormat.WDS:
        finalize_wds(spec, work_dir)
    elif spec.output_format is OutputFormat.HDF5:
        finalize_hdf5(spec, work_dir)
    elif spec.output_format is OutputFormat.LANCE:
        finalize_lance(spec, work_dir)
    else:
        raise ValueError(f"unsupported output_format: {spec.output_format!r}")

    marker_local = work_dir / DATASET_COMPLETE_FILENAME
    marker_local.touch()
    r2_io.upload(marker_local, marker_uri)
    logger.info("wrote dataset.complete to {}", marker_uri)


def _r2_to_s3_uri(r2_uri: str) -> str:
    """Rewrite an ``r2://`` URI to the ``s3://`` scheme W&B references record.

    R2 exposes an S3-compatible API; only the scheme differs, so the
    bucket/key path is preserved verbatim. ``storage-provenance-spec.md`` §4
    logs dataset references as ``s3://``.

    :param r2_uri: An ``r2://<bucket>/<key>`` URI (e.g. from ``R2Location``).
    :returns: The same location as ``s3://<bucket>/<key>``.
    :raises ValueError: ``r2_uri`` does not start with the ``r2://`` scheme.
    """
    scheme = "r2://"
    if not r2_uri.startswith(scheme):
        raise ValueError(f"expected an r2:// URI, got {r2_uri!r}")
    return f"s3://{r2_uri[len(scheme) :]}"


def _finalized_reference_uris(spec: DatasetSpec) -> list[str]:
    """Return the R2 URIs of the objects finalize materialized for this run.

    hdf5 reshards into per-split ``.h5`` files, so each non-empty split plus
    ``stats.npz`` is referenced. wds leaves shards in place under the run
    prefix, so the prefix dir (carrying the tars) plus ``stats.npz`` is
    referenced. Empty splits contribute nothing — finalize prunes them.

    :param spec: Validated dataset spec.
    :returns: Canonical ``r2://`` URIs, splits/prefix first then ``stats.npz``.
    """
    if spec.output_format is OutputFormat.HDF5:
        split_uris = [
            spec.r2.split_h5_uri(split)
            for split, (lo, hi) in spec.split_shard_ranges.items()
            if lo < hi
        ]
        return [*split_uris, spec.r2.stats_uri()]
    if spec.output_format is OutputFormat.LANCE:
        split_uris = [
            spec.r2.split_lance_uri(split)
            for split, (lo, hi) in spec.split_shard_ranges.items()
            if lo < hi
        ]
        return [*split_uris, spec.r2.stats_uri()]
    return [spec.r2.uri(spec.r2.prefix), spec.r2.stats_uri()]


def build_dataset_artifact(spec: DatasetSpec) -> wandb.Artifact:
    """Build the canonical ``dataset`` W&B artifact for a finalized run.

    Names the artifact ``data-{spec.task_name}`` (type ``dataset``) per
    ``storage-provenance-spec.md`` §4, references the finalized R2 objects as
    ``s3://`` URIs (split ``.h5`` files or the shard prefix, plus
    ``stats.npz``), and records ``shard_count`` / ``n_samples`` / ``git_sha``
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
        artifact.add_reference(_r2_to_s3_uri(r2_uri), checksum=False)
    return artifact


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
            lg.experiment.log_artifact(build_dataset_artifact(spec))
        except Exception as exc:  # noqa: BLE001 — wandb artifact failure must not abort finalize
            logger.warning(f"_log_dataset_artifact failed on {type(lg).__name__}: {exc}")


def finalize(cfg: DictConfig) -> None:  # noqa: DOC503
    """Finalize the R2 prefix at ``cfg.dataset_root_uri``; idempotent on ``dataset.complete``.

    Loads R2 creds and the spec from ``input_spec.json`` under
    ``cfg.dataset_root_uri``, delegates to
    :func:`finalize_from_spec` for the marker-probe → dispatch → marker-upload
    body, then logs the canonical ``dataset`` artifact to any configured
    ``WandbLogger`` (resuming the data-generation run pinned to ``spec.run_id``
    so the artifact lands on the producer node of the lineage DAG). The wandb
    run id is pinned and ``resume=allow`` is forced so finalize attaches to the
    generation run rather than minting a new one; both are no-ops when
    ``cfg`` carries no ``logger`` group (the wandb-free default). On any
    failure the loggers are still closed (status ``"failed"``) before the
    exception re-raises.

    :param cfg: Composed cfg with ``dataset_root_uri`` (the run-prefix dir
        accepted by :func:`~synth_setter.pipeline.spec_io.load_spec_from_root`),
        ``paths.output_dir`` (writable scratch dir; created if missing;
        retained after the call, multi-GB on the hdf5 branch), and an optional
        ``logger`` group instantiated for W&B artifact logging.
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
    try:
        finalize_from_spec(spec, Path(cfg.paths.output_dir))
        _log_dataset_artifact(loggers, spec)
    except BaseException:
        status = "failed"
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
    main()
