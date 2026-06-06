"""``synth-setter-finalize-dataset`` entrypoint: post-generate finalize stage.

Loads the frozen ``DatasetSpec`` from ``cfg.dataset_spec_uri`` (an R2 URI
produced by the upstream generate stage's ``upload_spec``) and dispatches
on ``spec.output_format``. Both branches upload their derived artifact(s)
and then write the ``dataset.complete`` marker last per
``pipeline/CLAUDE.md``. The wds branch streams train shards through
Welford row-by-row; the hdf5 branch downloads every shard, reshards into
``{train,val,test}.h5``, and computes ``stats.npz`` over the train split.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.constants import (
    DATASET_COMPLETE_FILENAME,
    INPUT_SPEC_FILENAME,
    STATS_NPZ_FILENAME,
)
from synth_setter.pipeline.data.reshard import reshard_dataset
from synth_setter.pipeline.data.stats import get_stats_hdf5, stream_stats_wds
from synth_setter.pipeline.schemas.prefix import assert_r2_prefix_matches
from synth_setter.pipeline.schemas.spec import DatasetSpec, OutputFormat
from synth_setter.pipeline.spec_io import load_spec_from_uri, write_spec_to_path
from synth_setter.workspace import operator_workspace

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
    via ``SurgeXTDataset.get_stats_file_path(train.h5)``); the post-call
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
            f"SurgeXTDataset.get_stats_file_path derivation."
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
    :raises ValueError: ``spec.r2.prefix`` does not match
        ``make_r2_prefix(spec.task_name, spec.run_id, spec.r2.prefix_root)``
        (prefix drift detected before any R2 writes), or ``spec.output_format``
        is neither ``"hdf5"`` nor ``"wds"``.
    """
    marker_uri = spec.r2.dataset_complete_marker_uri()
    if r2_io.object_size(marker_uri) is not None:
        logger.info("skip: {} already exists, run is finalized", marker_uri)
        return

    assert_r2_prefix_matches(spec.r2.prefix, spec.task_name, spec.run_id, spec.r2.prefix_root)
    work_dir.mkdir(parents=True, exist_ok=True)
    if spec.output_format is OutputFormat.WDS:
        finalize_wds(spec, work_dir)
    elif spec.output_format is OutputFormat.HDF5:
        finalize_hdf5(spec, work_dir)
    else:
        raise ValueError(f"unsupported output_format: {spec.output_format!r}")

    marker_local = work_dir / DATASET_COMPLETE_FILENAME
    marker_local.touch()
    r2_io.upload(marker_local, marker_uri)
    logger.info("wrote dataset.complete to {}", marker_uri)


def finalize(cfg: DictConfig) -> None:
    """Finalize the R2 prefix for ``cfg.dataset_spec_uri``; idempotent on ``dataset.complete``.

    Loads R2 creds and the spec from ``cfg.dataset_spec_uri``, then delegates
    to :func:`finalize_from_spec` for the marker-probe → dispatch → marker-upload
    body.

    :param cfg: Composed cfg with ``dataset_spec_uri`` (URI accepted by
        :func:`~synth_setter.pipeline.spec_io.load_spec_from_uri`) and
        ``paths.output_dir`` (writable scratch dir; created if missing;
        retained after the call, multi-GB on the hdf5 branch).
    """
    r2_io.ensure_r2_env_loaded()
    spec = load_spec_from_uri(cfg.dataset_spec_uri)
    finalize_from_spec(spec, Path(cfg.paths.output_dir))


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
