"""``synth-setter-finalize-dataset`` entrypoint: post-generate finalize stage.

Mirrors ``generate-dataset``'s operator-side shape — programmatic Hydra
compose, single ``DatasetSpec`` input, dispatch on ``spec.output_format``.
Both branches upload their derived artifact(s) and then write the
``dataset.complete`` marker last per ``pipeline/CLAUDE.md``. The wds
branch streams train shards through Welford row-by-row; the hdf5 branch
downloads every shard, reshards into ``{train,val,test}.h5``, and
computes ``stats.npz`` over the train split.
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import rootutils
from hydra import compose, initialize_config_dir
from loguru import logger

# Bootstrap PROJECT_ROOT + sys.path before sibling synth_setter imports.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np  # noqa: E402

from synth_setter.cli.generate_dataset import spec_from_cfg  # noqa: E402
from synth_setter.pipeline import r2_io  # noqa: E402
from synth_setter.pipeline.constants import (  # noqa: E402
    DATASET_COMPLETE_FILENAME,
    INPUT_SPEC_FILENAME,
    STATS_NPZ_FILENAME,
)
from synth_setter.pipeline.data import reshard  # noqa: E402
from synth_setter.pipeline.data.stats import get_stats_hdf5, stream_stats_wds  # noqa: E402
from synth_setter.pipeline.schemas.spec import DatasetSpec  # noqa: E402

# Resolve repo root from this file so the entrypoint is cwd-independent.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = _REPO_ROOT / "configs"


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
    ``ValueError`` at that helper.

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
    mean, std = stream_stats_wds(_download_train_shards_one_at_a_time(spec, work_dir))
    stats_npz = work_dir / STATS_NPZ_FILENAME
    np.savez(stats_npz, mean=mean, std=std)
    r2_io.upload(stats_npz, spec.r2.stats_uri())
    logger.info("uploaded stats to {}", spec.r2.stats_uri())


def finalize_hdf5(spec: DatasetSpec, work_dir: Path) -> None:
    """Download every shard, reshard into split files, compute stats, upload all artifacts.

    Materializes ``input_spec.json`` sibling to the shards so reshard's
    default spec-path resolution works without a ``--spec`` override. Stats
    land at ``work_dir / "stats.npz"`` per ``SurgeXTDataset.get_stats_file_path``.

    :param spec: Validated dataset spec (``output_format == "hdf5"``).
    :param work_dir: Scratch directory; shards, splits, stats and the spec
        copy live here transiently for the duration of the call.
    :raises ValueError: The train split is empty
        (``spec.split_shard_ranges["train"]`` has ``lo >= hi``); reshard
        would prune ``train.h5`` and stats compute would fail with a
        low-signal HDF5 error.
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
    (work_dir / INPUT_SPEC_FILENAME).write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    reshard.main.callback(dataset_root=work_dir, spec_uri=None)  # type: ignore[misc]
    get_stats_hdf5(str(work_dir / "train.h5"))
    # Reshard prunes empty splits — only upload the ones it actually wrote.
    # Iterate ``split_shard_ranges`` (Split-typed keys) so split_h5_uri's
    # Literal narrowing holds without a cast.
    for split in spec.split_shard_ranges:
        split_h5 = work_dir / f"{split}.h5"
        if split_h5.exists():
            r2_io.upload(split_h5, spec.r2.split_h5_uri(split))
    r2_io.upload(work_dir / STATS_NPZ_FILENAME, spec.r2.stats_uri())
    logger.info("uploaded h5 splits + stats to {}", spec.r2.rclone_prefix())


def main() -> None:
    """Operator CLI: compose dataset cfg, dispatch on ``output_format``, write marker last.

    Skips the body entirely when ``dataset.complete`` already exists at the
    run prefix — R2 is the source of truth (per ``pipeline/CLAUDE.md``), so a
    second invocation against a finalized prefix is a no-op rather than a
    full redo.

    :raises ValueError: ``spec.output_format`` is neither ``"hdf5"`` nor ``"wds"``.
    """
    overrides = list(sys.argv[1:])
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_DIR)):
        cfg = compose(config_name="dataset", overrides=overrides)

    # Pin paths.* so spec_from_cfg's resolve step does not trip on
    # ${hydra:runtime.output_dir} — programmatic compose leaves it unset.
    cfg.paths.root_dir = str(_REPO_ROOT)
    cfg.paths.output_dir = str(_REPO_ROOT)
    cfg.paths.work_dir = str(_REPO_ROOT)

    spec = spec_from_cfg(cfg)
    r2_io.ensure_r2_env_loaded()

    marker_uri = spec.r2.dataset_complete_marker_uri()
    if r2_io.object_size(marker_uri) is not None:
        logger.info("skip: {} already exists, run is finalized", marker_uri)
        return

    with tempfile.TemporaryDirectory() as raw_work_dir:
        work_dir = Path(raw_work_dir)
        if spec.output_format == "wds":
            finalize_wds(spec, work_dir)
        elif spec.output_format == "hdf5":
            finalize_hdf5(spec, work_dir)
        else:
            raise ValueError(f"unsupported output_format: {spec.output_format!r}")
        marker_local = work_dir / DATASET_COMPLETE_FILENAME
        marker_local.touch()
        r2_io.upload(marker_local, marker_uri)
    logger.info("wrote dataset.complete to {}", marker_uri)


if __name__ == "__main__":
    main()
