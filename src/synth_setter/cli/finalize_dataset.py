"""``synth-setter-finalize-dataset`` entrypoint: post-generate finalize stage.

Mirrors ``generate-dataset``'s operator-side shape — programmatic Hydra
compose, single ``DatasetSpec`` input, dispatch on ``spec.output_format``.
The ``hdf5`` branch raises ``NotImplementedError`` in this slice; the wds
branch downloads the train-split shards, computes stats, then writes the
``dataset.complete`` marker last per ``pipeline/CLAUDE.md`` invariants.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import NoReturn

import rootutils
from hydra import compose, initialize_config_dir
from loguru import logger

# Bootstrap PROJECT_ROOT + sys.path before sibling synth_setter imports.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from synth_setter.cli.generate_dataset import spec_from_cfg  # noqa: E402
from synth_setter.pipeline import r2_io  # noqa: E402
from synth_setter.pipeline.data.stats import get_stats_wds  # noqa: E402
from synth_setter.pipeline.schemas.spec import DatasetSpec  # noqa: E402

# Resolve repo root from this file so the entrypoint is cwd-independent.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = _REPO_ROOT / "configs"

_DATASET_COMPLETE_FILENAME = "dataset.complete"
_STATS_NPZ_FILENAME = "stats.npz"


def finalize_wds(spec: DatasetSpec, work_dir: Path) -> None:
    """Compute stats over the train-split shards and upload ``stats.npz``.

    Per-shard tar files stay in their original R2 location; only the
    derived ``stats.npz`` is materialized here. The wds brace pattern for
    each split is available at read time via
    ``spec.r2.split_wds_brace_uri(spec.split_shard_ranges[split])``.

    :param spec: Validated dataset spec (``output_format == "wds"``).
    :param work_dir: Scratch directory; train shards land under
        ``<work_dir>/train_shards/`` and ``stats.npz`` is written next to them.
    """
    train_lo, train_hi = spec.split_shard_ranges["train"]
    train_shards_dir = work_dir / "train_shards"
    train_shards_dir.mkdir()
    for shard in spec.shards[train_lo:train_hi]:
        r2_io.download_to_path(spec.r2.shard_uri(shard), train_shards_dir / shard.filename)
    get_stats_wds(train_shards_dir)
    r2_io.upload(train_shards_dir / _STATS_NPZ_FILENAME, spec.r2.stats_uri())
    logger.info("uploaded stats to {}", spec.r2.stats_uri())


def finalize_hdf5(spec: DatasetSpec, work_dir: Path) -> NoReturn:
    """Stub raising ``NotImplementedError``; the hdf5 body lands in Phase 2.

    :param spec: Validated dataset spec (``output_format == "hdf5"``); unused.
    :param work_dir: Scratch directory; unused.
    :raises NotImplementedError: Always — the hdf5 branch is deferred.
    """
    del spec, work_dir
    raise NotImplementedError("hdf5 finalize lands in Phase 2")


def main() -> None:
    """Operator CLI: compose dataset cfg from argv, then finalize the run.

    Programmatic compose (not ``@hydra.main``) so the branch decision on
    ``spec.output_format`` happens before any work runs and the operator
    surface stays compatible with a future SkyPilot dispatch branch
    (tracked as #1181). ``dataset.complete`` is uploaded strictly last;
    its presence is the canonical "this run is ready to train" signal.

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

    with tempfile.TemporaryDirectory() as raw_work_dir:
        work_dir = Path(raw_work_dir)
        if spec.output_format == "wds":
            finalize_wds(spec, work_dir)
        elif spec.output_format == "hdf5":
            finalize_hdf5(spec, work_dir)
        else:
            raise ValueError(f"unsupported output_format: {spec.output_format!r}")
        marker_local = work_dir / _DATASET_COMPLETE_FILENAME
        marker_local.touch()
        r2_io.upload(marker_local, spec.r2.dataset_complete_marker_uri())
    logger.info("wrote dataset.complete to {}", spec.r2.dataset_complete_marker_uri())


if __name__ == "__main__":
    main()
