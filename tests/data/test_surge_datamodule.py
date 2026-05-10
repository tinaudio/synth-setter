"""Tests for src/data/surge_datamodule.py — kwarg-swallowing contract."""

from __future__ import annotations

from pathlib import Path

from src.data.surge_datamodule import SurgeDataModule


def test_init_accepts_extra_datagen_kwargs(tmp_path: Path) -> None:
    """``SurgeDataModule`` swallows datagen-only kwargs that come from the shared
    ``configs/data/surge*.yaml`` group.

    The training-side LightningDataModule reads pre-rendered shards and has no
    use for ``train_val_test_sizes`` / ``train_val_test_seeds`` / ``base_seed``,
    but the dataset-pipeline launcher consumes the same group. ``__init__``
    must therefore accept-and-ignore them via ``**_unused_datagen_keys`` —
    pinned here so a future signature tightening that drops the catch-all
    fails this test instead of silently breaking Hydra composition for the
    training side.
    """
    module = SurgeDataModule(
        dataset_root=tmp_path,
        train_val_test_sizes=[100, 0, 0],
        train_val_test_seeds=[1, 2, 3],
        base_seed=42,
    )

    assert module.dataset_root == tmp_path
