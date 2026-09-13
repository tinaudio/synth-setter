"""Live end-to-end materializing hydration of ``LanceVSTDataModule`` from real R2.

No fakes, no mocks, no local-backend remote: ``prepare_data()`` streams the
projected split subsets from a production-written R2 dataset through the same
``r2_io`` credentials path training uses, and rclone fetches the non-Lance
sidecars. Read-only on R2 — everything lands on local disk only.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import lance
import numpy as np
import pytest
import torch

from synth_setter.conditioning import EmbeddingConditioningSpec
from synth_setter.data.lance_datamodule import LanceVSTDataModule
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.constants import conditioning_stats_filename
from synth_setter.pipeline.data.stats import get_conditioning_stats_lance
from tests.helpers.lance_fixtures import write_mel_stats, write_seeded_lance_shard

pytestmark = [pytest.mark.integration_r2, pytest.mark.r2, pytest.mark.slow]

# Small (1k/2k/2k-row) production-written dataset; read-only fixture for this test.
_ROOT_URI = (
    "r2://experiments/data/surge-simple-lance-1k-2k-2k/"
    "surge-simple-lance-1k-2k-2k-20260716T163226347Z"
)
_SUBSET_ROWS = 8
_BATCH_SIZE = 4


def test_prepare_data_live_r2_transfers_and_applies_conditioning_statistics(
    tmp_path: Path,
) -> None:
    """Real R2 hydration carries a required affine into its dataloader consumer.

    :param tmp_path: Local source and hydration workspace.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone missing or rclone lsd r2: failed)")
    r2_io.ensure_r2_env_loaded()

    source = tmp_path / "source"
    source.mkdir()
    for index, split in enumerate(("train", "val", "test"), start=1):
        write_seeded_lance_shard(source / f"{split}.lance", num_rows=2, seed=index)
    write_mel_stats(source)
    (source / "dataset.complete").touch()
    get_conditioning_stats_lance(
        source / "train.lance",
        column="music2latent",
        input_shape=(6, 7),
        normalization="per_channel",
    )

    prefix = f"pr-verification/conditioning-hydration-{uuid.uuid4().hex}/"
    source_uri = f"r2://test-bucket/{prefix}"
    try:
        r2_io.upload_dir(source, source_uri)
        module = LanceVSTDataModule(
            dataset_root=tmp_path / "hydrated",
            download_dataset_root_uri=source_uri,
            download_dataset_row_limit=2,
            batch_size=2,
            conditioning=EmbeddingConditioningSpec(
                column="music2latent",
                input_shape=(6, 7),
                normalization="per_channel",
            ),
            use_saved_mean_and_variance=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            param_spec_name=ParamSpecName("surge_xt"),
        )

        module.prepare_data()
        artifact = module.dataset_root / conditioning_stats_filename("music2latent")
        assert artifact.is_file()
        module.setup("validate")
        try:
            conditioning = next(iter(module.val_dataloader()))["conditioning"]
        finally:
            module.teardown()

        assert conditioning is not None
        with np.load(artifact) as stats:
            raw = (
                lance.dataset(str(module.dataset_root / "val.lance"))
                .to_table(columns=["music2latent"])
                .column(0)
                .combine_chunks()
                .to_numpy_ndarray()
            )
            expected = (raw - stats["mean"][:, None]) / stats["std"][:, None]
        torch.testing.assert_close(conditioning, torch.from_numpy(expected))
        assert torch.isfinite(conditioning).all()
    finally:
        r2_io.purge_prefix("test-bucket", prefix)


def test_prepare_data_live_r2_materializes_splits_and_feeds_dataloader(
    tmp_path: Path,
) -> None:
    """Full production hydration: latest row-limited splits land and train loads.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone missing or rclone lsd r2: failed)")
    r2_io.ensure_r2_env_loaded()

    destination = tmp_path / "data"
    module = LanceVSTDataModule(
        dataset_root=destination,
        download_dataset_root_uri=_ROOT_URI,
        download_dataset_row_limit=_SUBSET_ROWS,
        batch_size=_BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        param_spec_name=ParamSpecName("surge_simple"),
    )

    module.prepare_data()

    materialized_root = module.dataset_root
    assert materialized_root.parent == destination
    for split in ("train", "val"):
        dataset = lance.dataset(str(materialized_root / f"{split}.lance"))
        assert dataset.schema.names == ["param_array", "mel_spec"]
        assert dataset.count_rows() == _SUBSET_ROWS
    # test.lance doubles as the default predict split, so it retains audio.
    test_split = lance.dataset(str(materialized_root / "test.lance"))
    assert test_split.schema.names == ["param_array", "mel_spec", "audio"]
    assert test_split.count_rows() == _SUBSET_ROWS
    assert (materialized_root / "dataset.complete").is_file()
    assert (materialized_root / "stats.npz").is_file()
    # Pipeline-internal worker metadata must not ride along with the sidecars.
    assert not (materialized_root / "metadata").exists()

    module.setup("fit")
    try:
        batch = next(iter(module.train_dataloader()))
    finally:
        module.teardown()
    params = batch["params"]
    mel = batch["mel"]
    assert params is not None and params.shape[0] == _BATCH_SIZE
    assert mel is not None and mel.shape[0] == _BATCH_SIZE
