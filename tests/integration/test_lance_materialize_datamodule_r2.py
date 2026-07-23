"""Live end-to-end materializing hydration of ``LanceVSTDataModule`` from real R2.

No fakes, no mocks, no local-backend remote: ``prepare_data()`` streams the
projected split subsets from a production-written R2 dataset through the same
``r2_io`` credentials path training uses, and rclone fetches the non-Lance
sidecars. Read-only on R2 — everything lands on local disk only.
"""

from __future__ import annotations

from pathlib import Path

import lance
import pytest

from synth_setter.data.lance_datamodule import LanceVSTDataModule
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline import r2_io

pytestmark = [pytest.mark.integration_r2, pytest.mark.r2, pytest.mark.slow]

# Small (1k/2k/2k-row) production-written dataset; read-only fixture for this test.
_ROOT_URI = (
    "r2://experiments/data/surge-simple-lance-1k-2k-2k/"
    "surge-simple-lance-1k-2k-2k-20260716T163226347Z"
)
_SUBSET_ROWS = 8
_BATCH_SIZE = 4


def _live_txid(split_uri: str) -> str:
    """Pin a live split's current version by its real transaction uuid.

    :param split_uri: ``r2://`` URI of one split dataset.
    :return: Transaction uuid of the split's current version.
    """
    open_uri, storage_options = r2_io.lance_target(split_uri)
    dataset = lance.dataset(open_uri, storage_options=storage_options)
    transaction = dataset.read_transaction(dataset.version)
    assert transaction is not None
    return transaction.uuid


def test_prepare_data_live_r2_materializes_splits_and_feeds_dataloader(
    tmp_path: Path,
) -> None:
    """Full production hydration: txid-pinned subsets land locally and train loads.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip(
            "R2 not reachable (rclone missing, RCLONE_CONFIG_R2_* env vars missing, "
            "or rclone lsd r2: failed)"
        )
    r2_io.ensure_r2_env_loaded()
    txids = {split: _live_txid(f"{_ROOT_URI}/{split}.lance") for split in ("train", "val", "test")}

    destination = tmp_path / "data"
    module = LanceVSTDataModule(
        dataset_root=destination,
        download_dataset_root_uri=_ROOT_URI,
        materialize_columns=True,
        dataset_txids=txids,
        subset_rows=_SUBSET_ROWS,
        batch_size=_BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        param_spec_name=ParamSpecName("surge_simple"),
    )

    module.prepare_data()

    for split in ("train", "val"):
        dataset = lance.dataset(str(destination / f"{split}.lance"))
        assert dataset.schema.names == ["param_array", "mel_spec"]
        assert dataset.count_rows() == _SUBSET_ROWS
    # test.lance doubles as the default predict split, so it retains audio.
    test_split = lance.dataset(str(destination / "test.lance"))
    assert test_split.schema.names == ["param_array", "mel_spec", "audio"]
    assert test_split.count_rows() == _SUBSET_ROWS
    assert (destination / "stats.npz").is_file()
    # Pipeline-internal worker metadata must not ride along with the sidecars.
    assert not (destination / "metadata").exists()

    module.setup("fit")
    try:
        batch = next(iter(module.train_dataloader()))
    finally:
        module.teardown()
    params = batch["params"]
    mel_spec = batch["mel_spec"]
    assert params is not None and params.shape[0] == _BATCH_SIZE
    assert mel_spec is not None and mel_spec.shape[0] == _BATCH_SIZE
