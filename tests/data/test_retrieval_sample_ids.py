"""Transient evaluation identities follow real Lance source rows."""

import pickle
from pathlib import Path

import lance
import pytest
import torch

from synth_setter.data.lance_datamodule import (
    LanceVSTDataModule,
    PrepareBatchCollate,
    _RepeatFirstBatchDataset,
)
from synth_setter.data.lance_torch import LanceMapDataset, map_dataloader_over
from synth_setter.evaluation.paired_retrieval import paired_retrieval_metrics
from synth_setter.param_spec_name import ParamSpecName
from tests.helpers.lance_fixtures import write_seeded_lance_shard


def test_source_ids_shuffled_repeated_fetch_preserves_row_alignment(tmp_path: Path) -> None:
    """IDs survive shuffled, padded and ragged fetches into the real metric consumer.

    :param tmp_path: Local Lance destination.
    """
    source = write_seeded_lance_shard(tmp_path / "val.lance", 5)
    dataset = LanceMapDataset(
        tmp_path / "val.lance", columns=["param_array"], include_sample_id=True
    )
    collate = PrepareBatchCollate(mean=None, std=None, rescale_params=False, ot=False)
    loader = map_dataloader_over(
        dataset, batch_size=2, sampler=[4, 1, 4, 0, 3], collate_fn=collate
    )
    batches = list(loader)
    ids = torch.cat([batch["sample_id"] for batch in batches])
    params = torch.cat([batch["params"] for batch in batches])
    torch.testing.assert_close(ids, torch.tensor([4, 1, 4, 0, 3]))
    torch.testing.assert_close(params, torch.from_numpy(source["param_array"])[ids])
    assert paired_retrieval_metrics(params, params, ids.tolist())["gallery_size"] == 4


def test_source_ids_single_and_repeat_first_fetch_preserves_source_identity(
    tmp_path: Path,
) -> None:
    """Single fetch and repeat-first folding identify source rows, not wrapper positions.

    :param tmp_path: Local Lance destination.
    """
    write_seeded_lance_shard(tmp_path / "val.lance", 6)
    dataset = LanceMapDataset(
        tmp_path / "val.lance", columns=["param_array"], include_sample_id=True
    )
    assert dataset[4]["sample_id"].item() == 4
    repeated = _RepeatFirstBatchDataset(dataset, 2)
    torch.testing.assert_close(
        repeated.__getitems__([4, 3, 2])["sample_id"], torch.tensor([0, 1, 0])
    )


def test_source_ids_worker_reopen_keeps_pinned_version(tmp_path: Path) -> None:
    """Reopening a serialized worker cannot relabel changed source rows with old IDs.

    :param tmp_path: Local versioned Lance destination.
    """
    path = tmp_path / "val.lance"
    source = write_seeded_lance_shard(path, 3)
    dataset = LanceMapDataset(path, columns=["param_array"], include_sample_id=True)
    replacement = lance.dataset(str(path)).take([2, 1, 0])
    lance.write_dataset(replacement, str(path), mode="overwrite")
    reopened = pickle.loads(pickle.dumps(dataset))
    torch.testing.assert_close(reopened[0]["param_array"], torch.from_numpy(source["param_array"][0]))
    assert reopened[0]["sample_id"].item() == 0


def test_source_ids_default_loader_does_not_add_metadata(tmp_path: Path) -> None:
    """Unrelated callers retain the original projected-column contract.

    :param tmp_path: Local Lance destination.
    """
    write_seeded_lance_shard(tmp_path / "val.lance", 2)
    assert set(LanceMapDataset(tmp_path / "val.lance", columns=["param_array"])[0]) == {
        "param_array"
    }


def test_source_ids_ot_collate_rejects_misleading_identity() -> None:
    """Reordering must not silently associate predictions with stale source identities."""
    collate = PrepareBatchCollate(mean=None, std=None, rescale_params=False, ot=True)
    with pytest.raises(ValueError, match="sample_id.*OT"):
        collate({"param_array": torch.ones(2, 3), "sample_id": torch.tensor([0, 1])})


def test_source_ids_fake_data_rejects_unstable_identity(tmp_path: Path) -> None:
    """Random draws cannot be assigned stable row identities.

    :param tmp_path: Unused dataset location.
    """
    with pytest.raises(ValueError, match="fake"):
        LanceVSTDataModule(
            dataset_root=tmp_path,
            param_spec_name=ParamSpecName("surge_xt"),
            fake=True,
            eval_sample_ids=True,
        )


def test_source_ids_datamodule_opt_in_only_marks_evaluation(tmp_path: Path) -> None:
    """Only validation and test receive IDs when the datamodule opts in.

    :param tmp_path: Local split root.
    """
    write_seeded_lance_shard(tmp_path / "train.lance", 4)
    write_seeded_lance_shard(tmp_path / "val.lance", 3)
    write_seeded_lance_shard(tmp_path / "test.lance", 3)
    module = LanceVSTDataModule(
        dataset_root=tmp_path,
        param_spec_name=ParamSpecName("surge_xt"),
        use_saved_mean_and_variance=False,
        batch_size=2,
        eval_sample_ids=True,
    )
    module.setup()
    assert "sample_id" not in next(iter(module.train_dataloader()))
    torch.testing.assert_close(
        torch.cat([b["sample_id"] for b in module.val_dataloader()]), torch.tensor([0, 1, 2])
    )
    torch.testing.assert_close(
        torch.cat([b["sample_id"] for b in module.test_dataloader()]), torch.tensor([0, 1, 2])
    )
