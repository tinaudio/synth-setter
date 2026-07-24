"""Behavioral tests for txid-pinned materializing hydration in ``LanceVSTDataModule``.

Sources are real local Lance datasets written through the pipeline writer, so
``prepare_data()`` drives the real ``materialize_lance_subset`` path; only the
rclone sidecar boundary is replaced, mirroring the existing hydration tests.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import hydra
import lance
import pytest
from omegaconf import OmegaConf

from synth_setter.data.lance_datamodule import LanceVSTDataModule
from synth_setter.param_spec_name import ParamSpecName
from tests.helpers.lance_fixtures import (
    NUM_PARAMS,
    write_mel_stats,
    write_seeded_lance_shard,
)

_PARAM_SPEC = ParamSpecName("surge_xt")


@pytest.fixture
def source_root(tmp_path: Path) -> Path:
    """Build a hydration source with ``train/val/test.lance`` + ``stats.npz``.

    :param tmp_path: Per-test tmpdir.
    :return: Path to the populated source dataset root.
    """
    root = tmp_path / "source"
    root.mkdir()
    write_seeded_lance_shard(root / "train.lance", num_rows=8, seed=1)
    write_seeded_lance_shard(root / "val.lance", num_rows=6, seed=2)
    write_seeded_lance_shard(root / "test.lance", num_rows=6, seed=3)
    write_mel_stats(root)
    return root


def _split_txid(split_path: Path) -> str:
    """Read the transaction uuid of a split's current version.

    :param split_path: Local Lance dataset directory.
    :return: Transaction uuid pinning the current version.
    """
    dataset = lance.dataset(str(split_path))
    transaction = dataset.read_transaction(dataset.version)
    assert transaction is not None
    return transaction.uuid


def _txids(source_root: Path) -> dict[str, str]:
    """Pin every split of a source root by its current transaction uuid.

    :param source_root: Directory holding ``train/val/test.lance``.
    :return: ``{split: txid}`` mapping for all three splits.
    """
    return {
        split: _split_txid(source_root / f"{split}.lance")
        for split in ("train", "val", "test")
    }


def _sidecar_copier(
    source_root: Path,
) -> tuple[Callable[..., None], list[dict[str, object]]]:
    """Build a rclone-boundary stand-in that copies only ``stats.npz``.

    :param source_root: Hydration source directory holding ``stats.npz``.
    :return: Replacement for ``download_dir_no_overwrite`` and its call record.
    """
    calls: list[dict[str, object]] = []

    def hydrate(source_uri: str, dest_path: Path, exclude: str | None = None) -> None:
        calls.append({"source_uri": source_uri, "dest": dest_path, "exclude": exclude})
        dest_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_root / "stats.npz", dest_path / "stats.npz")

    return hydrate, calls


class TestMaterializeInitValidation:
    """``__init__`` fails loudly on inconsistent materialization configuration."""

    def test_init_materialize_missing_split_txid_raises(self, tmp_path: Path) -> None:
        """A mapping that omits a needed split is rejected.

        :param tmp_path: Local dataset root.
        """
        with pytest.raises(ValueError, match="val"):
            LanceVSTDataModule(
                dataset_root=tmp_path,
                download_dataset_root_uri="r2://experiments/data/ds",
                download_dataset_txids={"train": "t1", "test": "t3"},
                param_spec_name=_PARAM_SPEC,
            )

    def test_init_materialize_unknown_split_key_raises(self, tmp_path: Path) -> None:
        """A txid keyed by an unknown split name is rejected.

        :param tmp_path: Local dataset root.
        """
        with pytest.raises(ValueError, match="predict"):
            LanceVSTDataModule(
                dataset_root=tmp_path,
                download_dataset_root_uri="r2://experiments/data/ds",
                download_dataset_txids={
                    "train": "t1",
                    "val": "t2",
                    "test": "t3",
                    "predict": "t4",
                },
                param_spec_name=_PARAM_SPEC,
            )

    def test_init_materialize_without_download_uri_raises(self, tmp_path: Path) -> None:
        """Materialization without a hydration source is meaningless and rejected.

        :param tmp_path: Local dataset root.
        """
        with pytest.raises(ValueError, match="download_dataset_root_uri"):
            LanceVSTDataModule(
                dataset_root=tmp_path,
                download_dataset_txids={"train": "t1", "val": "t2", "test": "t3"},
                param_spec_name=_PARAM_SPEC,
            )

    def test_init_row_limit_without_txids_raises(self, tmp_path: Path) -> None:
        """A row cap without txids is rejected: a full download cannot cap rows.

        :param tmp_path: Local dataset root.
        """
        with pytest.raises(ValueError, match="download_dataset_row_limit"):
            LanceVSTDataModule(
                dataset_root=tmp_path,
                download_dataset_row_limit=100,
                param_spec_name=_PARAM_SPEC,
            )


class TestMaterializePrepareData:
    """``prepare_data()`` rematerializes projected, row-capped local splits."""

    def test_prepare_data_materialize_on_builds_projected_row_capped_splits(
        self, source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each split lands locally with only the derived columns and the row cap.

        :param source_root: Fixture-provided hydration source.
        :param tmp_path: Parent of the local dataset root.
        :param monkeypatch: Fixture replacing the separately tested rclone boundary.
        """
        destination = tmp_path / "local"
        hydrate, hydrate_calls = _sidecar_copier(source_root)
        monkeypatch.setattr(
            "synth_setter.data.vst_datamodule.r2_io.download_dir_no_overwrite", hydrate
        )
        module = LanceVSTDataModule(
            dataset_root=destination,
            download_dataset_root_uri=source_root.as_uri(),
            download_dataset_txids=_txids(source_root),
            download_dataset_row_limit=4,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            param_spec_name=_PARAM_SPEC,
        )

        module.prepare_data()

        for split in ("train", "val"):
            dataset = lance.dataset(str(destination / f"{split}.lance"))
            assert dataset.schema.names == ["param_array", "mel_spec"]
            assert dataset.count_rows() == 4
        # The default predict file is test.lance, so its projection keeps audio.
        test_split = lance.dataset(str(destination / "test.lance"))
        assert test_split.schema.names == ["param_array", "mel_spec", "audio"]
        assert test_split.count_rows() == 4
        assert hydrate_calls == [
            {
                "source_uri": source_root.as_uri(),
                "dest": destination,
                "exclude": "{*.lance/**,metadata/**}",
            }
        ]

    def test_instantiate_via_hydra_materialize_roundtrip(
        self, source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hydra instantiation materializes splits from a DictConfig txid map.

        Guards that ``download_dataset_txids`` arriving as an OmegaConf
        ``DictConfig`` converts to a plain ``dict`` at the constructor boundary
        and that ``prepare_data()`` then drives the real materialize path.

        :param source_root: Local multi-split Lance source.
        :param tmp_path: Fresh destination root.
        :param monkeypatch: Replaces the rclone sidecar boundary.
        """
        monkeypatch.setattr(
            "synth_setter.data.vst_datamodule.r2_io.download_dir_no_overwrite",
            _sidecar_copier(source_root)[0],
        )
        txids = _txids(source_root)
        dest_root = tmp_path / "root"
        cfg = OmegaConf.create(
            {
                "_target_": "synth_setter.data.lance_datamodule.LanceVSTDataModule",
                "dataset_root": str(dest_root),
                "download_dataset_root_uri": f"file://{source_root}",
                "download_dataset_txids": dict(txids),
                "download_dataset_row_limit": 4,
                "param_spec_name": "surge_xt",
            }
        )

        datamodule = hydra.utils.instantiate(cfg)

        assert isinstance(datamodule.download_dataset_txids, dict)
        assert datamodule.download_dataset_txids == dict(txids)
        datamodule.prepare_data()
        for split in ("train", "val", "test"):
            materialized = lance.dataset(str(dest_root / f"{split}.lance"))
            assert materialized.count_rows() == 4

    def test_prepare_data_materialized_root_feeds_train_dataloader(
        self, source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The materialized root is consumable by the normal Lightning flow.

        :param source_root: Fixture-provided hydration source.
        :param tmp_path: Parent of the local dataset root.
        :param monkeypatch: Fixture replacing the separately tested rclone boundary.
        """
        destination = tmp_path / "local"
        monkeypatch.setattr(
            "synth_setter.data.vst_datamodule.r2_io.download_dir_no_overwrite",
            _sidecar_copier(source_root)[0],
        )
        module = LanceVSTDataModule(
            dataset_root=destination,
            download_dataset_root_uri=source_root.as_uri(),
            download_dataset_txids=_txids(source_root),
            download_dataset_row_limit=4,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            param_spec_name=_PARAM_SPEC,
        )

        module.prepare_data()
        module.setup("fit")
        try:
            batch = next(iter(module.train_dataloader()))
        finally:
            module.teardown()

        assert batch["params"].shape == (2, NUM_PARAMS)
        assert batch["mel_spec"] is not None

    def test_prepare_data_materialize_external_predict_file_omits_audio(
        self, source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With predict served elsewhere, the test split drops the audio column.

        :param source_root: Fixture-provided hydration source.
        :param tmp_path: Parent of the local dataset root.
        :param monkeypatch: Fixture replacing the separately tested rclone boundary.
        """
        destination = tmp_path / "local"
        monkeypatch.setattr(
            "synth_setter.data.vst_datamodule.r2_io.download_dir_no_overwrite",
            _sidecar_copier(source_root)[0],
        )
        module = LanceVSTDataModule(
            dataset_root=destination,
            download_dataset_root_uri=source_root.as_uri(),
            download_dataset_txids=_txids(source_root),
            predict_file=tmp_path / "elsewhere" / "predict.lance",
            param_spec_name=_PARAM_SPEC,
        )

        module.prepare_data()

        test_split = lance.dataset(str(destination / "test.lance"))
        assert test_split.schema.names == ["param_array", "mel_spec"]

    def test_prepare_data_materialize_m2l_conditioning_projects_music2latent(
        self, source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The projected conditioning column follows the configured conditioning.

        :param source_root: Fixture-provided hydration source.
        :param tmp_path: Parent of the local dataset root.
        :param monkeypatch: Fixture replacing the separately tested rclone boundary.
        """
        destination = tmp_path / "local"
        monkeypatch.setattr(
            "synth_setter.data.vst_datamodule.r2_io.download_dir_no_overwrite",
            _sidecar_copier(source_root)[0],
        )
        module = LanceVSTDataModule(
            dataset_root=destination,
            download_dataset_root_uri=source_root.as_uri(),
            download_dataset_txids=_txids(source_root),
            conditioning="m2l",
            predict_file=tmp_path / "elsewhere" / "predict.lance",
            param_spec_name=_PARAM_SPEC,
        )

        module.prepare_data()

        train_split = lance.dataset(str(destination / "train.lance"))
        assert train_split.schema.names == ["param_array", "music2latent"]
