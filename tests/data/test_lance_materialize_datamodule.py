"""Behavioral tests for projected hydration in ``LanceVSTDataModule``.

Sources are real local Lance datasets written through the pipeline writer, so
``prepare_data()`` drives the real ``materialize_lance_subset`` path; only the
rclone sidecar boundary is replaced, mirroring the existing hydration tests.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import cast

import hydra
import lance
import pytest
from omegaconf import OmegaConf

from synth_setter.conditioning import ConditioningMode
from synth_setter.data.lance_datamodule import LanceVSTDataModule
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.data.lance_shard import LANCE_DATA_STORAGE_VERSION
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
    (root / "dataset.complete").touch()
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
        split: _split_txid(source_root / f"{split}.lance") for split in ("train", "val", "test")
    }


def _sidecar_copier(
    source_root: Path,
) -> tuple[Callable[..., None], list[dict[str, object]]]:
    """Build a rclone-boundary stand-in that copies root ``.npz`` sidecars.

    :param source_root: Hydration source directory holding statistics artifacts.
    :return: Replacement for ``download_dir_no_overwrite`` and its call record.
    """
    calls: list[dict[str, object]] = []

    def hydrate(source_uri: str, dest_path: Path, exclude: str | None = None) -> None:
        calls.append({"source_uri": source_uri, "dest": dest_path, "exclude": exclude})
        dest_path.mkdir(parents=True, exist_ok=True)
        for sidecar in source_root.glob("*.npz"):
            shutil.copy(sidecar, dest_path / sidecar.name)

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

    def test_init_row_limit_without_download_uri_raises(self, tmp_path: Path) -> None:
        """A row cap without a hydration source is rejected.

        :param tmp_path: Local dataset root.
        """
        with pytest.raises(ValueError, match="download_dataset_root_uri"):
            LanceVSTDataModule(
                dataset_root=tmp_path,
                download_dataset_row_limit=100,
                param_spec_name=_PARAM_SPEC,
            )

    @pytest.mark.parametrize("row_limit", [-1, 0])
    def test_init_non_positive_row_limit_raises(self, tmp_path: Path, row_limit: int) -> None:
        """A non-positive row cap is rejected before hydration.

        :param tmp_path: Local dataset root.
        :param row_limit: Invalid first-N row cap.
        """
        with pytest.raises(ValueError, match="download_dataset_row_limit"):
            LanceVSTDataModule(
                dataset_root=tmp_path,
                download_dataset_root_uri="r2://experiments/data/ds",
                download_dataset_row_limit=row_limit,
                param_spec_name=_PARAM_SPEC,
            )

    def test_init_boolean_row_limit_raises(self, tmp_path: Path) -> None:
        """A boolean is not accepted as an integer row limit.

        :param tmp_path: Local dataset root.
        """
        with pytest.raises(ValueError, match="download_dataset_row_limit"):
            LanceVSTDataModule(
                dataset_root=tmp_path,
                download_dataset_root_uri="r2://experiments/data/ds",
                download_dataset_row_limit=cast(int, True),
                param_spec_name=_PARAM_SPEC,
            )

    def test_init_numeric_txid_raises(self, tmp_path: Path) -> None:
        """Transaction identifiers must be strings at the config boundary.

        :param tmp_path: Local dataset root.
        """
        txids = cast(dict[str, str], {"train": 1, "val": 2, "test": 3})
        with pytest.raises(ValueError, match="download_dataset_txids"):
            LanceVSTDataModule(
                dataset_root=tmp_path,
                download_dataset_root_uri="r2://experiments/data/ds",
                download_dataset_txids=txids,
                param_spec_name=_PARAM_SPEC,
            )

    def test_init_row_limit_without_txids_succeeds(self, tmp_path: Path) -> None:
        """A row cap may select projected materialization from the latest snapshots.

        :param tmp_path: Local dataset root.
        """
        LanceVSTDataModule(
            dataset_root=tmp_path,
            download_dataset_root_uri="r2://experiments/data/ds",
            download_dataset_row_limit=100,
            param_spec_name=_PARAM_SPEC,
        )


class TestMaterializedSubsetLayout:
    """Hydration always projects, into a config-addressed subdirectory."""

    def test_prepare_data_download_uri_alone_projects_away_unread_columns(
        self, source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bare source URI hydrates the read set, not the whole dataset.

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
            predict_file=tmp_path / "elsewhere" / "predict.lance",
            param_spec_name=_PARAM_SPEC,
        )

        module.prepare_data()

        train_split = lance.dataset(str(module.dataset_root / "train.lance"))
        assert train_split.schema.names == ["param_array", "mel_spec"]
        assert train_split.count_rows() == 8

    def test_prepare_data_audio_conditioning_projects_waveform_once_per_split(
        self, source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raw-audio conditioning hydrates one audio column for train, val, and test.

        :param source_root: Fixture-provided hydration source.
        :param tmp_path: Parent of the local dataset root.
        :param monkeypatch: Fixture replacing the separately tested rclone boundary.
        """
        monkeypatch.setattr(
            "synth_setter.data.vst_datamodule.r2_io.download_dir_no_overwrite",
            _sidecar_copier(source_root)[0],
        )
        module = LanceVSTDataModule(
            dataset_root=tmp_path / "local",
            download_dataset_root_uri=source_root.as_uri(),
            conditioning="audio",
            param_spec_name=_PARAM_SPEC,
        )

        module.prepare_data()

        for split in ("train", "val", "test"):
            dataset = lance.dataset(str(module.dataset_root / f"{split}.lance"))
            assert dataset.schema.names == ["param_array", "audio"]

    def test_prepare_data_names_subset_directory_for_the_conditioning_column(
        self, source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The subset directory is a readable prefix plus a request digest.

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
            param_spec_name=_PARAM_SPEC,
        )

        module.prepare_data()

        assert module.dataset_root.parent == destination
        assert module.dataset_root.name.startswith("mel_spec-")

    def test_prepare_data_distinct_conditioning_hydrates_sibling_subsets(
        self, source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Changing conditioning hydrates a new subset instead of failing on the old one.

        :param source_root: Fixture-provided hydration source.
        :param tmp_path: Parent of the shared local dataset root.
        :param monkeypatch: Fixture replacing the separately tested rclone boundary.
        """
        destination = tmp_path / "local"
        monkeypatch.setattr(
            "synth_setter.data.vst_datamodule.r2_io.download_dir_no_overwrite",
            _sidecar_copier(source_root)[0],
        )

        def hydrate(conditioning: ConditioningMode) -> Path:
            module = LanceVSTDataModule(
                dataset_root=destination,
                download_dataset_root_uri=source_root.as_uri(),
                conditioning=conditioning,
                predict_file=tmp_path / "elsewhere" / "predict.lance",
                param_spec_name=_PARAM_SPEC,
            )
            module.prepare_data()
            return module.dataset_root

        mel_root = hydrate("mel")
        m2l_root = hydrate("m2l")

        assert mel_root != m2l_root
        assert lance.dataset(str(mel_root / "train.lance")).schema.names == [
            "param_array",
            "mel_spec",
        ]
        assert lance.dataset(str(m2l_root / "train.lance")).schema.names == [
            "param_array",
            "music2latent",
        ]

    def test_prepare_data_differing_projections_hydrate_separate_subsets(
        self, source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same conditioning, different read set — the subsets must not collide.

        Moving ``predict_file`` off the test split drops its ``audio`` column
        while leaving the conditioning column, so the directory name has to
        separate them on the projection alone.

        :param source_root: Fixture-provided hydration source.
        :param tmp_path: Parent of the shared local dataset root.
        :param monkeypatch: Fixture replacing the separately tested rclone boundary.
        """
        destination = tmp_path / "local"
        monkeypatch.setattr(
            "synth_setter.data.vst_datamodule.r2_io.download_dir_no_overwrite",
            _sidecar_copier(source_root)[0],
        )

        def hydrate(predict_file: Path | None) -> Path:
            module = LanceVSTDataModule(
                dataset_root=destination,
                download_dataset_root_uri=source_root.as_uri(),
                predict_file=predict_file,
                param_spec_name=_PARAM_SPEC,
            )
            module.prepare_data()
            return module.dataset_root

        serving_predict = hydrate(None)
        predict_elsewhere = hydrate(tmp_path / "elsewhere" / "predict.lance")

        assert serving_predict != predict_elsewhere
        assert lance.dataset(str(serving_predict / "test.lance")).schema.names == [
            "param_array",
            "mel_spec",
            "audio",
        ]
        assert lance.dataset(str(predict_elsewhere / "test.lance")).schema.names == [
            "param_array",
            "mel_spec",
        ]

    def test_prepare_data_repeated_identical_config_reuses_the_subset(
        self, source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Re-running the same configuration is a cache hit, not a hard failure.

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
            param_spec_name=_PARAM_SPEC,
        )

        module.prepare_data()
        module.prepare_data()

        assert lance.dataset(str(module.dataset_root / "train.lance")).count_rows() == 8

    def test_prepare_data_ignores_a_legacy_whole_dataset_copy_at_the_root(
        self, source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pre-existing flat copy neither blocks hydration nor gets read.

        :param source_root: Fixture-provided hydration source.
        :param tmp_path: Parent of the local dataset root.
        :param monkeypatch: Fixture replacing the separately tested rclone boundary.
        """
        destination = tmp_path / "local"
        shutil.copytree(source_root, destination)
        monkeypatch.setattr(
            "synth_setter.data.vst_datamodule.r2_io.download_dir_no_overwrite",
            _sidecar_copier(source_root)[0],
        )
        module = LanceVSTDataModule(
            dataset_root=destination,
            download_dataset_root_uri=source_root.as_uri(),
            predict_file=tmp_path / "elsewhere" / "predict.lance",
            param_spec_name=_PARAM_SPEC,
        )

        module.prepare_data()

        assert lance.dataset(str(module.dataset_root / "train.lance")).schema.names == [
            "param_array",
            "mel_spec",
        ]
        assert lance.dataset(str(destination / "train.lance")).schema.names[0] == "audio"

    def test_prepare_data_predict_file_in_configured_root_follows_the_subset(
        self, source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A predict split named against the configured root resolves to the subset.

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
            predict_file=destination / "test.lance",
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            param_spec_name=_PARAM_SPEC,
        )

        module.prepare_data()
        module.setup("predict")
        try:
            batch = next(iter(module.predict_dataloader()))
        finally:
            module.teardown()

        assert module.predict_file == module.dataset_root / "test.lance"
        assert batch["audio"].shape[0] == 2

    def test_materialized_split_pins_the_pipeline_lance_storage_version(
        self, source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Local subsets are written in the version the pipeline pins, not the pylance default.

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
            param_spec_name=_PARAM_SPEC,
        )

        module.prepare_data()

        materialized = lance.dataset(str(module.dataset_root / "train.lance"))
        assert materialized.data_storage_version == LANCE_DATA_STORAGE_VERSION


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

        expected_columns = {
            "train": ["param_array", "mel_spec"],
            "val": ["param_array", "mel_spec"],
            "test": ["param_array", "mel_spec", "audio"],
        }
        for split, columns in expected_columns.items():
            source = lance.dataset(str(source_root / f"{split}.lance"))
            materialized = lance.dataset(str(module.dataset_root / f"{split}.lance"))
            assert materialized.schema.names == columns
            assert materialized.count_rows() == 4
            for column in columns:
                assert materialized.schema.field(column).type == source.schema.field(column).type
            expected_params = source.scanner(columns=["param_array"], limit=4).to_table()
            actual_params = materialized.scanner(columns=["param_array"]).to_table()
            assert actual_params.equals(expected_params)
        assert hydrate_calls == [
            {
                "source_uri": source_root.as_uri(),
                "dest": module.dataset_root,
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
            materialized = lance.dataset(str(datamodule.dataset_root / f"{split}.lance"))
            assert materialized.count_rows() == 4

    def test_prepare_data_row_limit_without_txids_feeds_train_dataloader(
        self, source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Latest split subsets hydrate and feed the normal training data flow.

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

        for split in ("train", "val", "test"):
            source = lance.dataset(str(source_root / f"{split}.lance"))
            materialized = lance.dataset(str(module.dataset_root / f"{split}.lance"))
            expected_params = source.scanner(columns=["param_array"], limit=4).to_table()
            actual_params = materialized.scanner(columns=["param_array"]).to_table()
            assert actual_params.equals(expected_params)
        assert batch["params"].shape == (2, NUM_PARAMS)
        assert batch["mel"] is not None

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
        assert batch["mel"] is not None

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

        test_split = lance.dataset(str(module.dataset_root / "test.lance"))
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

        train_split = lance.dataset(str(module.dataset_root / "train.lance"))
        assert train_split.schema.names == ["param_array", "music2latent"]
