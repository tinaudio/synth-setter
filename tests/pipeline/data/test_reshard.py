"""Behavioral tests for `synth_setter.pipeline.data.reshard`.

The reshard CLI consumes ``DatasetSpec.train_val_test_sizes`` from
``<dataset_root>/input_spec.json`` as the authoritative source of per-split
sample counts. Shard counts are derived as
``size // render.samples_per_shard``; the legacy ``--train-samples`` /
``--val-samples`` / ``--test-samples`` defaults are gone.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from synth_setter.pipeline.constants import INPUT_SPEC_FILENAME
from synth_setter.pipeline.data import reshard as _reshard_module
from synth_setter.pipeline.schemas.spec import DatasetSpec

FIXED_NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)


def _render_kwargs(samples_per_shard: int) -> dict[str, Any]:
    """Return RenderConfig kwargs with the given ``samples_per_shard``.

    :param samples_per_shard: Sample count per shard, exercised by reshard's
        spec-vs-flag agreement check.
    :returns: Dict suitable as ``render`` input for ``DatasetSpec``.
    :rtype: dict[str, Any]
    """
    return {
        "plugin_path": "/fake/Plugin.vst3",
        "preset_path": "presets/surge-base.vstpreset",
        "param_spec_name": "surge_simple",
        "renderer_version": "1.3.4",
        "sample_rate": 16000,
        "channels": 2,
        "velocity": 100,
        "signal_duration_seconds": 4.0,
        "min_loudness": -55.0,
        "samples_per_render_batch": 32,
        "samples_per_shard": samples_per_shard,
    }


def _spec_kwargs(
    train: int,
    val: int,
    test: int,
    *,
    samples_per_shard: int,
) -> dict[str, Any]:
    """Return DatasetSpec kwargs that produce the requested (train, val, test) split.

    :param train: Sample count for the train split.
    :param val: Sample count for the val split.
    :param test: Sample count for the test split.
    :param samples_per_shard: Shard granularity; each split must be a multiple.
    :returns: Dict suitable as ``**kwargs`` for ``DatasetSpec``.
    :rtype: dict[str, Any]
    """
    return {
        "task_name": "reshard-test",
        "output_format": "hdf5",
        "train_val_test_sizes": [train, val, test],
        "base_seed": 42,
        "r2_bucket": "intermediate-data",
        "render": _render_kwargs(samples_per_shard),
    }


@pytest.fixture()
def patch_runtime_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub git/timestamp factories so DatasetSpec construction is deterministic.

    :param monkeypatch: Pytest monkeypatching fixture used to override the
        factory functions on ``synth_setter.pipeline.schemas.spec``.
    """
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._get_git_sha", lambda: "abc123def456")
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._is_repo_dirty", lambda: False)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._utc_now", lambda: FIXED_NOW)


def _write_shard(path: Path, shard_size: int) -> None:
    """Write a minimal HDF5 shard with the three datasets reshard expects.

    :param path: Destination filesystem path for the shard.
    :param shard_size: Number of samples (rows) the shard should advertise.
    """
    with h5py.File(path, "w") as f:
        f.create_dataset("audio", shape=(shard_size, 2, 64), dtype=np.float32)
        f.create_dataset("mel_spec", shape=(shard_size, 2, 8, 8), dtype=np.float32)
        f.create_dataset("param_array", shape=(shard_size, 12), dtype=np.float32)


def _materialize_dataset(
    dataset_root: Path,
    spec: DatasetSpec,
) -> None:
    """Write ``input_spec.json`` and all ``shard-NNNNNN.h5`` files under ``dataset_root``.

    :param dataset_root: Directory that will hold the spec JSON and shards.
    :param spec: Spec that describes the layout to materialize.
    """
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / INPUT_SPEC_FILENAME).write_text(spec.model_dump_json())
    shard_size = spec.render.samples_per_shard
    for shard in spec.shards:
        _write_shard(dataset_root / shard.filename, shard_size)


def _audio_rows(split_h5_path: Path) -> int:
    """Return the number of rows in the ``audio`` dataset at ``split_h5_path``.

    :param split_h5_path: Path to a ``{split}.h5`` virtual-dataset file.
    :returns: Leading-axis length of the ``audio`` dataset.
    :rtype: int
    """
    with h5py.File(split_h5_path, "r") as f:
        dataset = cast(h5py.Dataset, f["audio"])
        return int(dataset.shape[0])


@pytest.fixture()
def runner() -> CliRunner:
    """Return a fresh ``click.testing.CliRunner`` for each test.

    :returns: A new ``CliRunner`` instance.
    :rtype: CliRunner
    """
    return CliRunner()


class TestResharSpecDriven:
    """Reshard splits the local shard list using spec-derived counts."""

    def test_uses_spec_train_val_test_sizes(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """Three splits are sized by ``train_val_test_sizes // samples_per_shard``.

        :param tmp_path: Pytest tmp_path fixture for the dataset root.
        :param patch_runtime_io: Spec runtime-factory stub fixture.
        :param runner: Click test runner fixture.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)

        result = runner.invoke(_reshard_module.main, [str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert _audio_rows(tmp_path / "train.h5") == 20
        assert _audio_rows(tmp_path / "val.h5") == 10
        assert _audio_rows(tmp_path / "test.h5") == 10

    def test_assigns_shards_in_sorted_order(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """Train gets the first N shards, val the next, test the rest — in sorted order.

        :param tmp_path: Pytest tmp_path fixture for the dataset root.
        :param patch_runtime_io: Spec runtime-factory stub fixture.
        :param runner: Click test runner fixture.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)

        result = runner.invoke(_reshard_module.main, [str(tmp_path)])

        assert result.exit_code == 0, result.output
        for split, expected_total in (("train", 20), ("val", 10), ("test", 10)):
            assert _audio_rows(tmp_path / f"{split}.h5") == expected_total

    def test_zero_sized_split_is_skipped(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """A split with sample count 0 produces no output file.

        :param tmp_path: Pytest tmp_path fixture for the dataset root.
        :param patch_runtime_io: Spec runtime-factory stub fixture.
        :param runner: Click test runner fixture.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 0, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)

        result = runner.invoke(_reshard_module.main, [str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert (tmp_path / "train.h5").exists()
        assert not (tmp_path / "val.h5").exists()
        assert (tmp_path / "test.h5").exists()


class TestResharSpecPath:
    """Reshard reads the spec from ``<dataset_root>/input_spec.json`` by default."""

    def test_explicit_spec_path_overrides_default(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """``--spec`` accepts an arbitrary path outside the dataset root.

        :param tmp_path: Pytest tmp_path fixture for the dataset root.
        :param patch_runtime_io: Spec runtime-factory stub fixture.
        :param runner: Click test runner fixture.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        dataset_root = tmp_path / "data"
        _materialize_dataset(dataset_root, spec)
        (dataset_root / INPUT_SPEC_FILENAME).unlink()
        external_spec = tmp_path / "elsewhere.json"
        external_spec.write_text(spec.model_dump_json())

        result = runner.invoke(
            _reshard_module.main,
            [str(dataset_root), "--spec", str(external_spec)],
        )

        assert result.exit_code == 0, result.output
        assert (dataset_root / "train.h5").exists()

    def test_missing_spec_fails_loud(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """A dataset root without ``input_spec.json`` exits non-zero.

        :param tmp_path: Pytest tmp_path fixture for the dataset root.
        :param patch_runtime_io: Spec runtime-factory stub fixture.
        :param runner: Click test runner fixture.
        """
        tmp_path.mkdir(exist_ok=True)

        result = runner.invoke(_reshard_module.main, [str(tmp_path)])

        assert result.exit_code != 0


class TestResharShardSizeConflict:
    """``--shard-size`` must agree with ``spec.render.samples_per_shard`` or fail loudly."""

    def test_explicit_shard_size_matching_spec_is_ok(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """Passing ``--shard-size`` matching the spec value runs normally.

        :param tmp_path: Pytest tmp_path fixture for the dataset root.
        :param patch_runtime_io: Spec runtime-factory stub fixture.
        :param runner: Click test runner fixture.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)

        result = runner.invoke(
            _reshard_module.main, [str(tmp_path), "--shard-size", "10"]
        )

        assert result.exit_code == 0, result.output

    def test_explicit_shard_size_mismatching_spec_errors(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """Passing ``--shard-size`` different from the spec is rejected.

        :param tmp_path: Pytest tmp_path fixture for the dataset root.
        :param patch_runtime_io: Spec runtime-factory stub fixture.
        :param runner: Click test runner fixture.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)

        result = runner.invoke(
            _reshard_module.main, [str(tmp_path), "--shard-size", "9999"]
        )

        assert result.exit_code != 0
        assert "shard-size" in result.output.lower() or "samples_per_shard" in result.output


class TestResharLegacyFlagsRemoved:
    """The legacy ``--train-samples`` / ``--val-samples`` / ``--test-samples`` are gone."""

    @pytest.mark.parametrize("flag", ["--train-samples", "--val-samples", "--test-samples"])
    def test_legacy_flag_is_rejected(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
        flag: str,
    ) -> None:
        """Passing a removed flag fails with click's no-such-option error.

        :param tmp_path: Pytest tmp_path fixture for the dataset root.
        :param patch_runtime_io: Spec runtime-factory stub fixture.
        :param runner: Click test runner fixture.
        :param flag: Parametrized legacy flag name under test.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)

        result = runner.invoke(_reshard_module.main, [str(tmp_path), flag, "1"])

        assert result.exit_code != 0
        assert "no such option" in result.output.lower()
