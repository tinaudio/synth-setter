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
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

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


class TestReshardSpecDriven:
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


class TestReshardSpecShardsAreAuthoritative:
    """Reshard reads exactly the filenames in ``spec.shards``, no glob."""

    def test_missing_canonical_shard_fails_loud(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """A spec.shards filename missing on disk exits non-zero, even with a stale neighbor.

        :param tmp_path: Pytest tmp_path fixture for the dataset root.
        :param patch_runtime_io: Spec runtime-factory stub fixture.
        :param runner: Click test runner fixture.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)
        (tmp_path / spec.shards[0].filename).unlink()
        (tmp_path / "shard-999999.h5").touch()

        result = runner.invoke(_reshard_module.main, [str(tmp_path)])

        assert result.exit_code != 0


class TestReshardSpecPath:
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


class TestReshardRemovedFlagsRejected:
    """Flags that used to exist on reshard (``--train-samples`` / ``--val-samples`` / ``--test-
    samples`` from the original CLI; ``--shard-size`` from the interim parent #1092) must be
    rejected — the spec is now the single source of truth."""

    @pytest.mark.parametrize(
        "flag",
        ["--train-samples", "--val-samples", "--test-samples", "--shard-size"],
    )
    def test_removed_flag_is_rejected(
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
        :param flag: Parametrized removed flag name under test.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)

        result = runner.invoke(_reshard_module.main, [str(tmp_path), flag, "1"])

        assert result.exit_code != 0
        assert "no such option" in result.output.lower()


class TestReshardSplitDivisibility:
    """``train_val_test_sizes`` must be perfectly divisible by ``samples_per_shard``."""

    def test_non_divisible_split_size_raises_at_runtime(
        self,
        tmp_path: Path,
        runner: CliRunner,
    ) -> None:
        """A stale spec whose split size has a non-zero remainder is rejected loudly.

        Normally ``DatasetSpec._split_sizes_must_be_multiples_of_samples_per_shard``
        catches this at parse time; this test bypasses model validation (via
        ``SimpleNamespace``) to exercise the defensive guard inside ``main()``,
        protecting against a stale R2 spec that predates the model validator.

        :param tmp_path: Pytest tmp_path fixture for the dataset root.
        :param runner: Click test runner fixture.
        """
        tmp_path.mkdir(exist_ok=True)
        bad_spec = SimpleNamespace(
            render=SimpleNamespace(samples_per_shard=10),
            train_val_test_sizes=(15, 10, 10),  # 15 % 10 != 0
            shards=(),
        )

        with mock.patch.object(_reshard_module, "load_spec_from_uri", return_value=bad_spec):
            result = runner.invoke(_reshard_module.main, [str(tmp_path)])

        assert result.exit_code != 0
        assert "divisible" in result.output.lower() or "samples_per_shard" in result.output


class TestReshardR2SpecUri:
    """``--spec r2://...`` is loaded via ``load_spec_from_uri`` (no real R2 I/O)."""

    def test_r2_spec_uri_is_loaded_via_load_spec_from_uri(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """``--spec r2://...`` delegates to ``load_spec_from_uri`` (mocked).

        :param tmp_path: Pytest tmp_path fixture for the dataset root.
        :param patch_runtime_io: Spec runtime-factory stub fixture.
        :param runner: Click test runner fixture.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)
        # Remove the local fallback so the test fails if the URI isn't honored.
        (tmp_path / INPUT_SPEC_FILENAME).unlink()
        spec_uri = "r2://intermediate-data/data/foo/input_spec.json"

        with mock.patch.object(
            _reshard_module, "load_spec_from_uri", return_value=spec
        ) as loader:
            result = runner.invoke(
                _reshard_module.main,
                [str(tmp_path), "--spec", spec_uri],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        loader.assert_called_once_with(spec_uri)
        assert (tmp_path / "train.h5").exists()


class TestReshardVirtualDatasetIdentity:
    """Virtual-dataset rows actually surface the right source-shard bytes, in spec order."""

    def test_split_files_concatenate_source_shards_in_spec_order(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """Each split's audio[i*sps:(i+1)*sps] equals the i-th source shard's bytes.

        :param tmp_path: Pytest tmp_path fixture for the dataset root.
        :param patch_runtime_io: Spec runtime-factory stub fixture.
        :param runner: Click test runner fixture.
        """
        sps = 3
        spec = DatasetSpec(**_spec_kwargs(sps * 2, sps * 1, sps * 1, samples_per_shard=sps))
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / INPUT_SPEC_FILENAME).write_text(spec.model_dump_json())
        # Write shard `i` filled with `float(i + 1)` so each source's bytes are traceable.
        for i, shard in enumerate(spec.shards):
            fill = float(i + 1)
            with h5py.File(tmp_path / shard.filename, "w") as f:
                f.create_dataset(
                    "audio", data=np.full((sps, 2, 64), fill, dtype=np.float32)
                )
                f.create_dataset(
                    "mel_spec", data=np.full((sps, 2, 8, 8), fill, dtype=np.float32)
                )
                f.create_dataset(
                    "param_array", data=np.full((sps, 12), fill, dtype=np.float32)
                )

        result = runner.invoke(_reshard_module.main, [str(tmp_path)], catch_exceptions=False)
        assert result.exit_code == 0, result.output

        with h5py.File(tmp_path / "train.h5", "r") as f:
            audio = cast(h5py.Dataset, f["audio"])[:]
            params = cast(h5py.Dataset, f["param_array"])[:]
        np.testing.assert_array_equal(audio[:sps], np.full((sps, 2, 64), 1.0, dtype=np.float32))
        np.testing.assert_array_equal(audio[sps:], np.full((sps, 2, 64), 2.0, dtype=np.float32))
        np.testing.assert_array_equal(params[:sps], np.full((sps, 12), 1.0, dtype=np.float32))
        np.testing.assert_array_equal(params[sps:], np.full((sps, 12), 2.0, dtype=np.float32))

        with h5py.File(tmp_path / "val.h5", "r") as f:
            np.testing.assert_array_equal(
                cast(h5py.Dataset, f["audio"])[:],
                np.full((sps, 2, 64), 3.0, dtype=np.float32),
            )
        with h5py.File(tmp_path / "test.h5", "r") as f:
            np.testing.assert_array_equal(
                cast(h5py.Dataset, f["audio"])[:],
                np.full((sps, 2, 64), 4.0, dtype=np.float32),
            )
