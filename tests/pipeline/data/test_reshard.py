"""Behavioral tests for ``synth_setter.pipeline.data.reshard``.

The CLI reads ``DatasetSpec.train_val_test_sizes`` from
``<dataset_root>/input_spec.json`` and derives per-split shard counts as
``size // render.samples_per_shard``. Shard filenames are taken verbatim from
``spec.shards``; the on-disk shape and dtype of each shard are revalidated at
the reshard boundary before any output handle opens.
"""

from __future__ import annotations

import json
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

# DatasetSpec captures git-sha + UTC now in default-factory fields; pin them
# so spec construction is deterministic and the test fixtures stay hashable.
FIXED_NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)


def _render_kwargs(samples_per_shard: int) -> dict[str, Any]:
    """Return RenderConfig kwargs with the given ``samples_per_shard``.

    :param samples_per_shard: Per-shard row count; drives the spec-derived
        shard count via ``size // samples_per_shard``.
    :returns: Mapping suitable as ``render`` input for ``DatasetSpec``.
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
    """Return DatasetSpec kwargs producing the requested ``(train, val, test)`` split.

    :param train: Sample count for the train split; must be a multiple of ``samples_per_shard``.
    :param val: Sample count for the val split; must be a multiple of ``samples_per_shard``.
    :param test: Sample count for the test split; must be a multiple of ``samples_per_shard``.
    :param samples_per_shard: Shard granularity; passed through to ``render`` kwargs.
    :returns: Mapping suitable as ``**kwargs`` for ``DatasetSpec``.
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
        non-pure factory functions on ``synth_setter.pipeline.schemas.spec``.
    """
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._get_git_sha", lambda: "abc123def456")
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._is_repo_dirty", lambda: False)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._utc_now", lambda: FIXED_NOW)


def _write_shard(path: Path, shard_size: int) -> None:
    """Write a minimal HDF5 shard with the three datasets reshard expects.

    Shapes match :data:`_reshard_module._REQUIRED_DATASETS` and the dtypes the
    shard-contract check at :func:`reshard._check_shard_contracts` enforces.

    :param path: Destination filesystem path for the shard.
    :param shard_size: Required leading-axis length for every dataset.
    """
    with h5py.File(path, "w") as f:
        f.create_dataset("audio", shape=(shard_size, 2, 64), dtype=np.float32)
        f.create_dataset("mel_spec", shape=(shard_size, 2, 8, 8), dtype=np.float32)
        f.create_dataset("param_array", shape=(shard_size, 12), dtype=np.float32)


def _materialize_dataset(dataset_root: Path, spec: DatasetSpec) -> None:
    """Write ``input_spec.json`` and every ``shard-NNNNNN.h5`` under ``dataset_root``.

    :param dataset_root: Directory to populate; created if missing.
    :param spec: Spec whose ``shards`` define the layout to materialize.
    """
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / INPUT_SPEC_FILENAME).write_text(spec.model_dump_json())
    shard_size = spec.render.samples_per_shard
    for shard in spec.shards:
        _write_shard(dataset_root / shard.filename, shard_size)


def _audio_rows(split_h5_path: Path) -> int:
    """Return the leading-axis length of the ``audio`` dataset at ``split_h5_path``.

    :param split_h5_path: Path to a ``{split}.h5`` virtual-dataset file.
    :returns: Row count for the ``audio`` dataset.
    """
    with h5py.File(split_h5_path, "r") as f:
        dataset = cast(h5py.Dataset, f["audio"])
        return int(dataset.shape[0])


@pytest.fixture()
def runner() -> CliRunner:
    """Yield a fresh ``CliRunner`` per test so output buffers don't leak.

    :returns: A new ``click.testing.CliRunner`` instance.
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

        Also pins that every split file exposes the three expected datasets
        with ``np.float32`` dtype — guards against a regression that drops a
        dataset or flips its precision.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)

        result = runner.invoke(_reshard_module.main, [str(tmp_path)], catch_exceptions=False)

        assert result.exit_code == 0, result.output
        for split, expected_rows in [("train", 20), ("val", 10), ("test", 10)]:
            with h5py.File(tmp_path / f"{split}.h5", "r") as f:
                assert {"audio", "mel_spec", "param_array"} <= set(f.keys())
                for key in ("audio", "mel_spec", "param_array"):
                    dataset = cast(h5py.Dataset, f[key])
                    assert dataset.dtype == np.float32, f"{split}/{key}: {dataset.dtype}"
                    assert dataset.shape[0] == expected_rows

    def test_zero_sized_split_is_skipped(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """A split with sample count 0 produces no output file; siblings still get their rows.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 0, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)

        result = runner.invoke(_reshard_module.main, [str(tmp_path)], catch_exceptions=False)

        assert result.exit_code == 0, result.output
        assert _audio_rows(tmp_path / "train.h5") == 20
        assert not (tmp_path / "val.h5").exists()
        assert _audio_rows(tmp_path / "test.h5") == 10


class TestReshardSpecShardsAreAuthoritative:
    """Reshard reads exactly the filenames in ``spec.shards``, no glob."""

    def test_missing_canonical_shard_fails_loud(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """A missing ``spec.shards`` filename fails preflight with the path named in output.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)
        missing_name = spec.shards[0].filename
        (tmp_path / missing_name).unlink()
        (tmp_path / "shard-999999.h5").touch()

        result = runner.invoke(_reshard_module.main, [str(tmp_path)])

        assert result.exit_code != 0
        assert missing_name in result.output
        # Preflight must run before any output is written.
        assert not (tmp_path / "train.h5").exists()


class TestReshardSpecPath:
    """Reshard reads the spec from ``<dataset_root>/input_spec.json`` by default."""

    def test_explicit_spec_path_overrides_default(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """``--spec`` accepts an arbitrary path outside the dataset root.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
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
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert (dataset_root / "train.h5").exists()

    def test_missing_spec_fails_loud(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """A dataset root without ``input_spec.json`` exits with a user-oriented message.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        """
        tmp_path.mkdir(exist_ok=True)

        result = runner.invoke(_reshard_module.main, [str(tmp_path)], catch_exceptions=False)

        assert result.exit_code != 0
        assert "DatasetSpec not found" in result.output
        assert INPUT_SPEC_FILENAME in result.output


class TestReshardRemovedFlagsRejected:
    """Reshard rejects ``--train-samples``, ``--val-samples``, ``--test-samples``, ``--shard-
    size``.

    Regression net against re-adding any of the legacy/interim flags — the spec is the single
    source of truth and no override knob is allowed.
    """

    @pytest.mark.parametrize(
        "flag",
        # Closed-PR regression net: prevents re-introduction of any of these.
        ["--train-samples", "--val-samples", "--test-samples", "--shard-size"],
    )
    def test_removed_flag_is_rejected(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
        flag: str,
    ) -> None:
        """Click's unknown-option handling returns its canonical exit code.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        :param flag: Removed flag name under test.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)

        result = runner.invoke(_reshard_module.main, [str(tmp_path), flag, "1"])

        # exit_code 2 is click's canonical UsageError code, more behavioral than
        # asserting on click's English-language error string.
        assert result.exit_code == 2


class TestReshardSplitDivisibility:
    """Non-divisible ``train_val_test_sizes`` surfaces as a clean ClickException."""

    def test_stale_json_with_non_divisible_size_is_rejected(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """A tampered on-disk spec is rejected by the load-path error wrapper.

        ``DatasetSpec._split_sizes_must_be_multiples_of_samples_per_shard``
        raises ``ValidationError`` on parse; the CLI converts that to a
        ``click.ClickException`` so operators see a clean error message
        instead of a raw pydantic traceback.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)
        spec_path = tmp_path / INPUT_SPEC_FILENAME
        tampered = json.loads(spec_path.read_text())
        tampered["train_val_test_sizes"] = [25, 10, 10]
        spec_path.write_text(json.dumps(tampered))

        result = runner.invoke(_reshard_module.main, [str(tmp_path)])

        assert result.exit_code != 0
        assert "is invalid" in result.output
        assert "samples_per_shard" in result.output


class TestReshardOutputFormatGuard:
    """Reshard only supports ``output_format='hdf5'``; ``wds`` must be rejected early."""

    def test_wds_output_format_rejected_with_clickexception(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """A real WDS-format spec exits non-zero before any shard open attempt.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        """
        spec_kwargs = _spec_kwargs(20, 10, 10, samples_per_shard=10)
        spec_kwargs["output_format"] = "wds"
        spec = DatasetSpec(**spec_kwargs)
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / INPUT_SPEC_FILENAME).write_text(spec.model_dump_json())

        result = runner.invoke(_reshard_module.main, [str(tmp_path)])

        assert result.exit_code != 0
        assert "output_format" in result.output
        assert "hdf5" in result.output
        # Guard must fire before any shard is opened.
        assert not (tmp_path / "train.h5").exists()


class TestReshardR2SpecUri:
    """``--spec r2://...`` is loaded via ``load_spec_from_uri`` (no real R2 I/O)."""

    def test_r2_spec_uri_is_honored(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``--spec r2://...`` produces the same split files as a local spec.

        The local fallback is deleted before invocation, so a success here
        proves the URI argument was honored end-to-end (state-based check).

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        :param monkeypatch: Used to intercept the ``load_spec_from_uri`` import binding.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)
        (tmp_path / INPUT_SPEC_FILENAME).unlink()
        monkeypatch.setattr(_reshard_module, "load_spec_from_uri", lambda _uri: spec)

        result = runner.invoke(
            _reshard_module.main,
            [str(tmp_path), "--spec", "r2://intermediate-data/data/foo/input_spec.json"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert (tmp_path / "train.h5").exists()


class TestReshardShardContractValidation:
    """Per-shard structural validation catches drift before any output write."""

    def test_missing_required_dataset_fails_loud(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """A shard missing one of ``audio``/``mel_spec``/``param_array`` is rejected.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)
        # Rewrite shard 0 without ``mel_spec`` to simulate a corrupted upload.
        with h5py.File(tmp_path / spec.shards[0].filename, "w") as f:
            f.create_dataset("audio", shape=(10, 2, 64), dtype=np.float32)
            f.create_dataset("param_array", shape=(10, 12), dtype=np.float32)

        result = runner.invoke(_reshard_module.main, [str(tmp_path)])

        assert result.exit_code != 0
        assert "mel_spec" in result.output
        assert not (tmp_path / "train.h5").exists()

    def test_wrong_dtype_fails_loud(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """A shard whose dataset is not ``np.float32`` is rejected.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)
        with h5py.File(tmp_path / spec.shards[0].filename, "w") as f:
            f.create_dataset("audio", shape=(10, 2, 64), dtype=np.float64)
            f.create_dataset("mel_spec", shape=(10, 2, 8, 8), dtype=np.float32)
            f.create_dataset("param_array", shape=(10, 12), dtype=np.float32)

        result = runner.invoke(_reshard_module.main, [str(tmp_path)])

        assert result.exit_code != 0
        assert "float32" in result.output

    def test_wrong_row_count_fails_loud(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """A shard whose leading axis disagrees with ``samples_per_shard`` is rejected.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)
        with h5py.File(tmp_path / spec.shards[0].filename, "w") as f:
            # 7 rows instead of 10.
            f.create_dataset("audio", shape=(7, 2, 64), dtype=np.float32)
            f.create_dataset("mel_spec", shape=(7, 2, 8, 8), dtype=np.float32)
            f.create_dataset("param_array", shape=(7, 12), dtype=np.float32)

        result = runner.invoke(_reshard_module.main, [str(tmp_path)])

        assert result.exit_code != 0
        assert "samples_per_shard=10" in result.output

    def test_inconsistent_trailing_shape_fails_loud(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """Shards with different trailing shapes are rejected before VDS construction.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)
        # Rewrite shard 1 with a different audio tail to simulate config drift.
        with h5py.File(tmp_path / spec.shards[1].filename, "w") as f:
            f.create_dataset("audio", shape=(10, 2, 32), dtype=np.float32)
            f.create_dataset("mel_spec", shape=(10, 2, 8, 8), dtype=np.float32)
            f.create_dataset("param_array", shape=(10, 12), dtype=np.float32)

        result = runner.invoke(_reshard_module.main, [str(tmp_path)])

        assert result.exit_code != 0
        assert "trailing shape" in result.output


class TestReshardAtomicWrite:
    """Mid-loop failures don't leave stale ``<split>.h5`` files next to the inputs.

    Reshard stages each split under ``<dataset_root>/.tmp-<split>.h5`` and
    atomically renames into place after ``create_virtual_dataset`` succeeds.
    """

    def test_partial_split_files_are_not_left_behind_on_failure(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """A shard-contract failure leaves no ``train.h5`` and no ``.tmp-*.h5`` artifact.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)
        # Drift one shard so the contract check fails after the preflight existence
        # sweep but before any output handle opens.
        with h5py.File(tmp_path / spec.shards[2].filename, "w") as f:
            f.create_dataset("audio", shape=(7, 2, 64), dtype=np.float32)
            f.create_dataset("mel_spec", shape=(7, 2, 8, 8), dtype=np.float32)
            f.create_dataset("param_array", shape=(7, 12), dtype=np.float32)

        result = runner.invoke(_reshard_module.main, [str(tmp_path)])

        assert result.exit_code != 0
        for split in ("train", "val", "test"):
            assert not (tmp_path / f"{split}.h5").exists()
            assert not (tmp_path / f".tmp-{split}.h5").exists()


class TestReshardVirtualDatasetIdentity:
    """Virtual-dataset rows surface the right source-shard bytes, in spec order."""

    def test_split_files_concatenate_source_shards_in_spec_order(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """Each split's row ``i*sps:(i+1)*sps`` equals the i-th source shard's bytes.

        Pins identity for all three datasets (``audio``, ``mel_spec``,
        ``param_array``) so a swap-bug in the per-dataset loop can't slip
        past. Also confirms outputs are real ``VirtualDataset`` files, not
        materialized copies.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        """
        sps = 3
        spec = DatasetSpec(**_spec_kwargs(sps * 2, sps, sps, samples_per_shard=sps))
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / INPUT_SPEC_FILENAME).write_text(spec.model_dump_json())
        # Each shard filled with float(i+1) so virtual-dataset slices stay traceable.
        for i, shard in enumerate(spec.shards):
            fill = float(i + 1)
            with h5py.File(tmp_path / shard.filename, "w") as f:
                f.create_dataset("audio", data=np.full((sps, 2, 64), fill, dtype=np.float32))
                f.create_dataset("mel_spec", data=np.full((sps, 2, 8, 8), fill, dtype=np.float32))
                f.create_dataset("param_array", data=np.full((sps, 12), fill, dtype=np.float32))

        result = runner.invoke(_reshard_module.main, [str(tmp_path)], catch_exceptions=False)
        assert result.exit_code == 0, result.output

        with h5py.File(tmp_path / "train.h5", "r") as f:
            audio = cast(h5py.Dataset, f["audio"])
            mel = cast(h5py.Dataset, f["mel_spec"])
            param = cast(h5py.Dataset, f["param_array"])
            assert audio.is_virtual
            assert mel.is_virtual
            assert param.is_virtual
            np.testing.assert_array_equal(
                audio[:sps], np.full((sps, 2, 64), 1.0, dtype=np.float32)
            )
            np.testing.assert_array_equal(
                audio[sps:], np.full((sps, 2, 64), 2.0, dtype=np.float32)
            )
            np.testing.assert_array_equal(
                mel[:sps], np.full((sps, 2, 8, 8), 1.0, dtype=np.float32)
            )
            np.testing.assert_array_equal(
                mel[sps:], np.full((sps, 2, 8, 8), 2.0, dtype=np.float32)
            )
            np.testing.assert_array_equal(param[:sps], np.full((sps, 12), 1.0, dtype=np.float32))
            np.testing.assert_array_equal(param[sps:], np.full((sps, 12), 2.0, dtype=np.float32))

        with h5py.File(tmp_path / "val.h5", "r") as f:
            np.testing.assert_array_equal(
                cast(h5py.Dataset, f["audio"])[:],
                np.full((sps, 2, 64), 3.0, dtype=np.float32),
            )
            np.testing.assert_array_equal(
                cast(h5py.Dataset, f["mel_spec"])[:],
                np.full((sps, 2, 8, 8), 3.0, dtype=np.float32),
            )
        with h5py.File(tmp_path / "test.h5", "r") as f:
            np.testing.assert_array_equal(
                cast(h5py.Dataset, f["audio"])[:],
                np.full((sps, 2, 64), 4.0, dtype=np.float32),
            )
