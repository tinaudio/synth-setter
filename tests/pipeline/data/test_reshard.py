"""Behavioral tests for ``synth_setter.pipeline.data.reshard``.

The CLI reads ``DatasetSpec.train_val_test_sizes`` from
``<dataset_root>/input_spec.json`` and derives per-split shard counts as
``size // render.samples_per_shard``. Shard filenames are taken verbatim from
``spec.shards``; the on-disk shape and dtype of each shard are revalidated at
the reshard boundary before any output handle opens. All split outputs are
staged under ``.tmp-<split>.h5`` and renamed only after every split's staging
write succeeds (across-splits atomicity).
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
from synth_setter.pipeline.data.reshard import _REQUIRED_DATASETS
from synth_setter.pipeline.schemas.spec import DatasetSpec

# Pinned timestamp for ``patch_runtime_io``'s ``_utc_now`` monkeypatch.
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

    Shapes and dtypes match the contract that
    :func:`reshard._check_shard_contracts` enforces.

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


def _assert_no_outputs(dataset_root: Path) -> None:
    """Assert no ``{train,val,test}.h5`` and no ``.tmp-*.h5`` remain in ``dataset_root``.

    :param dataset_root: Directory under inspection.
    """
    for split in ("train", "val", "test"):
        assert not (dataset_root / f"{split}.h5").exists(), f"{split}.h5 lingered"
        assert not (dataset_root / f".tmp-{split}.h5").exists(), f".tmp-{split}.h5 lingered"


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
        with ``np.float32`` dtype.

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
        # Happy path leaves no staging artifacts behind.
        for split in ("train", "val", "test"):
            assert not (tmp_path / f".tmp-{split}.h5").exists()

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

        result = runner.invoke(_reshard_module.main, [str(tmp_path)], catch_exceptions=False)

        assert result.exit_code != 0
        assert missing_name in result.output
        _assert_no_outputs(tmp_path)


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

        assert result.exit_code == 2  # click's UsageError exit code


class TestReshardSplitDivisibility:
    """Non-divisible ``train_val_test_sizes`` surfaces as a clean ClickException."""

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_stale_json_with_non_divisible_size_is_rejected(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
        index: int,
    ) -> None:
        """A tampered split size at any index is rejected by the load-path wrapper.

        ``DatasetSpec._split_sizes_must_be_multiples_of_samples_per_shard``
        raises ``ValidationError`` on parse; the CLI converts that to a
        ``click.ClickException`` so operators see a clean error message
        instead of a raw pydantic traceback.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        :param index: ``train_val_test_sizes`` position to corrupt.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)
        spec_path = tmp_path / INPUT_SPEC_FILENAME
        tampered = json.loads(spec_path.read_text())
        tampered["train_val_test_sizes"][index] = 25  # 25 % 10 != 0
        spec_path.write_text(json.dumps(tampered))

        result = runner.invoke(_reshard_module.main, [str(tmp_path)], catch_exceptions=False)

        assert result.exit_code != 0
        assert "is invalid" in result.output
        assert "not a multiple" in result.output


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

        result = runner.invoke(_reshard_module.main, [str(tmp_path)], catch_exceptions=False)

        assert result.exit_code != 0
        assert "output_format" in result.output
        assert "hdf5" in result.output
        _assert_no_outputs(tmp_path)


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
        proves the URI argument was honored end-to-end.

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


class TestReshardShardIdOrdering:
    """Tampered specs with mis-ordered ``shards[i].shard_id`` are rejected."""

    def test_out_of_order_shard_id_fails_loud(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``shards[0].shard_id != 0`` exits non-zero before any output write.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        :param monkeypatch: Used to swap ``spec.shards`` for a mis-ordered tuple.
        """
        from synth_setter.pipeline.schemas.spec import ShardSpec

        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)
        reversed_shards = tuple(
            ShardSpec(
                shard_id=len(spec.shards) - 1 - i,
                filename=shard.filename,
                seed=shard.seed,
            )
            for i, shard in enumerate(spec.shards)
        )

        def fake_loader(_uri: str) -> DatasetSpec:
            # ``shards`` is a computed property; swap it on a real spec instance.
            object.__setattr__(spec, "__dict__", {**spec.__dict__, "shards": reversed_shards})
            return spec

        monkeypatch.setattr(_reshard_module, "load_spec_from_uri", fake_loader)

        result = runner.invoke(_reshard_module.main, [str(tmp_path)], catch_exceptions=False)

        assert result.exit_code != 0
        assert "shard_id" in result.output
        _assert_no_outputs(tmp_path)


class TestReshardShardContractValidation:
    """Per-shard structural validation catches drift before any output write."""

    @pytest.mark.parametrize("missing_key", _REQUIRED_DATASETS)
    def test_missing_required_dataset_fails_loud(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
        missing_key: str,
    ) -> None:
        """Any of the three required datasets missing from a shard is rejected.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        :param missing_key: Dataset key the test omits from the corrupted shard.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)
        corrupted_path = tmp_path / spec.shards[0].filename
        shape_map = {
            "audio": (10, 2, 64),
            "mel_spec": (10, 2, 8, 8),
            "param_array": (10, 12),
        }
        with h5py.File(corrupted_path, "w") as f:
            for key, shape in shape_map.items():
                if key != missing_key:
                    f.create_dataset(key, shape=shape, dtype=np.float32)

        result = runner.invoke(_reshard_module.main, [str(tmp_path)], catch_exceptions=False)

        assert result.exit_code != 0
        assert missing_key in result.output
        assert str(corrupted_path) in result.output
        _assert_no_outputs(tmp_path)

    def test_group_at_required_key_fails_loud(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """A Group (not a Dataset) at a required key is rejected with the key name.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)
        corrupted_path = tmp_path / spec.shards[0].filename
        with h5py.File(corrupted_path, "w") as f:
            f.create_group("audio")  # wrong h5py type at the required key
            f.create_dataset("mel_spec", shape=(10, 2, 8, 8), dtype=np.float32)
            f.create_dataset("param_array", shape=(10, 12), dtype=np.float32)

        result = runner.invoke(_reshard_module.main, [str(tmp_path)], catch_exceptions=False)

        assert result.exit_code != 0
        assert "audio" in result.output
        assert "Group" in result.output
        _assert_no_outputs(tmp_path)

    def test_wrong_dtype_fails_loud(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """A shard whose dataset is not ``np.float32`` is rejected with observed dtype.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)
        corrupted_path = tmp_path / spec.shards[0].filename
        with h5py.File(corrupted_path, "w") as f:
            f.create_dataset("audio", shape=(10, 2, 64), dtype=np.float64)
            f.create_dataset("mel_spec", shape=(10, 2, 8, 8), dtype=np.float32)
            f.create_dataset("param_array", shape=(10, 12), dtype=np.float32)

        result = runner.invoke(_reshard_module.main, [str(tmp_path)], catch_exceptions=False)

        assert result.exit_code != 0
        assert "audio" in result.output
        assert "float64" in result.output  # observed
        assert "float32" in result.output  # expected
        _assert_no_outputs(tmp_path)

    def test_wrong_row_count_fails_loud(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """Shards with a wrong leading-axis length are rejected with both observed and expected.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)
        corrupted_path = tmp_path / spec.shards[0].filename
        with h5py.File(corrupted_path, "w") as f:
            # 7 rows instead of 10.
            f.create_dataset("audio", shape=(7, 2, 64), dtype=np.float32)
            f.create_dataset("mel_spec", shape=(7, 2, 8, 8), dtype=np.float32)
            f.create_dataset("param_array", shape=(7, 12), dtype=np.float32)

        result = runner.invoke(_reshard_module.main, [str(tmp_path)], catch_exceptions=False)

        assert result.exit_code != 0
        assert "7 rows" in result.output
        assert "samples_per_shard" in result.output
        _assert_no_outputs(tmp_path)

    def test_inconsistent_trailing_shape_fails_loud(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """Shards with different trailing shapes name the offending shard and key.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)
        corrupted_path = tmp_path / spec.shards[1].filename
        with h5py.File(corrupted_path, "w") as f:
            f.create_dataset("audio", shape=(10, 2, 32), dtype=np.float32)
            f.create_dataset("mel_spec", shape=(10, 2, 8, 8), dtype=np.float32)
            f.create_dataset("param_array", shape=(10, 12), dtype=np.float32)

        result = runner.invoke(_reshard_module.main, [str(tmp_path)], catch_exceptions=False)

        assert result.exit_code != 0
        assert str(corrupted_path) in result.output
        assert "audio" in result.output
        _assert_no_outputs(tmp_path)


class TestReshardAtomicWrite:
    """A failure inside ``_write_split`` leaves zero artifacts under ``dataset_root``.

    Reshard stages every split as ``.tmp-<split>.h5`` and renames into place
    only after every staging write succeeds (across-splits atomicity).
    """

    def test_create_virtual_dataset_failure_cleans_all_staging_files(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A mid-write h5py failure leaves no ``.h5`` or ``.tmp-*.h5`` next to the inputs.

        Forces ``h5py.File.create_virtual_dataset`` to raise once the ``val``
        split is being staged. ``train`` will already have finished its
        staging write; the rollback path must unlink ``.tmp-train.h5`` and
        ``.tmp-val.h5`` (and never rename either) so no operator can mistake
        a partial run for a complete one.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        :param monkeypatch: Used to intercept ``h5py.File.create_virtual_dataset``.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)
        original = h5py.File.create_virtual_dataset

        def failing(self: h5py.File, name: str, layout: h5py.VirtualLayout) -> h5py.Dataset:
            if Path(self.filename).name == ".tmp-val.h5":
                raise RuntimeError("simulated h5py failure during val staging")
            return original(self, name, layout)

        monkeypatch.setattr(h5py.File, "create_virtual_dataset", failing)

        result = runner.invoke(_reshard_module.main, [str(tmp_path)])

        assert result.exit_code != 0
        assert "simulated h5py failure" in str(result.exception) or result.exception is not None
        # Across-splits atomicity: train staged successfully but must NOT be renamed
        # because val failed during staging.
        _assert_no_outputs(tmp_path)

    def test_preflight_failure_creates_no_staging_files(
        self,
        tmp_path: Path,
        patch_runtime_io: None,
        runner: CliRunner,
    ) -> None:
        """A failed preflight never reaches the staging phase, so no ``.tmp-*.h5`` is opened.

        :param tmp_path: Per-test dataset root.
        :param patch_runtime_io: Deterministic spec runtime stubs.
        :param runner: Click test runner.
        """
        spec = DatasetSpec(**_spec_kwargs(20, 10, 10, samples_per_shard=10))
        _materialize_dataset(tmp_path, spec)
        with h5py.File(tmp_path / spec.shards[2].filename, "w") as f:
            f.create_dataset("audio", shape=(7, 2, 64), dtype=np.float32)
            f.create_dataset("mel_spec", shape=(7, 2, 8, 8), dtype=np.float32)
            f.create_dataset("param_array", shape=(7, 12), dtype=np.float32)

        result = runner.invoke(_reshard_module.main, [str(tmp_path)], catch_exceptions=False)

        assert result.exit_code != 0
        _assert_no_outputs(tmp_path)


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
