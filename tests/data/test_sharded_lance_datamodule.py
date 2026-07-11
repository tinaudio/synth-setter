"""Behavioral tests for the sharded Lance reading layer in
:mod:`synth_setter.data.lance_datamodule`.

Covers:

* :class:`ShardedLanceFile` — one h5py-``File``-like read surface over an
  ordered list of ``shard-*.lance`` dataset directories (global row indexing,
  boundary-crossing reads, close semantics, per-shard row-count validation).
* :class:`ShardedLanceVSTDataset` — the sharded sibling of ``LanceVSTDataset``.
* :class:`ShardedLanceVSTDataModule` — Lightning wiring over a dataset run
  directory (``shard-*.lance`` + ``input_spec.json`` + ``stats.npz``), the
  layout the pipeline's generate + stats-only finalize leave on R2 and an
  ``rclone mount`` exposes as a local directory.

Fixtures are tiny (a few rows per shard, ~10-element tensor axes) — the goal
is contract coverage on shard-boundary arithmetic, shapes, and wiring,
mirroring ``tests/data/test_lance_datamodule.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from synth_setter.data.lance_datamodule import (
    ShardedLanceFile,
    ShardedLanceVSTDataModule,
    ShardedLanceVSTDataset,
)
from synth_setter.pipeline.constants import INPUT_SPEC_FILENAME
from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.pipeline.spec_io import write_spec_to_path
from tests.helpers.finalize_shards import build_lance_smoke_spec
from tests.helpers.lance_fixtures import write_lance_shard

_ROWS_PER_SHARD = 4
_NUM_SHARDS = 3
_NUM_ROWS = _ROWS_PER_SHARD * _NUM_SHARDS
_MEL_SHAPE = (2, 3, 5)
_NUM_PARAMS = 3


def _global_columns(num_rows: int = _NUM_ROWS) -> dict[str, np.ndarray]:
    """Build deterministic whole-dataset columns keyed by global row index.

    Every element of row ``r`` encodes ``r`` (offset per column), so any read
    can be checked against a slice of these arrays by exact equality.

    :param num_rows: Total rows across all shards.
    :return: Mapping of column name to ``(num_rows, ...)`` float32 array.
    """
    mel = np.arange(num_rows * np.prod(_MEL_SHAPE), dtype=np.float32).reshape(
        num_rows, *_MEL_SHAPE
    )
    params = np.arange(num_rows * _NUM_PARAMS, dtype=np.float32).reshape(num_rows, _NUM_PARAMS)
    return {"mel_spec": mel, "param_array": params}


def _write_sharded_dataset(
    root: Path,
    columns: dict[str, np.ndarray] | None = None,
    rows_per_shard: int = _ROWS_PER_SHARD,
) -> list[Path]:
    """Split ``columns`` into equal shards and write them as ``shard-*.lance`` dirs.

    :param root: Directory receiving the shard dataset directories.
    :param columns: Whole-dataset columns; defaults to :func:`_global_columns`.
    :param rows_per_shard: Rows per shard; must divide the row count evenly.
    :return: Ordered shard paths.
    """
    columns = columns if columns is not None else _global_columns()
    num_rows = next(iter(columns.values())).shape[0]
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for shard_index in range(num_rows // rows_per_shard):
        lo = shard_index * rows_per_shard
        shard_path = root / f"shard-{shard_index:06d}.lance"
        write_lance_shard(
            shard_path, {name: data[lo : lo + rows_per_shard] for name, data in columns.items()}
        )
        paths.append(shard_path)
    return paths


@pytest.fixture
def sharded_file(tmp_path: Path) -> ShardedLanceFile:
    """Open a 3-shard, 12-row ``ShardedLanceFile`` over freshly written shards.

    :param tmp_path: Per-test tmpdir.
    :return: Open sharded read handle.
    """
    return ShardedLanceFile(_write_sharded_dataset(tmp_path / "run"), _ROWS_PER_SHARD)


class TestShardedLanceFile:
    """Global-index read surface over an ordered list of Lance shard directories."""

    def test_column_shape_spans_all_shards(self, sharded_file: ShardedLanceFile) -> None:
        """``file[name].shape`` is ``(total_rows, *tensor_shape)`` across every shard.

        :param sharded_file: Fixture-provided 3-shard handle.
        """
        assert sharded_file["mel_spec"].shape == (_NUM_ROWS, *_MEL_SHAPE)
        assert sharded_file["param_array"].shape == (_NUM_ROWS, _NUM_PARAMS)

    def test_slice_within_single_shard_reads_exact_rows(
        self, sharded_file: ShardedLanceFile
    ) -> None:
        """A slice inside one shard returns exactly those global rows.

        :param sharded_file: Fixture-provided 3-shard handle.
        """
        expected = _global_columns()["param_array"][1:3]
        np.testing.assert_array_equal(sharded_file["param_array"][1:3], expected)

    def test_slice_crossing_shard_boundary_concatenates_in_order(
        self, sharded_file: ShardedLanceFile
    ) -> None:
        """A contiguous slice spanning two shards stitches rows in global order.

        ``ShiftedBatchSampler`` shifts batch bounds by a random offset, so
        boundary-crossing ``(start, stop)`` reads are the production hot path.

        :param sharded_file: Fixture-provided 3-shard handle.
        """
        expected = _global_columns()["mel_spec"][2:7]
        np.testing.assert_array_equal(sharded_file["mel_spec"][2:7], expected)

    def test_slice_spanning_all_shards_reads_everything(
        self, sharded_file: ShardedLanceFile
    ) -> None:
        """A full-range slice touches all three shards and returns every row once.

        :param sharded_file: Fixture-provided 3-shard handle.
        """
        expected = _global_columns()["param_array"]
        np.testing.assert_array_equal(sharded_file["param_array"][:], expected)

    def test_step_slice_reads_strided_rows_across_shards(
        self, sharded_file: ShardedLanceFile
    ) -> None:
        """``file[name][a:b:s]`` gathers exactly the strided global rows.

        :param sharded_file: Fixture-provided 3-shard handle.
        """
        expected = _global_columns()["param_array"][1:12:3]
        np.testing.assert_array_equal(sharded_file["param_array"][1:12:3], expected)

    def test_unsorted_fancy_index_across_shards_preserves_requested_order(
        self, sharded_file: ShardedLanceFile
    ) -> None:
        """Unsorted indices spanning shards return rows in the requested order.

        :param sharded_file: Fixture-provided 3-shard handle.
        """
        params = _global_columns()["param_array"]
        requested = [9, 1, 6, 2]
        np.testing.assert_array_equal(sharded_file["param_array"][requested], params[requested])

    def test_numpy_int_indices_are_accepted(self, sharded_file: ShardedLanceFile) -> None:
        """Samplers yield numpy integer arrays — the column accepts them directly.

        :param sharded_file: Fixture-provided 3-shard handle.
        """
        idx = np.array([3, 4, 11], dtype=np.int64)
        np.testing.assert_array_equal(
            sharded_file["param_array"][idx], _global_columns()["param_array"][idx]
        )

    def test_reads_return_writable_arrays(self, sharded_file: ShardedLanceFile) -> None:
        """Both read paths return writable arrays safe for ``torch.from_numpy``.

        :param sharded_file: Fixture-provided 3-shard handle.
        """
        assert sharded_file["mel_spec"][1:3].flags.writeable
        assert sharded_file["mel_spec"][[0, 5, 10]].flags.writeable

    def test_negative_step_slice_raises_value_error(self, sharded_file: ShardedLanceFile) -> None:
        """A negative-step slice is rejected — the same contract h5py enforces.

        :param sharded_file: Fixture-provided 3-shard handle.
        """
        with pytest.raises(ValueError, match="step"):
            _ = sharded_file["param_array"][::-1]

    def test_out_of_range_index_raises_index_error(self, sharded_file: ShardedLanceFile) -> None:
        """Indices past the last shard fail loudly instead of reading a wrong shard.

        :param sharded_file: Fixture-provided 3-shard handle.
        """
        with pytest.raises(IndexError, match="out of range"):
            _ = sharded_file["param_array"][[0, _NUM_ROWS]]

    def test_missing_column_raises_key_error_at_lookup(
        self, sharded_file: ShardedLanceFile
    ) -> None:
        """``file[name]`` for an absent column raises ``KeyError`` like h5py.

        :param sharded_file: Fixture-provided 3-shard handle.
        """
        with pytest.raises(KeyError, match="no-such-column"):
            _ = sharded_file["no-such-column"]

    def test_close_makes_file_falsy_and_reads_raise(self, sharded_file: ShardedLanceFile) -> None:
        """``close()`` flips the handle falsy and later reads raise ``ValueError``.

        :param sharded_file: Fixture-provided 3-shard handle.
        """
        column = sharded_file["param_array"]
        assert sharded_file
        sharded_file.close()
        assert not sharded_file
        with pytest.raises(ValueError, match="closed"):
            _ = sharded_file["param_array"]
        with pytest.raises(ValueError, match="closed"):
            _ = column[0:2]

    def test_empty_shard_list_raises_value_error(self) -> None:
        """Constructing over zero shards errors at construction, not first read."""
        with pytest.raises(ValueError, match="at least one shard"):
            ShardedLanceFile([], _ROWS_PER_SHARD)

    def test_shard_with_wrong_row_count_raises_on_first_touch(self, tmp_path: Path) -> None:
        """A shard whose actual rows differ from the declared geometry fails loudly.

        Row geometry is spec-driven (workers validate it before upload); a
        mismatched shard would silently misalign every global index, so the
        lazy open verifies the count.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        paths = _write_sharded_dataset(tmp_path / "run")
        short = _global_columns(num_rows=_ROWS_PER_SHARD - 1)
        write_lance_shard(tmp_path / "run" / "shard-000003.lance", short)
        paths.append(tmp_path / "run" / "shard-000003.lance")
        sharded = ShardedLanceFile(paths, _ROWS_PER_SHARD)
        with pytest.raises(ValueError, match="expected 4"):
            _ = sharded["param_array"][[_NUM_ROWS]]


_AUDIO_SHAPE = (2, 6)
_RUN_SIZES = (8, 4, 4)
_RUN_ROWS = sum(_RUN_SIZES)


def _run_columns(num_rows: int, mel_fill: float | None = None) -> dict[str, np.ndarray]:
    """Build whole-run columns carrying every field ``VSTDataset`` reads.

    ``param_array`` encodes the global row index element-wise so split routing
    is observable: a batch read through any split must return exactly the rows
    of that split's shards.

    :param num_rows: Total rows across all shards.
    :param mel_fill: When set, fill ``mel_spec`` with this constant — used by
        the normalization test to make ``(mel - mean) / std`` predictable.
    :return: Mapping of column name to ``(num_rows, ...)`` array.
    """
    mel = np.arange(num_rows * np.prod(_MEL_SHAPE), dtype=np.float32).reshape(
        num_rows, *_MEL_SHAPE
    )
    if mel_fill is not None:
        mel = np.full_like(mel, mel_fill)
    return {
        # float16 mirrors the pipeline's on-disk audio dtype.
        "audio": np.ones((num_rows, *_AUDIO_SHAPE), dtype=np.float16),
        "mel_spec": mel,
        "param_array": np.arange(num_rows * _NUM_PARAMS, dtype=np.float32).reshape(
            num_rows, _NUM_PARAMS
        ),
    }


def _write_run_root(
    root: Path,
    train_val_test_sizes: tuple[int, int, int] = _RUN_SIZES,
    mel_fill: float | None = None,
    stats_mean: float = 0.0,
    stats_std: float = 1.0,
) -> DatasetSpec:
    """Materialize a dataset run directory as the stats-only finalize leaves it.

    Layout: ``shard-*.lance`` per spec shard + ``input_spec.json`` +
    ``stats.npz`` — what an ``rclone mount`` of the R2 run prefix exposes.

    :param root: Run directory to create.
    :param train_val_test_sizes: Split sample counts (multiples of 4 — the
        smoke spec's ``samples_per_shard``).
    :param mel_fill: Forwarded to :func:`_run_columns`.
    :param stats_mean: Scalar mean broadcast over the written ``stats.npz``.
    :param stats_std: Scalar std broadcast over the written ``stats.npz``.
    :return: The frozen spec the run directory was written from.
    """
    spec = build_lance_smoke_spec(
        task_name="sharded-dm", train_val_test_sizes=train_val_test_sizes
    )
    root.mkdir(parents=True, exist_ok=True)
    columns = _run_columns(sum(train_val_test_sizes), mel_fill=mel_fill)
    per_shard = spec.render.samples_per_shard
    for index, shard in enumerate(spec.shards):
        lo = index * per_shard
        write_lance_shard(
            root / shard.filename,
            {name: data[lo : lo + per_shard] for name, data in columns.items()},
        )
    write_spec_to_path(spec, root / INPUT_SPEC_FILENAME)
    np.savez(
        root / "stats.npz",
        mean=np.full(_MEL_SHAPE, stats_mean, dtype=np.float32),
        std=np.full(_MEL_SHAPE, stats_std, dtype=np.float32),
    )
    return spec


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    """Write the default 4-shard (train 2 / val 1 / test 1) run directory.

    :param tmp_path: Per-test tmpdir.
    :return: Path to the populated run directory.
    """
    _write_run_root(tmp_path / "run")
    return tmp_path / "run"


class TestShardedLanceVSTDataset:
    """Split-path open contract: ``<root>/<split>.lance`` resolves via the sibling spec."""

    def test_len_counts_rows_across_all_split_shards(self, run_root: Path) -> None:
        """``__len__`` floor-divides the whole split's rows (all shards) by batch_size.

        :param run_root: Fixture-provided run directory.
        """
        dataset = ShardedLanceVSTDataset(
            run_root / "train.lance", batch_size=2, ot=False, use_saved_mean_and_variance=False
        )
        assert len(dataset) == 8 // 2

    def test_train_split_reads_train_shard_rows(self, run_root: Path) -> None:
        """A train batch crossing the shard boundary returns exactly global rows 2:6.

        :param run_root: Fixture-provided run directory.
        """
        dataset = ShardedLanceVSTDataset(
            run_root / "train.lance",
            batch_size=4,
            ot=False,
            use_saved_mean_and_variance=False,
            rescale_params=False,
        )
        params = _run_columns(_RUN_ROWS)["param_array"]
        np.testing.assert_array_equal(dataset[(2, 6)]["params"].numpy(), params[2:6])

    def test_val_split_reads_rows_after_train_shards(self, run_root: Path) -> None:
        """The val split's first batch is the run's global rows 8:10, not rows 0:2.

        :param run_root: Fixture-provided run directory.
        """
        dataset = ShardedLanceVSTDataset(
            run_root / "val.lance",
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=False,
            rescale_params=False,
        )
        params = _run_columns(_RUN_ROWS)["param_array"]
        np.testing.assert_array_equal(dataset[0]["params"].numpy(), params[8:10])

    def test_existing_lance_dir_opens_as_single_shard(self, run_root: Path) -> None:
        """A path to a real ``.lance`` dataset directory bypasses the spec (predict_file path).

        :param run_root: Fixture-provided run directory.
        """
        dataset = ShardedLanceVSTDataset(
            run_root / "shard-000003.lance",
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=False,
        )
        assert len(dataset) == 4 // 2

    def test_mel_normalized_with_run_root_stats(self, tmp_path: Path) -> None:
        """``stats.npz`` at the run root normalizes mel reads: ``(3 - 1) / 2 = 1``.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        _write_run_root(tmp_path / "run", mel_fill=3.0, stats_mean=1.0, stats_std=2.0)
        dataset = ShardedLanceVSTDataset(
            tmp_path / "run" / "train.lance",
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=True,
        )
        mel = dataset[0]["mel_spec"]
        assert mel is not None
        assert torch.allclose(mel, torch.ones_like(mel))

    def test_missing_input_spec_raises_file_not_found(self, tmp_path: Path) -> None:
        """A split path whose parent has no ``input_spec.json`` errors clearly.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="input_spec.json"):
            ShardedLanceVSTDataset(
                tmp_path / "empty" / "train.lance",
                batch_size=2,
                ot=False,
                use_saved_mean_and_variance=False,
            )

    def test_unknown_split_stem_raises_value_error(self, run_root: Path) -> None:
        """A virtual filename that is not a split name fails instead of guessing.

        :param run_root: Fixture-provided run directory.
        """
        with pytest.raises(ValueError, match="weird"):
            ShardedLanceVSTDataset(
                run_root / "weird.lance",
                batch_size=2,
                ot=False,
                use_saved_mean_and_variance=False,
            )

    def test_empty_split_raises_value_error(self, tmp_path: Path) -> None:
        """Opening a split whose shard range is empty errors at construction.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        _write_run_root(tmp_path / "run", train_val_test_sizes=(8, 4, 0))
        with pytest.raises(ValueError, match="test split is empty"):
            ShardedLanceVSTDataset(
                tmp_path / "run" / "test.lance",
                batch_size=2,
                ot=False,
                use_saved_mean_and_variance=False,
            )


class TestShardedLanceVSTDataModule:
    """Lightning wiring over a run directory: setup / dataloaders / teardown."""

    def _module(self, run_root: Path, **kwargs: object) -> ShardedLanceVSTDataModule:
        """Construct a module with the test-suite defaults.

        :param run_root: Run directory serving as ``dataset_root``.
        :param \\*\\*kwargs: Overrides forwarded to the datamodule.
        :return: Unsetup datamodule.
        """
        defaults: dict[str, object] = dict(
            dataset_root=run_root,
            batch_size=2,
            ot=False,
            num_workers=0,
            pin_memory=False,
            use_saved_mean_and_variance=False,
        )
        defaults.update(kwargs)
        return ShardedLanceVSTDataModule(**defaults)  # type: ignore[arg-type]

    def test_setup_builds_sharded_splits_with_correct_lengths(self, run_root: Path) -> None:
        """``setup()`` opens train/val/test as sharded datasets sized per the spec splits.

        :param run_root: Fixture-provided run directory.
        """
        module = self._module(run_root)
        module.setup()
        try:
            assert isinstance(module.train_dataset, ShardedLanceVSTDataset)
            assert len(module.train_dataset) == 8 // 2
            assert len(module.val_dataset) == 4 // 2
            assert len(module.test_dataset) == 4 // 2
        finally:
            module.teardown()

    def test_train_dataloader_yields_batches_across_shard_boundaries(
        self, run_root: Path
    ) -> None:
        """End-to-end smoke: the train loader iterates real sharded Lance reads.

        :param run_root: Fixture-provided run directory.
        """
        module = self._module(run_root)
        module.setup()
        try:
            item = next(iter(module.train_dataloader()))
            params = item["params"]
            assert params is not None and params.shape == (2, _NUM_PARAMS)
            mel = item["mel_spec"]
            assert mel is not None and mel.shape == (2, *_MEL_SHAPE)
        finally:
            module.teardown()

    def test_predict_dataloader_defaults_to_test_split_with_audio(self, run_root: Path) -> None:
        """The default predict dataset is the sharded test split and force-reads audio.

        :param run_root: Fixture-provided run directory.
        """
        module = self._module(run_root)
        module.setup()
        try:
            item = next(iter(module.predict_dataloader()))
            audio = item["audio"]
            assert audio is not None and audio.shape == (2, *_AUDIO_SHAPE)
        finally:
            module.teardown()

    def test_teardown_closes_every_split_handle(self, run_root: Path) -> None:
        """``teardown`` closes all four split handles so they read as falsy.

        :param run_root: Fixture-provided run directory.
        """
        module = self._module(run_root)
        module.setup()
        module.teardown()
        assert not module.train_dataset.dataset_file
        assert not module.val_dataset.dataset_file
        assert not module.test_dataset.dataset_file
        assert not module.predict_dataset.dataset_file
