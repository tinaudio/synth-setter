"""Behavioral tests for :mod:`synth_setter.data.lance_datamodule`.

Covers the public symbols exposed by the module:

* :class:`LanceShardFile` — the read-only, h5py-``File``-like adapter over a
  ``.lance`` shard file (column access, slice / fancy-index reads,
  ``shape``, ``close`` semantics).
* :class:`LanceVSTDataset` — the Lance-backed sibling of ``VSTDataset``:
  same batch-per-index ``__getitem__`` contract (float32 tensors, sibling
  ``stats.npz`` normalization, parameter rescaling, OT routing).
* :class:`LanceVSTDataModule` — Lightning ``setup`` / dataloader / ``teardown``
  wiring over ``train/val/test.lance`` splits.

Lance fixtures are tiny (a handful of rows, ~10-element mel/audio axes) — the
goal is contract coverage on shapes, flags, and call routing, not numerical
ML behavior, mirroring ``tests/data/test_surge_datamodule.py``.
"""

from __future__ import annotations

import contextlib
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

from synth_setter.data.lance_datamodule import (
    LanceShardFile,
    LanceVSTDataModule,
    LanceVSTDataset,
)
from synth_setter.data.surge_datamodule import ShiftedBatchSampler
from synth_setter.data.vst.param_spec_registry import param_specs
from tests.helpers.finalize_shards import build_lance_smoke_spec, write_minimal_lance_shard
from tests.helpers.lance_fixtures import write_lance_shard

_AUDIO_CHANNELS = 2
_AUDIO_SAMPLES = 16
_MEL_CHANNELS = 2
_MEL_N_MELS = 4
_MEL_N_FRAMES = 5
_M2L_DIM_1 = 6
_M2L_DIM_2 = 7
_NUM_PARAMS = 11

_ALL_TENSOR_KEYS = ("audio", "mel_spec", "m2l", "params", "noise")


def _make_columns(num_rows: int, *, params_seed: int = 0) -> dict[str, np.ndarray]:
    """Build the column arrays a Lance shard carries.

    :param num_rows: Number of rows along the first axis of every column.
    :param params_seed: Seed for all columns so different splits get
        distinguishable values when needed.

    :return: Mapping of column name to ``(num_rows, ...)`` float32 array.
    """
    rng = np.random.default_rng(params_seed)
    return {
        # float16 mirrors the pipeline's on-disk audio dtype (DATASET_FIELD_DTYPES).
        "audio": rng.standard_normal((num_rows, _AUDIO_CHANNELS, _AUDIO_SAMPLES)).astype(
            np.float16
        ),
        "mel_spec": rng.standard_normal(
            (num_rows, _MEL_CHANNELS, _MEL_N_MELS, _MEL_N_FRAMES)
        ).astype(np.float32),
        "music2latent": rng.standard_normal((num_rows, _M2L_DIM_1, _M2L_DIM_2)).astype(np.float32),
        # params in [0, 1) so the rescale_params=True branch lands in [-1, 1).
        "param_array": rng.random((num_rows, _NUM_PARAMS)).astype(np.float32),
    }


def _write_seeded_lance_shard(
    path: Path,
    num_rows: int,
    *,
    params_seed: int = 0,
    mel_fill: float | None = None,
) -> dict[str, np.ndarray]:
    """Write a tiny Lance shard with the columns ``LanceVSTDataset`` reads.

    :param path: Output ``.lance`` shard file path.
    :param num_rows: Number of rows along the first axis of every column.
    :param params_seed: Seed for the per-row arrays so different splits get
        distinguishable values when needed.
    :param mel_fill: When set, fill the ``mel_spec`` column with this constant
        instead of random values — used by the normalization test to make
        ``(mel - mean) / std`` produce a predictable result.

    :return: The column arrays that were written, for ground-truth assertions.
    """
    columns = _make_columns(num_rows, params_seed=params_seed)
    if mel_fill is not None:
        columns["mel_spec"] = np.full_like(columns["mel_spec"], mel_fill)
    write_lance_shard(path, columns)
    return columns


def _write_stats(
    dataset_dir: Path,
    *,
    mean: float = 0.0,
    std: float = 1.0,
) -> Path:
    """Write a sibling ``stats.npz`` whose mean/std broadcast against ``mel_spec``.

    :param dataset_dir: Directory holding ``*.lance``; ``stats.npz`` is written
        alongside.
    :param mean: Scalar mean — broadcast against every mel-spec element.
    :param std: Scalar std — broadcast against every mel-spec element.

    :return: Path to the written ``stats.npz``.
    """
    stats_path = dataset_dir / "stats.npz"
    mel_shape = (_MEL_CHANNELS, _MEL_N_MELS, _MEL_N_FRAMES)
    np.savez(
        stats_path,
        mean=np.full(mel_shape, mean, dtype=np.float32),
        std=np.full(mel_shape, std, dtype=np.float32),
    )
    return stats_path


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Build a ``dataset_root`` directory with ``train/val/test.lance`` + ``stats.npz``.

    :param tmp_path: Per-test tmpdir.
    :return: Path to the populated dataset root directory.
    """
    root = tmp_path / "data"
    root.mkdir()
    _write_seeded_lance_shard(root / "train.lance", num_rows=8, params_seed=1)
    _write_seeded_lance_shard(root / "val.lance", num_rows=8, params_seed=2)
    _write_seeded_lance_shard(root / "test.lance", num_rows=8, params_seed=3)
    _write_stats(root)
    return root


@pytest.fixture
def single_shard(tmp_path: Path) -> Path:
    """Write a single ``train.lance`` + sibling ``stats.npz`` for dataset-only tests.

    :param tmp_path: Per-test tmpdir.

    :return: Path to the written ``train.lance`` shard file.
    """
    shard_path = tmp_path / "train.lance"
    _write_seeded_lance_shard(shard_path, num_rows=8)
    _write_stats(tmp_path)
    return shard_path


@contextlib.contextmanager
def _set_up_module(**kwargs: object) -> Iterator[LanceVSTDataModule]:
    """Construct a ``LanceVSTDataModule`` from ``**kwargs``, ``setup``, yield, then ``teardown``.

    :param \\*\\*kwargs: Forwarded to ``LanceVSTDataModule``.
    :yields: The set-up datamodule for assertion work inside the ``with`` block.
    :ytype: LanceVSTDataModule
    """
    module = LanceVSTDataModule(**kwargs)  # type: ignore[arg-type]
    module.setup()
    try:
        yield module
    finally:
        module.teardown()


def _unwrap(maybe_tensor: torch.Tensor | None) -> torch.Tensor:
    """Assert ``maybe_tensor`` is non-None and return it as a ``torch.Tensor``.

    :param maybe_tensor: The dict value to narrow.
    :return: The same tensor, now typed as non-None.
    """
    assert maybe_tensor is not None
    return maybe_tensor


class TestLanceShardFile:
    """H5py-``File``-like adapter: column access, slicing, shape, close."""

    def test_column_slice_read_matches_source_rows(self, tmp_path: Path) -> None:
        """``file[name][a:b]`` returns the same rows as the written numpy array.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        columns = _write_seeded_lance_shard(tmp_path / "train.lance", num_rows=8)
        shard = LanceShardFile(tmp_path / "train.lance")
        out = shard["mel_spec"][2:5]
        assert out.shape == (3, _MEL_CHANNELS, _MEL_N_MELS, _MEL_N_FRAMES)
        assert out.dtype == columns["mel_spec"].dtype
        np.testing.assert_array_equal(out, columns["mel_spec"][2:5])

    def test_column_reads_return_writable_arrays(self, tmp_path: Path) -> None:
        """Reads are copied out of Arrow's read-only buffer before reaching torch.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        _write_seeded_lance_shard(tmp_path / "train.lance", num_rows=8)
        shard = LanceShardFile(tmp_path / "train.lance")
        assert shard["audio"][0:2].flags.writeable

    def test_column_open_ended_slice_reads_from_row_zero(self, tmp_path: Path) -> None:
        """``file[name][:k]`` (no explicit start) reads the first ``k`` rows.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        columns = _write_seeded_lance_shard(tmp_path / "train.lance", num_rows=8)
        shard = LanceShardFile(tmp_path / "train.lance")
        np.testing.assert_array_equal(shard["param_array"][:3], columns["param_array"][:3])

    def test_column_fancy_index_read_selects_rows(self, tmp_path: Path) -> None:
        """``file[name][[i, j, k]]`` gathers exactly those rows.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        columns = _write_seeded_lance_shard(tmp_path / "train.lance", num_rows=8)
        shard = LanceShardFile(tmp_path / "train.lance")
        np.testing.assert_array_equal(shard["audio"][[0, 3, 6]], columns["audio"][[0, 3, 6]])

    def test_column_numpy_int_indices_are_accepted(self, tmp_path: Path) -> None:
        """Samplers yield numpy integer arrays — the column accepts them directly.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        columns = _write_seeded_lance_shard(tmp_path / "train.lance", num_rows=8)
        shard = LanceShardFile(tmp_path / "train.lance")
        idx = np.array([1, 2, 4], dtype=np.int64)
        np.testing.assert_array_equal(shard["param_array"][idx], columns["param_array"][idx])

    def test_column_shape_reports_rows_and_tensor_dims(self, tmp_path: Path) -> None:
        """``file[name].shape`` mirrors h5py: ``(num_rows, *tensor_shape)``.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        _write_seeded_lance_shard(tmp_path / "train.lance", num_rows=8)
        shard = LanceShardFile(tmp_path / "train.lance")
        assert shard["mel_spec"].shape == (8, _MEL_CHANNELS, _MEL_N_MELS, _MEL_N_FRAMES)
        assert shard["param_array"].shape == (8, _NUM_PARAMS)

    def test_open_shard_is_truthy_closed_shard_is_falsy(self, tmp_path: Path) -> None:
        """``close()`` flips the handle falsy — the contract ``teardown`` relies on.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        _write_seeded_lance_shard(tmp_path / "train.lance", num_rows=8)
        shard = LanceShardFile(tmp_path / "train.lance")
        assert shard
        shard.close()
        assert not shard

    def test_column_access_after_close_raises_value_error(self, tmp_path: Path) -> None:
        """Reading from a closed shard fails loudly instead of returning stale data.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        _write_seeded_lance_shard(tmp_path / "train.lance", num_rows=8)
        shard = LanceShardFile(tmp_path / "train.lance")
        shard.close()
        with pytest.raises(ValueError, match="closed"):
            _ = shard["mel_spec"]

    def test_stale_column_read_after_close_raises_value_error(self, tmp_path: Path) -> None:
        """A column view obtained before ``close()`` cannot read afterwards.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        _write_seeded_lance_shard(tmp_path / "train.lance", num_rows=8)
        shard = LanceShardFile(tmp_path / "train.lance")
        column = shard["mel_spec"]
        shard.close()
        with pytest.raises(ValueError, match="closed"):
            _ = column[0:2]

    def test_missing_column_raises_key_error_at_lookup(self, tmp_path: Path) -> None:
        """``file[name]`` for an absent column raises ``KeyError`` like h5py, not on first read.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        _write_seeded_lance_shard(tmp_path / "train.lance", num_rows=8)
        shard = LanceShardFile(tmp_path / "train.lance")
        with pytest.raises(KeyError, match="no-such-column"):
            _ = shard["no-such-column"]

    def test_missing_shard_file_raises_value_error(self, tmp_path: Path) -> None:
        """Opening a nonexistent ``.lance`` path errors at construction, not first read.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        with pytest.raises(ValueError, match="does-not-exist"):
            LanceShardFile(tmp_path / "does-not-exist.lance")

    def test_directory_path_raises_value_error(self, tmp_path: Path) -> None:
        """A ``.lance`` *directory* (the Lance dataset format) is rejected — shards are files.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        dataset_dir = tmp_path / "old-format.lance"
        dataset_dir.mkdir()
        with pytest.raises(ValueError, match="directory"):
            LanceShardFile(dataset_dir)

    def test_column_step_slice_reads_strided_rows(self, tmp_path: Path) -> None:
        """``file[name][a:b:s]`` with a step gathers exactly the strided rows.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        columns = _write_seeded_lance_shard(tmp_path / "train.lance", num_rows=8)
        shard = LanceShardFile(tmp_path / "train.lance")
        np.testing.assert_array_equal(shard["param_array"][1:8:3], columns["param_array"][1:8:3])

    def test_column_negative_step_slice_raises_value_error(self, tmp_path: Path) -> None:
        """A negative-step slice is rejected — the same contract h5py enforces.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        _write_seeded_lance_shard(tmp_path / "train.lance", num_rows=8)
        shard = LanceShardFile(tmp_path / "train.lance")
        with pytest.raises(ValueError, match="step"):
            _ = shard["param_array"][::-1]

    def test_column_unsorted_fancy_index_raises_value_error(self, tmp_path: Path) -> None:
        """Fancy indices must be ascending — the same contract h5py enforces; samplers sort.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        _write_seeded_lance_shard(tmp_path / "train.lance", num_rows=8)
        shard = LanceShardFile(tmp_path / "train.lance")
        with pytest.raises(ValueError, match="ascending"):
            _ = shard["param_array"][[4, 1]]


class TestLanceVSTDataset:
    """Lance-backed dataset: same batch-per-index contract as ``VSTDataset``."""

    def test_explicit_stats_file_overrides_sibling_stats_npz(self, tmp_path: Path) -> None:
        """An explicit ``stats_file`` loads mel mean/std from that path, not the shard's sibling.

        Streaming reads the shard from R2 while ``stats.npz`` is fetched to a
        separate local dir, so the two locations must decouple.

        :param tmp_path: Hosts the shard with its sibling stats and the override stats.
        """
        shard_dir = tmp_path / "data"
        shard_dir.mkdir()
        _write_seeded_lance_shard(shard_dir / "train.lance", num_rows=8)
        _write_stats(shard_dir, mean=0.0, std=1.0)
        custom_dir = tmp_path / "elsewhere"
        custom_dir.mkdir()
        custom_stats = _write_stats(custom_dir, mean=7.0, std=3.0)

        dataset = LanceVSTDataset(
            shard_dir / "train.lance",
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=True,
            stats_file=custom_stats,
        )

        mel_shape = (_MEL_CHANNELS, _MEL_N_MELS, _MEL_N_FRAMES)
        np.testing.assert_array_equal(dataset.mean, np.full(mel_shape, 7.0, dtype=np.float32))
        np.testing.assert_array_equal(dataset.std, np.full(mel_shape, 3.0, dtype=np.float32))

    def test_remote_shard_without_stats_file_raises_before_open(self) -> None:
        """A remote shard with saved stats but no explicit ``stats_file`` fails fast.

        The sibling ``stats.npz`` derivation only works for local paths, so the
        constructor must reject the remote-without-``stats_file`` combination
        before it opens the (network) shard.
        """
        with pytest.raises(ValueError, match="stats_file is required"):
            LanceVSTDataset(
                "s3://intermediate-data/data/run/train.lance",
                batch_size=2,
                ot=False,
                use_saved_mean_and_variance=True,
            )

    def test_len_equals_num_rows_floor_divided_by_batch_size(self, single_shard: Path) -> None:
        """``__len__`` floor-divides the row count by ``batch_size``, dropping the ragged tail.

        :param single_shard: Fixture-provided single-shard Lance path.
        """
        dataset = LanceVSTDataset(single_shard, batch_size=3, ot=False)
        assert len(dataset) == 8 // 3

    def test_getitem_int_returns_batch_size_slice(self, single_shard: Path) -> None:
        """Integer index ``i`` reads rows ``[i*B : i*B+B]`` from each column.

        :param single_shard: Fixture-provided single-shard Lance path.
        """
        dataset = LanceVSTDataset(
            single_shard, batch_size=2, ot=False, use_saved_mean_and_variance=False
        )
        item = dataset[1]
        assert _unwrap(item["params"]).shape == (2, _NUM_PARAMS)
        assert _unwrap(item["mel_spec"]).shape == (
            2,
            _MEL_CHANNELS,
            _MEL_N_MELS,
            _MEL_N_FRAMES,
        )

    def test_getitem_int_maps_index_to_batch_rows(self, tmp_path: Path) -> None:
        """Integer index ``i`` maps to exactly the ``i``-th block of ``batch_size`` rows.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        columns = _write_seeded_lance_shard(tmp_path / "train.lance", num_rows=8)
        dataset = LanceVSTDataset(
            tmp_path / "train.lance",
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=False,
            rescale_params=False,
        )
        np.testing.assert_array_equal(
            _unwrap(dataset[1]["params"]).numpy(), columns["param_array"][2:4]
        )

    def test_getitem_tuple_returns_explicit_slice(self, tmp_path: Path) -> None:
        """A 2-tuple index ``(lo, hi)`` selects exactly rows ``[lo:hi]``.

        Pins the row values, not just the count — ``ShiftedBatchSampler`` feeds
        this path in production, so an off-by-one must fail here.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        columns = _write_seeded_lance_shard(tmp_path / "train.lance", num_rows=8)
        dataset = LanceVSTDataset(
            tmp_path / "train.lance",
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=False,
            rescale_params=False,
        )
        item = dataset[(1, 5)]
        np.testing.assert_array_equal(_unwrap(item["params"]).numpy(), columns["param_array"][1:5])

    def test_getitem_sequence_falls_through_to_fancy_indexing(self, single_shard: Path) -> None:
        """A non-int / non-2-tuple index gathers exactly those rows.

        :param single_shard: Fixture-provided single-shard Lance path.
        """
        dataset = LanceVSTDataset(
            single_shard, batch_size=2, ot=False, use_saved_mean_and_variance=False
        )
        item = dataset[[0, 2, 4]]
        assert _unwrap(item["params"]).shape[0] == 3

    def test_repeat_first_batch_ignores_idx(self, single_shard: Path) -> None:
        """``repeat_first_batch=True`` always returns the first ``batch_size`` rows.

        :param single_shard: Fixture-provided single-shard Lance path.
        """
        dataset = LanceVSTDataset(
            single_shard,
            batch_size=3,
            ot=False,
            use_saved_mean_and_variance=False,
            repeat_first_batch=True,
        )
        assert torch.equal(_unwrap(dataset[0]["params"]), _unwrap(dataset[2]["params"]))

    def test_returned_tensors_are_float32_and_contiguous(self, single_shard: Path) -> None:
        """All populated tensors are ``torch.float32`` and ``.contiguous()``.

        :param single_shard: Fixture-provided single-shard Lance path.
        """
        dataset = LanceVSTDataset(
            single_shard,
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=False,
            read_audio=True,
            read_mel=True,
            read_m2l=True,
        )
        item = dataset[0]
        for key in _ALL_TENSOR_KEYS:
            assert _unwrap(item[key]).dtype == torch.float32, key
            assert _unwrap(item[key]).is_contiguous(), f"{key} not contiguous"

    def test_read_flags_route_modalities(self, single_shard: Path) -> None:
        """``read_audio`` / ``read_mel`` / ``read_m2l`` toggle their dict slots.

        :param single_shard: Fixture-provided single-shard Lance path.
        """
        dataset = LanceVSTDataset(
            single_shard,
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=False,
            read_audio=False,
            read_mel=False,
            read_m2l=True,
        )
        item = dataset[0]
        assert item["audio"] is None
        assert item["mel_spec"] is None
        assert _unwrap(item["m2l"]).shape == (2, _M2L_DIM_1, _M2L_DIM_2)

    def test_rescale_params_centers_to_minus_one_to_one(self, single_shard: Path) -> None:
        """``rescale_params=True`` applies ``p * 2 - 1`` element-wise.

        :param single_shard: Fixture-provided single-shard Lance path.
        """
        dataset_raw = LanceVSTDataset(
            single_shard,
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=False,
            rescale_params=False,
        )
        dataset_rescaled = LanceVSTDataset(
            single_shard,
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=False,
            rescale_params=True,
        )
        raw = _unwrap(dataset_raw[0]["params"])
        rescaled = _unwrap(dataset_rescaled[0]["params"])
        assert torch.allclose(rescaled, raw * 2 - 1)

    def test_mel_spec_normalized_with_loaded_stats(self, tmp_path: Path) -> None:
        """When ``stats.npz`` is loaded, mel is returned as ``(mel - mean) / std``.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        _write_seeded_lance_shard(tmp_path / "train.lance", num_rows=4, mel_fill=3.0)
        _write_stats(tmp_path, mean=1.0, std=2.0)
        dataset = LanceVSTDataset(
            tmp_path / "train.lance",
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=True,
        )
        mel = _unwrap(dataset[0]["mel_spec"])
        expected = (3.0 - 1.0) / 2.0
        assert torch.allclose(mel, torch.full_like(mel, expected))

    def test_missing_stats_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """``use_saved_mean_and_variance=True`` with no sibling ``stats.npz`` errors clearly.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        _write_seeded_lance_shard(tmp_path / "train.lance", num_rows=4)
        with pytest.raises(FileNotFoundError, match="stats.npz"):
            LanceVSTDataset(
                tmp_path / "train.lance",
                batch_size=2,
                ot=False,
                use_saved_mean_and_variance=True,
            )

    def test_ot_true_applies_hungarian_match_to_real_batch(self, single_shard: Path) -> None:
        """``ot=True`` returns a noise/params pairing that is a row permutation of ``ot=False``.

        The Hungarian match may only reorder rows — every returned param row must still be one of
        the underlying batch rows.

        :param single_shard: Fixture-provided single-shard Lance path.
        """
        plain = LanceVSTDataset(
            single_shard,
            batch_size=4,
            ot=False,
            use_saved_mean_and_variance=False,
        )
        matched = LanceVSTDataset(
            single_shard,
            batch_size=4,
            ot=True,
            use_saved_mean_and_variance=False,
        )
        plain_params = _unwrap(plain[0]["params"])
        matched_params = _unwrap(matched[0]["params"])
        # Bijection check: sorting rows lexicographically must yield identical
        # tensors — a dropped or duplicated row fails, the identity permutation
        # is legitimately allowed (Hungarian may return it).
        plain_sorted = plain_params[torch.argsort(plain_params[:, 0])]
        matched_sorted = matched_params[torch.argsort(matched_params[:, 0])]
        assert torch.allclose(matched_sorted, plain_sorted)

    def test_fake_mode_skips_lance_entirely(self, tmp_path: Path) -> None:
        """``fake=True`` accepts a nonexistent path and synthesizes batches in memory.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        dataset = LanceVSTDataset(tmp_path / "missing.lance", batch_size=3, fake=True)
        assert dataset.dataset_file is None
        item = dataset[0]
        assert _unwrap(item["params"]).shape == (3, len(param_specs["surge_xt"]))


class TestLanceVSTDataModule:
    """Lightning datamodule: setup / dataloaders / teardown wiring over Lance splits."""

    def test_setup_creates_lance_backed_splits(self, dataset_root: Path) -> None:
        """``setup()`` opens the three required ``.lance`` splits as ``LanceVSTDataset``.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_module(dataset_root=dataset_root, batch_size=2, ot=False) as module:
            assert isinstance(module.train_dataset, LanceVSTDataset)
            assert isinstance(module.val_dataset, LanceVSTDataset)
            assert isinstance(module.test_dataset, LanceVSTDataset)

    def test_setup_without_predict_file_defaults_to_test_split(self, dataset_root: Path) -> None:
        """No ``predict_file``: ``predict_dataset`` defaults to the ``test.lance`` split.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_module(dataset_root=dataset_root, batch_size=2, ot=False) as module:
            assert module.predict_file == dataset_root / "test.lance"
            assert module.predict_dataset.read_audio is True

    def test_setup_val_and_test_force_ot_false(self, dataset_root: Path) -> None:
        """``setup`` hard-codes ``ot=False`` on val/test even when the module is ``ot=True``.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_module(dataset_root=dataset_root, batch_size=2, ot=True) as module:
            assert module.train_dataset.ot is True
            assert module.val_dataset.ot is False
            assert module.test_dataset.ot is False

    def test_conditioning_m2l_routes_to_m2l_reads(self, dataset_root: Path) -> None:
        """``conditioning='m2l'`` flips the read flags to the music2latent channel.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_module(
            dataset_root=dataset_root, batch_size=2, ot=False, conditioning="m2l"
        ) as module:
            for split in (module.train_dataset, module.val_dataset, module.test_dataset):
                assert split.read_mel is False
                assert split.read_m2l is True

    def test_train_dataloader_yields_well_shaped_batches(self, dataset_root: Path) -> None:
        """End-to-end smoke: the train loader iterates real Lance reads into batches.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_module(
            dataset_root=dataset_root,
            batch_size=2,
            ot=False,
            num_workers=0,
            pin_memory=False,
        ) as module:
            loader = module.train_dataloader()
            assert isinstance(loader.sampler, ShiftedBatchSampler)
            item = next(iter(loader))
            assert _unwrap(item["params"]).shape == (2, _NUM_PARAMS)
            assert _unwrap(item["mel_spec"]).shape == (
                2,
                _MEL_CHANNELS,
                _MEL_N_MELS,
                _MEL_N_FRAMES,
            )

    def test_val_dataloader_iterates_sequentially_over_all_batches(
        self, dataset_root: Path
    ) -> None:
        """The val loader walks every batch of ``val.lance`` in order without error.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_module(
            dataset_root=dataset_root,
            batch_size=2,
            ot=False,
            num_workers=0,
            pin_memory=False,
        ) as module:
            loader = module.val_dataloader()
            assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)
            batches = list(loader)
            assert len(batches) == 4
            for batch in batches:
                assert _unwrap(batch["params"]).shape == (2, _NUM_PARAMS)

    def test_predict_dataloader_reads_audio_for_rendering(self, dataset_root: Path) -> None:
        """The predict loader force-reads audio so ground truth can be rendered.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_module(
            dataset_root=dataset_root,
            batch_size=2,
            ot=False,
            num_workers=0,
            pin_memory=False,
        ) as module:
            item = next(iter(module.predict_dataloader()))
            assert _unwrap(item["audio"]).shape == (2, _AUDIO_CHANNELS, _AUDIO_SAMPLES)

    def test_teardown_closes_open_lance_handles(self, dataset_root: Path) -> None:
        """``teardown`` closes every split so handles read as falsy afterwards.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        module = LanceVSTDataModule(dataset_root=dataset_root, batch_size=2, ot=False)
        module.setup()
        module.teardown()
        assert not module.train_dataset.dataset_file
        assert not module.val_dataset.dataset_file
        assert not module.test_dataset.dataset_file
        assert not module.predict_dataset.dataset_file

    def test_fake_mode_setup_does_not_require_dataset_files(self, tmp_path: Path) -> None:
        """``fake=True`` setup never touches the dataset_root, so a fresh dir is enough.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        with _set_up_module(
            dataset_root=tmp_path,
            batch_size=2,
            ot=False,
            fake=True,
            use_saved_mean_and_variance=False,
        ) as module:
            assert module.train_dataset.fake is True
            item = next(iter(module.val_dataloader()))
            assert _unwrap(item["params"]).shape == (2, len(param_specs["surge_xt"]))

    def test_val_dataloader_multi_worker_matches_single_worker(self, dataset_root: Path) -> None:
        """``num_workers=2`` forked workers read the same batches as in-process loading.

        Lance handles are not fork-safe, so ``LanceShardFile`` reopens per
        worker — multi-worker loaders (the production default) must produce
        the same data as ``num_workers=0``.

        :param dataset_root: Fixture-provided dataset-root directory.
        """

        def collect(num_workers: int) -> torch.Tensor:
            with _set_up_module(
                dataset_root=dataset_root,
                batch_size=2,
                ot=False,
                num_workers=num_workers,
                pin_memory=False,
            ) as module:
                return torch.cat([_unwrap(b["params"]) for b in module.val_dataloader()])

        assert torch.allclose(collect(num_workers=2), collect(num_workers=0))


class TestPipelineWriterCompatibility:
    """Shards from the pipeline's Lance writer are readable by the datamodule classes."""

    def test_shard_file_reads_pipeline_written_shard(self, tmp_path: Path) -> None:
        """A shard written via the production ``lance_schema``/``write_lance_file`` path reads back.

        ``write_minimal_lance_shard`` fills ``mel_spec`` with ``np.arange`` and
        stores ``audio`` as float16, so this pins values, dtype, and the
        writer↔reader format contract across the pipeline boundary.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        spec = build_lance_smoke_spec()
        shard_path = tmp_path / "train.lance"
        write_minimal_lance_shard(shard_path, spec)
        shard = LanceShardFile(shard_path)
        rows = spec.render.samples_per_shard
        mel = shard["mel_spec"][0:rows]
        np.testing.assert_array_equal(
            mel, np.arange(np.prod(mel.shape), dtype=np.float32).reshape(mel.shape)
        )
        assert shard["audio"][0:rows].dtype == np.float16
        assert shard["param_array"].shape == (rows, spec.num_params)

    def test_dataset_batches_from_pipeline_written_shard(self, tmp_path: Path) -> None:
        """``LanceVSTDataset`` serves float32 batches from a pipeline-written shard.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        spec = build_lance_smoke_spec()
        shard_path = tmp_path / "train.lance"
        write_minimal_lance_shard(shard_path, spec)
        dataset = LanceVSTDataset(
            shard_path,
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=False,
            read_audio=True,
        )
        item = dataset[0]
        assert _unwrap(item["params"]).shape == (2, spec.num_params)
        assert _unwrap(item["audio"]).dtype == torch.float32
        assert _unwrap(item["mel_spec"]).dtype == torch.float32


@pytest.fixture()
def local_r2_remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Back the ``r2:`` rclone remote with the local filesystem for real-binary e2e.

    ``RCLONE_CONFIG_R2_TYPE=local`` resolves ``r2://<bucket>/<key>`` to
    ``<cwd>/<bucket>/<key>``, and the three secret keys satisfy
    ``ensure_r2_env_loaded`` / ``r2_storage_options`` presence checks (their
    values are unused by the local backend). Skips when ``rclone`` is absent.

    :param tmp_path: Pytest tmp dir; the returned subdir is the fake R2 root.
    :param monkeypatch: Sets the rclone env vars and chdirs into the remote root.
    :return: The fake R2 root; ``r2://<bucket>/<key>`` materializes under it.
    """
    if shutil.which("rclone") is None:
        pytest.skip("rclone binary not available on PATH")
    remote_root = tmp_path / "r2"
    remote_root.mkdir()
    monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "local")
    monkeypatch.setenv("RCLONE_CONFIG_R2_ACCESS_KEY_ID", "stub")
    monkeypatch.setenv("RCLONE_CONFIG_R2_SECRET_ACCESS_KEY", "stub")
    monkeypatch.setenv("RCLONE_CONFIG_R2_ENDPOINT", "stub")
    monkeypatch.chdir(remote_root)
    return remote_root


class TestLanceVSTDataModuleStreaming:
    """``stream_from_r2`` wiring: remote split URIs, stats-only hydration, validation."""

    _ROOT_URI = "r2://intermediate-data/data/run/"

    def test_stream_from_r2_without_root_uri_raises(self) -> None:
        """Streaming needs a dataset prefix to read from; omitting it fails fast."""
        with pytest.raises(ValueError, match="download_dataset_root_uri"):
            LanceVSTDataModule(dataset_root="/cache", stream_from_r2=True)

    def test_split_target_is_s3_uri_under_dataset_prefix(self) -> None:
        """Each split resolves to its ``s3://`` object under the dataset prefix."""
        module = LanceVSTDataModule(
            dataset_root="/cache",
            download_dataset_root_uri=self._ROOT_URI,
            stream_from_r2=True,
        )
        assert module._split_target("train.lance") == "s3://intermediate-data/data/run/train.lance"

    def test_predict_file_defaults_to_remote_test_split(self) -> None:
        """With no explicit predict_file, predict streams the remote test split."""
        module = LanceVSTDataModule(
            dataset_root="/cache",
            download_dataset_root_uri=self._ROOT_URI,
            stream_from_r2=True,
        )
        assert module.predict_file == "s3://intermediate-data/data/run/test.lance"

    def test_dataset_extra_kwargs_supplies_storage_options_and_local_stats(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Streaming hands each split the R2 storage_options plus the local stats path.

        :param monkeypatch: Sets the R2 secrets and stubs the auth ping.
        """
        monkeypatch.setattr(
            "synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda *a, **k: None
        )
        monkeypatch.setenv("RCLONE_CONFIG_R2_ACCESS_KEY_ID", "ak")
        monkeypatch.setenv("RCLONE_CONFIG_R2_SECRET_ACCESS_KEY", "sk")
        monkeypatch.setenv("RCLONE_CONFIG_R2_ENDPOINT", "https://acct.r2.cloudflarestorage.com")
        module = LanceVSTDataModule(
            dataset_root="/cache",
            download_dataset_root_uri=self._ROOT_URI,
            stream_from_r2=True,
        )

        extra = module._dataset_extra_kwargs()

        assert extra["stats_file"] == Path("/cache") / "stats.npz"
        assert extra["storage_options"] == {
            "aws_access_key_id": "ak",
            "aws_secret_access_key": "sk",
            "aws_endpoint": "https://acct.r2.cloudflarestorage.com",
            "aws_region": "auto",
        }

    def test_dataset_extra_kwargs_omits_stats_file_when_stats_unused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With stats off, no ``stats_file`` is injected — matching the hydration that skips it.

        :param monkeypatch: Sets the R2 secrets and stubs the auth ping.
        """
        monkeypatch.setattr(
            "synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda *a, **k: None
        )
        monkeypatch.setenv("RCLONE_CONFIG_R2_ACCESS_KEY_ID", "ak")
        monkeypatch.setenv("RCLONE_CONFIG_R2_SECRET_ACCESS_KEY", "sk")
        monkeypatch.setenv("RCLONE_CONFIG_R2_ENDPOINT", "https://acct.r2.cloudflarestorage.com")
        module = LanceVSTDataModule(
            dataset_root="/cache",
            download_dataset_root_uri=self._ROOT_URI,
            stream_from_r2=True,
            use_saved_mean_and_variance=False,
        )

        extra = module._dataset_extra_kwargs()

        assert "stats_file" not in extra
        assert "storage_options" in extra

    @pytest.mark.slow
    def test_prepare_data_streaming_fetches_only_stats_npz(
        self, local_r2_remote: Path, tmp_path: Path
    ) -> None:
        """Streaming hydration pulls just ``stats.npz`` — the splits stay in R2.

        :param local_r2_remote: Real rclone remote backed by the local filesystem.
        :param tmp_path: Holds the (initially absent) local cache root.
        """
        remote_prefix = local_r2_remote / "intermediate-data" / "data" / "run"
        remote_prefix.mkdir(parents=True)
        _write_stats(remote_prefix, mean=2.0, std=0.5)
        (remote_prefix / "train.lance").write_bytes(b"lance-bytes")
        dataset_root = tmp_path / "cache"

        module = LanceVSTDataModule(
            dataset_root=str(dataset_root),
            download_dataset_root_uri=self._ROOT_URI,
            stream_from_r2=True,
        )
        module.prepare_data()

        # The split stays remote; only stats.npz is fetched — and it is a real,
        # loadable .npz the dataset can normalize with, not opaque bytes.
        assert not (dataset_root / "train.lance").exists()
        with np.load(dataset_root / "stats.npz") as fetched:
            np.testing.assert_array_equal(
                fetched["mean"],
                np.full((_MEL_CHANNELS, _MEL_N_MELS, _MEL_N_FRAMES), 2.0, np.float32),
            )
