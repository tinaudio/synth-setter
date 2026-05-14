"""Tests for ``synth_setter.data.surge_datamodule``.

Covers :class:`SurgeXTDataset` (real and ``fake`` mode), the three samplers
(:class:`WithinChunkShuffledSampler`, :class:`ShuffledSampler`,
:class:`ShiftedBatchSampler`), and :class:`SurgeDataModule` (setup,
dataloader construction, teardown).

HDF5 fixtures are written to ``tmp_path`` with the layout
:class:`SurgeXTDataset` expects: ``audio``, ``mel_spec``, ``music2latent``,
``param_array`` plus a sibling ``stats.npz`` carrying ``mean`` and ``std``
for the mel spectrogram. The fixtures are tiny (a handful of rows) — the
goal is contract coverage on shapes, flags, and call routing, not numerical
ML behavior.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar
from unittest.mock import patch

import h5py
import hdf5plugin  # noqa: F401  side-effect import: registers HDF5 plugins for h5py
import numpy as np
import pytest
import torch

from synth_setter.data.surge_datamodule import (
    ShiftedBatchSampler,
    ShuffledSampler,
    SurgeDataModule,
    SurgeXTDataset,
    WithinChunkShuffledSampler,
)

# ``SurgeXTDataset.dataset_file`` is typed ``Optional[h5py.File]`` because fake
# mode leaves it ``None``. The narrowing helpers below let real-file tests
# access the handle without a per-call ``assert is not None`` cluttering each
# arrange/act/assert block.
_T = TypeVar("_T")


def _present(x: _T | None) -> _T:
    """Assert non-None and return — narrows ``Optional[T]`` to ``T`` for type checkers.

    :param x: Any value, typically a ``__getitem__`` result that may be ``None``.

    :returns: ``x`` itself, narrowed to its non-Optional type.
    :rtype: _T
    """
    assert x is not None
    return x


def _h5(ds: SurgeXTDataset) -> h5py.File:
    """Narrow :attr:`SurgeXTDataset.dataset_file` to :class:`h5py.File`.

    :param ds: A non-fake :class:`SurgeXTDataset` whose ``dataset_file`` is an open HDF5 handle.

    :returns: The open HDF5 file handle.
    :rtype: h5py.File
    """
    assert ds.dataset_file is not None
    return ds.dataset_file


def _h5_ds(ds: SurgeXTDataset, name: str) -> h5py.Dataset:
    """Look up ``name`` inside ``ds.dataset_file`` and narrow to :class:`h5py.Dataset`.

    :param ds: A non-fake :class:`SurgeXTDataset` with an open HDF5 handle.
    :param name: Dataset name inside the file (``audio``, ``mel_spec``, ``param_array``,
        ``music2latent``).

    :returns: The HDF5 Dataset object.
    :rtype: h5py.Dataset
    """
    obj = _h5(ds)[name]
    assert isinstance(obj, h5py.Dataset)
    return obj


def _close(ds: SurgeXTDataset) -> None:
    """Close ``ds.dataset_file`` when it is an open HDF5 handle.

    :param ds: A :class:`SurgeXTDataset`. No-op when the underlying file is ``None``.
    """
    if ds.dataset_file is not None:
        ds.dataset_file.close()


@contextlib.contextmanager
def _open_dataset(
    dataset_file: Path | str, **overrides: object
) -> Iterator[SurgeXTDataset]:
    """Build a real-file :class:`SurgeXTDataset` with ``ot=False``/no stats defaults; auto-close.

    The defaults (``batch_size=2``, ``use_saved_mean_and_variance=False``,
    ``ot=False``) are what every test in this file picks unless it is
    explicitly exercising the relevant flag. Tests opt out by passing
    ``ot=True`` / ``use_saved_mean_and_variance=True`` / ``batch_size=N`` /
    ``read_audio=True`` etc.

    :param dataset_file: Path to an HDF5 dataset file.
    :param **overrides: Keyword overrides forwarded to :class:`SurgeXTDataset`.

    :yields: A configured :class:`SurgeXTDataset` whose HDF5 handle is guaranteed closed on exit.
    :ytype: SurgeXTDataset
    """
    kwargs: dict[str, object] = {
        "batch_size": 2,
        "use_saved_mean_and_variance": False,
        "ot": False,
    }
    kwargs.update(overrides)
    ds = SurgeXTDataset(dataset_file=dataset_file, **kwargs)  # type: ignore[arg-type]
    try:
        yield ds
    finally:
        _close(ds)


# Smallest slice-axis sizes that still exercise the read paths in real-file
# tests. The audio "samples" axis is intentionally small so writing per-row
# fixtures stays fast.
_AUDIO_CHANNELS = 2
_AUDIO_SAMPLES_PER_ROW = 16
_MEL_CHANNELS = 2
_MEL_BANDS = 4
_MEL_FRAMES = 5
_M2L_BANDS = 4
_M2L_FRAMES = 3
_PARAM_LENGTH = 8

# Hard-coded shapes baked into ``SurgeXTDataset._get_fake_item``. Mirrored
# here so fake-mode tests reference a single source of truth and any
# source-side flip surfaces as one constant edit, not a sea of magic numbers.
_FAKE_LENGTH = 10000
_FAKE_MEL_SHAPE = (2, 128, 401)
_FAKE_M2L_SHAPE = (128, 42)
_FAKE_AUDIO_SAMPLES = 44100 * 4
_FAKE_PARAMS_DIM = 189


def _write_dataset_h5(
    path: Path,
    n_rows: int,
    *,
    seed: int = 0,
) -> None:
    """Write a minimal HDF5 dataset compatible with :class:`SurgeXTDataset`.

    :param path: File to create.
    :param n_rows: Number of rows along the leading axis of each dataset.
    :param seed: Seed for the numpy RNG that fills the arrays — different seeds across
        train/val/test files keep accidental cross-file collisions easy to spot in assertion
        failures.
    """
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "audio",
            data=rng.standard_normal(
                (n_rows, _AUDIO_CHANNELS, _AUDIO_SAMPLES_PER_ROW),
                dtype=np.float32,
            ),
        )
        f.create_dataset(
            "mel_spec",
            data=rng.standard_normal(
                (n_rows, _MEL_CHANNELS, _MEL_BANDS, _MEL_FRAMES),
                dtype=np.float32,
            ),
        )
        f.create_dataset(
            "music2latent",
            data=rng.standard_normal(
                (n_rows, _M2L_BANDS, _M2L_FRAMES),
                dtype=np.float32,
            ),
        )
        # ``param_array`` is rescaled by ``__getitem__`` (``x * 2 - 1``) when
        # ``rescale_params`` is True. Use [0, 1) values so the rescaled output
        # lands in [-1, 1) — easy to assert on.
        f.create_dataset(
            "param_array",
            data=rng.uniform(0.0, 1.0, size=(n_rows, _PARAM_LENGTH)).astype(np.float32),
        )


def _write_stats_npz(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Write a sibling ``stats.npz`` next to ``h5_path``.

    :param h5_path: Dataset file whose directory will receive ``stats.npz``.

    :returns: ``(mean, std)`` arrays written to disk, broadcastable against a single mel row.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    mean = np.full((_MEL_CHANNELS, _MEL_BANDS, _MEL_FRAMES), 0.25, dtype=np.float32)
    std = np.full((_MEL_CHANNELS, _MEL_BANDS, _MEL_FRAMES), 2.0, dtype=np.float32)
    stats_path = SurgeXTDataset.get_stats_file_path(h5_path)
    np.savez(stats_path, mean=mean, std=std)
    return mean, std


@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    """A directory holding ``train.h5``/``val.h5``/``test.h5`` and ``stats.npz``.

    :param tmp_path: pytest tmp dir fixture.
    :returns: The directory containing the fixtures.
    :rtype: Path
    """
    for name, seed in (("train", 1), ("val", 2), ("test", 3)):
        _write_dataset_h5(tmp_path / f"{name}.h5", n_rows=8, seed=seed)
    _write_stats_npz(tmp_path / "train.h5")
    return tmp_path


@pytest.fixture()
def single_h5(tmp_path: Path) -> Path:
    """A single HDF5 dataset file (no stats sidecar).

    :param tmp_path: pytest tmp dir fixture.
    :returns: Path to a freshly written HDF5 file with 8 rows.
    :rtype: Path
    """
    path = tmp_path / "train.h5"
    _write_dataset_h5(path, n_rows=8, seed=11)
    return path


# ---------------------------------------------------------------------------
# SurgeXTDataset — static helpers
# ---------------------------------------------------------------------------


class TestGetStatsFilePath:
    """``SurgeXTDataset.get_stats_file_path`` resolves the sibling stats file."""

    def test_returns_stats_npz_next_to_train_file(self, tmp_path: Path) -> None:
        """For ``/x/train.h5`` it returns ``/x/stats.npz``.

        :param tmp_path: pytest tmp dir fixture.
        """
        train = tmp_path / "train.h5"
        assert SurgeXTDataset.get_stats_file_path(train) == tmp_path / "stats.npz"

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """String inputs are converted via :class:`pathlib.Path`.

        :param tmp_path: pytest tmp dir fixture.
        """
        train = tmp_path / "train.h5"
        result = SurgeXTDataset.get_stats_file_path(str(train))
        assert result == tmp_path / "stats.npz"

    def test_resolves_relative_to_parent_directory(self, tmp_path: Path) -> None:
        """The stats file lives in the dataset file's parent, not next to the cwd.

        :param tmp_path: pytest tmp dir fixture.
        """
        nested = tmp_path / "nested" / "shards" / "val.h5"
        nested.parent.mkdir(parents=True)
        assert SurgeXTDataset.get_stats_file_path(nested) == nested.parent / "stats.npz"


# ---------------------------------------------------------------------------
# SurgeXTDataset — fake mode
# ---------------------------------------------------------------------------


class TestSurgeXTDatasetFakeMode:
    """``fake=True`` bypasses HDF5 entirely and yields random tensors."""

    def test_init_does_not_open_a_file(self) -> None:
        """``dataset_file`` stays ``None`` so no HDF5 handle is held."""
        ds = SurgeXTDataset(dataset_file="does-not-exist.h5", batch_size=4, fake=True)
        assert ds.dataset_file is None

    def test_length_is_fixed_in_fake_mode(self) -> None:
        """Fake-mode ``__len__`` ignores ``batch_size`` and returns the baked-in constant."""
        for bs in (1, 32, 1024):
            ds = SurgeXTDataset(dataset_file="anything", batch_size=bs, fake=True)
            assert len(ds) == _FAKE_LENGTH

    def test_default_flags_emit_audio_mel_params_and_noise(self) -> None:
        """Default fake item: ``audio`` (read_audio=False inverts), mel, params, noise; m2l absent.

        ``_get_fake_item`` gates the audio tensor on ``not self.read_audio`` —
        opposite to the mel/m2l flags. This test pins that current behaviour so
        any future flip is caught.
        """
        ds = SurgeXTDataset(dataset_file="x", batch_size=4, fake=True)
        item = ds[0]
        assert _present(item["mel_spec"]).shape == (4, *_FAKE_MEL_SHAPE)
        # In fake mode, ``params`` is a fixed-width tensor regardless of
        # the schema the real HDF5 file would use.
        assert _present(item["params"]).shape == (4, _FAKE_PARAMS_DIM)
        # Default ``rescale_params=True`` maps uniform[0,1) to [-1, 1).
        assert _present(item["params"]).min() >= -1.0
        assert _present(item["params"]).max() < 1.0
        assert _present(item["noise"]).shape == _present(item["params"]).shape
        # Default ``read_audio=False`` — the inverted gate produces a tensor.
        assert _present(item["audio"]).shape == (4, _AUDIO_CHANNELS, _FAKE_AUDIO_SAMPLES)
        assert item["m2l"] is None

    def test_read_audio_true_suppresses_audio_tensor(self) -> None:
        """Inverted audio gate: ``read_audio=True`` yields ``audio=None`` in fake mode.

        Mirror of ``test_default_flags_emit_audio_mel_params_and_noise`` — both
        directions of the inverted condition are pinned so the surprise doesn't
        regress silently.
        """
        ds = SurgeXTDataset(
            dataset_file="x",
            batch_size=2,
            fake=True,
            read_audio=True,
            read_mel=False,
        )
        item = ds[0]
        assert item["audio"] is None
        assert item["mel_spec"] is None

    def test_read_m2l_emits_m2l_tensor(self) -> None:
        """``read_m2l=True`` populates the m2l entry with the documented shape."""
        ds = SurgeXTDataset(
            dataset_file="x",
            batch_size=3,
            fake=True,
            read_mel=False,
            read_m2l=True,
        )
        item = ds[0]
        assert _present(item["m2l"]).shape == (3, *_FAKE_M2L_SHAPE)
        assert item["mel_spec"] is None

    def test_rescale_params_false_keeps_values_in_unit_interval(self) -> None:
        """Without rescaling, fake params stay in ``[0, 1)``."""
        ds = SurgeXTDataset(
            dataset_file="x",
            batch_size=8,
            fake=True,
            rescale_params=False,
        )
        item = ds[0]
        assert _present(item["params"]).min() >= 0.0
        assert _present(item["params"]).max() < 1.0

    def test_noise_matches_params_dtype_and_shape(self) -> None:
        """``noise`` is drawn from ``randn_like(params)`` — same shape and dtype."""
        ds = SurgeXTDataset(dataset_file="x", batch_size=4, fake=True)
        item = ds[0]
        assert _present(item["noise"]).shape == _present(item["params"]).shape
        assert _present(item["noise"]).dtype == _present(item["params"]).dtype


# ---------------------------------------------------------------------------
# SurgeXTDataset — real HDF5
# ---------------------------------------------------------------------------


class TestSurgeXTDatasetRealFile:
    """Behavior backed by an actual HDF5 dataset on disk."""

    def test_length_is_rows_floor_divided_by_batch_size(self, single_h5: Path) -> None:
        """``__len__`` reports complete-batches only: 8 rows // batch 3 == 2.

        :param single_h5: 8-row HDF5 fixture (fixture).
        """
        with _open_dataset(single_h5, batch_size=3, read_mel=False) as ds:
            assert len(ds) == 8 // 3

    def test_load_statistics_missing_stats_file_raises(self, single_h5: Path) -> None:
        """A missing ``stats.npz`` sibling triggers a clear ``FileNotFoundError``.

        :param single_h5: 8-row HDF5 fixture (fixture).
        """
        with pytest.raises(FileNotFoundError, match="stats.npz"):
            SurgeXTDataset(
                dataset_file=single_h5,
                batch_size=2,
                use_saved_mean_and_variance=True,
            )

    def test_load_statistics_populates_mean_and_std(self, single_h5: Path) -> None:
        """When ``stats.npz`` is present, ``mean`` and ``std`` attributes load.

        :param single_h5: 8-row HDF5 fixture (fixture).
        """
        mean, std = _write_stats_npz(single_h5)
        with _open_dataset(single_h5, use_saved_mean_and_variance=True) as ds:
            np.testing.assert_array_equal(ds.mean, mean)
            np.testing.assert_array_equal(ds.std, std)

    def test_use_saved_mean_and_variance_false_skips_stats(self, single_h5: Path) -> None:
        """Disabling stats loading leaves the class-default ``None`` in place.

        :param single_h5: 8-row HDF5 fixture (fixture).
        """
        with _open_dataset(single_h5) as ds:
            assert ds.mean is None
            assert ds.std is None


# ---------------------------------------------------------------------------
# SurgeXTDataset — _index_dataset slicing
# ---------------------------------------------------------------------------


class TestIndexDataset:
    """``_index_dataset`` dispatches on ``int``, ``tuple``, sequence, and
    ``repeat_first_batch``."""

    def test_int_index_slices_one_batch(self, single_h5: Path) -> None:
        """Integer ``idx`` returns rows ``[idx*bs : idx*bs + bs]``.

        :param single_h5: 8-row HDF5 fixture (fixture).
        """
        with _open_dataset(single_h5) as ds:
            pa = _h5_ds(ds, "param_array")
            full = pa[:]
            sliced = ds._index_dataset(pa, 1)
            np.testing.assert_array_equal(sliced, full[2:4])

    def test_tuple_index_slices_inclusive_of_start_exclusive_of_end(
        self, single_h5: Path
    ) -> None:
        """Tuple ``(start, end)`` slices the half-open range directly.

        :param single_h5: 8-row HDF5 fixture (fixture).
        """
        with _open_dataset(single_h5) as ds:
            pa = _h5_ds(ds, "param_array")
            full = pa[:]
            sliced = ds._index_dataset(pa, (1, 5))
            np.testing.assert_array_equal(sliced, full[1:5])

    def test_sequence_index_passes_through_to_h5py(self, single_h5: Path) -> None:
        """A list of indices delegates to ``ds[idx]``-style fancy indexing.

        :param single_h5: 8-row HDF5 fixture (fixture).
        """
        with _open_dataset(single_h5) as ds:
            pa = _h5_ds(ds, "param_array")
            full = pa[:]
            # h5py requires increasing index lists for fancy indexing.
            sliced = ds._index_dataset(pa, [0, 2, 5])
            np.testing.assert_array_equal(sliced, full[[0, 2, 5]])
            _close(ds)

    def test_repeat_first_batch_ignores_idx_and_returns_prefix(self, single_h5: Path) -> None:
        """With ``repeat_first_batch=True``, every call returns rows ``[:batch_size]``.

        :param single_h5: 8-row HDF5 fixture (fixture).
        """
        with _open_dataset(single_h5, repeat_first_batch=True) as ds:
            pa = _h5_ds(ds, "param_array")
            full = pa[:]
            np.testing.assert_array_equal(ds._index_dataset(pa, 0), full[:2])
            np.testing.assert_array_equal(ds._index_dataset(pa, 3), full[:2])
            np.testing.assert_array_equal(ds._index_dataset(pa, (5, 7)), full[:2])


# ---------------------------------------------------------------------------
# SurgeXTDataset — __getitem__
# ---------------------------------------------------------------------------


class TestGetItemMel:
    """``__getitem__`` with mel-only conditioning paths."""

    def test_emits_mel_params_and_noise_without_audio_or_m2l(self, single_h5: Path) -> None:
        """Default-style read: mel + params + noise; audio and m2l are ``None``.

        :param single_h5: 8-row HDF5 fixture (fixture).
        """
        with _open_dataset(single_h5) as ds:
            item = ds[0]
            assert _present(item["mel_spec"]).shape == (2, _MEL_CHANNELS, _MEL_BANDS, _MEL_FRAMES)
            assert _present(item["mel_spec"]).dtype == torch.float32
            assert _present(item["params"]).shape == (2, _PARAM_LENGTH)
            assert _present(item["noise"]).shape == _present(item["params"]).shape
            assert item["audio"] is None
            assert item["m2l"] is None

    def test_rescale_params_maps_unit_interval_to_signed_unit(self, single_h5: Path) -> None:
        """``rescale_params=True`` applies ``x * 2 - 1`` to ``param_array``.

        :param single_h5: 8-row HDF5 fixture (fixture).
        """
        with (
            _open_dataset(single_h5, rescale_params=True) as ds_rescaled,
            _open_dataset(single_h5, rescale_params=False) as ds_raw,
        ):
            expected = _present(ds_raw[0]["params"]) * 2 - 1
            torch.testing.assert_close(_present(ds_rescaled[0]["params"]), expected)

    def test_mel_is_normalized_when_stats_loaded(self, single_h5: Path) -> None:
        """When ``mean``/``std`` exist, mel is ``(x - mean) / std`` before tensor conversion.

        :param single_h5: 8-row HDF5 fixture (fixture).
        """
        mean, std = _write_stats_npz(single_h5)
        with _open_dataset(single_h5, use_saved_mean_and_variance=True) as ds:
            raw = _h5_ds(ds, "mel_spec")[0:2]
            expected = (raw - mean) / std
            np.testing.assert_allclose(_present(ds[0]["mel_spec"]).numpy(), expected, rtol=1e-6)

    def test_returned_tensors_are_contiguous(self, single_h5: Path) -> None:
        """``__getitem__`` calls ``.contiguous()`` on every non-``None`` tensor it returns.

        :param single_h5: 8-row HDF5 fixture (fixture).
        """
        with _open_dataset(single_h5, read_audio=True, read_m2l=True) as ds:
            item = ds[0]
            assert _present(item["mel_spec"]).is_contiguous()
            assert _present(item["m2l"]).is_contiguous()
            assert _present(item["params"]).is_contiguous()
            assert _present(item["noise"]).is_contiguous()
            assert _present(item["audio"]).is_contiguous()


class TestGetItemAudio:
    """Audio-read path: returns float32 tensor with the on-disk shape."""

    def test_emits_audio_when_read_audio_true(self, single_h5: Path) -> None:
        """``read_audio=True`` loads the audio dataset; mel/m2l respect their flags.

        :param single_h5: 8-row HDF5 fixture (fixture).
        """
        with _open_dataset(single_h5, read_audio=True, read_mel=False) as ds:
            item = ds[0]
            assert _present(item["audio"]).shape == (2, _AUDIO_CHANNELS, _AUDIO_SAMPLES_PER_ROW)
            assert _present(item["audio"]).dtype == torch.float32
            assert item["mel_spec"] is None


class TestGetItemM2l:
    """Music2Latent conditioning path."""

    def test_emits_m2l_when_read_m2l_true(self, single_h5: Path) -> None:
        """``read_m2l=True`` reads ``music2latent`` into a float32 tensor.

        :param single_h5: 8-row HDF5 fixture (fixture).
        """
        with _open_dataset(single_h5, read_mel=False, read_m2l=True) as ds:
            item = ds[0]
            assert _present(item["m2l"]).shape == (2, _M2L_BANDS, _M2L_FRAMES)
            assert _present(item["m2l"]).dtype == torch.float32
            assert item["mel_spec"] is None


class TestGetItemOptimalTransport:
    """``ot=True`` routes (noise, params[, mel, audio]) through ``_hungarian_match``."""

    def test_calls_hungarian_match_with_aligned_tensors(self, single_h5: Path) -> None:
        """The OT branch invokes ``_hungarian_match`` once, with the four expected tensors.

        :param single_h5: 8-row HDF5 fixture (fixture).
        """
        # Stub returns the four inputs unchanged so the rest of ``__getitem__``
        # keeps working — we only care that it was called with the right tuple.
        with (
            _open_dataset(single_h5, ot=True, read_audio=True) as ds,
            patch(
                "synth_setter.data.surge_datamodule._hungarian_match",
                side_effect=lambda n, p, m, a: (n, p, m, a),
            ) as patched,
        ):
            ds[0]
        patched.assert_called_once()
        args = patched.call_args.args
        # (noise, params, mel_spec, audio): four positional args.
        assert len(args) == 4
        assert args[0].shape == args[1].shape  # noise, params share shape
        assert args[2].shape[0] == 2  # mel: batch axis
        assert args[3].shape[0] == 2  # audio: batch axis

    def test_ot_false_leaves_noise_and_params_unmatched(self, single_h5: Path) -> None:
        """``ot=False`` never calls ``_hungarian_match``.

        :param single_h5: 8-row HDF5 fixture (fixture).
        """
        with (
            _open_dataset(single_h5) as ds,
            patch("synth_setter.data.surge_datamodule._hungarian_match") as patched,
        ):
            ds[0]
        patched.assert_not_called()


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


class TestWithinChunkShuffledSampler:
    """Yields batches that stay inside fixed-size groups to minimize cross-shard reads."""

    def test_len_reports_num_batches(self) -> None:
        """``__len__`` echoes ``num_batches``."""
        s = WithinChunkShuffledSampler(batch_size=4, num_batches=10, batches_per_group=3)
        assert len(s) == 10

    def test_iter_yields_num_batches_sorted_lists(self) -> None:
        """Exactly ``num_batches`` rows, each a sorted ``batch_size`` index list."""
        np.random.seed(0)
        s = WithinChunkShuffledSampler(batch_size=4, num_batches=6, batches_per_group=2)
        batches = list(iter(s))
        assert len(batches) == 6
        for row in batches:
            assert len(row) == 4
            assert row == sorted(row)
        # All indices fall inside ``[0, num_batches*batch_size)``.
        flat = [i for row in batches for i in row]
        assert min(flat) >= 0
        assert max(flat) < 6 * 4

    def test_indices_within_each_group_stay_inside_group_bounds(self) -> None:
        """Each ``batches_per_group`` batches share the same group window."""
        np.random.seed(0)
        bs = 4
        bpg = 3
        num_batches = bpg * 4  # exactly 4 full groups, no remainder
        s = WithinChunkShuffledSampler(
            batch_size=bs, num_batches=num_batches, batches_per_group=bpg
        )
        batches = list(iter(s))
        samples_per_group = bs * bpg
        # Bucket each batch by which group its smallest index falls into; every
        # batch must be fully contained in that group's window.
        for row in batches:
            group = row[0] // samples_per_group
            lower = group * samples_per_group
            upper = lower + samples_per_group
            assert all(lower <= i < upper for i in row), row

    def test_remainder_batches_form_a_smaller_final_group(self) -> None:
        """``num_batches % batches_per_group`` leftover batches become one short group."""
        np.random.seed(0)
        bs = 4
        bpg = 3
        num_batches = bpg * 2 + 1  # one leftover batch
        s = WithinChunkShuffledSampler(
            batch_size=bs, num_batches=num_batches, batches_per_group=bpg
        )
        batches = list(iter(s))
        assert len(batches) == num_batches
        # Total indices used spans only ``num_batches * batch_size`` slots.
        flat = sorted(i for row in batches for i in row)
        assert flat == list(range(num_batches * bs))


class TestShuffledSampler:
    """A plain permutation chunked into sorted batches."""

    def test_len_reports_num_batches(self) -> None:
        """``__len__`` matches the constructor arg."""
        assert len(ShuffledSampler(batch_size=8, num_batches=5)) == 5

    def test_iter_yields_full_permutation_in_sorted_batches(self) -> None:
        """Concatenated batches are a permutation of ``range(num_batches*batch_size)``."""
        np.random.seed(0)
        s = ShuffledSampler(batch_size=4, num_batches=6)
        batches = list(iter(s))
        assert len(batches) == 6
        flat = []
        for batch in batches:
            batch_list = list(batch)
            assert batch_list == sorted(batch_list)
            assert len(batch_list) == 4
            flat.extend(batch_list)
        assert sorted(flat) == list(range(6 * 4))


class TestShiftedBatchSampler:
    """Batch sampler over offset ``(start, end)`` windows."""

    def test_len_is_one_less_than_num_batches(self) -> None:
        """The trailing batch is dropped to make room for the random shift."""
        assert len(ShiftedBatchSampler(batch_size=4, num_batches=10)) == 9

    def test_iter_yields_pairs_offset_by_one_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each yielded pair is ``(i*bs + offset, (i+1)*bs + offset)`` for some shared offset.

        :param monkeypatch: pytest monkeypatch fixture used to pin ``random.randint``.
        """
        monkeypatch.setattr(random, "randint", lambda a, b: 1)
        np.random.seed(0)
        bs = 4
        num_batches = 5
        s = ShiftedBatchSampler(batch_size=bs, num_batches=num_batches)
        pairs = list(iter(s))
        assert len(pairs) == num_batches - 1
        for start, end in pairs:
            assert end - start == bs
            assert (start - 1) % bs == 0  # all starts share offset=1

    def test_iter_offset_zero_yields_aligned_pairs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``offset=0`` gives back the unshifted batch boundaries.

        :param monkeypatch: pytest monkeypatch fixture used to pin ``random.randint``.
        """
        monkeypatch.setattr(random, "randint", lambda a, b: 0)
        np.random.seed(0)
        bs = 3
        s = ShiftedBatchSampler(batch_size=bs, num_batches=4)
        pairs = list(iter(s))
        starts = sorted(start for start, _ in pairs)
        # batches 0..(num_batches-2) at boundary 0, bs, 2*bs:
        assert starts == [0, bs, 2 * bs]


# ---------------------------------------------------------------------------
# SurgeDataModule
# ---------------------------------------------------------------------------


class TestSurgeDataModuleInit:
    """``__init__`` stores every parameter as a public attribute for Lightning's ``hparams``."""

    def test_defaults_are_stored_verbatim(self, tmp_path: Path) -> None:
        """Defaults: batch_size=1024, ot=True, conditioning=mel, pin_memory=True, ...

        :param tmp_path: pytest tmp dir fixture, used only as a placeholder ``dataset_root``.
        """
        dm = SurgeDataModule(dataset_root=tmp_path)
        assert dm.dataset_root == tmp_path
        assert dm.use_saved_mean_and_variance is True
        assert dm.batch_size == 1024
        assert dm.ot is True
        assert dm.num_workers == 0
        assert dm.fake is False
        assert dm.repeat_first_batch is False
        assert dm.predict_file is None
        assert dm.conditioning == "mel"
        assert dm.pin_memory is True

    def test_string_dataset_root_becomes_path(self, tmp_path: Path) -> None:
        """A string ``dataset_root`` is normalized to :class:`pathlib.Path`.

        :param tmp_path: pytest tmp dir fixture.
        """
        dm = SurgeDataModule(dataset_root=str(tmp_path))
        assert isinstance(dm.dataset_root, Path)
        assert dm.dataset_root == tmp_path


class TestSurgeDataModuleSetupFake:
    """In ``fake`` mode, setup builds three :class:`SurgeXTDataset` instances; ``predict_dataset``
    is gated on ``predict_file``.

    Fake mode is used because it skips the HDF5 open path and lets us assert on the flag wiring
    without writing real fixtures for every conditioning combo.
    """

    def test_setup_creates_train_val_test_with_expected_flags(self, tmp_path: Path) -> None:
        """train: ot=True; val/test: ot=False; all three respect ``conditioning='mel'``.

        :param tmp_path: pytest tmp dir fixture, used as a placeholder ``dataset_root``.
        """
        dm = SurgeDataModule(
            dataset_root=tmp_path,
            fake=True,
            batch_size=4,
            ot=True,
            conditioning="mel",
        )
        dm.setup()
        for split in (dm.train_dataset, dm.val_dataset, dm.test_dataset):
            assert isinstance(split, SurgeXTDataset)
            assert split.read_mel is True
            assert split.read_m2l is False
            assert split.batch_size == 4
        assert dm.train_dataset.ot is True
        assert dm.val_dataset.ot is False
        assert dm.test_dataset.ot is False

    def test_predict_dataset_is_none_when_predict_file_omitted(self, tmp_path: Path) -> None:
        """Without ``predict_file``, ``predict_dataset`` stays ``None`` after setup.

        :param tmp_path: pytest tmp dir fixture, used as a placeholder ``dataset_root``.
        """
        dm = SurgeDataModule(dataset_root=tmp_path, fake=True, batch_size=4)
        dm.setup()
        assert dm.predict_dataset is None

    def test_predict_dataset_set_when_predict_file_given(self, tmp_path: Path) -> None:
        """``predict_file`` builds a predict dataset with ``read_audio=True``.

        :param tmp_path: pytest tmp dir fixture, used as a placeholder ``dataset_root``.
        """
        dm = SurgeDataModule(
            dataset_root=tmp_path,
            fake=True,
            batch_size=2,
            predict_file="anything.h5",
        )
        dm.setup()
        assert isinstance(dm.predict_dataset, SurgeXTDataset)
        assert dm.predict_dataset.read_audio is True
        assert dm.predict_dataset.ot is False

    @pytest.mark.parametrize(
        ("conditioning", "expected_mel", "expected_m2l"),
        [("mel", True, False), ("m2l", False, True)],
    )
    def test_conditioning_toggles_read_flags(
        self,
        tmp_path: Path,
        conditioning: str,
        expected_mel: bool,
        expected_m2l: bool,
    ) -> None:
        """``conditioning`` switches ``read_mel``/``read_m2l`` across all datasets.

        :param tmp_path: pytest tmp dir fixture, used as a placeholder ``dataset_root``.
        :param conditioning: Parametrize value (``"mel"`` or ``"m2l"``).
        :param expected_mel: Expected ``read_mel`` flag on each split's dataset.
        :param expected_m2l: Expected ``read_m2l`` flag on each split's dataset.
        """
        dm = SurgeDataModule(
            dataset_root=tmp_path,
            fake=True,
            batch_size=2,
            conditioning=conditioning,  # type: ignore[arg-type]
        )
        dm.setup()
        for split in (dm.train_dataset, dm.val_dataset, dm.test_dataset):
            assert split.read_mel is expected_mel
            assert split.read_m2l is expected_m2l


class TestSurgeDataModuleSetupReal:
    """End-to-end setup over a real on-disk dataset_root, including teardown."""

    def test_setup_opens_three_h5_files_and_teardown_closes_them(
        self, dataset_root: Path
    ) -> None:
        """All three split files are real h5py handles after setup, closed after teardown.

        :param dataset_root: Directory holding train/val/test HDF5 files (fixture).
        """
        dm = SurgeDataModule(
            dataset_root=dataset_root,
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=False,
            pin_memory=False,
        )
        dm.setup()
        for split in (dm.train_dataset, dm.val_dataset, dm.test_dataset):
            assert isinstance(split.dataset_file, h5py.File)
            assert split.dataset_file
        dm.teardown()
        # ``bool(h5py.File)`` is False once closed.
        for split in (dm.train_dataset, dm.val_dataset, dm.test_dataset):
            assert not split.dataset_file


class TestSurgeDataModuleDataloaders:
    """Each ``*_dataloader`` returns a :class:`torch.utils.data.DataLoader` for its split."""

    def _setup_fake(self, tmp_path: Path, **overrides: object) -> SurgeDataModule:
        """Build a fake-mode datamodule with sensible defaults plus overrides.

        :param tmp_path: Directory used as ``dataset_root`` (placeholder; fake mode does not open
            HDF5).
        :param **overrides: Keyword overrides forwarded to :class:`SurgeDataModule`.

        :returns: A datamodule already through :meth:`SurgeDataModule.setup`.
        :rtype: SurgeDataModule
        """
        kwargs: dict[str, object] = {
            "dataset_root": tmp_path,
            "fake": True,
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
        }
        kwargs.update(overrides)
        dm = SurgeDataModule(**kwargs)  # type: ignore[arg-type]
        dm.setup()
        return dm

    def test_train_dataloader_uses_shifted_batch_sampler(self, tmp_path: Path) -> None:
        """The train loader is wired to :class:`ShiftedBatchSampler` with the configured batch
        size.

        :param tmp_path: pytest tmp dir fixture, used as a placeholder ``dataset_root``.
        """
        dm = self._setup_fake(tmp_path)
        loader = dm.train_dataloader()
        assert isinstance(loader, torch.utils.data.DataLoader)
        assert isinstance(loader.sampler, ShiftedBatchSampler)
        assert loader.sampler.batch_size == 4
        assert loader.num_workers == 0
        assert loader.pin_memory is False
        # ``batch_size=None`` because the dataset already returns full batches.
        assert loader.batch_size is None

    @pytest.mark.parametrize("split", ["val_dataloader", "test_dataloader"])
    def test_eval_dataloaders_disable_shuffle(self, tmp_path: Path, split: str) -> None:
        """Val/test loaders are deterministic — no shuffle, no custom sampler.

        :param tmp_path: pytest tmp dir fixture, used as a placeholder ``dataset_root``.
        :param split: Parametrized loader attribute name on the datamodule.
        """
        dm = self._setup_fake(tmp_path)
        loader = getattr(dm, split)()
        assert isinstance(loader, torch.utils.data.DataLoader)
        assert loader.batch_size is None
        # Default ``shuffle=False`` is reflected in a ``SequentialSampler``.
        assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)

    def test_predict_dataloader_returns_dataloader_when_predict_file_set(
        self, tmp_path: Path
    ) -> None:
        """Predict loader is built only when ``predict_file`` is configured.

        :param tmp_path: pytest tmp dir fixture, used as a placeholder ``dataset_root``.
        """
        dm = self._setup_fake(tmp_path, predict_file="something.h5")
        loader = dm.predict_dataloader()
        assert isinstance(loader, torch.utils.data.DataLoader)
        assert loader.batch_size is None

    def test_dataloaders_propagate_num_workers_and_pin_memory(self, tmp_path: Path) -> None:
        """``num_workers`` and ``pin_memory`` flow through to each constructed loader.

        :param tmp_path: pytest tmp dir fixture, used as a placeholder ``dataset_root``.
        """
        dm = self._setup_fake(tmp_path, num_workers=2, pin_memory=True)
        for split in (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()):
            assert split.num_workers == 2
            assert split.pin_memory is True
