"""Behavioral tests for :mod:`synth_setter.data.surge_datamodule`.

Covers the four public symbols exposed by the module:

* :class:`SurgeXTDataset` — both the ``fake`` synthetic path and the real
  HDF5-backed path, with the three boolean read flags (``read_audio`` /
  ``read_mel`` / ``read_m2l``), OT matching toggle, parameter rescaling, the
  ``repeat_first_batch`` mode, and the sibling-``stats.npz`` loader.
* :class:`WithinChunkShuffledSampler`, :class:`ShuffledSampler`,
  :class:`ShiftedBatchSampler` — three batch-index samplers with distinct
  shuffle/strict-locality invariants.
* :class:`SurgeDataModule` — Lightning ``setup`` / dataloader / ``teardown``
  wiring, including the ``conditioning`` mel-vs-m2l switch and the
  ``predict_file`` optional split.

HDF5 fixtures are tiny (a handful of rows, ~10-element mel/audio axes) — the
goal is contract coverage on shapes, flags, and call routing, not numerical
ML behavior. The mel/audio dimensions deliberately do NOT match production
shapes (production uses ``(2, 128, 401)`` mel and ``2 * 44100 * 4`` audio
samples); shrinking them keeps the fixture under a few KB on disk while still
exercising every code branch in ``__getitem__`` and ``_index_dataset``.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import h5py
import hdf5plugin  # noqa: F401  side-effect import: registers HDF5 plugins so h5py can read Blosc2 filters
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

_AUDIO_CHANNELS = 2
_AUDIO_SAMPLES = 16
_MEL_CHANNELS = 2
_MEL_N_MELS = 4
_MEL_N_FRAMES = 5
_M2L_DIM_1 = 6
_M2L_DIM_2 = 7
_NUM_PARAMS = 11

_ALL_TENSOR_KEYS = ("audio", "mel_spec", "m2l", "params", "noise")


@contextlib.contextmanager
def _set_up_module(**kwargs: object) -> Iterator[SurgeDataModule]:  # noqa: DOC101,DOC103,DOC404
    """Construct a ``SurgeDataModule`` from ``**kwargs``, ``setup``, yield, then ``teardown``.

    Encapsulates the setup/teardown try/finally pattern every
    ``TestSurgeDataModule`` test needs so a forgotten ``teardown`` can't leak
    h5py handles into the next test. Also closes the ``predict_dataset``'s
    h5py file when one was opened, since ``teardown`` only closes the three
    train/val/test splits.

    :yields: The set-up datamodule for assertion work inside the ``with`` block.
    """
    module = SurgeDataModule(**kwargs)  # type: ignore[arg-type]
    module.setup()
    try:
        yield module
    finally:
        # SurgeDataModule.teardown() blindly closes train/val/test dataset_file
        # handles; in fake mode those are None and the close call would raise.
        # Tests that exercise teardown's real-mode behavior do so directly
        # (see test_teardown_closes_open_h5_handles) — skip it here when the
        # caller asked for fake mode.
        if not module.fake:
            module.teardown()
        predict_dataset = module.predict_dataset
        if predict_dataset is not None and predict_dataset.dataset_file is not None:
            predict_dataset.dataset_file.close()


def _read_h5_slice(h5_path: Path, key: str, idx: object) -> np.ndarray:
    """Open ``h5_path`` and return ``f[key][idx]`` narrowed to :class:`numpy.ndarray`.

    Pyright sees ``h5py.File.__getitem__`` as returning ``Group | Dataset | Datatype``,
    so plain ``f["param_array"][1:5]`` flags a ``reportIndexIssue``. Tests use this
    helper to centralize the cast and keep arrange blocks readable.

    :param h5_path: Path to a closed HDF5 file the helper will open read-only.
    :param key: Dataset name inside the file (``param_array``, ``mel_spec``, ...).
    :param idx: Any h5py-supported index (slice, list, tuple).

    :return: The raw NumPy slice; the caller is free to compare or transform it.
    :rtype: np.ndarray
    """
    with h5py.File(h5_path, "r") as f:
        dataset = f[key]
        assert isinstance(dataset, h5py.Dataset)
        return dataset[idx]


def _unwrap(maybe_tensor: torch.Tensor | None) -> torch.Tensor:
    """Assert ``maybe_tensor`` is non-None and return it as a ``torch.Tensor``.

    The dataset's ``__getitem__`` returns a dict where every value is typed
    ``Tensor | None`` (mel/audio/m2l really are optional depending on read
    flags; params/noise are unconditionally populated but share the same
    dict type). Tests that exercise the populated keys go through this
    helper so pyright narrows the type once, instead of every test
    inlining ``assert x is not None``.

    :param maybe_tensor: The dict value to narrow.

    :return: The same tensor, now typed as non-None.
    :rtype: torch.Tensor
    """
    assert maybe_tensor is not None
    return maybe_tensor


def _write_h5_shard(
    path: Path,
    num_rows: int,
    *,
    params_seed: int = 0,
    mel_fill: float | None = None,
) -> None:
    """Write a tiny HDF5 shard with the four datasets ``SurgeXTDataset`` reads.

    :param path: Output file path.
    :param num_rows: Number of rows along the first axis of every dataset.
    :param params_seed: Seed for the per-row parameter array so different
        splits get distinguishable values when needed.
    :param mel_fill: When set, fill the ``mel_spec`` dataset with this constant
        instead of random values — used by the normalization test to make
        ``(mel - mean) / std`` produce a predictable result.
    """
    rng = np.random.default_rng(params_seed)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "audio",
            data=rng.standard_normal((num_rows, _AUDIO_CHANNELS, _AUDIO_SAMPLES)).astype(
                np.float32
            ),
        )
        if mel_fill is None:
            mel_data = rng.standard_normal((num_rows, _MEL_CHANNELS, _MEL_N_MELS, _MEL_N_FRAMES))
        else:
            mel_data = np.full(
                (num_rows, _MEL_CHANNELS, _MEL_N_MELS, _MEL_N_FRAMES),
                fill_value=mel_fill,
                dtype=np.float32,
            )
        f.create_dataset("mel_spec", data=mel_data.astype(np.float32))
        f.create_dataset(
            "music2latent",
            data=rng.standard_normal((num_rows, _M2L_DIM_1, _M2L_DIM_2)).astype(np.float32),
        )
        # params in [0, 1) so the rescale_params=True branch lands in [-1, 1).
        f.create_dataset(
            "param_array",
            data=rng.random((num_rows, _NUM_PARAMS)).astype(np.float32),
        )


def _write_stats(
    dataset_dir: Path,
    *,
    mean: float = 0.0,
    std: float = 1.0,
) -> Path:
    """Write a sibling ``stats.npz`` whose mean/std broadcast against ``mel_spec``.

    :param dataset_dir: Directory holding ``*.h5``; ``stats.npz`` is written
        alongside.
    :param mean: Scalar mean — broadcast against every mel-spec element.
    :param std: Scalar std — broadcast against every mel-spec element.

    :return: Path to the written ``stats.npz``.
    :rtype: Path
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
    """Build a ``dataset_root`` directory with ``train/val/test.h5`` + ``stats.npz``.

    :param tmp_path: Per-test tmpdir.
    :return: Path to the populated dataset root directory.
    :rtype: Path
    """
    root = tmp_path / "data"
    root.mkdir()
    _write_h5_shard(root / "train.h5", num_rows=8, params_seed=1)
    _write_h5_shard(root / "val.h5", num_rows=8, params_seed=2)
    _write_h5_shard(root / "test.h5", num_rows=8, params_seed=3)
    _write_stats(root)
    return root


@pytest.fixture
def single_h5(tmp_path: Path) -> Path:
    """Write a single ``train.h5`` + sibling ``stats.npz`` for SurgeXTDataset-only tests.

    :param tmp_path: Per-test tmpdir.

    :return: Path to the written ``train.h5`` file.
    :rtype: Path
    """
    h5_path = tmp_path / "train.h5"
    _write_h5_shard(h5_path, num_rows=8)
    _write_stats(tmp_path)
    return h5_path


# --------------------------------------------------------------------------- #
# SurgeXTDataset — fake mode                                                  #
# --------------------------------------------------------------------------- #


class TestSurgeXTDatasetFakeMode:
    """``fake=True`` skips HDF5 entirely and returns randomly-generated tensors."""

    def test_fake_mode_does_not_open_h5_file(self, tmp_path: Path) -> None:  # noqa: DOC101,DOC103
        """``fake=True`` accepts a nonexistent path because ``__init__`` never reads it."""
        missing = tmp_path / "does-not-exist.h5"
        dataset = SurgeXTDataset(missing, batch_size=4, fake=True)
        assert dataset.dataset_file is None

    def test_fake_mode_len_is_fixed_constant(self) -> None:
        """``__len__`` returns the documented 10000 in fake mode regardless of batch_size."""
        small = SurgeXTDataset("ignored", batch_size=1, fake=True)
        large = SurgeXTDataset("ignored", batch_size=8192, fake=True)
        assert len(small) == 10000
        assert len(large) == 10000

    def test_fake_mode_default_flags_populate_mel_audio_params_noise(self) -> None:
        """With default flags ``audio``/``mel_spec``/``params``/``noise`` are populated; ``m2l`` is
        None.

        ``audio`` is populated here even though ``read_audio`` defaults to
        False — see ``test_fake_mode_read_audio_true_returns_none_audio``
        for the inverse pin on the asymmetric flag.
        """
        dataset = SurgeXTDataset("ignored", batch_size=3, fake=True)
        item = dataset[0]
        assert item["m2l"] is None
        assert _unwrap(item["audio"]).shape == (3, 2, 44100 * 4)
        assert _unwrap(item["mel_spec"]).shape == (3, 2, 128, 401)
        assert _unwrap(item["params"]).shape == (3, 189)
        assert _unwrap(item["noise"]).shape == (3, 189)

    def test_fake_mode_read_audio_true_returns_none_audio(self) -> None:
        """Pin the asymmetric fake-mode contract: ``read_audio=True`` returns audio=None.

        This is the *current* (and surprising) behavior of ``_get_fake_item``:
        the audio ternary inverts the flag (``... if not self.read_audio else None``)
        so the real-mode contract ``read_audio=True -> audio populated`` does
        not hold in fake mode. Pinned here so future changes flip the test
        either way — the asymmetry is intentional to surface in review.
        """
        dataset = SurgeXTDataset("ignored", batch_size=2, fake=True, read_audio=True)
        item = dataset[0]
        assert item["audio"] is None

    def test_fake_mode_read_m2l_returns_m2l_tensor(self) -> None:
        """``read_m2l=True`` populates the ``m2l`` slot with the documented shape."""
        dataset = SurgeXTDataset("ignored", batch_size=2, fake=True, read_m2l=True)
        assert _unwrap(dataset[0]["m2l"]).shape == (2, 128, 42)

    def test_fake_mode_read_mel_false_returns_none_mel(self) -> None:
        """``read_mel=False`` drops the ``mel_spec`` slot to ``None``."""
        dataset = SurgeXTDataset("ignored", batch_size=2, fake=True, read_mel=False)
        item = dataset[0]
        assert item["mel_spec"] is None

    def test_fake_mode_rescale_params_maps_into_minus_one_to_one(self) -> None:
        """``rescale_params=True`` rescales ``torch.rand`` from [0, 1) into [-1, 1)."""
        # read_mel=False + read_audio=True skips the ~190 MB mel/audio
        # allocations that fake mode would otherwise build at batch_size=128;
        # this test only inspects params.
        dataset = SurgeXTDataset(
            "ignored",
            batch_size=128,
            fake=True,
            rescale_params=True,
            read_mel=False,
            read_audio=True,
        )
        params = _unwrap(dataset[0]["params"])
        assert params.min().item() >= -1.0
        assert params.max().item() < 1.0
        # With 128*189 samples from rand()*2 - 1 we will reach into [-1, 0) and
        # [0, 1) with vanishing probability of staying on one side — pin the
        # rescaling behavior, not just the bounds.
        assert params.min().item() < 0.0
        assert params.max().item() > 0.0

    def test_fake_mode_no_rescale_params_stays_in_zero_to_one(self) -> None:
        """``rescale_params=False`` leaves params in ``torch.rand``'s native [0, 1) range."""
        dataset = SurgeXTDataset(
            "ignored",
            batch_size=128,
            fake=True,
            rescale_params=False,
            read_mel=False,
            read_audio=True,
        )
        params = _unwrap(dataset[0]["params"])
        assert params.min().item() >= 0.0
        assert params.max().item() < 1.0

    def test_fake_mode_noise_matches_param_shape(self) -> None:
        """``noise`` is allocated with ``torch.randn_like(params)``, so shapes match."""
        dataset = SurgeXTDataset("ignored", batch_size=5, fake=True)
        item = dataset[0]
        assert _unwrap(item["noise"]).shape == _unwrap(item["params"]).shape

    def test_fake_mode_returns_full_key_set(self) -> None:
        """The returned dict always exposes all five keys (some may be ``None``)."""
        dataset = SurgeXTDataset("ignored", batch_size=2, fake=True)
        item = dataset[0]
        assert set(item.keys()) == set(_ALL_TENSOR_KEYS)


# --------------------------------------------------------------------------- #
# SurgeXTDataset — real HDF5 mode                                             #
# --------------------------------------------------------------------------- #


class TestSurgeXTDatasetH5Mode:
    """HDF5-backed path: indexing, type conversion, OT routing, normalization."""

    def test_len_equals_num_rows_floor_divided_by_batch_size(self, single_h5: Path) -> None:  # noqa: DOC101,DOC103
        """``__len__`` uses integer division — 8 rows / batch_size 3 == 2 batches."""
        dataset = SurgeXTDataset(single_h5, batch_size=3, ot=False)
        assert len(dataset) == 8 // 3

    def test_getitem_int_returns_batch_size_slice(self, single_h5: Path) -> None:  # noqa: DOC101,DOC103
        """Integer index ``i`` reads rows ``[i*B : i*B+B]`` from each dataset."""
        # ``rescale_params=True`` (default) applies ``x * 2 - 1`` to params.
        expected_params = _read_h5_slice(single_h5, "param_array", slice(2, 4)) * 2 - 1
        expected_mel = _read_h5_slice(single_h5, "mel_spec", slice(2, 4))
        dataset = SurgeXTDataset(
            single_h5, batch_size=2, ot=False, use_saved_mean_and_variance=False
        )
        item = dataset[1]
        torch.testing.assert_close(
            _unwrap(item["params"]), torch.from_numpy(expected_params).to(torch.float32)
        )
        torch.testing.assert_close(
            _unwrap(item["mel_spec"]), torch.from_numpy(expected_mel).to(torch.float32)
        )

    def test_getitem_tuple_returns_explicit_slice(self, single_h5: Path) -> None:  # noqa: DOC101,DOC103
        """A 2-tuple index ``(lo, hi)`` selects rows ``[lo:hi]`` directly."""
        expected_params = _read_h5_slice(single_h5, "param_array", slice(1, 5)) * 2 - 1
        expected_mel = _read_h5_slice(single_h5, "mel_spec", slice(1, 5))
        dataset = SurgeXTDataset(
            single_h5, batch_size=2, ot=False, use_saved_mean_and_variance=False
        )
        item = dataset[(1, 5)]
        torch.testing.assert_close(
            _unwrap(item["params"]), torch.from_numpy(expected_params).to(torch.float32)
        )
        torch.testing.assert_close(
            _unwrap(item["mel_spec"]), torch.from_numpy(expected_mel).to(torch.float32)
        )

    def test_getitem_sequence_falls_through_to_ds_fancy_indexing(self, single_h5: Path) -> None:  # noqa: DOC101,DOC103
        """A non-int / non-2-tuple index falls through to ``ds[idx]`` fancy indexing."""
        expected_params = _read_h5_slice(single_h5, "param_array", [0, 2, 4]) * 2 - 1
        expected_mel = _read_h5_slice(single_h5, "mel_spec", [0, 2, 4])
        dataset = SurgeXTDataset(
            single_h5, batch_size=2, ot=False, use_saved_mean_and_variance=False
        )
        item = dataset[[0, 2, 4]]
        torch.testing.assert_close(
            _unwrap(item["params"]), torch.from_numpy(expected_params).to(torch.float32)
        )
        torch.testing.assert_close(
            _unwrap(item["mel_spec"]), torch.from_numpy(expected_mel).to(torch.float32)
        )

    def test_repeat_first_batch_ignores_idx(self, single_h5: Path) -> None:  # noqa: DOC101,DOC103
        """``repeat_first_batch=True`` always returns the first ``batch_size`` rows."""
        expected_params = _read_h5_slice(single_h5, "param_array", slice(None, 3)) * 2 - 1
        expected_mel = _read_h5_slice(single_h5, "mel_spec", slice(None, 3))
        dataset = SurgeXTDataset(
            single_h5,
            batch_size=3,
            ot=False,
            use_saved_mean_and_variance=False,
            repeat_first_batch=True,
        )
        first = dataset[0]
        later = dataset[2]
        # Both indices return the same prefix slice from disk — pin the actual
        # values, not just inter-call equality, so a regression that returns
        # any constant slice is still caught.
        for batch in (first, later):
            torch.testing.assert_close(
                _unwrap(batch["params"]),
                torch.from_numpy(expected_params).to(torch.float32),
            )
            torch.testing.assert_close(
                _unwrap(batch["mel_spec"]),
                torch.from_numpy(expected_mel).to(torch.float32),
            )

    def test_returned_tensors_are_float32(self, single_h5: Path) -> None:  # noqa: DOC101,DOC103
        """All numeric tensors come back as ``torch.float32`` for AMP compatibility."""
        dataset = SurgeXTDataset(
            single_h5,
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

    def test_returned_tensors_are_contiguous(self, single_h5: Path) -> None:  # noqa: DOC101,DOC103
        """Every populated tensor is ``.contiguous()`` so downstream cuda copies are cheap."""
        dataset = SurgeXTDataset(
            single_h5,
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=False,
            read_audio=True,
            read_mel=True,
            read_m2l=True,
        )
        item = dataset[0]
        for key in _ALL_TENSOR_KEYS:
            assert _unwrap(item[key]).is_contiguous(), f"{key} not contiguous"

    def test_read_audio_false_returns_none_audio(self, single_h5: Path) -> None:  # noqa: DOC101,DOC103
        """``read_audio=False`` (default) leaves the ``audio`` slot at ``None``."""
        dataset = SurgeXTDataset(
            single_h5, batch_size=2, ot=False, use_saved_mean_and_variance=False
        )
        item = dataset[0]
        assert item["audio"] is None

    def test_read_mel_false_returns_none_mel(self, single_h5: Path) -> None:  # noqa: DOC101,DOC103
        """``read_mel=False`` drops the ``mel_spec`` slot, even with stats on disk."""
        dataset = SurgeXTDataset(single_h5, batch_size=2, ot=False, read_mel=False)
        item = dataset[0]
        assert item["mel_spec"] is None

    def test_read_m2l_true_returns_m2l_tensor(self, single_h5: Path) -> None:  # noqa: DOC101,DOC103
        """``read_m2l=True`` reads the ``music2latent`` dataset under the ``m2l`` key."""
        dataset = SurgeXTDataset(
            single_h5,
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=False,
            read_m2l=True,
        )
        assert _unwrap(dataset[0]["m2l"]).shape == (2, _M2L_DIM_1, _M2L_DIM_2)

    def test_rescale_params_centers_to_minus_one_to_one(self, single_h5: Path) -> None:  # noqa: DOC101,DOC103
        """``rescale_params=True`` applies ``p * 2 - 1`` element-wise before tensor conversion."""
        dataset_raw = SurgeXTDataset(
            single_h5,
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=False,
            rescale_params=False,
        )
        dataset_rescaled = SurgeXTDataset(
            single_h5,
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=False,
            rescale_params=True,
        )
        raw = _unwrap(dataset_raw[0]["params"])
        rescaled = _unwrap(dataset_rescaled[0]["params"])
        assert torch.allclose(rescaled, raw * 2 - 1)

    def test_mel_spec_normalized_with_loaded_stats(self, tmp_path: Path) -> None:  # noqa: DOC101,DOC103
        """When ``stats.npz`` is loaded, mel is returned as ``(mel - mean) / std``."""
        h5_path = tmp_path / "train.h5"
        _write_h5_shard(h5_path, num_rows=4, mel_fill=3.0)
        _write_stats(tmp_path, mean=1.0, std=2.0)
        dataset = SurgeXTDataset(
            h5_path,
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=True,
        )
        mel = _unwrap(dataset[0]["mel_spec"])
        expected = (3.0 - 1.0) / 2.0
        assert torch.allclose(mel, torch.full_like(mel, expected))

    def test_no_stats_load_when_disabled(self, tmp_path: Path) -> None:  # noqa: DOC101,DOC103
        """``use_saved_mean_and_variance=False`` skips the npz read, even if it's missing."""
        h5_path = tmp_path / "train.h5"
        _write_h5_shard(h5_path, num_rows=4)
        dataset = SurgeXTDataset(
            h5_path,
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=False,
        )
        assert dataset.mean is None
        assert dataset.std is None

    def test_missing_stats_file_raises_file_not_found(self, tmp_path: Path) -> None:  # noqa: DOC101,DOC103
        """``use_saved_mean_and_variance=True`` with no sibling ``stats.npz`` errors clearly."""
        h5_path = tmp_path / "train.h5"
        _write_h5_shard(h5_path, num_rows=4)
        with pytest.raises(FileNotFoundError, match="stats.npz"):
            SurgeXTDataset(h5_path, batch_size=2, ot=False, use_saved_mean_and_variance=True)

    def test_get_stats_file_path_is_sibling_of_dataset(self, tmp_path: Path) -> None:  # noqa: DOC101,DOC103
        """The static helper returns ``parent_dir / 'stats.npz'`` for any input layout."""
        # str input
        assert SurgeXTDataset.get_stats_file_path(str(tmp_path / "train.h5")) == (
            tmp_path / "stats.npz"
        )
        # Path input
        assert SurgeXTDataset.get_stats_file_path(tmp_path / "val.h5") == tmp_path / "stats.npz"
        # Nested path
        nested = tmp_path / "shard0" / "data.h5"
        assert SurgeXTDataset.get_stats_file_path(nested) == tmp_path / "shard0" / "stats.npz"

    def test_resolves_relative_to_parent_directory(self, tmp_path: Path) -> None:  # noqa: DOC101,DOC103
        """The stats file lives in the dataset file's parent, not next to the cwd."""
        nested = tmp_path / "nested" / "shards" / "val.h5"
        nested.parent.mkdir(parents=True)
        assert SurgeXTDataset.get_stats_file_path(nested) == nested.parent / "stats.npz"

    def test_load_statistics_populates_mean_and_std(self, tmp_path: Path) -> None:  # noqa: DOC101,DOC103
        """When ``stats.npz`` is present, ``mean`` and ``std`` attributes load from disk."""
        h5_path = tmp_path / "train.h5"
        _write_h5_shard(h5_path, num_rows=4)
        stats_path = _write_stats(tmp_path, mean=0.25, std=2.0)
        with np.load(stats_path) as stats:
            expected_mean = stats["mean"]
            expected_std = stats["std"]
        dataset = SurgeXTDataset(
            h5_path,
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=True,
        )
        try:
            np.testing.assert_array_equal(dataset.mean, expected_mean)
            np.testing.assert_array_equal(dataset.std, expected_std)
        finally:
            if dataset.dataset_file is not None:
                dataset.dataset_file.close()

    def test_no_ot_does_not_call_hungarian_match(self, single_h5: Path) -> None:  # noqa: DOC101,DOC103
        """``ot=False`` short-circuits before ``_hungarian_match`` is invoked."""
        with patch("synth_setter.data.surge_datamodule._hungarian_match") as mock_match:
            dataset = SurgeXTDataset(
                single_h5,
                batch_size=2,
                ot=False,
                use_saved_mean_and_variance=False,
            )
            _ = dataset[0]
        mock_match.assert_not_called()

    def test_ot_true_calls_hungarian_match_with_all_four_tensors(self, single_h5: Path) -> None:  # noqa: DOC101,DOC103
        """``ot=True`` routes through ``_hungarian_match(noise, params, mel_spec, audio)``."""
        with patch(
            "synth_setter.data.surge_datamodule._hungarian_match",
            side_effect=lambda noise, params, *args: (noise, params, *args),
        ) as mock_match:
            dataset = SurgeXTDataset(
                single_h5,
                batch_size=2,
                ot=True,
                use_saved_mean_and_variance=False,
                read_audio=True,
            )
            _ = dataset[0]
        mock_match.assert_called_once()
        positional = mock_match.call_args.args
        assert len(positional) == 4
        # Positional contract: (noise, params, mel_spec, audio). Pin each slot's
        # shape so a regression that swaps positions is caught — bare arity does
        # not distinguish `(noise, params, audio, mel_spec)` from the correct order.
        noise, params, mel_spec, audio = positional
        assert isinstance(noise, torch.Tensor) and noise.shape == (2, _NUM_PARAMS)
        assert isinstance(params, torch.Tensor) and params.shape == (2, _NUM_PARAMS)
        assert isinstance(mel_spec, torch.Tensor) and mel_spec.shape == (
            2,
            _MEL_CHANNELS,
            _MEL_N_MELS,
            _MEL_N_FRAMES,
        )
        assert isinstance(audio, torch.Tensor) and audio.shape == (
            2,
            _AUDIO_CHANNELS,
            _AUDIO_SAMPLES,
        )

    def test_ot_with_disabled_modalities_passes_none_through(self, single_h5: Path) -> None:  # noqa: DOC101,DOC103
        """``_hungarian_match`` still receives ``None`` placeholders when modalities are off."""
        with patch(
            "synth_setter.data.surge_datamodule._hungarian_match",
            side_effect=lambda noise, params, *args: (noise, params, *args),
        ) as mock_match:
            dataset = SurgeXTDataset(
                single_h5,
                batch_size=2,
                ot=True,
                use_saved_mean_and_variance=False,
                read_audio=False,
                read_mel=False,
                read_m2l=False,
            )
            item = dataset[0]
        mock_match.assert_called_once()
        positional = mock_match.call_args.args
        # Positional contract: noise, params, mel_spec, audio — the trailing
        # two are None when their read flags are off.
        assert positional[2] is None
        assert positional[3] is None
        assert item["mel_spec"] is None
        assert item["audio"] is None

    def test_returned_dict_always_exposes_full_key_set(self, single_h5: Path) -> None:  # noqa: DOC101,DOC103
        """Every ``__getitem__`` return dict exposes the same five-key contract."""
        dataset = SurgeXTDataset(
            single_h5,
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=False,
        )
        item = dataset[0]
        assert set(item.keys()) == set(_ALL_TENSOR_KEYS)


# --------------------------------------------------------------------------- #
# WithinChunkShuffledSampler                                                  #
# --------------------------------------------------------------------------- #


class TestWithinChunkShuffledSampler:
    """Shard-aware sampler: shuffles within fixed-size groups to bound h5py reads."""

    def test_len_matches_num_batches(self) -> None:
        """``__len__`` is the declared ``num_batches``, independent of grouping math."""
        sampler = WithinChunkShuffledSampler(batch_size=4, num_batches=10, batches_per_group=3)
        assert len(sampler) == 10

    def test_iter_yields_num_batches_rows(self) -> None:
        """Exactly ``num_batches`` batches are emitted, regardless of remainder grouping."""
        sampler = WithinChunkShuffledSampler(batch_size=4, num_batches=10, batches_per_group=3)
        rows = list(sampler)
        assert len(rows) == 10

    def test_each_row_has_batch_size_indices(self) -> None:
        """Every emitted batch has exactly ``batch_size`` indices."""
        sampler = WithinChunkShuffledSampler(batch_size=4, num_batches=6, batches_per_group=2)
        for row in sampler:
            assert len(row) == 4

    def test_each_row_is_sorted_ascending(self) -> None:
        """Within a batch, indices are sorted (cheaper monotone h5py reads)."""
        sampler = WithinChunkShuffledSampler(batch_size=4, num_batches=6, batches_per_group=2)
        for row in sampler:
            assert row == sorted(row)

    def test_all_indices_unique_when_evenly_divisible(self) -> None:
        """No index is repeated across the full epoch when ``num_batches`` divides cleanly."""
        batch_size, num_batches = 4, 6
        sampler = WithinChunkShuffledSampler(
            batch_size=batch_size, num_batches=num_batches, batches_per_group=2
        )
        flat = [int(idx) for row in sampler for idx in row]
        # Strengthen the uniqueness check: the sampler must yield exactly the
        # expected index range, not just any unique values that pass the
        # ``len == len(set)`` check (e.g. emitting ``[1000, 1001, ...]`` would
        # still pass uniqueness while silently skipping rows 0..N-1).
        assert sorted(flat) == list(range(batch_size * num_batches))

    def test_all_indices_unique_with_remainder_group(self) -> None:
        """The remainder group keeps overall uniqueness intact."""
        batch_size, num_batches = 4, 7
        sampler = WithinChunkShuffledSampler(
            batch_size=batch_size, num_batches=num_batches, batches_per_group=3
        )
        flat = [int(idx) for row in sampler for idx in row]
        # Same strengthening as ``test_all_indices_unique_when_evenly_divisible``
        # — pin the actual index range covered, not just uniqueness.
        assert sorted(flat) == list(range(batch_size * num_batches))

    def test_indices_respect_group_boundaries(self) -> None:
        """Each batch's indices live entirely within one ``batches_per_group``-sized window."""
        batch_size = 4
        batches_per_group = 3
        sampler = WithinChunkShuffledSampler(
            batch_size=batch_size, num_batches=9, batches_per_group=batches_per_group
        )
        samples_per_group = batch_size * batches_per_group
        for row in sampler:
            group_ids = {idx // samples_per_group for idx in row}
            assert len(group_ids) == 1

    def test_remainder_group_produces_smaller_groups(self) -> None:
        """When ``num_batches % batches_per_group != 0`` the tail group still emits batches."""
        batch_size, num_batches, batches_per_group = 4, 5, 3
        sampler = WithinChunkShuffledSampler(
            batch_size=batch_size,
            num_batches=num_batches,
            batches_per_group=batches_per_group,
        )
        rows = list(sampler)
        assert len(rows) == num_batches
        # Strengthen: also assert the remainder tail keeps the contract that
        # every dataset row is visited exactly once — a regression that drops
        # the remainder group entirely would still pass ``len(rows) == 5``.
        flat = [int(idx) for row in rows for idx in row]
        assert sorted(flat) == list(range(batch_size * num_batches))
        # Pin the group-boundary contract on the remainder tail: every batch's
        # indices live inside one ``batches_per_group``-sized window — including
        # the tail group, which is smaller than the others.
        samples_per_group = batch_size * batches_per_group
        for row in rows:
            group_ids = {int(idx) // samples_per_group for idx in row}
            assert len(group_ids) == 1


# --------------------------------------------------------------------------- #
# ShuffledSampler                                                             #
# --------------------------------------------------------------------------- #


class TestShuffledSampler:
    """Global random permutation sampler — no locality guarantees."""

    def test_len_matches_num_batches(self) -> None:
        """``__len__`` reports the declared ``num_batches``."""
        assert len(ShuffledSampler(batch_size=4, num_batches=7)) == 7

    def test_iter_yields_num_batches_rows(self) -> None:
        """Iteration emits exactly ``num_batches`` rows."""
        rows = list(ShuffledSampler(batch_size=4, num_batches=7))
        assert len(rows) == 7

    def test_each_row_has_batch_size_indices(self) -> None:
        """Every emitted row has exactly ``batch_size`` indices."""
        for row in ShuffledSampler(batch_size=4, num_batches=5):
            assert len(row) == 4

    def test_each_row_is_sorted_ascending(self) -> None:
        """Within a row, indices are sorted (matches the contract used by ``_index_dataset``)."""
        for row in ShuffledSampler(batch_size=4, num_batches=5):
            assert list(row) == sorted(row)

    def test_indices_cover_full_range_without_duplicates(self) -> None:
        """Across the full epoch, every index in ``[0, B*N)`` appears exactly once."""
        batch_size, num_batches = 4, 5
        sampler = ShuffledSampler(batch_size=batch_size, num_batches=num_batches)
        flat = [int(idx) for row in sampler for idx in row]
        assert sorted(flat) == list(range(batch_size * num_batches))

    def test_iter_yields_full_permutation_in_sorted_batches(self) -> None:
        """Concatenated batches are a permutation of ``range(num_batches*batch_size)``."""
        np_state = np.random.get_state()
        try:
            np.random.seed(0)
            sampler = ShuffledSampler(batch_size=4, num_batches=6)
            batches = list(iter(sampler))
            assert len(batches) == 6
            flat: list[int] = []
            for batch in batches:
                batch_list = list(batch)
                assert batch_list == sorted(batch_list)
                assert len(batch_list) == 4
                flat.extend(batch_list)
            assert sorted(flat) == list(range(6 * 4))
        finally:
            np.random.set_state(np_state)


# --------------------------------------------------------------------------- #
# ShiftedBatchSampler                                                         #
# --------------------------------------------------------------------------- #


class TestShiftedBatchSampler:
    """Two-int ``(start, end)`` sampler that adds a random per-epoch offset."""

    def test_len_is_num_batches_minus_one(self) -> None:
        """``__len__`` is ``num_batches - 1`` — the trailing batch is dropped to make room for shift."""
        assert len(ShiftedBatchSampler(batch_size=4, num_batches=6)) == 5

    def test_iter_yields_two_tuples(self) -> None:
        """Each emission is a length-2 tuple ``(start, end)`` consumed by ``_index_dataset``."""
        for pair in ShiftedBatchSampler(batch_size=4, num_batches=6):
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    def test_consecutive_within_pair_one_batch_apart(self) -> None:
        """``end - start`` always equals ``batch_size`` (a single contiguous window)."""
        batch_size = 4
        for start, end in ShiftedBatchSampler(batch_size=batch_size, num_batches=6):
            assert end - start == batch_size

    def test_offset_is_in_range(self) -> None:
        """The shared epoch-wide offset is drawn from ``[0, batch_size)``."""
        batch_size, num_batches = 4, 6
        for start, _ in ShiftedBatchSampler(batch_size=batch_size, num_batches=num_batches):
            offset = start % batch_size
            assert 0 <= offset < batch_size

    def test_offset_is_constant_within_one_epoch(self) -> None:
        """All emitted pairs in one iteration share the same per-epoch offset."""
        batch_size = 4
        offsets = {
            start % batch_size for start, _ in ShiftedBatchSampler(batch_size, num_batches=8)
        }
        assert len(offsets) == 1

    def test_offset_can_vary_across_epochs(self) -> None:
        """A fresh ``iter()`` redraws the offset — over many epochs we see >1 distinct value."""
        # Seed Python's random + numpy so the test is deterministic but still
        # exercises the redraw. Save/restore the global states so other tests
        # in the same pytest-xdist worker don't see a leaked seed.
        py_state = random.getstate()
        np_state = np.random.get_state()
        try:
            random.seed(0)
            np.random.seed(0)
            batch_size = 4
            sampler = ShiftedBatchSampler(batch_size=batch_size, num_batches=8)
            offsets = set()
            for _ in range(200):
                start, _ = next(iter(sampler))
                offsets.add(start % batch_size)
            assert len(offsets) > 1
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    def test_iter_visits_each_consecutive_pair_index_once(self) -> None:
        """The ``num_batches - 1`` consecutive-pair indices are each visited exactly once."""
        batch_size, num_batches = 4, 6
        pair_indices = sorted(
            start // batch_size
            for start, _ in ShiftedBatchSampler(batch_size=batch_size, num_batches=num_batches)
        )
        assert pair_indices == list(range(num_batches - 1))

    def test_iter_offset_zero_yields_aligned_pairs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``offset=0`` gives back the unshifted batch boundaries.

        :param monkeypatch: pytest monkeypatch fixture used to pin ``random.randint``.
        """
        monkeypatch.setattr(random, "randint", lambda a, b: 0)
        np_state = np.random.get_state()
        try:
            np.random.seed(0)
            batch_size = 3
            sampler = ShiftedBatchSampler(batch_size=batch_size, num_batches=4)
            pairs = list(iter(sampler))
            starts = sorted(start for start, _ in pairs)
            # batches 0..(num_batches-2) at boundary 0, bs, 2*bs:
            assert starts == [0, batch_size, 2 * batch_size]
        finally:
            np.random.set_state(np_state)


# --------------------------------------------------------------------------- #
# SurgeDataModule                                                             #
# --------------------------------------------------------------------------- #


class TestSurgeDataModule:
    """Lightning datamodule: setup / dataloaders / teardown wiring."""

    def test_init_stores_dataset_root_as_path(self, tmp_path: Path) -> None:  # noqa: DOC101,DOC103
        """``dataset_root`` is normalized to ``pathlib.Path`` even when passed as a str."""
        module = SurgeDataModule(dataset_root=str(tmp_path))
        assert module.dataset_root == tmp_path
        assert isinstance(module.dataset_root, Path)

    def test_defaults_are_stored_verbatim(self, tmp_path: Path) -> None:  # noqa: DOC101,DOC103
        """Every ``__init__`` default lands on the matching public attribute for hparams
        logging."""
        module = SurgeDataModule(dataset_root=tmp_path)
        assert module.dataset_root == tmp_path
        assert module.use_saved_mean_and_variance is True
        assert module.batch_size == 1024
        assert module.ot is True
        assert module.num_workers == 0
        assert module.fake is False
        assert module.repeat_first_batch is False
        assert module.predict_file is None
        assert module.conditioning == "mel"
        assert module.pin_memory is True

    def test_setup_creates_train_val_test_splits(self, dataset_root: Path) -> None:  # noqa: DOC101,DOC103
        """``setup()`` opens the three required splits and exposes them as attrs."""
        with _set_up_module(dataset_root=dataset_root, batch_size=2, ot=False) as module:
            assert isinstance(module.train_dataset, SurgeXTDataset)
            assert isinstance(module.val_dataset, SurgeXTDataset)
            assert isinstance(module.test_dataset, SurgeXTDataset)

    def test_setup_without_predict_file_leaves_predict_none(self, dataset_root: Path) -> None:  # noqa: DOC101,DOC103
        """``predict_file=None`` leaves ``predict_dataset`` as ``None``."""
        with _set_up_module(dataset_root=dataset_root, batch_size=2, ot=False) as module:
            assert module.predict_dataset is None

    def test_setup_with_predict_file_builds_predict_dataset_with_audio(  # noqa: DOC101,DOC103
        self, dataset_root: Path
    ) -> None:
        """``predict_file`` set: predict-split dataset opens with ``read_audio=True``."""
        with _set_up_module(
            dataset_root=dataset_root,
            batch_size=2,
            ot=False,
            predict_file=str(dataset_root / "test.h5"),
        ) as module:
            assert isinstance(module.predict_dataset, SurgeXTDataset)
            assert module.predict_dataset.read_audio is True

    def test_setup_val_and_test_force_ot_false(self, dataset_root: Path) -> None:  # noqa: DOC101,DOC103
        """``setup`` hard-codes ``ot=False`` on val/test even when the module is ``ot=True``."""
        with _set_up_module(dataset_root=dataset_root, batch_size=2, ot=True) as module:
            assert module.train_dataset.ot is True
            assert module.val_dataset.ot is False
            assert module.test_dataset.ot is False

    def test_conditioning_mel_routes_to_mel_reads(self, dataset_root: Path) -> None:  # noqa: DOC101,DOC103
        """``conditioning='mel'`` toggles ``read_mel=True`` / ``read_m2l=False`` on every split."""
        with _set_up_module(
            dataset_root=dataset_root, batch_size=2, ot=False, conditioning="mel"
        ) as module:
            for split in (module.train_dataset, module.val_dataset, module.test_dataset):
                assert split.read_mel is True
                assert split.read_m2l is False

    def test_conditioning_m2l_routes_to_m2l_reads(self, dataset_root: Path) -> None:  # noqa: DOC101,DOC103
        """``conditioning='m2l'`` flips the read flags to the music2latent channel."""
        with _set_up_module(
            dataset_root=dataset_root, batch_size=2, ot=False, conditioning="m2l"
        ) as module:
            for split in (module.train_dataset, module.val_dataset, module.test_dataset):
                assert split.read_mel is False
                assert split.read_m2l is True

    def test_conditioning_m2l_also_routes_predict_split(self, dataset_root: Path) -> None:  # noqa: DOC101,DOC103
        """``predict_dataset`` follows the same conditioning routing as train/val/test."""
        with _set_up_module(
            dataset_root=dataset_root,
            batch_size=2,
            ot=False,
            conditioning="m2l",
            predict_file=str(dataset_root / "test.h5"),
        ) as module:
            assert module.predict_dataset is not None
            assert module.predict_dataset.read_mel is False
            assert module.predict_dataset.read_m2l is True

    def test_train_dataloader_uses_shifted_batch_sampler(self, dataset_root: Path) -> None:  # noqa: DOC101,DOC103
        """``train_dataloader`` wires the ``ShiftedBatchSampler`` (not the global random one)."""
        with _set_up_module(
            dataset_root=dataset_root,
            batch_size=2,
            ot=False,
            num_workers=0,
            pin_memory=False,
        ) as module:
            loader = module.train_dataloader()
            assert isinstance(loader.sampler, ShiftedBatchSampler)
            assert loader.batch_size is None
            assert loader.num_workers == 0
            assert loader.pin_memory is False

    def test_val_test_dataloaders_have_no_shuffle_sampler(self, dataset_root: Path) -> None:  # noqa: DOC101,DOC103
        """Val/test loaders use the default no-shuffle ``SequentialSampler``."""
        with _set_up_module(
            dataset_root=dataset_root,
            batch_size=2,
            ot=False,
            num_workers=0,
            pin_memory=False,
        ) as module:
            val_loader = module.val_dataloader()
            test_loader = module.test_dataloader()
            assert isinstance(val_loader.sampler, torch.utils.data.SequentialSampler)
            assert isinstance(test_loader.sampler, torch.utils.data.SequentialSampler)

    def test_dataloader_num_workers_and_pin_memory_propagate(self, dataset_root: Path) -> None:  # noqa: DOC101,DOC103
        """``num_workers`` / ``pin_memory`` kwargs are passed verbatim to every DataLoader."""
        with _set_up_module(
            dataset_root=dataset_root,
            batch_size=2,
            ot=False,
            num_workers=2,
            pin_memory=True,
        ) as module:
            for loader in (
                module.train_dataloader(),
                module.val_dataloader(),
                module.test_dataloader(),
            ):
                assert loader.num_workers == 2
                assert loader.pin_memory is True

    def test_predict_dataloader_returns_dataloader_when_predict_file_set(  # noqa: DOC101,DOC103
        self, dataset_root: Path
    ) -> None:
        """``predict_dataloader`` wraps the predict split in a no-shuffle loader."""
        with _set_up_module(
            dataset_root=dataset_root,
            batch_size=2,
            ot=False,
            predict_file=str(dataset_root / "test.h5"),
        ) as module:
            loader = module.predict_dataloader()
            assert isinstance(loader, torch.utils.data.DataLoader)
            assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)

    def test_predict_dataloader_propagates_num_workers_and_pin_memory(  # noqa: DOC101,DOC103
        self, dataset_root: Path
    ) -> None:
        """``num_workers`` / ``pin_memory`` reach the predict loader too (separate construction
        path)."""
        with _set_up_module(
            dataset_root=dataset_root,
            batch_size=2,
            ot=False,
            num_workers=2,
            pin_memory=True,
            predict_file=str(dataset_root / "test.h5"),
        ) as module:
            loader = module.predict_dataloader()
            assert loader.num_workers == 2
            assert loader.pin_memory is True

    def test_teardown_closes_open_h5_handles(self, dataset_root: Path) -> None:  # noqa: DOC101,DOC103
        """``teardown`` closes the three split files so the next setup can reopen them."""
        module = SurgeDataModule(dataset_root=dataset_root, batch_size=2, ot=False)
        module.setup()
        module.teardown()
        # h5py.File's truthiness reflects open-state; after close, the handle is falsy.
        assert not module.train_dataset.dataset_file
        assert not module.val_dataset.dataset_file
        assert not module.test_dataset.dataset_file

    def test_fake_mode_setup_does_not_require_dataset_files(self, tmp_path: Path) -> None:  # noqa: DOC101,DOC103
        """``fake=True`` setup never touches the dataset_root, so a fresh dir is enough."""
        with _set_up_module(
            dataset_root=tmp_path,
            batch_size=2,
            ot=False,
            fake=True,
            use_saved_mean_and_variance=False,
        ) as module:
            assert module.train_dataset.fake is True
            assert module.val_dataset.fake is True
            assert module.test_dataset.fake is True

    def test_fake_mode_train_dataloader_yields_well_shaped_items(self, tmp_path: Path) -> None:  # noqa: DOC101,DOC103
        """End-to-end smoke: fake-mode train loader iterates and produces sane shapes."""
        with _set_up_module(
            dataset_root=tmp_path,
            batch_size=2,
            ot=False,
            fake=True,
            use_saved_mean_and_variance=False,
            num_workers=0,
            pin_memory=False,
        ) as module:
            loader = module.train_dataloader()
            item = next(iter(loader))
            assert _unwrap(item["params"]).shape == (2, 189)
            assert _unwrap(item["mel_spec"]).shape == (2, 2, 128, 401)
