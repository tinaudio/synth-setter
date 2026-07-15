"""Behavioral tests for :mod:`synth_setter.data.vst_datamodule` over Lance shards.

Covers the four public symbols exposed by the module:

* :class:`VSTDataset` / :class:`LanceVSTDataset` — both the ``fake`` synthetic
  path and the real Lance-backed path, with the three boolean read flags
  (``read_audio`` / ``read_mel`` / ``read_m2l``), OT matching toggle, parameter
  rescaling, the ``repeat_first_batch`` mode, and the sibling-``stats.npz`` loader.
* :class:`WithinChunkShuffledSampler`, :class:`ShuffledSampler`,
  :class:`ShiftedBatchSampler` — three batch-index samplers with distinct
  shuffle/strict-locality invariants.
* :class:`VSTDataModule` / :class:`LanceVSTDataModule` — Lightning ``setup`` /
  dataloader / ``teardown`` wiring, including the ``conditioning`` mel-vs-m2l
  switch and the ``predict_file`` optional split.

Lance fixtures are tiny (a handful of rows, ~10-element mel/audio axes) — the
goal is contract coverage on shapes, flags, and call routing, not numerical
ML behavior. The mel/audio dimensions deliberately do NOT match production
shapes (production uses ``(2, 128, 401)`` mel and ``2 * 44100 * 4`` audio
samples); shrinking them keeps the fixture tiny on disk while still
exercising every code branch in ``__getitem__`` and ``_index_dataset``.
"""

from __future__ import annotations

import contextlib
import random
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from synth_setter.data import surge_datamodule
from synth_setter.data.lance_datamodule import LanceVSTDataModule, LanceVSTDataset
from synth_setter.data.ot import _hungarian_match
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.data.vst_datamodule import (
    ShiftedBatchSampler,
    ShuffledSampler,
    VSTDataset,
    WithinChunkShuffledSampler,
)
from synth_setter.param_spec_name import ParamSpecName
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
_FAKE_PARAM_WIDTH = 3


@pytest.fixture()
def local_r2_remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Back the ``r2:`` rclone remote with the local filesystem for real-binary e2e.

    ``SYNTH_SETTER_STORAGE_RCLONE_TYPE=local`` resolves ``r2://<bucket>/<key>`` to
    ``<cwd>/<bucket>/<key>``. Canonical storage settings satisfy the credential
    check; their rclone projection is unused by the local backend.

    :param tmp_path: Pytest tmp dir; the returned subdir is the fake R2 root.
    :param monkeypatch: Sets the rclone env vars and chdirs into the remote root.
    :return: The fake R2 root; ``r2://<bucket>/<key>`` materializes under it.
    """
    if shutil.which("rclone") is None:
        pytest.skip("rclone binary not available on PATH")
    remote_root = tmp_path / "r2"
    remote_root.mkdir()
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID", "stub")
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY", "stub")
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_ENDPOINT_URL", "http://localhost:0")
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_RCLONE_TYPE", "local")
    monkeypatch.chdir(remote_root)
    return remote_root


@contextlib.contextmanager
def _set_up_module(**kwargs: object) -> Iterator[LanceVSTDataModule]:
    """Construct a ``LanceVSTDataModule`` from ``**kwargs``, ``setup``, yield, then ``teardown``.

    Encapsulates the setup/teardown try/finally pattern every
    ``TestVSTDataModule`` test needs so a forgotten ``teardown`` can't leak
    shard handles into the next test.

    :param \\*\\*kwargs: Forwarded to ``LanceVSTDataModule``.
    :yields: The set-up datamodule for assertion work inside the ``with`` block.
    :ytype: LanceVSTDataModule
    """
    kwargs.setdefault("param_spec_name", ParamSpecName("surge_xt"))
    module = LanceVSTDataModule(**kwargs)  # type: ignore[arg-type]
    module.setup()
    try:
        yield module
    finally:
        module.teardown()


def _unwrap(maybe_tensor: torch.Tensor | None) -> torch.Tensor:
    """Assert ``maybe_tensor`` is non-None and narrow it for pyright.

    :param maybe_tensor: The dict value to narrow.
    :return: The same tensor, now typed as non-None.
    """
    assert maybe_tensor is not None
    return maybe_tensor


def _write_lance_shard(
    path: Path,
    num_rows: int,
    *,
    params_seed: int = 0,
    mel_fill: float | None = None,
    include_audio: bool = True,
) -> None:
    """Write a tiny Lance shard with the columns ``VSTDataset`` reads.

    :param path: Output ``.lance`` dataset directory.
    :param num_rows: Number of rows along the first axis of every column.
    :param params_seed: Seed for the per-row parameter array so different
        splits get distinguishable values when needed.
    :param mel_fill: When set, fill the ``mel_spec`` column with this constant
        instead of random values — used by the normalization test to make
        ``(mel - mean) / std`` produce a predictable result.
    :param include_audio: When ``False``, omit the ``audio`` column so the
        row-count path off ``param_array`` is exercised on an audio-less shard.
    """
    rng = np.random.default_rng(params_seed)
    if mel_fill is None:
        mel_data = rng.standard_normal((num_rows, _MEL_CHANNELS, _MEL_N_MELS, _MEL_N_FRAMES))
    else:
        mel_data = np.full(
            (num_rows, _MEL_CHANNELS, _MEL_N_MELS, _MEL_N_FRAMES),
            fill_value=mel_fill,
            dtype=np.float32,
        )
    columns: dict[str, np.ndarray] = {
        "mel_spec": mel_data.astype(np.float32),
        "music2latent": rng.standard_normal((num_rows, _M2L_DIM_1, _M2L_DIM_2)).astype(np.float32),
        # params in [0, 1) so the rescale_params=True branch lands in [-1, 1).
        "param_array": rng.random((num_rows, _NUM_PARAMS)).astype(np.float32),
    }
    if include_audio:
        # Audio stays in [-1, 1]: the read path validates the range.
        columns["audio"] = rng.uniform(
            -1.0, 1.0, (num_rows, _AUDIO_CHANNELS, _AUDIO_SAMPLES)
        ).astype(np.float32)
    write_lance_shard(path, columns)


def _write_stats(
    dataset_dir: Path,
    *,
    mean: float = 0.0,
    std: float = 1.0,
) -> Path:
    """Write a sibling ``stats.npz`` whose mean/std broadcast against ``mel_spec``.

    :param dataset_dir: Directory holding the split shards; ``stats.npz`` is written
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
    """Build a ``dataset_root`` directory with ``train/val/test.lance`` + ``stats.npz``.

    :param tmp_path: Per-test tmpdir.
    :return: Path to the populated dataset root directory.
    :rtype: Path
    """
    root = tmp_path / "data"
    root.mkdir()
    _write_lance_shard(root / "train.lance", num_rows=8, params_seed=1)
    _write_lance_shard(root / "val.lance", num_rows=8, params_seed=2)
    _write_lance_shard(root / "test.lance", num_rows=8, params_seed=3)
    _write_stats(root)
    return root


@pytest.fixture
def single_lance(tmp_path: Path) -> Path:
    """Write a single ``train.lance`` + sibling ``stats.npz`` for VSTDataset-only tests.

    :param tmp_path: Per-test tmpdir.

    :return: Path to the written ``train.lance`` dataset directory.
    :rtype: Path
    """
    lance_path = tmp_path / "train.lance"
    _write_lance_shard(lance_path, num_rows=8)
    _write_stats(tmp_path)
    return lance_path


class TestVSTDatasetFakeMode:
    """``fake=True`` skips the shard entirely and returns randomly-generated tensors."""

    def test_fake_mode_does_not_open_shard_file(self, tmp_path: Path) -> None:
        """``fake=True`` accepts a nonexistent path because ``__init__`` never reads it.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        missing = tmp_path / "does-not-exist.lance"
        dataset = VSTDataset(missing, batch_size=4, fake=True, num_params=_FAKE_PARAM_WIDTH)
        assert dataset.dataset_file is None

    def test_fake_mode_len_is_fixed_constant(self) -> None:
        """``__len__`` returns the documented 10000 in fake mode regardless of batch_size."""
        small = VSTDataset("ignored", batch_size=1, fake=True, num_params=_FAKE_PARAM_WIDTH)
        large = VSTDataset("ignored", batch_size=8192, fake=True, num_params=_FAKE_PARAM_WIDTH)
        assert len(small) == 10000
        assert len(large) == 10000

    def test_fake_mode_default_flags_populate_mel_params_noise(self) -> None:
        """Default flags populate ``mel_spec``/``params``/``noise``; ``audio``/``m2l`` are None.

        ``read_audio`` and ``read_m2l`` both default to False, so their slots
        drop to None — the same flag→slot mapping as real mode.
        """
        dataset = VSTDataset("ignored", batch_size=3, fake=True, num_params=_FAKE_PARAM_WIDTH)
        item = dataset[0]
        assert item["audio"] is None
        assert item["m2l"] is None
        assert _unwrap(item["mel_spec"]).shape == (3, 2, 128, 401)
        assert _unwrap(item["params"]).shape == (3, _FAKE_PARAM_WIDTH)
        assert _unwrap(item["noise"]).shape == (3, _FAKE_PARAM_WIDTH)

    def test_fake_mode_read_audio_true_returns_audio_tensor(self) -> None:
        """``read_audio=True`` populates the ``audio`` slot, mirroring real mode."""
        dataset = VSTDataset(
            "ignored", batch_size=2, fake=True, read_audio=True, num_params=_FAKE_PARAM_WIDTH
        )
        item = dataset[0]
        assert _unwrap(item["audio"]).shape == (2, 2, 44100 * 4)

    def test_fake_mode_read_m2l_returns_m2l_tensor(self) -> None:
        """``read_m2l=True`` populates the ``m2l`` slot with the documented shape."""
        dataset = VSTDataset(
            "ignored", batch_size=2, fake=True, read_m2l=True, num_params=_FAKE_PARAM_WIDTH
        )
        assert _unwrap(dataset[0]["m2l"]).shape == (2, 128, 42)

    def test_fake_mode_read_mel_false_returns_none_mel(self) -> None:
        """``read_mel=False`` drops the ``mel_spec`` slot to ``None``."""
        dataset = VSTDataset(
            "ignored", batch_size=2, fake=True, read_mel=False, num_params=_FAKE_PARAM_WIDTH
        )
        item = dataset[0]
        assert item["mel_spec"] is None

    def test_fake_mode_rescale_params_maps_into_minus_one_to_one(self) -> None:
        """``rescale_params=True`` rescales ``torch.rand`` from [0, 1) into [-1, 1)."""
        # batch_size=128 with mel/audio enabled allocates ~190 MB; disable both
        # since this test only inspects params.
        dataset = VSTDataset(
            "ignored",
            batch_size=128,
            fake=True,
            rescale_params=True,
            read_mel=False,
            read_audio=False,
            num_params=_FAKE_PARAM_WIDTH,
        )
        params = _unwrap(dataset[0]["params"])
        assert params.min().item() >= -1.0
        assert params.max().item() < 1.0
        # 128*num_params samples make staying on one side vanishingly unlikely;
        # assert both to pin the rescaling, not just the bounds.
        assert params.min().item() < 0.0
        assert params.max().item() > 0.0

    def test_fake_mode_no_rescale_params_stays_in_zero_to_one(self) -> None:
        """``rescale_params=False`` leaves params in ``torch.rand``'s native [0, 1) range."""
        dataset = VSTDataset(
            "ignored",
            batch_size=128,
            fake=True,
            rescale_params=False,
            read_mel=False,
            read_audio=False,
            num_params=_FAKE_PARAM_WIDTH,
        )
        params = _unwrap(dataset[0]["params"])
        assert params.min().item() >= 0.0
        assert params.max().item() < 1.0

    def test_fake_mode_noise_matches_param_shape(self) -> None:
        """``noise`` is allocated with ``torch.randn_like(params)``, so shapes match."""
        dataset = VSTDataset("ignored", batch_size=5, fake=True, num_params=_FAKE_PARAM_WIDTH)
        item = dataset[0]
        assert _unwrap(item["noise"]).shape == _unwrap(item["params"]).shape

    def test_fake_mode_returns_full_key_set(self) -> None:
        """The returned dict always exposes all five keys (some may be ``None``)."""
        dataset = VSTDataset("ignored", batch_size=2, fake=True, num_params=_FAKE_PARAM_WIDTH)
        item = dataset[0]
        assert set(item.keys()) == set(_ALL_TENSOR_KEYS)

    def test_fake_mode_without_num_params_raises_value_error(self) -> None:
        """Fake datasets require their caller to select a parameter width explicitly."""
        with pytest.raises(ValueError, match="num_params"):
            VSTDataset("ignored", batch_size=4, fake=True)

    def test_fake_mode_num_params_overrides_param_width(self) -> None:
        """An explicit ``num_params`` sets the fake param width; noise tracks it."""
        dataset = VSTDataset("ignored", batch_size=2, fake=True, num_params=92)
        item = dataset[0]
        assert _unwrap(item["params"]).shape == (2, 92)
        assert _unwrap(item["noise"]).shape == (2, 92)


class TestVSTDatasetLanceMode:
    """Lance-backed path: indexing, type conversion, OT routing, normalization."""

    def test_len_equals_num_rows_floor_divided_by_batch_size(self, single_lance: Path) -> None:
        """``__len__`` uses integer division — 8 rows / batch_size 3 == 2 batches.

        :param single_lance: Fixture-provided single-shard Lance path.
        """
        dataset = LanceVSTDataset(single_lance, batch_size=3, ot=False)
        assert len(dataset) == 8 // 3

    def test_len_counts_param_array_rows_without_audio_column(self, tmp_path: Path) -> None:
        """``__len__`` counts rows off ``param_array`` so an audio-less shard still works.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        lance_path = tmp_path / "train.lance"
        _write_lance_shard(lance_path, num_rows=8, include_audio=False)
        dataset = LanceVSTDataset(
            lance_path, batch_size=3, ot=False, use_saved_mean_and_variance=False
        )
        assert len(dataset) == 8 // 3

    def test_getitem_int_returns_batch_size_slice(self, single_lance: Path) -> None:
        """Integer index ``i`` reads rows ``[i*B : i*B+B]`` from each dataset.

        :param single_lance: Fixture-provided single-shard Lance path.
        """
        dataset = LanceVSTDataset(
            single_lance, batch_size=2, ot=False, use_saved_mean_and_variance=False
        )
        item = dataset[1]
        assert _unwrap(item["params"]).shape[0] == 2
        assert _unwrap(item["mel_spec"]).shape[0] == 2

    def test_getitem_tuple_returns_explicit_slice(self, single_lance: Path) -> None:
        """A 2-tuple index ``(lo, hi)`` selects rows ``[lo:hi]`` directly.

        :param single_lance: Fixture-provided single-shard Lance path.
        """
        dataset = LanceVSTDataset(
            single_lance, batch_size=2, ot=False, use_saved_mean_and_variance=False
        )
        item = dataset[(1, 5)]
        assert _unwrap(item["params"]).shape[0] == 4

    def test_getitem_sequence_falls_through_to_ds_fancy_indexing(self, single_lance: Path) -> None:
        """A non-int / non-2-tuple index falls through to ``ds[idx]`` fancy indexing.

        :param single_lance: Fixture-provided single-shard Lance path.
        """
        dataset = LanceVSTDataset(
            single_lance, batch_size=2, ot=False, use_saved_mean_and_variance=False
        )
        item = dataset[[0, 2, 4]]
        assert _unwrap(item["params"]).shape[0] == 3

    def test_repeat_first_batch_ignores_idx(self, single_lance: Path) -> None:
        """``repeat_first_batch=True`` always returns the first ``batch_size`` rows.

        :param single_lance: Fixture-provided single-shard Lance path.
        """
        dataset = LanceVSTDataset(
            single_lance,
            batch_size=3,
            ot=False,
            use_saved_mean_and_variance=False,
            repeat_first_batch=True,
        )
        first = dataset[0]
        later = dataset[2]
        assert torch.equal(_unwrap(first["params"]), _unwrap(later["params"]))

    def test_returned_tensors_are_float32(self, single_lance: Path) -> None:
        """All numeric tensors come back as ``torch.float32`` for AMP compatibility.

        :param single_lance: Fixture-provided single-shard Lance path.
        """
        dataset = LanceVSTDataset(
            single_lance,
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

    def test_returned_tensors_are_contiguous(self, single_lance: Path) -> None:
        """Every populated tensor is ``.contiguous()`` so downstream cuda copies are cheap.

        :param single_lance: Fixture-provided single-shard Lance path.
        """
        dataset = LanceVSTDataset(
            single_lance,
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

    def test_read_audio_false_returns_none_audio(self, single_lance: Path) -> None:
        """``read_audio=False`` (default) leaves the ``audio`` slot at ``None``.

        :param single_lance: Fixture-provided single-shard Lance path.
        """
        dataset = LanceVSTDataset(
            single_lance, batch_size=2, ot=False, use_saved_mean_and_variance=False
        )
        item = dataset[0]
        assert item["audio"] is None

    def test_read_mel_false_returns_none_mel(self, single_lance: Path) -> None:
        """``read_mel=False`` drops the ``mel_spec`` slot, even with stats on disk.

        :param single_lance: Fixture-provided single-shard Lance path.
        """
        dataset = LanceVSTDataset(single_lance, batch_size=2, ot=False, read_mel=False)
        item = dataset[0]
        assert item["mel_spec"] is None

    def test_read_m2l_true_returns_m2l_tensor(self, single_lance: Path) -> None:
        """``read_m2l=True`` reads the ``music2latent`` dataset under the ``m2l`` key.

        :param single_lance: Fixture-provided single-shard Lance path.
        """
        dataset = LanceVSTDataset(
            single_lance,
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=False,
            read_m2l=True,
        )
        assert _unwrap(dataset[0]["m2l"]).shape == (2, _M2L_DIM_1, _M2L_DIM_2)

    def test_rescale_params_centers_to_minus_one_to_one(self, single_lance: Path) -> None:
        """``rescale_params=True`` applies ``p * 2 - 1`` element-wise before tensor conversion.

        :param single_lance: Fixture-provided single-shard Lance path.
        """
        dataset_raw = LanceVSTDataset(
            single_lance,
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=False,
            rescale_params=False,
        )
        dataset_rescaled = LanceVSTDataset(
            single_lance,
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
        lance_path = tmp_path / "train.lance"
        _write_lance_shard(lance_path, num_rows=4, mel_fill=3.0)
        _write_stats(tmp_path, mean=1.0, std=2.0)
        dataset = LanceVSTDataset(
            lance_path,
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=True,
        )
        mel = _unwrap(dataset[0]["mel_spec"])
        expected = (3.0 - 1.0) / 2.0
        assert torch.allclose(mel, torch.full_like(mel, expected))

    def test_no_stats_load_when_disabled(self, tmp_path: Path) -> None:
        """``use_saved_mean_and_variance=False`` skips the npz read, even if it's missing.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        lance_path = tmp_path / "train.lance"
        _write_lance_shard(lance_path, num_rows=4)
        dataset = LanceVSTDataset(
            lance_path,
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=False,
        )
        assert dataset.mean is None
        assert dataset.std is None

    def test_missing_stats_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """``use_saved_mean_and_variance=True`` with no sibling ``stats.npz`` errors clearly.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        lance_path = tmp_path / "train.lance"
        _write_lance_shard(lance_path, num_rows=4)
        with pytest.raises(FileNotFoundError, match="stats.npz"):
            LanceVSTDataset(lance_path, batch_size=2, ot=False, use_saved_mean_and_variance=True)

    def test_get_stats_file_path_is_sibling_of_dataset(self, tmp_path: Path) -> None:
        """The static helper returns ``parent_dir / 'stats.npz'`` for any input layout.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        # str input
        assert VSTDataset.get_stats_file_path(str(tmp_path / "train.lance")) == (
            tmp_path / "stats.npz"
        )
        # Path input
        assert VSTDataset.get_stats_file_path(tmp_path / "val.lance") == tmp_path / "stats.npz"
        # Nested path
        nested = tmp_path / "shard0" / "data.lance"
        assert VSTDataset.get_stats_file_path(nested) == tmp_path / "shard0" / "stats.npz"

    def test_no_ot_does_not_call_hungarian_match(self, single_lance: Path) -> None:
        """``ot=False`` short-circuits before ``_hungarian_match`` is invoked.

        :param single_lance: Fixture-provided single-shard Lance path.
        """
        with patch("synth_setter.data.vst_datamodule._hungarian_match") as mock_match:
            dataset = LanceVSTDataset(
                single_lance,
                batch_size=2,
                ot=False,
                use_saved_mean_and_variance=False,
            )
            _ = dataset[0]
        mock_match.assert_not_called()

    def test_ot_with_disabled_modalities_passes_none_through(self, single_lance: Path) -> None:
        """``_hungarian_match`` still receives ``None`` placeholders when modalities are off.

        :param single_lance: Fixture-provided single-shard Lance path.
        """
        with patch(
            "synth_setter.data.vst_datamodule._hungarian_match",
            side_effect=lambda noise, params, *args: (noise, params, *args),
        ) as mock_match:
            dataset = LanceVSTDataset(
                single_lance,
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

    def test_returned_dict_always_exposes_full_key_set(self, single_lance: Path) -> None:
        """Every ``__getitem__`` return dict exposes the same five-key contract.

        :param single_lance: Fixture-provided single-shard Lance path.
        """
        dataset = LanceVSTDataset(
            single_lance,
            batch_size=2,
            ot=False,
            use_saved_mean_and_variance=False,
        )
        item = dataset[0]
        assert set(item.keys()) == set(_ALL_TENSOR_KEYS)


def _reference_getitem(
    dataset: VSTDataset, idx: int | list[int] | tuple[int, int], *, seed: int
) -> dict[str, torch.Tensor | None]:
    """Apply the batch math against the same shard independently of ``__getitem__``.

    Draws noise via ``torch.randn`` on a fresh generator seeded by ``seed``; a
    same-seeded ``VSTDataset.generator`` reproduces this bit-for-bit (equivalence
    guarded by ``test_noise_draw_apis_same_seed_bit_identical`` in
    ``test_prepare_batch.py``), so a match proves the delegation is a no-op.
    Deliberately white-box: it reads ``dataset.dataset_file`` and calls the
    private ``_index_dataset`` so the golden mirrors the production read path —
    renaming those members must update this helper in the same change.

    :param dataset: Open Lance-backed dataset to read columns from.
    :param idx: Batch index, ``(start, stop)`` pair, or explicit row list.
    :param seed: Seed for the golden's own noise generator.
    :raises ValueError: If ``dataset`` has no open shard handle.
    :returns: The batch dict ``__getitem__`` is expected to return.
    """
    shard = dataset.dataset_file
    if shard is None:
        raise ValueError("dataset has no open shard handle")
    if dataset.read_audio:
        audio = dataset._index_dataset(shard["audio"], idx)
        audio = torch.from_numpy(audio).to(dtype=torch.float32)
    else:
        audio = None

    if dataset.read_mel:
        mel_spec = dataset._index_dataset(shard["mel_spec"], idx)
        if dataset.mean is not None and dataset.std is not None:
            mel_spec = (mel_spec - dataset.mean) / dataset.std
        mel_spec = torch.from_numpy(mel_spec).to(dtype=torch.float32)
    else:
        mel_spec = None

    if dataset.read_m2l:
        m2l = dataset._index_dataset(shard["music2latent"], idx)
        m2l = torch.from_numpy(m2l).to(dtype=torch.float32)
    else:
        m2l = None

    param_array = dataset._index_dataset(shard["param_array"], idx)
    if dataset.rescale_params:
        param_array = param_array * 2 - 1
    param_array = torch.from_numpy(param_array).to(dtype=torch.float32)
    noise = torch.randn(param_array.shape, generator=torch.Generator().manual_seed(seed))
    if dataset.ot:
        noise, param_array, mel_spec, m2l, audio = _hungarian_match(
            noise, param_array, mel_spec, m2l, audio
        )

    return dict(
        mel_spec=mel_spec.contiguous() if mel_spec is not None else None,
        m2l=m2l.contiguous() if m2l is not None else None,
        params=param_array.contiguous(),
        noise=noise.contiguous(),
        audio=audio.contiguous() if audio is not None else None,
    )


class TestGetitemNoOpAfterExtraction:
    """The post-extraction ``__getitem__`` matches the independent golden."""

    @pytest.mark.parametrize("idx", [0, 1, [0, 2, 5, 7], (2, 6)])
    def test_getitem_unchanged_after_extraction(
        self, single_lance: Path, idx: int | list[int] | tuple[int, int]
    ) -> None:
        """Seeding ``dataset.generator`` reproduces the golden at each index form.

        :param single_lance: Fixture-provided single-shard Lance path.
        :param idx: Batch index, row list, or ``(start, stop)`` pair — every
            index shape ``_index_dataset`` documents, so a regression in any
            slicing branch (or an off-by-one past the first batch) surfaces.
        """
        seed = 0
        dataset = LanceVSTDataset(
            single_lance,
            batch_size=4,
            ot=True,
            use_saved_mean_and_variance=True,
            read_audio=True,
            read_mel=True,
            read_m2l=True,
        )
        golden = _reference_getitem(dataset, idx, seed=seed)
        dataset.generator.manual_seed(seed)
        out = dataset[idx]
        for key in _ALL_TENSOR_KEYS:
            if golden[key] is None:
                assert out[key] is None, key
                continue
            # atol=rtol=0: the extraction is a pure structural move, so output
            # must be bit-identical under the matched seed.
            torch.testing.assert_close(out[key], golden[key], atol=0.0, rtol=0.0)


class TestNoiseGeneratorSeeding:
    """Production seeding contract: the global seed governs the noise generator."""

    def test_getitem_noise_same_global_seed_reproduces(self, single_lance: Path) -> None:
        """Datasets built under the same global seed draw identical batch noise.

        Pins that ``seed_everything(cfg.seed)`` governs the production noise
        path (via the constructor's global-RNG draw) exactly as the
        pre-refactor global-RNG ``randn_like`` did.

        :param single_lance: Fixture-provided single-shard Lance path.
        """
        draws = []
        for _ in range(2):
            with torch.random.fork_rng():
                torch.manual_seed(7)
                dataset = LanceVSTDataset(
                    single_lance, batch_size=4, ot=False, use_saved_mean_and_variance=False
                )
                draws.append(dataset[0]["noise"])
        torch.testing.assert_close(draws[0], draws[1], atol=0.0, rtol=0.0)

    def test_fake_mode_construction_leaves_global_rng_untouched(self) -> None:
        """Constructing a fake dataset must not consume from the global RNG.

        Fake noise never uses ``self.generator``, so the constructor skips the
        global-RNG seed draw in fake mode — otherwise fake-mode runs would see
        a shifted global stream for no benefit.
        """
        with torch.random.fork_rng():
            torch.manual_seed(3)
            expected = torch.randn(4)
        with torch.random.fork_rng():
            torch.manual_seed(3)
            VSTDataset("unused.lance", batch_size=2, fake=True, num_params=_FAKE_PARAM_WIDTH)
            actual = torch.randn(4)
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def test_generator_unseeded_constructions_get_distinct_seeds(self, single_lance: Path) -> None:
        """Back-to-back real datasets without any seeding must not share a noise stream.

        Guards the constructor against regressing to a bare ``torch.Generator()``,
        whose fixed default seed would silently make every run's noise identical.

        :param single_lance: Fixture-provided single-shard Lance path.
        """
        first = LanceVSTDataset(
            single_lance, batch_size=2, ot=False, use_saved_mean_and_variance=False
        )
        second = LanceVSTDataset(
            single_lance, batch_size=2, ot=False, use_saved_mean_and_variance=False
        )
        assert first.generator.initial_seed() != second.generator.initial_seed()

    @pytest.mark.skipif(
        sys.platform != "linux",
        reason="needs forked workers: spawn (macOS default) would have to pickle the "
        "open Lance handle; production multi-worker runs are Linux/fork",
    )
    def test_dataloader_two_workers_reproduce_and_decorrelate_noise(
        self, single_lance: Path
    ) -> None:
        """Real forked workers draw seed-reproducible yet per-worker-distinct noise.

        Two epochs under one global seed must yield a bit-identical noise stream (the DataLoader
        base seed derives from the global RNG), while batches from different workers must differ —
        without the per-worker re-seed, every fork would inherit identical generator state and the
        first batch of worker 0 and worker 1 would be bit-equal.

        :param single_lance: Fixture-provided single-shard Lance path.
        """

        def collect_noise() -> list[torch.Tensor]:
            with torch.random.fork_rng():
                torch.manual_seed(11)
                dataset = LanceVSTDataset(
                    single_lance, batch_size=2, ot=False, use_saved_mean_and_variance=False
                )
                loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=2)
                draws = []
                for batch in loader:
                    noise = batch["noise"]
                    assert noise is not None
                    draws.append(noise.clone())
                return draws

        first_epoch = collect_noise()
        second_epoch = collect_noise()
        for ours, theirs in zip(first_epoch, second_epoch, strict=True):
            torch.testing.assert_close(ours, theirs, atol=0.0, rtol=0.0)
        # Sequential batches alternate workers round-robin, so index 0 vs 1 is
        # worker 0's first draw vs worker 1's first draw.
        assert not torch.equal(first_epoch[0], first_epoch[1])

    def test_parent_read_before_fork_does_not_disarm_worker_reseed(
        self, single_lance: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A parent-process read must not latch the once-per-worker re-seed flag.

        If it did, workers forked afterwards would inherit the latched flag, skip re-seeding, and
        draw correlated noise — e.g. after a debugging or preflight batch read in the parent before
        the DataLoader forks.

        :param single_lance: Fixture-provided single-shard Lance path.
        :param monkeypatch: Pytest monkeypatch fixture.
        """
        dataset = LanceVSTDataset(
            single_lance, batch_size=4, ot=False, use_saved_mean_and_variance=False
        )
        dataset[0]  # parent read: get_worker_info() is genuinely None here
        worker_info = SimpleNamespace(seed=777, num_workers=1)
        monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: worker_info)
        noise = dataset[0]["noise"]
        assert noise is not None
        expected = torch.randn(noise.shape, generator=torch.Generator().manual_seed(777))
        torch.testing.assert_close(noise, expected, atol=0.0, rtol=0.0)

    def test_getitem_in_worker_reseeds_generator_once_from_worker_seed(
        self, single_lance: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inside a dataloader worker, the first read re-seeds the forked generator.

        ``DataLoader`` hands each worker a distinct ``seed`` (``base_seed +
        worker_id``); re-seeding from it on first access decorrelates the noise
        the forked generator copies would otherwise draw identically. Later
        reads must continue that stream, not re-pin it per batch.

        :param single_lance: Fixture-provided single-shard Lance path.
        :param monkeypatch: Pytest monkeypatch fixture.
        """
        dataset = LanceVSTDataset(
            single_lance, batch_size=4, ot=False, use_saved_mean_and_variance=False
        )
        worker_info = SimpleNamespace(seed=777, num_workers=1)
        monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: worker_info)
        first = dataset[0]["noise"]
        second = dataset[1]["noise"]
        assert first is not None and second is not None
        expected_stream = torch.Generator().manual_seed(777)
        torch.testing.assert_close(
            first, torch.randn(first.shape, generator=expected_stream), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            second, torch.randn(second.shape, generator=expected_stream), atol=0.0, rtol=0.0
        )


class TestWithinChunkShuffledSampler:
    """Shard-aware sampler: shuffles within fixed-size groups to bound shard reads."""

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
        """Within a batch, indices are sorted (cheaper monotone shard reads)."""
        sampler = WithinChunkShuffledSampler(batch_size=4, num_batches=6, batches_per_group=2)
        for row in sampler:
            assert row == sorted(row)

    def test_all_indices_unique_when_evenly_divisible(self) -> None:
        """No index is repeated across the full epoch when ``num_batches`` divides cleanly."""
        sampler = WithinChunkShuffledSampler(batch_size=4, num_batches=6, batches_per_group=2)
        flat = [idx for row in sampler for idx in row]
        assert len(flat) == len(set(flat))

    def test_all_indices_unique_with_remainder_group(self) -> None:
        """The remainder group keeps overall uniqueness intact."""
        sampler = WithinChunkShuffledSampler(batch_size=4, num_batches=7, batches_per_group=3)
        flat = [idx for row in sampler for idx in row]
        assert len(flat) == len(set(flat))

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
        sampler = WithinChunkShuffledSampler(batch_size=4, num_batches=5, batches_per_group=3)
        rows = list(sampler)
        assert len(rows) == 5


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
        # Seed global RNGs for determinism; save/restore so xdist workers don't
        # see a leaked state.
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


class TestVSTDataModule:
    """Lightning datamodule: setup / dataloaders / teardown wiring."""

    def test_init_stores_dataset_root_as_path(self, tmp_path: Path) -> None:
        """``dataset_root`` is normalized to ``pathlib.Path`` even when passed as a str.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        module = LanceVSTDataModule(
            dataset_root=str(tmp_path), param_spec_name=ParamSpecName("surge_xt")
        )
        assert module.dataset_root == tmp_path
        assert isinstance(module.dataset_root, Path)

    def test_prepare_data_hydrates_dataset_root_from_r2_when_uri_set(
        self, local_r2_remote: Path, tmp_path: Path
    ) -> None:
        """``prepare_data`` with a ``download_dataset_root_uri`` fills ``dataset_root`` from R2.

        :param local_r2_remote: Real rclone remote backed by the local filesystem.
        :param tmp_path: Holds the (initially absent) download destination root.
        """
        remote_prefix = local_r2_remote / "intermediate-data" / "dataset"
        remote_prefix.mkdir(parents=True)
        (remote_prefix / "train.lance").write_bytes(b"train-bytes")
        (remote_prefix / "stats.npz").write_bytes(b"stats-bytes")
        dataset_root = tmp_path / "downloaded"

        module = LanceVSTDataModule(
            dataset_root=str(dataset_root),
            download_dataset_root_uri="r2://intermediate-data/dataset",
            param_spec_name=ParamSpecName("surge_xt"),
        )
        module.prepare_data()

        assert (dataset_root / "train.lance").read_bytes() == b"train-bytes"
        assert (dataset_root / "stats.npz").read_bytes() == b"stats-bytes"

    def test_prepare_data_no_download_when_uri_none(
        self, local_r2_remote: Path, tmp_path: Path
    ) -> None:
        """Default ``None`` URI leaves ``dataset_root`` untouched.

        :param local_r2_remote: Real rclone remote — present so a regression copies rather than
            hangs.
        :param tmp_path: Holds the pre-existing, empty dataset root.
        """
        dataset_root = tmp_path / "downloaded"
        dataset_root.mkdir()

        module = LanceVSTDataModule(
            dataset_root=str(dataset_root), param_spec_name=ParamSpecName("surge_xt")
        )
        module.prepare_data()

        assert list(dataset_root.iterdir()) == []

    def test_setup_creates_train_val_test_splits(self, dataset_root: Path) -> None:
        """``setup()`` opens the three required splits and exposes them as attrs.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_module(dataset_root=dataset_root, batch_size=2, ot=False) as module:
            assert isinstance(module.train_dataset, VSTDataset)
            assert isinstance(module.val_dataset, VSTDataset)
            assert isinstance(module.test_dataset, VSTDataset)

    def test_setup_without_predict_file_defaults_to_test_split(self, dataset_root: Path) -> None:
        """No ``predict_file``: ``predict_dataset`` defaults to the ``test.lance`` split.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_module(dataset_root=dataset_root, batch_size=2, ot=False) as module:
            assert module.predict_file == dataset_root / "test.lance"
            assert isinstance(module.predict_dataset, VSTDataset)

    def test_setup_with_predict_file_builds_predict_dataset_with_audio(
        self, dataset_root: Path
    ) -> None:
        """``predict_file`` set: predict-split dataset opens with ``read_audio=True``.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_module(
            dataset_root=dataset_root,
            batch_size=2,
            ot=False,
            predict_file=str(dataset_root / "test.lance"),
        ) as module:
            assert isinstance(module.predict_dataset, VSTDataset)
            assert module.predict_dataset.read_audio is True

    def test_setup_val_and_test_force_ot_false(self, dataset_root: Path) -> None:
        """``setup`` hard-codes ``ot=False`` on val/test even when the module is ``ot=True``.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_module(dataset_root=dataset_root, batch_size=2, ot=True) as module:
            assert isinstance(module.train_dataset, VSTDataset)
            assert isinstance(module.val_dataset, VSTDataset)
            assert isinstance(module.test_dataset, VSTDataset)
            assert module.train_dataset.ot is True
            assert module.val_dataset.ot is False
            assert module.test_dataset.ot is False

    def test_conditioning_mel_routes_to_mel_reads(self, dataset_root: Path) -> None:
        """``conditioning='mel'`` toggles ``read_mel=True`` / ``read_m2l=False`` on every split.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_module(
            dataset_root=dataset_root, batch_size=2, ot=False, conditioning="mel"
        ) as module:
            for split in (module.train_dataset, module.val_dataset, module.test_dataset):
                assert isinstance(split, VSTDataset)
                assert split.read_mel is True
                assert split.read_m2l is False

    def test_conditioning_m2l_routes_to_m2l_reads(self, dataset_root: Path) -> None:
        """``conditioning='m2l'`` flips the read flags to the music2latent channel.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_module(
            dataset_root=dataset_root, batch_size=2, ot=False, conditioning="m2l"
        ) as module:
            for split in (module.train_dataset, module.val_dataset, module.test_dataset):
                assert isinstance(split, VSTDataset)
                assert split.read_mel is False
                assert split.read_m2l is True

    def test_conditioning_m2l_also_routes_predict_split(self, dataset_root: Path) -> None:
        """``predict_dataset`` follows the same conditioning routing as train/val/test.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_module(
            dataset_root=dataset_root,
            batch_size=2,
            ot=False,
            conditioning="m2l",
            predict_file=str(dataset_root / "test.lance"),
        ) as module:
            assert isinstance(module.predict_dataset, VSTDataset)
            assert module.predict_dataset.read_mel is False
            assert module.predict_dataset.read_m2l is True

    def test_train_dataloader_uses_shifted_batch_sampler(self, dataset_root: Path) -> None:
        """``train_dataloader`` wires the ``ShiftedBatchSampler`` (not the global random one).

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
            assert loader.batch_size is None
            assert loader.num_workers == 0
            assert loader.pin_memory is False

    def test_val_test_dataloaders_have_no_shuffle_sampler(self, dataset_root: Path) -> None:
        """Val/test loaders use the default no-shuffle ``SequentialSampler``.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
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

    def test_dataloader_num_workers_and_pin_memory_propagate(self, dataset_root: Path) -> None:
        """``num_workers`` / ``pin_memory`` kwargs are passed verbatim to every DataLoader.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
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

    def test_dataloaders_leave_worker_init_fn_unset_for_lightning(
        self, dataset_root: Path
    ) -> None:
        """No split's DataLoader installs a custom ``worker_init_fn``.

        Lightning's ``seed_everything(workers=True)`` auto-adds
        ``pl_worker_init_function`` only while ``worker_init_fn is None``; a
        custom hook here would silently displace it and de-seed worker global
        RNGs (fake mode's noise). The dataset re-seeds its own generator lazily
        in ``__getitem__`` instead.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_module(
            dataset_root=dataset_root,
            batch_size=2,
            ot=False,
            predict_file=str(dataset_root / "test.lance"),
        ) as module:
            for loader in (
                module.train_dataloader(),
                module.val_dataloader(),
                module.test_dataloader(),
                module.predict_dataloader(),
            ):
                assert loader.worker_init_fn is None

    def test_predict_dataloader_returns_dataloader_when_predict_file_set(
        self, dataset_root: Path
    ) -> None:
        """``predict_dataloader`` wraps the predict split in a no-shuffle loader.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_module(
            dataset_root=dataset_root,
            batch_size=2,
            ot=False,
            predict_file=str(dataset_root / "test.lance"),
        ) as module:
            loader = module.predict_dataloader()
            assert isinstance(loader, torch.utils.data.DataLoader)
            assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)

    def test_predict_dataloader_propagates_num_workers_and_pin_memory(
        self, dataset_root: Path
    ) -> None:
        """``num_workers`` / ``pin_memory`` reach the separately constructed predict loader too.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        with _set_up_module(
            dataset_root=dataset_root,
            batch_size=2,
            ot=False,
            num_workers=2,
            pin_memory=True,
            predict_file=str(dataset_root / "test.lance"),
        ) as module:
            loader = module.predict_dataloader()
            assert loader.num_workers == 2
            assert loader.pin_memory is True

    def test_teardown_closes_open_shard_handles(self, dataset_root: Path) -> None:
        """``teardown`` closes every split file so the next setup can reopen them.

        :param dataset_root: Fixture-provided dataset-root directory.
        """
        module = LanceVSTDataModule(
            dataset_root=dataset_root,
            batch_size=2,
            ot=False,
            param_spec_name=ParamSpecName("surge_xt"),
        )
        module.setup()
        module.teardown()
        # LanceShardFile truthiness reflects open-state; after close, the handle is falsy.
        assert isinstance(module.train_dataset, VSTDataset)
        assert isinstance(module.val_dataset, VSTDataset)
        assert isinstance(module.test_dataset, VSTDataset)
        assert isinstance(module.predict_dataset, VSTDataset)
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
            assert isinstance(module.train_dataset, VSTDataset)
            assert isinstance(module.val_dataset, VSTDataset)
            assert isinstance(module.test_dataset, VSTDataset)
            assert module.train_dataset.fake is True
            assert module.val_dataset.fake is True
            assert module.test_dataset.fake is True

    def test_fake_mode_train_dataloader_yields_well_shaped_items(self, tmp_path: Path) -> None:
        """End-to-end smoke: fake-mode train loader iterates and produces sane shapes.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
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
            assert _unwrap(item["params"]).shape == (2, len(param_specs["surge_xt"]))
            assert _unwrap(item["mel_spec"]).shape == (2, 2, 128, 401)

    def test_fake_mode_param_spec_name_sizes_param_width(self, tmp_path: Path) -> None:
        """``param_spec_name`` selects the registry spec that sizes fake param width.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        with _set_up_module(
            dataset_root=tmp_path,
            batch_size=2,
            ot=False,
            fake=True,
            use_saved_mean_and_variance=False,
            num_workers=0,
            pin_memory=False,
            param_spec_name="surge_simple",
        ) as module:
            item = next(iter(module.train_dataloader()))
            assert _unwrap(item["params"]).shape == (2, len(param_specs["surge_simple"]))

    def test_setup_unknown_param_spec_name_raises_key_error(self, tmp_path: Path) -> None:
        """An unregistered ``param_spec_name`` fails fast at ``setup()`` with ``KeyError``.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        module = LanceVSTDataModule(
            dataset_root=tmp_path,
            batch_size=2,
            ot=False,
            fake=True,
            use_saved_mean_and_variance=False,
            param_spec_name=ParamSpecName("does_not_exist"),
        )
        with pytest.raises(KeyError, match="does_not_exist"):
            module.setup()


class TestBackCompatAliases:
    """The old Surge-prefixed names bind to the concrete Lance-backed classes."""

    def test_surge_data_module_alias_is_lance_vst_data_module(self) -> None:
        """``SurgeDataModule`` resolves to ``LanceVSTDataModule`` so old ``_target_``s run."""
        assert surge_datamodule.SurgeDataModule is LanceVSTDataModule

    def test_surge_xt_dataset_alias_is_lance_vst_dataset(self) -> None:
        """``SurgeXTDataset`` resolves to ``LanceVSTDataset`` so old ``_target_``s run."""
        assert surge_datamodule.SurgeXTDataset is LanceVSTDataset
