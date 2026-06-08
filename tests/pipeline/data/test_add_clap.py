"""Tests for `synth_setter.pipeline.data.add_clap` HDF5 embedding injection.

The CLAP model itself is never loaded here: the HDF5-plumbing tests inject a
fake ``encode`` callable (dataset creation, batching, channel downmix,
sample-rate propagation, idempotency), and the encoder-contract test mocks the
``transformers`` boundary to pin the v5 call shape (``audio=`` kwarg, 48 kHz
resample, ``pooler_output`` projection).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

import h5py
import hdf5plugin  # noqa: F401  # side-effect import: registers Blosc2 so h5py can read the clap field
import numpy as np
import pytest

from synth_setter.pipeline.data import add_clap


def _clap(f: h5py.File) -> h5py.Dataset:
    """Return the ``clap`` dataset narrowed from h5py's group/dataset union.

    :param f: Open HDF5 file holding a ``clap`` dataset.
    :returns: The ``clap`` dataset.
    """
    return cast(h5py.Dataset, f["clap"])


def _write_audio_h5(
    path: Path,
    audio: np.ndarray,
    *,
    sample_rate: int | None = None,
) -> None:
    """Write a one-dataset shard holding only ``audio`` (optionally a sample_rate attr).

    :param path: Destination ``.h5`` file.
    :param audio: ``(N, C, T)`` array written verbatim as the ``audio`` dataset.
    :param sample_rate: When set, stored as the ``sample_rate`` attr the encoder reads.
    """
    with h5py.File(path, "w") as f:
        dset = f.create_dataset("audio", data=audio.astype(np.float16))
        if sample_rate is not None:
            dset.attrs["sample_rate"] = sample_rate


def _width_four_encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """Fake encoder: column 0 = per-row mean, column 1 = sample_rate, rest zero.

    Encoding the mono mean lets a test assert row order and channel downmix; the
    sample_rate column lets a test assert the attr was propagated.

    :param mono: ``(B, T)`` mono batch.
    :param sample_rate: Rate forwarded from the caller.
    :returns: ``(B, 4)`` float32 embedding batch.
    """
    out = np.zeros((mono.shape[0], 4), dtype=np.float32)
    out[:, 0] = mono.mean(axis=1)
    out[:, 1] = sample_rate
    return out


def test_add_clap_embeddings_creates_field_with_encoder_width_and_float32(
    tmp_path: Path,
) -> None:
    """The ``clap`` dataset is created with the encoder's output width as float32.

    :param tmp_path: Per-test tmpdir.
    """
    path = tmp_path / "train.h5"
    _write_audio_h5(path, np.zeros((5, 2, 8), dtype=np.float32))

    add_clap.add_clap_embeddings(path, _width_four_encode, batch_size=2)

    with h5py.File(path, "r") as f:
        assert _clap(f).shape == (5, 4)
        assert _clap(f).dtype == np.float32


def test_add_clap_embeddings_preserves_row_order_across_ragged_batches(
    tmp_path: Path,
) -> None:
    """Each row's embedding stays aligned to its audio row when N is not a batch multiple.

    :param tmp_path: Per-test tmpdir.
    """
    path = tmp_path / "train.h5"
    # Row i is filled with the constant i, so the fake encoder's mean column == i.
    audio = np.stack([np.full((2, 8), float(i)) for i in range(5)]).astype(np.float32)
    _write_audio_h5(path, audio)

    add_clap.add_clap_embeddings(path, _width_four_encode, batch_size=2)

    with h5py.File(path, "r") as f:
        assert _clap(f)[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_add_clap_embeddings_downmixes_stereo_to_mono_mean(tmp_path: Path) -> None:
    """A row's channels are averaged before encoding (L=2, R=4 → mono 3).

    :param tmp_path: Per-test tmpdir.
    """
    path = tmp_path / "train.h5"
    audio = np.empty((1, 2, 8), dtype=np.float32)
    audio[:, 0, :] = 2.0
    audio[:, 1, :] = 4.0
    _write_audio_h5(path, audio)

    add_clap.add_clap_embeddings(path, _width_four_encode, batch_size=4)

    with h5py.File(path, "r") as f:
        assert _clap(f)[0, 0] == 3.0


def test_add_clap_embeddings_reads_sample_rate_from_audio_attrs(tmp_path: Path) -> None:
    """The encoder receives the shard's stored ``sample_rate`` attr.

    :param tmp_path: Per-test tmpdir.
    """
    path = tmp_path / "train.h5"
    _write_audio_h5(path, np.zeros((1, 2, 8), dtype=np.float32), sample_rate=48000)

    add_clap.add_clap_embeddings(path, _width_four_encode, batch_size=4)

    with h5py.File(path, "r") as f:
        assert _clap(f)[0, 1] == 48000.0


def test_add_clap_embeddings_defaults_sample_rate_when_attr_absent(tmp_path: Path) -> None:
    """With no ``sample_rate`` attr the encoder receives the 44100 project default.

    :param tmp_path: Per-test tmpdir.
    """
    path = tmp_path / "train.h5"
    _write_audio_h5(path, np.zeros((1, 2, 8), dtype=np.float32))

    add_clap.add_clap_embeddings(path, _width_four_encode, batch_size=4)

    with h5py.File(path, "r") as f:
        assert _clap(f)[0, 1] == float(add_clap.DEFAULT_SAMPLE_RATE)


def _exploding_encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """Encoder that fails if invoked — proves a skip path bypasses encoding.

    :param mono: Unused mono batch.
    :param sample_rate: Unused sample rate.
    :returns: Never returns.
    :raises AssertionError: Always.
    """
    raise AssertionError("encoder must not run on a completed field")


def _write_clap_field(path: Path, data: np.ndarray, *, complete: bool) -> None:
    """Pre-create a ``clap`` dataset, optionally marked complete.

    :param path: HDF5 file to mutate.
    :param data: Values for the ``clap`` dataset.
    :param complete: Whether to set the completion marker attr.
    """
    with h5py.File(path, "r+") as f:
        dset = f.create_dataset("clap", data=data)
        if complete:
            dset.attrs[add_clap.COMPLETE_ATTR] = True


def test_add_clap_embeddings_skips_completed_field_without_overwrite(tmp_path: Path) -> None:
    """A field carrying the completion marker is left untouched and the encoder never runs.

    :param tmp_path: Per-test tmpdir.
    """
    path = tmp_path / "train.h5"
    _write_audio_h5(path, np.zeros((2, 2, 8), dtype=np.float32))
    _write_clap_field(path, np.full((2, 4), 7.0, dtype=np.float32), complete=True)

    written = add_clap.add_clap_embeddings(path, _exploding_encode, batch_size=2)

    assert written == 0
    with h5py.File(path, "r") as f:
        assert np.all(_clap(f)[:] == 7.0)


def test_add_clap_embeddings_recomputes_incomplete_field(tmp_path: Path) -> None:
    """A present-but-incomplete field (crashed run) is recomputed, not trusted.

    :param tmp_path: Per-test tmpdir.
    """
    path = tmp_path / "train.h5"
    _write_audio_h5(path, np.zeros((2, 2, 8), dtype=np.float32))
    _write_clap_field(path, np.full((2, 4), 7.0, dtype=np.float32), complete=False)

    written = add_clap.add_clap_embeddings(path, _width_four_encode, batch_size=2)

    assert written == 2
    with h5py.File(path, "r") as f:
        assert np.all(_clap(f)[:, 0] == 0.0)


def test_add_clap_embeddings_marks_field_complete_after_writing(tmp_path: Path) -> None:
    """A fully-written field carries the completion marker so reruns can skip it.

    :param tmp_path: Per-test tmpdir.
    """
    path = tmp_path / "train.h5"
    _write_audio_h5(path, np.zeros((3, 2, 8), dtype=np.float32))

    add_clap.add_clap_embeddings(path, _width_four_encode, batch_size=2)

    with h5py.File(path, "r") as f:
        assert _clap(f).attrs[add_clap.COMPLETE_ATTR]


def test_add_clap_embeddings_overwrites_completed_field_when_requested(tmp_path: Path) -> None:
    """``overwrite=True`` recomputes even a completed ``clap`` field.

    :param tmp_path: Per-test tmpdir.
    """
    path = tmp_path / "train.h5"
    _write_audio_h5(path, np.zeros((2, 2, 8), dtype=np.float32))
    _write_clap_field(path, np.full((2, 4), 7.0, dtype=np.float32), complete=True)

    add_clap.add_clap_embeddings(path, _width_four_encode, batch_size=2, overwrite=True)

    with h5py.File(path, "r") as f:
        assert np.all(_clap(f)[:, 0] == 0.0)


def test_add_clap_embeddings_raises_when_width_differs_from_expected_dim(tmp_path: Path) -> None:
    """A ``expected_dim`` mismatch with the encoder width raises before writing.

    :param tmp_path: Per-test tmpdir.
    """
    path = tmp_path / "train.h5"
    _write_audio_h5(path, np.zeros((2, 2, 8), dtype=np.float32))

    # _width_four_encode emits width 4; expecting 512 must fail.
    with pytest.raises(ValueError, match="width 4, expected 512"):
        add_clap.add_clap_embeddings(path, _width_four_encode, batch_size=2, expected_dim=512)

    with h5py.File(path, "r") as f:
        assert "clap" not in f


def test_add_clap_embeddings_accepts_matching_expected_dim(tmp_path: Path) -> None:
    """A ``expected_dim`` equal to the encoder width writes normally.

    :param tmp_path: Per-test tmpdir.
    """
    path = tmp_path / "train.h5"
    _write_audio_h5(path, np.zeros((2, 2, 8), dtype=np.float32))

    add_clap.add_clap_embeddings(path, _width_four_encode, batch_size=2, expected_dim=4)

    with h5py.File(path, "r") as f:
        assert _clap(f).shape == (2, 4)


def test_load_clap_audio_encoder_resamples_to_48k_and_returns_pooler_output() -> None:
    """The encoder closure resamples to 48 kHz, passes ``audio=``, and returns pooler_output.

    Mocks the ``transformers``/``torchaudio`` boundary so the v5 call contract is
    pinned without a model download (regression guard for the ``audio=`` kwarg,
    the 48 kHz resample, and the ``pooler_output`` projection).
    """
    import torch

    captured: dict[str, object] = {}

    def fake_processor(audio: object, sampling_rate: int, return_tensors: str) -> dict:
        captured["sampling_rate"] = sampling_rate
        captured["batch"] = len(audio)  # type: ignore[arg-type]
        return {"input_features": torch.zeros((len(audio), 1))}  # type: ignore[arg-type]

    def fake_get_audio_features(**kwargs: object) -> SimpleNamespace:
        rows = kwargs["input_features"].shape[0]  # type: ignore[union-attr]
        return SimpleNamespace(pooler_output=torch.ones((rows, 4), dtype=torch.float32))

    fake_model = mock.Mock()
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model
    fake_model.get_audio_features.side_effect = fake_get_audio_features

    with (
        mock.patch("transformers.ClapModel.from_pretrained", return_value=fake_model),
        mock.patch("transformers.ClapProcessor.from_pretrained", return_value=fake_processor),
        mock.patch("torch.cuda.is_available", return_value=False),
        mock.patch(
            "torchaudio.functional.resample", side_effect=lambda wav, orig, new: wav
        ) as resample,
    ):
        encode = add_clap.load_clap_audio_encoder()
        out = encode(np.zeros((2, 16), dtype=np.float32), add_clap.DEFAULT_SAMPLE_RATE)

    assert out.shape == (2, 4)
    assert captured["sampling_rate"] == add_clap.CLAP_SAMPLE_RATE
    resample.assert_called_once()
    assert resample.call_args.args[1:] == (add_clap.DEFAULT_SAMPLE_RATE, add_clap.CLAP_SAMPLE_RATE)
