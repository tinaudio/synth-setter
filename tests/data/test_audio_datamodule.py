"""Tests for ``AudioFolderDataset`` file discovery."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from pedalboard.io import AudioFile
from torch.utils.data import SequentialSampler

from synth_setter.data.audio_datamodule import AudioDataModule, AudioFolderDataset


def _write_wav(path: Path, seconds: float = 0.5, sample_rate: int = 44100) -> Path:
    """Write a small stereo float32 WAV.

    :param path: Destination path.
    :param seconds: Duration.
    :param sample_rate: Sample rate.
    :returns: ``path``.
    """
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    with AudioFile(str(path), "w", samplerate=float(sample_rate), num_channels=2) as f:
        f.write(np.stack([tone, tone]))
    return path


def test_default_glob_discovers_only_wav_files(tmp_path: Path) -> None:
    """The root glob picks up .wav files and ignores other extensions.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    kept = _write_wav(tmp_path / "keep.wav")
    (tmp_path / "skip.txt").write_text("not audio")
    (tmp_path / "skip.wav.tmp").write_text("partial capture")

    dataset = AudioFolderDataset(root=str(tmp_path))

    assert dataset.files == [kept]


def test_default_glob_orders_audio_files_by_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prediction order is stable regardless of directory enumeration order.

    :param tmp_path: Audio root populated in reverse lexical order.
    :param monkeypatch: Forces the filesystem glob to return reverse lexical order.
    """
    second = _write_wav(tmp_path / "b.wav")
    first = _write_wav(tmp_path / "a.wav")
    original_glob = Path.glob

    def _reverse_glob(path: Path, pattern: str) -> Iterator[Path]:
        return reversed(list(original_glob(path, pattern)))

    monkeypatch.setattr(Path, "glob", _reverse_glob)

    dataset = AudioFolderDataset(root=str(tmp_path))

    assert dataset.files == [first, second]


def test_explicit_files_skip_the_root_glob(tmp_path: Path) -> None:
    """An explicit file list is used verbatim, ignoring the root's contents.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    _write_wav(tmp_path / "neighbor.wav")
    target = _write_wav(tmp_path / "target.wav")

    dataset = AudioFolderDataset(root=str(tmp_path), files=[target])

    assert dataset.files == [target]
    assert dataset[0]["mel"].shape[0] == 2


def test_audio_folder_dataset_preserves_leading_pad_and_half_scale(
    tmp_path: Path,
) -> None:
    """Dataset audio retains its 50 ms leading pad and half-amplitude policy.

    :param tmp_path: Holds the source WAV.
    """
    source = _write_wav(tmp_path / "source.wav")

    audio = AudioFolderDataset(root=str(tmp_path), files=[source])[0]["audio"]

    assert audio.shape == (2, 176400)
    assert audio.dtype == torch.float32
    assert torch.count_nonzero(audio[:, :2205]) == 0
    assert audio.abs().max().item() == pytest.approx(0.1, abs=1e-4)


def test_empty_root_yields_empty_dataset(tmp_path: Path) -> None:
    """An empty root produces a zero-length dataset rather than raising.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    dataset = AudioFolderDataset(root=str(tmp_path))

    assert len(dataset) == 0


def test_predict_dataloader_preserves_dataset_order(tmp_path: Path) -> None:
    """Audio prediction iterates sequentially so outputs retain source ordering.

    :param tmp_path: Empty audio root used to initialize the prediction dataset.
    """
    datamodule = AudioDataModule(root=str(tmp_path))
    datamodule.setup("predict")

    assert isinstance(datamodule.predict_dataloader().sampler, SequentialSampler)
