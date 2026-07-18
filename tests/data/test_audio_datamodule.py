"""Tests for ``AudioFolderDataset`` file discovery."""

from pathlib import Path

import numpy as np
from pedalboard.io import AudioFile

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


def test_explicit_files_skip_the_root_glob(tmp_path: Path) -> None:
    """An explicit file list is used verbatim, ignoring the root's contents.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    _write_wav(tmp_path / "neighbor.wav")
    target = _write_wav(tmp_path / "target.wav")

    dataset = AudioFolderDataset(root=str(tmp_path), files=[target])

    assert dataset.files == [target]
    assert dataset[0]["mel_spec"].shape[0] == 2


def test_empty_root_yields_empty_dataset(tmp_path: Path) -> None:
    """An empty root produces a zero-length dataset rather than raising.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    dataset = AudioFolderDataset(root=str(tmp_path))

    assert len(dataset) == 0


def test_predict_dataloader_param_spec_name_keeps_audio_batches_unchanged(
    tmp_path: Path,
) -> None:
    """``param_spec_name`` is accepted as metadata without altering audio batches.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    _write_wav(tmp_path / "capture.wav")

    baseline = AudioDataModule(root=str(tmp_path), batch_size=1, shuffle=False)
    baseline.setup(stage="predict")
    baseline_batch = next(iter(baseline.predict_dataloader()))

    with_spec = AudioDataModule(
        root=str(tmp_path),
        batch_size=1,
        shuffle=False,
        param_spec_name="surge_xt",
    )
    with_spec.setup(stage="predict")
    with_spec_batch = next(iter(with_spec.predict_dataloader()))

    assert with_spec.param_spec_name == "surge_xt"
    assert baseline_batch["audio"].shape == with_spec_batch["audio"].shape
    assert baseline_batch["mel_spec"].shape == with_spec_batch["mel_spec"].shape
    assert np.array_equal(baseline_batch["audio"].numpy(), with_spec_batch["audio"].numpy())
    assert np.array_equal(
        baseline_batch["mel_spec"].numpy(), with_spec_batch["mel_spec"].numpy()
    )
