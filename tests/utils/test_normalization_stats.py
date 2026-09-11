"""Tests for training-time normalization calibration."""

import random
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset

from synth_setter.models.components.spec_encoder import LogMelFrontend, SpecEncoder
from synth_setter.utils.normalization_stats import NormalizationStatsCallback

_SIGNAL_LENGTH = 4_410
_SAMPLE_RATE = 44_100


class _WaveformDataset(Dataset):
    """Provide deterministic tones through map-style dataset semantics."""

    def __init__(self) -> None:
        """Create six distinct fixed-length waveforms."""
        time = torch.arange(_SIGNAL_LENGTH, dtype=torch.float32) / _SAMPLE_RATE
        self.rows = [
            torch.sin(2 * torch.pi * frequency * time)
            for frequency in (110.0, 220.0, 330.0, 440.0, 550.0, 660.0)
        ]

    def __len__(self) -> int:
        """Return the number of waveforms.

        :returns: Number of available rows.
        """
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one waveform under the production batch key.

        :param index: Row index.
        :returns: Audio batch item.
        """
        return {"audio": self.rows[index]}


class _DataModule:
    """Expose a map-style training loader and dataset statistics root."""

    def __init__(self, dataset_root: Path) -> None:
        """Bind the local dataset root and waveform rows.

        :param dataset_root: Directory where dataset statistics may exist.
        """
        self.dataset_root = dataset_root
        self.dataset = _WaveformDataset()

    def train_dataloader(self) -> DataLoader:
        """Return the shuffled production-shaped waveform loader.

        :returns: Training waveform loader.
        """
        return DataLoader(self.dataset, batch_size=2, shuffle=True)


def _runtime(
    tmp_path: Path,
) -> tuple[Trainer, LightningModule, LogMelFrontend, _DataModule]:
    """Build callback hook arguments around a real frontend.

    :param tmp_path: Temporary dataset root parent.
    :returns: Trainer, model, frontend, and datamodule test collaborators.
    """
    frontend = LogMelFrontend(_SIGNAL_LENGTH, sample_rate=_SAMPLE_RATE)
    model = SimpleNamespace(encoder=SpecEncoder(frontend=frontend, backbone=torch.nn.Identity()))
    trainer = SimpleNamespace(
        datamodule=_DataModule(tmp_path / "dataset"),
        is_global_zero=True,
        lightning_module=model,
        strategy=SimpleNamespace(broadcast=lambda value, src=0: value),
    )
    return cast(Trainer, trainer), cast(LightningModule, model), frontend, trainer.datamodule


def test_callback_existing_dataset_statistics_are_reused(tmp_path: Path) -> None:
    """Dataset-root statistics apply without writing output statistics.

    :param tmp_path: Temporary dataset and output roots.
    """
    trainer, model, frontend, datamodule = _runtime(tmp_path)
    datamodule.dataset_root.mkdir()
    np.savez(
        datamodule.dataset_root / "stats.npz",
        mean=np.array(-2.0, dtype=np.float32),
        std=np.array(4.0, dtype=np.float32),
    )
    waveform = datamodule.dataset.rows[0].unsqueeze(0)
    raw = frontend.forward_raw(waveform)

    NormalizationStatsCallback(tmp_path / "output").on_fit_start(trainer, model)

    torch.testing.assert_close(frontend(waveform), (raw + 2.0) / 4.0)
    assert not (tmp_path / "output" / "stats.npz").exists()


def test_callback_existing_output_statistics_are_reused(tmp_path: Path) -> None:
    """A prior output estimate is reused without calibration.

    :param tmp_path: Temporary dataset and output roots.
    """
    trainer, model, frontend, datamodule = _runtime(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    np.savez(
        output_dir / "stats.npz",
        mean=np.array(-2.0, dtype=np.float32),
        std=np.array(4.0, dtype=np.float32),
    )
    waveform = datamodule.dataset.rows[0].unsqueeze(0)
    raw = frontend.forward_raw(waveform)

    NormalizationStatsCallback(output_dir).on_fit_start(trainer, model)

    torch.testing.assert_close(frontend(waveform), (raw + 2.0) / 4.0)


def test_callback_invalid_existing_statistics_propagate_rank_zero_error(tmp_path: Path) -> None:
    """Invalid rank-zero artifacts become a broadcast-safe failure.

    :param tmp_path: Temporary output root.
    """
    trainer, model, _, _ = _runtime(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    np.savez(
        output_dir / "stats.npz",
        mean=np.array(float("nan"), dtype=np.float32),
        std=np.array(1.0, dtype=np.float32),
    )

    with pytest.raises(RuntimeError, match="failed on rank 0.*finite"):
        NormalizationStatsCallback(output_dir).on_fit_start(trainer, model)


def test_callback_missing_dataset_statistics_are_estimated_and_persisted(tmp_path: Path) -> None:
    """Missing statistics are persisted and consumed by the real frontend.

    :param tmp_path: Temporary dataset and output roots.
    """
    trainer, model, frontend, datamodule = _runtime(tmp_path)

    NormalizationStatsCallback(tmp_path / "output", seed=7, sample_limit=4).on_fit_start(
        trainer, model
    )

    stats_path = tmp_path / "output" / "stats.npz"
    assert stats_path.is_file()
    waveform = datamodule.dataset.rows[0].unsqueeze(0)
    with np.load(stats_path) as stats:
        expected = (
            frontend.forward_raw(waveform) - torch.from_numpy(stats["mean"])
        ) / torch.from_numpy(stats["std"])
    torch.testing.assert_close(frontend(waveform), expected)


def test_callback_seeded_sampling_does_not_change_global_rng(tmp_path: Path) -> None:
    """Calibration leaves the training process's global RNGs untouched.

    :param tmp_path: Temporary output root.
    """
    trainer, model, _, _ = _runtime(tmp_path)
    torch.manual_seed(91)
    np.random.seed(91)
    random.seed(91)
    torch_state = torch.random.get_rng_state().clone()
    expected_numpy = np.random.random(3)
    np.random.seed(91)
    python_state = random.getstate()

    NormalizationStatsCallback(tmp_path / "output", seed=7, sample_limit=4).on_fit_start(
        trainer, model
    )

    assert torch.equal(torch.random.get_rng_state(), torch_state)
    np.testing.assert_array_equal(np.random.random(3), expected_numpy)
    assert random.getstate() == python_state


def test_callback_restored_frontend_statistics_skip_artifacts(tmp_path: Path) -> None:
    """Restored frontend buffers bypass all calibration artifacts.

    :param tmp_path: Temporary output root.
    """
    trainer, model, frontend, _ = _runtime(tmp_path)
    frontend.set_normalization_statistics(torch.zeros(1), torch.ones(1))

    NormalizationStatsCallback(tmp_path / "output").on_fit_start(trainer, model)

    assert not (tmp_path / "output" / "stats.npz").exists()
