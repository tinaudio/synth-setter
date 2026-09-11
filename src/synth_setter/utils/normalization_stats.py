"""Calibrate online log-mel normalization after checkpoint restoration."""

import copy
import random
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from torch.utils.data import DataLoader

from synth_setter.data.normalization_stats import (
    NORMALIZATION_SAMPLE_LIMIT,
    estimate_log_mel_statistics,
)
from synth_setter.data.vst_datamodule import load_mel_statistics
from synth_setter.models.components.spec_encoder import LogMelFrontend, SpecEncoder

DEFAULT_NORMALIZATION_ESTIMATION_SEED = 1234

type _BroadcastPayload = tuple[str, str | None, np.ndarray | None, np.ndarray | None]


class NormalizationStatsCallback(Callback):
    """Load or estimate online frontend statistics once at fit start."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: int = DEFAULT_NORMALIZATION_ESTIMATION_SEED,
        sample_limit: int = NORMALIZATION_SAMPLE_LIMIT,
    ) -> None:
        """Configure deterministic rank-zero calibration.

        :param output_dir: Fallback directory for newly estimated ``stats.npz``.
        :param seed: Seed for the calibration-only subset and collate RNG.
        :param sample_limit: Maximum number of training waveforms to estimate from.
        """
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.sample_limit = sample_limit

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Apply identical statistics on every rank after checkpoint restoration.

        :param trainer: Active trainer with a fit-stage datamodule and strategy.
        :param pl_module: Model expected to own a :class:`SpecEncoder` at ``encoder``.
        :raises RuntimeError: If rank-zero loading or estimation fails.
        """
        frontend = _frontend_from_model(pl_module)
        payload = None
        if trainer.is_global_zero:
            payload = self._rank_zero_payload(trainer, frontend)
        received = trainer.strategy.broadcast(payload, src=0)
        action, error, mean, std = cast(_BroadcastPayload, received)
        if error is not None:
            raise RuntimeError(f"Normalization statistics calibration failed on rank 0: {error}")
        if action == "apply":
            if mean is None or std is None:
                raise RuntimeError("rank-zero calibration returned no statistics")
            frontend.set_normalization_statistics(torch.from_numpy(mean), torch.from_numpy(std))

    def _rank_zero_payload(self, trainer: Trainer, frontend: LogMelFrontend) -> _BroadcastPayload:
        if frontend.normalization_enabled:
            return "skip", None, None, None
        try:
            datamodule = getattr(trainer, "datamodule", None)
            if datamodule is None:
                raise RuntimeError("normalization estimation requires a datamodule")
            stats_path = _existing_dataset_stats(datamodule)
            output_stats = self.output_dir / "stats.npz"
            if stats_path is None and output_stats.is_file():
                stats_path = output_stats
            if stats_path is not None:
                mean, std = _load_statistics(stats_path)
                frontend.set_normalization_statistics(
                    torch.from_numpy(mean), torch.from_numpy(std)
                )
            else:
                with _seeded_collate_rng(self.seed):
                    mean, std = estimate_log_mel_statistics(
                        _calibration_audio_batches(datamodule, self.seed, self.sample_limit),
                        frontend,
                        self.sample_limit,
                        mask_degenerate=True,
                    )
                frontend.set_normalization_statistics(
                    torch.from_numpy(mean), torch.from_numpy(std)
                )
                _save_statistics(output_stats, mean, std)
        except Exception as exc:  # noqa: BLE001 - every rank must reach the broadcast
            return "error", f"{type(exc).__name__}: {exc}", None, None
        return "apply", None, mean, std


def _frontend_from_model(model: LightningModule) -> LogMelFrontend:
    encoder = getattr(model, "encoder", None)
    frontend = getattr(encoder, "frontend", None)
    if not isinstance(encoder, SpecEncoder) or not isinstance(frontend, LogMelFrontend):
        raise TypeError(
            "estimate_normalization_stats requires a LogMelFrontend at model.encoder.frontend"
        )
    return frontend


def _existing_dataset_stats(datamodule: Any) -> Path | None:
    dataset_root = getattr(datamodule, "dataset_root", None)
    if dataset_root is None:
        return None
    stats_path = Path(dataset_root) / "stats.npz"
    return stats_path if stats_path.is_file() else None


def _load_statistics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mean, std = load_mel_statistics(path)
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def _save_statistics(path: Path, mean: np.ndarray, std: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as file:
        np.savez(file, mean=mean, std=std)
    temporary.replace(path)


def _calibration_audio_batches(
    datamodule: Any, seed: int, sample_limit: int
) -> Iterable[torch.Tensor]:
    loader = datamodule.train_dataloader()
    dataset = loader.dataset
    sample_count = min(len(dataset), sample_limit)
    indices = np.random.default_rng(seed).choice(len(dataset), sample_count, replace=False)
    collate_fn = copy.copy(loader.collate_fn)
    if hasattr(collate_fn, "ot"):
        collate_fn.ot = False
    batch_size = loader.batch_size or getattr(datamodule, "batch_size", 1)
    calibration_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=indices,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return _audio_only(calibration_loader)


def _audio_only(loader: Iterable[dict[str, torch.Tensor]]) -> Iterator[torch.Tensor]:
    for batch in loader:
        audio = batch.get("audio")
        if audio is None:
            raise ValueError("training calibration requires waveform batches under 'audio'")
        if audio.ndim == 3 and audio.shape[1] == 1:
            audio = audio[:, 0]
        yield audio


@contextmanager
def _seeded_collate_rng(seed: int) -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
