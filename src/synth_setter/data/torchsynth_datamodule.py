"""Online TorchSynth datasets and Lightning data module."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from functools import cache, partial
from typing import TYPE_CHECKING

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from synth_setter.data.ot import regular_collate_fn

if TYPE_CHECKING:
    from torchsynth.parameter import ModuleParameter
    from torchsynth.synth import Voice


@cache
def _torchsynth_types() -> tuple[type, type]:
    """Import TorchSynth's ``SynthConfig`` and ``Voice``, restoring the module it expects.

    torchsynth 1.0.2 imports ``pytorch_lightning.core.lightning``, a path removed in
    pytorch-lightning >= 2.0; shim it back once before importing so the package loads.

    :returns: TorchSynth's ``SynthConfig`` and ``Voice`` types.
    """
    try:
        import pytorch_lightning.core.lightning  # noqa: F401
    except ModuleNotFoundError:
        import pytorch_lightning

        shim = types.ModuleType("pytorch_lightning.core.lightning")
        setattr(shim, "LightningModule", pytorch_lightning.LightningModule)
        sys.modules["pytorch_lightning.core.lightning"] = shim
    from torchsynth.config import SynthConfig
    from torchsynth.synth import Voice

    return SynthConfig, Voice


def _make_voice(sample_rate: int, signal_length: int, batch_size: int = 1) -> Voice:
    synth_config, voice = _torchsynth_types()
    return voice(
        synthconfig=synth_config(
            batch_size=batch_size,
            sample_rate=sample_rate,
            buffer_size_seconds=signal_length / sample_rate,
            reproducible=False,
        )
    )


def _synth_parameters(voice: Voice) -> list[ModuleParameter]:
    return [
        parameter
        for (module, _), parameter in voice.get_parameters().items()
        if module != "keyboard"
    ]


def render_torchsynth(params: torch.Tensor, *, sample_rate: int, signal_length: int) -> torch.Tensor:
    """Render normalized TorchSynth parameters into a mono audio batch.

    :param params: Normalized parameter rows in TorchSynth's native order.
    :param sample_rate: Audio sample rate in Hz.
    :param signal_length: Number of output samples.
    :returns: Audio shaped ``(batch, signal_length)``.
    :raises ValueError: The parameter width does not match TorchSynth's voice.
    """
    voice = _make_voice(sample_rate, signal_length, len(params))
    voice.to(params.device)
    all_parameters = voice.get_parameters()
    native = _synth_parameters(voice)
    if params.shape[1] != len(native):
        raise ValueError(f"Expected {len(native)} TorchSynth parameters, got {params.shape[1]}")
    for values, parameter in zip(params.T, native, strict=True):
        parameter.data.copy_(values.clamp(0.0, 1.0))
    for name, value in (("midi_f0", 60.0), ("duration", signal_length / sample_rate)):
        keyboard = all_parameters[("keyboard", name)]
        keyboard.to_0to1(torch.full((len(params),), value, device=params.device))
    with torch.no_grad():
        return voice.output()


class TorchSynthDataset(Dataset):
    """Deterministic parameters rendered on demand instead of stored as audio."""

    def __init__(self, num_samples: int, seed: int, sample_rate: int, signal_length: int) -> None:
        """Bind the sampling seed and audio shape for on-demand rendering.

        :param num_samples: Number of parameter rows the dataset yields.
        :param seed: Base seed folded with the index to draw each row's parameters.
        :param sample_rate: Audio sample rate in Hz.
        :param signal_length: Number of output samples per rendered row.
        """
        self.num_samples = num_samples
        self.seed = seed
        self.sample_rate = sample_rate
        self.signal_length = signal_length
        self.num_params = len(_synth_parameters(_make_voice(sample_rate, signal_length)))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        sample_seed = (self.seed * 0x9E3779B97F4A7C15 + index) % sys.maxsize
        generator = torch.Generator().manual_seed(sample_seed)
        params = torch.rand((1, self.num_params), generator=generator)
        render_fn = partial(
            render_torchsynth,
            sample_rate=self.sample_rate,
            signal_length=self.signal_length,
        )
        return render_fn(params), params, render_fn


class TorchSynthDataModule(LightningDataModule):
    """Serve train, validation, and test audio rendered locally by TorchSynth."""

    def __init__(
        self,
        sample_rate: int = 44_100,
        signal_length: int = 4_410,
        num_params: int = 76,
        train_val_test_sizes: tuple[int, int, int] = (100_000, 10_000, 10_000),
        train_val_test_seeds: tuple[int, int, int] = (123, 456, 789),
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        """Configure the online TorchSynth train, validation, and test splits.

        :param sample_rate: Audio sample rate in Hz.
        :param signal_length: Number of output samples per rendered row.
        :param num_params: Expected parameter width, validated against TorchSynth in ``setup``.
        :param train_val_test_sizes: Row counts for the train, validation, and test splits.
        :param train_val_test_seeds: Base seeds for the train, validation, and test splits.
        :param batch_size: DataLoader batch size.
        :param num_workers: DataLoader worker process count.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.signal_length = signal_length
        self.num_params = num_params
        self.train_val_test_sizes = train_val_test_sizes
        self.train_val_test_seeds = train_val_test_seeds
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        datasets = [
            TorchSynthDataset(size, seed, self.sample_rate, self.signal_length)
            for size, seed in zip(self.train_val_test_sizes, self.train_val_test_seeds, strict=True)
        ]
        discovered = datasets[0].num_params
        if self.num_params != discovered:
            raise ValueError(f"Configured num_params={self.num_params}, TorchSynth exposes {discovered}")
        self.train, self.val, self.test = datasets

    def _loader(self, dataset: Dataset, *, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=regular_collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test)
