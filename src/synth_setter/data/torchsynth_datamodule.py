"""Online TorchSynth datasets and Lightning data module.

Compose ``experiment=torchsynth/ffn`` to sample parameters and render every
audio batch on the training machine without materializing an audio dataset.
"""

from __future__ import annotations

import sys
import threading
import types
from dataclasses import dataclass
from collections.abc import Callable
from functools import cache, partial
from typing import TYPE_CHECKING, TypeAlias, cast

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from synth_setter.data.ot import regular_collate_fn

if TYPE_CHECKING:
    from torchsynth.parameter import ModuleParameter
    from torchsynth.synth import Voice

TorchSynthItem: TypeAlias = tuple[
    torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]
]
TorchSynthBatch: TypeAlias = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]
]
# The odd 64-bit golden-ratio multiplier diffuses nearby split seeds into distinct RNG streams.
_SEED_MIXER = 0x9E3779B97F4A7C15


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


@dataclass
class _Renderer:
    """Own one mutable voice and serialize access to its parameter state.

    .. attribute :: voice

       Cached TorchSynth voice.

    .. attribute :: lock

       Guards parameter mutation and rendering.
    """

    voice: Voice
    lock: threading.Lock


@cache
def _make_renderer(
    sample_rate: int, signal_length: int, batch_size: int = 1, device: str = "cpu"
) -> _Renderer:
    synth_config, voice = _torchsynth_types()
    instance = voice(
        synthconfig=synth_config(
            batch_size=batch_size,
            sample_rate=sample_rate,
            buffer_size_seconds=signal_length / sample_rate,
            reproducible=False,
        )
    )
    return _Renderer(instance.to(torch.device(device)), threading.Lock())


def _synth_parameters(voice: Voice) -> list[ModuleParameter]:
    return [
        parameter
        for (module, _), parameter in voice.get_parameters().items()
        if module != "keyboard"
    ]


def render_torchsynth(
    params: torch.Tensor, *, sample_rate: int, signal_length: int, midi_pitch: int
) -> torch.Tensor:
    """Render normalized TorchSynth parameters into a mono audio batch.

    :param params: Normalized parameter rows in TorchSynth's native order.
    :param sample_rate: Audio sample rate in Hz.
    :param signal_length: Number of output samples.
    :param midi_pitch: Fixed MIDI note rendered for every parameter row.
    :returns: Audio shaped ``(batch, signal_length)``.
    :raises ValueError: The parameter width or rendered audio violates the data contract.
    """
    renderer = _make_renderer(sample_rate, signal_length, len(params), str(params.device))
    voice = renderer.voice
    with renderer.lock:
        all_parameters = voice.get_parameters()
        native = _synth_parameters(voice)
        if params.shape[1] != len(native):
            raise ValueError(f"Expected {len(native)} TorchSynth parameters, got {params.shape[1]}")
        for values, parameter in zip(params.T, native, strict=True):
            parameter.data.copy_(values.nan_to_num(0.5).clamp(1e-4, 1 - 1e-4))
        for name, value in (
            ("midi_f0", float(midi_pitch)),
            ("duration", signal_length / sample_rate),
        ):
            keyboard = all_parameters[("keyboard", name)]
            keyboard.to_0to1(torch.full((len(params),), value, device=params.device))
        with torch.no_grad():
            audio = voice.output()
    if not torch.isfinite(audio).all():
        raise ValueError("TorchSynth audio must be finite")
    return audio.clamp(-1, 1)


class TorchSynthDataset(Dataset[TorchSynthItem]):
    """Deterministic parameters rendered on demand instead of stored as audio."""

    def __init__(
        self, num_samples: int, seed: int, sample_rate: int, signal_length: int, midi_pitch: int
    ) -> None:
        """Bind the sampling seed and audio shape for on-demand rendering.

        :param num_samples: Number of parameter rows the dataset yields.
        :param seed: Base seed folded with the index to draw each row's parameters.
        :param sample_rate: Audio sample rate in Hz.
        :param signal_length: Number of output samples per rendered row.
        :param midi_pitch: Fixed MIDI note rendered for every row.
        """
        self.num_samples = num_samples
        self.seed = seed
        self.sample_rate = sample_rate
        self.signal_length = signal_length
        self.midi_pitch = midi_pitch
        renderer = _make_renderer(sample_rate, signal_length)
        self.num_params = len(_synth_parameters(renderer.voice))

    def __len__(self) -> int:
        """Return the logical number of online samples.

        :returns: Configured split length.
        """
        return self.num_samples

    def __getitem__(
        self, index: int
    ) -> TorchSynthItem:
        """Sample and render one deterministic parameter row.

        :param index: Logical row index.
        :returns: Audio, parameters, and the callable used to render them.
        """
        sample_seed = (self.seed * _SEED_MIXER + index) % sys.maxsize
        generator = torch.Generator().manual_seed(sample_seed)
        params = torch.rand((1, self.num_params), generator=generator)
        render_fn = partial(
            render_torchsynth,
            sample_rate=self.sample_rate,
            signal_length=self.signal_length,
            midi_pitch=self.midi_pitch,
        )
        return render_fn(params), params, render_fn


class TorchSynthDataModule(LightningDataModule):
    """Serve train, validation, and test audio rendered locally by TorchSynth."""

    def __init__(
        self,
        sample_rate: int = 44_100,
        signal_length: int = 4_410,
        midi_pitch: int = 60,
        num_params: int = 76,
        train_val_test_sizes: tuple[int, int, int] = (100_000, 10_000, 10_000),
        train_val_test_seeds: tuple[int, int, int] = (123, 456, 789),
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        """Configure the online TorchSynth train, validation, and test splits.

        :param sample_rate: Audio sample rate in Hz.
        :param signal_length: Number of output samples per rendered row.
        :param midi_pitch: Fixed MIDI note rendered for every parameter row.
        :param num_params: Expected parameter width, validated against TorchSynth in ``setup``.
        :param train_val_test_sizes: Row counts for the train, validation, and test splits.
        :param train_val_test_seeds: Base seeds for the train, validation, and test splits.
        :param batch_size: DataLoader batch size.
        :param num_workers: DataLoader worker process count.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.signal_length = signal_length
        self.midi_pitch = midi_pitch
        self.num_params = num_params
        self.train_val_test_sizes = train_val_test_sizes
        self.train_val_test_seeds = train_val_test_seeds
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        """Build only the splits required for the requested Lightning stage.

        :param stage: Lightning stage name, or ``None`` to build every split.
        :raises ValueError: Configured parameter width differs from TorchSynth.
        """

        def dataset(size: int, seed: int) -> TorchSynthDataset:
            return TorchSynthDataset(
                size,
                seed,
                self.sample_rate,
                self.signal_length,
                self.midi_pitch,
            )

        train_size, val_size, test_size = self.train_val_test_sizes
        train_seed, val_seed, test_seed = self.train_val_test_seeds
        if stage in (None, "fit"):
            self.train, self.val = dataset(train_size, train_seed), dataset(val_size, val_seed)
        elif stage == "validate":
            self.val = dataset(val_size, val_seed)
        if stage in (None, "test", "predict"):
            self.test = dataset(test_size, test_seed)
        renderer = _make_renderer(self.sample_rate, self.signal_length)
        discovered = len(_synth_parameters(renderer.voice))
        if self.num_params != discovered:
            raise ValueError(f"Configured num_params={self.num_params}, TorchSynth exposes {discovered}")

    def _loader(
        self, dataset: Dataset[TorchSynthItem], *, shuffle: bool = False
    ) -> DataLoader[TorchSynthBatch]:
        return cast(
            DataLoader[TorchSynthBatch],
            DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                collate_fn=regular_collate_fn,
            ),
        )

    def train_dataloader(self) -> DataLoader[TorchSynthBatch]:
        """Return the shuffled online training loader.

        :returns: Batched online training data.
        """
        return self._loader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader[TorchSynthBatch]:
        """Return the deterministic online validation loader.

        :returns: Batched online validation data.
        """
        return self._loader(self.val)

    def test_dataloader(self) -> DataLoader[TorchSynthBatch]:
        """Return the deterministic online test loader.

        :returns: Batched online test data.
        """
        return self._loader(self.test)
