"""Online TorchSynth datasets and Lightning data module.

Compose ``experiment=torchsynth/ffn`` to sample parameters and render every
audio batch on the training machine without materializing an audio dataset.
"""

from __future__ import annotations

import sys
import threading
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cache, partial
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler

from synth_setter.data.ot import regular_collate_fn
from synth_setter.data.sample_seed import derive_sample_seed

# Re-exported for backward compat: training code imports these names from this module.
from synth_setter.data.vst.torchsynth_param_spec import (
    INFERABLE_SPEC as INFERABLE_SPEC,
    NUM_PARAMS as NUM_PARAMS,
    PARAM_SPEC as PARAM_SPEC,
    TorchSynthParam as TorchSynthParam,
)
from synth_setter.data.vst.torchsynth_param_spec import (
    TORCHSYNTH_FULL_PARAM_SPEC,
    note_on_duration,
)
from synth_setter.data.vst.torchsynth_param_spec import (
    verify_voice_matches_spec as _verify_voice_matches_spec,
)

if TYPE_CHECKING:
    from torchsynth.synth import Voice

TorchSynthItem: TypeAlias = tuple[
    torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]
]
TorchSynthBatch: TypeAlias = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]
]
# Finite params clamp into the open interval (0, 1) because model predictions are unconstrained.
# NaN/Inf signal divergence or a pipeline bug and raise instead.
_PARAM_CLAMP_EPS = 1e-4


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
        # setattr (not ``shim.LightningModule = ...``) so pyright doesn't flag the
        # attribute as unknown on a dynamically created ModuleType.
        setattr(shim, "LightningModule", pytorch_lightning.LightningModule)
        sys.modules["pytorch_lightning.core.lightning"] = shim
    from torchsynth.config import SynthConfig
    from torchsynth.synth import Voice

    return SynthConfig, Voice


@dataclass
class _Renderer:
    """Own one mutable voice and serialize access to its parameter state.

    .. attribute :: voice

       Mutated only while ``lock`` is held.

    .. attribute :: lock

       Serializes callers sharing the cached voice.
    """

    voice: Voice
    lock: threading.Lock


# Production caches only batch_size=1 items and metric re-render val batch sizes.
# Batch/GPU rendering needs eviction or a fixed renderer size — see #1820.
@cache
def _make_renderer(
    sample_rate: int, signal_length: int, batch_size: int = 1, device: str = "cpu"
) -> _Renderer:
    """Return the process-local renderer for one audio geometry and device.

    :param sample_rate: Audio sample rate in Hz.
    :param signal_length: Number of output samples.
    :param batch_size: Voice batch size.
    :param device: Torch device string.
    :returns: Cached voice and its mutation lock.
    """
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


def _delay_by_note_start(
    audio: torch.Tensor, start_seconds: torch.Tensor, sample_rate: int
) -> torch.Tensor:
    """Shift each row right by its note-on offset, zero-filling the head.

    The voice always starts its note at sample 0, so the offset a note window declares is
    emulated here; a note starting at or past the buffer end renders as silence.

    :param audio: Rendered audio shaped ``(batch, signal_length)``.
    :param start_seconds: Non-negative note-on offset per row, shaped ``(batch,)``.
    :param sample_rate: Audio sample rate in Hz.
    :returns: Delayed audio of the same shape and dtype.
    """
    offsets = (start_seconds * sample_rate).round().to(torch.int64).unsqueeze(1)
    source = torch.arange(audio.shape[1], device=audio.device).unsqueeze(0) - offsets
    return torch.where(source >= 0, audio.gather(1, source.clamp(min=0)), 0.0)


def render_torchsynth(
    params: torch.Tensor, *, sample_rate: int, signal_length: int
) -> torch.Tensor:
    """Render ``torchsynth_full``-encoded rows into a mono audio batch.

    A row is the only thing that determines its audio: pitch and the note window come
    from the row's own note columns, so no caller can supply note conditioning that
    contradicts what the row encodes. Note columns clamp into ``[0, 1]`` and the window
    they decode to clamps into ``KEYBOARD_DURATION_BOUNDS`` — unconstrained model
    predictions and short spec draws must stay renderable rather than raise.

    :param params: Finite float32 rows shaped ``(batch, torchsynth_full encoded width)``;
        synth values clamp strictly inside ``(0, 1)``.
    :param sample_rate: Audio sample rate in Hz.
    :param signal_length: Number of output samples.
    :returns: Float32 audio shaped ``(batch, signal_length)``.
    :raises ValueError: The row width, a non-finite value, or the rendered audio
        violates the data contract.
    """
    if not torch.isfinite(params).all():
        raise ValueError("TorchSynth params must be finite")
    width = TORCHSYNTH_FULL_PARAM_SPEC.encoded_width
    if params.shape[1] != width:
        raise ValueError(f"Expected {width} encoded parameter columns, got {params.shape[1]}")
    notes = [
        TORCHSYNTH_FULL_PARAM_SPEC.decode(row)[1]
        for row in params.detach().clamp(0, 1).cpu().numpy()
    ]
    column = partial(torch.tensor, dtype=torch.float32, device=params.device)
    # Slice the synth columns explicitly: only they reach the voice's parameters. The note
    # columns carry no gradient by construction — pitch is a discrete category, and duration
    # lands on ADSR segment boundaries through integer sample arithmetic — so a differentiable
    # render must never see them as if they were continuous knobs.
    synth_params = params[:, TORCHSYNTH_FULL_PARAM_SPEC.synth_columns]
    renderer = _make_renderer(sample_rate, signal_length, len(params), str(params.device))
    voice = renderer.voice
    with renderer.lock:
        all_parameters = voice.get_parameters()
        native = [all_parameters[(spec.module, spec.name)] for spec in INFERABLE_SPEC]
        for values, parameter in zip(synth_params.T, native, strict=True):
            parameter.data.copy_(values.clamp(_PARAM_CLAMP_EPS, 1 - _PARAM_CLAMP_EPS))
        keyboard = (
            ("midi_f0", column([note["pitch"] for note in notes])),
            (
                "duration",
                column([note_on_duration(note["note_start_and_end"]) for note in notes]),
            ),
        )
        for name, value in keyboard:
            all_parameters[("keyboard", name)].to_0to1(value)
        with torch.no_grad():
            audio = voice.output()
    if not torch.isfinite(audio).all():
        raise ValueError("TorchSynth audio must be finite")
    starts = column([note["note_start_and_end"][0] for note in notes])
    return _delay_by_note_start(audio.as_subclass(torch.Tensor).clamp(-1, 1), starts, sample_rate)


class TorchSynthDataset(Dataset[TorchSynthItem]):
    """Deterministic parameters rendered on demand instead of stored as audio."""

    def __init__(
        self, num_samples: int, seed: int, sample_rate: int, signal_length: int
    ) -> None:
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

    def __len__(self) -> int:
        """Return the logical number of online samples.

        :returns: Configured split length.
        """
        return self.num_samples

    def __getitem__(self, index: int) -> TorchSynthItem:
        """Sample and render one deterministic parameter row.

        Synth values, pitch, and the note window all come from ``torchsynth_full``'s own
        sampler, so the row is the spec's encoding — the same contract every offline path
        emits — and the render is driven by the row rather than by configuration.

        :param index: Logical row index.
        :returns: Float32 audio shaped ``(1, signal_length)``, float32 parameters shaped ``(1,
            torchsynth_full encoded width)``, and the renderer callable.
        """
        rng = np.random.default_rng(derive_sample_seed(self.seed, index))
        synth_values, note_params = TORCHSYNTH_FULL_PARAM_SPEC.sample(rng)
        row = TORCHSYNTH_FULL_PARAM_SPEC.encode(synth_values, note_params)
        params = torch.from_numpy(row).unsqueeze(0)
        # Per-sample CPU render; render_fn is passed through so a future collate can
        # batch/GPU-render instead of paying Voice.output() per row — see #1820.
        render_fn = partial(
            render_torchsynth,
            sample_rate=self.sample_rate,
            signal_length=self.signal_length,
        )
        return render_fn(params), params, render_fn


class _FreshEpochSampler(Sampler[int]):
    """Yield a never-repeating index block per epoch so each epoch draws fresh rows.

    Indices map to i.i.d. seeded parameter rows, so sequential blocks are already unordered draws
    and need no within-epoch shuffle.
    """

    def __init__(self, num_samples: int) -> None:
        """Bind the per-epoch block length.

        :param num_samples: Number of indices yielded per epoch.
        """
        self.num_samples = num_samples
        self._epoch = 0

    def __iter__(self) -> Iterator[int]:
        """Advance to the next index block.

        :returns: Iterator over this epoch's fresh logical indices.
        """
        start = self._epoch * self.num_samples
        self._epoch += 1
        return iter(range(start, start + self.num_samples))

    def __len__(self) -> int:
        """Return the per-epoch sample count.

        :returns: Configured block length.
        """
        return self.num_samples


class TorchSynthDataModule(LightningDataModule):
    """Serve train, validation, and test audio rendered locally by TorchSynth."""

    def __init__(
        self,
        sample_rate: int = 44_100,
        signal_length: int = 4_410,
        num_params: int = TORCHSYNTH_FULL_PARAM_SPEC.encoded_width,
        train_val_test_sizes: tuple[int, int, int] = (100_000, 10_000, 10_000),
        train_val_test_seeds: tuple[int, int, int] = (123, 456, 789),
        batch_size: int = 32,
        num_workers: int = 0,
        resample_train_per_epoch: bool = False,
    ) -> None:
        """Configure the online TorchSynth train, validation, and test splits.

        :param sample_rate: Audio sample rate in Hz.
        :param signal_length: Number of output samples per rendered row.
        :param num_params: Expected row width, validated against the spec in ``setup``.
        :param train_val_test_sizes: Row counts for the train, validation, and test splits.
        :param train_val_test_seeds: Base seeds for the train, validation, and test splits.
        :param batch_size: DataLoader batch size.
        :param num_workers: DataLoader worker process count.
        :param resample_train_per_epoch: Draw fresh train rows every epoch (truly online
            training) instead of revisiting one fixed split; validation and test stay fixed.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.signal_length = signal_length
        self.num_params = num_params
        self.train_val_test_sizes = train_val_test_sizes
        self.train_val_test_seeds = train_val_test_seeds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resample_train_per_epoch = resample_train_per_epoch

    def setup(self, stage: str | None = None) -> None:
        """Build only the splits required for the requested Lightning stage.

        :param stage: Lightning stage name, or ``None`` to build every split.
        :raises ValueError: The live voice drifts from ``PARAM_SPEC``, or the
            configured row width differs from the ``torchsynth_full`` spec.
        """

        def dataset(size: int, seed: int) -> TorchSynthDataset:
            return TorchSynthDataset(size, seed, self.sample_rate, self.signal_length)

        train_size, val_size, test_size = self.train_val_test_sizes
        train_seed, val_seed, test_seed = self.train_val_test_seeds
        if stage in (None, "fit"):
            self.train, self.val = dataset(train_size, train_seed), dataset(val_size, val_seed)
        elif stage == "validate":
            self.val = dataset(val_size, val_seed)
        if stage in (None, "test", "predict"):
            self.test = dataset(test_size, test_seed)
        renderer = _make_renderer(self.sample_rate, self.signal_length)
        _verify_voice_matches_spec(renderer.voice)
        width = TORCHSYNTH_FULL_PARAM_SPEC.encoded_width
        if self.num_params != width:
            raise ValueError(
                f"Configured num_params={self.num_params}, torchsynth_full encodes {width}"
            )

    def _loader(
        self,
        dataset: Dataset[TorchSynthItem],
        *,
        shuffle: bool = False,
        sampler: Sampler[int] | None = None,
    ) -> DataLoader[TorchSynthBatch]:
        """Wrap one online split with the shared tuple collator.

        :param dataset: Online split to load.
        :param shuffle: Whether to shuffle logical row indices; exclusive with ``sampler``.
        :param sampler: Index sampler overriding the default order.
        :returns: Batched online data loader.
        """
        # persistent_workers / pin_memory are unset — per-epoch worker Voice rebuilds
        # and the host→GPU copy are tunable throughput wins, deferred to #1820.
        return cast(
            DataLoader[TorchSynthBatch],
            DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=regular_collate_fn,
            ),
        )

    def train_dataloader(self) -> DataLoader[TorchSynthBatch]:
        """Return the online training loader, shuffled or freshly resampled per epoch.

        :returns: Batched online training data.
        """
        if self.resample_train_per_epoch:
            return self._loader(self.train, sampler=_FreshEpochSampler(len(self.train)))
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
