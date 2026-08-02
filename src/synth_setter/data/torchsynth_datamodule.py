"""Online TorchSynth datasets and Lightning data module.

Compose ``experiment=torchsynth/ffn`` to sample parameters and render every
audio batch on the training machine without materializing an audio dataset.
"""

from __future__ import annotations

import sys
import threading
import types
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache, partial
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler

from synth_setter.conditioning import ConditioningMode
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
TorchSynthBatch: TypeAlias = dict[str, torch.Tensor]
TorchSynthCollateFn: TypeAlias = Callable[[Sequence[TorchSynthItem]], TorchSynthBatch]
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
    # Context and removal criteria: https://github.com/tinaudio/synth-setter/issues/2659.
    lock: threading.Lock


def _pad_to_render_size(params: torch.Tensor, render_batch_size: int) -> torch.Tensor:
    """Fill a short parameter batch up to the renderer's fixed row count.

    Row 0 is repeated because it is already a valid parameter row; the caller slices
    the padding rows back off the rendered audio.

    :param params: Parameter rows shaped ``(batch, NUM_PARAMS)``.
    :param render_batch_size: Fixed row count the renderer's voice was built for.
    :returns: Rows shaped ``(render_batch_size, NUM_PARAMS)``.
    :raises ValueError: ``params`` carries more rows than the renderer can hold.
    """
    missing = render_batch_size - len(params)
    if missing < 0:
        raise ValueError(
            f"render_batch_size={render_batch_size} cannot hold {len(params)} parameter rows"
        )
    if missing == 0:
        return params
    return torch.cat([params, params[:1].detach().expand(missing, -1)])


# Sized by configuration, never by the observed batch length — that bound is the #1820 fix. The
# separate size-1 voice is deliberate: the per-row target path costs ~3.8x if folded into it.
# https://github.com/tinaudio/synth-setter/issues/1820#issuecomment-5111622505
@cache
def _make_renderer(
    sample_rate: int, signal_length: int, render_batch_size: int = 1, device: str = "cpu"
) -> _Renderer:
    """Return the process-local renderer for one audio geometry and device.

    :param sample_rate: Audio sample rate in Hz.
    :param signal_length: Number of output samples.
    :param render_batch_size: Fixed voice batch size; shorter batches pad up to it.
    :param device: Torch device string.
    :returns: Cached voice and its mutation lock.
    """
    synth_config, voice = _torchsynth_types()
    # The cache outlives whatever scope first fills it. Built inside a Lightning validation
    # loop the voice's parameters would be inference tensors, which track no version counter
    # and so break every later gradient render in the process (#2744).
    with torch.inference_mode(False):
        instance = voice(
            synthconfig=synth_config(
                batch_size=render_batch_size,
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
    params: torch.Tensor,
    *,
    sample_rate: int,
    signal_length: int,
    render_batch_size: int = 1,
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
    :param render_batch_size: Fixed row count of the voice this render runs on, taken
        from configuration rather than from ``params`` so the renderer cache stays
        bounded (#1820); defaults to the row-at-a-time target render.
    :returns: Float32 audio shaped ``(batch, signal_length)``.
    :raises ValueError: The row width, a non-finite value, a batch exceeding
        ``render_batch_size``, or the rendered audio violates the data contract.
    """
    if not torch.isfinite(params).all():
        raise ValueError("TorchSynth params must be finite")
    width = TORCHSYNTH_FULL_PARAM_SPEC.encoded_width
    if params.shape[1] != width:
        raise ValueError(f"Expected {width} encoded parameter columns, got {params.shape[1]}")
    padded = _pad_to_render_size(params, render_batch_size)
    # Decode the padded rows, not just the real ones: the voice holds render_batch_size
    # keyboard entries and every one of them must be set from the row it renders.
    notes = [
        TORCHSYNTH_FULL_PARAM_SPEC.decode(row)[1]
        for row in padded.detach().clamp(0, 1).cpu().numpy()
    ]
    column = partial(torch.tensor, dtype=torch.float32, device=params.device)
    # Slice the synth columns explicitly: only they reach the voice's parameters. The note
    # columns carry no gradient by construction — pitch is a discrete category, and duration
    # lands on ADSR segment boundaries through integer sample arithmetic — so a differentiable
    # render must never see them as if they were continuous knobs.
    synth_params = padded[:, TORCHSYNTH_FULL_PARAM_SPEC.synth_columns]
    renderer = _make_renderer(sample_rate, signal_length, render_batch_size, str(params.device))
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
            audio = voice.output()[: len(params)]
    if not torch.isfinite(audio).all():
        raise ValueError("TorchSynth audio must be finite")
    starts = column([note["note_start_and_end"][0] for note in notes[: len(params)]])
    return _delay_by_note_start(audio.as_subclass(torch.Tensor).clamp(-1, 1), starts, sample_rate)


def collate_audio_dict(batch: Sequence[TorchSynthItem]) -> TorchSynthBatch:
    """Collate online rows into the audio-conditioned VST batch contract.

    :param batch: Rows carrying audio shaped ``(1, samples)`` and params shaped
        ``(1, NUM_PARAMS)`` in ``[0, 1]``.
    :returns: Float32 ``params``/``noise`` shaped ``(batch, NUM_PARAMS)`` and ``audio``
        shaped ``(batch, samples)``; params use the model's ``[-1, 1]`` space.
    """
    audio = torch.cat([row[0] for row in batch], dim=0)
    params01 = torch.cat([row[1] for row in batch], dim=0)
    params = params01 * 2 - 1
    return {
        "params": params,
        "noise": torch.randn_like(params),
        "audio": audio,
    }


def collate_vst_dict(batch: Sequence[TorchSynthItem], sample_rate: float) -> TorchSynthBatch:
    """Collate online rows into the mel-capable VST batch contract.

    ``mel`` uses the dataset generator's frontend so online and Lance-hydrated
    batches share shape and scaling.

    :param batch: Rows carrying audio shaped ``(1, samples)`` and params shaped
        ``(1, NUM_PARAMS)`` in ``[0, 1]``.
    :param sample_rate: Audio sample rate in Hz for the mel frontend.
    :returns: The audio-conditioned contract plus float32 ``mel`` shaped
        ``(batch, 1, mels, frames)``.
    """
    from synth_setter.data.vst.generate_vst_dataset import make_spectrogram

    collated = collate_audio_dict(batch)
    mel = torch.stack(
        [torch.from_numpy(make_spectrogram(row.numpy(), sample_rate)) for row in collated["audio"]]
    ).unsqueeze(1)
    collated["mel"] = mel.to(torch.float32)
    return collated


class TorchSynthDataset(Dataset[TorchSynthItem]):
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
        collate_fn: TorchSynthCollateFn | None = None,
        resample_train_per_epoch: bool = False,
        drop_last: bool = False,
        conditioning: ConditioningMode = "audio",
        *,
        val_num_workers: int = 0,
    ) -> None:
        """Configure the online TorchSynth train, validation, and test splits.

        :param sample_rate: Audio sample rate in Hz.
        :param signal_length: Number of output samples per rendered row.
        :param num_params: Expected row width, validated against the spec in ``setup``.
        :param train_val_test_sizes: Row counts for the train, validation, and test splits.
        :param train_val_test_seeds: Base seeds for the train, validation, and test splits.
        :param batch_size: DataLoader batch size.
        :param num_workers: Worker processes for training and test loaders.
        :param collate_fn: Fully configured row collator; defaults to mel-capable batches.
        :param resample_train_per_epoch: Draw fresh train rows every epoch (truly online
            training) instead of revisiting one fixed split; validation and test stay fixed.
        :param drop_last: Whether training discards a trailing partial batch when the split
            contains at least one full batch.
        :param conditioning: Model-batch modality; TorchSynth supports raw audio only.
        :param val_num_workers: Worker processes for the validation loader.
        :raises ValueError: If conditioning does not select raw audio.
        """
        if conditioning != "audio":
            raise ValueError("TorchSynth conditioning must be 'audio'")
        super().__init__()
        self.sample_rate = sample_rate
        self.signal_length = signal_length
        self.num_params = num_params
        self.train_val_test_sizes = train_val_test_sizes
        self.train_val_test_seeds = train_val_test_seeds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_num_workers = val_num_workers
        self.collate_fn = (
            collate_fn
            if collate_fn is not None
            else partial(collate_vst_dict, sample_rate=sample_rate)
        )
        self.resample_train_per_epoch = resample_train_per_epoch
        self.drop_last = drop_last
        self.conditioning = conditioning

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
        num_workers: int,
        shuffle: bool = False,
        sampler: Sampler[int] | None = None,
        drop_last: bool = False,
    ) -> DataLoader[TorchSynthBatch]:
        """Wrap one online split with the configured collator.

        :param dataset: Online split to load.
        :param num_workers: Worker processes for this split.
        :param shuffle: Whether to shuffle logical row indices; exclusive with ``sampler``.
        :param sampler: Index sampler overriding the default order.
        :param drop_last: Whether to drop a trailing partial batch. Set on training only;
            evaluation keeps its remainder rather than silently losing rows.
        :returns: Batched online data loader.
        """
        # persistent_workers / pin_memory are unset — per-epoch worker Voice rebuilds
        # and the host→GPU copy are tunable throughput wins, deferred to #1820.
        # The cast re-types the loader by its collate output; DataLoader's generic only
        # tracks the dataset's item type.
        return cast(
            DataLoader[TorchSynthBatch],
            DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=num_workers,
                drop_last=drop_last,
                collate_fn=self.collate_fn,
            ),
        )

    def train_dataloader(self) -> DataLoader[TorchSynthBatch]:
        """Return the online training loader, shuffled or freshly resampled per epoch.

        :returns: Batched online training data.
        """
        # An undersized split must still yield its single partial batch, not nothing.
        drop_last = self.drop_last and len(self.train) >= self.batch_size
        if self.resample_train_per_epoch:
            return self._loader(
                self.train,
                num_workers=self.num_workers,
                sampler=_FreshEpochSampler(len(self.train)),
                drop_last=drop_last,
            )
        return self._loader(
            self.train,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=drop_last,
        )

    def val_dataloader(self) -> DataLoader[TorchSynthBatch]:
        """Return the deterministic online validation loader.

        :returns: Batched online validation data.
        """
        return self._loader(self.val, num_workers=self.val_num_workers)

    def test_dataloader(self) -> DataLoader[TorchSynthBatch]:
        """Return the deterministic online test loader.

        :returns: Batched online test data.
        """
        return self._loader(self.test, num_workers=self.num_workers)
