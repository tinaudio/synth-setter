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

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler

from synth_setter.data.sample_seed import derive_sample_seed

# Re-exported for backward compat: training code imports these names from this module.
from synth_setter.data.vst.torchsynth_param_spec import (
    INFERABLE_SPEC as INFERABLE_SPEC,
    NUM_PARAMS as NUM_PARAMS,
    PARAM_SPEC as PARAM_SPEC,
    TorchSynthParam as TorchSynthParam,
)
from synth_setter.data.vst.torchsynth_param_spec import (
    KEYBOARD_DURATION_BOUNDS,
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


# One voice per configured render size, never per observed batch length, so a partial
# batch neither allocates a second voice nor changes the render — see #1820.
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
    instance = voice(
        synthconfig=synth_config(
            batch_size=render_batch_size,
            sample_rate=sample_rate,
            buffer_size_seconds=signal_length / sample_rate,
            reproducible=False,
        )
    )
    return _Renderer(instance.to(torch.device(device)), threading.Lock())


def render_torchsynth(
    params: torch.Tensor,
    *,
    sample_rate: int,
    signal_length: int,
    midi_pitch: int,
    note_duration_seconds: float | None = None,
    render_batch_size: int = 1,
) -> torch.Tensor:
    """Render normalized TorchSynth parameters into a mono audio batch.

    :param params: Finite float32 parameter rows shaped ``(batch, NUM_PARAMS)`` in
        TorchSynth's native order; values clamp strictly inside ``(0, 1)``.
    :param sample_rate: Audio sample rate in Hz.
    :param signal_length: Number of output samples.
    :param midi_pitch: Fixed MIDI note rendered for every parameter row.
    :param note_duration_seconds: Note-on length before the release stage; ``None``
        holds the note for the whole buffer. Must lie within the keyboard
        parameter's pinned human range (``KEYBOARD_DURATION_BOUNDS``).
    :param render_batch_size: Fixed row count of the voice this render runs on;
        defaults to the row-at-a-time target render.
    :returns: Float32 audio shaped ``(batch, signal_length)``.
    :raises ValueError: The parameter width, a non-finite parameter, an
        out-of-range note duration, a batch exceeding ``render_batch_size``, or the
        rendered audio violates the data contract.
    """
    if not torch.isfinite(params).all():
        raise ValueError("TorchSynth params must be finite")
    if params.shape[1] != NUM_PARAMS:
        raise ValueError(f"Expected {NUM_PARAMS} TorchSynth parameters, got {params.shape[1]}")
    duration = (
        note_duration_seconds
        if note_duration_seconds is not None
        else signal_length / sample_rate
    )
    minimum_duration, maximum_duration = KEYBOARD_DURATION_BOUNDS
    if not minimum_duration <= duration <= maximum_duration:
        # Validate here (not via torchsynth's bare asserts, stripped under -O) so
        # out-of-range durations fail the documented ValueError contract.
        raise ValueError(
            f"note duration {duration}s outside the keyboard's pinned range "
            f"[{minimum_duration}, {maximum_duration}]s"
        )
    padded = _pad_to_render_size(params, render_batch_size)
    renderer = _make_renderer(sample_rate, signal_length, render_batch_size, str(params.device))
    voice = renderer.voice
    with renderer.lock:
        all_parameters = voice.get_parameters()
        native = [all_parameters[(spec.module, spec.name)] for spec in INFERABLE_SPEC]
        for values, parameter in zip(padded.T, native, strict=True):
            parameter.data.copy_(values.clamp(_PARAM_CLAMP_EPS, 1 - _PARAM_CLAMP_EPS))
        for name, value in (
            ("midi_f0", float(midi_pitch)),
            ("duration", duration),
        ):
            keyboard = all_parameters[("keyboard", name)]
            keyboard.to_0to1(torch.full((render_batch_size,), value, device=params.device))
        with torch.no_grad():
            audio = voice.output()[: len(params)]
    if not torch.isfinite(audio).all():
        raise ValueError("TorchSynth audio must be finite")
    return audio.as_subclass(torch.Tensor).clamp(-1, 1)


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


def collate_vst_dict(
    batch: Sequence[TorchSynthItem], sample_rate: float
) -> TorchSynthBatch:
    """Collate online rows into the mel-capable VST batch contract.

    ``mel_spec`` uses the dataset generator's frontend so online and Lance-hydrated
    batches share shape and scaling.

    :param batch: Rows carrying audio shaped ``(1, samples)`` and params shaped
        ``(1, NUM_PARAMS)`` in ``[0, 1]``.
    :param sample_rate: Audio sample rate in Hz for the mel frontend.
    :returns: The audio-conditioned contract plus float32 ``mel_spec`` shaped
        ``(batch, 1, mels, frames)``.
    """
    from synth_setter.data.vst.generate_vst_dataset import make_spectrogram

    collated = collate_audio_dict(batch)
    mel = torch.stack(
        [
            torch.from_numpy(make_spectrogram(row.numpy(), sample_rate))
            for row in collated["audio"]
        ]
    ).unsqueeze(1)
    collated["mel_spec"] = mel.to(torch.float32)
    return collated


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
        self.num_params = NUM_PARAMS

    def __len__(self) -> int:
        """Return the logical number of online samples.

        :returns: Configured split length.
        """
        return self.num_samples

    def __getitem__(self, index: int) -> TorchSynthItem:
        """Sample and render one deterministic parameter row.

        :param index: Logical row index.
        :returns: Float32 audio shaped ``(1, signal_length)``, float32 parameters shaped ``(1,
            NUM_PARAMS)``, and the renderer callable.
        """
        sample_seed = derive_sample_seed(self.seed, index)
        generator = torch.Generator().manual_seed(sample_seed)
        params = torch.rand((1, self.num_params), generator=generator).clamp(
            _PARAM_CLAMP_EPS, 1 - _PARAM_CLAMP_EPS
        )
        # Per-sample CPU render; render_fn is passed through so a future collate can
        # batch/GPU-render instead of paying Voice.output() per row — see #1820.
        render_fn = partial(
            render_torchsynth,
            sample_rate=self.sample_rate,
            signal_length=self.signal_length,
            midi_pitch=self.midi_pitch,
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
        midi_pitch: int = 60,
        num_params: int = NUM_PARAMS,
        train_val_test_sizes: tuple[int, int, int] = (100_000, 10_000, 10_000),
        train_val_test_seeds: tuple[int, int, int] = (123, 456, 789),
        batch_size: int = 32,
        num_workers: int = 0,
        collate_fn: TorchSynthCollateFn | None = None,
        resample_train_per_epoch: bool = False,
        drop_last: bool = False,
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
        :param collate_fn: Fully configured row collator; defaults to mel-capable batches.
        :param resample_train_per_epoch: Draw fresh train rows every epoch (truly online
            training) instead of revisiting one fixed split; validation and test stay fixed.
        :param drop_last: Whether training discards a trailing partial batch when the split
            contains at least one full batch.
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
        self.collate_fn = (
            collate_fn
            if collate_fn is not None
            else partial(collate_vst_dict, sample_rate=sample_rate)
        )
        self.resample_train_per_epoch = resample_train_per_epoch
        self.drop_last = drop_last

    def setup(self, stage: str | None = None) -> None:
        """Build only the splits required for the requested Lightning stage.

        :param stage: Lightning stage name, or ``None`` to build every split.
        :raises ValueError: The live voice drifts from ``PARAM_SPEC``, or the
            configured parameter width differs from TorchSynth.
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
        _verify_voice_matches_spec(renderer.voice)
        if self.num_params != NUM_PARAMS:
            raise ValueError(
                f"Configured num_params={self.num_params}, TorchSynth exposes {NUM_PARAMS}"
            )

    def _loader(
        self,
        dataset: Dataset[TorchSynthItem],
        *,
        shuffle: bool = False,
        sampler: Sampler[int] | None = None,
        drop_last: bool = False,
    ) -> DataLoader[TorchSynthBatch]:
        """Wrap one online split with the configured collator.

        :param dataset: Online split to load.
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
                num_workers=self.num_workers,
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
                self.train, sampler=_FreshEpochSampler(len(self.train)), drop_last=drop_last
            )
        return self._loader(self.train, shuffle=True, drop_last=drop_last)

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
