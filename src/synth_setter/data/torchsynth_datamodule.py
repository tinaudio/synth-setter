"""Online TorchSynth datasets and Lightning data module.

Compose ``experiment=torchsynth/ffn`` to sample parameters and render every
audio batch on the training machine without materializing an audio dataset.
"""

from __future__ import annotations

import math
import sys
import threading
import types
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, partial
from typing import TYPE_CHECKING, TypeAlias, cast

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from synth_setter.data.ot import regular_collate_fn

if TYPE_CHECKING:
    from torchsynth.synth import Voice

TorchSynthItem: TypeAlias = tuple[
    torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]
]
TorchSynthBatch: TypeAlias = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]
]
# The odd 64-bit golden-ratio multiplier diffuses nearby split seeds into distinct RNG streams.
_SEED_MIXER = 0x9E3779B97F4A7C15
# Params must stay strictly inside the open interval (0, 1). Finite out-of-range
# values are expected (raw model predictions are unconstrained) and clamp away from
# the endpoints; NaN/Inf signal divergence or a pipeline bug and raise instead.
_PARAM_CLAMP_EPS = 1e-4


@dataclass(frozen=True)
class TorchSynthParam:
    """Identity and human range of one voice parameter, pinned to detect torchsynth drift.

    .. attribute :: module

       Owning synth module name (e.g. ``adsr_1``).

    .. attribute :: name

       Parameter name within the module (e.g. ``attack``).

    .. attribute :: minimum

       Human-unit range minimum ``from_0to1`` maps onto.

    .. attribute :: maximum

       Human-unit range maximum ``from_0to1`` maps onto.

    .. attribute :: curve

       Normalization curve exponent (1 is linear).

    .. attribute :: symmetric

       Whether the curve is mirrored around the range midpoint.
    """

    module: str
    name: str
    minimum: float
    maximum: float
    curve: float
    symmetric: bool


# Snapshot of every torchsynth 1.0.2 Voice parameter in ``get_parameters()`` order.
# The model's targets map onto these columns positionally, so any drift — a rename,
# reorder, or range change in a torchsynth bump — must fail loudly (see setup()).
PARAM_SPEC: tuple[TorchSynthParam, ...] = (
    TorchSynthParam("adsr_1", "attack", 0.0, 2.0, 0.5, False),
    TorchSynthParam("adsr_1", "decay", 0.0, 2.0, 0.5, False),
    TorchSynthParam("adsr_1", "sustain", 0.0, 1.0, 1, False),
    TorchSynthParam("adsr_1", "release", 0.0, 5.0, 0.5, False),
    TorchSynthParam("adsr_1", "alpha", 0.1, 6.0, 1, False),
    TorchSynthParam("adsr_2", "attack", 0.0, 2.0, 0.5, False),
    TorchSynthParam("adsr_2", "decay", 0.0, 2.0, 0.5, False),
    TorchSynthParam("adsr_2", "sustain", 0.0, 1.0, 1, False),
    TorchSynthParam("adsr_2", "release", 0.0, 5.0, 0.5, False),
    TorchSynthParam("adsr_2", "alpha", 0.1, 6.0, 1, False),
    TorchSynthParam("keyboard", "midi_f0", 0.0, 127.0, 1.0, False),
    TorchSynthParam("keyboard", "duration", 0.01, 4.0, 0.5, False),
    TorchSynthParam("lfo_1", "frequency", 0.0, 20.0, 0.25, False),
    TorchSynthParam("lfo_1", "mod_depth", -10.0, 20.0, 0.5, True),
    TorchSynthParam("lfo_1", "initial_phase", -3.1415927410125732, 3.1415927410125732, 1, False),
    TorchSynthParam("lfo_1", "sin", 0.0, 1.0, 1, False),
    TorchSynthParam("lfo_1", "tri", 0.0, 1.0, 1, False),
    TorchSynthParam("lfo_1", "saw", 0.0, 1.0, 1, False),
    TorchSynthParam("lfo_1", "rsaw", 0.0, 1.0, 1, False),
    TorchSynthParam("lfo_1", "sqr", 0.0, 1.0, 1, False),
    TorchSynthParam("lfo_1_amp_adsr", "attack", 0.0, 2.0, 0.5, False),
    TorchSynthParam("lfo_1_amp_adsr", "decay", 0.0, 2.0, 0.5, False),
    TorchSynthParam("lfo_1_amp_adsr", "sustain", 0.0, 1.0, 1, False),
    TorchSynthParam("lfo_1_amp_adsr", "release", 0.0, 5.0, 0.5, False),
    TorchSynthParam("lfo_1_amp_adsr", "alpha", 0.1, 6.0, 1, False),
    TorchSynthParam("lfo_1_rate_adsr", "attack", 0.0, 2.0, 0.5, False),
    TorchSynthParam("lfo_1_rate_adsr", "decay", 0.0, 2.0, 0.5, False),
    TorchSynthParam("lfo_1_rate_adsr", "sustain", 0.0, 1.0, 1, False),
    TorchSynthParam("lfo_1_rate_adsr", "release", 0.0, 5.0, 0.5, False),
    TorchSynthParam("lfo_1_rate_adsr", "alpha", 0.1, 6.0, 1, False),
    TorchSynthParam("lfo_2", "frequency", 0.0, 20.0, 0.25, False),
    TorchSynthParam("lfo_2", "mod_depth", -10.0, 20.0, 0.5, True),
    TorchSynthParam("lfo_2", "initial_phase", -3.1415927410125732, 3.1415927410125732, 1, False),
    TorchSynthParam("lfo_2", "sin", 0.0, 1.0, 1, False),
    TorchSynthParam("lfo_2", "tri", 0.0, 1.0, 1, False),
    TorchSynthParam("lfo_2", "saw", 0.0, 1.0, 1, False),
    TorchSynthParam("lfo_2", "rsaw", 0.0, 1.0, 1, False),
    TorchSynthParam("lfo_2", "sqr", 0.0, 1.0, 1, False),
    TorchSynthParam("lfo_2_amp_adsr", "attack", 0.0, 2.0, 0.5, False),
    TorchSynthParam("lfo_2_amp_adsr", "decay", 0.0, 2.0, 0.5, False),
    TorchSynthParam("lfo_2_amp_adsr", "sustain", 0.0, 1.0, 1, False),
    TorchSynthParam("lfo_2_amp_adsr", "release", 0.0, 5.0, 0.5, False),
    TorchSynthParam("lfo_2_amp_adsr", "alpha", 0.1, 6.0, 1, False),
    TorchSynthParam("lfo_2_rate_adsr", "attack", 0.0, 2.0, 0.5, False),
    TorchSynthParam("lfo_2_rate_adsr", "decay", 0.0, 2.0, 0.5, False),
    TorchSynthParam("lfo_2_rate_adsr", "sustain", 0.0, 1.0, 1, False),
    TorchSynthParam("lfo_2_rate_adsr", "release", 0.0, 5.0, 0.5, False),
    TorchSynthParam("lfo_2_rate_adsr", "alpha", 0.1, 6.0, 1, False),
    TorchSynthParam("mixer", "vco_1", 0.0, 1.0, 1.0, False),
    TorchSynthParam("mixer", "vco_2", 0.0, 1.0, 1.0, False),
    TorchSynthParam("mixer", "noise", 0.0, 1.0, 0.025, False),
    TorchSynthParam("mod_matrix", "adsr_1->vco_1_pitch", 0.0, 1.0, 0.5, False),
    TorchSynthParam("mod_matrix", "adsr_1->vco_1_amp", 0.0, 1.0, 0.5, False),
    TorchSynthParam("mod_matrix", "adsr_1->vco_2_pitch", 0.0, 1.0, 0.5, False),
    TorchSynthParam("mod_matrix", "adsr_1->vco_2_amp", 0.0, 1.0, 0.5, False),
    TorchSynthParam("mod_matrix", "adsr_1->noise_amp", 0.0, 1.0, 0.5, False),
    TorchSynthParam("mod_matrix", "adsr_2->vco_1_pitch", 0.0, 1.0, 0.5, False),
    TorchSynthParam("mod_matrix", "adsr_2->vco_1_amp", 0.0, 1.0, 0.5, False),
    TorchSynthParam("mod_matrix", "adsr_2->vco_2_pitch", 0.0, 1.0, 0.5, False),
    TorchSynthParam("mod_matrix", "adsr_2->vco_2_amp", 0.0, 1.0, 0.5, False),
    TorchSynthParam("mod_matrix", "adsr_2->noise_amp", 0.0, 1.0, 0.5, False),
    TorchSynthParam("mod_matrix", "lfo_1->vco_1_pitch", 0.0, 1.0, 0.5, False),
    TorchSynthParam("mod_matrix", "lfo_1->vco_1_amp", 0.0, 1.0, 0.5, False),
    TorchSynthParam("mod_matrix", "lfo_1->vco_2_pitch", 0.0, 1.0, 0.5, False),
    TorchSynthParam("mod_matrix", "lfo_1->vco_2_amp", 0.0, 1.0, 0.5, False),
    TorchSynthParam("mod_matrix", "lfo_1->noise_amp", 0.0, 1.0, 0.5, False),
    TorchSynthParam("mod_matrix", "lfo_2->vco_1_pitch", 0.0, 1.0, 0.5, False),
    TorchSynthParam("mod_matrix", "lfo_2->vco_1_amp", 0.0, 1.0, 0.5, False),
    TorchSynthParam("mod_matrix", "lfo_2->vco_2_pitch", 0.0, 1.0, 0.5, False),
    TorchSynthParam("mod_matrix", "lfo_2->vco_2_amp", 0.0, 1.0, 0.5, False),
    TorchSynthParam("mod_matrix", "lfo_2->noise_amp", 0.0, 1.0, 0.5, False),
    TorchSynthParam("vco_1", "tuning", -24.0, 24.0, 1, False),
    TorchSynthParam("vco_1", "mod_depth", -96.0, 96.0, 0.2, True),
    TorchSynthParam("vco_1", "initial_phase", -3.1415927410125732, 3.1415927410125732, 1, False),
    TorchSynthParam("vco_2", "tuning", -24.0, 24.0, 1, False),
    TorchSynthParam("vco_2", "mod_depth", -96.0, 96.0, 0.2, True),
    TorchSynthParam("vco_2", "initial_phase", -3.1415927410125732, 3.1415927410125732, 1, False),
    TorchSynthParam("vco_2", "shape", 0.0, 1.0, 1, False),
)
# The keyboard's midi_f0 and duration are fixed by the renderer (constants of the
# task), so they are excluded from the model's positional prediction targets.
_FIXED_MODULES = frozenset({"keyboard"})
INFERABLE_SPEC: tuple[TorchSynthParam, ...] = tuple(
    param for param in PARAM_SPEC if param.module not in _FIXED_MODULES
)
NUM_PARAMS = len(INFERABLE_SPEC)


def _spec_from_voice(voice: Voice) -> tuple[TorchSynthParam, ...]:
    """Extract the live voice's parameter spec in ``get_parameters()`` order.

    :param voice: Live torchsynth voice to snapshot.
    :returns: One ``TorchSynthParam`` per voice parameter, in native order.
    """
    spec = []
    for (module, name), parameter in voice.get_parameters().items():
        # getattr: torchsynth sets parameter_range dynamically in __new__, so pyright
        # cannot see it as a ModuleParameter attribute.
        parameter_range = getattr(parameter, "parameter_range")
        spec.append(
            TorchSynthParam(
                module,
                name,
                float(parameter_range.minimum),
                float(parameter_range.maximum),
                float(parameter_range.curve),
                bool(parameter_range.symmetric),
            )
        )
    return tuple(spec)


def _verify_voice_matches_spec(
    voice: Voice, spec: tuple[TorchSynthParam, ...] = PARAM_SPEC
) -> None:
    """Raise unless the live voice's parameters match the pinned spec.

    Identity fields compare exactly; range floats compare via ``math.isclose`` so an
    upstream float-precision wobble does not masquerade as drift.

    :param voice: Live torchsynth voice to check.
    :param spec: Expected parameter snapshot, defaulting to the checked-in ``PARAM_SPEC``.
    :raises ValueError: The live voice's parameter set drifts from the spec.
    """
    live = _spec_from_voice(voice)
    if len(live) != len(spec):
        raise ValueError(f"TorchSynth exposes {len(live)} parameters, spec pins {len(spec)}")
    for index, (actual, expected) in enumerate(zip(live, spec, strict=True)):
        identity_matches = (actual.module, actual.name, actual.symmetric) == (
            expected.module,
            expected.name,
            expected.symmetric,
        )
        range_matches = all(
            math.isclose(actual_value, expected_value, rel_tol=1e-6, abs_tol=1e-9)
            for actual_value, expected_value in (
                (actual.minimum, expected.minimum),
                (actual.maximum, expected.maximum),
                (actual.curve, expected.curve),
            )
        )
        if not (identity_matches and range_matches):
            raise ValueError(
                f"TorchSynth parameter {index} drifted from PARAM_SPEC:"
                f" expected {expected}, got {actual}"
            )


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


# Unbounded cache, but production holds only a few entries: batch_size=1 items plus
# the metric re-render's val batch sizes; batch/GPU rendering would need eviction or
# a fixed renderer size — see #1820.
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


def render_torchsynth(
    params: torch.Tensor, *, sample_rate: int, signal_length: int, midi_pitch: int
) -> torch.Tensor:
    """Render normalized TorchSynth parameters into a mono audio batch.

    :param params: Finite parameter rows in TorchSynth's native order; values are
        clamped strictly inside ``(0, 1)``.
    :param sample_rate: Audio sample rate in Hz.
    :param signal_length: Number of output samples.
    :param midi_pitch: Fixed MIDI note rendered for every parameter row.
    :returns: Audio shaped ``(batch, signal_length)``.
    :raises ValueError: The parameter width, a non-finite parameter, or the rendered
        audio violates the data contract.
    """
    if not torch.isfinite(params).all():
        raise ValueError("TorchSynth params must be finite")
    renderer = _make_renderer(sample_rate, signal_length, len(params), str(params.device))
    voice = renderer.voice
    with renderer.lock:
        all_parameters = voice.get_parameters()
        if params.shape[1] != NUM_PARAMS:
            raise ValueError(
                f"Expected {NUM_PARAMS} TorchSynth parameters, got {params.shape[1]}"
            )
        native = [all_parameters[(spec.module, spec.name)] for spec in INFERABLE_SPEC]
        for values, parameter in zip(params.T, native, strict=True):
            parameter.data.copy_(values.clamp(_PARAM_CLAMP_EPS, 1 - _PARAM_CLAMP_EPS))
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
        self.num_params = NUM_PARAMS

    def __len__(self) -> int:
        """Return the logical number of online samples.

        :returns: Configured split length.
        """
        return self.num_samples

    def __getitem__(self, index: int) -> TorchSynthItem:
        """Sample and render one deterministic parameter row.

        :param index: Logical row index.
        :returns: Audio, parameters, and the callable used to render them.
        """
        sample_seed = (self.seed * _SEED_MIXER + index) % sys.maxsize
        generator = torch.Generator().manual_seed(sample_seed)
        params = torch.rand((1, self.num_params), generator=generator)
        # Per-sample CPU render; render_fn is passed through so a future collate can
        # batch/GPU-render instead of paying Voice.output() per row — see #1820.
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
        num_params: int = NUM_PARAMS,
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
        self, dataset: Dataset[TorchSynthItem], *, shuffle: bool = False
    ) -> DataLoader[TorchSynthBatch]:
        """Wrap one online split with the shared tuple collator.

        :param dataset: Online split to load.
        :param shuffle: Whether to shuffle logical row indices.
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
