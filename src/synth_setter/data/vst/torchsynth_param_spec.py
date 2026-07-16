"""Pinned torchsynth ``Voice`` parameter identity and the registry ``ParamSpec``s.

Pure-Python (no torch / torchsynth imports) so the pedalboard-free registry can
load it in interpreter-only contexts. Hosts the checked-in snapshot of every
voice parameter (identity + human range), the curve math mirrored from
``torchsynth.parameter.ModuleParameterRange``, the baseline patch reduced specs
render against, and the ``torchsynth_adsr`` / ``torchsynth_simple`` /
``torchsynth_full`` sampling specs. Synth params are keyed ``module.name``
(e.g. ``adsr_1.attack``), matching ``Voice.get_parameters()`` identities.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from synth_setter.renderer_backend import TORCHSYNTH_PLUGIN_NAME as TORCHSYNTH_PLUGIN_NAME

from synth_setter.data.vst.param_spec import (
    ContinuousParameter,
    DiscreteLiteralParameter,
    NoteDurationParameter,
    Parameter,
    ParamSpec,
)

if TYPE_CHECKING:
    from torchsynth.synth import Voice



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

    @property
    def key(self) -> str:
        """Return the ``module.name`` identity used as the spec/renderer dict key.

        :returns: Dotted parameter key (e.g. ``adsr_1.attack``).
        """
        return f"{self.module}.{self.name}"

    def from_0to1(self, normalized: float) -> float:
        """Denormalize a machine value, mirroring ``ModuleParameterRange.from_0to1``.

        :param normalized: Machine-range value in ``[0, 1]``.
        :returns: Human-unit value in ``[minimum, maximum]``.
        """
        if not self.symmetric:
            if self.curve != 1.0:
                normalized = normalized ** (1.0 / self.curve)
            return self.minimum + (self.maximum - self.minimum) * normalized
        dist = 2.0 * normalized - 1.0
        if self.curve != 1.0 and dist != 0.0:
            dist = math.copysign(abs(dist) ** (1.0 / self.curve), dist)
        return self.minimum + (self.maximum - self.minimum) / 2.0 * (dist + 1.0)

    def to_0to1(self, value: float) -> float:
        """Normalize a human-unit value, mirroring ``ModuleParameterRange.to_0to1``.

        :param value: Human-unit value in ``[minimum, maximum]``.
        :returns: Machine-range value in ``[0, 1]``.
        """
        normalized = (value - self.minimum) / (self.maximum - self.minimum)
        if not self.symmetric:
            if self.curve != 1.0:
                normalized = normalized**self.curve
            return normalized
        dist = 2.0 * normalized - 1.0
        return (1.0 + math.copysign(abs(dist) ** self.curve, dist)) / 2.0


# Snapshot of torchsynth 1.0.2 Voice parameters in ``get_parameters()`` order; targets
# map positionally, so any drift must fail loudly (``verify_voice_matches_spec``).
PARAM_SPEC: tuple[TorchSynthParam, ...] = (
    TorchSynthParam("adsr_1", "attack", 0.0, 2.0, 0.5, False),
    TorchSynthParam("adsr_1", "decay", 0.0, 2.0, 0.5, False),
    TorchSynthParam("adsr_1", "sustain", 0.0, 1.0, 1.0, False),
    TorchSynthParam("adsr_1", "release", 0.0, 5.0, 0.5, False),
    TorchSynthParam("adsr_1", "alpha", 0.1, 6.0, 1.0, False),
    TorchSynthParam("adsr_2", "attack", 0.0, 2.0, 0.5, False),
    TorchSynthParam("adsr_2", "decay", 0.0, 2.0, 0.5, False),
    TorchSynthParam("adsr_2", "sustain", 0.0, 1.0, 1.0, False),
    TorchSynthParam("adsr_2", "release", 0.0, 5.0, 0.5, False),
    TorchSynthParam("adsr_2", "alpha", 0.1, 6.0, 1.0, False),
    TorchSynthParam("keyboard", "midi_f0", 0.0, 127.0, 1.0, False),
    TorchSynthParam("keyboard", "duration", 0.01, 4.0, 0.5, False),
    TorchSynthParam("lfo_1", "frequency", 0.0, 20.0, 0.25, False),
    TorchSynthParam("lfo_1", "mod_depth", -10.0, 20.0, 0.5, True),
    TorchSynthParam("lfo_1", "initial_phase", -3.1415927410125732, 3.1415927410125732, 1.0, False),
    TorchSynthParam("lfo_1", "sin", 0.0, 1.0, 1.0, False),
    TorchSynthParam("lfo_1", "tri", 0.0, 1.0, 1.0, False),
    TorchSynthParam("lfo_1", "saw", 0.0, 1.0, 1.0, False),
    TorchSynthParam("lfo_1", "rsaw", 0.0, 1.0, 1.0, False),
    TorchSynthParam("lfo_1", "sqr", 0.0, 1.0, 1.0, False),
    TorchSynthParam("lfo_1_amp_adsr", "attack", 0.0, 2.0, 0.5, False),
    TorchSynthParam("lfo_1_amp_adsr", "decay", 0.0, 2.0, 0.5, False),
    TorchSynthParam("lfo_1_amp_adsr", "sustain", 0.0, 1.0, 1.0, False),
    TorchSynthParam("lfo_1_amp_adsr", "release", 0.0, 5.0, 0.5, False),
    TorchSynthParam("lfo_1_amp_adsr", "alpha", 0.1, 6.0, 1.0, False),
    TorchSynthParam("lfo_1_rate_adsr", "attack", 0.0, 2.0, 0.5, False),
    TorchSynthParam("lfo_1_rate_adsr", "decay", 0.0, 2.0, 0.5, False),
    TorchSynthParam("lfo_1_rate_adsr", "sustain", 0.0, 1.0, 1.0, False),
    TorchSynthParam("lfo_1_rate_adsr", "release", 0.0, 5.0, 0.5, False),
    TorchSynthParam("lfo_1_rate_adsr", "alpha", 0.1, 6.0, 1.0, False),
    TorchSynthParam("lfo_2", "frequency", 0.0, 20.0, 0.25, False),
    TorchSynthParam("lfo_2", "mod_depth", -10.0, 20.0, 0.5, True),
    TorchSynthParam("lfo_2", "initial_phase", -3.1415927410125732, 3.1415927410125732, 1.0, False),
    TorchSynthParam("lfo_2", "sin", 0.0, 1.0, 1.0, False),
    TorchSynthParam("lfo_2", "tri", 0.0, 1.0, 1.0, False),
    TorchSynthParam("lfo_2", "saw", 0.0, 1.0, 1.0, False),
    TorchSynthParam("lfo_2", "rsaw", 0.0, 1.0, 1.0, False),
    TorchSynthParam("lfo_2", "sqr", 0.0, 1.0, 1.0, False),
    TorchSynthParam("lfo_2_amp_adsr", "attack", 0.0, 2.0, 0.5, False),
    TorchSynthParam("lfo_2_amp_adsr", "decay", 0.0, 2.0, 0.5, False),
    TorchSynthParam("lfo_2_amp_adsr", "sustain", 0.0, 1.0, 1.0, False),
    TorchSynthParam("lfo_2_amp_adsr", "release", 0.0, 5.0, 0.5, False),
    TorchSynthParam("lfo_2_amp_adsr", "alpha", 0.1, 6.0, 1.0, False),
    TorchSynthParam("lfo_2_rate_adsr", "attack", 0.0, 2.0, 0.5, False),
    TorchSynthParam("lfo_2_rate_adsr", "decay", 0.0, 2.0, 0.5, False),
    TorchSynthParam("lfo_2_rate_adsr", "sustain", 0.0, 1.0, 1.0, False),
    TorchSynthParam("lfo_2_rate_adsr", "release", 0.0, 5.0, 0.5, False),
    TorchSynthParam("lfo_2_rate_adsr", "alpha", 0.1, 6.0, 1.0, False),
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
    TorchSynthParam("vco_1", "tuning", -24.0, 24.0, 1.0, False),
    TorchSynthParam("vco_1", "mod_depth", -96.0, 96.0, 0.2, True),
    TorchSynthParam("vco_1", "initial_phase", -3.1415927410125732, 3.1415927410125732, 1.0, False),
    TorchSynthParam("vco_2", "tuning", -24.0, 24.0, 1.0, False),
    TorchSynthParam("vco_2", "mod_depth", -96.0, 96.0, 0.2, True),
    TorchSynthParam("vco_2", "initial_phase", -3.1415927410125732, 3.1415927410125732, 1.0, False),
    TorchSynthParam("vco_2", "shape", 0.0, 1.0, 1.0, False),
)
# The keyboard's midi_f0 and duration are fixed by the renderer (constants of the
# task), so they are excluded from the model's positional prediction targets.
_FIXED_MODULES = frozenset({"keyboard"})
INFERABLE_SPEC: tuple[TorchSynthParam, ...] = tuple(
    param for param in PARAM_SPEC if param.module not in _FIXED_MODULES
)
NUM_PARAMS = len(INFERABLE_SPEC)


def spec_from_voice(voice: Voice) -> tuple[TorchSynthParam, ...]:
    """Extract the live voice's parameter spec in ``get_parameters()`` order.

    :param voice: Live torchsynth voice to snapshot.
    :returns: One ``TorchSynthParam`` per voice parameter, in native order.
    """
    spec = []
    for (module, name), parameter in voice.get_parameters().items():
        # TorchSynth sets parameter_range dynamically in __new__, beyond pyright's stub.
        parameter_range = parameter.parameter_range  # type: ignore[attr-defined]
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


def verify_voice_matches_spec(
    voice: Voice, spec: tuple[TorchSynthParam, ...] = PARAM_SPEC
) -> None:
    """Raise unless the live voice's parameters match the pinned spec.

    Identity fields compare exactly; range floats compare via ``math.isclose`` so an
    upstream float-precision wobble does not masquerade as drift.

    :param voice: Live torchsynth voice to check.
    :param spec: Expected parameter snapshot, defaulting to the checked-in ``PARAM_SPEC``.
    :raises ValueError: The live voice's parameter set drifts from the spec.
    """
    live = spec_from_voice(voice)
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


# Human-unit baseline pinning every un-sampled knob: adsr_1 is the open amp
# route (silence otherwise); other open routes are inert while mod_depth is 0.
DEFAULT_PATCH: Mapping[str, float] = MappingProxyType({
    "adsr_1.attack": 0.05,
    "adsr_1.decay": 0.2,
    "adsr_1.sustain": 0.7,
    "adsr_1.release": 0.3,
    "adsr_1.alpha": 3.0,
    "adsr_2.attack": 0.05,
    "adsr_2.decay": 0.2,
    "adsr_2.sustain": 0.7,
    "adsr_2.release": 0.3,
    "adsr_2.alpha": 3.0,
    "lfo_1.frequency": 2.0,
    "lfo_1.mod_depth": 0.0,
    "lfo_1.initial_phase": 0.0,
    "lfo_1.sin": 1.0,
    "lfo_1.tri": 0.0,
    "lfo_1.saw": 0.0,
    "lfo_1.rsaw": 0.0,
    "lfo_1.sqr": 0.0,
    "lfo_1_amp_adsr.attack": 0.0,
    "lfo_1_amp_adsr.decay": 0.0,
    "lfo_1_amp_adsr.sustain": 1.0,
    "lfo_1_amp_adsr.release": 0.5,
    "lfo_1_amp_adsr.alpha": 3.0,
    "lfo_1_rate_adsr.attack": 0.0,
    "lfo_1_rate_adsr.decay": 0.0,
    "lfo_1_rate_adsr.sustain": 1.0,
    "lfo_1_rate_adsr.release": 0.5,
    "lfo_1_rate_adsr.alpha": 3.0,
    "lfo_2.frequency": 2.0,
    "lfo_2.mod_depth": 0.0,
    "lfo_2.initial_phase": 0.0,
    "lfo_2.sin": 1.0,
    "lfo_2.tri": 0.0,
    "lfo_2.saw": 0.0,
    "lfo_2.rsaw": 0.0,
    "lfo_2.sqr": 0.0,
    "lfo_2_amp_adsr.attack": 0.0,
    "lfo_2_amp_adsr.decay": 0.0,
    "lfo_2_amp_adsr.sustain": 1.0,
    "lfo_2_amp_adsr.release": 0.5,
    "lfo_2_amp_adsr.alpha": 3.0,
    "lfo_2_rate_adsr.attack": 0.0,
    "lfo_2_rate_adsr.decay": 0.0,
    "lfo_2_rate_adsr.sustain": 1.0,
    "lfo_2_rate_adsr.release": 0.5,
    "lfo_2_rate_adsr.alpha": 3.0,
    "mixer.vco_1": 0.0,
    "mixer.vco_2": 1.0,
    "mixer.noise": 0.0,
    "mod_matrix.adsr_1->vco_1_pitch": 0.0,
    "mod_matrix.adsr_1->vco_1_amp": 1.0,
    "mod_matrix.adsr_1->vco_2_pitch": 0.0,
    "mod_matrix.adsr_1->vco_2_amp": 1.0,
    "mod_matrix.adsr_1->noise_amp": 1.0,
    "mod_matrix.adsr_2->vco_1_pitch": 1.0,
    "mod_matrix.adsr_2->vco_1_amp": 0.0,
    "mod_matrix.adsr_2->vco_2_pitch": 1.0,
    "mod_matrix.adsr_2->vco_2_amp": 0.0,
    "mod_matrix.adsr_2->noise_amp": 0.0,
    "mod_matrix.lfo_1->vco_1_pitch": 1.0,
    "mod_matrix.lfo_1->vco_1_amp": 0.0,
    "mod_matrix.lfo_1->vco_2_pitch": 1.0,
    "mod_matrix.lfo_1->vco_2_amp": 0.0,
    "mod_matrix.lfo_1->noise_amp": 0.0,
    "mod_matrix.lfo_2->vco_1_pitch": 0.0,
    "mod_matrix.lfo_2->vco_1_amp": 0.0,
    "mod_matrix.lfo_2->vco_2_pitch": 0.0,
    "mod_matrix.lfo_2->vco_2_amp": 0.0,
    "mod_matrix.lfo_2->noise_amp": 0.0,
    "vco_1.tuning": 0.0,
    "vco_1.mod_depth": 0.0,
    "vco_1.initial_phase": 0.0,
    "vco_2.tuning": 0.0,
    "vco_2.mod_depth": 0.0,
    "vco_2.initial_phase": 0.0,
    "vco_2.shape": 0.0,
})

# The baseline patch as machine-range values in ``INFERABLE_SPEC`` order.
DEFAULT_NORMALIZED_ROW: tuple[float, ...] = tuple(
    param.to_0to1(DEFAULT_PATCH[param.key]) for param in INFERABLE_SPEC
)
# Renderer lookups: dotted key -> positional slot, and the keyboard's pinned
# human duration range (torchsynth asserts on out-of-range note durations).
PARAM_INDEX: Mapping[str, int] = MappingProxyType(
    {param.key: index for index, param in enumerate(INFERABLE_SPEC)}
)
KEYBOARD_DURATION_BOUNDS: tuple[float, float] = next(
    (param.minimum, param.maximum) for param in PARAM_SPEC if param.key == "keyboard.duration"
)


def _note_params() -> list[Parameter]:
    """Build the shared note-conditioning params (fresh instances per spec).

    :returns: Pitch and note-window params matching the surge specs' ranges.
    """
    return [
        DiscreteLiteralParameter(name="pitch", min=48, max=72),
        NoteDurationParameter(name="note_start_and_end", max_note_duration_seconds=4.0),
    ]


def _continuous(names: list[str]) -> list[Parameter]:
    """Build one full-range normalized continuous param per dotted key.

    :param names: Dotted ``module.name`` keys to sample.
    :returns: ``ContinuousParameter``s in the given order.
    """
    return [ContinuousParameter(name=name, min=0.0, max=1.0) for name in names]


# Amp envelope + waveform morph on the one audible VCO at baseline.
TORCHSYNTH_ADSR_PARAM_SPEC = ParamSpec(
    _continuous(
        [
            "adsr_1.attack",
            "adsr_1.decay",
            "adsr_1.sustain",
            "adsr_1.release",
            "vco_2.shape",
        ]
    ),
    _note_params(),
)

# Fixed-routing analogue of ``surge_simple``: amp/pitch envelopes and vibrato
# ride the baseline routes; vco_2 detune/morph/depth and mixer levels sampled.
TORCHSYNTH_SIMPLE_PARAM_SPEC = ParamSpec(
    _continuous(
        [
            "adsr_1.attack",
            "adsr_1.decay",
            "adsr_1.sustain",
            "adsr_1.release",
            "adsr_2.attack",
            "adsr_2.decay",
            "adsr_2.sustain",
            "adsr_2.release",
            "lfo_1.frequency",
            "lfo_1.mod_depth",
            "vco_2.tuning",
            "vco_2.mod_depth",
            "vco_2.shape",
            "mixer.vco_1",
            "mixer.vco_2",
            "mixer.noise",
        ]
    ),
    _note_params(),
)

# Every inferable voice param in native order, mirroring the online datamodule.
TORCHSYNTH_FULL_PARAM_SPEC = ParamSpec(
    _continuous([param.key for param in INFERABLE_SPEC]),
    _note_params(),
)
