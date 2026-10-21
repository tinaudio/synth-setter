"""Isolated single-note rendering through the dexed-py native DX7 engine."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from numbers import Integral, Real

import numpy as np

from synth_setter.data.vst.dexed_param_spec import DEXED_PARAMETERS, DEXED_VERSION
from synth_setter.data.vst.param_spec import DiscreteLiteralParameter, ParameterValue
from synth_setter.data.vst.renderers import AudioRenderer


def _preset_array(params: Mapping[str, ParameterValue]) -> np.ndarray:
    """Apply validated scalar overrides to a fresh upstream default preset.

    :param params: Native values keyed by named Preset coordinates.
    :returns: The complete native 145-coordinate array.
    :raises ValueError: A coordinate is unknown, non-finite, or outside its native domain.
    """
    from dexed import Preset

    values = Preset().to_array()
    parameters = {parameter.name: parameter for parameter in DEXED_PARAMETERS}
    unknown = params.keys() - parameters.keys()
    if unknown:
        raise ValueError(f"unknown Dexed parameters: {sorted(unknown)}")
    for index, (name, parameter) in enumerate(parameters.items()):
        if name not in params:
            continue
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite scalar")
        scalar = float(value)
        if not parameter.min <= scalar <= parameter.max:
            raise ValueError(f"{name} must be in [{parameter.min}, {parameter.max}]")
        if isinstance(parameter, DiscreteLiteralParameter) and not scalar.is_integer():
            raise ValueError(f"{name} must be an integer")
        values[index] = scalar
    return values


@dataclass(kw_only=True)
class DexedRenderer(AudioRenderer):
    """Render a fresh native synth per call; stereo duplicates the mono engine output.

    .. attribute :: synth_version

       Required dexed-py package version, pinned in the persisted synth identity.
    """

    synth_version: str = DEXED_VERSION

    def __post_init__(self) -> None:
        if self.plugin_path != "dexed" or self.plugin_state_path:
            raise ValueError("Dexed requires plugin_path='dexed' and no plugin_state_path")
        if self.channels not in (1, 2):
            raise ValueError("Dexed supports one or two channels")
        if not math.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be finite and positive")
        if not math.isfinite(self.signal_duration_seconds) or self.signal_duration_seconds <= 0:
            raise ValueError("signal_duration_seconds must be finite and positive")
        if int(self.sample_rate * self.signal_duration_seconds) < 1:
            raise ValueError("render duration must contain at least one sample")
        installed = version("dexed-py")
        if installed != DEXED_VERSION or self.synth_version != DEXED_VERSION:
            raise ValueError(
                f"Dexed requires dexed-py {DEXED_VERSION}; installed={installed}, "
                f"configured={self.synth_version}"
            )

    def render(
        self,
        params: Mapping[str, ParameterValue],
        midi_note: int,
        velocity: int,
        note_start_and_end: tuple[float, float],
        *,
        warmup: bool = False,
    ) -> np.ndarray:
        """Render native preset values without clipping or per-patch normalization.

        :param params: Partial scalar Preset fields; dotted indices select array coordinates.
        :param midi_note: MIDI pitch in [0, 127].
        :param velocity: MIDI velocity in [0, 127]; zero produces silence.
        :param note_start_and_end: Ordered note endpoints within the output duration, in seconds.
        :param warmup: Unsupported editor warm-up request.
        :returns: Finite float32 audio shaped (channels, samples).
        :raises ValueError: Parameters, MIDI values, timing, or warm-up are invalid.
        :raises RuntimeError: The native engine returns invalid audio.
        """
        from dexed import DexedSynth, Preset

        if warmup:
            raise ValueError("Dexed has no editor to warm up")
        for name, value in (("midi_note", midi_note), ("velocity", velocity)):
            if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= 127:
                raise ValueError(f"{name} must be an integer in [0, 127]")
        start, end = note_start_and_end
        if not (math.isfinite(start) and math.isfinite(end)) or not (
            0 <= start <= end <= self.signal_duration_seconds
        ):
            raise ValueError("note endpoints must satisfy 0 <= start <= end <= duration")

        preset_values = _preset_array(params)

        samples = int(self.sample_rate * self.signal_duration_seconds)
        start_sample = math.ceil(start * self.sample_rate)
        audio = np.zeros((self.channels, samples), dtype=np.float32)
        if start_sample >= samples or start == end or velocity == 0:
            return audio
        synth = DexedSynth(sample_rate=self.sample_rate)
        synth.load_preset(Preset.from_array(preset_values))
        frames = samples - start_sample
        # One guard sample avoids native float-to-frame truncation losing the final frame.
        native = synth.render(
            midi_note=int(midi_note),
            velocity=int(velocity),
            note_duration=end - start,
            render_duration=(frames + 1) / self.sample_rate,
        )
        if native.ndim != 1 or native.size < frames or not np.isfinite(native).all():
            raise RuntimeError("Dexed returned invalid native audio")
        audio[:, start_sample:] = native[:frames]
        return audio
