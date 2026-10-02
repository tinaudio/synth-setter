"""Render existing normalized KR106 parameter dictionaries without a VST host."""

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from synth_setter.data.vst.kr106_native_runtime import (
    KR106NativeModule,
    NativeBaseline,
    NativeParameter,
    import_kr106_native,
)
from synth_setter.data.vst.param_map import SynthParamMap
from synth_setter.data.vst.param_spec import ParameterValue, require_scalar_synth_params
from synth_setter.data.vst.renderers import AudioRenderer, _validate_rendered_audio
from synth_setter.workspace import operator_workspace


@dataclass(kw_only=True)
class KR106NativeRenderer(AudioRenderer):
    """Render isolated KR106 clips with verified native identities and baseline state.

    .. attribute :: plugin_path
        In-process ``kr106_native`` sentinel.
    .. attribute :: sample_rate
        Output frames per second.
    .. attribute :: channels
        One for downmixed mono or two for stereo.
    .. attribute :: signal_duration_seconds
        Exact clip duration including release.
    .. attribute :: plugin_state_path
        Pinned baseline, relative to the workspace or absolute.
    .. attribute :: parameter_map
        Normalized-name to native-control identities and provenance.
    .. attribute :: block_size
        Maximum DSP block length before note-boundary splitting.
    """

    plugin_path: str
    sample_rate: float
    channels: int
    signal_duration_seconds: float
    plugin_state_path: str = ""
    parameter_map: SynthParamMap
    block_size: int = 512
    _native: KR106NativeModule = field(init=False, repr=False)
    _parameters: dict[str, NativeParameter] = field(init=False, repr=False)
    _baseline: dict[int, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the engine, baseline and every mapped native identity.

        :raises ValueError: Configuration, baseline, or parameter provenance is invalid.
        """
        if self.plugin_path != "kr106_native":
            raise ValueError("KR106-native renderer requires plugin_path='kr106_native'")
        if not self.plugin_state_path:
            raise ValueError("KR106-native renderer requires a pinned baseline")
        if self.channels not in (1, 2):
            raise ValueError("KR106-native supports one or two output channels")
        if (
            not math.isfinite(self.sample_rate)
            or self.sample_rate <= 0
            or not math.isfinite(self.signal_duration_seconds)
            or self.signal_duration_seconds <= 0
            or self.block_size <= 0
        ):
            raise ValueError("sample rate, duration and block size must be positive and finite")
        self._native = import_kr106_native()
        metadata = [NativeParameter.model_validate(item) for item in self._native.get_parameters()]
        by_id = {parameter.id: parameter for parameter in metadata}
        snapshot = self.parameter_map.kr106_native
        if (
            snapshot is None
            or snapshot.plugin_version != self._native.get_version()
            or snapshot.parameter_count != len(metadata)
            or len(by_id) != len(metadata)
        ):
            raise ValueError("KR106-native parameter snapshot does not match the engine")
        self._parameters = {}
        for name, ref in self.parameter_map.kr106_native_params().items():
            parameter = by_id.get(ref.native_id)
            if parameter is None or parameter.name != ref.name:
                raise ValueError(f"stale KR106-native identity for {name!r}")
            self._parameters[name] = parameter
        baseline_path = Path(self.plugin_state_path).expanduser()
        if not baseline_path.is_absolute():
            baseline_path = operator_workspace() / baseline_path
        baseline_bytes = baseline_path.read_bytes()
        if (
            hashlib.sha256(baseline_bytes).hexdigest()
            != self.parameter_map.kr106_native_preset_sha256
        ):
            raise ValueError("KR106-native baseline SHA-256 does not match the parameter map")
        baseline = NativeBaseline.model_validate_json(baseline_bytes)
        if baseline.source_vstpreset_sha256 != self.parameter_map.preset_sha256:
            raise ValueError("KR106-native baseline does not originate from the mapped VST preset")
        self._baseline = {int(key): value for key, value in baseline.parameters.items()}
        if set(self._baseline) != set(range(56)):
            raise ValueError("KR106-native baseline must cover all 56 DSP parameters")
        for param_id, value in self._baseline.items():
            parameter = by_id[param_id]
            if not math.isfinite(value) or not parameter.minimum <= value <= parameter.maximum:
                raise ValueError(f"invalid KR106-native baseline value for {parameter.name}")

    def native_parameter_values(self, params: Mapping[str, ParameterValue]) -> dict[str, float]:
        """Convert existing normalized labels to native control values.

        :param params: Repository-named normalized KR106 controls.
        :returns: Native values keyed by the same repository names.
        :raises KeyError: A requested control has no verified native mapping.
        :raises ValueError: A value is non-finite or outside the normalized range.
        """
        values = require_scalar_synth_params(params)
        unknown = sorted(values.keys() - self._parameters.keys())
        if unknown:
            raise KeyError(f"unknown KR106-native parameters: {', '.join(unknown)}")
        result: dict[str, float] = {}
        for name, value in values.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("KR106 parameter values must be finite and in [0, 1]")
            parameter = self._parameters[name]
            native_value = parameter.minimum + value * (parameter.maximum - parameter.minimum)
            if parameter.kind == "bool":
                native_value = float(value > 0.5)
            elif parameter.kind == "int":
                native_value = float(math.floor(native_value + 0.5))
            result[name] = native_value
        return result

    def render(
        self,
        params: Mapping[str, ParameterValue],
        midi_note: int,
        velocity: int,
        note_start_and_end: tuple[float, float],
        *,
        warmup: bool = False,
    ) -> np.ndarray:
        """Render a fresh native engine into channel-leading float32 audio.

        :param params: Normalized controls using the unchanged KR106 ParamSpec.
        :param midi_note: MIDI pitch in [0, 127].
        :param velocity: MIDI velocity in [0, 127].
        :param note_start_and_end: Start and end times within the configured clip.
        :param warmup: Unused because the native backend has no editor.
        :returns: Finite audio with exactly the configured channels and frame count.
        :raises ValueError: Pitch, velocity, timing or parameter values are invalid.
        """
        del warmup
        values = self.native_parameter_values(params)
        start, end = note_start_and_end
        if not 0 <= start < end <= self.signal_duration_seconds:
            raise ValueError("note times must satisfy 0 <= start < end <= signal duration")
        if not 0 <= midi_note <= 127 or not 0 <= velocity <= 127:
            raise ValueError("MIDI pitch and velocity must be in [0, 127]")
        samples = int(self.sample_rate * self.signal_duration_seconds)
        audio = self._native.render_note(
            {self._parameters[name].id: value for name, value in values.items()},
            midi_note,
            velocity,
            min(samples, math.ceil(start * self.sample_rate)),
            min(samples, math.ceil(end * self.sample_rate)),
            samples,
            self.sample_rate,
            self.block_size,
            baseline_parameters=self._baseline,
        )
        output = audio if self.channels == 2 else audio.mean(axis=0, keepdims=True)
        return _validate_rendered_audio(output, channels=self.channels, samples=samples)
