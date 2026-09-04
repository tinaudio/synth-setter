"""Common audio-rendering interface and VST backend implementations.

Usage::

    renderer = PedalboardRenderer(
        plugin_path="synth.vst3",
        sample_rate=44100,
        channels=2,
        signal_duration_seconds=1.0,
    )
    audio = renderer.render({"cutoff": 0.5}, 60, 100, (0.0, 0.5))
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import numpy as np

from synth_setter.data.vst.dawdreamer_runtime import settle_dawdreamer_preset
from synth_setter.data.vst.param_map import SynthParamMap
from synth_setter.data.vst.param_spec import ParameterValue, require_scalar_synth_params
from synth_setter.data.vst.torchsynth_param_spec import (
    DEFAULT_NORMALIZED_PATCH,
    TORCHSYNTH_FULL_PARAM_SPEC,
)
from synth_setter.data.vst.surgepy_runtime import (
    SurgePyModule,
    SurgePyNamedParam,
    SurgePySynth,
    import_surgepy,
    iter_surgepy_named_params,
)
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.renderer_backend import FAUST_PLUGIN_NAME, SURGEPY_PLUGIN_NAME

if TYPE_CHECKING:
    from pedalboard import VST3Plugin

# Both DawDreamer hosts pin the engine block size as a runtime invariant.
DAWDREAMER_BLOCK_SIZE = 2048
# Surge's legacy automation scale is load-bearing for saved host parameter values.
_SURGE_INT_NORMALIZED_OFFSET = 0.005
_SURGE_INT_NORMALIZED_SCALE = 0.99
_SURGEPY_NATIVE_CHANNELS = 2


def _sample_index_at_or_after(time_seconds: float, sample_rate: float) -> int:
    """Return the first sample at or after an event time.

    :param time_seconds: Requested event time in seconds.
    :param sample_rate: Samples per second.
    :returns: First sample at or after the event.
    """
    sample_position = time_seconds * sample_rate
    sample_before = math.floor(sample_position)
    if time_seconds <= sample_before / sample_rate:
        return sample_before
    return sample_before + 1


def _align_native_attack(
    audio: np.ndarray,
    *,
    samples: int,
    start_sample: int,
    source_start: int,
) -> np.ndarray:
    """Place the first native attack sample at the requested output sample.

    :param audio: Native buffer shaped ``(2, N)`` with channels first.
    :param samples: Retained output sample count.
    :param start_sample: Requested first output sample.
    :param source_start: Native buffer index where the attack begins.
    :returns: Stereo float32 output with a silent pre-event prefix.
    :raises ValueError: If the buffer shape or sample interval cannot satisfy the output.
    """
    source_end = source_start + samples - start_sample
    if (
        audio.ndim != 2
        or audio.shape[0] != _SURGEPY_NATIVE_CHANNELS
        or not 0 <= start_sample <= samples
        or source_start < 0
        or audio.shape[1] < source_end
    ):
        raise ValueError("invalid SurgePy attack buffer or sample interval")
    rendered = np.asarray(audio, dtype=np.float32)
    aligned = np.zeros((_SURGEPY_NATIVE_CHANNELS, samples), dtype=np.float32)
    aligned[:, start_sample:] = rendered[:, source_start:source_end]
    return aligned


class _DawDreamerParameterDescription(TypedDict):
    """Parameter identity fields returned by DawDreamer.

    .. attribute :: index

       Host parameter index.

    .. attribute :: name

       Host parameter display name.
    """

    index: int
    name: str


class _DawDreamerPlugin(Protocol):
    """DawDreamer plugin-processor surface used by the renderer."""

    def get_parameters_description(self) -> list[_DawDreamerParameterDescription]: ...

    def clear_midi(self) -> None: ...

    def set_parameter(self, index: int, value: float) -> None: ...

    def add_midi_note(self, pitch: int, velocity: int, start: float, duration: float) -> None: ...

    def load_vst3_preset(self, path: str) -> None: ...

    def load_preset(self, path: str) -> None: ...


class _DawDreamerFaustProcessor(Protocol):
    """DawDreamer Faust-processor surface used by the renderer.

    .. attribute :: num_voices

       Polyphonic voice count configured before compilation.

    .. attribute :: group_voices

       Whether polyphonic controls share one parameter address.
    """

    num_voices: int
    group_voices: bool

    def set_dsp_string(self, source: str) -> None:
        """Set the checked-in Faust source string.

        :param source: Complete Faust program.
        """
        ...

    def compile(self) -> None:
        """Compile the configured source."""
        ...

    def get_parameters_description(self) -> list[_DawDreamerParameterDescription]:
        """Return compiled parameter metadata.

        :returns: Parameter descriptions in compiled address order.
        """
        ...

    def clear_midi(self) -> None:
        """Clear scheduled MIDI events."""
        ...

    def set_parameter(self, address: str, value: float) -> bool:
        """Set one native value by exact address.

        :param address: Exact compiled Faust address.
        :param value: Renderer-native value.
        :returns: Whether the address accepted the value.
        """
        ...

    def set_automation(self, address: str, values: np.ndarray) -> bool:
        """Set audio-rate values by exact address.

        :param address: Exact compiled Faust address.
        :param values: One renderer-native value per output sample.
        :returns: Whether the address accepted the automation.
        """
        ...

    def add_midi_note(
        self, pitch: int, velocity: int, start: float, duration: float
    ) -> None:
        """Schedule one MIDI note.

        :param pitch: MIDI note number.
        :param velocity: MIDI velocity.
        :param start: Note-on time in seconds.
        :param duration: Note duration in seconds.
        """
        ...


type _DawDreamerProcessor = _DawDreamerPlugin | _DawDreamerFaustProcessor


class _DawDreamerEngine(Protocol):
    """DawDreamer render-engine surface used by the renderer."""

    def make_plugin_processor(self, name: str, path: str) -> _DawDreamerPlugin: ...

    def make_faust_processor(self, name: str) -> _DawDreamerFaustProcessor: ...

    def load_graph(self, graph: list[tuple[_DawDreamerProcessor, list[object]]]) -> None: ...

    def render(self, duration: float) -> None: ...

    def get_audio(self) -> np.ndarray: ...


class _DawDreamerModule(Protocol):
    """Lazily imported DawDreamer module surface."""

    def RenderEngine(self, sample_rate: float, block_size: int) -> _DawDreamerEngine: ...


class NonFiniteAudioError(ValueError):
    """Rendered audio contained NaN or infinity."""


def _validate_rendered_audio(
    audio: np.ndarray,
    *,
    channels: int,
    samples: int,
) -> np.ndarray:
    """Validate the shared backend output contract without changing samples.

    :param audio: Channel-leading rendered audio.
    :param channels: Required output channel count.
    :param samples: Required output sample count.
    :returns: The validated audio without replacement.
    :raises ValueError: If the shape is invalid.
    :raises NonFiniteAudioError: If any sample is non-finite.
    """
    if audio.shape != (channels, samples):
        raise ValueError(
            f"rendered audio shape {audio.shape} != expected {(channels, samples)}"
        )
    if not np.isfinite(audio).all():
        raise NonFiniteAudioError("rendered audio must contain only finite samples")
    return audio


@dataclass(kw_only=True)
class AudioRenderer(ABC):
    """Render one parameterized MIDI note through a synthesizer plugin.

    .. attribute :: plugin_path

       Plugin path or interpreter-resolved backend sentinel.

    .. attribute :: sample_rate

       Render sample rate in Hz.

    .. attribute :: channels

       Requested output channel count.

    .. attribute :: signal_duration_seconds

       Duration of each rendered sample.

    .. attribute :: plugin_state_path

       Optional baseline preset path.
    """

    plugin_path: str
    sample_rate: float
    channels: int
    signal_duration_seconds: float
    plugin_state_path: str | None = None

    @abstractmethod
    def render(
        self,
        params: Mapping[str, ParameterValue],
        midi_note: int,
        velocity: int,
        note_start_and_end: tuple[float, float],
        *,
        warmup: bool = False,
    ) -> np.ndarray:
        """Render one note and return audio shaped ``(channels, samples)``.

        :param params: Renderer-native parameter values keyed by renderer identity.
        :param midi_note: MIDI pitch of the note.
        :param velocity: MIDI note velocity in the inclusive range ``[0, 127]``.
        :param note_start_and_end: Note start and end times in seconds.
        :param warmup: Whether to perform the backend's optional editor warm-up.
        :returns: Rendered audio with channels on the first axis.
        """


@dataclass(kw_only=True)
class PedalboardRenderer(AudioRenderer):
    """Render through the existing pedalboard implementation.

    .. attribute :: plugin

       Optional preloaded pedalboard plugin instance.
    """

    plugin: VST3Plugin | None = field(default=None, repr=False)

    def render(
        self,
        params: Mapping[str, ParameterValue],
        midi_note: int,
        velocity: int,
        note_start_and_end: tuple[float, float],
        *,
        warmup: bool = False,
    ) -> np.ndarray:
        """Render one note through pedalboard's VST3 host.

        :param params: Normalized plugin parameter values keyed by plugin name.
        :param midi_note: MIDI pitch of the note.
        :param velocity: MIDI note velocity in the inclusive range ``[0, 127]``.
        :param note_start_and_end: Note start and end times in seconds.
        :param warmup: Whether to perform pedalboard's optional editor warm-up.
        :returns: Rendered audio with channels on the first axis.
        """
        from synth_setter.data.vst.core import render_params

        scalar_params = require_scalar_synth_params(params)
        return _validate_rendered_audio(
            render_params(
                self.plugin_path,
                scalar_params,
                midi_note,
                velocity,
                note_start_and_end,
                self.signal_duration_seconds,
                self.sample_rate,
                self.channels,
                plugin_state_path=self.plugin_state_path,
                plugin=self.plugin,
                warmup=warmup,
            ),
            channels=self.channels,
            samples=int(self.sample_rate * self.signal_duration_seconds),
        )


@dataclass(kw_only=True)
class SurgePyRenderer(AudioRenderer):
    """Render Surge patches directly through the in-process Surge engine.

    .. attribute :: parameter_map
        :type: SynthParamMap

        Verified cross-host parameter identities and patch provenance.

    .. attribute :: synth
        :type: SurgePySynth

        Fresh native engine used by the current render.
    """

    parameter_map: SynthParamMap = field(kw_only=True)
    synth: SurgePySynth = field(init=False, repr=False)
    _parameter_ids: dict[str, SurgePyNamedParam] = field(init=False, repr=False)
    _surgepy: SurgePyModule = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate immutable resources before the first native render.

        :raises ValueError: If the backend identity, channels, or patch provenance is invalid.
        """
        if self.plugin_path != SURGEPY_PLUGIN_NAME:
            raise ValueError('SurgePy renderer requires plugin_path="surgepy"')
        if self.channels not in (1, _SURGEPY_NATIVE_CHANNELS):
            raise ValueError("SurgePy renderer supports one or two output channels")
        if not self.plugin_state_path:
            raise ValueError("SurgePy renderer requires an .fxp patch")
        patch_path = Path(self.plugin_state_path).expanduser().resolve()
        if patch_path.suffix.casefold() != ".fxp":
            raise ValueError("SurgePy renderer requires an .fxp patch")
        if not self.parameter_map.surgepy_preset_sha256:
            raise ValueError("parameter map has no SurgePy patch provenance")
        self.plugin_state_path = str(patch_path)
        self._verify_patch_digest()
        self._surgepy = import_surgepy()

    def _verify_patch_digest(self) -> None:
        """Reject patch bytes that differ from the mapped baseline.

        :raises ValueError: If the patch changed after map generation.
        """
        expected_digest = cast(str, self.parameter_map.surgepy_preset_sha256)
        patch_path = Path(cast(str, self.plugin_state_path))
        actual_digest = hashlib.sha256(patch_path.read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError("SurgePy patch SHA-256 does not match the parameter map")

    def _initialize_synth(self) -> None:
        """Create a fresh preset-loaded synth and validate mapped native identities.

        :raises RuntimeError: If the patch cannot be loaded.
        :raises ValueError: If the live engine, patch, or parameter identities drifted.
        """
        self._verify_patch_digest()
        self.synth = self._surgepy.createSurge(self.sample_rate)
        if not self.synth.loadPatch(cast(str, self.plugin_state_path)):
            raise RuntimeError(f"SurgePy could not load patch {self.plugin_state_path!r}")
        snapshot = self.parameter_map.surgepy
        if snapshot is None:
            raise ValueError("parameter map has no SurgePy snapshot")
        version = self._surgepy.getVersion()
        if version != snapshot.plugin_version:
            raise ValueError(f"SurgePy version {version!r} != map {snapshot.plugin_version!r}")
        by_id: dict[int, SurgePyNamedParam] = {}
        for parameter in iter_surgepy_named_params(self.synth.getPatch()):
            synth_side_id = parameter.getId().getSynthSideId()
            if synth_side_id not in by_id:
                by_id[synth_side_id] = parameter
        if len(by_id) != snapshot.parameter_count:
            raise ValueError(
                f"SurgePy parameter count {len(by_id)} != map {snapshot.parameter_count}"
            )
        self._parameter_ids = {}
        for name, reference in self.parameter_map.surgepy_params().items():
            parameter = by_id.get(reference.synth_side_id)
            if parameter is None or parameter.getName() != reference.name:
                raise ValueError(
                    f"stale SurgePy identity for {name!r} at ID {reference.synth_side_id}"
                )
            self._parameter_ids[name] = parameter

    def _native_parameter_value(
        self,
        parameter: SurgePyNamedParam,
        normalized_value: float,
    ) -> float:
        """Convert one normalized host value to the Surge-native parameter range.

        :param parameter: Live native parameter handle.
        :param normalized_value: VST-normalized value in the inclusive range ``[0, 1]``.
        :returns: Native float, rounded for integer and Boolean parameters.
        :raises ValueError: If the normalized value or native type is invalid.
        """
        if not math.isfinite(normalized_value) or not 0.0 <= normalized_value <= 1.0:
            raise ValueError("SurgePy parameter values must be finite and in [0, 1]")
        minimum = self.synth.getParamMin(parameter)
        maximum = self.synth.getParamMax(parameter)
        native_value = minimum + normalized_value * (maximum - minimum)
        value_type = self.synth.getParamValType(parameter)
        if value_type == "float":
            return native_value
        if value_type == "bool":
            return float(normalized_value > 0.5)
        if value_type == "int":
            span = maximum - minimum
            surge_scaled = (
                int(
                    (normalized_value - _SURGE_INT_NORMALIZED_OFFSET)
                    * span
                    / _SURGE_INT_NORMALIZED_SCALE
                    + 0.5
                )
                + int(minimum)
            )
            return float(min(max(surge_scaled, int(minimum)), int(maximum)))
        raise ValueError(f"unsupported SurgePy parameter type {value_type!r}")

    def _apply_parameters(self, params: dict[str, float]) -> None:
        """Apply repository-normalized values to the fresh native synth.

        Unmapped keys raise ``KeyError`` and invalid values ``ValueError``,
        both before any native write.

        :param params: Normalized values keyed by repository parameter name.
        """
        self._reject_unknown_parameters(params)
        for name, value in params.items():
            parameter = self._parameter_ids[name]
            self.synth.setParamVal(parameter, self._native_parameter_value(parameter, value))

    def _reject_unknown_parameters(self, params: dict[str, float]) -> None:
        """Fail before writing to the engine when a key has no mapped identity.

        :param params: Normalized values keyed by repository parameter name.
        :raises KeyError: If a requested parameter has no mapped native identity.
        """
        unknown = sorted(params.keys() - self._parameter_ids.keys())
        if unknown:
            raise KeyError(f"unknown SurgePy parameter key(s): {', '.join(unknown)}")

    def native_parameter_values(self, params: dict[str, float]) -> dict[str, float]:
        """Resolve normalized values on a fresh verified engine, without rendering.

        Shares the render path's validation: unmapped keys raise ``KeyError``,
        out-of-range or non-finite values ``ValueError``.

        :param params: Normalized values keyed by repository parameter name.
        :returns: Surge-native values keyed by repository parameter name.
        """
        self._initialize_synth()
        self._reject_unknown_parameters(params)
        return {
            name: self._native_parameter_value(self._parameter_ids[name], value)
            for name, value in params.items()
        }

    def _render_note_blocks(
        self,
        *,
        midi_note: int,
        velocity: int,
        samples: int,
        start: float,
        end: float,
    ) -> np.ndarray:
        """Render one note over complete native processing blocks.

        :param midi_note: MIDI pitch of the note.
        :param velocity: MIDI velocity in the inclusive range ``[0, 127]``.
        :param samples: Exact output sample count after block trimming.
        :param start: Requested note start in seconds.
        :param end: Requested note end in seconds.
        :returns: Stereo output with the native attack aligned to the requested start.
        """
        start_sample = _sample_index_at_or_after(start, self.sample_rate)
        if start_sample >= samples:
            return np.zeros((_SURGEPY_NATIVE_CHANNELS, samples), dtype=np.float32)
        block_size = self.synth.getBlockSize()
        num_blocks = math.ceil(samples / block_size)
        start_block = start_sample // block_size
        end_sample = _sample_index_at_or_after(end, self.sample_rate)
        note_samples = max(1, end_sample - start_sample)
        note_blocks = math.ceil(note_samples / block_size)
        end_block = min(num_blocks, start_block + note_blocks)
        audio = self.synth.createMultiBlock(num_blocks)
        try:
            self._process_blocks(audio, start_block=0, num_blocks=start_block)
            self.synth.playNote(0, midi_note, velocity)
            self._process_blocks(
                audio,
                start_block=start_block,
                num_blocks=end_block - start_block,
            )
            self.synth.releaseNote(0, midi_note)
            self._process_blocks(
                audio,
                start_block=end_block,
                num_blocks=num_blocks - end_block,
            )
        finally:
            self.synth.allNotesOff()
        return _align_native_attack(
            audio,
            samples=samples,
            start_sample=start_sample,
            source_start=start_block * block_size,
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
        """Render one isolated note through a fresh SurgePy synth.

        :param params: Normalized values keyed by repository parameter name.
        :param midi_note: MIDI pitch of the note.
        :param velocity: MIDI velocity in the inclusive range ``[0, 127]``.
        :param note_start_and_end: Note start and end times in seconds, expanded to
            cover complete native processing blocks.
        :param warmup: Unused; SurgePy has no plugin editor.
        :returns: Rendered channel-leading float32 audio.
        :raises ValueError: If note timing is invalid.
        """
        del warmup
        scalar_params = require_scalar_synth_params(params)
        self._initialize_synth()
        self._apply_parameters(scalar_params)
        start, end = note_start_and_end
        if not 0.0 <= start < end <= self.signal_duration_seconds:
            raise ValueError("note times must satisfy 0 <= start < end <= signal duration")
        samples = int(self.sample_rate * self.signal_duration_seconds)
        trimmed = self._render_note_blocks(
            midi_note=midi_note,
            velocity=velocity,
            samples=samples,
            start=start,
            end=end,
        )
        matched = (
            trimmed
            if self.channels == _SURGEPY_NATIVE_CHANNELS
            else trimmed.mean(axis=0, keepdims=True)
        )
        return _validate_rendered_audio(matched, channels=self.channels, samples=samples)

    def _process_blocks(
        self,
        audio: np.ndarray,
        *,
        start_block: int,
        num_blocks: int,
    ) -> None:
        """Process a valid block interval into the shared stereo output buffer.

        :param audio: Two-dimensional SurgePy buffer with two channels and enough samples for the
            requested block interval.
        :param start_block: Non-negative first output block to populate.
        :param num_blocks: Non-negative number of synth blocks to process.
        :raises ValueError: If the buffer cannot contain the requested interval.
        """
        block_size = self.synth.getBlockSize()
        required_samples = (start_block + num_blocks) * block_size
        if (
            audio.ndim != 2
            or audio.shape[0] != _SURGEPY_NATIVE_CHANNELS
            or start_block < 0
            or num_blocks < 0
            or audio.shape[1] < required_samples
        ):
            raise ValueError("invalid SurgePy output buffer or block interval")
        if num_blocks > 0:
            self.synth.processMultiBlock(audio, start_block, num_blocks)


@dataclass(kw_only=True)
class TorchSynthRenderer(AudioRenderer):
    """Render through the in-process torchsynth ``Voice`` (no plugin host).

    Shares the online datamodule's cached voice and render path
    (``render_torchsynth``): sampled params (normalized ``[0, 1]`` values keyed
    ``module.name``) override the spec module's baseline patch, so every
    un-sampled knob is pinned. ``plugin_path`` is the bare backend name
    (``"torchsynth"``) and ``plugin_state_path`` is unused. MIDI velocity is
    ignored — the voice has no velocity input, and production configs hold it
    constant per run. The note-on offset is emulated by delaying the rendered
    audio, and the voice's mono output is repeated across requested channels.
    """

    def __post_init__(self) -> None:
        """Verify the live voice against the pinned spec (``ValueError`` on drift)."""
        # Lazy: pulls torch + lightning, which this module must not import eagerly.
        from synth_setter.data.torchsynth_datamodule import _make_renderer
        from synth_setter.data.vst.torchsynth_param_spec import verify_voice_matches_spec

        verify_voice_matches_spec(
            _make_renderer(int(self.sample_rate), self._signal_length()).voice
        )

    def _signal_length(self) -> int:
        """Return the render length in samples.

        :returns: Configured duration at the configured sample rate.
        """
        return int(self.sample_rate * self.signal_duration_seconds)

    def render(
        self,
        params: Mapping[str, ParameterValue],
        midi_note: int,
        velocity: int,
        note_start_and_end: tuple[float, float],
        *,
        warmup: bool = False,
    ) -> np.ndarray:
        """Write params over the baseline patch and render one note in-process.

        :param params: Normalized values keyed ``module.name``, overriding the
            baseline patch; keys outside the pinned voice spec are rejected.
        :param midi_note: MIDI pitch of the note.
        :param velocity: Ignored; torchsynth's voice has no velocity input.
        :param note_start_and_end: Note start and end times in seconds. The
            note-on length clamps into the keyboard's pinned duration range and
            the start offset delays the audio with a zero-filled head.
        :param warmup: Unused; there is no plugin editor to warm up.
        :returns: Rendered audio with channels on the first axis.
        :raises KeyError: A requested key has no matching voice parameter.
        """
        del velocity, warmup
        params = require_scalar_synth_params(params)
        # Lazy: pulls torch + lightning, which this module must not import eagerly.
        import torch

        from synth_setter.data.torchsynth_datamodule import render_torchsynth

        unknown = sorted(params.keys() - DEFAULT_NORMALIZED_PATCH.keys())
        if unknown:
            raise KeyError(f"unknown torchsynth parameter key(s): {', '.join(unknown)}")
        row = TORCHSYNTH_FULL_PARAM_SPEC.encode(
            {**DEFAULT_NORMALIZED_PATCH, **params},
            {"pitch": midi_note, "note_start_and_end": note_start_and_end},
        )
        samples = self._signal_length()
        audio = render_torchsynth(
            torch.tensor([row], dtype=torch.float32),
            sample_rate=int(self.sample_rate),
            signal_length=samples,
        ).numpy()
        # The note-on delay is applied by the render; the mono voice now fans out to channels.
        if self.channels > 1:
            audio = np.repeat(audio, self.channels, axis=0)
        return _validate_rendered_audio(audio, channels=self.channels, samples=samples)


@dataclass(kw_only=True)
class DawDreamerFaustRenderer(AudioRenderer):
    """Compile and render one checked-in Faust program through DawDreamer.

    .. attribute :: param_spec_name

       Shared source and exact-address parameter-spec identity.

    .. attribute :: block_size

       DawDreamer engine block size.

    .. attribute :: reload_processor_each_render

       Whether subsequent calls compile a fresh native graph.

    .. attribute :: engine

       DawDreamer render engine holding the compiled graph.

    .. attribute :: processor

       Compiled Faust processor.
    """

    param_spec_name: ParamSpecName = field(kw_only=True)
    block_size: int = DAWDREAMER_BLOCK_SIZE
    reload_processor_each_render: bool = True
    engine: _DawDreamerEngine = field(init=False, repr=False)
    processor: _DawDreamerFaustProcessor = field(init=False, repr=False)
    _daw: _DawDreamerModule = field(init=False, repr=False)
    _dsp_source: str = field(init=False, repr=False)
    _num_voices: int = field(init=False, repr=False)
    _parameter_addresses: tuple[str, ...] = field(init=False, repr=False)
    _has_rendered: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve checked-in identities and compile the first native graph.

        :raises ValueError: The direct renderer configuration accepts an external resource.
        """
        from synth_setter.data.vst.faust_param_spec import resolve_faust_param_spec
        from synth_setter.data.vst.faust_sources import resolve_faust_dsp

        if self.plugin_path != FAUST_PLUGIN_NAME:
            raise ValueError('Faust renderer requires plugin_path="faust"')
        if self.plugin_state_path:
            raise ValueError("Faust renderer does not accept plugin_state_path")
        dsp = resolve_faust_dsp(self.param_spec_name)
        spec = resolve_faust_param_spec(self.param_spec_name)
        self._dsp_source = dsp.source
        self._num_voices = dsp.num_voices
        self._parameter_addresses = tuple(spec.synth_param_names)
        self._daw = cast(_DawDreamerModule, import_module("dawdreamer"))
        self._initialize_graph()

    def _initialize_graph(self) -> None:
        """Compile one source string and verify its exact parameter addresses.

        :raises ValueError: Compiled addresses drift from the registered specification.
        """
        self.engine = self._daw.RenderEngine(self.sample_rate, self.block_size)
        self.processor = self.engine.make_faust_processor("synth")
        self.processor.num_voices = self._num_voices
        self.processor.group_voices = True
        self.processor.set_dsp_string(self._dsp_source)
        self.processor.compile()
        self.engine.load_graph([(self.processor, [])])
        compiled_addresses = tuple(
            str(item["name"]) for item in self.processor.get_parameters_description()
        )
        if compiled_addresses != self._parameter_addresses:
            raise ValueError("compiled Faust parameter addresses do not match the registered spec")

    def render(
        self,
        params: Mapping[str, ParameterValue],
        midi_note: int,
        velocity: int,
        note_start_and_end: tuple[float, float],
        *,
        warmup: bool = False,
    ) -> np.ndarray:
        """Apply native values by exact address and render one MIDI note.

        :param params: Complete renderer-native values keyed by exact Faust address.
        :param midi_note: MIDI pitch of the note.
        :param velocity: MIDI note velocity in the inclusive range ``[0, 127]``.
        :param note_start_and_end: Note start and end times in seconds.
        :param warmup: Unused; a source processor has no plugin editor.
        :returns: Rendered audio with channels on the first axis.
        """
        del warmup
        params = require_scalar_synth_params(params)
        self._validate_parameter_addresses(params)
        if self.reload_processor_each_render and self._has_rendered:
            self._initialize_graph()
        self._has_rendered = True
        self.processor.clear_midi()
        try:
            samples = int(self.sample_rate * self.signal_duration_seconds)
            self._apply_parameters(params, samples)
            start, end = note_start_and_end
            self.processor.add_midi_note(midi_note, velocity, start, end - start)
            self.engine.render(self.signal_duration_seconds)
            audio = np.asarray(self.engine.get_audio())
        finally:
            self.processor.clear_midi()
        return _validate_rendered_audio(
            audio,
            channels=self.channels,
            samples=samples,
        )

    def _apply_parameters(self, params: dict[str, float], samples: int) -> None:
        """Apply a complete native patch to the compiled processor.

        :param params: Exact compiled Faust addresses mapped to native values.
        :param samples: Number of output samples in the render.
        :raises RuntimeError: A native parameter or automation write is rejected.
        """
        for address, value in params.items():
            if not self.processor.set_parameter(address, value):
                raise RuntimeError(f"Faust rejected parameter address {address!r}")
            # DawDreamer 0.8.3 needs automation to propagate one-voice shared controls.
            if self._num_voices == 1 and not self.processor.set_automation(
                address, np.full(samples, value, dtype=np.float32)
            ):
                raise RuntimeError(f"Faust rejected automation address {address!r}")

    def _validate_parameter_addresses(self, params: dict[str, float]) -> None:
        """Require one complete patch under the compiled exact-address identity.

        :param params: Requested native values keyed by exact Faust address.
        :raises KeyError: A requested address is absent from the registered spec.
        :raises ValueError: One or more registered addresses are missing.
        """
        expected = set(self._parameter_addresses)
        unknown = sorted(params.keys() - expected)
        if unknown:
            raise KeyError(f"unknown Faust parameter address(es): {', '.join(unknown)}")
        missing = sorted(expected - params.keys())
        if missing:
            raise ValueError(f"missing Faust parameter address(es): {', '.join(missing)}")


@dataclass(kw_only=True)
class DawDreamerRenderer(AudioRenderer):
    """Render through DawDreamer's JUCE-backed VST host.

    .. attribute :: block_size

       DawDreamer engine block size.

    .. attribute :: parameter_map

       Validated immutable cross-host identity map.

    .. attribute :: reload_plugin_each_render

       Whether subsequent calls replace the initialized plugin graph.

    .. attribute :: engine

       DawDreamer render engine instance.

    .. attribute :: plugin

       DawDreamer plugin processor instance.
    """

    block_size: int = DAWDREAMER_BLOCK_SIZE
    parameter_map: SynthParamMap = field(kw_only=True)
    reload_plugin_each_render: bool = True
    engine: _DawDreamerEngine = field(init=False, repr=False)
    plugin: _DawDreamerPlugin = field(init=False, repr=False)
    _parameter_indices: dict[str, int] = field(init=False, repr=False)
    _daw: _DawDreamerModule = field(init=False, repr=False)
    _has_rendered: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        """Create the DawDreamer engine and load the plugin graph."""
        self.plugin_path = str(Path(self.plugin_path).expanduser().absolute())
        if self.plugin_state_path is not None:
            self.plugin_state_path = str(Path(self.plugin_state_path).expanduser().resolve())
        self._daw = cast(_DawDreamerModule, import_module("dawdreamer"))
        self._initialize_graph()

    def _initialize_graph(self) -> None:
        """Create and validate one preset-loaded plugin graph."""
        self._create_graph()
        self._load_preset()
        self._validate_parameter_map()

    def _create_graph(self) -> None:
        """Create a fresh engine and open the plugin under its managed lease."""
        from synth_setter.plugin_runtime import validated_bundle_lease

        self.engine = self._daw.RenderEngine(self.sample_rate, self.block_size)
        with validated_bundle_lease(Path(self.plugin_path)) as validated_bundle:
            self.plugin = self.engine.make_plugin_processor("synth", str(validated_bundle))
            self.engine.load_graph([(self.plugin, [])])
        self._parameter_indices = self.parameter_map.dawdreamer_indices()

    def _validate_parameter_map(self) -> None:
        """Validate the live plugin and preset against committed provenance.

        :raises ValueError: If the version, count, preset hash, index, or stored name drifted.
        """
        descriptions = self.plugin.get_parameters_description()
        snapshot = self.parameter_map.dawdreamer
        if len(descriptions) != snapshot.parameter_count:
            raise ValueError(
                f"DawDreamer parameter count {len(descriptions)} != map {snapshot.parameter_count}"
            )
        if snapshot.plugin_version:
            from synth_setter.data.vst.core import extract_renderer_version

            version = extract_renderer_version(Path(self.plugin_path))
        else:
            version = ""
        if snapshot.plugin_version and version != snapshot.plugin_version:
            raise ValueError(f"plugin version {version!r} != map {snapshot.plugin_version!r}")
        if self.parameter_map.preset_sha256 and self.plugin_state_path is None:
            raise ValueError("DawDreamer rendering requires the mapped preset")
        digest = (
            hashlib.sha256(Path(self.plugin_state_path).read_bytes()).hexdigest()
            if self.parameter_map.preset_sha256 and self.plugin_state_path
            else ""
        )
        if self.parameter_map.preset_sha256 and digest != self.parameter_map.preset_sha256:
            raise ValueError("preset SHA-256 does not match the parameter map")
        by_index = {int(item["index"]): str(item["name"]) for item in descriptions}
        for name, identity in self.parameter_map.params.items():
            ref = identity.dawdreamer
            if by_index.get(ref.index) != ref.name:
                raise ValueError(f"stale DawDreamer identity for {name!r} at index {ref.index}")

    def render(
        self,
        params: Mapping[str, ParameterValue],
        midi_note: int,
        velocity: int,
        note_start_and_end: tuple[float, float],
        *,
        warmup: bool = False,
    ) -> np.ndarray:
        """Set parameters, schedule one note, and render through DawDreamer.

        :param params: Normalized DawDreamer parameter values keyed by plugin name.
        :param midi_note: MIDI pitch of the note.
        :param velocity: MIDI note velocity in the inclusive range ``[0, 127]``.
        :param note_start_and_end: Note start and end times in seconds.
        :param warmup: Unused; DawDreamer has no non-blocking editor warm-up API.
        :returns: Rendered audio with channels on the first axis.
        """
        del warmup
        params = require_scalar_synth_params(params)
        self._validate_parameter_dispatch(params)
        if self.reload_plugin_each_render and self._has_rendered:
            self._initialize_graph()
        self._has_rendered = True
        self.plugin.clear_midi()
        try:
            for name, value in params.items():
                self.plugin.set_parameter(self._parameter_indices[name], value)
            start, end = note_start_and_end
            self.plugin.add_midi_note(midi_note, velocity, start, end - start)
            self.engine.render(self.signal_duration_seconds)
            audio = np.asarray(self.engine.get_audio())
        finally:
            self.plugin.clear_midi()
        matched = self._match_channels(audio)
        return _validate_rendered_audio(
            matched,
            channels=self.channels,
            samples=int(self.sample_rate * self.signal_duration_seconds),
        )

    def _validate_parameter_dispatch(self, params: dict[str, float]) -> None:
        """Require every requested key to target exactly one distinct host parameter.

        :param params: Requested normalized plugin values keyed by repository parameter name.
        :raises KeyError: If a requested parameter has no verified host mapping.
        :raises ValueError: If requested parameters share a host index.
        """
        unknown = sorted(params.keys() - self._parameter_indices.keys())
        if unknown:
            raise KeyError(f"unknown DawDreamer parameter key(s): {', '.join(unknown)}")
        seen: dict[int, str] = {}
        for name in params:
            index = self._parameter_indices[name]
            if previous := seen.get(index):
                raise ValueError(
                    f"{previous!r} and {name!r} target the same DawDreamer parameter index {index}"
                )
            seen[index] = name

    def _match_channels(self, audio: np.ndarray) -> np.ndarray:
        """Convert native output to the configured channel count.

        :param audio: Channel-leading audio returned by DawDreamer.
        :returns: Audio with ``self.channels`` channels.
        :raises ValueError: If the audio is not channel-leading or conversion is unsupported.
        """
        if audio.ndim != 2:
            raise ValueError(f"expected channel-leading audio, got shape {audio.shape}")
        expected_samples = int(self.sample_rate * self.signal_duration_seconds)
        if audio.shape[1] != expected_samples:
            raise ValueError(
                f"DawDreamer sample count {audio.shape[1]} != expected {expected_samples}"
            )
        if audio.shape[0] == self.channels:
            return audio
        if self.channels == 1:
            return audio.mean(axis=0, keepdims=True)
        if audio.shape[0] == 1:
            return np.repeat(audio, self.channels, axis=0)
        raise ValueError(
            f"cannot convert DawDreamer audio with {audio.shape[0]} channels "
            f"to requested {self.channels} channels"
        )

    def _load_preset(self) -> None:
        """Load the configured preset and settle it before any parameter dispatch."""
        if self.plugin_state_path is None:
            return
        if self.plugin_state_path.endswith(".vstpreset"):
            self.plugin.load_vst3_preset(self.plugin_state_path)
        else:
            self.plugin.load_preset(self.plugin_state_path)
        settle_dawdreamer_preset(
            self.engine,
            sample_rate=self.sample_rate,
            block_size=self.block_size,
        )
