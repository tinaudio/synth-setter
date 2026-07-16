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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import numpy as np

from synth_setter.data.vst.param_map import SynthParamMap
from synth_setter.data.vst.torchsynth_param_spec import (
    DEFAULT_NORMALIZED_ROW,
    KEYBOARD_DURATION_BOUNDS,
    PARAM_INDEX,
)

if TYPE_CHECKING:
    from pedalboard import VST3Plugin


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


class _DawDreamerEngine(Protocol):
    """DawDreamer render-engine surface used by the renderer."""

    def make_plugin_processor(self, name: str, path: str) -> _DawDreamerPlugin: ...

    def load_graph(self, graph: list[tuple[_DawDreamerPlugin, list[object]]]) -> None: ...

    def render(self, duration: float) -> None: ...

    def get_audio(self) -> np.ndarray: ...


class _DawDreamerModule(Protocol):
    """Lazily imported DawDreamer module surface."""

    def RenderEngine(self, sample_rate: float, block_size: int) -> _DawDreamerEngine: ...


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
    :returns: The validated audio without clipping or replacement.
    :raises ValueError: If shape, finiteness, or normalized amplitude is invalid.
    """
    if audio.shape != (channels, samples):
        raise ValueError(
            f"rendered audio shape {audio.shape} != expected {(channels, samples)}"
        )
    if not np.isfinite(audio).all():
        raise ValueError("rendered audio must contain only finite samples")
    if np.any(np.abs(audio) > 1.0):
        raise ValueError("rendered audio samples must be within [-1, 1]")
    return audio


@dataclass(kw_only=True)
class AudioRenderer(ABC):
    """Render one parameterized MIDI note through a synthesizer plugin.

    .. attribute :: plugin_path

       Filesystem path to the plugin bundle.

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
        params: dict[str, float],
        midi_note: int,
        velocity: int,
        note_start_and_end: tuple[float, float],
        *,
        warmup: bool = False,
    ) -> np.ndarray:
        """Render one note and return audio shaped ``(channels, samples)``.

        :param params: Normalized plugin parameter values keyed by plugin name.
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
        params: dict[str, float],
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

        return _validate_rendered_audio(
            render_params(
                self.plugin_path,
                params,
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
        params: dict[str, float],
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
        # Lazy: pulls torch + lightning, which this module must not import eagerly.
        import torch

        from synth_setter.data.torchsynth_datamodule import render_torchsynth

        unknown = sorted(params.keys() - PARAM_INDEX.keys())
        if unknown:
            raise KeyError(f"unknown torchsynth parameter key(s): {', '.join(unknown)}")
        row = list(DEFAULT_NORMALIZED_ROW)
        for key, value in params.items():
            row[PARAM_INDEX[key]] = value
        start, end = note_start_and_end
        minimum_duration, maximum_duration = KEYBOARD_DURATION_BOUNDS
        duration = min(max(end - start, minimum_duration), maximum_duration)
        samples = self._signal_length()
        audio = render_torchsynth(
            torch.tensor([row], dtype=torch.float32),
            sample_rate=int(self.sample_rate),
            signal_length=samples,
            midi_pitch=midi_note,
            note_duration_seconds=duration,
        ).numpy()
        # Clamp: a note starting at/after the buffer end is silence (matching a VST
        # host), not a negative-slice shape error; the loudness gate rejects it.
        offset = min(int(round(start * self.sample_rate)), samples)
        if offset:
            delayed = np.zeros_like(audio)
            delayed[:, offset:] = audio[:, : samples - offset]
            audio = delayed
        # Independent of the delay above: the mono voice fans out to the requested channels.
        if self.channels > 1:
            audio = np.repeat(audio, self.channels, axis=0)
        return _validate_rendered_audio(audio, channels=self.channels, samples=samples)


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

    block_size: int = 2048
    parameter_map: SynthParamMap = field(kw_only=True)
    reload_plugin_each_render: bool = True
    engine: _DawDreamerEngine = field(init=False, repr=False)
    plugin: _DawDreamerPlugin = field(init=False, repr=False)
    _parameter_indices: dict[str, int] = field(init=False, repr=False)
    _daw: _DawDreamerModule = field(init=False, repr=False)
    _has_rendered: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        """Create the DawDreamer engine and load the plugin graph."""
        self.plugin_path = str(Path(self.plugin_path).expanduser().resolve())
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
        """Create a fresh engine, plugin processor, graph, and parameter dispatch."""
        self.engine = self._daw.RenderEngine(self.sample_rate, self.block_size)
        self.plugin = self.engine.make_plugin_processor("synth", self.plugin_path)
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
        params: dict[str, float],
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
        """Load the configured preset into the current fresh plugin instance."""
        if self.plugin_state_path is None:
            return
        if self.plugin_state_path.endswith(".vstpreset"):
            self.plugin.load_vst3_preset(self.plugin_state_path)
        else:
            self.plugin.load_preset(self.plugin_state_path)
