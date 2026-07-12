"""Common audio-rendering interface and VST backend implementations."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from synth_setter.data.vst.param_map import SynthParamMap


@dataclass
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


@dataclass
class PedalboardRenderer(AudioRenderer):
    """Render through the existing pedalboard implementation.

    .. attribute :: plugin

       Optional preloaded pedalboard plugin instance.
    """

    plugin: Any = field(default=None, repr=False)

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

        return render_params(
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
        )


@dataclass
class DawDreamerRenderer(AudioRenderer):
    """Render through DawDreamer's JUCE-backed VST host.

    .. attribute :: block_size

       DawDreamer engine block size.

    .. attribute :: parameter_map

       Validated immutable cross-host identity map.

    .. attribute :: engine

       DawDreamer render engine instance.

    .. attribute :: plugin

       DawDreamer plugin processor instance.
    """

    block_size: int = 2048
    parameter_map: SynthParamMap = field(kw_only=True)
    engine: Any = field(init=False, repr=False)
    plugin: Any = field(init=False, repr=False)
    _parameter_indices: dict[str, int] = field(init=False, repr=False)
    _daw: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Create the DawDreamer engine and load the plugin graph."""
        self.plugin_path = str(Path(self.plugin_path).expanduser().resolve())
        if self.plugin_state_path is not None:
            self.plugin_state_path = str(Path(self.plugin_state_path).expanduser().resolve())
        self._daw = import_module("dawdreamer")
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
        self._create_graph()
        self._load_preset()
        self._validate_parameter_dispatch(params)
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
        return self._match_channels(audio)

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
