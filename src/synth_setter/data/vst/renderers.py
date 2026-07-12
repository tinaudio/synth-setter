"""Common audio-rendering interface and VST backend implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from synth_setter.data.vst.dawdreamer_map import dawdreamer_parameter_key


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

    .. attribute :: preset_path

       Optional baseline preset path.
    """

    plugin_path: str
    sample_rate: float
    channels: int
    signal_duration_seconds: float
    preset_path: str | None = None

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
            preset_path=self.preset_path,
            plugin=self.plugin,
            warmup=warmup,
        )


@dataclass
class DawDreamerRenderer(AudioRenderer):
    """Render through DawDreamer's JUCE-backed VST host.

    .. attribute :: block_size

       DawDreamer engine block size.

    .. attribute :: engine

       DawDreamer render engine instance.

    .. attribute :: plugin

       DawDreamer plugin processor instance.
    """

    block_size: int = 2048
    engine: Any = field(init=False, repr=False)
    plugin: Any = field(init=False, repr=False)
    _parameter_indices: dict[str, int] = field(init=False, repr=False)
    _preset_loaded: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Create the DawDreamer engine and load the plugin graph."""
        self.plugin_path = str(Path(self.plugin_path).expanduser().resolve())
        if self.preset_path is not None:
            self.preset_path = str(Path(self.preset_path).expanduser().resolve())
        daw: Any = import_module("dawdreamer")
        self.engine = daw.RenderEngine(self.sample_rate, self.block_size)
        self.plugin = self.engine.make_plugin_processor("synth", self.plugin_path)
        self.engine.load_graph([(self.plugin, [])])
        self._parameter_indices = {}
        for description in self.plugin.get_parameters_description():
            key = dawdreamer_parameter_key(description["name"])
            index = description["index"]
            self._parameter_indices[key] = index
            if key.endswith("_shape"):
                for waveform in ("sawtooth", "pulse", "triangle"):
                    self._parameter_indices[key.removesuffix("shape") + waveform] = index
            if key.endswith("_width_1"):
                self._parameter_indices[key.removesuffix("_1")] = index

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
        self._load_preset_once()
        self.plugin.clear_midi()
        try:
            for name, value in params.items():
                index = self._parameter_indices.get(name)
                if index is not None:
                    self.plugin.set_parameter(index, value)
            start, end = note_start_and_end
            self.plugin.add_midi_note(midi_note, velocity, start, end - start)
            self.engine.render(self.signal_duration_seconds)
            audio = np.asarray(self.engine.get_audio())
        finally:
            self.plugin.clear_midi()
        return self._match_channels(audio)

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

    def _load_preset_once(self) -> None:
        """Load the configured preset once before the first render."""
        if self._preset_loaded or self.preset_path is None:
            return
        if self.preset_path.endswith(".vstpreset"):
            self.plugin.load_vst3_preset(self.preset_path)
        else:
            self.plugin.load_preset(self.preset_path)
        self._preset_loaded = True
