"""Common audio-rendering interface and VST backend implementations."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np

from synth_setter.data.vst.clap_map import load_clap_map
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
    _daw: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Create the DawDreamer engine and load the plugin graph."""
        self.plugin_path = str(Path(self.plugin_path).expanduser().resolve())
        if self.preset_path is not None:
            self.preset_path = str(Path(self.preset_path).expanduser().resolve())
        self._daw = import_module("dawdreamer")
        self._create_graph()

    def _create_graph(self) -> None:
        """Create a fresh engine, plugin processor, graph, and parameter dispatch."""
        self.engine = self._daw.RenderEngine(self.sample_rate, self.block_size)
        self.plugin = self.engine.make_plugin_processor("synth", self.plugin_path)
        self.engine.load_graph([(self.plugin, [])])
        self._build_parameter_indices()

    def _build_parameter_indices(self) -> None:
        """Build a collision-free dispatch from the current plugin descriptions.

        :raises ValueError: If normalized host keys collide.
        """
        descriptions = self.plugin.get_parameters_description()
        if "surge" in Path(self.plugin_path).name.lower():
            self._parameter_indices = self._build_surge_parameter_indices(descriptions)
            return
        self._parameter_indices = {}
        for description in descriptions:
            key = dawdreamer_parameter_key(str(description["name"]))
            index = cast(int, description["index"])
            if key in self._parameter_indices:
                raise ValueError(f"duplicate DawDreamer parameter key {key!r}")
            self._parameter_indices[key] = index

    def _build_surge_parameter_indices(
        self, descriptions: list[dict[str, object]]
    ) -> dict[str, int]:
        """Map Surge semantic keys via its committed CLAP display-name contract.

        :param descriptions: DawDreamer parameter descriptions for the loaded Surge plugin.
        :returns: Repository parameter keys mapped to distinct DawDreamer host indices.
        :raises ValueError: If indices, display names, or FX slot positions are ambiguous.
        """
        by_name: dict[str, list[int]] = {}
        by_index: dict[int, str] = {}
        for description in descriptions:
            name = str(description["name"])
            index = cast(int, description["index"])
            by_name.setdefault(name, []).append(index)
            if index in by_index:
                raise ValueError(f"duplicate DawDreamer parameter index {index}")
            by_index[index] = name

        format_map = load_clap_map(Path(__file__).with_name("surge_xt_clap_map.json"))
        parameter_indices: dict[str, int] = {}
        for key, ref in format_map.params.items():
            match = re.fullmatch(r"FX (A[1-4]) Param (\d+)", ref.clap_name)
            if match:
                bank, slot_text = match.groups()
                anchors = by_name.get(f"FX {bank} FX Type", [])
                if len(anchors) != 1:
                    continue
                target = anchors[0] + int(slot_text)
                name = by_index.get(target)
                if name is None:
                    later_slot_exists = any(
                        index > target and candidate.startswith(f"FX {bank}")
                        for index, candidate in by_index.items()
                    )
                    if later_slot_exists:
                        raise ValueError(
                            f"{ref.clap_name} expected DawDreamer parameter index {target}"
                        )
                    continue
                if not name.startswith(f"FX {bank}"):
                    raise ValueError(
                        f"{ref.clap_name} expected DawDreamer parameter index {target}, "
                        f"found {name!r}"
                    )
                parameter_indices[key] = target
                continue

            indices = by_name.get(ref.clap_name, [])
            if len(indices) > 1:
                raise ValueError(f"ambiguous Surge parameter display name {ref.clap_name!r}")
            if indices:
                parameter_indices[key] = indices[0]

        duplicates: dict[int, list[str]] = {}
        for key, index in parameter_indices.items():
            duplicates.setdefault(index, []).append(key)
        collisions = {index: keys for index, keys in duplicates.items() if len(keys) > 1}
        if collisions:
            raise ValueError(f"Surge parameter map has index collisions: {collisions}")
        return parameter_indices

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
        self._build_parameter_indices()
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
        if self.preset_path is None:
            return
        if self.preset_path.endswith(".vstpreset"):
            self.plugin.load_vst3_preset(self.preset_path)
        else:
            self.plugin.load_preset(self.preset_path)
