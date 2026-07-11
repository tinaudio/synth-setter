from __future__ import annotations

import sys
import types
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from synth_setter.data.vst.renderers import (
    AudioRenderer,
    DawDreamerRenderer,
    PedalboardRenderer,
)


def test_audio_renderer_is_an_abstract_dataclass() -> None:
    """Abstract renderer construction is rejected while dataclass fields remain available."""
    assert hasattr(AudioRenderer, "__dataclass_fields__")
    assert fields(AudioRenderer)
    with pytest.raises(TypeError):
        AudioRenderer("plugin.vst3", 44100, 2, 1.0)  # pyright: ignore[reportAbstractUsage]


def test_pedalboard_renderer_uses_common_render_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pedalboard implements the shared renderer call and return-value contract.

    :param monkeypatch: Patches the existing pedalboard render seam.
    """
    expected = np.ones((2, 32), dtype=np.float32)
    seen: dict[str, object] = {}

    def fake_render_params(*args: object, **kwargs: object) -> np.ndarray:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return expected

    monkeypatch.setattr("synth_setter.data.vst.core.render_params", fake_render_params)
    renderer = PedalboardRenderer("plugin.vst3", 44100, 2, 1.0, "preset.vstpreset")

    result = renderer.render({"cutoff": 0.5}, 60, 100, (0.0, 0.25))

    assert result is expected
    assert seen == {
        "args": (
            "plugin.vst3",
            {"cutoff": 0.5},
            60,
            100,
            (0.0, 0.25),
            1.0,
            44100,
            2,
        ),
        "kwargs": {"preset_path": "preset.vstpreset", "plugin": None, "warmup": False},
    }


def test_dawdreamer_renderer_loads_graph_and_renders_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """DawDreamer loads a graph, applies parameters, schedules MIDI, and returns audio.

    :param monkeypatch: Installs a fake DawDreamer module.
    """
    class FakeProcessor:
        """Minimal DawDreamer processor surface used by the backend contract test."""

        def __init__(self) -> None:
            self.parameters: dict[int, float] = {}
            self.midi: list[tuple[int, int, float, float]] = []

        def set_parameter(self, name: int, value: float) -> None:
            """Record a parameter assignment.

            :param name: Host parameter index.
            :param value: Normalized parameter value.
            """
            self.parameters[name] = value

        def get_parameters_description(self) -> list[dict[str, object]]:
            """Return one named parameter with its host index.

            :returns: One fake parameter description.
            """
            return [{"index": 0, "name": "cutoff"}]

        def add_midi_note(self, pitch: int, velocity: int, start: float, duration: float) -> None:
            """Record one scheduled MIDI note.

            :param pitch: MIDI pitch.
            :param velocity: MIDI velocity.
            :param start: Note start time in seconds.
            :param duration: Note duration in seconds.
            """
            self.midi.append((pitch, velocity, start, duration))

        def clear_midi(self) -> None:
            """Clear scheduled MIDI notes."""
            self.midi.clear()

        def load_vst3_preset(self, path: str) -> None:
            """Record the loaded VST3 preset path.

            :param path: Preset path.
            """
            self.preset = path

    class FakeEngine:
        """Minimal DawDreamer engine surface used by the backend contract test."""

        def __init__(self, sample_rate: float, block_size: int) -> None:
            """Create an empty fake engine.

            :param sample_rate: Render sample rate.
            :param block_size: Audio block size.
            """
            self.sample_rate = sample_rate
            self.block_size = block_size
            self.processor = FakeProcessor()
            self.rendered_duration = 0.0

        def make_plugin_processor(self, name: str, path: str) -> FakeProcessor:
            """Create and return the fake plugin processor.

            :param name: Processor name.
            :param path: Plugin path.
            :returns: The fake plugin processor.
            """
            self.name = name
            self.path = path
            return self.processor

        def load_graph(self, graph: object) -> None:
            """Record the graph supplied by the renderer.

            :param graph: DawDreamer graph definition.
            """
            self.graph = graph

        def render(self, duration: float) -> None:
            """Record the requested render duration.

            :param duration: Duration in seconds.
            """
            self.rendered_duration = duration

        def get_audio(self) -> np.ndarray:
            """Return deterministic stereo test audio.

            :returns: A fixed stereo audio buffer.
            """
            return np.ones((2, 16), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "dawdreamer", types.SimpleNamespace(RenderEngine=FakeEngine))

    renderer = DawDreamerRenderer("plugin.vst3", 44100, 2, 1.0, "preset.vstpreset")
    result = renderer.render({"cutoff": 0.5}, 60, 100, (0.1, 0.35))

    assert result.shape == (2, 16)
    assert renderer.plugin.parameters == {0: 0.5}
    assert renderer.plugin.midi == []
    assert renderer.plugin.preset == str(Path("preset.vstpreset").resolve())
    assert renderer.engine.rendered_duration == 1.0

    mono_renderer = DawDreamerRenderer("plugin.vst3", 44100, 1, 1.0)
    assert mono_renderer.render({"cutoff": 0.5}, 60, 100, (0.1, 0.35)).shape == (1, 16)
