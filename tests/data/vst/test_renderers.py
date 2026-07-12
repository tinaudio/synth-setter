from __future__ import annotations

import sys
import types
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from synth_setter.data.vst.generate_vst_dataset import generate_sample
from synth_setter.data.vst.param_spec import ContinuousParameter, NoteDurationParameter, ParamSpec
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


def test_generate_sample_uses_common_renderer_backend() -> None:
    """Dataset sample generation forwards note and params through an injected backend."""

    class Renderer:
        sample_rate = 44100
        channels = 2

        def render(self, params, midi_note, velocity, note_start_and_end, *, warmup=False):
            assert params == {"cutoff": 0.5}
            assert (midi_note, velocity, note_start_and_end, warmup) == (
                60,
                100,
                (0.0, 0.25),
                False,
            )
            return np.ones((2, 44100), dtype=np.float32)

    sample = generate_sample(
        renderer=Renderer(),  # type: ignore[arg-type]
        velocity=100,
        min_loudness=-100.0,
        param_spec=ParamSpec(
            [ContinuousParameter("cutoff", 0.0, 1.0)],
            [NoteDurationParameter("note_start_and_end", 1.0)],
        ),
        fixed_synth_params={"cutoff": 0.5},
        fixed_note_params={"pitch": 60, "note_start_and_end": (0.0, 0.25)},
    )

    assert sample.audio.shape == (44100, 2)


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


def test_dawdreamer_renderer_loads_graph_and_renders_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DawDreamer loads a graph, applies parameters, schedules MIDI, and returns audio.

    :param monkeypatch: Installs a fake DawDreamer module.
    """

    class FakeProcessor:
        """Minimal DawDreamer processor surface used by the backend contract test."""

        def __init__(self) -> None:
            self.parameters: dict[int, float] = {}
            self.midi: list[tuple[int, int, float, float]] = []
            self.midi_history: list[tuple[int, int, float, float]] = []

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
            return [
                {"index": 0, "name": "A Filter 1 Cutoff"},
                {"index": 1, "name": "A Osc 1 Shape"},
                {"index": 2, "name": "A Osc 1 Width 1"},
                {"index": 3, "name": "A Osc 1 Width 2"},
                {"index": 4, "name": "A Osc 1 Sub Mix"},
            ]

        def add_midi_note(self, pitch: int, velocity: int, start: float, duration: float) -> None:
            """Record one scheduled MIDI note.

            :param pitch: MIDI pitch.
            :param velocity: MIDI velocity.
            :param start: Note start time in seconds.
            :param duration: Note duration in seconds.
            """
            note = (pitch, velocity, start, duration)
            self.midi.append(note)
            self.midi_history.append(note)

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

    renderer = DawDreamerRenderer("Surge XT.vst3", 44100, 2, 1.0, "preset.vstpreset")
    result = renderer.render(
        {
            "a_filter_1_cutoff": 0.5,
            "a_osc_1_sawtooth": 0.1,
            "a_osc_1_pulse": 0.2,
            "a_osc_1_triangle": 0.3,
            "a_osc_1_width": 0.4,
        },
        60,
        100,
        (0.1, 0.35),
    )

    assert result.shape == (2, 16)
    assert renderer.plugin.parameters == {0: 0.5, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
    assert renderer.plugin.midi == []
    assert renderer.plugin.midi_history == [pytest.approx((60, 100, 0.1, 0.25))]
    assert renderer.plugin.preset == str(Path("preset.vstpreset").resolve())
    assert renderer.engine.rendered_duration == 1.0

    mono_renderer = DawDreamerRenderer("Surge XT.vst3", 44100, 1, 1.0)
    assert mono_renderer.render({"a_filter_1_cutoff": 0.5}, 60, 100, (0.1, 0.35)).shape == (
        1,
        16,
    )

    first_render_engine = renderer.engine
    renderer.render({"a_filter_1_cutoff": 0.75}, 61, 90, (0.0, 0.2))
    assert renderer.engine is not first_render_engine
    assert renderer.plugin.parameters == {0: 0.75}
    assert renderer.plugin.midi_history == [pytest.approx((61, 90, 0.0, 0.2))]
    assert renderer.plugin.midi == []


def test_dawdreamer_renderer_rejects_invalid_parameter_dispatch_before_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown keys and aliases targeting one host index fail before rendering.

    :param monkeypatch: Installs a fake DawDreamer module.
    """

    class FakeProcessor:
        def get_parameters_description(self) -> list[dict[str, object]]:
            return [{"index": 7, "name": "Cutoff"}]

        def clear_midi(self) -> None:
            pass

    class FakeEngine:
        def __init__(self, sample_rate: float, block_size: int) -> None:
            self.processor = FakeProcessor()
            self.render_calls = 0

        def make_plugin_processor(self, name: str, path: str) -> FakeProcessor:
            return self.processor

        def load_graph(self, graph: object) -> None:
            pass

        def render(self, duration: float) -> None:
            self.render_calls += 1

    monkeypatch.setitem(sys.modules, "dawdreamer", types.SimpleNamespace(RenderEngine=FakeEngine))
    renderer = DawDreamerRenderer("plugin.vst3", 44100, 2, 1.0)

    with pytest.raises(KeyError, match="unknown DawDreamer parameter.*missing"):
        renderer.render({"missing": 0.5}, 60, 100, (0.0, 0.2))
    assert renderer.engine.render_calls == 0


def test_dawdreamer_renderer_uses_explicit_surge_map_for_runtime_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Surge semantic keys map through CLAP names and positional FX slots.

    :param monkeypatch: Installs a fake DawDreamer module.
    """
    descriptions = [
        {"index": 40, "name": "A Osc 1 Shape"},
        {"index": 100, "name": "FX A1 FX Type"},
        {"index": 101, "name": "FX A1 -"},
        {"index": 102, "name": "FX A1 -"},
    ]

    class FakeProcessor:
        def get_parameters_description(self) -> list[dict[str, object]]:
            return descriptions

    class FakeEngine:
        def __init__(self, sample_rate: float, block_size: int) -> None:
            self.processor = FakeProcessor()

        def make_plugin_processor(self, name: str, path: str) -> FakeProcessor:
            return self.processor

        def load_graph(self, graph: object) -> None:
            pass

    monkeypatch.setitem(sys.modules, "dawdreamer", types.SimpleNamespace(RenderEngine=FakeEngine))

    renderer = DawDreamerRenderer("Surge XT.vst3", 44100, 2, 1.0)

    assert renderer._parameter_indices["a_osc_1_sawtooth"] == 40
    assert renderer._parameter_indices["fx_a1_delay_time"] == 101
    assert renderer._parameter_indices["fx_a1_modulation_rate"] == 102


def test_dawdreamer_renderer_rejects_noncontiguous_surge_fx_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Surge FX slots must occupy the verified indices following their type anchor.

    :param monkeypatch: Installs a fake DawDreamer module.
    """

    class FakeProcessor:
        def get_parameters_description(self) -> list[dict[str, object]]:
            return [
                {"index": 100, "name": "FX A1 FX Type"},
                {"index": 102, "name": "FX A1 -"},
            ]

    class FakeEngine:
        def __init__(self, sample_rate: float, block_size: int) -> None:
            self.processor = FakeProcessor()

        def make_plugin_processor(self, name: str, path: str) -> FakeProcessor:
            return self.processor

        def load_graph(self, graph: object) -> None:
            pass

    monkeypatch.setitem(sys.modules, "dawdreamer", types.SimpleNamespace(RenderEngine=FakeEngine))

    with pytest.raises(ValueError, match="FX A1 Param 1.*index 101"):
        DawDreamerRenderer("surge-xt.vst3", 44100, 2, 1.0)
