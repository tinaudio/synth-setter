from __future__ import annotations

import sys
import types
from dataclasses import fields
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from synth_setter.data.vst.generate_vst_dataset import generate_sample
from synth_setter.data.vst.param_map import (
    BackendSnapshot,
    DawDreamerParamRef,
    ParamIdentity,
    PedalboardParamRef,
    SynthParamMap,
)
from synth_setter.data.vst.param_spec import ContinuousParameter, NoteDurationParameter, ParamSpec
from synth_setter.data.vst.renderers import (
    AudioRenderer,
    DawDreamerRenderer,
    PedalboardRenderer,
)
from synth_setter.param_spec_name import ParamSpecName


def _test_param_map(params: dict[str, tuple[int, str]], count: int) -> SynthParamMap:
    """Build an explicit immutable map for a fake host.

    :param params: Repository keys mapped to fake index and display name pairs.
    :param count: Full fake host parameter count.
    :returns: A map whose backend snapshots share the requested fake host cardinality.
    """
    snapshot = BackendSnapshot(plugin_version="", parameter_count=count)
    return SynthParamMap(
        plugin="test",
        param_spec_name=ParamSpecName("test"),
        preset_resource="",
        preset_sha256="",
        pedalboard=snapshot,
        clap=snapshot,
        dawdreamer=snapshot,
        params={
            key: ParamIdentity(
                pedalboard=PedalboardParamRef(index=index, name=name),
                clap=None,
                dawdreamer=DawDreamerParamRef(index=index, name=name),
            )
            for key, (index, name) in params.items()
        },
    )


def test_audio_renderer_is_an_abstract_dataclass() -> None:
    """Abstract renderer construction is rejected while dataclass fields remain available."""
    assert hasattr(AudioRenderer, "__dataclass_fields__")
    assert fields(AudioRenderer)
    with pytest.raises(TypeError):
        AudioRenderer(  # pyright: ignore[reportAbstractUsage]
            plugin_path="plugin.vst3",
            sample_rate=44100,
            channels=2,
            signal_duration_seconds=1.0,
        )


def test_renderer_dataclasses_reject_positional_configuration() -> None:
    """Renderer configuration must be explicit at construction sites."""
    with pytest.raises(TypeError):
        PedalboardRenderer("plugin.vst3", 44100, 2, 1.0)  # type: ignore[misc]


def test_generate_sample_uses_common_renderer_backend() -> None:
    """Dataset sample generation forwards note and params through an injected backend."""

    class Renderer:
        """Minimal injected renderer for sample-generation coverage.

        .. attribute :: sample_rate

           Fake sample rate.

        .. attribute :: channels

           Fake output channel count.
        """

        sample_rate = 44100
        channels = 2

        def render(
            self,
            params: dict[str, float],
            midi_note: int,
            velocity: int,
            note_start_and_end: tuple[float, float],
            *,
            warmup: bool = False,
        ) -> np.ndarray:
            """Validate forwarded values and return audible fake audio.

            :param params: Synth parameter values.
            :param midi_note: MIDI note number.
            :param velocity: MIDI velocity.
            :param note_start_and_end: Note time bounds.
            :param warmup: Whether warm-up was requested.
            :returns: Non-silent stereo audio for the sample-generation assertion.
            """
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
    renderer = PedalboardRenderer(
        plugin_path="plugin.vst3",
        sample_rate=32,
        channels=2,
        signal_duration_seconds=1.0,
        plugin_state_path="preset.vstpreset",
    )

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
            32,
            2,
        ),
        "kwargs": {"plugin_state_path": "preset.vstpreset", "plugin": None, "warmup": False},
    }


@pytest.mark.parametrize(
    "audio",
    [
        np.array([[0.0, np.nan], [0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 1.01], [0.0, 0.0]], dtype=np.float32),
    ],
)
def test_pedalboard_renderer_rejects_invalid_audio(
    monkeypatch: pytest.MonkeyPatch,
    audio: np.ndarray,
) -> None:
    """The shared renderer contract rejects unsafe Pedalboard output.

    :param monkeypatch: Patches the Pedalboard render seam.
    :param audio: Invalid backend output under test.
    """
    monkeypatch.setattr("synth_setter.data.vst.core.render_params", lambda *args, **kwargs: audio)
    renderer = PedalboardRenderer(
        plugin_path="plugin.vst3",
        sample_rate=2,
        channels=2,
        signal_duration_seconds=1.0,
    )

    with pytest.raises(ValueError, match="rendered audio"):
        renderer.render({"cutoff": 0.5}, 60, 100, (0.0, 0.25))


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

            :returns: The fixed indexed host enumeration used for parameter dispatch.
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
            :returns: The engine-owned processor after recording its graph identity.
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

            :returns: Deterministic non-silent stereo audio for channel-shape assertions.
            """
            return np.ones((2, 16), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "dawdreamer", types.SimpleNamespace(RenderEngine=FakeEngine))

    parameter_map = _test_param_map(
        {
            "a_filter_1_cutoff": (0, "A Filter 1 Cutoff"),
            "a_osc_1_sawtooth": (1, "A Osc 1 Shape"),
            "a_osc_1_pulse": (2, "A Osc 1 Width 1"),
            "a_osc_1_triangle": (3, "A Osc 1 Width 2"),
            "a_osc_1_width": (4, "A Osc 1 Sub Mix"),
        },
        5,
    )
    renderer = DawDreamerRenderer(
        plugin_path="Surge XT.vst3",
        sample_rate=16,
        channels=2,
        signal_duration_seconds=1.0,
        plugin_state_path="preset.vstpreset",
        parameter_map=parameter_map,
    )
    initial_engine = renderer.engine
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
    plugin = cast(FakeProcessor, renderer.plugin)
    engine = cast(FakeEngine, renderer.engine)
    assert plugin.parameters == {0: 0.5, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
    assert plugin.midi == []
    assert plugin.midi_history == [pytest.approx((60, 100, 0.1, 0.25))]
    assert plugin.preset == str(Path("preset.vstpreset").resolve())
    assert engine.rendered_duration == 1.0
    assert renderer.engine is initial_engine

    mono_renderer = DawDreamerRenderer(
        plugin_path="Surge XT.vst3",
        sample_rate=16,
        channels=1,
        signal_duration_seconds=1.0,
        parameter_map=parameter_map,
    )
    assert mono_renderer.render({"a_filter_1_cutoff": 0.5}, 60, 100, (0.1, 0.35)).shape == (
        1,
        16,
    )

    first_render_engine = renderer.engine
    renderer.render({"a_filter_1_cutoff": 0.75}, 61, 90, (0.0, 0.2))
    assert renderer.engine is not first_render_engine
    reloaded_plugin = cast(FakeProcessor, renderer.plugin)
    assert reloaded_plugin.parameters == {0: 0.75}
    assert reloaded_plugin.midi_history == [pytest.approx((61, 90, 0.0, 0.2))]
    assert reloaded_plugin.midi == []


def test_dawdreamer_renderer_rejects_invalid_parameter_dispatch_before_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown keys and aliases targeting one host index fail before rendering.

    :param monkeypatch: Installs a fake DawDreamer module.
    """

    class FakeProcessor:
        """Expose one mapped parameter and MIDI cleanup."""

        def get_parameters_description(self) -> list[dict[str, object]]:
            """Return the fake host enumeration.

            :returns: The host enumeration containing the mapped Cutoff identity.
            """
            return [{"index": 7, "name": "Cutoff"}]

        def clear_midi(self) -> None:
            """Accept MIDI cleanup."""
            pass

    class FakeEngine:
        """Track whether audio rendering begins."""

        def __init__(self, sample_rate: float, block_size: int) -> None:
            """Create the fake processor.

            :param sample_rate: Render sample rate.
            :param block_size: Render block size.
            """
            self.processor = FakeProcessor()
            self.render_calls = 0

        def make_plugin_processor(self, name: str, path: str) -> FakeProcessor:
            """Return the fake processor.

            :param name: Graph processor name.
            :param path: Plugin path.
            :returns: The engine-owned processor exposing the only valid parameter identity.
            """
            return self.processor

        def load_graph(self, graph: object) -> None:
            """Accept a fake graph.

            :param graph: Graph definition.
            """
            pass

        def render(self, duration: float) -> None:
            """Record a render attempt.

            :param duration: Render duration.
            """
            self.render_calls += 1

    monkeypatch.setitem(sys.modules, "dawdreamer", types.SimpleNamespace(RenderEngine=FakeEngine))
    renderer = DawDreamerRenderer(
        plugin_path="plugin.vst3",
        sample_rate=44100,
        channels=2,
        signal_duration_seconds=1.0,
        parameter_map=_test_param_map({"cutoff": (7, "Cutoff")}, 1),
    )

    with pytest.raises(KeyError, match="unknown DawDreamer parameter.*missing"):
        renderer.render({"missing": 0.5}, 60, 100, (0.0, 0.2))
    assert cast(FakeEngine, renderer.engine).render_calls == 0


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
        """Expose the preset-specific fake enumeration."""

        def get_parameters_description(self) -> list[dict[str, object]]:
            """Return the fake host enumeration.

            :returns: The preset-specific enumeration with positional duplicate-name FX slots.
            """
            return descriptions

    class FakeEngine:
        """Provide the fake processor to the renderer."""

        def __init__(self, sample_rate: float, block_size: int) -> None:
            """Create the processor.

            :param sample_rate: Render sample rate.
            :param block_size: Render block size.
            """
            self.processor = FakeProcessor()

        def make_plugin_processor(self, name: str, path: str) -> FakeProcessor:
            """Return the fake processor.

            :param name: Graph processor name.
            :param path: Plugin path.
            :returns: The processor backed by that preset-specific enumeration.
            """
            return self.processor

        def load_graph(self, graph: object) -> None:
            """Accept a fake graph.

            :param graph: Graph definition.
            """
            pass

    monkeypatch.setitem(sys.modules, "dawdreamer", types.SimpleNamespace(RenderEngine=FakeEngine))

    renderer = DawDreamerRenderer(
        plugin_path="Surge XT.vst3",
        sample_rate=44100,
        channels=2,
        signal_duration_seconds=1.0,
        parameter_map=_test_param_map(
            {
                "a_osc_1_sawtooth": (40, "A Osc 1 Shape"),
                "fx_a1_delay_time": (101, "FX A1 -"),
                "fx_a1_modulation_rate": (102, "FX A1 -"),
            },
            4,
        ),
    )

    assert renderer._parameter_indices["a_osc_1_sawtooth"] == 40
    assert renderer._parameter_indices["fx_a1_delay_time"] == 101
    assert renderer._parameter_indices["fx_a1_modulation_rate"] == 102


def test_dawdreamer_renderer_rejects_stale_mapped_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stored index whose live name changed is rejected during initialization.

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

    with pytest.raises(ValueError, match="stale DawDreamer identity.*index 101"):
        DawDreamerRenderer(
            plugin_path="surge-xt.vst3",
            sample_rate=44100,
            channels=2,
            signal_duration_seconds=1.0,
            parameter_map=_test_param_map({"fx": (101, "FX A1 -")}, 2),
        )


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("count", "parameter count 1 != map 2"),
        ("version", "plugin version 'actual' != map 'expected'"),
        ("missing_preset", "requires the mapped preset"),
        ("preset_sha", "preset SHA-256 does not match"),
    ],
)
def test_dawdreamer_renderer_rejects_provenance_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    drift: str,
    message: str,
) -> None:
    """Live plugin provenance must match every committed snapshot field.

    :param monkeypatch: Installs a fake DawDreamer module and plugin version.
    :param tmp_path: Provides the mismatched preset fixture.
    :param drift: Provenance mismatch under test.
    :param message: Expected validation-error fragment.
    """

    class FakeProcessor:
        """Expose one stable parameter and accept preset loading."""

        def get_parameters_description(self) -> list[dict[str, object]]:
            """Return the one-parameter live enumeration.

            :returns: The live enumeration used for provenance validation.
            """
            return [{"index": 0, "name": "Cutoff"}]

        def load_vst3_preset(self, path: str) -> None:
            """Accept a VST3 preset path.

            :param path: Preset path.
            """

    class FakeEngine:
        """Provide one fake processor graph."""

        def __init__(self, sample_rate: float, block_size: int) -> None:
            """Create the fake processor.

            :param sample_rate: Render sample rate.
            :param block_size: Render block size.
            """
            self.processor = FakeProcessor()

        def make_plugin_processor(self, name: str, path: str) -> FakeProcessor:
            """Return the engine-owned processor.

            :param name: Graph processor name.
            :param path: Plugin path.
            :returns: The processor used for provenance validation.
            """
            return self.processor

        def load_graph(self, graph: object) -> None:
            """Accept the processor graph.

            :param graph: Graph definition.
            """

    monkeypatch.setitem(sys.modules, "dawdreamer", types.SimpleNamespace(RenderEngine=FakeEngine))
    monkeypatch.setattr("synth_setter.data.vst.core.extract_renderer_version", lambda path: "actual")
    parameter_map = _test_param_map({"cutoff": (0, "Cutoff")}, 1)
    preset_path: str | None = None
    if drift == "count":
        parameter_map = parameter_map.model_copy(
            update={"dawdreamer": BackendSnapshot(plugin_version="", parameter_count=2)}
        )
    elif drift == "version":
        parameter_map = parameter_map.model_copy(
            update={"dawdreamer": BackendSnapshot(plugin_version="expected", parameter_count=1)}
        )
    else:
        parameter_map = parameter_map.model_copy(update={"preset_sha256": "0" * 64})
        if drift == "preset_sha":
            preset = tmp_path / "preset.vstpreset"
            preset.write_bytes(b"different preset")
            preset_path = str(preset)

    with pytest.raises(ValueError, match=message):
        DawDreamerRenderer(
            plugin_path="plugin.vst3",
            sample_rate=44100,
            channels=2,
            signal_duration_seconds=1.0,
            plugin_state_path=preset_path,
            parameter_map=parameter_map,
        )


def test_dawdreamer_renderer_once_cadence_reuses_loaded_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two renders reuse one graph when plugin reload cadence is once.

    :param monkeypatch: Installs a stateful fake DawDreamer module.
    """

    class FakeProcessor:
        """Record MIDI scheduled across repeated renders."""

        def __init__(self) -> None:
            self.notes: list[tuple[int, int, float, float]] = []

        def get_parameters_description(self) -> list[dict[str, object]]:
            """Expose one mapped parameter.

            :returns: The fixed indexed host enumeration used for parameter dispatch.
            """
            return [{"index": 0, "name": "Cutoff"}]

        def clear_midi(self) -> None:
            """Accept MIDI cleanup between renders."""

        def set_parameter(self, index: int, value: float) -> None:
            """Accept a normalized parameter assignment.

            :param index: Host parameter index.
            :param value: Normalized parameter value.
            """

        def add_midi_note(self, pitch: int, velocity: int, start: float, duration: float) -> None:
            """Record one scheduled MIDI note.

            :param pitch: MIDI pitch.
            :param velocity: MIDI velocity.
            :param start: Note start time in seconds.
            :param duration: Note duration in seconds.
            """
            self.notes.append((pitch, velocity, start, duration))

    class FakeEngine:
        """Expose deterministic audio from one plugin graph.

        .. attribute :: created

           Number of fake graphs initialized by the renderer.
        """

        created = 0

        def __init__(self, sample_rate: float, block_size: int) -> None:
            """Create one fake plugin graph.

            :param sample_rate: Render sample rate.
            :param block_size: Render block size.
            """
            FakeEngine.created += 1
            self.processor = FakeProcessor()

        def make_plugin_processor(self, name: str, path: str) -> FakeProcessor:
            """Return the engine-owned processor.

            :param name: Graph processor name.
            :param path: Plugin path.
            :returns: The engine-owned processor after recording its graph identity.
            """
            return self.processor

        def load_graph(self, graph: object) -> None:
            """Accept a fake graph.

            :param graph: Graph definition.
            """

        def render(self, duration: float) -> None:
            """Accept a render request.

            :param duration: Render duration in seconds.
            """

        def get_audio(self) -> np.ndarray:
            """Return valid stereo audio.

            :returns: Deterministic non-silent stereo audio for channel-shape assertions.
            """
            return np.full((2, 4), 0.25, dtype=np.float32)

    monkeypatch.setitem(sys.modules, "dawdreamer", types.SimpleNamespace(RenderEngine=FakeEngine))
    renderer = DawDreamerRenderer(
        plugin_path="plugin.vst3",
        sample_rate=4,
        channels=2,
        signal_duration_seconds=1.0,
        parameter_map=_test_param_map({"cutoff": (0, "Cutoff")}, 1),
        reload_plugin_each_render=False,
    )
    engine = renderer.engine
    plugin = cast(FakeProcessor, renderer.plugin)

    renderer.render({"cutoff": 0.25}, 60, 100, (0.0, 0.25))
    renderer.render({"cutoff": 0.75}, 61, 90, (0.0, 0.5))

    assert FakeEngine.created == 1
    assert renderer.engine is engine
    assert renderer.plugin is plugin
    assert plugin.notes == [
        (60, 100, 0.0, 0.25),
        (61, 90, 0.0, 0.5),
    ]


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        (np.zeros((2, 3), dtype=np.float32), "sample count"),
        (np.array([[0.0, np.nan], [0.0, 0.0]], dtype=np.float32), "finite"),
        (np.array([[0.0, np.inf], [0.0, 0.0]], dtype=np.float32), "finite"),
        (np.array([[0.0, 1.0001], [0.0, 0.0]], dtype=np.float32), r"\[-1, 1\]"),
    ],
)
def test_dawdreamer_renderer_rejects_invalid_audio(
    monkeypatch: pytest.MonkeyPatch,
    audio: np.ndarray,
    message: str,
) -> None:
    """Malformed or unsafe backend audio fails before reaching dataset writers.

    :param monkeypatch: Installs a fake DawDreamer module.
    :param audio: Invalid backend output under test.
    :param message: Expected validation-error fragment.
    """

    class FakeProcessor:
        """Expose the minimum processor render surface."""

        def get_parameters_description(self) -> list[dict[str, object]]:
            """Expose one mapped parameter.

            :returns: The fixed indexed host enumeration used for parameter dispatch.
            """
            return [{"index": 0, "name": "Cutoff"}]

        def clear_midi(self) -> None:
            """Accept MIDI cleanup."""

        def set_parameter(self, index: int, value: float) -> None:
            """Accept one parameter assignment.

            :param index: Host parameter index.
            :param value: Normalized parameter value.
            """

        def add_midi_note(self, pitch: int, velocity: int, start: float, duration: float) -> None:
            """Accept one MIDI note.

            :param pitch: MIDI pitch.
            :param velocity: MIDI velocity.
            :param start: Note start time in seconds.
            :param duration: Note duration in seconds.
            """

    class FakeEngine:
        """Return the parametrized invalid output."""

        def __init__(self, sample_rate: float, block_size: int) -> None:
            """Create the fake processor.

            :param sample_rate: Render sample rate.
            :param block_size: Render block size.
            """
            self.processor = FakeProcessor()

        def make_plugin_processor(self, name: str, path: str) -> FakeProcessor:
            """Return the engine-owned processor.

            :param name: Graph processor name.
            :param path: Plugin path.
            :returns: The engine-owned processor after recording its graph identity.
            """
            return self.processor

        def load_graph(self, graph: object) -> None:
            """Accept a fake graph.

            :param graph: Graph definition.
            """

        def render(self, duration: float) -> None:
            """Accept a render request.

            :param duration: Render duration in seconds.
            """

        def get_audio(self) -> np.ndarray:
            """Return the configured invalid audio.

            :returns: Backend audio for production validation.
            """
            return audio

    monkeypatch.setitem(sys.modules, "dawdreamer", types.SimpleNamespace(RenderEngine=FakeEngine))
    renderer = DawDreamerRenderer(
        plugin_path="plugin.vst3",
        sample_rate=2,
        channels=2,
        signal_duration_seconds=1.0,
        parameter_map=_test_param_map({"cutoff": (0, "Cutoff")}, 1),
    )

    with pytest.raises(ValueError, match=message):
        renderer.render({"cutoff": 0.5}, 60, 100, (0.0, 0.25))


def test_dawdreamer_renderer_accepts_full_scale_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact normalized full scale remains valid.

    :param monkeypatch: Installs a fake DawDreamer module.
    """

    class FakeProcessor:
        def get_parameters_description(self) -> list[dict[str, object]]:
            return [{"index": 0, "name": "Cutoff"}]

        def clear_midi(self) -> None:
            pass

        def set_parameter(self, index: int, value: float) -> None:
            pass

        def add_midi_note(self, pitch: int, velocity: int, start: float, duration: float) -> None:
            pass

    class FakeEngine:
        def __init__(self, sample_rate: float, block_size: int) -> None:
            self.processor = FakeProcessor()

        def make_plugin_processor(self, name: str, path: str) -> FakeProcessor:
            return self.processor

        def load_graph(self, graph: object) -> None:
            pass

        def render(self, duration: float) -> None:
            pass

        def get_audio(self) -> np.ndarray:
            return np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float32)

    monkeypatch.setitem(sys.modules, "dawdreamer", types.SimpleNamespace(RenderEngine=FakeEngine))
    renderer = DawDreamerRenderer(
        plugin_path="plugin.vst3",
        sample_rate=2,
        channels=2,
        signal_duration_seconds=1.0,
        parameter_map=_test_param_map({"cutoff": (0, "Cutoff")}, 1),
    )

    audio = renderer.render({"cutoff": 0.5}, 60, 100, (0.0, 0.25))

    assert np.array_equal(audio, np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float32))
