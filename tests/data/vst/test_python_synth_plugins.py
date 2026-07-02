"""Behavior tests for the Python-synth backends hosted behind the VST plugin surface.

``torchsynth`` / ``synthax`` are dispatched from ``core.load_plugin`` by bare
plugin-path name and must duck-type the ``pedalboard.VST3Plugin`` render
surface (the same contract ``FakeVST3Plugin`` pins), so every downstream
consumer — ``render_params``, the shard writers, cadence machinery — works
unchanged.
"""

import threading
from importlib.metadata import version as dist_version
from pathlib import Path

import numpy as np
import pytest

from synth_setter.data.vst.core import extract_renderer_version, load_plugin, make_midi_events
from synth_setter.data.vst.python_synth import (
    PythonSynthPlugin,
    SynthaxPlugin,
    TorchSynthPlugin,
    is_python_synth,
    load_python_synth,
)
from tests.data.vst._fake_plugin import FakeVST3Plugin

_SAMPLE_RATE = 44100.0
_CHANNELS = 2
_BLOCK_SIZE = 2048
_DURATION_S = 1.0
_NUM_SAMPLES = int(_DURATION_S * _SAMPLE_RATE)


def _note_events(
    pitch: int = 60, velocity: int = 100, start: float = 0.0, end: float = 0.5
) -> tuple:
    """Build note-on/off MIDI events via the production helper.

    :param pitch: MIDI pitch of the note.
    :param velocity: MIDI velocity of the note.
    :param start: Note-on time in seconds.
    :param end: Note-off time in seconds.
    :returns: The ``core.make_midi_events`` event tuple.
    """
    return make_midi_events(pitch, velocity, start, end)


class TestLoadPluginDispatch:
    """``core.load_plugin`` routes Python synth names without disturbing VST3 paths."""

    def test_load_plugin_torchsynth_name_returns_torchsynth_plugin(self) -> None:
        """Bare name \"torchsynth\" dispatches to the torchsynth adapter."""
        assert isinstance(load_plugin("torchsynth"), TorchSynthPlugin)

    def test_load_plugin_synthax_name_returns_synthax_plugin(self) -> None:
        """Bare name \"synthax\" dispatches to the synthax adapter."""
        assert isinstance(load_plugin("synthax"), SynthaxPlugin)

    def test_load_plugin_vst3_path_still_uses_pedalboard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``.vst3`` path still constructs the pedalboard plugin class.

        :param monkeypatch: Swaps ``core.VST3Plugin`` for the no-X11 fake.
        """
        monkeypatch.setattr("synth_setter.data.vst.core.VST3Plugin", FakeVST3Plugin)
        assert isinstance(load_plugin("plugins/Anything.vst3"), FakeVST3Plugin)

    def test_is_python_synth_vst3_path_false(self) -> None:
        """A filesystem plugin path is not classified as a Python synth."""
        assert not is_python_synth("plugins/Surge XT.vst3")


class TestExtractRendererVersion:
    """``extract_renderer_version`` handles Python synth names and real paths."""

    def test_extract_renderer_version_torchsynth_returns_package_version(self) -> None:
        """Torchsynth reports the installed distribution version."""
        assert extract_renderer_version(Path("torchsynth")) == dist_version("torchsynth")

    def test_extract_renderer_version_synthax_returns_package_version(self) -> None:
        """Synthax reports the installed distribution version."""
        assert extract_renderer_version(Path("synthax")) == dist_version("synthax")

    def test_extract_renderer_version_missing_path_still_raises(self) -> None:
        """A nonexistent ``.vst3`` path still raises ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError):
            extract_renderer_version(Path("plugins/DoesNotExist.vst3"))


@pytest.fixture(
    scope="module",
    params=["torchsynth", pytest.param("synthax", marks=pytest.mark.slow)],
)
def plugin(request: pytest.FixtureRequest) -> PythonSynthPlugin:
    """Provide one adapter per backend, shared across the surface tests.

    :param request: Parametrized with each Python synth backend name.
    :returns: A fresh plugin adapter for the backend under test.
    """
    return load_python_synth(request.param)


class TestPluginSurface:
    """Both backends honour the pedalboard render-surface contract."""

    def test_version_matches_installed_distribution(self, plugin: PythonSynthPlugin) -> None:
        """``version`` equals the installed distribution version.

        :param plugin: The backend adapter under test.
        """
        assert plugin.version == dist_version(plugin._dist_name)  # noqa: SLF001

    def test_parameters_expose_raw_values_in_unit_interval(
        self, plugin: PythonSynthPlugin
    ) -> None:
        """Every exposed parameter carries a ``raw_value`` in [0, 1].

        :param plugin: The backend adapter under test.
        """
        params = plugin.parameters
        assert len(params) > 0
        assert all(0.0 <= p.raw_value <= 1.0 for p in params.values())

    def test_parameters_exclude_keyboard_note_controls(self, plugin: PythonSynthPlugin) -> None:
        """Keyboard pitch/duration are MIDI-driven, not exposed parameters.

        :param plugin: The backend adapter under test.
        """
        assert not any(name.startswith("keyboard.") for name in plugin.parameters)

    def test_process_empty_midi_returns_zero_length_flush_buffer(
        self, plugin: PythonSynthPlugin
    ) -> None:
        """Empty MIDI events (pipeline flush calls) yield a zero-length buffer.

        :param plugin: The backend adapter under test.
        """
        out = plugin.process([], 32.0, _SAMPLE_RATE, _CHANNELS, _BLOCK_SIZE, True)
        assert out.shape == (_CHANNELS, 0)

    def test_process_note_on_returns_channels_by_duration_samples(
        self, plugin: PythonSynthPlugin
    ) -> None:
        """A note render matches pedalboard's output shape/dtype and is nonsilent.

        :param plugin: The backend adapter under test.
        """
        out = plugin.process(
            _note_events(), _DURATION_S, _SAMPLE_RATE, _CHANNELS, _BLOCK_SIZE, True
        )
        assert out.shape == (_CHANNELS, _NUM_SAMPLES)
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))
        assert float(np.abs(out).max()) > 0.0

    def test_process_same_inputs_twice_identical_output(self, plugin: PythonSynthPlugin) -> None:
        """Identical params and note produce byte-identical audio.

        :param plugin: The backend adapter under test.
        """
        args = (_note_events(), _DURATION_S, _SAMPLE_RATE, _CHANNELS, _BLOCK_SIZE, True)
        assert np.array_equal(plugin.process(*args), plugin.process(*args))

    def test_process_note_start_delays_onset(self, plugin: PythonSynthPlugin) -> None:
        """Audio before the note-on time is silent; the note still sounds.

        :param plugin: The backend adapter under test.
        """
        start = 0.25
        out = plugin.process(
            _note_events(start=start, end=0.75),
            _DURATION_S,
            _SAMPLE_RATE,
            _CHANNELS,
            _BLOCK_SIZE,
            True,
        )
        head = out[:, : int(start * _SAMPLE_RATE) - 1]
        assert float(np.abs(head).max()) == 0.0
        assert float(np.abs(out).max()) > 0.0

    def test_process_changing_param_raw_value_changes_audio(
        self, plugin: PythonSynthPlugin
    ) -> None:
        """Writing ``raw_value`` changes the rendered audio.

        :param plugin: The backend adapter under test.
        """
        args = (_note_events(), _DURATION_S, _SAMPLE_RATE, _CHANNELS, _BLOCK_SIZE, True)
        saved = {name: param.raw_value for name, param in plugin.parameters.items()}
        try:
            for param in plugin.parameters.values():
                param.raw_value = 0.25
            low = plugin.process(*args)
            for param in plugin.parameters.values():
                param.raw_value = 0.75
            high = plugin.process(*args)
        finally:
            for name, param in plugin.parameters.items():
                param.raw_value = saved[name]
        assert not np.array_equal(low, high)

    def test_load_preset_and_reset_are_noops(self, plugin: PythonSynthPlugin) -> None:
        """``load_preset`` and ``reset`` accept calls without effect or error.

        :param plugin: The backend adapter under test.
        """
        plugin.load_preset("")
        plugin.reset()

    def test_show_editor_returns_once_close_event_set(self, plugin: PythonSynthPlugin) -> None:
        """``show_editor`` honours the close-event contract without blocking.

        :param plugin: The backend adapter under test.
        """
        close = threading.Event()
        close.set()
        plugin.show_editor(close)
