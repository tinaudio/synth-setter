"""Unit tests for src/data/vst/core.py.

Adheres to professional testing standards:
- Tests public behavior, not implementation details.
- Uses Fakes for domain objects to test state changes.
- Uses dependency injection for side-effects (time, threading, I/O).
"""

import threading
from types import SimpleNamespace
from typing import Dict, List, Tuple

import mido
import numpy as np
import pytest

from src.data.vst.core import (
    _call_with_interrupt,
    load_plugin,
    make_midi_events,
    render_params,
    set_params,
    write_wav,
)

# ---------------------------------------------------------------------------
# Fakes (Behavioral Simulators)
# ---------------------------------------------------------------------------


class FakeParameter:
    """Mock parameter for VST3Plugin."""

    def __init__(self, value=0.0):
        self.raw_value = value


class FakeVSTPlugin:
    """A realistic Fake for VST3Plugin.

    Stores logical state (parameters, current preset) and tracks history of 'heavy' operations like
    processing.
    """

    def __init__(self, path: str = "fake.vst3"):
        self.path = path
        self.parameters: Dict[str, FakeParameter] = {
            "osc1_pitch": FakeParameter(0.0),
            "filter_cutoff": FakeParameter(0.0),
        }
        self.preset_loaded: str = None
        self.process_history: List[List[bytes]] = []
        self.editor_shown = False
        self.reset_count = 0

    def load_preset(self, path: str):
        """Simulate loading a preset."""
        self.preset_loaded = path

    def show_editor(self):
        """Simulate opening the plugin editor."""
        self.editor_shown = True
        # Simulate blocking behavior that needs external interrupt
        # In a real test we just mark the flag, we don't actually block here
        # unless we were testing the threading harness specifically.
        pass

    def process(self, midi_events, duration, sample_rate, channels, *args):
        """Simulate audio processing and MIDI event handling."""
        # Log the midi events received during this block
        self.process_history.append(midi_events)
        # Return silence of appropriate shape
        n_samples = int(duration * sample_rate) if duration > 0.0 else 2048
        return np.zeros((channels, n_samples))

    def reset(self):
        """Increment the reset counter."""
        self.reset_count += 1


class FakeAudioFile:
    """Minimal fake for AudioFile."""

    def __init__(self, path, mode, sample_rate, num_channels):
        """Initialize with file path and settings."""
        self.path = path
        self.sample_rate = sample_rate
        self.written_data = None

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, *args):
        """Exit context manager."""
        pass

    def write(self, data):
        """Simulate writing audio data."""
        self.written_data = data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_plugin():
    """Create a fresh FakeVSTPlugin instance."""
    return FakeVSTPlugin()


# ---------------------------------------------------------------------------
# Core Logic Tests (Pure Functions)
# ---------------------------------------------------------------------------


def test_make_midi_events_structure():
    """Verify strictly the data structure required by pedalboard."""
    events = make_midi_events(pitch=60, velocity=100, note_start=0.5, note_end=2.0)

    assert len(events) == 2
    (msg1, time1), (msg2, time2) = events

    # Check Times
    assert time1 == 0.5
    assert time2 == 2.0

    # Check Messages
    # msg1 is a list of integers (bytes) from .bytes() in the source code
    m1 = mido.Message.from_bytes(msg1)
    assert m1.type == "note_on"
    assert m1.note == 60
    assert m1.velocity == 100

    m2 = mido.Message.from_bytes(msg2)
    assert m2.type == "note_off"
    assert m2.note == 60
    # Note-off velocity is often 0 or the release velocity depending on implementation,
    # but checking type and note is usually sufficient.


# ---------------------------------------------------------------------------
# Domain Logic Tests (Public API)
# ---------------------------------------------------------------------------


def test_load_plugin_lifecycle(mocker):
    """Test that loading a plugin initializes it correctly and simulates the UI hack."""
    # We trap the thread/sleep logic to avoid test slowness/flakes
    mocker.patch("src.data.vst.core._call_with_interrupt", lambda fn, **kw: fn())

    # Inject our fake factory
    plugin = load_plugin("path/to/synth.vst3", plugin_factory=FakeVSTPlugin)

    assert isinstance(plugin, FakeVSTPlugin)
    assert plugin.editor_shown is True  # Verify the UI hack ran


def test_set_params_updates_state(fake_plugin):
    """Test parameter setting logic against our fake state."""
    target_params = {"osc1_pitch": 0.5, "filter_cutoff": 0.9}
    set_params(fake_plugin, target_params)

    assert fake_plugin.parameters["osc1_pitch"].raw_value == 0.5
    assert fake_plugin.parameters["filter_cutoff"].raw_value == 0.9


def test_render_params_full_flow(fake_plugin):
    """
    Verify the entire render pipeline behavior:
    1. Preset loading
    2. Flushing (resetting state)
    3. Parameter setting
    4. Audio generation
    """
    params = {"osc1_pitch": 0.75}
    output = render_params(
        fake_plugin,
        params=params,
        midi_note=60,
        velocity=100,
        note_start_and_end=(0.1, 1.0),
        signal_duration_seconds=1.0,
        sample_rate=44100,
        channels=2,
        preset_path="presets/bass.vstpreset",
    )

    # 1. Preset loaded?
    assert fake_plugin.preset_loaded == "presets/bass.vstpreset"

    # 2. Parameters set?
    assert fake_plugin.parameters["osc1_pitch"].raw_value == 0.75

    # 3. Flushed correctly? (We expect resets before render)
    assert fake_plugin.reset_count > 0

    # 4. Process called with MIDI?
    # We expect some "empty" flush calls, and at least one "real" call with our notes
    real_calls = [log for log in fake_plugin.process_history if len(log) > 0]
    assert len(real_calls) == 1

    # Verify the MIDI notes passed to the "real" render
    midi_block = real_calls[0]
    assert len(midi_block) == 2  # On/Off


def test_render_params_no_preset(fake_plugin):
    """Verify behavior when no preset is provided."""
    render_params(
        fake_plugin,
        params={},
        midi_note=60,
        velocity=100,
        note_start_and_end=(0, 1),
        signal_duration_seconds=1.0,
        sample_rate=44100,
        channels=2,
        preset_path=None,
    )
    assert fake_plugin.preset_loaded is None


def test_write_wav(tmp_path, mocker):
    """Verify file writing using injected factory."""
    # We use a factory mock so we can inject our FakeAudioFile
    fake_file_instance = FakeAudioFile("dummy", "w", 44100, 2)
    factory_mock = mocker.Mock(return_value=fake_file_instance)

    audio_data = np.random.rand(2, 100).astype(np.float32)
    path = str(tmp_path / "test.wav")

    write_wav(audio_data, path, 44100, 2, audio_file_factory=factory_mock)

    # Check arguments passed to factory
    factory_mock.assert_called_once_with(path, "w", 44100, 2)

    # Check data transposition (AudioFile expects frames x channels, we usually work in channels x frames)
    # The source code does: f.write(audio.T)
    np.testing.assert_array_equal(fake_file_instance.written_data, audio_data.T)


# ---------------------------------------------------------------------------
# Utility Logic Tests (Infrastructure)
# ---------------------------------------------------------------------------


def test_call_with_interrupt_logic(mocker):
    """Test the threading mechanics of _call_with_interrupt isolated from any VST logic.

    We verify it creates a thread and sleeps.
    """
    mock_sleep = mocker.Mock()
    mock_interrupt = mocker.Mock()
    mock_fn = mocker.Mock()

    _call_with_interrupt(
        mock_fn, sleep_time=10.0, sleep_fn=mock_sleep, interrupt_fn=mock_interrupt
    )

    # The function should have run
    mock_fn.assert_called_once()

    # The helper thread should have slept and fired interrupt
    mock_sleep.assert_called_with(10.0)
    mock_interrupt.assert_called_once()
