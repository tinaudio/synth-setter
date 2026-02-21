"""Core functionality for VST plugin manipulation and audio rendering."""

import _thread
import threading
import time
from typing import Callable, Optional, Tuple

import mido
import numpy as np
from loguru import logger
from pedalboard import VST3Plugin
from pedalboard.io import AudioFile


def _call_with_interrupt(
    fn: Callable,
    sleep_time: float = 2.0,
    sleep_fn=time.sleep,
    interrupt_fn=_thread.interrupt_main,
):
    """Calls the function fn on the main thread, while another thread sends a KeyboardInterrupt
    (SIGINT) to the main thread."""

    def send_interrupt():
        # Brief sleep so that fn starts before we send the interrupt
        sleep_fn(sleep_time)
        interrupt_fn()

    # Create and start the thread that sends the interrupt
    t = threading.Thread(target=send_interrupt)
    t.start()

    try:
        fn()
    except KeyboardInterrupt:
        print("Interrupted main thread.")
    finally:
        t.join()


def _prepare_plugin(plugin: VST3Plugin) -> None:
    """Prepare plugin for usage by showing editor (hacky fix for some VSTs)."""
    _call_with_interrupt(plugin.show_editor, sleep_time=2.0)


def load_plugin(plugin_path: str, plugin_factory=VST3Plugin) -> VST3Plugin:
    """Load a VST3 plugin from a path."""
    logger.info(f"Loading plugin {plugin_path}")
    p = plugin_factory(plugin_path)
    logger.info(f"Plugin {plugin_path} loaded")
    logger.info("Preparing plugin for preset load...")
    _prepare_plugin(p)
    logger.info("Plugin ready")
    return p


def load_preset(plugin: VST3Plugin, preset_path: str) -> None:
    """Load a specific preset file into the plugin."""
    logger.info(f"Loading preset {preset_path}")
    plugin.load_preset(preset_path)
    logger.info(f"Preset {preset_path} loaded")


def set_params(plugin: VST3Plugin, params: dict[str, float]) -> None:
    """Set the plugin parameters from a dictionary of parameter names and values."""
    for k, v in params.items():
        plugin.parameters[k].raw_value = v


def write_wav(
    audio: np.ndarray,
    path: str,
    sample_rate: float,
    channels: int,
    audio_file_factory: Callable = AudioFile,
) -> None:
    """Write audio data to a WAV file."""
    with audio_file_factory(str(path), "w", sample_rate, channels) as f:
        f.write(audio.T)


def render_params(
    plugin: VST3Plugin,
    params: dict[str, float],
    midi_note: int,
    velocity: int,
    note_start_and_end: Tuple[float, float],
    signal_duration_seconds: float,
    sample_rate: float,
    channels: int,
    preset_path: Optional[str] = None,
) -> np.ndarray:
    """Render audio by setting parameters and sending a MIDI note to the plugin."""
    if preset_path is not None:
        load_preset(plugin, preset_path)

    logger.debug("post-load flush")
    plugin.process([], 32.0, sample_rate, channels, 2048, True)  # flush
    plugin.reset()

    logger.debug("setting params")
    set_params(plugin, params)
    # plugin.reset()

    logger.debug("post-param flush")
    plugin.process([], 32.0, sample_rate, channels, 2048, True)  # flush
    plugin.reset()

    midi_events = make_midi_events(midi_note, velocity, *note_start_and_end)

    logger.debug("rendering audio")
    output = plugin.process(
        midi_events, signal_duration_seconds, sample_rate, channels, 2048, True
    )

    logger.debug("post-render flush")
    plugin.process([], 32.0, sample_rate, channels, 2048, True)  # flush
    plugin.reset()

    return output


def make_midi_events(pitch: int, velocity: int, note_start: float, note_end: float):
    """Create a tuple of MIDI events (note on/off messages) for pedalboard."""
    events = []
    note_on = mido.Message("note_on", note=pitch, velocity=velocity, time=0)
    events.append((note_on.bytes(), note_start))
    note_off = mido.Message("note_off", note=pitch, velocity=velocity, time=0)
    events.append((note_off.bytes(), note_end))

    return tuple(events)
