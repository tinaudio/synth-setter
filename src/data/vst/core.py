import _thread
import os
import threading
import time
from collections.abc import Callable
from typing import Optional

import mido
import numpy as np
from loguru import logger
from pedalboard import VST3Plugin
from pedalboard.io import AudioFile


def _call_with_interrupt(fn: Callable, sleep_time: float = 2.0):
    """Calls the function fn on the main thread, while another thread sends a KeyboardInterrupt
    (SIGINT) to the main thread."""

    def send_interrupt():
        # Brief sleep so that fn starts before we send the interrupt
        time.sleep(sleep_time)
        _thread.interrupt_main()

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
    """Show and dismiss the editor to initialize plugin state."""
    _call_with_interrupt(plugin.show_editor, sleep_time=2.0)


def load_plugin(plugin_path: str) -> VST3Plugin:
    """Load a VST3 plugin and prepare it for preset loading."""
    logger.info(f"Loading plugin {plugin_path}")
    p = VST3Plugin(plugin_path)
    logger.info(f"Plugin {plugin_path} loaded")
    logger.info("Preparing plugin for preset load...")
    _prepare_plugin(p)
    logger.info("Plugin ready")
    return p


def load_preset(plugin: VST3Plugin, preset_path: str) -> None:
    """Load a preset file into the plugin."""
    logger.info(f"Loading preset {preset_path}")
    plugin.load_preset(preset_path)
    logger.info(f"Preset {preset_path} loaded")


def set_params(plugin: VST3Plugin, params: dict[str, float]) -> None:
    """Set raw parameter values on the plugin."""
    for k, v in params.items():
        plugin.parameters[k].raw_value = v


def write_wav(audio: np.ndarray, path: str, sample_rate: float, channels: int) -> None:
    """Write audio array to a WAV file."""
    with AudioFile(str(path), "w", sample_rate, channels) as f:
        f.write(audio.T)


def render_params(
    plugin: VST3Plugin,
    params: dict[str, float],
    midi_note: int,
    velocity: int,
    note_start_and_end: tuple[float, float],
    signal_duration_seconds: float,
    sample_rate: float,
    channels: int,
    preset_path: str | None = None,
    full_flush: bool | None = None,
) -> np.ndarray:
    """Load preset, set params, and render audio through the plugin."""
    if full_flush is None:
        full_flush = os.environ.get("FULL_FLUSH", "0") == "1"

    if preset_path is not None:
        load_preset(plugin, preset_path)
        # Always flush+reset after preset load — presets change oscillator type,
        # which dynamically exposes different parameters (e.g. sawtooth, pulse).
        # Without this, set_params fails with KeyError on preset-dependent params.
        logger.debug("post-load flush")
        plugin.process([], 32.0, sample_rate, channels, 2048, True)
        plugin.reset()

    logger.debug("setting params")
    set_params(plugin, params)

    if full_flush:
        logger.debug("post-param flush")
        plugin.process([], 32.0, sample_rate, channels, 2048, True)
        plugin.reset()

    midi_events = make_midi_events(midi_note, velocity, *note_start_and_end)

    # reset=True (the last arg) clears plugin state before rendering.
    # This is pedalboard's default — no manual post-render flush needed.
    logger.debug("rendering audio")
    output = plugin.process(
        midi_events, signal_duration_seconds, sample_rate, channels, 2048, True
    )

    if full_flush:
        # Explicit post-render flush for conservative mode. Redundant when
        # the next process() call uses reset=True (default), but ensures
        # clean state if plugin is inspected between renders.
        logger.debug("post-render flush")
        plugin.process([], 32.0, sample_rate, channels, 2048, True)
        plugin.reset()

    return output


def make_midi_events(pitch: int, velocity: int, note_start: float, note_end: float):
    """Create MIDI note-on/note-off event pairs for rendering."""
    events = []
    note_on = mido.Message("note_on", note=pitch, velocity=velocity, time=0)
    events.append((note_on.bytes(), note_start))
    note_off = mido.Message("note_off", note=pitch, velocity=velocity, time=0)
    events.append((note_off.bytes(), note_end))

    return tuple(events)
