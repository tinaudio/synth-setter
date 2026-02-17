import _thread
import threading
import time
from typing import Callable, Optional, Tuple

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
    _call_with_interrupt(plugin.show_editor, sleep_time=2.0)


def load_plugin(plugin_path: str) -> VST3Plugin:
    """Load a VST3 plugin from disk and run preparation warmup.

    :param plugin_path: Filesystem path to the ``.vst3`` plugin bundle.
    :return: Prepared plugin instance.
    """
    logger.info(f"Loading plugin {plugin_path}")
    p = VST3Plugin(plugin_path)
    logger.info(f"Plugin {plugin_path} loaded")
    logger.info("Preparing plugin for preset load...")
    _prepare_plugin(p)
    logger.info("Plugin ready")
    return p


def load_preset(plugin: VST3Plugin, preset_path: str) -> None:
    """Load a preset file into an already-instantiated plugin.

    :param plugin: Plugin instance to mutate.
    :param preset_path: Filesystem path to a preset file.
    """
    logger.info(f"Loading preset {preset_path}")
    plugin.load_preset(preset_path)
    logger.info(f"Preset {preset_path} loaded")


def set_params(plugin: VST3Plugin, params: dict[str, float]) -> None:
    """Set plugin parameters by raw value.

    :param plugin: Plugin instance whose parameters will be updated.
    :param params: Mapping of parameter names to raw scalar values.
    """
    for k, v in params.items():
        plugin.parameters[k].raw_value = v


def write_wav(audio: np.ndarray, path: str, sample_rate: float, channels: int) -> None:
    """Write channel-first audio to a WAV file.

    :param audio: Audio array shaped ``(channels, samples)``.
    :param path: Output WAV path.
    :param sample_rate: Output sample rate in Hz.
    :param channels: Number of channels to write.
    """
    with AudioFile(str(path), "w", sample_rate, channels) as f:
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
    """Render audio from a plugin after optional preset load and param update.

    :param plugin: Prepared VST3 plugin instance.
    :param params: Mapping of parameter names to raw values.
    :param midi_note: MIDI pitch for the rendered note.
    :param velocity: MIDI note-on velocity.
    :param note_start_and_end: Tuple ``(note_on_time_s, note_off_time_s)``.
    :param signal_duration_seconds: Total audio duration to render.
    :param sample_rate: Render sample rate in Hz.
    :param channels: Number of output channels.
    :param preset_path: Optional preset to load before setting params.
    :return: Rendered audio array.
    """
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
    """Create note-on and note-off MIDI events for pedalboard processing.

    :param pitch: MIDI note number.
    :param velocity: MIDI velocity value.
    :param note_start: Note-on event time in seconds.
    :param note_end: Note-off event time in seconds.
    :return: Tuple of ``(message_bytes, event_time_seconds)`` pairs.
    """
    events = []
    note_on = mido.Message("note_on", note=pitch, velocity=velocity, time=0)
    events.append((note_on.bytes(), note_start))
    note_off = mido.Message("note_off", note=pitch, velocity=velocity, time=0)
    events.append((note_off.bytes(), note_end))

    return tuple(events)
