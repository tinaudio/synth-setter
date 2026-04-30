import sys
import threading
import time
from typing import Optional, Tuple

import mido
import numpy as np
from loguru import logger
from pedalboard import VST3Plugin
from pedalboard.io import AudioFile

# How long the helper thread waits before signalling ``show_editor`` to close.
_PREPARE_PLUGIN_SLEEP_SECONDS = 0.5

# Upper bound for how long load_plugin waits for the helper thread to exit
# after signalling its stop event. The helper only sleeps for
# ``_PREPARE_PLUGIN_SLEEP_SECONDS`` so 1.0s is a generous ceiling.
_PREPARE_PLUGIN_JOIN_TIMEOUT_SECONDS = 1.0


def _prepare_plugin(
    stop_event: threading.Event,
    sleep_time: float = _PREPARE_PLUGIN_SLEEP_SECONDS,
) -> None:
    """Sleep, then signal ``show_editor`` to close.

    Used by :func:`load_plugin` to drive ``show_editor`` long enough that the
    plugin completes initialisation, but no longer than necessary for batch
    rendering throughput.
    """
    time.sleep(sleep_time)
    stop_event.set()


def load_plugin(plugin_path: str) -> VST3Plugin:
    """Load a VST3 plugin (with a brief editor warmup on non-Darwin — see comment below for
    rationale)."""
    logger.info(f"Loading plugin {plugin_path}")
    p = VST3Plugin(plugin_path)
    logger.info(f"Plugin {plugin_path} loaded")
    # show_editor accumulates AppKit/CGS commit-handler state per call in
    # unbundled python and triggers SIGTRAP after ~3-4 plugin reloads on
    # Darwin (#714). The post-load process() flush in render_params is
    # sufficient to commit Surge XT's preset state — see preset-coverage
    # audit on #714 for the empirical justification.
    if sys.platform != "darwin":
        logger.info("Preparing plugin for preset load...")
        stop_event = threading.Event()
        t = threading.Thread(target=_prepare_plugin, args=(stop_event,))
        t.start()
        try:
            p.show_editor(stop_event)
        finally:
            stop_event.set()
            t.join(timeout=_PREPARE_PLUGIN_JOIN_TIMEOUT_SECONDS)
    return p


def load_preset(plugin: VST3Plugin, preset_path: str) -> None:
    logger.info(f"Loading preset {preset_path}")
    plugin.load_preset(preset_path)
    logger.info(f"Preset {preset_path} loaded")


def set_params(plugin: VST3Plugin, params: dict[str, float]) -> None:
    for k, v in params.items():
        plugin.parameters[k].raw_value = v


def write_wav(audio: np.ndarray, path: str, sample_rate: float, channels: int) -> None:
    with AudioFile(str(path), "w", sample_rate, channels) as f:
        f.write(audio.T)


def render_params(
    plugin_path: str,
    params: dict[str, float],
    midi_note: int,
    velocity: int,
    note_start_and_end: Tuple[float, float],
    signal_duration_seconds: float,
    sample_rate: float,
    channels: int,
    preset_path: Optional[str] = None,
) -> np.ndarray:
    """Render a single audio sample by loading the plugin fresh per call.

    Reloads the plugin on every call to work around stale-state bug #489. This incurs an extra
    plugin-load per render; see #705 for the perf follow-up.
    """
    plugin = load_plugin(plugin_path)
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
    events = []
    note_on = mido.Message("note_on", note=pitch, velocity=velocity, time=0)
    events.append((note_on.bytes(), note_start))
    note_off = mido.Message("note_off", note=pitch, velocity=velocity, time=0)
    events.append((note_off.bytes(), note_end))

    return tuple(events)
