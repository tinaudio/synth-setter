import json
import plistlib
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

import mido
import numpy as np
from loguru import logger
from pedalboard import VST3Plugin
from pedalboard.io import AudioFile

# How long the editor stays open before we signal it to close.
_EDITOR_INIT_DELAY_SECONDS = 0.5


def extract_renderer_version(plugin_path: Path) -> str:
    """Extract the version string from a VST3 plugin bundle.

    Tries the static-metadata files first (`Contents/moduleinfo.json` on Linux,
    `Contents/Info.plist` on macOS), then falls back to loading the plugin via
    pedalboard and reading `plugin.version`. The fallback requires a usable
    X11 display, so callers in interpreter-only contexts (the SkyPilot
    launcher) must avoid it — they pin `renderer_version` in the dataset
    config that produces the spec and let the worker compare against this
    function's output before rendering (see
    `synth_setter.cli.generate_dataset.run`).

    :raises FileNotFoundError: plugin_path does not exist.
    :raises RuntimeError: version cannot be extracted by any method.
    :raises json.JSONDecodeError: moduleinfo.json is malformed.
    :raises plistlib.InvalidFileException: Info.plist is malformed.
    """
    if not plugin_path.exists():
        raise FileNotFoundError(f"Plugin path does not exist: {plugin_path}")

    moduleinfo = plugin_path / "Contents" / "moduleinfo.json"
    if moduleinfo.is_file():
        return json.loads(moduleinfo.read_text())["Version"]

    plist = plugin_path / "Contents" / "Info.plist"
    if plist.is_file():
        return plistlib.loads(plist.read_bytes())["CFBundleShortVersionString"]

    # Pedalboard fallback: prebuilt plugin bundles (e.g. Surge XT shipped via
    # .deb) don't always carry moduleinfo.json. Loading the .so via pedalboard
    # gives us VST3 factory metadata; this requires X11.
    plugin = VST3Plugin(str(plugin_path))
    version = plugin.version
    if not version:
        raise RuntimeError(f"Could not extract version from {plugin_path}")
    return version


def load_plugin(plugin_path: str, *, open_gui: bool = True) -> VST3Plugin:
    """Load a VST3 plugin, optionally running a brief editor warm-up on non-Darwin.

    The editor warm-up runs by default to preserve historical behaviour. Pass
    ``open_gui=False`` to skip it (e.g. when the caller already loaded the
    plugin once for a long-lived shard render — see ``writers._render_in_batches``).
    The warm-up is unconditionally skipped on Darwin: ``show_editor``
    accumulates AppKit/CGS commit-handler state per call in unbundled python
    and triggers SIGTRAP after ~3-4 plugin reloads (#714).

    :param plugin_path: Path to the VST3 bundle to load.
    :param open_gui: When True (default), run the ``show_editor`` warm-up on
        non-Darwin platforms; ignored on Darwin where the warm-up is never run.
    :returns: The freshly-loaded plugin.
    :rtype: VST3Plugin
    """
    logger.info(f"Loading plugin {plugin_path}")
    p = VST3Plugin(plugin_path)
    logger.info(f"Plugin {plugin_path} loaded")
    if open_gui and sys.platform != "darwin":
        logger.info("Preparing plugin for preset load...")
        close_editor = threading.Event()
        timer = threading.Timer(_EDITOR_INIT_DELAY_SECONDS, close_editor.set)
        timer.daemon = True
        timer.start()
        try:
            p.show_editor(close_editor)
        finally:
            timer.cancel()
            close_editor.set()  # defensive: ensure show_editor unblocks even if Timer fails
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
    *,
    plugin: Optional[VST3Plugin] = None,
    open_gui: bool = True,
) -> np.ndarray:
    """Render a single audio sample, optionally reusing a pre-loaded plugin.

    Default path (``plugin=None``): load the plugin fresh and apply the preset
    per call — preserves the historical work-around for stale-state bug #489
    at the cost of a ~7s plugin load per render (see #705 for the perf
    follow-up).

    Cached path (``plugin`` supplied): skip ``load_plugin`` and ``load_preset``
    — the caller is responsible for having done both once before the loop.
    The existing flush sequence still runs on every call, which is what makes
    preset state deterministic per #489.

    :param plugin: Optional pre-loaded plugin instance. When supplied, both
        ``load_plugin`` and ``load_preset`` are skipped and ``plugin_path`` /
        ``preset_path`` are ignored.
    :param open_gui: Forwarded to ``load_plugin`` on the default path; ignored
        when ``plugin`` is supplied (the caller chose the warm-up policy when
        they loaded the plugin).
    """
    if plugin is None:
        plugin = load_plugin(plugin_path, open_gui=open_gui)
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
