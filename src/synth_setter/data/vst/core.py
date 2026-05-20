import json
import plistlib
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import mido
import numpy as np
from loguru import logger
from pedalboard import VST3Plugin
from pedalboard.io import AudioFile

# How long the editor stays open before we signal it to close.
_EDITOR_INIT_DELAY_SECONDS = 0.5
# Upper bound on how long ``run_with_editor_held_open`` waits for the render
# worker to drain after ``show_editor`` returns; a generous safety net for the
# held-open path (#1187), not a normal-case timing parameter.
_EDITOR_JOIN_TIMEOUT_SECONDS = 2.0

_BodyResult = TypeVar("_BodyResult")


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


def load_plugin(plugin_path: str) -> VST3Plugin:
    """Load a VST3 plugin instance.

    No warm-up — see ``warmup_plugin``.
    """
    logger.info(f"Loading plugin {plugin_path}")
    p = VST3Plugin(plugin_path)
    logger.info(f"Plugin {plugin_path} loaded")
    return p


def warmup_plugin(plugin: VST3Plugin) -> None:
    """Run the ``show_editor`` warm-up to nudge the plugin's commit-handler state.

    Side-effect only; the plugin must already be loaded. Callers are responsible
    for not exceeding the empirical ~3-4 calls-per-process threshold on Darwin
    (#714) — ``RenderConfig.gui_toggle_cadence`` enforces this for the renderer
    paths by rejecting ``"render"`` on Darwin.

    :param plugin: A loaded VST3 plugin; ``show_editor`` is invoked once on it.
    """
    logger.info("Warming up plugin via show_editor (commit-handler state)...")
    close_editor = threading.Event()
    timer = threading.Timer(_EDITOR_INIT_DELAY_SECONDS, close_editor.set)
    timer.daemon = True
    timer.start()
    try:
        plugin.show_editor(close_editor)
    finally:
        timer.cancel()
        close_editor.set()  # defensive: ensure show_editor unblocks even if Timer fails


def run_with_editor_held_open(plugin: VST3Plugin, body: Callable[[], _BodyResult]) -> _BodyResult:
    """Invoke ``body()`` on a worker thread while the caller blocks in ``plugin.show_editor``.

    Pedalboard 0.9.x requires ``show_editor`` to run on the process main
    thread; the worker handles renders so the editor stays realised for the
    duration of ``body()`` (#1187). Worker exceptions propagate to the caller
    after ``show_editor`` returns. If the worker outlives the
    ``_EDITOR_JOIN_TIMEOUT_SECONDS`` join window after the close event is set,
    a warning records the leak.

    :param plugin: A loaded VST3 plugin whose editor is realised for the call.
    :param body: Callable executed on the worker thread; its return value is
        returned to the caller.
    :returns: Whatever ``body()`` returned.
    :raises BaseException: Re-raised from the worker thread if ``body()``
        raised — typically a ``RuntimeError`` from the VST3 host or a render
        failure.
    """
    close_editor = threading.Event()
    result: list[_BodyResult] = []
    captured: list[BaseException] = []

    def _worker() -> None:
        try:
            result.append(body())
        except BaseException as exc:  # noqa: BLE001 — propagate to caller after show_editor returns
            captured.append(exc)
        finally:
            close_editor.set()

    worker = threading.Thread(target=_worker, name="render-worker")
    worker.start()
    try:
        plugin.show_editor(close_editor)
    finally:
        close_editor.set()
        worker.join(timeout=_EDITOR_JOIN_TIMEOUT_SECONDS)
        if worker.is_alive():
            logger.warning(
                "render-worker did not drain within {}s; thread leaks (reaped at process exit)",
                _EDITOR_JOIN_TIMEOUT_SECONDS,
            )
    if captured:
        exc: BaseException = captured[0]
        raise exc
    if not result:
        # Worker outlived the join window without completing — the leak warning above
        # records this; there's no body return value to surface.
        return None  # type: ignore[return-value]
    return result[0]


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
    note_start_and_end: tuple[float, float],
    signal_duration_seconds: float,
    sample_rate: float,
    channels: int,
    preset_path: str | None = None,
    *,
    plugin: VST3Plugin | None = None,
    warmup: bool = False,
) -> np.ndarray:
    """Render a single audio sample; reuse ``plugin`` if supplied, else load fresh.

    The flush sequence runs every call (preset-state determinism, #489). When
    ``plugin`` is supplied, ``plugin_path`` / ``preset_path`` are ignored; the
    caller owns load + preset placement. When ``warmup`` is True, ``warmup_plugin``
    runs after loading (or directly on the supplied plugin) and before the flush
    sequence. See #705 for the load-once-per-shard motivation.
    """
    if plugin is None:
        plugin = load_plugin(plugin_path)
        if preset_path is not None:
            load_preset(plugin, preset_path)

    if warmup:
        warmup_plugin(plugin)

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
