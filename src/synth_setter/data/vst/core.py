import contextlib
import json
import plistlib
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import mido
import numpy as np
from loguru import logger
from pedalboard import VST3Plugin
from pedalboard.io import AudioFile

# How long the editor stays open before we signal it to close.
_EDITOR_INIT_DELAY_SECONDS = 0.5
# Upper bound on how long ``editor_held_open`` waits for the editor thread to
# drain after the close event is set; a generous safety net for the held-open
# path (#1187), not a normal-case timing parameter.
_EDITOR_JOIN_TIMEOUT_SECONDS = 2.0
# Upper bound on the editor-thread start handshake; a miss raises
# ``EditorStartTimeout`` rather than proceeding silently (#1198, #1204).
_EDITOR_START_TIMEOUT_SECONDS = 5.0


class EditorStartTimeout(RuntimeError):
    """Raised when the ``editor_held_open`` start handshake misses its deadline.

    Surfaces a slow editor-thread bring-up as a loud failure so the render aborts and the trace is
    attributable, rather than degrading to a warning that would be lost in normal log noise
    (#1204).
    """


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


@contextlib.contextmanager
def editor_held_open(plugin: VST3Plugin) -> Iterator[None]:
    """Hold ``plugin.show_editor`` open on a background thread for the ``with`` body.

    The editor runs on a daemon thread blocking inside ``show_editor(close_event)``.
    ``__exit__`` sets the event, joins the thread (bounded by
    ``_EDITOR_JOIN_TIMEOUT_SECONDS``), and re-raises any exception the editor
    thread raised — already logged at the moment of failure via
    ``logger.exception``. **Body exceptions always win:** if the ``with`` body
    raises, that exception propagates and a captured editor-thread exception
    is logged but not re-raised (it would otherwise mask the original error in
    the ``finally``-clause raise). If the join times out the daemon thread is
    left to be reaped at process exit; a warning records the leak.

    Before yielding, blocks on a start handshake bounded by
    ``_EDITOR_START_TIMEOUT_SECONDS`` (#1198). A miss raises
    ``EditorStartTimeout`` — the body never runs, so a stuck editor bring-up
    fails loud and traceable rather than masquerading as a slow render
    (#1204). The event proves the editor thread reached the line just before
    ``show_editor`` without the host raising, not that ``show_editor`` has
    realised the window — pedalboard exposes no editor-ready signal.

    :param plugin: A loaded VST3 plugin whose editor is realised for the block.
    :raises EditorStartTimeout: ``editor_started`` did not fire within
        ``_EDITOR_START_TIMEOUT_SECONDS``; the ``with`` body is skipped and
        the editor thread is joined under the same bounded teardown as the
        success path so a late-starting daemon cannot leak past the failed
        context-entry.
    :raises Exception: Propagated from the editor thread at ``__exit__`` only
        when the ``with`` body itself raised nothing — typically a
        ``RuntimeError`` from the VST3 host.
    """
    close_editor = threading.Event()
    editor_started = threading.Event()
    captured: list[Exception] = []

    def _run_editor() -> None:
        try:
            editor_started.set()
            plugin.show_editor(close_editor)
        except Exception as exc:  # noqa: BLE001 — fan-in for any host-side failure
            logger.exception("vst-editor-window crashed: {}", exc)
            captured.append(exc)

    editor_thread = threading.Thread(target=_run_editor, daemon=True, name="vst-editor-window")
    editor_thread.start()
    try:
        if not editor_started.wait(timeout=_EDITOR_START_TIMEOUT_SECONDS):
            raise EditorStartTimeout(
                f"vst-editor-window did not signal start within {_EDITOR_START_TIMEOUT_SECONDS}s"
            )
        yield
    finally:
        close_editor.set()
        editor_thread.join(timeout=_EDITOR_JOIN_TIMEOUT_SECONDS)
        if editor_thread.is_alive():
            logger.warning(
                "vst-editor-window did not drain within {}s; daemon thread "
                "leaks (reaped at process exit)",
                _EDITOR_JOIN_TIMEOUT_SECONDS,
            )
        # sys.exc_info() returns the body's exception if `with` body raised,
        # else (None, None, None). Only surface the captured editor-thread
        # exception when the body succeeded — re-raising during an active
        # body exception would mask the original error (raise-in-finally wins).
        body_exc_active = sys.exc_info()[0] is not None
        if captured and not body_exc_active:
            exc: Exception = captured[0]
            raise exc
        if captured and body_exc_active:
            logger.error(
                "vst-editor-window also crashed during body exception: {}",
                captured[0],
            )


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
