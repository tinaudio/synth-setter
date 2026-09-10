import importlib.metadata
import json
import re
import shutil
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import mido
import numpy as np
from loguru import logger
from pedalboard import VST3Plugin
from pedalboard.io import AudioFile

from synth_setter.data.vst.torchsynth_param_spec import TORCHSYNTH_PLUGIN_NAME
from synth_setter.plugin_runtime import plugin_bundle_version, validated_bundle_lease
from synth_setter.resources import faustwasm_dir
from synth_setter.renderer_backend import (
    FAUST_PLUGIN_NAME,
    PEDALBOARD_BLOCK_SIZE,
    PYFDN_PLUGIN_NAME,
    SURGEPY_PLUGIN_NAME,
    FlushBlocks,
    pedalboard_flush_blocks,
)

# How long the editor stays open before we signal it to close.
_EDITOR_INIT_DELAY_SECONDS = 0.5
# Hard ceiling on how long ``run_with_editor_held_open`` waits for the render
# worker to drain after ``show_editor`` returns; exceeding it raises
# ``RenderWorkerLeaked`` (#1204) so the helper never returns with a live
# worker thread.
_EDITOR_JOIN_TIMEOUT_SECONDS = 2.0

_BodyResult = TypeVar("_BodyResult")


class RenderWorkerLeaked(RuntimeError):
    """Render worker outlived the join window — body never reached a terminal state.

    Raised by ``run_with_editor_held_open`` when ``show_editor`` returns but the
    worker thread is still alive past ``_EDITOR_JOIN_TIMEOUT_SECONDS``, or when
    the worker exited without capturing a result or exception. Either branch
    means the helper cannot honour its "returns whatever body() returned"
    contract; surfacing it as an exception keeps the caller from treating an
    in-flight or empty render as success (#1204).
    """


def extract_backend_version(renderer_backend: str) -> str:
    """Return the installed version of a separately versioned rendering host.

    :param renderer_backend: Rendering host whose distribution version is required.
    :returns: Installed host distribution version.
    :raises ValueError: The backend has no separate version contract.
    :raises RuntimeError: Host version probing or package metadata inspection fails.
    """
    if renderer_backend == "dawdreamer":
        return importlib.metadata.version("dawdreamer")
    if renderer_backend == "faustcpp":
        if shutil.which("faust") is None or shutil.which("g++") is None:
            raise RuntimeError("install the Faust CLI and g++ to use renderer_backend='faustcpp'")
        try:
            result = subprocess.run(  # noqa: S603
                ["faust", "--version"],  # noqa: S607
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Faust CLI version probe failed: {error.stderr}") from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("Faust CLI version probe timed out after 10 seconds") from error
        match = re.search(r"FAUST Version ([0-9]+(?:\.[0-9]+)+)", result.stdout)
        if match is None:
            raise RuntimeError("Faust CLI returned an unrecognized version string")
        return match.group(1)
    if renderer_backend == "faustwasm":
        package = faustwasm_dir() / "vendor" / "package.json"
        if not package.is_file():
            raise RuntimeError("packaged @grame/faustwasm metadata is unavailable")
        try:
            version = json.loads(package.read_text()).get("version")
        except (json.JSONDecodeError, AttributeError) as error:
            raise RuntimeError("packaged @grame/faustwasm metadata is malformed") from error
        if not isinstance(version, str) or not version.strip():
            raise RuntimeError("packaged @grame/faustwasm metadata is malformed")
        return version
    raise ValueError(f"renderer backend has no separate version contract: {renderer_backend!r}")


def extract_renderer_version(plugin_path: Path) -> str:
    """Extract the version string from a VST3 plugin bundle or Python backend.

    Bare Python-backend names resolve to their installed package versions
    (propagating ``importlib.metadata.PackageNotFoundError``
    when the package is missing). Otherwise, tries the static-metadata files first
    (`Contents/moduleinfo.json` on Linux, `Contents/Info.plist` on macOS), then
    falls back to loading the plugin via pedalboard and reading
    `plugin.version`. The fallback requires a usable X11 display, so callers in
    interpreter-only contexts (the SkyPilot launcher) must avoid it — they pin
    `synth.synth_version` in the dataset config (serialized as
    `render.synth.synth_version` in the spec) and let the
    worker compare against this function's output before rendering (see
    `synth_setter.cli.generate_dataset.generate`).

    :raises FileNotFoundError: The bundle path or required managed-integrity record is absent.
    """
    if str(plugin_path) == FAUST_PLUGIN_NAME:
        return extract_backend_version("dawdreamer")
    if str(plugin_path) == TORCHSYNTH_PLUGIN_NAME:
        return importlib.metadata.version(TORCHSYNTH_PLUGIN_NAME)
    if str(plugin_path) == PYFDN_PLUGIN_NAME:
        return importlib.metadata.version("pyFDN")
    if str(plugin_path) == SURGEPY_PLUGIN_NAME:
        from synth_setter.data.vst.surgepy_runtime import import_surgepy

        return import_surgepy().getVersion()
    if not plugin_path.exists():
        raise FileNotFoundError(f"Plugin path does not exist: {plugin_path}")

    return plugin_bundle_version(plugin_path)


def load_plugin(plugin_path: str, plugin_name: str | None = None) -> VST3Plugin:
    """Load a VST3 plugin instance after validating manager-owned integrity.

    No warm-up — see ``warmup_plugin``.

    :param plugin_path: Path to the ``.vst3`` bundle.
    :param plugin_name: Factory class to open from a multi-class bundle; ``None``
        opens the sole class, and pedalboard raises ``ValueError`` listing the
        classes when a bundle exposes more than one.
    :returns: The loaded plugin.
    """
    logger.info(f"Loading plugin {plugin_path}")
    with validated_bundle_lease(Path(plugin_path)) as validated_bundle:
        validated_path = str(validated_bundle)
        p = (
            VST3Plugin(validated_path)
            if plugin_name is None
            else VST3Plugin(validated_path, plugin_name=plugin_name)
        )
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
    thread; a daemon worker handles renders so the editor stays realised for
    the duration of ``body()`` (#1187). Worker exceptions propagate to the
    caller after ``show_editor`` returns and take precedence over the leak
    signal so a body that raised while finishing slowly still surfaces its
    own exception. The helper never returns with a live worker thread:
    ``BodyResult`` is delivered only when the worker reached a terminal state
    inside the join window; any other outcome raises ``RenderWorkerLeaked``.

    :param plugin: A loaded VST3 plugin whose editor is realised for the call.
    :param body: Callable executed on the worker thread; its return value is
        returned to the caller.
    :returns: Whatever ``body()`` returned.
    :raises BaseException: Re-raised from the worker thread if ``body()``
        raised — typically a ``RuntimeError`` from the VST3 host or a render
        failure.
    :raises RenderWorkerLeaked: Worker outlived ``_EDITOR_JOIN_TIMEOUT_SECONDS``
        after the close event was set, or exited without producing a result
        or exception (no body return value can be surfaced safely).
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

    worker = threading.Thread(target=_worker, name="render-worker", daemon=True)
    worker.start()
    try:
        plugin.show_editor(close_editor)
    finally:
        close_editor.set()
        worker.join(timeout=_EDITOR_JOIN_TIMEOUT_SECONDS)
    if captured:
        exc: BaseException = captured[0]
        raise exc
    if worker.is_alive():
        logger.warning(
            "render-worker did not drain within {}s — raising RenderWorkerLeaked"
            " (the body must complete or raise before show_editor returns)",
            _EDITOR_JOIN_TIMEOUT_SECONDS,
        )
        raise RenderWorkerLeaked(
            f"render-worker still alive after {_EDITOR_JOIN_TIMEOUT_SECONDS}s join window;"
            " body did not complete or raise before show_editor returned"
        )
    if not result:
        raise RenderWorkerLeaked(
            "render-worker exited without producing a result or exception;"
            " body() must either return or raise"
        )
    return result[0]


def load_preset(plugin: VST3Plugin, plugin_state_path: str) -> None:
    logger.info(f"Loading preset {plugin_state_path}")
    plugin.load_preset(plugin_state_path)
    logger.info(f"Preset {plugin_state_path} loaded")


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
    plugin_state_path: str | None = None,
    *,
    plugin: VST3Plugin | None = None,
    warmup: bool = False,
    flush_blocks: FlushBlocks | None = None,
) -> np.ndarray:
    """Render a single audio sample; reuse ``plugin`` if supplied, else load fresh.

    Each non-zero ``flush_blocks`` step processes that many silent blocks and then
    resets the plugin (preset-state determinism, #489). When
    ``plugin`` is supplied, ``plugin_path`` / ``plugin_state_path`` are ignored; the
    caller owns load + preset placement. When ``warmup`` is True, ``warmup_plugin``
    runs after loading (or directly on the supplied plugin) and before the flush
    sequence. See #705 for the load-once-per-shard motivation.

    :param plugin_path: Filesystem path to the VST3 plugin.
    :param params: Synthesizer parameter values to apply before rendering.
    :param midi_note: MIDI note number to render.
    :param velocity: MIDI note velocity.
    :param note_start_and_end: Note-on and note-off times in seconds.
    :param signal_duration_seconds: Duration of the rendered signal in seconds.
    :param sample_rate: Audio sample rate in Hz.
    :param channels: Number of output channels.
    :param plugin_state_path: Optional pedalboard plugin-state file to load.
    :param plugin: Existing plugin instance to reuse.
    :param warmup: Whether to run the plugin warm-up sequence.
    :param flush_blocks: Silent block counts after load, parameter writes, and the render;
        ``None`` covers ``PEDALBOARD_FLUSH_SECONDS`` at ``sample_rate`` for each step.
    :returns: Rendered audio as a channel-first NumPy array.
    """
    if plugin is None:
        plugin = load_plugin(plugin_path)
        if plugin_state_path is not None:
            load_preset(plugin, plugin_state_path)

    if warmup:
        warmup_plugin(plugin)
    if flush_blocks is None:
        flush_blocks = pedalboard_flush_blocks(sample_rate)
    host = _HostFormat(sample_rate=sample_rate, channels=channels)

    _flush_and_reset(plugin, host, blocks=flush_blocks.post_load, step="post-load")

    logger.debug("setting params")
    set_params(plugin, params)

    _flush_and_reset(plugin, host, blocks=flush_blocks.post_param, step="post-param")

    midi_events = make_midi_events(midi_note, velocity, *note_start_and_end)

    logger.debug("rendering audio")
    output = plugin.process(
        midi_events, signal_duration_seconds, sample_rate, channels, PEDALBOARD_BLOCK_SIZE, True
    )

    _flush_and_reset(plugin, host, blocks=flush_blocks.post_render, step="post-render")

    return output


@dataclass(frozen=True)
class _HostFormat:
    """Sample rate and channel count every host ``process`` call shares.

    .. attribute :: sample_rate

       Audio sample rate in Hz.

    .. attribute :: channels

       Number of output channels.
    """

    sample_rate: float
    channels: int


def _flush_and_reset(plugin: VST3Plugin, host: _HostFormat, *, blocks: int, step: str) -> None:
    """Process ``blocks`` silent host blocks and reset the plugin; zero blocks skips both.

    :param plugin: Loaded plugin instance.
    :param host: Sample rate and channel count of the silent blocks.
    :param blocks: Number of silent host blocks to process.
    :param step: Render step name for the debug log.
    """
    if blocks == 0:
        return
    logger.debug(f"{step} flush")
    seconds = blocks * PEDALBOARD_BLOCK_SIZE / host.sample_rate
    plugin.process([], seconds, host.sample_rate, host.channels, PEDALBOARD_BLOCK_SIZE, True)
    plugin.reset()


def make_midi_events(pitch: int, velocity: int, note_start: float, note_end: float):
    events = []
    note_on = mido.Message("note_on", note=pitch, velocity=velocity, time=0)
    events.append((note_on.bytes(), note_start))
    note_off = mido.Message("note_off", note=pitch, velocity=velocity, time=0)
    events.append((note_off.bytes(), note_end))

    return tuple(events)
