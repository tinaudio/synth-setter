"""Interactive Surge XT preview with real-time audio streaming via pedalboard."""

import logging
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import rootutils

_REPO_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import click  # noqa: E402
import h5py  # noqa: E402
import hdf5plugin  # noqa: F401, E402  side-effect: registers HDF5_PLUGIN_PATH for Blosc2 filters
import mido  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pedalboard import VST3Plugin  # noqa: E402
from pedalboard.io import AudioFile, AudioStream, StreamResampler  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.logging import RichHandler  # noqa: E402

from src.data.vst import load_plugin, load_preset, param_specs  # noqa: E402
from src.data.vst.core import make_midi_events, set_params  # noqa: E402
from src.data.vst.generate_vst_dataset import make_dataset  # noqa: E402
from src.data.vst.param_spec import ParamSpec  # noqa: E402

MIDI_LISTEN_MESSAGE_TYPES = ("note_on", "note_off", "control_change", "pitchwheel", "aftertouch")

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Install the Rich root-logger handler used when this script runs as a CLI.

    Kept out of import-time side effects so importing the module (e.g. from the test suite) doesn't
    reconfigure the root logger or construct a Console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=Console(width=200),
                rich_tracebacks=True,
                markup=False,
                show_path=True,
            )
        ],
    )


CHANNELS = 2
SAMPLE_RATE = 44100
BUFFER_SIZE = 512
MAKE_DATASET_VELOCITY = 100
MAKE_DATASET_SIGNAL_DURATION_SECONDS = 4.0
MAKE_DATASET_MIN_LOUDNESS = -50.0
MAKE_DATASET_SAMPLE_BATCH_SIZE = 32

# Real-time playback rate. Set this to whatever the default output device supports
# if it differs from ``SAMPLE_RATE`` — ``play_audio`` will resample on the fly.
# Resampling adds overhead, so it's optional and disabled by default.
PLAYBACK_SAMPLE_RATE = SAMPLE_RATE

# Maximum time to wait for the audio thread to drain after ``stop_event`` is set.
AUDIO_THREAD_DRAIN_TIMEOUT_SECONDS = 2

# Deterministic recording clip when ``--session-recording-path`` is set.
# The point is reproducibility: same plugin + preset + params -> same WAV.
# A held middle-C note is rendered from ``NOTE_START`` to ``NOTE_END``,
# inside a fixed-length window so any release tail is captured.
SESSION_RECORDING_DURATION_SECONDS = 10.0
SESSION_RECORDING_MIDI_NOTE = 60  # middle C (C4)
SESSION_RECORDING_VELOCITY = 100
SESSION_RECORDING_NOTE_START_SECONDS = 2.0
SESSION_RECORDING_NOTE_END_SECONDS = 4.0
# Buffer size used for the offline ``plugin.process(...)`` render. Mirrors
# the value used in ``src.data.vst.core.render_params``.
_SESSION_RECORDING_BUFFER_SIZE = 2048

# Plugin-flush parameters used by the post-load / pre-render flush pattern; see
# ``_flush_plugin``. Mirror of the values used in ``src.data.vst.core.render_params``.
_PLUGIN_FLUSH_DURATION_SECONDS = 32.0
_PLUGIN_FLUSH_BUFFER_SIZE = 2048

# Return signals for ``keyboard_loop`` actions — distinguishes "user requested
# quit" from "action completed, keep listening".
_KEEP_LOOPING = True
_STOP_LOOPING = False

_DRIFT_TOLERANCE = 1e-6

# Cap how many queued MIDI events ``play_audio`` drains per buffer so a high-rate
# CC stream can't extend the realtime audio callback enough to cause underruns.
# Excess events stay queued and are processed on the next buffer (~12ms later at
# 44.1k/512), which is well within the human-perceptible latency budget.
_MAX_MIDI_EVENTS_PER_BUFFER = 64

# Polling interval for ``midi_listener`` between non-blocking ``port.poll()`` calls.
# Short enough to make the listener responsive to ``stop_event`` (~10ms worst case)
# without busy-spinning on the GIL.
_MIDI_POLL_INTERVAL_SECONDS = 0.01

_VST_SUBPROCESS_TIMEOUT_SECONDS = 300
_EVAL_SUBPROCESS_TIMEOUT_SECONDS = 600
_METRICS_SUBPROCESS_TIMEOUT_SECONDS = 300
_VST_HEADLESS_WRAPPER = _REPO_ROOT / "scripts" / "run-linux-vst-headless.sh"
_EVAL_SCRIPT = _REPO_ROOT / "src" / "eval.py"
_PREDICT_VST_AUDIO_SCRIPT = _REPO_ROOT / "scripts" / "predict_vst_audio.py"
_COMPUTE_AUDIO_METRICS_SCRIPT = _REPO_ROOT / "scripts" / "compute_audio_metrics.py"

# Below this peak, librosa RMS norms underflow and ``compute_rms`` produces
# 0/0 → NaN (see ``compute_rms`` in ``scripts/compute_audio_metrics.py``).
SILENCE_PEAK_THRESHOLD = 1e-4

_METRIC_COLUMNS: frozenset[str] = frozenset({"mss", "wmfcc", "sot", "rms"})


# ----- Test seams ---------------------------------------------------------------------
# Narrow dependency-injection points so tests can substitute fakes that capture state
# instead of monkey-patching module-level functions. Defaults preserve production behavior;
# CLI surface is unchanged. Each seam is keyword-only at the call sites that accept it.
# Refs #844.

# Returns whatever stdlib ``subprocess.run`` / ``subprocess.check_call`` return; production
# callers either ignore the result or read ``.returncode`` from a ``CompletedProcess``.
SubprocessRunner = Callable[..., object]

# ``mido.open_input`` returns an ``IOPort`` context-managed object that exposes ``.poll()``
# (and historically iteration). We type the value as a context manager whose entered handle
# satisfies ``midi_listener``'s structural needs.
PortOpener = Callable[[str], AbstractContextManager[object]]

# A no-arg factory returning an entered audio-output stream. Production binds
# ``AudioStream.default_output_device_name`` lazily so test factories never trigger a
# real-device probe.
AudioStreamFactory = Callable[[], AbstractContextManager[object]]


def _default_audio_stream_factory() -> AbstractContextManager[object]:
    """Build the production ``AudioStream`` used by ``play_audio``.

    Wrapped in a closure so the default-device lookup happens only when the seam is left at its
    default — fake factories used in tests never trigger PortAudio device probing.
    """
    return AudioStream(
        output_device_name=AudioStream.default_output_device_name,
        sample_rate=PLAYBACK_SAMPLE_RATE,
        buffer_size=BUFFER_SIZE,
    )


@dataclass(frozen=True)
class _MetricsFileSpec:
    """Expected shape of a metrics CSV produced by ``compute_audio_metrics.py``."""

    rows: int
    columns: frozenset[str]


def _expected_prediction_filenames(num_samples: int) -> list[str]:
    """Return the sorted list of filenames ``PredictionWriter`` writes per sample."""
    return sorted(
        f"{prefix}-{i}.pt"
        for prefix in ("pred", "target-audio", "target-params")
        for i in range(num_samples)
    )


def _validate_metrics_df(
    metrics_path: Path,
    metrics_df: pd.DataFrame,
    expected: _MetricsFileSpec,
) -> None:
    """Verify ``metrics_df`` has the expected row count and a superset of expected columns, and
    that the expected columns are entirely finite."""
    if len(metrics_df) != expected.rows:
        raise ValueError(f"{metrics_path}: expected {expected.rows} rows, got {len(metrics_df)}")
    missing_columns = expected.columns - set(metrics_df.columns)
    if missing_columns:
        raise ValueError(
            f"{metrics_path}: missing expected columns {sorted(missing_columns)}; "
            f"got {sorted(metrics_df.columns)}"
        )
    numeric = metrics_df[sorted(expected.columns)].to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError(f"{metrics_path} contains NaN/Inf:\n{metrics_df}")


@dataclass(frozen=True)
class PredictionRef:
    """Identifier for a single predicted parameter row on disk."""

    path: Path
    batch_idx: int


@dataclass(frozen=True)
class DatasetRef:
    """Identifier for a single dataset row on disk."""

    path: Path
    batch_idx: int


class PredictionRefType(click.ParamType):
    """Click parser for ``PATH:BATCH_IDX`` prediction references."""

    name = "pred_ref"

    def convert(
        self,
        value: str | PredictionRef,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> PredictionRef:
        """Parse a ``PATH:BATCH_IDX`` string (or pass-through ``PredictionRef``) into a
        ``PredictionRef``."""
        if isinstance(value, PredictionRef):
            return value
        path_str, sep, idx_str = value.rpartition(":")
        if not sep or not path_str or not idx_str:
            self.fail(f"expected PATH:BATCH_IDX (e.g. 'pred-n.pt:n'), got {value!r}", param, ctx)
        try:
            batch_idx = int(idx_str)
        except ValueError:
            self.fail(f"batch index must be an integer, got {idx_str!r}", param, ctx)
        if batch_idx < 0:
            self.fail(f"batch index must be non-negative, got {batch_idx}", param, ctx)
        return PredictionRef(path=Path(path_str), batch_idx=batch_idx)


class DatasetRefType(click.ParamType):
    """Click parser for ``PATH:DATASET_IDX`` dataset references."""

    name = "dataset_ref"

    def convert(
        self,
        value: str | DatasetRef,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> DatasetRef:
        """Parse a ``PATH:DATASET_IDX`` string (or pass-through ``DatasetRef``) into a
        ``DatasetRef``."""
        if isinstance(value, DatasetRef):
            return value
        path_str, sep, idx_str = value.rpartition(":")
        if not sep or not path_str or not idx_str:
            self.fail(f"expected PATH:DATASET_IDX (e.g. 'test.h5:0'), got {value!r}", param, ctx)
        try:
            batch_idx = int(idx_str)
        except ValueError:
            self.fail(f"batch index must be an integer, got {idx_str!r}", param, ctx)
        if batch_idx < 0:
            self.fail(f"dataset index must be non-negative, got {batch_idx}", param, ctx)
        return DatasetRef(path=Path(path_str), batch_idx=batch_idx)


def _load_pred_tensor(pred_path: Path) -> torch.Tensor:
    """Load a prediction tensor from disk (imperative shell — does I/O only).

    :param pred_path: Path to a ``pred-*.pt`` file produced by ``predict_vst_audio.py``.
    :returns: A float tensor of shape ``(batch_size, num_params)`` with values in
        the model output range ``[-1, 1]`` (the inverse of the ``(x + 1) / 2``
        scale used by ``predict_vst_audio.py``). Loaded with
        ``weights_only=True`` since predictions are plain tensors.
    """
    return torch.load(pred_path, map_location="cpu", weights_only=True)


def decode_prediction_row(
    pred_tensor: torch.Tensor,
    batch_idx: int,
    param_spec_name: str,
) -> dict[str, float]:
    """Decode a single predicted row into raw VST synth params (functional core — pure transform).

    :param pred_tensor: Float tensor of shape ``(batch_size, num_params)`` with
        values in ``[-1, 1]`` (inverse of the ``(x + 1) / 2`` scale used by
        ``predict_vst_audio.py``). Out-of-range values are clipped to ``[0, 1]``
        after rescaling.
    :param batch_idx: Row index within the prediction tensor to decode. Must be
        in ``[0, batch_size)``.
    :param param_spec_name: Parameter spec name (key into ``param_specs``) used
        to decode the rescaled row into raw VST parameter values.
    :returns: Dict mapping VST parameter name to its raw (decoded) value.
    :raises IndexError: when ``batch_idx`` is out of range for ``pred_tensor``.
    """
    if batch_idx < 0 or batch_idx >= pred_tensor.shape[0]:
        raise IndexError(f"batch_idx {batch_idx} out of range (batch size {pred_tensor.shape[0]})")

    spec = param_specs[param_spec_name]
    row = pred_tensor[batch_idx].detach().cpu().float().numpy()
    row_scaled = np.clip((row + 1) / 2, 0, 1)
    synth_params, _ = spec.decode(row_scaled)
    return synth_params


def load_dataset_synth_params(
    ref: DatasetRef,
    param_spec_name: str,
) -> dict[str, float]:
    """Load a single row from an h5 dataset's ``param_array`` and decode it into synth params.

    Unlike prediction tensors, h5 dataset rows are already in encoded form
    (output of :meth:`ParamSpec.encode`, values in ``[0, 1]``), so no
    ``(x + 1) / 2`` rescaling is applied.

    :param ref: Reference identifying the h5 file path and row to decode.
    :param param_spec_name: Parameter spec name (key into ``param_specs``) used
        to decode the row into raw VST parameter values.
    :returns: Dict mapping VST parameter name to its raw (decoded) value.
    """
    spec = param_specs[param_spec_name]
    with h5py.File(ref.path, "r") as f:
        param_array = f["param_array"]
        if not isinstance(param_array, h5py.Dataset):
            raise TypeError(
                f"expected h5py.Dataset for 'param_array', got {type(param_array).__name__}"
            )
        row = np.asarray(param_array[ref.batch_idx], dtype=np.float32)

    synth_params, _ = spec.decode(row)
    return synth_params


def load_prediction_synth_params(
    ref: PredictionRef,
    param_spec_name: str,
) -> dict[str, float]:
    """Load a single predicted row from a pred-*.pt file and decode it into synth params.

    Thin orchestrator: reads the prediction tensor from disk via
    :func:`_load_pred_tensor` and decodes the requested row via
    :func:`decode_prediction_row`.

    :param ref: Reference identifying the file path and row to decode.
    :param param_spec_name: Parameter spec name (key into ``param_specs``) used
        to decode the rescaled row into raw VST parameter values.
    :returns: Dict mapping VST parameter name to its raw (decoded) value. The
        underlying tensor has shape ``(batch_size, num_params)`` and dtype
        float, with values in ``[-1, 1]`` (the inverse of the ``(x + 1) / 2``
        scale used by ``predict_vst_audio.py``).
    :raises IndexError: when ``ref.batch_idx`` is out of range for the loaded tensor.
    """
    pred_tensor = _load_pred_tensor(ref.path)
    return decode_prediction_row(pred_tensor, ref.batch_idx, param_spec_name)


def _flush_plugin(plugin: VST3Plugin) -> None:
    """Process an empty buffer and reset the plugin to flush internal state.

    Mirrors the post-load / pre-render flush pattern in
    ``src.data.vst.core.render_params``.
    """
    plugin.process(
        [],
        _PLUGIN_FLUSH_DURATION_SECONDS,
        SAMPLE_RATE,
        CHANNELS,
        _PLUGIN_FLUSH_BUFFER_SIZE,
        True,
    )
    plugin.reset()


def play_audio(
    plugin: VST3Plugin,
    stop_event: threading.Event,
    midi_queue: "queue.Queue[tuple[list[int], float]] | None",
    *,
    audio_stream_factory: AudioStreamFactory | None = None,
) -> None:
    """Stream Surge XT output to the audio device, processing any pending MIDI messages.

    Runs until ``stop_event`` is set (typically by ``keyboard_loop``'s quit action or by
    ``main`` after the plugin editor is closed). When ``midi_queue`` is provided, drains it
    each iteration so the plugin sees incoming notes/CCs from the MIDI listener thread.
    Resamples to ``PLAYBACK_SAMPLE_RATE`` if it differs from ``SAMPLE_RATE`` so the output
    device gets a rate it supports.

    ``audio_stream_factory`` exists for test injection (#844). ``None`` (the default)
    builds the real pedalboard ``AudioStream`` at call time so legacy module-level
    monkeypatches of ``AudioStream`` keep working.
    """
    needs_resample = SAMPLE_RATE != PLAYBACK_SAMPLE_RATE
    stream_resampler = (
        StreamResampler(SAMPLE_RATE, PLAYBACK_SAMPLE_RATE, CHANNELS) if needs_resample else None
    )
    buffer_duration_seconds = BUFFER_SIZE / SAMPLE_RATE
    factory = (
        audio_stream_factory if audio_stream_factory is not None else _default_audio_stream_factory
    )
    with factory() as stream:
        while not stop_event.is_set():
            messages: list[tuple[list[int], float]] = []
            if midi_queue is not None:
                # Cap per-buffer drain — see ``_MAX_MIDI_EVENTS_PER_BUFFER`` rationale.
                for _ in range(_MAX_MIDI_EVENTS_PER_BUFFER):
                    try:
                        messages.append(midi_queue.get_nowait())
                    except queue.Empty:
                        break
            # ``reset`` passed positionally for consistency with ``_flush_plugin`` /
            # ``play_audio_recorded`` / ``src/data/vst/core.py`` and to avoid relying on
            # pedalboard's C-extension keyword-arg support across unpinned versions.
            synth_output = plugin.process(
                messages, buffer_duration_seconds, SAMPLE_RATE, CHANNELS, BUFFER_SIZE, False
            )
            if synth_output.shape != (CHANNELS, BUFFER_SIZE):
                raise ValueError(
                    f"expected synth output shape ({CHANNELS}, {BUFFER_SIZE}), "
                    f"got {synth_output.shape}"
                )
            if stream_resampler is not None:
                stream.write(stream_resampler.process(synth_output), PLAYBACK_SAMPLE_RATE)  # pyright: ignore[reportAttributeAccessIssue]
            else:
                stream.write(synth_output, SAMPLE_RATE)  # pyright: ignore[reportAttributeAccessIssue]


def midi_listener(
    port_name: str,
    midi_queue: "queue.Queue[tuple[list[int], float]]",
    stop_event: threading.Event,
    *,
    port_opener: PortOpener | None = None,
) -> None:
    """Listen on a MIDI input port and push ``(list[int], float)`` tuples onto ``midi_queue``.

    Filters to performance-relevant types (notes, CC, pitch wheel, aftertouch); other
    message types (e.g. ``polytouch``, ``sysex``, ``clock``) are dropped. Each forwarded
    message is converted to ``(msg.bytes(), 0.0)`` so ``plugin.process`` schedules it at the
    start of the next audio buffer — the format used elsewhere in the repo (see
    :func:`src.data.vst.core.make_midi_events`). ``mido.Message.bytes()`` returns
    ``list[int]`` (a sequence of MIDI status bytes), matching the ``List[int]`` form
    accepted by pedalboard's ``plugin.process(...)``.

    Polls ``port.poll()`` non-blockingly so the loop checks ``stop_event`` every
    ``_MIDI_POLL_INTERVAL_SECONDS`` and exits cleanly when ``main`` signals shutdown
    — otherwise the queue would keep growing after the audio thread stops (e.g. while
    ``main`` waits at the post-editor "press any key" prompt).

    ``port_opener`` exists for test injection (#844). ``None`` (the default) resolves to
    ``mido.open_input`` at call time so legacy ``monkeypatch.setattr(surge_xt_interactive.mido,
    "open_input", ...)`` tests keep working until they migrate to direct injection.
    """
    logger.info("Listening on MIDI port: %s", port_name)
    opener = port_opener if port_opener is not None else mido.open_input  # pyright: ignore[reportAttributeAccessIssue]
    try:
        with opener(port_name) as port_handle:
            while not stop_event.is_set():
                msg = port_handle.poll()  # pyright: ignore[reportAttributeAccessIssue]
                if msg is None:
                    time.sleep(_MIDI_POLL_INTERVAL_SECONDS)
                    continue
                if msg.type in MIDI_LISTEN_MESSAGE_TYPES:
                    midi_queue.put((msg.bytes(), 0.0))
    except Exception:
        logger.exception("MIDI listener thread aborted on port %r", port_name)


def _resolve_midi_port(requested: str, available: Sequence[str]) -> str:
    """Map the ``--midi-port`` flag value to a concrete port name.

    ``requested == ""`` selects ``available[0]`` (auto-pick); a non-empty value
    must match one of ``available`` exactly. Either form raises ``click.UsageError``
    when ``available`` is empty or the requested name is not present.
    """
    if not available:
        raise click.UsageError("--midi-port set but no MIDI input ports are available.")
    if requested == "":
        return available[0]
    if requested in available:
        return requested
    raise click.UsageError(f"--midi-port {requested!r} not in available inputs {available!r}")


def _validate_no_drift(
    plugin: VST3Plugin, spec: ParamSpec, default_params: dict[str, float]
) -> None:
    """Raise ``ValueError`` if any non-spec plugin param drifted from its default.

    Only parameters absent from ``spec.synth_param_names`` are checked; spec params
    are expected to vary between recordings.
    """
    for param_name in plugin.parameters:  # pyright: ignore[reportAttributeAccessIssue]
        if param_name in spec.synth_param_names:
            continue
        current = plugin.parameters[param_name].raw_value  # pyright: ignore[reportAttributeAccessIssue]
        default = default_params[param_name]
        if not math.isclose(current, default, abs_tol=_DRIFT_TOLERANCE):
            raise ValueError(
                f"plugin parameter {param_name!r} drifted from default "
                f"{default:.6f} to {current:.6f}; revert it before recording"
            )


def play_audio_recorded(plugin: VST3Plugin, session_recording_path: Path) -> None:
    """Render a deterministic clip through ``plugin`` to ``session_recording_path``.

    The clip is a fixed ``SESSION_RECORDING_DURATION_SECONDS`` window with a
    held middle-C note from ``SESSION_RECORDING_NOTE_START_SECONDS`` to
    ``SESSION_RECORDING_NOTE_END_SECONDS``; the surrounding silence captures
    any release tail. The output depends only on the loaded plugin state
    (preset + ``--pred`` / ``--dataset-ref`` params), so the same inputs
    always produce the same WAV.

    Headless alternative to :func:`play_audio` — uses
    ``pedalboard.io.AudioFile`` instead of an ``AudioStream``, so it works in
    environments without an audio output device.
    """
    midi_events = make_midi_events(
        SESSION_RECORDING_MIDI_NOTE,
        SESSION_RECORDING_VELOCITY,
        SESSION_RECORDING_NOTE_START_SECONDS,
        SESSION_RECORDING_NOTE_END_SECONDS,
    )
    logger.info(
        "Rendering %.1fs deterministic clip (note %d, %.1f-%.1fs) to %s",
        SESSION_RECORDING_DURATION_SECONDS,
        SESSION_RECORDING_MIDI_NOTE,
        SESSION_RECORDING_NOTE_START_SECONDS,
        SESSION_RECORDING_NOTE_END_SECONDS,
        session_recording_path,
    )
    output = plugin.process(
        list(midi_events),
        SESSION_RECORDING_DURATION_SECONDS,
        SAMPLE_RATE,
        CHANNELS,
        _SESSION_RECORDING_BUFFER_SIZE,
        True,
    )
    expected_frames = int(SESSION_RECORDING_DURATION_SECONDS * SAMPLE_RATE)
    if output.shape != (CHANNELS, expected_frames):
        raise ValueError(
            f"expected output shape ({CHANNELS}, {expected_frames}), got {output.shape}"
        )
    # AudioFile.write expects (frames, channels); plugin output is (channels, frames).
    with AudioFile(str(session_recording_path), "w", SAMPLE_RATE, CHANNELS) as f:
        f.write(output.T)
    logger.info("Recording wrote %d frames", expected_frames)


def keyboard_loop(
    plugin: VST3Plugin,
    stop_event: threading.Event,
    param_spec_name: str,
    default_params: dict[str, float],
) -> list[dict[str, float]]:
    """Read keystrokes and snapshot the live plugin params into a list of patches.

    Keys:

    - ``p`` — record the current values of every synth param in
      ``param_specs[param_spec_name]`` as a patch dict. Fails if any non-spec parameter
      has drifted from its post-preset-load default (caught at record time so the user
      can revert before the editor closes).
    - ``q`` — set ``stop_event`` and exit.

    Also exits when ``stop_event`` is set externally (e.g. when the plugin editor is
    closed). Returns the list of recorded patches.
    """
    spec = param_specs[param_spec_name]
    synth_patches: list[dict[str, float]] = []

    def quit_action() -> bool:
        stop_event.set()
        logger.info("Quitting...")
        return _STOP_LOOPING

    def record_patch() -> bool:
        logger.info("Recording patch...")
        _validate_no_drift(plugin, spec, default_params)
        patch: dict[str, float] = {}
        for param_name in spec.synth_param_names:
            if param_name not in plugin.parameters:  # pyright: ignore[reportAttributeAccessIssue]
                raise KeyError(f"parameter {param_name!r} not found in plugin parameters")
            patch[param_name] = plugin.parameters[param_name].raw_value  # pyright: ignore[reportAttributeAccessIssue]
        synth_patches.append(patch)
        logger.info("patch recorded: %s", synth_patches[-1])
        return _KEEP_LOOPING

    actions = {
        "q": quit_action,
        "p": record_patch,
    }
    # NOTE: ``click.getchar()`` blocks until a key is pressed, so this loop only
    # checks ``stop_event`` between keystrokes. If the editor closes from another
    # thread, the loop won't exit until the user presses any key. Acceptable for
    # the typical editor-close-then-press-q flow; a sentinel-key approach would
    # be needed for fully event-driven shutdown.
    while not stop_event.is_set():
        ch = click.getchar()
        action = actions.get(ch)
        if action is None:
            continue
        try:
            if action() == _STOP_LOOPING:
                return synth_patches
        except (KeyError, ValueError):
            # Log now so the traceback isn't buffered until ``main`` calls ``.result()``.
            logger.exception("keyboard action %r failed", ch)
            stop_event.set()
            raise
    return synth_patches


def _run_predict(
    checkpoint_path: Path,
    dataset_root_dir: Path,
    predict_file: Path,
    predictions_output_dir: Path,
    param_spec_name: str,
    *,
    subprocess_runner: SubprocessRunner | None = None,
) -> None:
    """Run model prediction via ``src/eval.py`` with ``mode=predict``.

    Paths are passed as absolute (``.resolve()``) because ``src/eval.py`` runs under Hydra, which
    chdirs into its own output dir before the job starts; relative paths would otherwise resolve
    against the wrong cwd. ``model.net.d_out`` is overridden from ``len(param_specs[...])`` to
    satisfy the mandatory-override sentinel in ``configs/experiment/surge/test.yaml``.

    ``subprocess_runner`` exists for test injection (#844). ``None`` (the default) resolves to
    ``subprocess.check_call`` at call time so legacy ``monkeypatch.setattr(subprocess, ...)``
    tests keep working until they migrate to direct injection.
    """
    runner = subprocess_runner if subprocess_runner is not None else subprocess.check_call
    encoded_width = len(param_specs[param_spec_name])
    runner(  # noqa: S603
        [
            sys.executable,
            str(_EVAL_SCRIPT),
            "experiment=surge/test",
            "ckpt_path=" + str(checkpoint_path.resolve()),
            "data.predict_file=" + str(predict_file.resolve()),
            "data.dataset_root=" + str(dataset_root_dir.resolve()),
            "callbacks.prediction_writer.output_dir=" + str(predictions_output_dir.resolve()),
            f"model.net.d_out={encoded_width}",
            "mode=predict",
        ],
        timeout=_EVAL_SUBPROCESS_TIMEOUT_SECONDS,
    )


def _validate_predictions(predictions_output_dir: Path, num_samples: int) -> None:
    """Verify ``PredictionWriter`` (``src/utils/callbacks.py``) wrote the expected per-sample
    ``pred-{i}.pt``, ``target-audio-{i}.pt``, and ``target-params-{i}.pt`` files, and that
    prediction tensors are finite.

    Tensors are loaded onto CPU regardless of the device they were saved from so this works across
    mps/cuda/cpu predict runs.

    :raises FileNotFoundError: if the expected files are missing or extras are present.
    :raises ValueError: if any ``pred-{i}.pt`` tensor contains NaN/Inf.
    """
    expected_names = _expected_prediction_filenames(num_samples)
    actual_names = sorted(p.name for p in predictions_output_dir.iterdir())
    if actual_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_names))
        unexpected = sorted(set(actual_names) - set(expected_names))
        raise FileNotFoundError(
            f"unexpected prediction outputs in {predictions_output_dir}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for i in range(num_samples):
        pred_path = predictions_output_dir / f"pred-{i}.pt"
        pred = torch.load(pred_path, map_location="cpu", weights_only=True)
        if not torch.isfinite(pred).all():
            raise ValueError(f"{pred_path} contains NaN/Inf")


def _render_predicted_audio(
    predictions_output_dir: Path,
    audio_dir: Path,
    num_samples: int,
    param_spec_name: str,
    preset_path: str,
    *,
    subprocess_runner: SubprocessRunner | None = None,
) -> None:
    """Render audio for the predicted patches and validate per-sample outputs (file presence and
    non-silent WAVs).

    ``param_spec_name`` and ``preset_path`` must match the values used to capture the patches —
    otherwise ``predict_vst_audio.py`` would fall back to its own defaults (``surge_xt`` /
    ``presets/surge-base.vstpreset``) and decode/render against a mismatched spec.

    :raises FileNotFoundError: if the headless wrapper is missing on Linux, if any sample directory
        is missing, or if a per-sample artifact (target.wav, pred.wav, spec.png, params.csv) is
        absent.
    :raises ValueError: if a rendered WAV's peak amplitude is below ``SILENCE_PEAK_THRESHOLD``.
    """
    args: list[str] = []
    if sys.platform == "linux":
        if not _VST_HEADLESS_WRAPPER.is_file():
            raise FileNotFoundError(
                f"VST headless wrapper not found at {_VST_HEADLESS_WRAPPER}; "
                f"this script needs it on Linux to run predict_vst_audio.py headlessly."
            )
        args.append(str(_VST_HEADLESS_WRAPPER))
    args += [
        sys.executable,
        str(_PREDICT_VST_AUDIO_SCRIPT),
        str(predictions_output_dir),
        str(audio_dir),
        "--param_spec",
        param_spec_name,
        "--preset_path",
        preset_path,
        "-t",
    ]
    runner = subprocess_runner if subprocess_runner is not None else subprocess.run
    try:
        runner(  # noqa: S603
            args,
            check=True,
            timeout=_VST_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        logger.error(
            "predict_vst_audio timed out after %ss; command: %s",
            _VST_SUBPROCESS_TIMEOUT_SECONDS,
            args,
        )
        raise

    sample_dirs = sorted(d for d in audio_dir.iterdir() if d.is_dir())
    actual_sample_names = [d.name for d in sample_dirs]
    expected_sample_names = [f"sample_{i}" for i in range(num_samples)]
    if actual_sample_names != expected_sample_names:
        raise FileNotFoundError(
            f"unexpected sample directories in {audio_dir}: "
            f"got {actual_sample_names}, expected {expected_sample_names}"
        )
    for sample_dir in sample_dirs:
        for fname in ("target.wav", "pred.wav", "spec.png", "params.csv"):
            artifact_path = sample_dir / fname
            if not artifact_path.is_file():
                raise FileNotFoundError(f"{artifact_path} not found")
        for wav_name in ("target.wav", "pred.wav"):
            with AudioFile(str(sample_dir / wav_name)) as f:
                audio = f.read(f.frames)
            peak = float(np.abs(audio).max())
            if peak <= SILENCE_PEAK_THRESHOLD:
                raise ValueError(f"{sample_dir.name}/{wav_name} is silent (peak={peak:.2e})")


def _compute_and_validate_metrics(
    audio_dir: Path,
    metrics_dir: Path,
    num_samples: int,
    *,
    subprocess_runner: SubprocessRunner | None = None,
) -> None:
    """Compute MSS / wMFCC / SOT / RMS metrics on the rendered pairs and verify the CSV outputs.

    ``subprocess_runner`` exists for test injection (#844). ``None`` (the default) resolves to
    ``subprocess.check_call`` at call time so legacy module-level monkeypatches keep working.

    :raises FileNotFoundError: if either metrics CSV is missing.
    :raises ValueError: if a metrics CSV has unexpected shape or columns, or contains NaN/Inf.
    """
    metrics_file_expectations: dict[str, _MetricsFileSpec] = {
        "aggregated_metrics.csv": _MetricsFileSpec(
            rows=len(_METRIC_COLUMNS), columns=frozenset({"mean", "std"})
        ),
        "metrics.csv": _MetricsFileSpec(rows=num_samples, columns=_METRIC_COLUMNS),
    }
    runner = subprocess_runner if subprocess_runner is not None else subprocess.check_call
    runner(  # noqa: S603
        [
            sys.executable,
            str(_COMPUTE_AUDIO_METRICS_SCRIPT),
            str(audio_dir),
            str(metrics_dir),
            "-w",
            "1",
        ],
        timeout=_METRICS_SUBPROCESS_TIMEOUT_SECONDS,
    )
    for metrics_file, expected in metrics_file_expectations.items():
        metrics_path = metrics_dir / metrics_file
        if not metrics_path.is_file():
            raise FileNotFoundError(f"{metrics_file} not found in {metrics_dir}")
        metrics_df = pd.read_csv(metrics_path)
        _validate_metrics_df(metrics_path, metrics_df, expected)


def eval_patches(
    num_samples: int,
    dataset_root_dir: Path,
    checkpoint_path: Path,
    param_spec_name: str,
    preset_path: str,
    *,
    subprocess_runner: SubprocessRunner | None = None,
) -> None:
    """Run model eval on captured patches end-to-end.

    Pipeline (each step gated on the previous one's success):

    1. Predict params via ``src/eval.py mode=predict`` (``_run_predict``).
    2. Verify the per-sample ``pred-{i}.pt`` / ``target-audio-{i}.pt`` / ``target-params-{i}.pt``
       files were written and that prediction tensors are finite (``_validate_predictions``).
    3. Render predicted vs. target audio via ``scripts/predict_vst_audio.py`` and verify the
       per-sample artifacts (``_render_predicted_audio``).
    4. Compute MSS / wMFCC / SOT / RMS metrics via ``scripts/compute_audio_metrics.py`` and verify
       the resulting CSVs (``_compute_and_validate_metrics``).

    Re-runs against the same ``dataset_root_dir`` clear the previous prediction/audio/metrics
    output directories so stale files don't leak into validation.

    :param num_samples: Number of patches predicted/rendered (matches ``predict.h5``'s row count).
    :param dataset_root_dir: Directory containing ``predict.h5``; receives ``prediction_outputs/``,
        ``audio/``, and ``metrics/`` subdirectories.
    :param checkpoint_path: Path to the ``.ckpt`` file to load weights from.
    :param param_spec_name: Parameter spec name (key into ``param_specs``) used to set the model's
        ``d_out`` and the decoder used to render predicted audio. Must match the spec used when
        the patches were captured.
    :param preset_path: Base preset to load when rendering predicted audio. Must match the preset
        used when the patches were captured.
    :param subprocess_runner: Test seam (#844) — when set, forwarded to every subprocess-using
        helper so a single fake records all three external invocations. ``None`` (the default)
        preserves production behavior by letting each helper bind its own
        ``subprocess.run``/``subprocess.check_call`` default.
    :raises FileNotFoundError: if ``checkpoint_path`` is missing, ``dataset_root_dir`` is not a
        directory, ``predict.h5`` is missing, or any expected pipeline output is absent.
    :raises ValueError: if predictions contain NaN/Inf, a rendered WAV is silent, or a metrics CSV
        has the wrong shape/columns or contains NaN/Inf.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not dataset_root_dir.is_dir():
        raise NotADirectoryError(f"dataset root directory not found: {dataset_root_dir}")
    predict_file = dataset_root_dir / "predict.h5"
    if not predict_file.is_file():
        raise FileNotFoundError(f"predict.h5 not found: {predict_file}")

    predictions_output_dir = dataset_root_dir / "prediction_outputs"
    audio_dir = dataset_root_dir / "audio"
    metrics_dir = dataset_root_dir / "metrics"

    # Stale outputs from a prior run would leak into validation (e.g. extra ``pred-N.pt`` files
    # from a larger previous run). Clear and recreate per-run output dirs up front.
    for output_dir in (predictions_output_dir, audio_dir, metrics_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

    runner_kwargs: dict[str, SubprocessRunner] = (
        {"subprocess_runner": subprocess_runner} if subprocess_runner is not None else {}
    )
    _run_predict(
        checkpoint_path,
        dataset_root_dir,
        predict_file,
        predictions_output_dir,
        param_spec_name,
        **runner_kwargs,
    )
    _validate_predictions(predictions_output_dir, num_samples)
    _render_predicted_audio(
        predictions_output_dir,
        audio_dir,
        num_samples,
        param_spec_name,
        preset_path,
        **runner_kwargs,
    )
    _compute_and_validate_metrics(audio_dir, metrics_dir, num_samples, **runner_kwargs)


@click.command()
@click.option("--plugin-path", "-p", default="plugins/Surge XT.vst3", help="Path to VST3 plugin.")
@click.option(
    "--pred",
    type=PredictionRefType(),
    default=None,
    help=(
        "Prediction reference as PATH:BATCH_IDX (e.g. 'outputs/pred-0.pt:0'). "
        "When set, the predicted row is decoded and applied to the plugin "
        "before the editor opens."
    ),
)
@click.option(
    "--dataset-ref",
    type=DatasetRefType(),
    default=None,
    help=(
        "Dataset reference as PATH:DATASET_IDX (e.g. 'outputs/test.h5:0'). "
        "When set, the dataset row is decoded and applied to the plugin "
        "before the editor opens."
    ),
)
@click.option(
    "--preset-path",
    "-r",
    type=str,
    default="presets/surge-base.vstpreset",
    help="Base preset to load before applying any --pred / --dataset-ref params.",
)
@click.option(
    "--param-spec-name",
    type=str,
    default="surge_xt",
    help=(
        "Parameter spec name (key into ``param_specs``) used to decode prediction/dataset "
        "rows applied to the plugin and to enumerate which synth params are captured when "
        "recording patches."
    ),
)
@click.option(
    "--output-dataset-dir-path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help=(
        "Directory to create for the recorded patches. Must not already exist — "
        "``make_dataset`` writes fixed-size HDF5 datasets without ``maxshape`` and cannot "
        "append to existing files. After the editor is closed, patches captured via the "
        "keyboard loop (press 'p' to record, 'q' to quit) are rendered through the plugin "
        "and written to ``train.h5`` inside this directory via "
        "``src.data.vst.generate_vst_dataset.make_dataset`` (plus ``val.h5``/``test.h5``/"
        "``predict.h5`` siblings when ``--checkpoint-path`` is set)."
    ),
)
@click.option(
    "--midi-port",
    type=str,
    default=None,
    help=(
        "Name of a MIDI input port to listen on (matched against ``mido.get_input_names()``). "
        "When set, a daemon thread forwards MIDI note/CC/pitch-wheel/aftertouch messages to "
        "the plugin while the editor is open. Pass an empty string ``''`` to auto-select the "
        "first available input. When unset, no MIDI input is opened. List ports with "
        '``python -c "import mido; print(mido.get_input_names())"``.'
    ),
)
@click.option(
    "--checkpoint-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=("Optional checkpoint path to run standalone eval on after rendering captured patches. "),
)
@click.option(
    "--session-recording-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        f"Optional WAV file to render a deterministic "
        f"{int(SESSION_RECORDING_DURATION_SECONDS)}s test clip to "
        f"(middle C from {SESSION_RECORDING_NOTE_START_SECONDS:.1f}-"
        f"{SESSION_RECORDING_NOTE_END_SECONDS:.1f}s). When set, replaces the live "
        f"audio stream — output depends only on plugin state, so the same inputs "
        f"always produce the same WAV. No-op when not set."
    ),
)
def main(
    plugin_path: str,
    pred: PredictionRef | None,
    dataset_ref: DatasetRef | None,
    preset_path: str,
    param_spec_name: str,
    output_dataset_dir_path: Path | None,
    midi_port: str | None,
    checkpoint_path: Path | None,
    session_recording_path: Path | None,
) -> None:
    """Open Surge XT GUI with real-time audio streaming and record patches to an HDF5 dataset.

    Flow:

    1. Load the plugin and base preset.
    2. If ``--pred`` or ``--dataset-ref`` is provided, decode the referenced row and apply
       it to the plugin before the editor opens.
    3. Open the plugin's GUI editor; in parallel, stream audio to the default output device
       and run a keyboard loop (press ``p`` to snapshot the current synth params as a patch,
       ``q`` to quit).
    4. After the editor is closed, render every recorded patch through the plugin and write
       the resulting samples to ``train.h5`` inside ``--output-dataset-dir-path`` via
       ``make_dataset``.
    5. If ``--checkpoint-path`` is also set, copy ``train.h5`` to ``val.h5``/``test.h5``/
       ``predict.h5`` siblings (rolled back if any copy fails) and call ``eval_patches`` to
       run ``src/eval.py mode=predict`` followed by audio rendering
       (``predict_vst_audio.py``) and metric computation (``compute_audio_metrics.py``) on
       the captured patches.
    """
    if dataset_ref is not None and pred is not None:
        raise click.UsageError(
            "--pred and --dataset-ref are mutually exclusive; pass at most one."
        )

    # ``--session-recording-path`` skips ``play_audio``, so a MIDI listener would enqueue
    # forever with no consumer. Reject the combination up front.
    if session_recording_path is not None and midi_port is not None:
        raise click.UsageError(
            "--midi-port and --session-recording-path are mutually exclusive; "
            "the live audio thread doesn't run during deterministic clip rendering."
        )

    # Fail fast — ``make_dataset`` writes fixed-size HDF5 datasets without
    # ``maxshape`` and cannot append, so a pre-existing path would either
    # silently overwrite (when re-creating datasets) or fail mid-render after
    # patches have been captured. Better to reject up front.
    if output_dataset_dir_path is not None and output_dataset_dir_path.exists():
        raise click.UsageError(
            f"--output-dataset-dir-path {output_dataset_dir_path} already exists; "
            f"this script creates a new directory (fixed-size HDF5 datasets cannot be "
            f"appended to). Choose a path that does not exist yet."
        )

    plugin = load_plugin(plugin_path)
    if not isinstance(plugin, VST3Plugin):
        raise TypeError(f"expected VST3Plugin, got {type(plugin).__name__}")

    load_preset(plugin, preset_path)

    # Two flushes ensure the plugin's full parameter dict is populated and any
    # transient state from preset load is cleared before applying user params.
    _flush_plugin(plugin)
    _flush_plugin(plugin)

    # Snapshot the post-preset-flush parameter values so ``record_patch`` can detect drift
    # in any non-spec parameter at record time.
    default_params: dict[str, float] = {
        name: plugin.parameters[name].raw_value  # pyright: ignore[reportAttributeAccessIssue]
        for name in plugin.parameters  # pyright: ignore[reportAttributeAccessIssue]
    }

    if dataset_ref is not None:
        synth_params = load_dataset_synth_params(dataset_ref, param_spec_name)
        set_params(plugin, synth_params)
    elif pred is not None:
        synth_params = load_prediction_synth_params(pred, param_spec_name)
        set_params(plugin, synth_params)

    # Render the deterministic clip before opening the editor so the WAV depends
    # only on the initially-loaded plugin state. Running it concurrently with
    # show_editor would let the user twist knobs mid-render and break that
    # guarantee.
    if session_recording_path is not None:
        play_audio_recorded(plugin, session_recording_path)

    # Created up front so ``midi_listener`` can observe shutdown the moment ``main``
    # sets it (the same event also drives ``play_audio``, ``keyboard_loop``, and the
    # plugin editor's blocking close signal).
    stop_event = threading.Event()

    midi_queue: queue.Queue[tuple[list[int], float]] | None = None
    if midi_port is not None:
        resolved_port = _resolve_midi_port(
            midi_port,
            mido.get_input_names(),  # pyright: ignore[reportAttributeAccessIssue]
        )
        if midi_port == "":
            logger.info("--midi-port='' — auto-selected first input: %s", resolved_port)
        midi_queue = queue.Queue()
        threading.Thread(
            target=midi_listener,
            args=(resolved_port, midi_queue, stop_event),
            daemon=True,
        ).start()

    pool = ThreadPoolExecutor()
    audio_timed_out = False
    try:
        audio_future = (
            pool.submit(play_audio, plugin, stop_event, midi_queue)
            if session_recording_path is None
            else None
        )
        keyboard_future = pool.submit(
            keyboard_loop, plugin, stop_event, param_spec_name, default_params
        )

        try:
            plugin.show_editor(stop_event)
        finally:
            stop_event.set()
            # Surface any exception from the audio thread. Catch TimeoutError so
            # a slow stream-close on shutdown doesn't crash the script — log it
            # and skip the wait-for-completion in pool teardown to avoid hanging.
            if audio_future is not None:
                try:
                    audio_future.result(timeout=AUDIO_THREAD_DRAIN_TIMEOUT_SECONDS)
                except TimeoutError:
                    audio_timed_out = True
                    logger.warning(
                        "audio thread did not exit within %ss; "
                        "cancelling pool and continuing shutdown",
                        AUDIO_THREAD_DRAIN_TIMEOUT_SECONDS,
                    )
        logger.info(
            "Editor closed; press any key in this terminal to finish "
            "(captured patches will then be rendered)..."
        )
        synth_patches = keyboard_future.result()
        logger.info("Recorded %d patches: %s", len(synth_patches), synth_patches)
    finally:
        # When the audio thread timed out, don't let pool teardown block waiting
        # on it again — cancel pending work and return immediately. Otherwise
        # wait for any remaining work to drain normally.
        pool.shutdown(wait=not audio_timed_out, cancel_futures=audio_timed_out)

    if output_dataset_dir_path is None:
        logger.info("No --output-dataset-dir-path provided; skipping dataset creation.")
        return
    if not synth_patches:
        logger.info("No patches recorded, skipping dataset creation.")
        return
    output_dataset_dir_path.mkdir(parents=True, exist_ok=False)
    patch_file_path = output_dataset_dir_path / "train.h5"
    with h5py.File(patch_file_path, "w") as f:
        make_dataset(
            hdf5_file=f,
            num_samples=len(synth_patches),
            plugin_path=plugin_path,
            preset_path=preset_path,
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            velocity=MAKE_DATASET_VELOCITY,
            signal_duration_seconds=MAKE_DATASET_SIGNAL_DURATION_SECONDS,
            min_loudness=MAKE_DATASET_MIN_LOUDNESS,
            param_spec=param_specs[param_spec_name],
            sample_batch_size=MAKE_DATASET_SAMPLE_BATCH_SIZE,
            fixed_synth_params_list=synth_patches,
        )
    _maybe_eval_captured_patches(
        patch_file_path,
        output_dataset_dir_path,
        len(synth_patches),
        checkpoint_path,
        param_spec_name,
        preset_path,
    )


EvalRunner = Callable[[int, Path, Path, str, str], None]


def _maybe_eval_captured_patches(
    patch_file_path: Path,
    output_dataset_dir_path: Path,
    num_patches: int,
    checkpoint_path: Path | None,
    param_spec_name: str,
    preset_path: str,
    *,
    eval_runner: EvalRunner | None = None,
) -> None:
    """Replicate captured patches into the four eval-pipeline splits and run eval_patches if a
    checkpoint is provided; no-op otherwise.

    The Click ``--checkpoint-path`` option already validates ``exists=True``, so when this is
    invoked from ``main`` ``checkpoint_path`` is guaranteed to refer to an existing file.
    ``param_spec_name`` and ``preset_path`` are forwarded to ``eval_patches`` so the predict /
    render / metrics steps decode and re-render against the same spec + preset that were used
    when the patches were captured.

    ``eval_runner`` exists for test injection (#844). ``None`` (the default) resolves to the
    module-level :func:`eval_patches` at call time so legacy ``monkeypatch.setattr(module,
    "eval_patches", ...)`` tests keep working until they migrate to direct injection.
    """
    if checkpoint_path is None:
        logger.info("No --checkpoint-path provided; skipping patch evaluation.")
        return
    runner = eval_runner if eval_runner is not None else eval_patches
    sibling_paths = [
        output_dataset_dir_path / name for name in ("test.h5", "val.h5", "predict.h5")
    ]
    copied: list[Path] = []
    try:
        for sibling_path in sibling_paths:
            shutil.copyfile(patch_file_path, sibling_path)
            copied.append(sibling_path)
    except OSError:
        for created in copied:
            try:
                os.remove(created)
            except OSError:
                logger.exception("failed to roll back partial sibling copy at %s", created)
        raise
    runner(num_patches, output_dataset_dir_path, checkpoint_path, param_spec_name, preset_path)


if __name__ == "__main__":
    _configure_logging()
    main()  # type: ignore[call-arg]
