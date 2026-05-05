"""Interactive Surge XT preview with real-time audio streaming via pedalboard."""

import logging
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import click  # noqa: E402
import h5py  # noqa: E402
import hdf5plugin  # noqa: F401, E402  side-effect: registers HDF5_PLUGIN_PATH for Blosc2 filters
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pedalboard import VST3Plugin  # noqa: E402
from pedalboard.io import AudioFile, AudioStream, StreamResampler  # noqa: E402

from src.data.vst import load_plugin, load_preset, param_specs  # noqa: E402
from src.data.vst.core import make_midi_events, set_params  # noqa: E402
from src.data.vst.generate_vst_dataset import make_dataset  # noqa: E402

logger = logging.getLogger(__name__)

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

_VST_SUBPROCESS_TIMEOUT_SECONDS = 300
VST_HEADLESS_WRAPPER = "scripts/run-linux-vst-headless.sh"


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


def play_audio(plugin: VST3Plugin, stop_event: threading.Event) -> None:
    """Stream silence through Surge XT and write synthesized audio to the output device.

    Runs until ``stop_event`` is set (typically by ``keyboard_loop``'s quit action or by
    ``main`` after the plugin editor is closed). Resamples to ``PLAYBACK_SAMPLE_RATE`` if
    it differs from ``SAMPLE_RATE`` so the output device gets a rate it supports.
    """
    silence = np.zeros((CHANNELS, BUFFER_SIZE), dtype=np.float32)
    needs_resample = SAMPLE_RATE != PLAYBACK_SAMPLE_RATE
    stream_resampler = (
        StreamResampler(SAMPLE_RATE, PLAYBACK_SAMPLE_RATE, CHANNELS) if needs_resample else None
    )
    with AudioStream(
        output_device_name=AudioStream.default_output_device_name,
        sample_rate=PLAYBACK_SAMPLE_RATE,
        buffer_size=BUFFER_SIZE,
    ) as stream:
        while not stop_event.is_set():
            synth_output = plugin(silence, SAMPLE_RATE, reset=False)
            if synth_output.shape != (CHANNELS, BUFFER_SIZE):
                raise ValueError(
                    f"expected synth output shape ({CHANNELS}, {BUFFER_SIZE}), "
                    f"got {synth_output.shape}"
                )
            if stream_resampler is not None:
                stream.write(stream_resampler.process(synth_output), PLAYBACK_SAMPLE_RATE)
            else:
                stream.write(synth_output, SAMPLE_RATE)


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
    plugin: VST3Plugin, stop_event: threading.Event, param_spec_name: str
) -> list[dict[str, float]]:
    """Read keystrokes and snapshot the live plugin params into a list of patches.

    Keys:

    - ``p`` — record the current values of every synth param in
      ``param_specs[param_spec_name]`` as a patch dict.
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
        patch: dict[str, float] = {}
        logger.info("Recording patch...")
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
        if action is not None and action() == _STOP_LOOPING:
            return synth_patches
    return synth_patches


def eval_patches(num_samples: int, dataset_root_dir: Path, checkpoint_path: Path) -> None:
    """Render the captured patches through the plugin and optionally run eval on them."""
    NUM_AUDIO_METRICS = 4  # mss, wmfcc, sot, rms
    METRICS_FILE_EXPECTATIONS = {
        "aggregated_metrics.csv": {
            "rows": NUM_AUDIO_METRICS,
            "columns": {"mean", "std"},
        },
        "metrics.csv": {
            "rows": num_samples,
            "columns": {"mss", "wmfcc", "sot", "rms"},
        },
    }

    predict_file = dataset_root_dir / "predict.h5"
    # base_dir = Path("data/junk3")
    # checkpoint_path = base_dir / "junk.ckpt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not dataset_root_dir.is_dir():
        raise NotADirectoryError(f"dataset root directory not found: {dataset_root_dir}")
    if not (dataset_root_dir / "predict.h5").is_file():
        raise FileNotFoundError(
            f"predict.h5 not found in dataset root directory: {dataset_root_dir}"
        )
    if not predict_file.is_file():
        raise FileNotFoundError(f"predict.h5 not found: {predict_file}")

    audio_dir = dataset_root_dir / "audio"
    metrics_dir = dataset_root_dir / "metrics"
    predictions_output_dir = dataset_root_dir / "prediction_outputs"

    assert checkpoint_path.is_file(), f"checkpoint not found: {checkpoint_path}"
    # evaluate(cfg_surge_xt_eval)
    subprocess.check_call(  # noqa: S603 — args built from validated spec
        [
            sys.executable,
            "src/eval.py",
            "experiment=surge/test",
            "ckpt_path=" + str(checkpoint_path),
            "data.predict_file=" + str(predict_file),
            "data.dataset_root=" + str(dataset_root_dir),
            "callbacks.prediction_writer.output_dir=" + str(predictions_output_dir),
            "mode=predict",
        ],
    )

    # `PredictionWriter` (in `src/utils/callbacks.py`) with `write_interval=batch` saves three
    # tensors per predict batch: `pred-{i}.pt`, `target-audio-{i}.pt`, `target-params-{i}.pt`.
    expected_names = sorted(
        f"{prefix}-{i}.pt"
        for prefix in ("pred", "target-audio", "target-params")
        for i in range(num_samples)
    )
    assert sorted(p.name for p in predictions_output_dir.iterdir()) == expected_names

    for i in range(num_samples):
        pred = torch.load(predictions_output_dir / f"pred-{i}.pt", weights_only=True)
        assert torch.isfinite(pred).all(), f"pred-{i}.pt contains NaN/Inf"

    args = []
    if sys.platform == "linux":
        args.append(VST_HEADLESS_WRAPPER)

    args += [
        sys.executable,
        "scripts/predict_vst_audio.py",
        str(predictions_output_dir),
        str(audio_dir),
        "-t",
    ]
    try:
        result = subprocess.run(  # noqa: S603, S607
            args,
            text=True,
            check=False,
            timeout=_VST_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        logger.error(
            f"predict_vst_audio timed out after {_VST_SUBPROCESS_TIMEOUT_SECONDS}s\n"
            f"command: {args}\n"
            f"(child stdout/stderr printed above; rerun with `pytest -s` if captured)",
        )
        raise
    if result.returncode != 0:
        logger.error(
            f"predict_vst_audio failed (exit {result.returncode})\n"
            f"command: {args}\n"
            f"(child stdout/stderr printed above; rerun with `pytest -s` if captured)",
        )

    sample_dirs = sorted(d for d in audio_dir.iterdir() if d.is_dir())
    assert [d.name for d in sample_dirs] == [f"sample_{i}" for i in range(num_samples)]
    # ~-80 dBFS — below this, librosa RMS norms underflow and `compute_rms`
    # produces 0/0 → NaN (see `compute_rms` in `scripts/compute_audio_metrics.py`).
    SILENCE_PEAK_THRESHOLD = 1e-4
    for sample_dir in sample_dirs:
        assert (sample_dir / "target.wav").is_file()
        assert (sample_dir / "pred.wav").is_file()
        assert (sample_dir / "spec.png").is_file()
        assert (sample_dir / "params.csv").is_file()

        for wav_name in ("target.wav", "pred.wav"):
            with AudioFile(str(sample_dir / wav_name)) as f:
                audio = f.read(f.frames)
            peak = float(np.abs(audio).max())
            assert peak > SILENCE_PEAK_THRESHOLD, (
                f"{sample_dir.name}/{wav_name} is silent (peak={peak:.2e})"
            )

    # Compute audio distance metrics (MSS, wMFCC, SOT, RMS) on the rendered pairs.
    subprocess.check_call(  # noqa: S603 — args built from validated spec
        [
            sys.executable,
            "scripts/compute_audio_metrics.py",
            str(audio_dir),
            str(metrics_dir),
            "-w",
            "1",
        ],
    )

    for metrics_file, expected in METRICS_FILE_EXPECTATIONS.items():
        assert (metrics_dir / metrics_file).is_file(), f"{metrics_file} not found in {metrics_dir}"
        metrics_df = pd.read_csv(metrics_dir / metrics_file)
        assert len(metrics_df) == expected["rows"]
        assert expected["columns"].issubset(metrics_df.columns)
        numeric = metrics_df[sorted(expected["columns"])].to_numpy()
        assert np.isfinite(numeric).all(), f"{metrics_file} contains NaN/Inf:\n{metrics_df}"


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
    "--checkpoint-path",
    type=click.Path(dir_okay=False, path_type=Path),
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
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if dataset_ref is not None and pred is not None:
        raise click.UsageError(
            "--pred and --dataset-ref are mutually exclusive; pass at most one."
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

    stop_event = threading.Event()
    pool = ThreadPoolExecutor()
    audio_timed_out = False
    try:
        audio_future = (
            pool.submit(play_audio, plugin, stop_event) if session_recording_path is None else None
        )
        keyboard_future = pool.submit(keyboard_loop, plugin, stop_event, param_spec_name)

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
        logger.info("No output dataset path provided, skipping dataset creation.")
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
    if checkpoint_path is None:
        logger.info("No checkpoint path provided, skipping patch evaluation.")
        return
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    shutil.copyfile(patch_file_path, output_dataset_dir_path / "test.h5")
    shutil.copyfile(patch_file_path, output_dataset_dir_path / "val.h5")
    shutil.copyfile(patch_file_path, output_dataset_dir_path / "predict.h5")
    eval_patches(len(synth_patches), output_dataset_dir_path, checkpoint_path)


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
