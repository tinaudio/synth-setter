"""Interactive Surge XT preview with real-time audio streaming via pedalboard."""

import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

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
from src.data.vst.core import set_params  # noqa: E402
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
PLAYBACK_SAMPLE_RATE = SAMPLE_RATE

# Maximum time to wait for the audio thread to drain after ``stop_event`` is set.
AUDIO_THREAD_DRAIN_TIMEOUT_SECONDS = 2

# Plugin-flush parameters used by the post-load / pre-render flush pattern; see
# ``_flush_plugin``. Mirror of the values used in ``src.data.vst.core.render_params``.
_PLUGIN_FLUSH_DURATION_SECONDS = 32.0
_PLUGIN_FLUSH_BUFFER_SIZE = 2048

# Return signals for ``keyboard_loop`` actions — distinguishes "user requested
# quit" from "action completed, keep listening".
_KEEP_LOOPING = True
_STOP_LOOPING = False


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
        return DatasetRef(path=Path(path_str), batch_idx=batch_idx)


def render_to_wav(plugin: VST3Plugin, output_path: Path, duration_seconds: float) -> None:
    """Render synthesized audio through ``plugin`` to a WAV file at ``output_path``.

    Offline alternative to :func:`play_audio` for headless environments (e.g. Docker
    on Linux without ALSA/PulseAudio) where ``AudioStream`` cannot open a device.

    :param plugin: Loaded VST3 plugin (post preset/parameter setup).
    :param output_path: WAV destination. Parent directory must exist.
    :param duration_seconds: Seconds of audio to render. The actual length is
        rounded up to the nearest ``BUFFER_SIZE`` samples.
    """
    silence = np.zeros((CHANNELS, BUFFER_SIZE), dtype=np.float32)
    num_buffers = math.ceil(duration_seconds * SAMPLE_RATE / BUFFER_SIZE)
    with AudioFile(str(output_path), "w", SAMPLE_RATE, CHANNELS) as f:
        for _ in range(num_buffers):
            synth_output = plugin(silence, SAMPLE_RATE, reset=False)
            f.write(synth_output)


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
        sample_rate=SAMPLE_RATE,
        buffer_size=BUFFER_SIZE,
    ) as stream:
        while not stop_event.is_set():
            synth_output = plugin(silence, SAMPLE_RATE, reset=False)
            if stream_resampler is not None:
                stream.write(stream_resampler.process(synth_output), PLAYBACK_SAMPLE_RATE)
            else:
                stream.write(synth_output, SAMPLE_RATE)


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
    "--output-dataset-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        "HDF5 file to be created with the recorded patches. Must not already exist — "
        "``make_dataset`` writes fixed-size HDF5 datasets without ``maxshape`` and cannot "
        "append to existing files. After the editor is closed, patches captured via the "
        "keyboard loop (press 'p' to record, 'q' to quit) are rendered through the plugin "
        "and written to this file via ``src.data.vst.generate_vst_dataset.make_dataset``."
    ),
)
def main(
    plugin_path: str,
    pred: PredictionRef | None,
    dataset_ref: DatasetRef | None,
    preset_path: str,
    param_spec_name: str,
    output_dataset_path: Path | None,
) -> None:
    """Open Surge XT GUI with real-time audio streaming and record patches to an HDF5 dataset.

    Flow:

    1. Load the plugin and base preset.
    2. If ``--pred`` or ``--dataset-ref`` is provided, decode the referenced row and apply
       it to the plugin before the editor opens.
    3. Open the plugin's GUI editor; in parallel, stream audio to the default output device
       and run a keyboard loop (press ``p`` to snapshot the current synth params as a patch,
       ``q`` to quit).
    4. After the editor is closed, render every recorded patch through the plugin and append
       the resulting samples to ``--output-dataset-path`` via ``make_dataset``.
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
    if output_dataset_path is not None and output_dataset_path.exists():
        raise click.UsageError(
            f"--output-dataset-path {output_dataset_path} already exists; "
            f"this script writes a new file (fixed-size HDF5 datasets cannot be "
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

    stop_event = threading.Event()
    with ThreadPoolExecutor() as pool:
        audio_future = pool.submit(play_audio, plugin, stop_event)
        keyboard_future = pool.submit(keyboard_loop, plugin, stop_event, param_spec_name)

        try:
            plugin.show_editor(stop_event)
        finally:
            stop_event.set()
            # Surface any exception from the audio thread. Catch TimeoutError so
            # a slow stream-close on shutdown doesn't crash the script — log it
            # and continue draining; cancel-style errors will resurface elsewhere.
            try:
                audio_future.result(timeout=AUDIO_THREAD_DRAIN_TIMEOUT_SECONDS)
            except TimeoutError:
                logger.warning(
                    "audio thread did not exit within %ss; continuing shutdown",
                    AUDIO_THREAD_DRAIN_TIMEOUT_SECONDS,
                )
        logger.info("Editor closed, waiting for keyboard thread to finish...")
        synth_patches = keyboard_future.result()
        logger.info("Recorded %d patches: %s", len(synth_patches), synth_patches)

    if output_dataset_path is None:
        logger.info("No output dataset path provided, skipping dataset creation.")
        return
    if not synth_patches:
        logger.info("No patches recorded, skipping dataset creation.")
        return
    with h5py.File(output_dataset_path, "w") as f:
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


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
