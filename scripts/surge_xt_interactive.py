"""Interactive Surge XT preview with real-time audio streaming via pedalboard."""

import math
import threading
from dataclasses import dataclass
from pathlib import Path

import click
import h5py
import hdf5plugin  # noqa: F401   side-effect import: registers HDF5_PLUGIN_PATH so h5py can load Blosc2 filters in fixtures
import numpy as np
import rootutils
import torch
from pedalboard import VST3Plugin
from pedalboard.io import AudioFile, AudioStream, StreamResampler
from rich import print

from src.data.vst import load_plugin, load_preset, param_specs
from src.data.vst.core import set_params

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


CHANNELS = 2
SAMPLE_RATE = 44100
BUFFER_SIZE = 512

PLAYBACK_SAMPLE_RATE = (
    44100  # Sample rate to use for real-time playback. Must be supported by the output device.
)


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
    row = pred_tensor[batch_idx].float().numpy()
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
        assert isinstance(param_array, h5py.Dataset), "expected h5py.Dataset for 'param_array'"
        print(
            f"Loaded param_array from {ref.path}: shape={param_array.shape}, dtype={param_array.dtype}"
        )
        print(f"param array keys: {list(f.keys())}")
        print(f"param_array : {param_array}")
        row = np.asarray(param_array[ref.batch_idx], dtype=np.float32)
        print(f"Loaded row from {ref.path}: {row}")

    synth_params, _ = spec.decode(row)
    print(f"Decoded synth params: {synth_params}")
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


def play_audio(plugin: VST3Plugin, close_window_event: threading.Event) -> None:
    """Stream silence through Surge XT and write synthesized audio to the output device."""
    silence = np.zeros((CHANNELS, BUFFER_SIZE), dtype=np.float32)
    # output_device = None if sys.platform == "Linux" else AudioStream.default_output_device_name
    stream_resampler = None
    with AudioStream(
        output_device_name=AudioStream.default_output_device_name,
        sample_rate=SAMPLE_RATE,
        buffer_size=BUFFER_SIZE,
    ) as stream:
        while not close_window_event.is_set():
            if SAMPLE_RATE != PLAYBACK_SAMPLE_RATE and stream_resampler is None:
                stream_resampler = StreamResampler(SAMPLE_RATE, PLAYBACK_SAMPLE_RATE, CHANNELS)
            synth_output = plugin(silence, SAMPLE_RATE, reset=False)
            if stream_resampler:
                resampled = stream_resampler.process(synth_output)
                stream.write(resampled, PLAYBACK_SAMPLE_RATE)
            else:
                stream.write(synth_output, SAMPLE_RATE)


def wait_and_watch(plugin: VST3Plugin, close_window_event: threading.Event) -> None:
    def quit_action():
        close_window_event.set()
        return False  # stop loop

    def show_params():
        print(f"params: {dict(plugin.parameters)}")  # pyright: ignore[reportAttributeAccessIssue]
        return True  # keep going

    actions = {
        "q": quit_action,
        "p": show_params,
    }
    while not close_window_event.is_set():
        ch = click.getchar()
        action = actions.get(ch)
        if action and not action():
            return


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
    help="Base preset to load before applying predicted params.",
)
@click.option(
    "--param-spec",
    type=str,
    default="surge_xt",
    help="Parameter spec name used to decode the prediction tensor.",
)
@click.option(
    "--output-wav",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Render audio offline to this WAV file instead of opening an output device. "
        "Use in headless environments (e.g. Docker on Linux). "
        "When set, the plugin GUI is not opened."
    ),
)
@click.option(
    "--duration",
    type=float,
    default=5.0,
    help="Seconds of audio to render when --output-wav is set.",
)
def main(
    plugin_path: str,
    pred: PredictionRef | None,
    dataset_ref: DatasetRef | None,
    preset_path: str,
    param_spec: str,
    output_wav: Path | None,
    duration: float,
) -> None:
    """Open Surge XT GUI with real-time audio streaming.

    When --pred is provided, the predicted parameters from predict_vst_audio.py output are applied
    to the plugin before the editor opens.
    """
    plugin = load_plugin(plugin_path)
    load_preset(plugin, preset_path)

    # flush plugin to ensure that full parameter dict is populated.
    plugin.process([], 32.0, SAMPLE_RATE, CHANNELS, 2048, True)  # flush
    plugin.reset()
    plugin.process([], 32.0, SAMPLE_RATE, CHANNELS, 2048, True)  # flush
    plugin.reset()

    if dataset_ref is not None:
        synth_params = load_dataset_synth_params(dataset_ref, param_spec)
        set_params(plugin, synth_params)
    elif pred is not None:
        synth_params = load_prediction_synth_params(pred, param_spec)
        set_params(plugin, synth_params)
    else:
        plugin = VST3Plugin(plugin_path)

    if output_wav is not None:
        render_to_wav(plugin, output_wav, duration)
        return

    close_window_event = threading.Event()
    audio_thread = threading.Thread(
        target=play_audio, args=(plugin, close_window_event), daemon=True
    )
    audio_thread.start()
    threading.Thread(target=wait_and_watch, args=(plugin, close_window_event), daemon=True).start()

    plugin.show_editor(close_window_event)
    # editor closed manually → still shut down audio
    audio_thread.join(timeout=2)


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
