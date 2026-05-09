import hashlib
import io
import json
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import click
import h5py
import hdf5plugin
import librosa
import numpy as np
import rootutils
from loguru import logger
from pyloudnorm import Meter
from tqdm import trange

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.vst import param_specs, render_params  # noqa
from src.data.vst.param_spec import ParamSpec  # noqa


@dataclass
class WDSBuffers:
    """In-memory batched arrays mirroring the HDF5 datasets.

    Filled in-step with HDF5 writes when wds_file is set; flushed once at the end via
    _write_wds_shard.
    """

    audio: np.ndarray
    mel: np.ndarray
    param_array: np.ndarray


@dataclass
class VSTDataSample:
    synth_params: dict[str, float]
    note_params: dict[str, float]

    sample_rate: float
    channels: int

    param_spec: ParamSpec

    audio: np.ndarray
    mel_spec: np.ndarray
    param_array: np.ndarray = None

    def __post_init__(self):
        self.param_array = self.param_spec.encode(self.synth_params, self.note_params)


def make_spectrogram(audio: np.ndarray, sample_rate: float) -> np.ndarray:
    """Values hardcoded to be roughly like those used by the audio spectrogram transformer.

    i.e. 100 frames per second, 128 mels, ~25ms window, hamming window.
    """

    n_fft = int(0.025 * sample_rate)
    hop_length = int(sample_rate / 100.0)
    window = "hamming"

    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=128,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)
    return spec_db


def generate_sample(
    plugin_path: str,
    velocity: int,
    signal_duration_seconds: float,
    sample_rate: float,
    channels: int,
    min_loudness: float,
    param_spec: ParamSpec,
    preset_path: Optional[str],
    fixed_synth_params: dict[str, float] | None = None,
    fixed_note_params: dict[str, int | tuple[float, float]] | None = None,
) -> VSTDataSample:
    """Render a single VST sample.

    When ``fixed_synth_params`` and/or ``fixed_note_params`` are supplied, they take
    precedence over the values drawn from ``param_spec.sample()`` for deterministic
    rendering. When ``fixed_synth_params`` is supplied (with or without
    ``fixed_note_params``), the function raises ``ValueError`` on loudness fail
    rather than retrying — the synth patch is the dominant determinant of loudness,
    so re-sampling note params alone almost never lifts a silent patch above
    ``min_loudness`` and the loop would run forever. When only ``fixed_note_params``
    is supplied, the synth is re-sampled each retry and the loop remains meaningful.
    """
    while True:
        if fixed_synth_params is None or fixed_note_params is None:
            logger.debug("sampling params")
            sampled_synth, sampled_note = param_spec.sample()
            synth_params = (
                fixed_synth_params if fixed_synth_params is not None else sampled_synth
            )
            note_params = fixed_note_params if fixed_note_params is not None else sampled_note
        else:
            synth_params = fixed_synth_params
            note_params = fixed_note_params

        output = render_params(
            plugin_path,
            synth_params,
            note_params["pitch"],
            velocity,
            note_params["note_start_and_end"],
            signal_duration_seconds,
            sample_rate,
            channels,
            preset_path=preset_path,
        )

        meter = Meter(sample_rate)
        loudness = meter.integrated_loudness(output.T)
        logger.debug(f"loudness: {loudness}")
        if loudness < min_loudness:
            if fixed_synth_params is not None:
                raise ValueError(
                    f"fixed_synth_params render produced loudness {loudness:.2f} dB "
                    f"below min_loudness {min_loudness:.2f} dB. The synth patch is "
                    f"held constant and dominates loudness, so retrying is futile "
                    f"(the fully-fixed case has no re-sample input at all; the "
                    f"only-synth-fixed case re-samples note params, which rarely "
                    f"lifts a silent patch above the threshold). Provide a louder "
                    f"patch."
                )
            logger.debug("loudness too low, skipping")
            continue

        break

    logger.debug("making spectrogram")
    spectrogram = make_spectrogram(output, sample_rate)

    return VSTDataSample(
        synth_params=synth_params,
        note_params=note_params,
        audio=output.T,
        mel_spec=spectrogram,
        sample_rate=sample_rate,
        channels=channels,
        param_spec=param_spec,
    )


def save_sample(
    sample: VSTDataSample,
    audio_dataset: h5py.Dataset,
    mel_dataset: h5py.Dataset,
    param_dataset: h5py.Dataset,
    idx: int,
) -> None:
    logger.info(f"Saving sample {idx}...")
    audio_dataset[idx, :, :] = sample.audio.T
    mel_dataset[idx, :, :] = sample.mel_spec
    param_dataset[idx, :] = sample.param_array
    logger.info(f"Sample {idx} written!")


def save_samples(
    samples: List[VSTDataSample],
    audio_dataset: h5py.Dataset,
    mel_dataset: h5py.Dataset,
    param_dataset: h5py.Dataset,
    start_idx: int,
    wds_buffers: Optional[WDSBuffers] = None,
) -> None:
    logger.info(f"Saving {len(samples)} samples...")
    audios = np.stack([s.audio.T for s in samples], axis=0)
    mel_specs = np.stack([s.mel_spec for s in samples], axis=0)
    param_arrays = np.stack([s.param_array for s in samples], axis=0)

    end = start_idx + len(samples)
    audio_dataset[start_idx:end, :, :] = audios
    mel_dataset[start_idx:end, :, :] = mel_specs
    param_dataset[start_idx:end, :] = param_arrays

    # The wds tar is the shard's mirror of the HDF5 file; the rows must come
    # back from h5py to capture lossy float16 round-tripping in the audio
    # dataset (test_h5_and_wds_outputs_are_equivalent asserts byte equality).
    if wds_buffers is not None:
        wds_buffers.audio[start_idx:end] = audio_dataset[start_idx:end]
        wds_buffers.mel[start_idx:end] = mel_dataset[start_idx:end]
        wds_buffers.param_array[start_idx:end] = param_dataset[start_idx:end]

    logger.info(f"{len(samples)} samples written!")


def get_first_unwritten_idx(dataset: h5py.Dataset) -> int:
    num_rows, *_ = dataset.shape
    for i in range(num_rows):
        row = dataset[num_rows - i - 1]
        if not np.all(row == 0):
            return num_rows - i
        logger.debug(f"Row {num_rows - i - 1} is empty...")

    return 0


def create_dataset_and_get_first_unwritten_idx(
    h5py_file: h5py.File,
    name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    compression: Any,
) -> Tuple[h5py.Dataset, int]:
    logger.info(f"Looking for dataset {name}...")
    if name in h5py_file:
        logger.info(f"Found dataset {name}, looking for first unwritten row.")
        dataset = h5py_file[name]
        return dataset, get_first_unwritten_idx(dataset)

    dataset = h5py_file.create_dataset(
        name, shape=shape, dtype=dtype, compression=compression
    )
    return dataset, 0


def create_datasets_and_get_start_idx(
    hdf5_file: h5py.File,
    num_samples: int,
    channels: int,
    sample_rate: float,
    signal_duration_seconds: float,
    num_params: int,
):
    audio_dataset, audio_start_idx = create_dataset_and_get_first_unwritten_idx(
        hdf5_file,
        "audio",
        (num_samples, channels, sample_rate * signal_duration_seconds),
        dtype=np.float16,
        compression=hdf5plugin.Blosc2(),
    )
    mel_dataset, mel_start_idx = create_dataset_and_get_first_unwritten_idx(
        hdf5_file,
        "mel_spec",
        (num_samples, 2, 128, 401),
        dtype=np.float32,
        compression=hdf5plugin.Blosc2(),
    )
    param_dataset, param_start_idx = create_dataset_and_get_first_unwritten_idx(
        hdf5_file,
        "param_array",
        (num_samples, num_params),  # +1 for MIDI note
        dtype=np.float32,
        compression=hdf5plugin.Blosc2(),
    )

    return (
        audio_dataset,
        mel_dataset,
        param_dataset,
        min(audio_start_idx, mel_start_idx, param_start_idx),
    )


def _allocate_wds_buffers(
    num_samples: int,
    channels: int,
    sample_rate: float,
    signal_duration_seconds: float,
    num_params: int,
) -> WDSBuffers:
    """Allocate batched in-memory buffers matching the HDF5 dataset shapes/dtypes."""
    return WDSBuffers(
        audio=np.empty(
            (num_samples, channels, int(sample_rate * signal_duration_seconds)),
            dtype=np.float16,
        ),
        mel=np.empty((num_samples, 2, 128, 401), dtype=np.float32),
        param_array=np.empty((num_samples, num_params), dtype=np.float32),
    )


def _write_wds_shard(
    path: Path | str,
    buffers: WDSBuffers,
    metadata: dict[str, Any],
) -> None:
    """Write a wds tar shard with the four-member layout."""

    def _add_npy(tar: tarfile.TarFile, name: str, arr: np.ndarray) -> None:
        buf = io.BytesIO()
        np.save(buf, arr)
        payload = buf.getvalue()
        info = tarfile.TarInfo(name=name)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    def _add_json(tar: tarfile.TarFile, name: str, value: dict[str, Any]) -> None:
        payload = json.dumps(value).encode("utf-8")
        info = tarfile.TarInfo(name=name)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    with tarfile.open(path, "w") as tar:
        _add_npy(tar, "audio.npy", buffers.audio)
        _add_npy(tar, "mel.npy", buffers.mel)
        _add_npy(tar, "param_array.npy", buffers.param_array)
        _add_json(tar, "metadata.json", metadata)


def make_dataset(
    hdf5_file: Path | str,
    num_samples: int,
    plugin_path: str,
    preset_path: Optional[str],
    sample_rate: float,
    channels: int,
    velocity: int,
    signal_duration_seconds: float,
    min_loudness: float,
    param_spec: ParamSpec,
    sample_batch_size: int,
    fixed_synth_params_list: list[dict[str, float]] | None = None,
    fixed_note_params_list: list[dict[str, int | tuple[float, float]]] | None = None,
    wds_file: Path | str | None = None,
) -> None:
    with h5py.File(hdf5_file, "a") as h5:
        audio_dataset, mel_dataset, param_dataset, start_idx = (
            create_datasets_and_get_start_idx(
                hdf5_file=h5,
                num_samples=num_samples,
                channels=channels,
                sample_rate=sample_rate,
                signal_duration_seconds=signal_duration_seconds,
                num_params=len(param_spec),
            )
        )

        expected_fixed_len = num_samples - start_idx
        for name, lst in [
            ("fixed_synth_params_list", fixed_synth_params_list),
            ("fixed_note_params_list", fixed_note_params_list),
        ]:
            if lst is not None and len(lst) < expected_fixed_len:
                raise ValueError(
                    f"{name} has length {len(lst)}, expected at least "
                    f"num_samples - start_idx = {expected_fixed_len} "
                    f"(num_samples={num_samples}, start_idx={start_idx})"
                )

        audio_dataset.attrs["velocity"] = velocity
        audio_dataset.attrs["signal_duration_seconds"] = signal_duration_seconds
        audio_dataset.attrs["sample_rate"] = sample_rate
        audio_dataset.attrs["channels"] = channels
        audio_dataset.attrs["min_loudness"] = min_loudness

        wds_buffers: Optional[WDSBuffers] = None
        if wds_file is not None:
            wds_buffers = _allocate_wds_buffers(
                num_samples=num_samples,
                channels=channels,
                sample_rate=sample_rate,
                signal_duration_seconds=signal_duration_seconds,
                num_params=len(param_spec),
            )

        sample_batch = []
        sample_batch_start = start_idx

        for i in trange(start_idx, num_samples):
            logger.info(f"Making sample {i}")
            fixed_idx = i - start_idx
            sample = generate_sample(
                plugin_path,
                velocity=velocity,
                signal_duration_seconds=signal_duration_seconds,
                sample_rate=sample_rate,
                channels=channels,
                min_loudness=min_loudness,
                param_spec=param_spec,
                preset_path=preset_path,
                fixed_synth_params=(
                    fixed_synth_params_list[fixed_idx]
                    if fixed_synth_params_list is not None
                    else None
                ),
                fixed_note_params=(
                    fixed_note_params_list[fixed_idx]
                    if fixed_note_params_list is not None
                    else None
                ),
            )

            sample_batch.append(sample)
            if len(sample_batch) == sample_batch_size:
                save_samples(
                    sample_batch,
                    audio_dataset,
                    mel_dataset,
                    param_dataset,
                    sample_batch_start,
                    wds_buffers=wds_buffers,
                )
                sample_batch = []
                sample_batch_start += sample_batch_size

        if len(sample_batch) > 0:
            save_samples(
                sample_batch,
                audio_dataset,
                mel_dataset,
                param_dataset,
                sample_batch_start,
                wds_buffers=wds_buffers,
            )

        if wds_file is not None and wds_buffers is not None:
            metadata = {
                "velocity": velocity,
                "signal_duration_seconds": signal_duration_seconds,
                "sample_rate": sample_rate,
                "channels": channels,
                "min_loudness": min_loudness,
            }
            _write_wds_shard(wds_file, wds_buffers, metadata)


@click.command()
@click.argument("data_file", type=str, required=True)
@click.argument("num_samples", type=int, required=True)
@click.option("--plugin_path", "-p", type=str, default="plugins/Surge XT.vst3")
@click.option("--preset_path", "-r", type=str, default="presets/surge-base.vstpreset")
@click.option("--sample_rate", "-s", type=float, default=44100.0)
@click.option("--channels", "-c", type=int, default=2)
@click.option("--velocity", "-v", type=int, default=100)
@click.option("--signal_duration_seconds", "-d", type=float, default=4.0)
@click.option("--min_loudness", "-l", type=float, default=-55.0)
@click.option("--param_spec", "-t", type=str, default="surge_xt")
@click.option("--sample_batch_size", "-b", type=int, default=32)
@click.option(
    "--wds-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path for a wds tar shard mirror of the HDF5 output.",
)
def main(
    data_file: str,
    num_samples: int,
    plugin_path: str = "plugins/Surge XT.vst3",
    preset_path: str = "presets/surge-base.vstpreset",
    sample_rate: float = 44100.0,
    channels: int = 2,
    velocity: int = 100,
    signal_duration_seconds: float = 4.0,
    min_loudness: float = -50.0,
    param_spec: str = "surge_xt",
    sample_batch_size: int = 32,
    wds_out: Optional[Path] = None,
):
    param_spec = param_specs[param_spec]
    make_dataset(
        hdf5_file=data_file,
        num_samples=num_samples,
        plugin_path=plugin_path,
        preset_path=preset_path,
        sample_rate=sample_rate,
        channels=channels,
        velocity=velocity,
        signal_duration_seconds=signal_duration_seconds,
        min_loudness=min_loudness,
        param_spec=param_spec,
        sample_batch_size=sample_batch_size,
        wds_file=wds_out,
    )


if __name__ == "__main__":
    main()
