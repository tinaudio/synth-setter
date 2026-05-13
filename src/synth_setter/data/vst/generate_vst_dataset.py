import hashlib
import random
from dataclasses import dataclass
from typing import Any, List, Tuple

import h5py
import hdf5plugin
import librosa
import numpy as np
import rootutils
from loguru import logger
from pyloudnorm import Meter
from pydantic_settings import BaseSettings, CliApp, CliPositionalArg, SettingsConfigDict
from tqdm import trange

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from synth_setter.pipeline.schemas.spec import RenderConfig  # noqa: E402
from synth_setter.data.vst import param_specs  # noqa
from synth_setter.data.vst.core import render_params  # noqa
from synth_setter.data.vst.param_spec import ParamSpec  # noqa


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
    preset_path: str,
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
) -> None:
    logger.info(f"Saving {len(samples)} samples...")
    audios = np.stack([s.audio.T for s in samples], axis=0)
    mel_specs = np.stack([s.mel_spec for s in samples], axis=0)
    param_arrays = np.stack([s.param_array for s in samples], axis=0)

    audio_dataset[start_idx : start_idx + len(samples), :, :] = audios
    mel_dataset[start_idx : start_idx + len(samples), :, :] = mel_specs
    param_dataset[start_idx : start_idx + len(samples), :] = param_arrays

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


def make_dataset(
    hdf5_file: h5py.File,
    render_cfg: RenderConfig,
    *,
    fixed_synth_params_list: list[dict[str, float]] | None = None,
    fixed_note_params_list: list[dict[str, int | tuple[float, float]]] | None = None,
) -> None:
    """Render ``render_cfg.batch_per_shard`` samples and append them to ``hdf5_file``.

    Resumable: a partially-written file picks up at the first all-zero row, so a
    crashed worker can re-run with the same ``render_cfg`` and only the missing tail
    is regenerated. The ``param_spec_name`` is resolved against the in-process
    registry here (not at the launcher) so the per-shard config stays JSON-only.
    """
    param_spec = param_specs[render_cfg.param_spec_name]
    num_samples = render_cfg.batch_per_shard

    audio_dataset, mel_dataset, param_dataset, start_idx = (
        create_datasets_and_get_start_idx(
            hdf5_file=hdf5_file,
            num_samples=num_samples,
            channels=render_cfg.channels,
            sample_rate=render_cfg.sample_rate,
            signal_duration_seconds=render_cfg.signal_duration_seconds,
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

    audio_dataset.attrs["velocity"] = render_cfg.velocity
    audio_dataset.attrs["signal_duration_seconds"] = render_cfg.signal_duration_seconds
    audio_dataset.attrs["sample_rate"] = render_cfg.sample_rate
    audio_dataset.attrs["channels"] = render_cfg.channels
    audio_dataset.attrs["min_loudness"] = render_cfg.min_loudness

    sample_batch = []
    sample_batch_start = start_idx
    batch_size = render_cfg.sample_batch_size

    for i in trange(start_idx, num_samples):
        logger.info(f"Making sample {i}")
        fixed_idx = i - start_idx
        sample = generate_sample(
            render_cfg.plugin_path,
            velocity=render_cfg.velocity,
            signal_duration_seconds=render_cfg.signal_duration_seconds,
            sample_rate=render_cfg.sample_rate,
            channels=render_cfg.channels,
            min_loudness=render_cfg.min_loudness,
            param_spec=param_spec,
            preset_path=render_cfg.preset_path,
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
        if len(sample_batch) == batch_size:
            save_samples(
                sample_batch,
                audio_dataset,
                mel_dataset,
                param_dataset,
                sample_batch_start,
            )
            sample_batch = []
            sample_batch_start += batch_size

    if len(sample_batch) > 0:
        save_samples(
            sample_batch,
            audio_dataset,
            mel_dataset,
            param_dataset,
            sample_batch_start,
        )


class _GenerateCliArgs(RenderConfig, BaseSettings):
    """Pydantic-settings CLI binding for ``generate_vst_dataset.py``.

    Inherits every ``RenderConfig`` field so the CLI flag set tracks the model
    automatically — adding or removing a field on ``RenderConfig`` extends or
    shrinks the CLI surface without a parallel update here. Adds ``data_file``
    as the sole positional arg (the destination path for the HDF5 shard).
    """

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_prog_name="generate_vst_dataset",
        cli_kebab_case=False,
        strict=True,
        extra="forbid",
    )

    data_file: CliPositionalArg[str]


def main() -> None:
    """Entry point — parse CLI args into a ``RenderConfig`` and render one shard."""
    args = CliApp.run(_GenerateCliArgs)
    render_cfg = RenderConfig(**args.model_dump(exclude={"data_file"}))
    with h5py.File(args.data_file, "a") as f:
        make_dataset(f, render_cfg)


if __name__ == "__main__":
    main()
