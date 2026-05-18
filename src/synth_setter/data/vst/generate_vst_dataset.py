from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import h5py
import hdf5plugin
import librosa
import numpy as np
import rootutils
from loguru import logger
from pedalboard import VST3Plugin
from pyloudnorm import Meter
from pydantic_settings import BaseSettings, CliApp, CliPositionalArg, SettingsConfigDict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from synth_setter.pipeline.schemas.spec import (  # noqa: E402
    EXTENSION_TO_OUTPUT_FORMAT,
    RenderConfig,
)
from synth_setter.data.vst import param_specs  # noqa
from synth_setter.data.vst.core import render_params  # noqa
from synth_setter.data.vst.param_spec import ParamSpec  # noqa
from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    MEL_N_MELS,
    MEL_SPEC_FIELD,
    MEL_WINDOW,
    PARAM_ARRAY_FIELD,
    audio_dataset_shape,
    mel_dataset_shape,
    mel_hop_length,
    mel_n_fft,
    param_array_dataset_shape,
)


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
    """Per-channel mel-spectrogram in dB; STFT params come from module-level constants."""
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=MEL_N_MELS,
        n_fft=mel_n_fft(sample_rate),
        hop_length=mel_hop_length(sample_rate),
        window=MEL_WINDOW,
        center=True,
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
    *,
    plugin: VST3Plugin | None = None,
    open_gui: bool = True,
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

    :param plugin: Optional pre-loaded plugin instance threaded through to
        ``render_params``. When supplied, ``plugin_path`` / ``preset_path`` are
        ignored on the render path (caller already loaded the plugin and applied
        the preset). The shard-level cached-plugin path in ``writers._render_in_batches``
        uses this; per-call callers leave it ``None``.
    :param open_gui: Forwarded to ``render_params`` (and from there to
        ``load_plugin``) on the per-call reload path. Ignored when ``plugin``
        is supplied because the caller chose the warm-up policy when loading.
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
            plugin=plugin,
            open_gui=open_gui,
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
        AUDIO_FIELD,
        audio_dataset_shape(num_samples, channels, sample_rate, signal_duration_seconds),
        dtype=np.float16,
        compression=hdf5plugin.Blosc2(),
    )
    mel_dataset, mel_start_idx = create_dataset_and_get_first_unwritten_idx(
        hdf5_file,
        MEL_SPEC_FIELD,
        mel_dataset_shape(num_samples, channels, sample_rate, signal_duration_seconds),
        dtype=np.float32,
        compression=hdf5plugin.Blosc2(),
    )
    param_dataset, param_start_idx = create_dataset_and_get_first_unwritten_idx(
        hdf5_file,
        PARAM_ARRAY_FIELD,
        param_array_dataset_shape(num_samples, num_params),
        dtype=np.float32,
        compression=hdf5plugin.Blosc2(),
    )

    return (
        audio_dataset,
        mel_dataset,
        param_dataset,
        min(audio_start_idx, mel_start_idx, param_start_idx),
    )


class _GenerateCliArgs(RenderConfig, BaseSettings):
    """Pydantic-settings CLI binding for ``generate_vst_dataset.py``.

    Inherits every ``RenderConfig`` field so the CLI flag set tracks the model
    automatically — adding or removing a field on ``RenderConfig`` extends or
    shrinks the CLI surface without a parallel update here. Adds ``data_file``
    as the sole positional arg (the destination shard path; suffix selects
    writer via ``EXTENSION_TO_OUTPUT_FORMAT``).
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
    """Entry point — parse CLI args into a ``RenderConfig`` and render one shard.

    The writer is dispatched on ``data_file``'s suffix via
    ``EXTENSION_TO_OUTPUT_FORMAT`` (``.h5`` → HDF5, ``.tar`` → wds). An unknown
    suffix raises ``SystemExit`` rather than silently producing a half-written
    file in the wrong format.
    """
    # Import lazily so that the writer module's webdataset dep only loads when
    # this CLI entrypoint is invoked, not when callers merely import this
    # module to reach VSTDataSample / generate_sample. (h5py is already a
    # module-level import here, so it is not what the lazy load avoids.)
    from synth_setter.data.vst.writers import make_hdf5_dataset, make_wds_dataset

    args = CliApp.run(_GenerateCliArgs)
    render_cfg = RenderConfig(**args.model_dump(exclude={"data_file"}))
    suffix = Path(args.data_file).suffix
    fmt = EXTENSION_TO_OUTPUT_FORMAT.get(suffix)
    if fmt == "hdf5":
        make_hdf5_dataset(args.data_file, render_cfg)
    elif fmt == "wds":
        make_wds_dataset(args.data_file, render_cfg)
    else:
        raise SystemExit(
            f"data_file must end in one of {sorted(EXTENSION_TO_OUTPUT_FORMAT)}, "
            f"got {suffix!r}"
        )


if __name__ == "__main__":
    main()
