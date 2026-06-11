from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin
import librosa
import numpy as np
from loguru import logger
from pedalboard import VST3Plugin
from pydantic_settings import BaseSettings, CliApp, CliPositionalArg, SettingsConfigDict
from pyloudnorm import Meter

from synth_setter.data.vst.core import render_params
from synth_setter.data.vst.param_spec import NoteParams, ParamSpec
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
from synth_setter.pipeline.schemas.spec import (
    OutputFormat,
    RenderConfig,
)
from synth_setter.pipeline.spec_io import join_uri, localized_uri


@dataclass
class VSTDataSample:
    synth_params: dict[str, float]
    note_params: NoteParams

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
    fixed_note_params: NoteParams | None = None,
    *,
    plugin: VST3Plugin | None = None,
    warmup: bool = False,
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

    :param plugin: Forwarded to ``render_params``; when set, the renderer
        skips ``load_plugin``/``load_preset``.
    :param warmup: Forwarded to ``render_params``; runs the ``show_editor``
        warm-up on the plugin used for this render (newly loaded or cached).
        Applied at most once per ``generate_sample`` call — the loudness-gate
        retry loop drops ``warmup`` to ``False`` after the first attempt so a
        retrying sample never exceeds the per-shard cadence budget (#714).
    """
    while True:
        if fixed_synth_params is None or fixed_note_params is None:
            logger.debug("sampling params")
            sampled_synth, sampled_note = param_spec.sample()
            synth_params = fixed_synth_params if fixed_synth_params is not None else sampled_synth
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
            warmup=warmup,
        )
        warmup = False

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
    shape: tuple[int, ...],
    dtype: np.dtype,
    compression: Any,
) -> tuple[h5py.Dataset, int]:
    logger.info(f"Looking for dataset {name}...")
    if name in h5py_file:
        logger.info(f"Found dataset {name}, looking for first unwritten row.")
        dataset = h5py_file[name]
        return dataset, get_first_unwritten_idx(dataset)

    dataset = h5py_file.create_dataset(name, shape=shape, dtype=dtype, compression=compression)
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


def fixed_params_from_dataset(
    source_shard: Path,
    param_spec: ParamSpec,
) -> tuple[list[dict[str, float]], list[NoteParams]]:
    """Decode an existing shard's ``param_array`` into fixed synth/note param lists.

    Reads the ``param_array`` dataset from ``source_shard`` and decodes each row
    via ``param_spec.decode``, yielding the per-row fixed params that reproduce
    the source dataset under a fresh render. The two returned lists are
    row-aligned and ready to pass to ``make_hdf5_dataset`` / ``make_wds_dataset``
    as ``fixed_synth_params_list`` / ``fixed_note_params_list``.

    :param source_shard: Source shard whose ``param_array`` rows are decoded.
    :param param_spec: Param spec the source was encoded with; its encoded width
        must equal the source's ``param_array`` column count.
    :returns: ``(synth_params_list, note_params_list)``, one dict per source row.
    :raises ValueError: ``source_shard`` lacks a ``param_array`` dataset, that
        dataset is not 2-D, or its column count does not match ``len(param_spec)``
        — any of which would make the per-row decode mis-align.
    """
    with h5py.File(source_shard, "r") as h5:
        if PARAM_ARRAY_FIELD not in h5:
            raise ValueError(
                f"source shard {source_shard} has no {PARAM_ARRAY_FIELD!r} dataset; "
                "it is not a parameter-bearing dataset shard."
            )
        param_array = h5[PARAM_ARRAY_FIELD][:]
    if param_array.ndim != 2:
        raise ValueError(
            f"source shard {source_shard} {PARAM_ARRAY_FIELD!r} must be 2-D "
            f"(num_rows, num_params), got shape {param_array.shape}."
        )
    if param_array.shape[1] != len(param_spec):
        raise ValueError(
            f"source shard {source_shard} param_array has {param_array.shape[1]} columns "
            f"but param_spec has {len(param_spec)}; the copy source must share the target's "
            "param_spec_name (same encoding width)."
        )
    synth_params_list: list[dict[str, float]] = []
    note_params_list: list[NoteParams] = []
    for row in param_array:
        synth_params, note_params = param_spec.decode(row)
        synth_params_list.append(synth_params)
        note_params_list.append(note_params)
    return synth_params_list, note_params_list


class _GenerateCliArgs(RenderConfig, BaseSettings):
    """Pydantic-settings CLI binding for ``generate_vst_dataset.py``.

    Inherits every ``RenderConfig`` field so the CLI flag set tracks the model
    automatically — adding or removing a field on ``RenderConfig`` extends or
    shrinks the CLI surface without a parallel update here. Adds ``data_file``
    as the sole positional arg (the destination shard path; suffix selects
    writer via ``OutputFormat.from_extension``), and an optional
    ``copy_dataset_root_uri`` that triggers the param-copy path.

    .. attribute :: copy_dataset_root_uri

        Optional copy source root URI; when set, the same-named source shard's
        params are decoded and re-rendered instead of sampling fresh ones.
    """

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_prog_name="generate_vst_dataset",
        cli_kebab_case=False,
        strict=True,
        extra="forbid",
    )

    data_file: CliPositionalArg[str]
    copy_dataset_root_uri: str | None = None


def main() -> None:
    """Entry point — parse CLI args into a ``RenderConfig`` and render one shard.

    The writer is dispatched on ``data_file``'s suffix via
    ``OutputFormat.from_extension`` (``.h5`` → HDF5, ``.tar`` → wds,
    ``.lance`` → Lance). An unknown suffix raises ``SystemExit`` rather than
    silently producing a half-written file in the wrong format.

    When ``--copy_dataset_root_uri`` is set, the params of the same-named source
    shard under that root URI are decoded into fixed synth/note lists and
    re-rendered instead of sampling fresh params. The root URI may be a bare
    path, ``file://`` URI, or ``r2://`` URI (an ``r2://`` shard is downloaded to
    a tempfile for the decode). The source is read as an HDF5 ``param_array``, so
    copy is supported for hdf5 output only (a ``.tar`` output has no same-named
    HDF5 source); a non-hdf5 output with ``--copy_dataset_root_uri`` raises
    ``SystemExit``.
    """
    # Import lazily so that the writer module's webdataset dep only loads when
    # this CLI entrypoint is invoked, not when callers merely import this
    # module to reach VSTDataSample / generate_sample. (h5py is already a
    # module-level import here, so it is not what the lazy load avoids.)
    from synth_setter.data.vst.param_spec_registry import param_specs
    from synth_setter.data.vst.writers import (
        make_hdf5_dataset,
        make_lance_dataset,
        make_wds_dataset,
    )

    args = CliApp.run(_GenerateCliArgs)
    render_cfg = RenderConfig(**args.model_dump(exclude={"data_file", "copy_dataset_root_uri"}))

    suffix = Path(args.data_file).suffix
    fmt = OutputFormat.from_extension(suffix)
    if fmt is None:
        raise SystemExit(
            f"data_file must end in one of {sorted(f.extension for f in OutputFormat)}, "
            f"got {suffix!r}"
        )

    fixed_synth_params_list = None
    fixed_note_params_list = None
    if args.copy_dataset_root_uri is not None:
        if fmt is not OutputFormat.HDF5:
            raise SystemExit(
                "--copy_dataset_root_uri supports hdf5 output only; the source params are "
                f"read from a same-named HDF5 shard, but data_file suffix is {suffix!r}."
            )
        # Same shard filename under the copy root URI names the source HDF5 shard;
        # localized_uri downloads it locally first when the root is an r2:// URI.
        source_shard_uri = join_uri(args.copy_dataset_root_uri, Path(args.data_file).name)
        with localized_uri(source_shard_uri) as source_shard:
            fixed_synth_params_list, fixed_note_params_list = fixed_params_from_dataset(
                source_shard, param_specs[render_cfg.param_spec_name]
            )

    if fmt is OutputFormat.HDF5:
        make_hdf5_dataset(
            args.data_file,
            render_cfg,
            fixed_synth_params_list=fixed_synth_params_list,
            fixed_note_params_list=fixed_note_params_list,
        )
    elif fmt is OutputFormat.WDS:
        make_wds_dataset(
            args.data_file,
            render_cfg,
            fixed_synth_params_list=fixed_synth_params_list,
            fixed_note_params_list=fixed_note_params_list,
        )
    else:
        make_lance_dataset(
            args.data_file,
            render_cfg,
            fixed_synth_params_list=fixed_synth_params_list,
            fixed_note_params_list=fixed_note_params_list,
        )


if __name__ == "__main__":
    main()
