import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, cast

import click
import h5py
import hdf5plugin
import librosa
import numpy as np
import rootutils
import webdataset as wds
from loguru import logger
from pyloudnorm import Meter
from tqdm import trange

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.pipeline.schemas.shard_metadata import ShardMetadata  # noqa
from src.pipeline.schemas.spec import RenderConfig  # noqa
from src.data.vst import param_specs  # noqa
from src.data.vst.core import render_params  # noqa
from src.data.vst.param_spec import ParamSpec  # noqa


class _WdsTarSink(Protocol):
    """Minimal surface from ``wds.TarWriter`` used by the wds writer path.

    The webdataset library lacks PEP 561 type stubs, so direct ``wds.TarWriter``
    references trigger ``reportAttributeAccessIssue`` under pyright. Typing the
    helper signatures against this Protocol keeps the call surface narrow and
    confines the type-ignore to the single ``wds.TarWriter(...)`` instantiation.
    """

    def write(self, sample: dict[str, Any]) -> None: ...

    def close(self) -> None: ...

    def __enter__(self) -> "_WdsTarSink": ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...


@dataclass
class VSTDataSample:
    synth_params: dict[str, float]
    note_params: dict[str, float]

    sample_rate: float
    channels: int

    param_spec: ParamSpec

    audio: np.ndarray
    mel_spec: np.ndarray
    param_array: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.param_array = self.param_spec.encode(self.synth_params, self.note_params)


# AST-style mel front-end. Centralised so ``_mel_dataset_shape`` and
# ``make_spectrogram`` agree on the time-axis size.
_MEL_FRAMES_PER_SECOND = 100
_MEL_N_MELS = 128
_MEL_N_FFT_FRACTION_OF_SAMPLE_RATE = 0.025
_MEL_WINDOW = "hamming"


def _mel_hop_length(sample_rate: float) -> int:
    return int(sample_rate / _MEL_FRAMES_PER_SECOND)


def _mel_n_fft(sample_rate: float) -> int:
    return int(_MEL_N_FFT_FRACTION_OF_SAMPLE_RATE * sample_rate)


def _mel_n_frames(sample_rate: float, signal_duration_seconds: float) -> int:
    """Compute the number of mel-time frames librosa produces for the given audio length.

    Mirrors librosa.feature.melspectrogram's frame count when ``center=True`` (its default):
    ``1 + audio_length // hop_length``. Adding a row to ``hdf5_file`` for any duration
    other than the legacy 4s-at-16kHz default would silently produce wrong-shape mel
    data without this helper — see ml-pipeline:block #3 in PR #883.
    """
    audio_length = int(sample_rate * signal_duration_seconds)
    return 1 + audio_length // _mel_hop_length(sample_rate)


def _mel_dataset_shape(
    num_samples: int, channels: int, sample_rate: float, signal_duration_seconds: float
) -> tuple[int, int, int, int]:
    """Mel-spectrogram dataset shape ``(N, C, n_mels, n_frames)`` for the writer."""
    return (
        num_samples,
        channels,
        _MEL_N_MELS,
        _mel_n_frames(sample_rate, signal_duration_seconds),
    )


def make_spectrogram(audio: np.ndarray, sample_rate: float) -> np.ndarray:
    """Per-channel mel-spectrogram in dB; STFT params come from module-level constants."""
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=_MEL_N_MELS,
        n_fft=_mel_n_fft(sample_rate),
        hop_length=_mel_hop_length(sample_rate),
        window=_MEL_WINDOW,
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
    preset_path: str | None,
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


def save_hdf5_samples(
    samples: list[VSTDataSample],
    audio_dataset: h5py.Dataset,
    mel_dataset: h5py.Dataset,
    param_dataset: h5py.Dataset,
    start_idx: int,
) -> None:
    """Append a batch of rendered samples to the three HDF5 datasets in place."""
    logger.info(f"Saving {len(samples)} samples to hdf5...")
    audios = np.stack([s.audio.T for s in samples], axis=0)
    mel_specs = np.stack([s.mel_spec for s in samples], axis=0)
    param_arrays = np.stack([s.param_array for s in samples], axis=0)

    end = start_idx + len(samples)
    audio_dataset[start_idx:end, :, :] = audios
    mel_dataset[start_idx:end, :, :] = mel_specs
    param_dataset[start_idx:end, :] = param_arrays

    logger.info(f"{len(samples)} hdf5 samples written!")


def save_wds_samples(
    samples: list[VSTDataSample],
    sink: _WdsTarSink,
    start_idx: int,
) -> None:
    """Write a batch of rendered samples as a per-batch-keyed tar entry.

    Audio is cast to ``float16`` to match the h5 path's storage precision so
    consumers see the same dtype regardless of which writer produced the shard;
    mel_spec and param_array stay ``float32``.
    """
    logger.info(f"Saving {len(samples)} samples to wds...")
    audios = np.stack([s.audio.T for s in samples], axis=0).astype(np.float16)
    mel_specs = np.stack([s.mel_spec for s in samples], axis=0)
    param_arrays = np.stack([s.param_array for s in samples], axis=0)

    sink.write(
        {
            "__key__": f"{start_idx:08d}",
            "audio.npy": audios,
            "mel_spec.npy": mel_specs,
            "param_array.npy": param_arrays,
        }
    )

    logger.info(f"{len(samples)} wds samples written!")


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
) -> tuple[h5py.Dataset, h5py.Dataset, h5py.Dataset, int]:
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
        _mel_dataset_shape(num_samples, channels, sample_rate, signal_duration_seconds),
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


def _validate_fixed_params_lengths(
    *,
    num_samples: int,
    start_idx: int,
    fixed_synth_params_list: list[dict[str, float]] | None,
    fixed_note_params_list: list[dict[str, int | tuple[float, float]]] | None,
) -> None:
    """Raise ``ValueError`` if a fixed-params list is shorter than the remaining renders."""
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


def _generate_sample_for_index(
    i: int,
    start_idx: int,
    *,
    plugin_path: str,
    preset_path: str | None,
    velocity: int,
    signal_duration_seconds: float,
    sample_rate: float,
    channels: int,
    min_loudness: float,
    param_spec: ParamSpec,
    fixed_synth_params_list: list[dict[str, float]] | None,
    fixed_note_params_list: list[dict[str, int | tuple[float, float]]] | None,
) -> VSTDataSample:
    """Render the ``i``-th sample, picking up the ``(i - start_idx)``-th fixed-params entry."""
    fixed_idx = i - start_idx
    return generate_sample(
        plugin_path,
        velocity=velocity,
        signal_duration_seconds=signal_duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
        min_loudness=min_loudness,
        param_spec=param_spec,
        preset_path=preset_path,
        fixed_synth_params=(
            fixed_synth_params_list[fixed_idx] if fixed_synth_params_list is not None else None
        ),
        fixed_note_params=(
            fixed_note_params_list[fixed_idx] if fixed_note_params_list is not None else None
        ),
    )


def make_hdf5_dataset(
    hdf5_file: Path | str,
    render_cfg: RenderConfig,
    *,
    fixed_synth_params_list: list[dict[str, float]] | None = None,
    fixed_note_params_list: list[dict[str, int | tuple[float, float]]] | None = None,
) -> None:
    """Render ``render_cfg.batch_per_shard`` samples to an HDF5 file at ``hdf5_file``.

    Resumable: a partially-written file picks up at the first all-zero row, so a
    crashed worker can re-run with the same args and only the missing tail is
    rendered. Audio is stored as ``float16`` (Blosc2-compressed); mel_spec and
    param_array are ``float32``. The five sidecar attrs (velocity, signal duration,
    sample rate, channels, min_loudness) are written to ``audio.attrs`` from a
    single ``ShardMetadata`` instance — the same instance the wds writer uses for
    its ``metadata.json`` member, so both formats expose identical metadata.
    """
    param_spec = param_specs[render_cfg.param_spec_name]
    num_samples = render_cfg.batch_per_shard
    meta = _shard_metadata_from_render(render_cfg)
    with h5py.File(hdf5_file, "a") as h5:
        audio_dataset, mel_dataset, param_dataset, start_idx = (
            create_datasets_and_get_start_idx(
                hdf5_file=h5,
                num_samples=num_samples,
                channels=render_cfg.channels,
                sample_rate=render_cfg.sample_rate,
                signal_duration_seconds=render_cfg.signal_duration_seconds,
                num_params=len(param_spec),
            )
        )

        _validate_fixed_params_lengths(
            num_samples=num_samples,
            start_idx=start_idx,
            fixed_synth_params_list=fixed_synth_params_list,
            fixed_note_params_list=fixed_note_params_list,
        )

        for k, v in meta.model_dump().items():
            audio_dataset.attrs[k] = v

        sample_batch: list[VSTDataSample] = []
        sample_batch_start = start_idx
        for i in trange(start_idx, num_samples):
            logger.info(f"Making sample {i}")
            sample_batch.append(
                _generate_sample_for_index(
                    i,
                    start_idx,
                    plugin_path=render_cfg.plugin_path,
                    preset_path=render_cfg.preset_path,
                    velocity=render_cfg.velocity,
                    signal_duration_seconds=render_cfg.signal_duration_seconds,
                    sample_rate=render_cfg.sample_rate,
                    channels=render_cfg.channels,
                    min_loudness=render_cfg.min_loudness,
                    param_spec=param_spec,
                    fixed_synth_params_list=fixed_synth_params_list,
                    fixed_note_params_list=fixed_note_params_list,
                )
            )
            if len(sample_batch) == render_cfg.sample_batch_size:
                save_hdf5_samples(
                    sample_batch, audio_dataset, mel_dataset, param_dataset, sample_batch_start
                )
                sample_batch = []
                sample_batch_start += render_cfg.sample_batch_size

        if sample_batch:
            save_hdf5_samples(
                sample_batch, audio_dataset, mel_dataset, param_dataset, sample_batch_start
            )


def make_wds_dataset(
    wds_file: Path | str,
    render_cfg: RenderConfig,
    *,
    fixed_synth_params_list: list[dict[str, float]] | None = None,
    fixed_note_params_list: list[dict[str, int | tuple[float, float]]] | None = None,
) -> None:
    """Render ``render_cfg.batch_per_shard`` samples to a webdataset tar at ``wds_file``.

    Not resumable — if ``wds_file`` exists it is overwritten on open. Audio is cast
    to ``float16`` to match the h5 path's storage precision; consumers can upcast
    on read if higher precision is required. The shard's ``metadata.json`` member
    is built from the same ``ShardMetadata`` instance the h5 path uses for its
    ``audio.attrs``, so both formats expose identical metadata.
    """
    param_spec = param_specs[render_cfg.param_spec_name]
    num_samples = render_cfg.batch_per_shard
    meta = _shard_metadata_from_render(render_cfg)
    _validate_fixed_params_lengths(
        num_samples=num_samples,
        start_idx=0,
        fixed_synth_params_list=fixed_synth_params_list,
        fixed_note_params_list=fixed_note_params_list,
    )
    with cast(
        _WdsTarSink,
        wds.TarWriter(str(wds_file)),  # pyright: ignore[reportAttributeAccessIssue]
    ) as sink:
        sample_batch: list[VSTDataSample] = []
        sample_batch_start = 0
        for i in trange(num_samples):
            logger.info(f"Making sample {i}")
            sample_batch.append(
                _generate_sample_for_index(
                    i,
                    0,
                    plugin_path=render_cfg.plugin_path,
                    preset_path=render_cfg.preset_path,
                    velocity=render_cfg.velocity,
                    signal_duration_seconds=render_cfg.signal_duration_seconds,
                    sample_rate=render_cfg.sample_rate,
                    channels=render_cfg.channels,
                    min_loudness=render_cfg.min_loudness,
                    param_spec=param_spec,
                    fixed_synth_params_list=fixed_synth_params_list,
                    fixed_note_params_list=fixed_note_params_list,
                )
            )
            if len(sample_batch) == render_cfg.sample_batch_size:
                save_wds_samples(sample_batch, sink, sample_batch_start)
                sample_batch = []
                sample_batch_start += render_cfg.sample_batch_size

        if sample_batch:
            save_wds_samples(sample_batch, sink, sample_batch_start)

        sink.write({"__key__": "metadata", "json": meta.model_dump()})


def _shard_metadata_from_render(render_cfg: RenderConfig) -> ShardMetadata:
    """Project a ``RenderConfig`` onto the per-shard sidecar metadata fields."""
    return ShardMetadata(
        velocity=render_cfg.velocity,
        signal_duration_seconds=render_cfg.signal_duration_seconds,
        sample_rate=render_cfg.sample_rate,
        channels=render_cfg.channels,
        min_loudness=render_cfg.min_loudness,
    )


@click.command()
@click.argument("data_file", type=str, required=True)
@click.option(
    "--render-cfg-json",
    "render_cfg_json",
    type=str,
    required=True,
    help="JSON-serialized RenderConfig (model_dump_json() output).",
)
def main(data_file: str, render_cfg_json: str) -> None:
    """Render a single shard at ``data_file`` (suffix selects writer)."""
    render_cfg = RenderConfig.model_validate_json(render_cfg_json)
    suffix = Path(data_file).suffix
    if suffix == ".h5":
        make_hdf5_dataset(hdf5_file=data_file, render_cfg=render_cfg)
        return
    if suffix == ".tar":
        make_wds_dataset(wds_file=data_file, render_cfg=render_cfg)
        return
    raise click.BadParameter(
        f"data_file must end in .h5 (hdf5) or .tar (wds), got {suffix!r}",
        param_hint="data_file",
    )


if __name__ == "__main__":
    main()
