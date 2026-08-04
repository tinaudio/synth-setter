"""Shared writers and column builders for Lance shard test fixtures."""

import io
from collections.abc import Mapping, Sequence
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from pedalboard.io import AudioFile

from synth_setter.data.vst.audio_preview import (
    DEFAULT_MP3_BITRATE_KBPS,
    audio_uuid,
    encode_audio_to_mp3,
)
from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    AUDIO_MP3_FIELD,
    AUDIO_UUID_FIELD,
    SKETCH_STRUCT_FIELD,
)
from synth_setter.pipeline.data.lance_shard import (
    sketch_struct_array,
    tensor_array,
    write_lance_dataset,
)

# Tiny per-row shapes shared by the datamodule test fixtures: large enough to
# expose shape mix-ups (every axis distinct), small enough for sub-second tests.
AUDIO_CHANNELS = 2
AUDIO_SAMPLES = 16
MEL_CHANNELS = 2
MEL_N_MELS = 4
MEL_N_FRAMES = 5
M2L_DIM_1 = 6
M2L_DIM_2 = 7
NUM_PARAMS = 11

MEL_SHAPE = (MEL_CHANNELS, MEL_N_MELS, MEL_N_FRAMES)


def with_preview_columns(
    columns: Mapping[str, np.ndarray],
    sample_rate: int,
) -> dict[str, np.ndarray | Sequence[bytes] | Sequence[str]]:
    """Add MP3 and UUID values derived from the persisted audio rows.

    :param columns: Tensor columns containing ``audio`` shaped ``(rows, channels, time)``.
    :param sample_rate: Playback rate used for MP3 encoding.
    :returns: A new mapping with the two canonical preview columns.
    """
    audio_rows = columns[AUDIO_FIELD]
    return {
        **columns,
        AUDIO_MP3_FIELD: [
            encode_audio_to_mp3(row, sample_rate, DEFAULT_MP3_BITRATE_KBPS) for row in audio_rows
        ],
        AUDIO_UUID_FIELD: [audio_uuid(row) for row in audio_rows],
    }


def make_shard_columns(
    num_rows: int, *, num_params: int = NUM_PARAMS, seed: int = 0
) -> dict[str, np.ndarray]:
    """Build the column arrays a VST Lance shard carries.

    :param num_rows: Number of rows along the first axis of every column.
    :param num_params: Width of the ``param_array`` column.
    :param seed: Seed for all columns so splits get distinguishable values.
    :return: Mapping of column name to ``(num_rows, ...)`` array.
    """
    rng = np.random.default_rng(seed)
    return {
        # float16 mirrors the pipeline's on-disk audio dtype (DATASET_FIELD_DTYPES).
        "audio": rng.uniform(-1.0, 1.0, (num_rows, AUDIO_CHANNELS, AUDIO_SAMPLES)).astype(
            np.float16
        ),
        "mel_spec": rng.standard_normal((num_rows, *MEL_SHAPE)).astype(np.float32),
        "music2latent": rng.standard_normal((num_rows, M2L_DIM_1, M2L_DIM_2)).astype(np.float32),
        # params in [0, 1) so the rescale_params=True branch lands in [-1, 1).
        "param_array": rng.random((num_rows, num_params)).astype(np.float32),
    }


def write_seeded_lance_shard(
    path: Path,
    num_rows: int,
    *,
    num_params: int = NUM_PARAMS,
    seed: int = 0,
    mel_fill: float | None = None,
) -> dict[str, np.ndarray]:
    """Write a tiny Lance shard and return its source arrays for assertions.

    :param path: Output ``.lance`` dataset directory.
    :param num_rows: Number of rows along the first axis of every column.
    :param num_params: Width of the ``param_array`` column.
    :param seed: Seed for the per-row arrays.
    :param mel_fill: When set, fill ``mel_spec`` with this constant so
        normalization tests can pin ``(mel - mean) / std`` exactly.
    :return: The column arrays that were written.
    """
    columns = make_shard_columns(num_rows, num_params=num_params, seed=seed)
    if mel_fill is not None:
        columns["mel_spec"] = np.full_like(columns["mel_spec"], mel_fill)
    write_lance_shard(path, columns)
    return columns


def write_mel_stats(dataset_dir: Path, *, mean: float = 0.0, std: float = 1.0) -> None:
    """Write a sibling ``stats.npz`` whose mean/std broadcast against ``mel_spec``.

    :param dataset_dir: Directory holding the ``*.lance`` splits.
    :param mean: Scalar mean broadcast against every mel-spec element.
    :param std: Scalar std broadcast against every mel-spec element.
    """
    np.savez(
        dataset_dir / "stats.npz",
        mean=np.full(MEL_SHAPE, mean, dtype=np.float32),
        std=np.full(MEL_SHAPE, std, dtype=np.float32),
    )


def shard_record_batch(columns: Mapping[str, np.ndarray]) -> pa.RecordBatch:
    """Encode column arrays as one record batch of fixed-shape tensor columns.

    :param columns: Mapping of column name to ``(num_rows, ...)`` array.
    :returns: Record batch carrying the shard schema the pipeline emits.
    """
    items = list(columns.items())
    fields = [
        pa.field(
            name,
            pa.fixed_shape_tensor(pa.from_numpy_dtype(data.dtype), data.shape[1:]),
            nullable=False,
        )
        for name, data in items
    ]
    schema = pa.schema(fields)
    return pa.record_batch(
        [tensor_array(data, data.dtype, data.shape[1:]) for _, data in items],
        schema=schema,
    )


def write_lance_shard(path: Path, columns: Mapping[str, np.ndarray]) -> None:
    """Write ``columns`` as a Lance dataset directory with one fixed-shape tensor column each.

    Goes through the pipeline's :func:`write_lance_dataset` so fixtures carry the
    exact on-disk format the finalize step emits.

    :param path: Output ``.lance`` dataset directory.
    :param columns: Mapping of column name to ``(num_rows, ...)`` array.
    """
    batch = shard_record_batch(columns)
    write_lance_dataset(path, batch.schema, [batch])


def write_lance_shard_with_sketch(
    path: Path, columns: Mapping[str, np.ndarray], sketch_controls: np.ndarray
) -> None:
    """Write a shard carrying the nested sketch struct column (#2707).

    :param path: Output ``.lance`` dataset directory.
    :param columns: Mapping of column name to ``(num_rows, ...)`` array.
    :param sketch_controls: ``(num_rows, NUM_SKETCH_CONTROLS, F)`` float32 stack
        split into the storage struct exactly as the add-embeddings writer does.
    """
    batch = shard_record_batch(columns)
    array = sketch_struct_array(sketch_controls)
    extended = batch.append_column(
        pa.field(SKETCH_STRUCT_FIELD, array.type, nullable=False), array
    )
    write_lance_dataset(path, extended.schema, [extended])


def wav_bytes(clip: np.ndarray, sample_rate: int) -> bytes:
    """Encode one mono clip as WAV container bytes.

    :param clip: ``(frames,)`` float32 samples.
    :param sample_rate: Encoded sample rate in Hz.
    :returns: WAV container bytes.
    """
    buffer = io.BytesIO()
    with AudioFile(buffer, "w", format="wav", samplerate=sample_rate, num_channels=1) as handle:
        handle.write(clip.reshape(1, -1).astype(np.float32))
    return buffer.getvalue()


def write_blob_audio_corpus(
    path: Path,
    clips: Sequence[np.ndarray],
    *,
    sample_rate: int,
    audio_column: str = AUDIO_FIELD,
    with_sample_rate_column: bool = False,
) -> None:
    """Write a third-party-style corpus storing WAV bytes in a blob column.

    Mirrors the published ``r2:experiments/third_party`` layout: source
    containers are stored verbatim, with no fixed-shape audio column.

    :param path: Destination Lance dataset.
    :param clips: One mono float32 clip per row.
    :param sample_rate: Encoded sample rate for every clip.
    :param audio_column: Blob column name; published corpora differ.
    :param with_sample_rate_column: Whether to store the rate alongside, as NSynth does
        and ESC50 does not.
    """

    def _wav_bytes(clip: np.ndarray) -> bytes:
        return wav_bytes(clip, sample_rate)

    fields = [
        pa.field(
            audio_column,
            pa.large_binary(),
            nullable=False,
            metadata={b"lance-encoding:blob": b"true"},
        )
    ]
    columns: dict[str, pa.Array] = {
        audio_column: pa.array([_wav_bytes(clip) for clip in clips], pa.large_binary())
    }
    if with_sample_rate_column:
        fields.append(pa.field("sample_rate", pa.int64(), nullable=False))
        columns["sample_rate"] = pa.array([sample_rate] * len(clips), pa.int64())
    table = pa.table(columns, schema=pa.schema(fields))
    lance.write_dataset(table, path, mode="create", data_storage_version="2.1")
