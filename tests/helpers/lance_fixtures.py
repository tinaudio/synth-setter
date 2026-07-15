"""Shared writers and column builders for Lance shard test fixtures."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyarrow as pa

from synth_setter.pipeline.data.lance_shard import tensor_array, write_lance_dataset

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


def write_lance_shard(path: Path, columns: Mapping[str, np.ndarray]) -> None:
    """Write ``columns`` as a Lance dataset directory with one fixed-shape tensor column each.

    Goes through the pipeline's :func:`write_lance_dataset` so fixtures carry the
    exact on-disk format the finalize step emits.

    :param path: Output ``.lance`` dataset directory.
    :param columns: Mapping of column name to ``(num_rows, ...)`` array.
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
    batch = pa.record_batch(
        [tensor_array(data, data.dtype, data.shape[1:]) for _, data in items],
        schema=schema,
    )
    write_lance_dataset(path, schema, [batch])
