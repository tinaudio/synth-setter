"""Tests for the Lance CLAP-embedding augmentation core.

The encoder is injected as a plain callable, so every path here runs without
loading a CLAP checkpoint; :func:`load_clap_audio_encoder` (the real model
shell) is exercised manually — see the module docstring of ``clap_lance``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from lance.file import LanceFileReader

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    CLAP_FIELD,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
)
from synth_setter.pipeline.data.clap_lance import (
    EncodeFn,
    clap_augmented_schema,
    iter_clap_batches,
)
from synth_setter.pipeline.data.lance_shard import iter_lance_column_rows, write_lance_file

# Small embedding width so fixtures stay tiny; unrelated to the real 512-d model.
_TEST_DIM = 4
_SAMPLE_RATE = 16000


def _write_shard(path: Path, audio: np.ndarray, mel: np.ndarray, params: np.ndarray) -> None:
    """Write a single-file Lance shard carrying the three core dataset columns.

    :param path: Output ``.lance`` shard file.
    :param audio: ``(N, C, T)`` audio rows.
    :param mel: ``(N, ...)`` mel-spectrogram rows.
    :param params: ``(N, P)`` parameter rows.
    """
    from tests.helpers.lance_fixtures import write_lance_shard

    write_lance_shard(
        path,
        {AUDIO_FIELD: audio, MEL_SPEC_FIELD: mel, PARAM_ARRAY_FIELD: params},
    )


def _read_schema_and_batches(path: Path) -> tuple[pa.Schema, list[pa.RecordBatch]]:
    """Read a Lance file's schema and all record batches, as finalize does.

    :param path: Lance shard file to read.
    :returns: The file's schema and its record batches.
    :rtype: tuple[pa.Schema, list[pa.RecordBatch]]
    """
    reader = LanceFileReader(str(path))
    return reader.metadata().schema, reader.read_all().to_batches()


def _constant_audio(num_rows: int, channels: int, time: int) -> np.ndarray:
    """Build an audio batch whose channel ``c`` is filled with ``c + 1``.

    :param num_rows: Row count ``N``.
    :param channels: Channel count ``C``.
    :param time: Sample count ``T``.
    :returns: ``(N, C, T)`` float16 audio.
    :rtype: np.ndarray
    """
    rows = np.empty((num_rows, channels, time), dtype=np.float16)
    for channel in range(channels):
        rows[:, channel, :] = channel + 1
    return rows


def _row_indexed_encoder(dim: int) -> EncodeFn:
    """Build a deterministic encoder mapping a mono batch to ``mean(row) + arange(dim)``.

    :param dim: Embedding width the encoder emits.
    :returns: An encode callable producing distinct, reproducible rows.
    :rtype: EncodeFn
    """

    def encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:  # noqa: ARG001
        return (mono.mean(axis=1, keepdims=True) + np.arange(dim)[None, :]).astype(np.float32)

    return encode


def test_clap_augmented_schema_appends_fixed_shape_tensor_clap_column(tmp_path: Path) -> None:
    """The augmented schema keeps every original column and adds a ``(dim,)`` float32 ``clap``.

    :param tmp_path: Pytest tmp dir hosting the fixture shard.
    """
    _write_shard(
        tmp_path / "shard.lance",
        audio=_constant_audio(2, 2, 8),
        mel=np.zeros((2, 2, 3), dtype=np.float32),
        params=np.zeros((2, 5), dtype=np.float32),
    )
    schema, _ = _read_schema_and_batches(tmp_path / "shard.lance")

    augmented = clap_augmented_schema(schema, _TEST_DIM)

    assert augmented.names == [AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD, CLAP_FIELD]
    clap_type = augmented.field(CLAP_FIELD).type
    assert tuple(clap_type.shape) == (_TEST_DIM,)
    assert clap_type.value_type == pa.float32()


def test_clap_augmented_schema_preserves_shard_metadata() -> None:
    """Schema-level metadata (ShardMetadata payload) survives the column append."""
    base = pa.schema(
        [pa.field(AUDIO_FIELD, pa.fixed_shape_tensor(pa.float16(), (2, 8)), nullable=False)],
        metadata={b"synth_setter.shard_metadata": b'{"sample_rate": 16000}'},
    )

    augmented = clap_augmented_schema(base, _TEST_DIM)

    assert augmented.metadata == base.metadata


def test_iter_clap_batches_writes_encoder_output_as_clap_column(tmp_path: Path) -> None:
    """Each row's ``clap`` column equals the injected encoder's output for that row's audio.

    :param tmp_path: Pytest tmp dir hosting the input and output shards.
    """
    audio = _constant_audio(num_rows=3, channels=2, time=8)
    _write_shard(
        tmp_path / "in.lance",
        audio=audio,
        mel=np.zeros((3, 2, 3), dtype=np.float32),
        params=np.zeros((3, 5), dtype=np.float32),
    )
    schema, batches = _read_schema_and_batches(tmp_path / "in.lance")

    out_schema = clap_augmented_schema(schema, _TEST_DIM)
    out_batches = iter_clap_batches(
        schema, batches, _row_indexed_encoder(_TEST_DIM), _SAMPLE_RATE, dim=_TEST_DIM
    )
    write_lance_file(tmp_path / "out.lance", out_schema, out_batches)

    clap_rows = list(iter_lance_column_rows(tmp_path / "out.lance", CLAP_FIELD))
    # Every channel mean is (1+2)/2 = 1.5, so each row is 1.5 + arange(dim).
    expected_row = (1.5 + np.arange(_TEST_DIM)).astype(np.float32)
    assert len(clap_rows) == 3
    for row in clap_rows:
        np.testing.assert_array_equal(row, expected_row)


def test_iter_clap_batches_downmixes_channels_to_mono_before_encoding(tmp_path: Path) -> None:
    """The encoder receives a mono ``(B, T)`` batch averaged across the channel axis.

    :param tmp_path: Pytest tmp dir hosting the input shard.
    """
    audio = _constant_audio(num_rows=2, channels=2, time=4)
    _write_shard(
        tmp_path / "in.lance",
        audio=audio,
        mel=np.zeros((2, 2, 3), dtype=np.float32),
        params=np.zeros((2, 5), dtype=np.float32),
    )
    schema, batches = _read_schema_and_batches(tmp_path / "in.lance")
    seen: list[np.ndarray] = []

    def recording_encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:  # noqa: ARG001
        seen.append(mono.copy())
        return np.zeros((mono.shape[0], _TEST_DIM), dtype=np.float32)

    list(iter_clap_batches(schema, batches, recording_encode, _SAMPLE_RATE, dim=_TEST_DIM))

    assert seen[0].shape == (2, 4)
    np.testing.assert_array_equal(seen[0], np.full((2, 4), 1.5, dtype=np.float32))


def test_iter_clap_batches_passes_sample_rate_to_encoder(tmp_path: Path) -> None:
    """The sample rate threaded into the iterator reaches the encoder unchanged.

    :param tmp_path: Pytest tmp dir hosting the input shard.
    """
    _write_shard(
        tmp_path / "in.lance",
        audio=_constant_audio(1, 2, 4),
        mel=np.zeros((1, 2, 3), dtype=np.float32),
        params=np.zeros((1, 5), dtype=np.float32),
    )
    schema, batches = _read_schema_and_batches(tmp_path / "in.lance")
    seen_rates: list[int] = []

    def recording_encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:
        seen_rates.append(sample_rate)
        return np.zeros((mono.shape[0], _TEST_DIM), dtype=np.float32)

    list(iter_clap_batches(schema, batches, recording_encode, 22050, dim=_TEST_DIM))

    assert seen_rates == [22050]


def test_iter_clap_batches_raises_when_encoder_width_differs_from_dim(tmp_path: Path) -> None:
    """A width guard rejects an encoder whose output dimension is not the declared ``dim``.

    :param tmp_path: Pytest tmp dir hosting the input shard.
    """
    _write_shard(
        tmp_path / "in.lance",
        audio=_constant_audio(2, 2, 4),
        mel=np.zeros((2, 2, 3), dtype=np.float32),
        params=np.zeros((2, 5), dtype=np.float32),
    )
    schema, batches = _read_schema_and_batches(tmp_path / "in.lance")

    def wrong_width_encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:  # noqa: ARG001
        return np.zeros((mono.shape[0], _TEST_DIM + 1), dtype=np.float32)

    with pytest.raises(ValueError, match="width"):
        list(iter_clap_batches(schema, batches, wrong_width_encode, _SAMPLE_RATE, dim=_TEST_DIM))


def test_iter_clap_batches_preserves_existing_columns(tmp_path: Path) -> None:
    """Augmentation is additive: the original audio/mel/param rows round-trip unchanged.

    :param tmp_path: Pytest tmp dir hosting the input and output shards.
    """
    mel = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    params = np.arange(2 * 5, dtype=np.float32).reshape(2, 5)
    _write_shard(
        tmp_path / "in.lance",
        audio=_constant_audio(2, 2, 4),
        mel=mel,
        params=params,
    )
    schema, batches = _read_schema_and_batches(tmp_path / "in.lance")

    out_schema = clap_augmented_schema(schema, _TEST_DIM)
    out_batches = iter_clap_batches(
        schema, batches, _row_indexed_encoder(_TEST_DIM), _SAMPLE_RATE, dim=_TEST_DIM
    )
    write_lance_file(tmp_path / "out.lance", out_schema, out_batches)

    mel_rows = list(iter_lance_column_rows(tmp_path / "out.lance", MEL_SPEC_FIELD))
    param_rows = list(iter_lance_column_rows(tmp_path / "out.lance", PARAM_ARRAY_FIELD))
    np.testing.assert_array_equal(np.stack(mel_rows), mel)
    np.testing.assert_array_equal(np.stack(param_rows), params)
