"""Behavior tests for the MP3-audio column adder.

Encoded bytes are validated structurally (MP3 frame sync, decoded sample rate / channel count)
rather than byte-for-byte, since LAME output is not promised to be bit-reproducible across
pedalboard builds. The decode side uses pedalboard's reader — never the encoder under test — so a
sample-rate or channel bug can't corrupt both sides identically and pass.
"""

from __future__ import annotations

import io
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
from click.testing import CliRunner
from pedalboard.io import AudioFile

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
)
from synth_setter.pipeline.data.add_mp3_audio import (
    AUDIO_MP3_FIELD,
    add_mp3_audio_column,
    encode_audio_to_mp3,
    main,
)
from synth_setter.pipeline.data.lance_shard import (
    lance_schema,
    record_batch_from_arrays,
    write_lance_dataset,
)
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata

_SAMPLE_RATE = 8000
_CHANNELS = 2
_DURATION_SECONDS = 0.1
_TIME_SAMPLES = int(_SAMPLE_RATE * _DURATION_SECONDS)
_ROWS = 3

# A full second at CD rate: enough frames that the 320-vs-128 kbps size
# difference shows. Sub-second / low-rate clips are dominated by padding.
_BITRATE_PROBE_RATE = 44100

# Concert-pitch A; an audible test tone, value not otherwise significant.
_A4_HZ = 440.0

# Source columns the backfill must preserve untouched.
_SOURCE_FIELDS = (AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD)

# Inner shapes (without the leading row axis) are arbitrary for mel/param here —
# only ``audio`` is read by the column adder.
_FIELD_SHAPES: dict[str, tuple[int, ...]] = {
    AUDIO_FIELD: (_ROWS, _CHANNELS, _TIME_SAMPLES),
    MEL_SPEC_FIELD: (_ROWS, 2, 3, 4),
    PARAM_ARRAY_FIELD: (_ROWS, 7),
}


def _sine_rows(sample_rate: int = _SAMPLE_RATE) -> np.ndarray:
    """Build ``(_ROWS, _CHANNELS, T)`` float16 sine audio, one pitch per row.

    :param sample_rate: Rate used to space the time axis so the tone is real.
    :returns: Float16 audio with a distinct frequency per row.
    """
    t = np.arange(_TIME_SAMPLES) / sample_rate
    rows = [np.sin(2 * np.pi * freq * t) for freq in (220.0, 440.0, 660.0)]
    mono = np.stack(rows)[:, None, :]
    # broadcast_to is a read-only view, but the trailing .astype materializes a
    # fresh writable float16 array, so the returned value is safe to mutate.
    return np.broadcast_to(mono, (_ROWS, _CHANNELS, _TIME_SAMPLES)).astype(np.float16)


def _write_smoke_dataset(path: Path, *, sample_rate: int = _SAMPLE_RATE) -> None:
    """Write a smoke Lance dataset: real sine ``audio`` plus zeroed mel/param tensors.

    :param path: Destination ``.lance`` directory.
    :param sample_rate: Sample rate embedded in the shard metadata.
    """
    metadata = ShardMetadata(
        velocity=100,
        # Derive from the fixed frame count so duration stays consistent with the
        # written tensor at any sample_rate (_TIME_SAMPLES is sized for 8 kHz).
        signal_duration_seconds=_TIME_SAMPLES / sample_rate,
        sample_rate=sample_rate,
        channels=_CHANNELS,
        min_loudness=-55.0,
    )
    schema = lance_schema(_FIELD_SHAPES, metadata)
    arrays = {
        AUDIO_FIELD: _sine_rows(sample_rate),
        MEL_SPEC_FIELD: np.zeros(_FIELD_SHAPES[MEL_SPEC_FIELD], dtype=np.float32),
        PARAM_ARRAY_FIELD: np.zeros(_FIELD_SHAPES[PARAM_ARRAY_FIELD], dtype=np.float32),
    }
    write_lance_dataset(path, schema, [record_batch_from_arrays(arrays, schema)])


def _decode_mp3(payload: bytes) -> tuple[np.ndarray, int]:
    """Decode MP3 bytes back to samples via pedalboard's reader.

    :param payload: A complete MP3 bitstream as produced by ``encode_audio_to_mp3``.
    :returns: ``(samples, sample_rate)`` with ``samples`` shaped ``(channels, frames)``.
    """
    with AudioFile(io.BytesIO(payload)) as f:
        return f.read(f.frames), int(f.samplerate)


def _read_mp3_blobs(uri: Path, indices: list[int]) -> list[bytes]:
    """Read back ``audio_mp3`` blob cells as raw MP3 bytes via Lance's blob API.

    :param uri: The ``.lance`` dataset directory.
    :param indices: Row indices to fetch, in the order returned.
    :returns: Per-row MP3 byte strings, in ``indices`` order.
    """
    dataset = lance.dataset(str(uri))
    return [
        blob.readall() for blob in dataset.take_blobs(blob_column=AUDIO_MP3_FIELD, indices=indices)
    ]


def test_encode_audio_to_mp3_contains_frame_sync() -> None:
    """A valid MP3 stream carries the 11-bit frame-sync word (0xFF 0xEy/0xFy).

    Scans the head rather than asserting offset 0: an MP3 may legally lead with
    an ID3 tag or padding before the first frame.
    """
    payload = encode_audio_to_mp3(_sine_rows()[0], _SAMPLE_RATE, 128)

    assert len(payload) > 0
    head = payload[:1024]
    assert any(head[i] == 0xFF and head[i + 1] & 0xE0 == 0xE0 for i in range(len(head) - 1))


def test_encode_audio_to_mp3_higher_bitrate_yields_larger_payload() -> None:
    """320 kbps encodes the same signal to more bytes than 128 kbps.

    Uses a full second of 44.1 kHz audio: a sub-second low-rate clip is dominated by header/padding
    frames that mask the bitrate difference.
    """
    t = np.arange(_BITRATE_PROBE_RATE) / _BITRATE_PROBE_RATE
    row = np.sin(2 * np.pi * _A4_HZ * t).astype(np.float16)[None, :]

    assert len(encode_audio_to_mp3(row, _BITRATE_PROBE_RATE, 320)) > len(
        encode_audio_to_mp3(row, _BITRATE_PROBE_RATE, 128)
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_encode_audio_to_mp3_accepts_any_float_dtype(dtype: type[np.floating]) -> None:
    """The documented "any float dtype" contract holds for float16/32/64 input.

    :param dtype: Float dtype the audio row is built with.
    """
    t = np.arange(_TIME_SAMPLES) / _SAMPLE_RATE
    row = np.sin(2 * np.pi * _A4_HZ * t).astype(dtype)[None, :]

    payload = encode_audio_to_mp3(row, _SAMPLE_RATE, 128)

    assert payload[0] == 0xFF


def test_encode_audio_to_mp3_mono_input_decodes_to_one_channel() -> None:
    """A mono ``(1, T)`` row (mono-only plugins) encodes and decodes to one channel."""
    t = np.arange(_TIME_SAMPLES) / _SAMPLE_RATE
    mono = np.sin(2 * np.pi * _A4_HZ * t).astype(np.float16)[None, :]

    samples, _ = _decode_mp3(encode_audio_to_mp3(mono, _SAMPLE_RATE, 128))

    assert samples.shape[0] == 1


def test_encode_audio_to_mp3_silent_input_yields_valid_stream() -> None:
    """All-zero audio (a silent patch) still encodes to a finite-decoding MP3 stream."""
    silence = np.zeros((_CHANNELS, _TIME_SAMPLES), dtype=np.float16)

    samples, _ = _decode_mp3(encode_audio_to_mp3(silence, _SAMPLE_RATE, 128))

    assert np.isfinite(samples).all()


@pytest.mark.parametrize(
    "shape",
    [(_TIME_SAMPLES,), (0, _TIME_SAMPLES), (_CHANNELS, 0)],
    ids=["1d-no-channel-axis", "empty-channel-axis", "empty-time-axis"],
)
def test_encode_audio_to_mp3_malformed_shape_raises(shape: tuple[int, ...]) -> None:
    """Non-2-D or empty-axis audio is rejected before reaching the encoder.

    :param shape: A malformed audio shape the guard must reject.
    """
    with pytest.raises(ValueError, match="2-D"):
        encode_audio_to_mp3(np.zeros(shape, dtype=np.float16), _SAMPLE_RATE, 128)


@pytest.mark.parametrize("sample_rate", [8000, 16000, 44100])
def test_encode_audio_to_mp3_preserves_sample_rate_and_channels(sample_rate: int) -> None:
    """The decoded stream reports the sample rate and channel count it was encoded with.

    :param sample_rate: Rate the row is encoded at and expected to decode back to.
    """
    t = np.arange(_TIME_SAMPLES) / sample_rate
    row = np.broadcast_to(
        np.sin(2 * np.pi * _A4_HZ * t)[None, :], (_CHANNELS, _TIME_SAMPLES)
    ).astype(np.float16)

    samples, decoded_rate = _decode_mp3(encode_audio_to_mp3(row, sample_rate, 128))

    assert decoded_rate == sample_rate
    assert samples.shape[0] == _CHANNELS


def test_add_mp3_audio_column_adds_blob_column_for_every_row(tmp_path: Path) -> None:
    """Every row gains a decodable ``audio_mp3`` blob cell; source columns and row count unchanged.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    uri = tmp_path / "shard-000000.lance"
    _write_smoke_dataset(uri)
    before = {f: lance.dataset(str(uri)).schema.field(f).type for f in _SOURCE_FIELDS}

    add_mp3_audio_column(uri)

    ds = lance.dataset(str(uri))
    for field in _SOURCE_FIELDS:
        assert ds.schema.field(field).type == before[field]
    assert ds.schema.field(AUDIO_MP3_FIELD).type == lance.blob_field(AUDIO_MP3_FIELD).type
    assert ds.count_rows() == _ROWS
    payloads = _read_mp3_blobs(uri, list(range(_ROWS)))
    assert all(len(p) > 0 for p in payloads)
    # Decode one cell to confirm the batch_udf path emits a real MP3, not just bytes.
    samples, _ = _decode_mp3(payloads[0])
    assert samples.shape[0] == _CHANNELS


def test_add_mp3_audio_column_tags_field_with_audio_mime_type(tmp_path: Path) -> None:
    """The added column carries ``mime_type: audio/mpeg`` so Lance viewers auto-play it.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    uri = tmp_path / "shard-000000.lance"
    _write_smoke_dataset(uri)

    add_mp3_audio_column(uri)

    field = lance.dataset(str(uri)).schema.field(AUDIO_MP3_FIELD)
    assert field.metadata == {b"mime_type": b"audio/mpeg"}


def test_add_mp3_audio_column_uses_sample_rate_from_metadata(tmp_path: Path) -> None:
    """The encoded column honors the shard metadata's sample rate, not a hardcoded value.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    uri = tmp_path / "shard-000000.lance"
    _write_smoke_dataset(uri, sample_rate=16000)

    add_mp3_audio_column(uri)

    first = _read_mp3_blobs(uri, [0])[0]
    _, decoded_rate = _decode_mp3(first)
    assert decoded_rate == 16000


def test_add_mp3_audio_column_existing_column_raises(tmp_path: Path) -> None:
    """A second add onto a dataset that already has ``audio_mp3`` fails fast.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    uri = tmp_path / "shard-000000.lance"
    _write_smoke_dataset(uri)
    add_mp3_audio_column(uri)

    with pytest.raises(ValueError, match=AUDIO_MP3_FIELD):
        add_mp3_audio_column(uri)


def test_add_mp3_audio_column_missing_audio_column_raises(tmp_path: Path) -> None:
    """A dataset without the source audio column raises a clear error.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    # The audio-column guard precedes the metadata read, so no shard metadata
    # is needed to reach it.
    uri = tmp_path / "no-audio.lance"
    lance.write_dataset(pa.table({"other": [1, 2]}), str(uri), mode="overwrite")

    with pytest.raises(ValueError, match=AUDIO_FIELD):
        add_mp3_audio_column(uri)


def test_main_adds_column_and_reports_success(tmp_path: Path) -> None:
    """The CLI entrypoint encodes the column and echoes a success line.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    uri = tmp_path / "shard-000000.lance"
    _write_smoke_dataset(uri)

    result = CliRunner().invoke(main, [str(uri)])

    assert result.exit_code == 0
    assert AUDIO_MP3_FIELD in result.output
    assert AUDIO_MP3_FIELD in lance.dataset(str(uri)).schema.names


def test_main_existing_column_exits_nonzero_with_message(tmp_path: Path) -> None:
    """A re-run via the CLI surfaces the duplicate-column error as a non-zero exit.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    uri = tmp_path / "shard-000000.lance"
    _write_smoke_dataset(uri)
    add_mp3_audio_column(uri)

    result = CliRunner().invoke(main, [str(uri)])

    assert result.exit_code != 0
    assert AUDIO_MP3_FIELD in result.output


def test_add_mp3_audio_column_missing_shard_metadata_raises(tmp_path: Path) -> None:
    """An ``audio`` column with no embedded ShardMetadata raises a clear error.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    uri = tmp_path / "no-metadata.lance"
    audio = pa.FixedShapeTensorArray.from_numpy_ndarray(_sine_rows())
    # Plain schema with no SHARD_METADATA_SCHEMA_KEY, so read_shard_metadata fails.
    lance.write_dataset(pa.table({AUDIO_FIELD: audio}), str(uri), mode="overwrite")

    with pytest.raises(ValueError, match="metadata"):
        add_mp3_audio_column(uri)


def test_main_rejects_out_of_range_bitrate(tmp_path: Path) -> None:
    """A non-positive ``--bitrate-kbps`` is rejected at the Click layer, not the encoder.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    uri = tmp_path / "shard-000000.lance"
    _write_smoke_dataset(uri)

    result = CliRunner().invoke(main, [str(uri), "--bitrate-kbps", "0"])

    assert result.exit_code != 0
    assert "bitrate-kbps" in result.output


def test_main_bitrate_option_threads_through(tmp_path: Path) -> None:
    """The ``--bitrate-kbps`` option is accepted and the column is still produced.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    uri = tmp_path / "shard-000000.lance"
    _write_smoke_dataset(uri)

    result = CliRunner().invoke(main, [str(uri), "--bitrate-kbps", "64"])

    assert result.exit_code == 0
    assert AUDIO_MP3_FIELD in lance.dataset(str(uri)).schema.names


def test_main_rewrites_r2_uri_and_forwards_storage_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ``r2://`` URI is rewritten to ``s3://`` and env credentials reach the column adder.

    :param monkeypatch: Pytest fixture stubbing the r2_io helpers and the column adder.
    """
    captured: dict[str, object] = {}
    monkeypatch.setattr("synth_setter.pipeline.r2_io.is_r2_uri", lambda uri: True)
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.to_s3_uri", lambda uri: "s3://bucket/key.lance"
    )
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda: None)
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.r2_storage_options", lambda: {"aws_secret": "x"}
    )

    def _spy(uri: str, *, bitrate_kbps: int, storage_options: dict[str, str] | None) -> None:
        captured.update(uri=uri, bitrate_kbps=bitrate_kbps, storage_options=storage_options)

    monkeypatch.setattr("synth_setter.pipeline.data.add_mp3_audio.add_mp3_audio_column", _spy)

    result = CliRunner().invoke(main, ["r2://bucket/key.lance", "--bitrate-kbps", "192"])

    assert result.exit_code == 0
    assert captured == {
        "uri": "s3://bucket/key.lance",
        "bitrate_kbps": 192,
        "storage_options": {"aws_secret": "x"},
    }


def test_main_credentials_bare_s3_uri_as_r2(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare ``s3://`` URI is treated as R2 and credentialed, matching add_embeddings.

    R2 datasets are commonly referenced as ``s3://`` in this repo, so an ``s3://``
    input must still receive ``r2_storage_options`` rather than relying on ambient
    AWS env config.

    :param monkeypatch: Pytest fixture stubbing the r2_io helpers and the column adder.
    """
    captured: dict[str, object] = {}
    monkeypatch.setattr("synth_setter.pipeline.r2_io.is_r2_uri", lambda uri: False)
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda: None)
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.r2_storage_options", lambda: {"aws_secret": "x"}
    )

    def _spy(uri: str, *, bitrate_kbps: int, storage_options: dict[str, str] | None) -> None:
        captured.update(uri=uri, bitrate_kbps=bitrate_kbps, storage_options=storage_options)

    monkeypatch.setattr("synth_setter.pipeline.data.add_mp3_audio.add_mp3_audio_column", _spy)

    result = CliRunner().invoke(main, ["s3://bucket/key.lance"])

    assert result.exit_code == 0
    assert captured["uri"] == "s3://bucket/key.lance"
    assert captured["storage_options"] == {"aws_secret": "x"}


def test_main_local_path_passes_no_storage_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """A local path is opened without storage options.

    :param monkeypatch: Pytest fixture stubbing the r2_io helpers and the column adder.
    """
    captured: dict[str, object] = {}
    monkeypatch.setattr("synth_setter.pipeline.r2_io.is_r2_uri", lambda uri: False)

    def _spy(uri: str, *, bitrate_kbps: int, storage_options: dict[str, str] | None) -> None:
        captured.update(uri=uri, storage_options=storage_options)

    monkeypatch.setattr("synth_setter.pipeline.data.add_mp3_audio.add_mp3_audio_column", _spy)

    result = CliRunner().invoke(main, ["/local/path/key.lance"])

    assert result.exit_code == 0
    assert captured == {"uri": "/local/path/key.lance", "storage_options": None}
