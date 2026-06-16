"""Add ``audio_mp3`` and ``audio_uuid`` preview columns to a Lance dataset.

Backfills two derived columns onto a dataset written by
``synth-setter-generate-dataset`` or ``synth-setter-finalize-dataset``, in a
single Lance ``add_columns`` transaction:

- ``audio_mp3`` — each row's float16 ``audio`` tensor encoded to a CBR MP3
  (pedalboard) and stored as a Lance blob v2 column tagged
  ``mime_type: audio/mpeg``, so open-source Lance viewers auto-play a per-row
  preview. Blob v2 (storage version >= 2.2) keeps per-row reads lazy, so a scan
  need not materialize every MP3. A lossy auditioning preview, never a training
  input; the ``audio`` tensor stays the source of truth.
- ``audio_uuid`` — a deterministic :rfc:`4122` UUIDv5 fingerprint of the same
  ``audio`` tensor bytes, so the same rendered waveform always maps to the same
  id (content-addressed, stable across re-runs of this tool).
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path

import click
import lance
import numpy as np
import pyarrow as pa
from pedalboard.io import AudioFile

from synth_setter.data.vst.shapes import AUDIO_FIELD
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_shard import read_shard_metadata

AUDIO_MP3_FIELD = "audio_mp3"
AUDIO_UUID_FIELD = "audio_uuid"

DEFAULT_MP3_BITRATE_KBPS = 128

# Viewer hint so Lance UIs offer per-row playback; a repo convention, not a
# Lance-defined contract. Arrow field metadata is a bytes->bytes map.
_MP3_FIELD_METADATA: dict[bytes, bytes] = {b"mime_type": b"audio/mpeg"}

# Project-scoped UUIDv5 namespace, derived deterministically from a DNS name so
# the constant is reproducible and self-documenting (no opaque literal). UUIDv5
# hashes (namespace, name) with SHA-1, so a fixed namespace makes audio_uuid a
# pure function of the audio bytes.
_AUDIO_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "synth-setter.tinaudio.com")


def encode_audio_to_mp3(audio: np.ndarray, sample_rate: int, bitrate_kbps: int) -> bytes:
    """Encode one ``(channels, time)`` audio tensor to a CBR MP3 byte string.

    :param audio: One row of audio shaped ``(channels, time_samples)``; any
        float dtype (on-disk ``float16`` is upcast to ``float32`` for the
        encoder). Values outside ``[-1.0, 1.0]`` are clipped by the encoder.
    :param sample_rate: Playback rate in Hz the stream is encoded at.
    :param bitrate_kbps: Constant bitrate in kbps (pedalboard's ``quality``).
    :returns: The complete CBR MP3 bitstream encoded at ``bitrate_kbps``.
    :raises ValueError: ``audio`` is not 2-D ``(channels, time)`` with both axes non-empty.
    """
    if audio.ndim != 2 or 0 in audio.shape:
        raise ValueError(
            f"audio must be 2-D (channels, time) with non-empty axes, got shape {audio.shape}"
        )
    buffer = io.BytesIO()
    with AudioFile(
        buffer,
        "w",
        samplerate=sample_rate,
        num_channels=audio.shape[0],
        format="mp3",
        quality=str(bitrate_kbps),
    ) as out:
        # ascontiguousarray is a no-op when audio is already float32 and
        # C-contiguous, avoiding a redundant copy in that common case.
        out.write(np.ascontiguousarray(audio, dtype=np.float32))
    return buffer.getvalue()


def audio_uuid(audio: np.ndarray) -> str:
    """Compute the deterministic UUIDv5 fingerprint of one audio tensor.

    The id is a pure function of the raw on-disk ``audio`` bytes (dtype and
    shape included via ``tobytes``), so the same rendered waveform always yields
    the same uuid; a different render — even one sample — yields a different one.

    :param audio: One row of audio of any shape/dtype; hashed by its exact bytes.
    :returns: The canonical hyphenated UUIDv5 string under the project namespace.
    """
    # tobytes() always copies into C-contiguous order, so a non-contiguous view
    # hashes identically to its contiguous twin. hex() keeps the name a str
    # (uuid5 takes bytes names only on Python >= 3.12; the repo targets 3.11).
    return str(uuid.uuid5(_AUDIO_UUID_NAMESPACE, audio.tobytes().hex()))


def _encode_preview_columns(
    batch: pa.RecordBatch, sample_rate: int, bitrate_kbps: int
) -> pa.RecordBatch:
    """Derive the ``audio_mp3`` and ``audio_uuid`` columns from a batch's ``audio`` column.

    :param batch: Record batch projecting the ``audio`` fixed-shape tensor column.
    :param sample_rate: Playback rate in Hz every row is encoded at.
    :param bitrate_kbps: Constant MP3 bitrate in kbps for every row.
    :returns: A two-column batch (``audio_mp3`` blob array, ``audio_uuid`` string
        array), one cell per input row, in batch row order.
    :raises ValueError: The ``audio`` column is not a fixed-shape tensor column,
        or a row fails to encode (the message names the offending row index).
    """
    column = batch.column(AUDIO_FIELD)
    if not isinstance(column, pa.FixedShapeTensorArray):
        raise ValueError(f"{AUDIO_FIELD!r} must be a fixed-shape tensor column, got {column.type}")
    rows = column.to_numpy_ndarray()
    mp3_blobs = []
    uuids = []
    for row_index, row in enumerate(rows):
        try:
            mp3_blobs.append(encode_audio_to_mp3(row, sample_rate, bitrate_kbps))
        except (ValueError, RuntimeError, OSError) as exc:
            raise ValueError(
                f"failed to encode audio row {row_index} (shape {row.shape}): {exc}"
            ) from exc
        uuids.append(audio_uuid(row))
    return pa.record_batch(
        [lance.blob_array(mp3_blobs), pa.array(uuids, type=pa.string())],
        names=[AUDIO_MP3_FIELD, AUDIO_UUID_FIELD],
    )


def add_preview_columns(
    uri: Path | str,
    *,
    bitrate_kbps: int = DEFAULT_MP3_BITRATE_KBPS,
    storage_options: dict[str, str] | None = None,
) -> None:
    """Backfill ``audio_mp3`` and ``audio_uuid`` columns onto the Lance dataset at ``uri``.

    Commits a new dataset version with both added columns; the source ``audio``
    column and all others are left untouched. ``add_columns`` commits both
    columns in a single Lance transaction, so an interrupted run leaves the
    dataset on its prior version — re-running is safe.

    :param uri: Lance dataset directory (local path or ``s3://`` URI).
    :param bitrate_kbps: Applied uniformly; pedalboard takes it as a string ``quality``.
    :param storage_options: Object-store config for a cloud ``uri`` (see
        :func:`synth_setter.pipeline.r2_io.r2_storage_options`); ``None`` local.
    :raises ValueError: ``uri`` lacks an ``audio`` column, already has an
        ``audio_mp3`` or ``audio_uuid`` column, or carries no readable shard metadata.
    """
    dataset = lance.dataset(str(uri), storage_options=storage_options)
    if AUDIO_FIELD not in dataset.schema.names:
        raise ValueError(f"dataset at {uri} has no {AUDIO_FIELD!r} column to encode")
    existing = [f for f in (AUDIO_MP3_FIELD, AUDIO_UUID_FIELD) if f in dataset.schema.names]
    if existing:
        raise ValueError(f"dataset at {uri} already has preview column(s): {existing}")
    sample_rate = read_shard_metadata(dataset.schema).sample_rate
    preview_schema = pa.schema(
        [
            lance.blob_field(AUDIO_MP3_FIELD).with_metadata(_MP3_FIELD_METADATA),
            pa.field(AUDIO_UUID_FIELD, pa.string()),
        ]
    )

    # output_schema makes Lance skip the first-batch inference probe (which would
    # encode that batch twice) and carries the blob type + mime metadata onto the
    # new columns.
    @lance.batch_udf(output_schema=preview_schema)
    def _to_preview(batch: pa.RecordBatch) -> pa.RecordBatch:
        return _encode_preview_columns(batch, sample_rate, bitrate_kbps)

    dataset.add_columns(_to_preview, read_columns=[AUDIO_FIELD])


@click.command()
@click.argument("uri", type=str)
@click.option(
    "--bitrate-kbps",
    type=click.IntRange(min=8, max=320),
    default=DEFAULT_MP3_BITRATE_KBPS,
    show_default=True,
    help="Constant MP3 bitrate in kbps applied to every row (valid CBR range 8-320).",
)
def main(uri: str, bitrate_kbps: int) -> None:
    """Add ``audio_mp3`` and ``audio_uuid`` preview columns to the Lance dataset at ``URI``.

    URI is a ``.lance`` dataset directory written by ``synth-setter-generate-dataset``
    or ``synth-setter-finalize-dataset``.

    :param uri: A ``.lance`` dataset; a local path is used as-is, an ``r2://`` URI
        is rewritten to ``s3://``, and any ``s3://`` URI is treated as the project's
        R2 endpoint and credentialed with env-derived credentials (mirroring
        ``add_embeddings``; generic non-R2 S3 buckets are not a supported input).
    :param bitrate_kbps: Forwarded to :func:`add_preview_columns`; default shown in ``--help``.
    :raises click.ClickException: The dataset is missing its ``audio`` column,
        already has an ``audio_mp3`` or ``audio_uuid`` column, lacks readable
        shard metadata, cannot be opened (e.g. a cloud I/O error), or an R2 URI
        is given with missing/blank R2 credentials.
    """
    # Lance opens R2 over its S3-compatible API: rewrite r2:// to s3:// and treat
    # any s3:// as R2, passing env-derived credentials (mirroring add_embeddings).
    # R2 credential resolution raises RuntimeError on missing/blank env, so it
    # stays inside the try to surface as a clean ClickException.
    try:
        resolved_uri = r2_io.to_s3_uri(uri) if r2_io.is_r2_uri(uri) else uri
        storage_options: dict[str, str] | None = None
        if resolved_uri.startswith("s3://"):
            r2_io.ensure_r2_env_loaded()
            storage_options = r2_io.r2_storage_options()
        add_preview_columns(
            resolved_uri, bitrate_kbps=bitrate_kbps, storage_options=storage_options
        )
    except (ValueError, OSError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"added {AUDIO_MP3_FIELD!r} and {AUDIO_UUID_FIELD!r} columns to {uri}")


if __name__ == "__main__":
    main()
