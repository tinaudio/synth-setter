"""Add an ``audio_mp3`` MP3-preview column to a generate/finalize Lance dataset.

Encodes each row's float16 ``audio`` tensor to a CBR MP3 (pedalboard) and appends
it as a Lance blob v2 column tagged ``mime_type: audio/mpeg`` via Lance's
``batch_udf`` ``add_columns`` — backfilled in place. Blob v2 (dataset storage
version >= 2.2) keeps per-row reads lazy, so a scan need not materialize every
MP3. A lossy preview for auditioning patches, never a training input; the
``audio`` tensor stays the source of truth.
"""

from __future__ import annotations

import io
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

DEFAULT_MP3_BITRATE_KBPS = 128

# Viewer hint so Lance UIs offer per-row playback; a repo convention, not a
# Lance-defined contract. Arrow field metadata is a bytes->bytes map.
_MP3_FIELD_METADATA: dict[bytes, bytes] = {b"mime_type": b"audio/mpeg"}


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


def _encode_audio_column(batch: pa.RecordBatch, sample_rate: int, bitrate_kbps: int) -> pa.Array:
    """Encode a batch's ``audio`` tensor column into a Lance blob array of MP3 bytes.

    :param batch: Record batch projecting the ``audio`` fixed-shape tensor column.
    :param sample_rate: Playback rate in Hz every row is encoded at.
    :param bitrate_kbps: Constant MP3 bitrate in kbps for every row.
    :returns: One Lance blob array of per-row MP3 bytes, in batch row order.
    :raises ValueError: The ``audio`` column is not a fixed-shape tensor column,
        or a row fails to encode (the message names the offending row index).
    """
    column = batch.column(AUDIO_FIELD)
    if not isinstance(column, pa.FixedShapeTensorArray):
        raise ValueError(f"{AUDIO_FIELD!r} must be a fixed-shape tensor column, got {column.type}")
    rows = column.to_numpy_ndarray()
    encoded = []
    for row_index, row in enumerate(rows):
        try:
            encoded.append(encode_audio_to_mp3(row, sample_rate, bitrate_kbps))
        except (ValueError, RuntimeError, OSError) as exc:
            raise ValueError(
                f"failed to encode audio row {row_index} (shape {row.shape}): {exc}"
            ) from exc
    return lance.blob_array(encoded)


def add_mp3_audio_column(
    uri: Path | str,
    *,
    bitrate_kbps: int = DEFAULT_MP3_BITRATE_KBPS,
    storage_options: dict[str, str] | None = None,
) -> None:
    """Backfill an ``audio_mp3`` preview column onto the Lance dataset at ``uri``.

    Commits a new dataset version with the added column; the source ``audio``
    column and all others are left untouched. ``add_columns`` commits the new
    column in a single Lance transaction, so an interrupted run leaves the
    dataset on its prior version — re-running is safe.

    :param uri: Lance dataset directory (local path or ``s3://`` URI).
    :param bitrate_kbps: Applied uniformly; pedalboard takes it as a string ``quality``.
    :param storage_options: Object-store config for a cloud ``uri`` (see
        :func:`synth_setter.pipeline.r2_io.r2_storage_options`); ``None`` local.
    :raises ValueError: ``uri`` lacks an ``audio`` column, already has an
        ``audio_mp3`` column, or carries no readable shard metadata.
    """
    dataset = lance.dataset(str(uri), storage_options=storage_options)
    if AUDIO_FIELD not in dataset.schema.names:
        raise ValueError(f"dataset at {uri} has no {AUDIO_FIELD!r} column to encode")
    if AUDIO_MP3_FIELD in dataset.schema.names:
        raise ValueError(f"dataset at {uri} already has an {AUDIO_MP3_FIELD!r} column")
    sample_rate = read_shard_metadata(dataset.schema).sample_rate
    mp3_schema = pa.schema([lance.blob_field(AUDIO_MP3_FIELD).with_metadata(_MP3_FIELD_METADATA)])

    # output_schema makes Lance skip the first-batch inference probe (which would
    # encode that batch twice) and carries the blob type + mime metadata onto the
    # new column.
    @lance.batch_udf(output_schema=mp3_schema)
    def _to_mp3(batch: pa.RecordBatch) -> pa.RecordBatch:
        column = _encode_audio_column(batch, sample_rate, bitrate_kbps)
        return pa.record_batch([column], names=[AUDIO_MP3_FIELD])

    dataset.add_columns(_to_mp3, read_columns=[AUDIO_FIELD])


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
    """Add an ``audio_mp3`` preview column to the Lance dataset at ``URI``.

    URI is a ``.lance`` dataset directory written by ``synth-setter-generate-dataset``
    or ``synth-setter-finalize-dataset``.

    :param uri: A ``.lance`` dataset; a local path is used as-is, an ``r2://`` URI
        is rewritten to ``s3://``, and any ``s3://`` URI is treated as the project's
        R2 endpoint and credentialed with env-derived credentials (mirroring
        ``add_embeddings``; generic non-R2 S3 buckets are not a supported input).
    :param bitrate_kbps: Forwarded to :func:`add_mp3_audio_column`; default shown in ``--help``.
    :raises click.ClickException: The dataset is missing its ``audio`` column,
        already has an ``audio_mp3`` column, lacks readable shard metadata,
        cannot be opened (e.g. a cloud I/O error), or an R2 URI is given
        with missing/blank R2 credentials.
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
        add_mp3_audio_column(
            resolved_uri, bitrate_kbps=bitrate_kbps, storage_options=storage_options
        )
    except (ValueError, OSError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"added {AUDIO_MP3_FIELD!r} column to {uri}")


if __name__ == "__main__":
    main()
