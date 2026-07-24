"""Add ``audio_mp3`` and ``audio_uuid`` preview columns to a Lance dataset.

Backfills two derived columns onto a legacy dataset that predates initial-write
previews, in a single Lance ``add_columns`` transaction. Current
``synth-setter-generate-dataset`` output already carries both columns:

- ``audio_mp3`` — each row's ``audio`` tensor encoded to a CBR MP3 (pedalboard),
  stored as an Arrow binary column tagged ``mime_type: audio/mpeg`` so viewers
  can scan the table through DuckDB without Lance blob support.
- ``audio_uuid`` — a deterministic UUIDv5 fingerprint of the same ``audio`` bytes,
  so the same rendered waveform always maps to the same id (content-addressed).

Neither column is a training input; the ``audio`` tensor stays the source of truth.
"""

from __future__ import annotations

from pathlib import Path

import click
import lance
import pyarrow as pa
import structlog

from synth_setter.data.vst.generate_vst_dataset import (
    DEFAULT_MP3_BITRATE_KBPS,
    audio_uuid,
    encode_audio_to_mp3,
)
from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    AUDIO_MP3_FIELD,
    AUDIO_UUID_FIELD,
)
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_shard import PREVIEW_SCHEMA, read_shard_metadata

logger = structlog.get_logger(__name__)

def _encode_preview_columns(
    batch: pa.RecordBatch, sample_rate: int, bitrate_kbps: int
) -> pa.RecordBatch:
    """Derive the ``audio_mp3`` and ``audio_uuid`` columns from a batch's ``audio`` column.

    :param batch: Record batch projecting the ``audio`` fixed-shape tensor column.
    :param sample_rate: Playback rate in Hz every row is encoded at.
    :param bitrate_kbps: Constant MP3 bitrate in kbps for every row.
    :returns: A two-column batch (``audio_mp3`` binary array, ``audio_uuid`` string
        array), one cell per input row, in batch row order.
    :raises ValueError: The ``audio`` column is not a fixed-shape tensor column,
        or a row fails to encode (the message names the offending row index).
    """
    column = batch.column(AUDIO_FIELD)
    if not isinstance(column, pa.FixedShapeTensorArray):
        raise ValueError(f"{AUDIO_FIELD!r} must be a fixed-shape tensor column, got {column.type}")
    rows = column.to_numpy_ndarray()
    mp3_payloads = []
    uuids = []
    for row_index, row in enumerate(rows):
        try:
            mp3 = encode_audio_to_mp3(row, sample_rate, bitrate_kbps)
        except (ValueError, RuntimeError, OSError) as exc:
            raise ValueError(
                f"failed to encode audio row {row_index} (shape {row.shape}): {exc}"
            ) from exc
        # Append as a pair only after the encode succeeds, so a failed row never
        # leaves the two columns at mismatched lengths.
        mp3_payloads.append(mp3)
        uuids.append(audio_uuid(row))
    return pa.record_batch(
        [pa.array(mp3_payloads, type=pa.binary()), pa.array(uuids, type=pa.string())],
        schema=PREVIEW_SCHEMA,
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
    # output_schema skips Lance's first-batch inference probe (avoids a double
    # encode) and carries the binary type + mime metadata onto the new columns.
    @lance.batch_udf(output_schema=PREVIEW_SCHEMA)
    def _to_preview(batch: pa.RecordBatch) -> pa.RecordBatch:
        return _encode_preview_columns(batch, sample_rate, bitrate_kbps)

    dataset.add_columns(_to_preview, read_columns=[AUDIO_FIELD])
    logger.info(
        "added_preview_columns",
        uri=str(uri),
        columns=[AUDIO_MP3_FIELD, AUDIO_UUID_FIELD],
        rows=dataset.count_rows(),
    )


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
    :param bitrate_kbps: See :func:`add_preview_columns` (Click validates the 8-320 range).
    :raises click.ClickException: The dataset is missing its ``audio`` column,
        already has an ``audio_mp3`` or ``audio_uuid`` column, lacks readable
        shard metadata, cannot be opened (e.g. a cloud I/O error), or an R2 URI
        is given with missing/blank R2 credentials.
    """
    # R2 credential resolution raises RuntimeError on missing/blank env, so keep
    # it inside the try to surface as a clean ClickException (mirrors add_embeddings).
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
