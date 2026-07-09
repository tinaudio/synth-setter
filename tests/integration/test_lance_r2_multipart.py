"""End-to-end Lance multipart writes against real Cloudflare R2."""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Iterator

import lance
import pyarrow as pa
import pytest

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_shard import write_lance_dataset

pytestmark = [pytest.mark.integration_r2, pytest.mark.r2, pytest.mark.slow]

_R2_BUCKET = "intermediate-data"
_LANCE_UPLOAD_PART_BYTES = 5 * 1024 * 1024
# Lance grows multipart part size after 100 parts unless it detects R2.
# 103 one-row batches cross that boundary and reproduce R2 InvalidPart pre-fix.
_MULTIPART_REGRESSION_ROWS = 103


def _unique_test_prefix() -> str:
    """Build a purge-safe R2 prefix for one Lance multipart regression run.

    :returns: Trailing-slash-terminated R2 key prefix.
    """
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "0")
    return f"ci-lance-r2-multipart/{run_id}/{run_attempt}/{uuid.uuid4().hex[:8]}/"


def _large_binary_batch(schema: pa.Schema, row_id: int) -> pa.RecordBatch:
    """Build one deterministic non-compressible batch for a row id.

    :param schema: Arrow schema shared by every emitted record batch.
    :param row_id: Stable integer seed for the row payload.
    :returns: A one-row batch with a 5 MiB binary payload.
    """
    seed = row_id.to_bytes(length=4, byteorder="little", signed=False)
    payload = hashlib.shake_256(seed).digest(_LANCE_UPLOAD_PART_BYTES)
    return pa.record_batch([pa.array([payload], type=pa.binary())], schema=schema)


def _large_binary_batches(schema: pa.Schema) -> Iterator[pa.RecordBatch]:
    """Return streaming batches for a multipart Lance data file.

    :param schema: Arrow schema shared by every emitted record batch.
    :returns: Iterator of deterministic one-row batches.
    """
    return (_large_binary_batch(schema, row_id) for row_id in range(_MULTIPART_REGRESSION_ROWS))


def test_lance_write_dataset_large_multipart_object_completes_on_real_r2() -> None:
    """A large Lance data file writes to real R2 without ``InvalidPart``.

    Exercises the pull-based ``write_lance_dataset`` writer against R2 with a
    payload size that crosses Lance's multipart part-growth boundary. The
    fragment path production staging uses (``lance_fragment``) has no
    ``max_bytes_per_file`` knob; its bound is enforced up front by
    ``stage_lance_shard_attempt``'s shard-size guard instead.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip(
            "R2 not reachable (rclone missing, RCLONE_CONFIG_R2_* env vars missing, "
            "or rclone lsd r2: failed)"
        )
    r2_io.ensure_r2_env_loaded()
    previous_upload_size = os.environ.get("LANCE_INITIAL_UPLOAD_SIZE")
    os.environ["LANCE_INITIAL_UPLOAD_SIZE"] = str(_LANCE_UPLOAD_PART_BYTES)

    prefix = _unique_test_prefix()
    dataset_r2_uri = f"r2://{_R2_BUCKET}/{prefix}regression.lance"
    dataset_s3_uri = r2_io.to_s3_uri(dataset_r2_uri)
    schema = pa.schema([pa.field("payload", pa.binary(), nullable=False)])

    try:
        write_lance_dataset(
            dataset_s3_uri,
            schema,
            _large_binary_batches(schema),
            storage_options=r2_io.r2_storage_options(),
        )

        dataset = lance.dataset(dataset_s3_uri, storage_options=r2_io.r2_storage_options())
        assert dataset.count_rows() == _MULTIPART_REGRESSION_ROWS
    finally:
        if previous_upload_size is None:
            os.environ.pop("LANCE_INITIAL_UPLOAD_SIZE", None)
        else:
            os.environ["LANCE_INITIAL_UPLOAD_SIZE"] = previous_upload_size
        r2_io.purge_prefix(_R2_BUCKET, prefix)
