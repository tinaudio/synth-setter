"""End-to-end proof that the Lance fragment-based finalize model works.

This is a *model* proof, not a test of pipeline glue: it drives the real Lance
codec (``lance_fragment`` → sidecar JSON → ``commit_lance_dataset`` →
``iter_lance_column_rows``) on a real local filesystem, no mocks, to pin the
library behaviours the design in ``docs/design/data-pipeline.md`` depends on:

* a worker writes an uncommitted fragment straight into ``{split}.lance/data/``,
  finalize commits only its serialized ``FragmentMetadata`` — no row rewrite;
* the winner set commits as one atomic transaction (single manifest version);
* committing one winner of several duplicate attempts yields exactly that
  shard's rows — duplicate attempts never double rows;
* re-committing the same winner set is idempotent (``Overwrite`` replaces);
* a fragment is only readable from the dataset whose ``data/`` dir physically
  holds its file — and ``count_rows`` (manifest metadata) cannot catch a
  dangling fragment, so validation must read rows.

Every expected array is built directly in NumPy — never through the codec under
test — so a row-order or reshape bug cannot corrupt both sides identically.
"""

from __future__ import annotations

import json
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    DATASET_FIELD_DTYPES,
    DATASET_FIELD_NAMES,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
)
from synth_setter.pipeline.data.lance_shard import (
    commit_lance_dataset,
    iter_lance_column_rows,
    lance_fragment,
    lance_schema,
    record_batch_from_arrays,
)
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata

# Small shapes; every element is a distinct value exactly representable as
# float16 (<= 2048), so equality is exact across the writer dtypes.
_FIELD_SHAPES: dict[str, tuple[int, ...]] = {
    AUDIO_FIELD: (2, 2, 5),
    MEL_SPEC_FIELD: (2, 2, 3, 4),
    PARAM_ARRAY_FIELD: (2, 7),
}
_ROWS_PER_SHARD = _FIELD_SHAPES[PARAM_ARRAY_FIELD][0]
# Offset stride between shards/attempts: keeps their element values disjoint
# while the largest offset element stays under the float16-exact 2048 ceiling.
_VALUE_STRIDE = 1000
_METADATA = ShardMetadata(
    velocity=100,
    signal_duration_seconds=1.0,
    sample_rate=100,
    channels=2,
    min_loudness=-55.0,
    base_seed=42,
    attempts_per_sample=100,
)


def _arange_arrays(offset: int) -> dict[str, np.ndarray]:
    """Build one shard's arrays; ``offset`` keeps two shards value-disjoint.

    :param offset: First ``arange`` value for every field.
    :returns: Mapping keyed by ``DATASET_FIELD_NAMES`` with writer dtypes.
    """
    return {
        field: np.arange(
            offset, offset + int(np.prod(shape)), dtype=DATASET_FIELD_DTYPES[field]
        ).reshape(shape)
        for field, shape in _FIELD_SHAPES.items()
    }


def _worker_writes_fragment(
    split_uri: Path, schema: pa.Schema, arrays: dict[str, np.ndarray], fragment_id: int
) -> tuple[lance.fragment.FragmentMetadata, str]:
    """Simulate a worker: write a fragment under ``split_uri`` and serialize it.

    The returned string is the Lance fragment-metadata JSON payload (the value
    that will live in the sidecar's ``fragment_json`` field): a ``json.dumps`` of
    Lance's ``FragmentMetadata.to_json()`` dict, which ``from_json`` re-parses.

    :param split_uri: Destination split dataset dir (``{split}.lance``).
    :param schema: Arrow schema shared by every fragment.
    :param arrays: One shard's field arrays.
    :param fragment_id: Zero-based fragment index within the dataset.
    :returns: The live fragment metadata and its sidecar JSON string.
    """
    batch = record_batch_from_arrays(arrays, schema)
    frag = lance_fragment(split_uri, schema, batch, fragment_id=fragment_id)
    return frag, json.dumps(frag.to_json())


def _read_columns(uri: Path) -> dict[str, np.ndarray]:
    """Read every dataset field back from a committed Lance dataset.

    Uses ``iter_lance_column_rows`` (a real data read), not ``count_rows`` —
    the latter trusts manifest ``physical_rows`` and cannot see a missing file.

    :param uri: Committed Lance dataset directory.
    :returns: Field name to row-stacked decoded array.
    """
    return {
        field: np.stack(list(iter_lance_column_rows(uri, field)), axis=0)
        for field in DATASET_FIELD_NAMES
    }


def test_worker_fragment_sidecar_round_trips_and_commit_reads_back_written_values(
    tmp_path: Path,
) -> None:
    """Worker fragment → sidecar JSON → finalize commit reads back exact values.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    train_uri = tmp_path / "train.lance"
    written = _arange_arrays(offset=0)

    _, sidecar_json = _worker_writes_fragment(train_uri, schema, written, fragment_id=0)
    # Cross the R2 boundary: the sidecar survives a disk round-trip as text.
    sidecar_path = tmp_path / "shard-000000.fragment.json"
    sidecar_path.write_text(sidecar_json)
    winner = lance.fragment.FragmentMetadata.from_json(sidecar_path.read_text())

    commit_lance_dataset(train_uri, schema, [winner])

    decoded = _read_columns(train_uri)
    for field in DATASET_FIELD_NAMES:
        np.testing.assert_array_equal(decoded[field], written[field])


def test_duplicate_attempts_commit_winner_only_yields_single_shard_rows(
    tmp_path: Path,
) -> None:
    """Two attempts for one shard, commit one winner: exactly the winner's rows.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    train_uri = tmp_path / "train.lance"
    winner_arrays = _arange_arrays(offset=0)
    loser_arrays = _arange_arrays(offset=_VALUE_STRIDE)

    # Both attempts write real fragment data into the same split dir.
    winner, _ = _worker_writes_fragment(train_uri, schema, winner_arrays, fragment_id=0)
    _worker_writes_fragment(train_uri, schema, loser_arrays, fragment_id=1)

    # Finalize commits the winner only.
    commit_lance_dataset(train_uri, schema, [winner])

    decoded = _read_columns(train_uri)
    assert decoded[PARAM_ARRAY_FIELD].shape[0] == _ROWS_PER_SHARD
    for field in DATASET_FIELD_NAMES:
        np.testing.assert_array_equal(decoded[field], winner_arrays[field])


def test_winner_set_commits_atomically_as_one_manifest_version(tmp_path: Path) -> None:
    """Three winner fragments commit as a single all-or-nothing transaction.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    train_uri = tmp_path / "train.lance"
    winners = [
        _worker_writes_fragment(train_uri, schema, _arange_arrays(i * _VALUE_STRIDE), i)[0]
        for i in range(3)
    ]

    commit_lance_dataset(train_uri, schema, winners)

    dataset = lance.dataset(str(train_uri))
    assert dataset.version == 1
    assert len(dataset.get_fragments()) == 3
    assert dataset.count_rows() == 3 * _ROWS_PER_SHARD


def test_recommitting_winner_set_is_idempotent(tmp_path: Path) -> None:
    """Re-committing the same winner set leaves rows and values unchanged.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    train_uri = tmp_path / "train.lance"
    written = _arange_arrays(offset=0)
    winner, _ = _worker_writes_fragment(train_uri, schema, written, fragment_id=0)

    commit_lance_dataset(train_uri, schema, [winner])
    commit_lance_dataset(train_uri, schema, [winner])

    decoded = _read_columns(train_uri)
    assert decoded[PARAM_ARRAY_FIELD].shape[0] == _ROWS_PER_SHARD
    np.testing.assert_array_equal(decoded[PARAM_ARRAY_FIELD], written[PARAM_ARRAY_FIELD])


def test_fragment_not_colocated_with_dataset_fails_on_read(tmp_path: Path) -> None:
    """A dangling fragment commit fails on read even though ``count_rows`` passes.

    The fragment's file lives under another dataset's ``data/`` dir; manifest
    metadata still reports its rows, so validation must read rows, not counts.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    staging_uri = tmp_path / "staging.lance"
    train_uri = tmp_path / "train.lance"
    # Worker writes the fragment data under staging, not under train.lance.
    foreign, _ = _worker_writes_fragment(staging_uri, schema, _arange_arrays(0), 0)

    commit_lance_dataset(train_uri, schema, [foreign])

    # Manifest metadata reports rows even though the data file is absent...
    assert lance.dataset(str(train_uri)).count_rows() == _ROWS_PER_SHARD
    # ...but an actual read fails, proving the co-location requirement.
    with pytest.raises(pa.ArrowInvalid, match="LanceError|Object at location"):
        list(iter_lance_column_rows(train_uri, PARAM_ARRAY_FIELD))
