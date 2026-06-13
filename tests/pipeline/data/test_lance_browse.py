"""Tests for exporting single-file Lance shards into browsable Lance datasets."""

from __future__ import annotations

from pathlib import Path

import lance
import numpy as np
import pytest

from synth_setter.pipeline.data.lance_browse import (
    build_browse_db,
    duplicate_stems,
    export_shard_to_dataset,
)
from synth_setter.pipeline.data.lance_shard import (
    SHARD_METADATA_SCHEMA_KEY,
    lance_schema,
    record_batch_from_arrays,
    write_lance_file,
)
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata
from tests.helpers.lance_fixtures import write_lance_shard

_METADATA = ShardMetadata(
    velocity=64,
    signal_duration_seconds=0.5,
    sample_rate=8,
    channels=1,
    min_loudness=-40.0,
)


def _write_two_row_shard(path: Path) -> dict[str, np.ndarray]:
    """Write a two-row shard with the three tensor columns and embedded ShardMetadata.

    :param path: Destination ``.lance`` shard file.
    :returns: The exact column arrays written, for round-trip comparison.
    """
    arrays = {
        "audio": np.arange(2 * 1 * 4, dtype=np.float16).reshape(2, 1, 4),
        "mel_spec": np.arange(2 * 1 * 2 * 3, dtype=np.float32).reshape(2, 1, 2, 3),
        "param_array": np.arange(2 * 5, dtype=np.float32).reshape(2, 5),
    }
    schema = lance_schema({k: v.shape for k, v in arrays.items()}, _METADATA)
    write_lance_file(path, schema, [record_batch_from_arrays(arrays, schema)])
    return arrays


def test_export_shard_to_dataset_preserves_rows_and_column_values(tmp_path: Path) -> None:
    """Every row, shape, dtype, and tensor value survives the shard-to-dataset round trip.

    :param tmp_path: Holds the source shard and its exported dataset.
    """
    shard = tmp_path / "shard-000000.lance"
    arrays = _write_two_row_shard(shard)

    dataset_dir = export_shard_to_dataset(shard, tmp_path / "train.lance")

    dataset = lance.dataset(str(dataset_dir))
    assert dataset.count_rows() == 2
    table = dataset.to_table()
    assert set(table.column_names) == set(arrays)
    for name, expected in arrays.items():
        restored = table.column(name).combine_chunks().to_numpy_ndarray()
        # Shape and dtype, not just values: a flattened tensor or a silent
        # float16 -> float32 upcast would slip past assert_array_equal alone.
        assert restored.shape == expected.shape
        assert restored.dtype == expected.dtype
        np.testing.assert_array_equal(restored, expected)


def test_export_shard_to_dataset_preserves_schema_metadata(tmp_path: Path) -> None:
    """The embedded ShardMetadata travels intact with the dataset schema.

    :param tmp_path: Holds the source shard and its exported dataset.
    """
    shard = tmp_path / "shard-000000.lance"
    _write_two_row_shard(shard)

    dataset_dir = export_shard_to_dataset(shard, tmp_path / "train.lance")

    metadata = lance.dataset(str(dataset_dir)).schema.metadata or {}
    assert SHARD_METADATA_SCHEMA_KEY in metadata
    payload = ShardMetadata.model_validate_json(metadata[SHARD_METADATA_SCHEMA_KEY])
    assert payload == _METADATA


def test_export_shard_to_dataset_overwrites_with_second_shards_data(tmp_path: Path) -> None:
    """A second export replaces the dataset — surviving rows are the new shard's.

    :param tmp_path: Holds the source shards and the reused destination dataset.
    """
    two_row = tmp_path / "two.lance"
    one_row = tmp_path / "one.lance"
    _write_two_row_shard(two_row)
    one_row_values = np.arange(100, 105, dtype=np.float32).reshape(1, 5)
    write_lance_shard(one_row, {"param_array": one_row_values})
    dest = tmp_path / "train.lance"

    export_shard_to_dataset(two_row, dest)
    export_shard_to_dataset(one_row, dest)

    table = lance.dataset(str(dest)).to_table()
    # One row whose values are the second shard's — not an older version fragment.
    assert table.num_rows == 1
    restored = table.column("param_array").combine_chunks().to_numpy_ndarray()
    np.testing.assert_array_equal(restored, one_row_values)


def test_export_shard_to_dataset_replaces_a_plain_file_destination(tmp_path: Path) -> None:
    """A pre-existing plain file at the destination is replaced, not an error.

    :param tmp_path: Holds the source shard and the file-occupied destination.
    """
    shard = tmp_path / "shard-000000.lance"
    _write_two_row_shard(shard)
    dest = tmp_path / "train.lance"
    dest.write_text("stale file where a dataset dir should go")

    export_shard_to_dataset(shard, dest)

    assert lance.dataset(str(dest)).count_rows() == 2


def test_export_shard_to_dataset_in_place_export_raises_and_keeps_source(tmp_path: Path) -> None:
    """Exporting a shard onto its own path is rejected, leaving the source intact.

    :param tmp_path: Holds the shard used as both source and (rejected) destination.
    """
    shard = tmp_path / "train.lance"
    _write_two_row_shard(shard)

    with pytest.raises(ValueError, match="in-place export"):
        export_shard_to_dataset(shard, shard)
    assert shard.is_file()


def test_export_shard_to_dataset_directory_source_raises_value_error(tmp_path: Path) -> None:
    """A directory source is rejected — the pipeline emits single-file shards, not datasets.

    :param tmp_path: Parents the directory masquerading as a shard.
    """
    source = tmp_path / "already-a-dir.lance"
    source.mkdir()

    with pytest.raises(ValueError, match="single-file Lance shard"):
        export_shard_to_dataset(source, tmp_path / "out.lance")


def test_export_shard_to_dataset_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """A missing source path raises rather than writing an empty dataset.

    :param tmp_path: Parents the nonexistent source path.
    """
    with pytest.raises(FileNotFoundError, match="not found"):
        export_shard_to_dataset(tmp_path / "absent.lance", tmp_path / "out.lance")


def test_build_browse_db_routes_each_source_to_its_own_table(tmp_path: Path) -> None:
    """Each shard lands as ``<stem>.lance`` with its own data, in input order.

    :param tmp_path: Holds the source shards and the browse-db root.
    """
    train = tmp_path / "train.lance"
    val = tmp_path / "val.lance"
    # Distinct row counts so a swapped return order or mis-routed data is caught.
    write_lance_shard(train, {"param_array": np.zeros((1, 5), dtype=np.float32)})
    write_lance_shard(val, {"param_array": np.zeros((2, 5), dtype=np.float32)})

    db_dir = tmp_path / "browse"
    tables = build_browse_db([train, val], db_dir)

    assert [t.name for t in tables] == ["train.lance", "val.lance"]
    assert all(t.parent == db_dir for t in tables)
    assert lance.dataset(str(db_dir / "train.lance")).count_rows() == 1
    assert lance.dataset(str(db_dir / "val.lance")).count_rows() == 2


def test_build_browse_db_creates_nested_db_dir(tmp_path: Path) -> None:
    """A multi-level browse-db root is created on demand (``parents=True``).

    :param tmp_path: Parents the not-yet-existing nested browse-db root.
    """
    shard = tmp_path / "train.lance"
    write_lance_shard(shard, {"param_array": np.zeros((1, 5), dtype=np.float32)})

    db_dir = tmp_path / "a" / "b" / "browse"
    build_browse_db([shard], db_dir)

    assert lance.dataset(str(db_dir / "train.lance")).count_rows() == 1


def test_build_browse_db_duplicate_stems_raise_value_error(tmp_path: Path) -> None:
    """Two sources sharing a stem are rejected, and the offending stem is named.

    :param tmp_path: Parents the two same-stem source shards.
    """
    one = tmp_path / "a" / "train.lance"
    two = tmp_path / "b" / "train.lance"
    one.parent.mkdir()
    two.parent.mkdir()
    write_lance_shard(one, {"param_array": np.zeros((1, 5), dtype=np.float32)})
    write_lance_shard(two, {"param_array": np.zeros((1, 5), dtype=np.float32)})

    with pytest.raises(ValueError, match="duplicate table name") as exc:
        build_browse_db([one, two], tmp_path / "browse")
    assert "train" in str(exc.value)


def test_build_browse_db_empty_sources_raise_value_error(tmp_path: Path) -> None:
    """An empty source list is a usage error — there is nothing to browse.

    :param tmp_path: Parents the browse-db root that is never written.
    """
    with pytest.raises(ValueError, match="at least one"):
        build_browse_db([], tmp_path / "browse")


def test_duplicate_stems_all_distinct_returns_empty() -> None:
    """Distinct stems yield no collisions."""
    assert duplicate_stems(["a/train.lance", "b/val.lance", "r2://x/test.lance"]) == []


def test_duplicate_stems_repeated_names_returns_them_sorted() -> None:
    """Repeated stems are reported once each, sorted."""
    paths = ["x/train.lance", "y/train.lance", "p/val.lance", "q/val.lance", "z/test.lance"]
    assert duplicate_stems(paths) == ["train", "val"]
