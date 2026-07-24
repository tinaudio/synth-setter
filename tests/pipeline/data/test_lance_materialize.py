"""Behavior tests for txid-pinned Lance subset materialization."""

from __future__ import annotations

import json
from pathlib import Path

import lance
import pyarrow as pa
import pytest

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_materialize import (
    MaterializeManifest,
    materialize_lance_subset,
    materialize_splits,
    resolve_txid_version,
    sidecar_path,
)


@pytest.fixture
def two_version_source(tmp_path: Path) -> tuple[str, str]:
    """Write a two-version local Lance dataset and pin its first version's txid.

    :param tmp_path: Pytest temp dir holding the source dataset.
    :returns: ``(source_uri, txid_of_version_1)``; version 1 has rows a=1..3,
        version 2 appends a=4..5.
    """
    source = str(tmp_path / "source.lance")
    lance.write_dataset(pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]}), source)
    ds = lance.write_dataset(
        pa.table({"a": [4, 5], "b": ["p", "q"]}), source, mode="append"
    )
    transaction = ds.read_transaction(1)
    assert transaction is not None
    return source, transaction.uuid


def test_resolve_txid_version_known_txid_returns_matching_version(
    two_version_source: tuple[str, str],
) -> None:
    """A recorded txid resolves back to the version that committed it.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    """
    source, txid = two_version_source
    ds = lance.dataset(source)
    assert resolve_txid_version(ds, txid) == 1


def test_resolve_txid_version_unknown_txid_raises_lookup_error(
    two_version_source: tuple[str, str],
) -> None:
    """An unknown txid fails loudly instead of falling back to latest.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    """
    source, _ = two_version_source
    ds = lance.dataset(source)
    with pytest.raises(LookupError, match="no-such-txid"):
        resolve_txid_version(ds, "no-such-txid")


def test_materialize_column_projection_subset_columns_only_requested_schema(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """The materialized dataset carries only the requested columns.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, txid = two_version_source
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(source, dest, txid=txid, columns=("a",))
    out = lance.dataset(str(dest))
    assert out.schema.names == ["a"]


def test_materialize_file_uri_source_resolves_local_path(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """A file URI source materializes the pinned local dataset snapshot.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, txid = two_version_source
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(f"file://{source}", dest, txid=txid, columns=("a",))
    out = lance.dataset(str(dest))
    assert out.schema.names == ["a"]
    assert out.to_table().column("a").to_pylist() == [1, 2, 3]


def test_materialize_splits_builds_projected_capped_splits_per_txid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each pinned split materializes with its requested projection and row cap.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param monkeypatch: Fixture replacing the rclone sidecar boundary.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    txids: dict[str, str] = {}
    for split in ("train", "val", "test"):
        dataset = lance.write_dataset(
            pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]}),
            str(source_root / f"{split}.lance"),
        )
        transaction = dataset.read_transaction(dataset.version)
        assert transaction is not None
        txids[split] = transaction.uuid

    calls: list[tuple[str, Path, str | None]] = []

    def download_spy(source_uri: str, dest_path: Path, *, exclude: str | None = None) -> None:
        calls.append((source_uri, dest_path, exclude))

    def columns_for(_split: str) -> tuple[str, ...]:
        return ("a",)

    monkeypatch.setattr(r2_io, "download_dir_no_overwrite", download_spy)
    dest_root = tmp_path / "dest"
    materialize_splits(
        str(source_root),
        dest_root,
        txids=txids,
        columns_for=columns_for,
        row_limit=2,
        shard_suffix=".lance",
    )

    for split in ("train", "val", "test"):
        dataset = lance.dataset(str(dest_root / f"{split}.lance"))
        assert dataset.schema.names == ["a"]
        assert dataset.count_rows() == 2


def test_materialize_splits_downloads_sidecars_with_lance_metadata_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sidecar hydration excludes split datasets and pipeline metadata.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param monkeypatch: Fixture replacing the rclone sidecar boundary.
    """
    calls: list[tuple[str, Path, str | None]] = []

    def download_spy(source_uri: str, dest_path: Path, *, exclude: str | None = None) -> None:
        calls.append((source_uri, dest_path, exclude))

    monkeypatch.setattr(r2_io, "download_dir_no_overwrite", download_spy)
    source_root = str(tmp_path / "source")
    dest_root = tmp_path / "dest"
    materialize_splits(
        source_root,
        dest_root,
        txids={},
        columns_for=lambda _split: (),
        row_limit=None,
        shard_suffix=".lance",
    )

    assert calls == [(source_root, dest_root, "{*.lance/**,metadata/**}")]


def test_materialize_row_limit_limit_two_row_count_matches(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """``limit`` caps the materialized row count.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, txid = two_version_source
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(source, dest, txid=txid, columns=("a",), limit=2)
    assert lance.dataset(str(dest)).count_rows() == 2


def test_materialize_snapshot_pinning_appends_after_pin_yields_pinned_rows(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """Appends after the pin do not leak into the materialized subset.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, txid = two_version_source
    lance.write_dataset(pa.table({"a": [99], "b": ["late"]}), source, mode="append")
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(source, dest, txid=txid, columns=("a",))
    out = lance.dataset(str(dest))
    assert out.to_table().column("a").to_pylist() == [1, 2, 3]


def test_materialize_writes_sidecar_manifest_fields_match_request(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """The sidecar manifest records the request that produced the dataset.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, txid = two_version_source
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(source, dest, txid=txid, columns=("a",), limit=2)
    manifest = MaterializeManifest.model_validate_json(
        sidecar_path(dest).read_text(encoding="utf-8")
    )
    assert manifest.source_uri == source
    assert manifest.txid == txid
    assert manifest.resolved_version == 1
    assert manifest.columns == ("a",)
    assert manifest.limit == 2


def test_materialize_cache_hit_same_request_returns_without_rewrite(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """An identical rerun reuses the local dataset without rewriting it.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, txid = two_version_source
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(source, dest, txid=txid, columns=("a",))
    version_after_first = lance.dataset(str(dest)).version
    result = materialize_lance_subset(source, dest, txid=txid, columns=("a",))
    assert result == dest
    assert lance.dataset(str(dest)).version == version_after_first


def test_materialize_rerun_different_limit_raises(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """A rerun with a different limit refuses the stale local subset.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, txid = two_version_source
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(source, dest, txid=txid, columns=("a",), limit=2)
    with pytest.raises(ValueError, match="hash"):
        materialize_lance_subset(source, dest, txid=txid, columns=("a",), limit=3)


def test_materialize_rerun_different_columns_raises(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """A rerun with different columns refuses the stale local subset.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, txid = two_version_source
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(source, dest, txid=txid, columns=("a",))
    with pytest.raises(ValueError, match="hash"):
        materialize_lance_subset(source, dest, txid=txid, columns=("a", "b"))


def test_materialize_dest_without_sidecar_raises(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """A dataset without its sidecar is untrusted and refused.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, txid = two_version_source
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(source, dest, txid=txid, columns=("a",))
    sidecar_path(dest).unlink()
    with pytest.raises(ValueError, match="sidecar"):
        materialize_lance_subset(source, dest, txid=txid, columns=("a",))


def test_materialize_garbled_sidecar_raises(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """A corrupt sidecar is refused instead of being reinterpreted.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, txid = two_version_source
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(source, dest, txid=txid, columns=("a",))
    sidecar_path(dest).write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar"):
        materialize_lance_subset(source, dest, txid=txid, columns=("a",))


def test_materialize_stamps_cloned_from_txn_transaction_property(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """Provenance is stamped in the output's transaction properties.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, txid = two_version_source
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(source, dest, txid=txid, columns=("a",))
    out = lance.dataset(str(dest))
    txn = out.read_transaction(out.version)
    assert txn is not None and txn.transaction_properties is not None
    assert txn.transaction_properties["cloned_from_txn"] == txid


def test_materialize_tampered_sidecar_hash_raises(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """A sidecar whose hash no longer covers its fields is refused.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, txid = two_version_source
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(source, dest, txid=txid, columns=("a",), limit=2)
    payload = json.loads(sidecar_path(dest).read_text(encoding="utf-8"))
    payload["limit"] = 3
    sidecar_path(dest).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        materialize_lance_subset(source, dest, txid=txid, columns=("a",), limit=3)
