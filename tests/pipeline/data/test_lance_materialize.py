"""Behavior tests for Lance subset materialization."""

from __future__ import annotations

import json
import os
import shutil
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from pathlib import Path
from typing import cast

import lance
import pyarrow as pa
import pytest
from structlog.testing import capture_logs

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_materialize import (
    MaterializeManifest,
    _open_source,
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


def _raise_advice_error(*_args: int) -> None:
    raise OSError(5, "advice failed")


def test_materialize_lance_subset_evicts_written_data_files(
    tmp_path: Path,
    two_version_source: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Published Lance data files are advised out of the filesystem cache.

    :param tmp_path: Isolates the published destination.
    :param two_version_source: Supplies a real version-pinned Lance source.
    :param monkeypatch: Records the OS advice boundary while preserving real materialization.
    """
    source, txid = two_version_source
    advised_fds: list[int] = []

    def record_advice(fd: int, offset: int, length: int, advice: int) -> None:
        advised_fds.append(fd)
        assert offset == 0
        assert length == 0
        assert advice == os.POSIX_FADV_DONTNEED

    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.setattr(os, "posix_fadvise", record_advice, raising=False)
    destination = tmp_path / "materialized.lance"

    materialize_lance_subset(source, destination, txid=txid, columns=("a",))
    assert advised_fds
    (destination / "data" / "additional-fragment.lance").write_bytes(b"fragment")
    advised_fds.clear()

    materialize_lance_subset(source, destination, txid=txid, columns=("a",))

    data_files = [path for path in (destination / "data").rglob("*") if path.is_file()]
    assert len(data_files) >= 2
    assert len(advised_fds) == len(data_files)
    assert lance.dataset(str(destination)).count_rows() == 3


@pytest.mark.skipif(
    not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"),
    reason="POSIX_FADV_DONTNEED is unavailable on this platform",
)
def test_materialize_lance_subset_real_cache_evict_remains_consumable(
    tmp_path: Path,
    two_version_source: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real OS eviction path preserves the published Lance dataset.

    :param tmp_path: Isolates the published destination.
    :param two_version_source: Supplies a real version-pinned Lance source.
    :param monkeypatch: Wraps the real syscall to prove the production path invokes it.
    """
    source, txid = two_version_source
    real_advice = os.posix_fadvise
    advised_fds: list[int] = []

    def record_real_advice(fd: int, offset: int, length: int, advice: int) -> None:
        real_advice(fd, offset, length, advice)
        advised_fds.append(fd)

    monkeypatch.setattr(os, "posix_fadvise", record_real_advice)
    destination = tmp_path / "materialized.lance"

    materialize_lance_subset(source, destination, txid=txid, columns=("a",))

    assert advised_fds
    materialized = lance.dataset(str(destination))
    assert materialized.count_rows() == 3
    assert materialized.schema.names == ["a"]


@pytest.mark.parametrize("missing_attribute", ["POSIX_FADV_DONTNEED", "posix_fadvise"])
def test_materialize_lance_subset_without_cache_advice_remains_consumable(
    missing_attribute: str,
    tmp_path: Path,
    two_version_source: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Platforms without POSIX cache advice still publish a readable dataset.

    :param missing_attribute: Optional OS cache-advice attribute to remove.
    :param tmp_path: Isolates the published destination.
    :param two_version_source: Supplies a real version-pinned Lance source.
    :param monkeypatch: Removes one optional OS attribute.
    """
    source, txid = two_version_source
    monkeypatch.delattr(os, missing_attribute, raising=False)
    destination = tmp_path / "materialized.lance"

    materialize_lance_subset(source, destination, txid=txid, columns=("a",))

    assert lance.dataset(str(destination)).count_rows() == 3


def test_materialize_lance_subset_cache_advice_error_remains_consumable(
    tmp_path: Path,
    two_version_source: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An OS advice failure does not invalidate the published dataset.

    :param tmp_path: Isolates the published destination.
    :param two_version_source: Supplies a real version-pinned Lance source.
    :param monkeypatch: Injects the advisory syscall failure.
    """
    source, txid = two_version_source

    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.setattr(os, "posix_fadvise", _raise_advice_error, raising=False)
    destination = tmp_path / "materialized.lance"

    materialize_lance_subset(source, destination, txid=txid, columns=("a",))

    assert lance.dataset(str(destination)).count_rows() == 3


def test_resolve_txid_version_known_txid_returns_matching_version(
    two_version_source: tuple[str, str],
) -> None:
    """A recorded txid resolves back to the version that committed it.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    """
    source, txid = two_version_source
    ds = lance.dataset(source)
    assert resolve_txid_version(ds, txid) == 1


def test_resolve_txid_version_transient_version_list_retries_without_leaking(
    two_version_source: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient version-list read retries without logging exception details.

    :param two_version_source: Local two-version source dataset and version-1 txid.
    :param monkeypatch: Fixture injecting the transient version-list failure.
    """
    source, txid = two_version_source
    dataset = lance.dataset(source)
    real_versions = dataset.versions
    attempts = 0

    def flaky_versions() -> Sequence[Mapping[str, object]]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError(
                "LanceError(IO): Generic S3 error: 502 Bad Gateway; token top-secret"
            )
        return real_versions()

    monkeypatch.setattr(dataset, "versions", flaky_versions)
    with capture_logs() as logs:
        resolved_version = resolve_txid_version(dataset, txid)

    assert resolved_version == 1
    assert attempts == 2
    retry_logs = [log for log in logs if log.get("event") == "lance_read_attempt_failed"]
    assert [(log["operation"], log["attempt"], log["max_attempts"]) for log in retry_logs] == [
        ("version_list", 1, 3)
    ]
    assert "top-secret" not in repr(logs)


def test_resolve_txid_version_transient_transaction_read_retries_without_leaking(
    two_version_source: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient transaction read retries without logging exception details.

    :param two_version_source: Local two-version source dataset and version-1 txid.
    :param monkeypatch: Fixture injecting the transient transaction-read failure.
    """
    source, txid = two_version_source
    dataset = lance.dataset(source)
    real_read_transaction = dataset.read_transaction
    attempts = 0

    def flaky_read_transaction(version: int) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError(
                "LanceError(IO): Generic S3 error: HTTP status server error "
                "(503 Service Unavailable); credential top-secret"
            )
        return real_read_transaction(version)

    monkeypatch.setattr(dataset, "read_transaction", flaky_read_transaction)
    with capture_logs() as logs:
        resolved_version = resolve_txid_version(dataset, txid)

    assert resolved_version == 1
    assert attempts == 2
    retry_logs = [log for log in logs if log.get("event") == "lance_read_attempt_failed"]
    assert [(log["operation"], log["attempt"], log["max_attempts"]) for log in retry_logs] == [
        ("transaction_read", 1, 3)
    ]
    assert "top-secret" not in repr(logs)


def test_resolve_txid_version_schema_error_does_not_retry(
    two_version_source: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A permanent transaction schema error propagates on its first attempt.

    :param two_version_source: Local two-version source dataset and version-1 txid.
    :param monkeypatch: Fixture injecting the permanent transaction-read failure.
    """
    source, txid = two_version_source
    dataset = lance.dataset(source)
    attempts = 0

    def invalid_transaction_schema(_version: int) -> object:
        nonlocal attempts
        attempts += 1
        raise ValueError("Schema mismatch: transaction field is invalid")

    monkeypatch.setattr(dataset, "read_transaction", invalid_transaction_schema)
    with capture_logs() as logs, pytest.raises(ValueError, match="Schema mismatch"):
        resolve_txid_version(dataset, txid)

    assert attempts == 1
    assert not [log for log in logs if log.get("event") == "lance_read_attempt_failed"]


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


def test_materialize_source_open_transient_object_store_error_retries_without_leaking(
    two_version_source: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient source-open failure retries without logging exception details.

    :param two_version_source: Local source used after the injected object-store failure.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param monkeypatch: Fixture injecting the transient ``lance.dataset`` failure.
    """
    source, txid = two_version_source
    real_dataset = lance.dataset
    attempts = 0

    def flaky_dataset(
        uri: str, *, storage_options: dict[str, str] | None = None
    ) -> lance.LanceDataset:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError(
                "LanceError(IO): Generic S3 error: HTTP error: error sending request "
                "with token top-secret"
            )
        return real_dataset(uri, storage_options=storage_options)

    monkeypatch.setattr(lance, "dataset", flaky_dataset)
    destination = tmp_path / "out" / "train.lance"
    with capture_logs() as logs:
        materialize_lance_subset(source, destination, txid=txid, columns=("a",))

    assert real_dataset(str(destination)).to_table().column("a").to_pylist() == [1, 2, 3]
    assert attempts == 2
    retry_logs = [log for log in logs if log.get("event") == "lance_read_attempt_failed"]
    assert [(log["operation"], log["attempt"], log["max_attempts"]) for log in retry_logs] == [
        ("source_open", 1, 3)
    ]
    assert "top-secret" not in repr(logs)


def test_materialize_source_open_auth_error_does_not_retry(
    two_version_source: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A permanent object-store auth error propagates on its first attempt.

    :param two_version_source: Local source whose txid forms the materialization request.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param monkeypatch: Fixture injecting the permanent source-open failure.
    """
    source, txid = two_version_source
    attempts = 0

    def denied_dataset(*_args: object, **_kwargs: object) -> lance.LanceDataset:
        nonlocal attempts
        attempts += 1
        raise ValueError("LanceError(IO): 403 Forbidden: AccessDenied")

    monkeypatch.setattr(lance, "dataset", denied_dataset)
    destination = tmp_path / "out" / "train.lance"
    with capture_logs() as logs, pytest.raises(ValueError, match="AccessDenied"):
        materialize_lance_subset(source, destination, txid=txid, columns=("a",))

    assert attempts == 1
    assert not [log for log in logs if log.get("event") == "lance_read_attempt_failed"]


def test_materialize_source_open_transient_exhaustion_fails_closed(
    two_version_source: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient source-open failure re-raises after the bounded attempt budget.

    :param two_version_source: Local source whose txid forms the materialization request.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param monkeypatch: Fixture injecting persistent transient source-open failures.
    """
    source, txid = two_version_source
    failure = ValueError(
        "LanceError(IO): Generic S3 error: HTTP error: error sending request "
        "with token top-secret"
    )
    attempts = 0

    def unavailable_dataset(*_args: object, **_kwargs: object) -> lance.LanceDataset:
        nonlocal attempts
        attempts += 1
        raise failure

    monkeypatch.setattr(lance, "dataset", unavailable_dataset)
    destination = tmp_path / "out" / "train.lance"
    with capture_logs() as logs, pytest.raises(
        RuntimeError, match="source_open failed after 3 transient attempts"
    ) as exc_info:
        materialize_lance_subset(source, destination, txid=txid, columns=("a",))

    assert attempts == 3
    retry_logs = [log for log in logs if log.get("event") == "lance_read_attempt_failed"]
    assert [log["attempt"] for log in retry_logs] == [1, 2, 3]
    assert "top-secret" not in repr(logs)
    assert "top-secret" not in "".join(traceback.format_exception(exc_info.value))
    assert not destination.exists()


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


def test_open_source_absolute_file_uri_from_other_cwd_preserves_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file URI remains absolute while Lance reads it outside the source cwd.

    :param tmp_path: Pytest fixture providing source and unrelated working directories.
    :param monkeypatch: Fixture changing the process working directory.
    """
    source = tmp_path / "network volume" / "train.lance"
    source.parent.mkdir()
    lance.write_dataset(pa.table({"a": [1, 2, 3]}), str(source))
    unrelated_cwd = tmp_path / "worker"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    dataset = _open_source(source.as_uri())

    assert dataset.uri == source.as_uri()
    assert dataset.to_table().column("a").to_pylist() == [1, 2, 3]


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


def test_materialize_splits_missing_local_completion_marker_writes_nothing(
    tmp_path: Path,
) -> None:
    """An incomplete local source is rejected before any split is published.

    :param tmp_path: Pytest fixture providing fresh source and destination roots.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    lance.write_dataset(pa.table({"a": [1]}), source_root / "train.lance")
    destination = tmp_path / "destination"

    with pytest.raises(FileNotFoundError, match="dataset.complete"):
        materialize_splits(
            str(source_root),
            destination,
            txids=None,
            projection={"train": ("a",)},
            row_limit=None,
            shard_suffix=".lance",
        )

    assert not destination.exists()


def test_materialize_splits_missing_file_uri_completion_marker_writes_nothing(
    tmp_path: Path,
) -> None:
    """An incomplete file-URI source is rejected before local publication.

    :param tmp_path: Pytest fixture providing fresh source and destination roots.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    destination = tmp_path / "destination"

    with pytest.raises(FileNotFoundError, match="dataset.complete"):
        materialize_splits(
            source_root.as_uri(),
            destination,
            txids=None,
            projection={},
            row_limit=None,
            shard_suffix=".lance",
        )

    assert not destination.exists()


def test_materialize_splits_missing_r2_completion_marker_writes_nothing(
    fake_r2_remote: Path,
) -> None:
    """An incomplete R2 source is rejected before local publication.

    :param fake_r2_remote: Real rclone local backend and destination parent.
    """
    destination = fake_r2_remote / "destination"

    with pytest.raises(FileNotFoundError, match="dataset.complete"):
        materialize_splits(
            "r2://bucket/incomplete",
            destination,
            txids=None,
            projection={},
            row_limit=None,
            shard_suffix=".lance",
        )

    assert not destination.exists()


def test_materialize_splits_builds_projected_capped_splits_per_txid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each pinned split materializes with its requested projection and row cap.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param monkeypatch: Fixture replacing the rclone sidecar boundary.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "dataset.complete").touch()
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

    monkeypatch.setattr(r2_io, "download_dir_no_overwrite", download_spy)
    dest_root = tmp_path / "dest"
    materialize_splits(
        str(source_root),
        dest_root,
        txids=txids,
        projection={split: ("a",) for split in ("train", "val", "test")},
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
    source_path = tmp_path / "source"
    source_path.mkdir()
    (source_path / "dataset.complete").touch()
    source_root = str(source_path)
    dest_root = tmp_path / "dest"
    materialize_splits(
        source_root,
        dest_root,
        txids={},
        projection={},
        row_limit=None,
        shard_suffix=".lance",
    )

    assert calls == [(source_root, dest_root, "{*.lance/**,metadata/**}")]


def test_materialize_latest_snapshot_transient_identity_read_retries(
    two_version_source: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Latest-snapshot hydration retries its transaction identity read.

    :param two_version_source: Local two-version source dataset.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param monkeypatch: Fixture injecting the transient identity-read failure.
    """
    source, _ = two_version_source
    real_dataset = lance.dataset
    source_dataset = real_dataset(source)
    real_read_transaction = source_dataset.read_transaction
    attempts = 0

    def flaky_read_transaction(version: int) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError(
                "LanceError(IO): Generic S3 error: 504 Gateway Timeout"
            )
        return real_read_transaction(version)

    def open_dataset(
        uri: str, *, storage_options: dict[str, str] | None = None
    ) -> lance.LanceDataset:
        if uri == source:
            return source_dataset
        return real_dataset(uri, storage_options=storage_options)

    monkeypatch.setattr(source_dataset, "read_transaction", flaky_read_transaction)
    monkeypatch.setattr(lance, "dataset", open_dataset)
    destination = tmp_path / "out" / "train.lance"

    materialize_lance_subset(source, destination, txid=None, columns=("a",), limit=4)

    assert attempts == 2
    assert real_dataset(str(destination)).to_table().column("a").to_pylist() == [1, 2, 3, 4]


def test_materialize_without_txid_uses_latest_version(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """An unpinned materialization reads the source's latest snapshot.

    :param two_version_source: Local two-version source dataset.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, _ = two_version_source
    dest = tmp_path / "out" / "train.lance"

    materialize_lance_subset(source, dest, txid=None, columns=("a",), limit=4)

    out = lance.dataset(str(dest))
    assert out.to_table().column("a").to_pylist() == [1, 2, 3, 4]
    manifest = MaterializeManifest.model_validate_json(
        sidecar_path(dest).read_text(encoding="utf-8")
    )
    assert manifest.txid is None
    assert manifest.resolved_version == 2


def test_materialize_without_txid_unchanged_source_reuses_cache(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """An unpinned rerun reuses its cache while the source snapshot is unchanged.

    :param two_version_source: Local two-version source dataset.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, _ = two_version_source
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(source, dest, txid=None, columns=("a",), limit=4)
    version_after_first = lance.dataset(str(dest)).version

    result = materialize_lance_subset(source, dest, txid=None, columns=("a",), limit=4)

    assert result == dest
    assert lance.dataset(str(dest)).version == version_after_first


def test_materialize_without_txid_replaced_source_rejects_stale_cache(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """An unpinned cache rejects a different dataset at the same URI and version.

    :param two_version_source: Local two-version source dataset.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, _ = two_version_source
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(source, dest, txid=None, columns=("a",), limit=4)
    shutil.rmtree(source)
    lance.write_dataset(pa.table({"a": [10], "b": ["new"]}), source)
    lance.write_dataset(pa.table({"a": [11], "b": ["newer"]}), source, mode="append")

    with pytest.raises(ValueError, match="hash"):
        materialize_lance_subset(source, dest, txid=None, columns=("a",), limit=4)


def test_materialize_without_txid_source_advance_rejects_stale_cache(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """An unpinned cache cannot masquerade as latest after the source advances.

    :param two_version_source: Local two-version source dataset.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, _ = two_version_source
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(source, dest, txid=None, columns=("a",), limit=4)
    lance.write_dataset(pa.table({"a": [6], "b": ["r"]}), source, mode="append")

    with pytest.raises(ValueError, match="hash"):
        materialize_lance_subset(source, dest, txid=None, columns=("a",), limit=4)


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
    destination = lance.dataset(str(dest))
    transaction = destination.read_transaction(destination.version)
    assert transaction is not None
    assert manifest.materialized_txid == transaction.uuid
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


def test_materialize_cache_transient_destination_open_retries(
    two_version_source: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient local cache read retries before declaring corruption.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param monkeypatch: Fixture injecting one transient destination-open failure.
    """
    source, txid = two_version_source
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(source, dest, txid=txid, columns=("a",))
    real_dataset = lance.dataset
    attempts = 0

    def transient_destination_open(
        uri: str, *, storage_options: dict[str, str] | None = None
    ) -> lance.LanceDataset:
        nonlocal attempts
        if uri == str(dest):
            attempts += 1
            if attempts == 1:
                raise TimeoutError("transient destination timeout")
        return real_dataset(uri, storage_options=storage_options)

    monkeypatch.setattr(lance, "dataset", transient_destination_open)

    assert materialize_lance_subset(source, dest, txid=txid, columns=("a",)) == dest
    assert attempts == 2


def test_materialize_cache_replaced_destination_raises(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """A valid but replaced local dataset is not accepted as a cache hit.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, txid = two_version_source
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(source, dest, txid=txid, columns=("a",))
    shutil.rmtree(dest)
    lance.write_dataset(pa.table({"a": [999]}), dest)

    with pytest.raises(ValueError, match="sidecar"):
        materialize_lance_subset(source, dest, txid=txid, columns=("a",))


def test_materialize_cache_missing_data_file_raises(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """A structurally corrupt local dataset is not accepted as a cache hit.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, txid = two_version_source
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(source, dest, txid=txid, columns=("a",))
    next((dest / "data").glob("*.lance")).unlink()

    with pytest.raises(ValueError, match="validation"):
        materialize_lance_subset(source, dest, txid=txid, columns=("a",))


def test_materialize_legacy_manifest_without_destination_identity_raises(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """A legacy sidecar cannot retroactively trust an unknown destination.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, txid = two_version_source
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(source, dest, txid=txid, columns=("a",))
    payload = json.loads(sidecar_path(dest).read_text(encoding="utf-8"))
    del payload["materialized_txid"]
    sidecar_path(dest).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="destination identity"):
        materialize_lance_subset(source, dest, txid=txid, columns=("a",))


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


def test_materialize_concurrent_identical_requests_publish_one_valid_cache(
    two_version_source: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent writers converge on one destination-manifest identity.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param monkeypatch: Fixture synchronizing the two real Lance writes.
    """
    source, txid = two_version_source
    dest = tmp_path / "out" / "train.lance"
    barrier = Barrier(2)
    real_write_dataset = cast(Callable[..., lance.LanceDataset], lance.write_dataset)

    def synchronized_write(*args: object, **kwargs: object) -> lance.LanceDataset:
        barrier.wait()
        return real_write_dataset(*args, **kwargs)

    monkeypatch.setattr(lance, "write_dataset", synchronized_write)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                materialize_lance_subset,
                source,
                dest,
                txid=txid,
                columns=("a",),
            )
            for _ in range(2)
        ]
        assert [future.result() for future in futures] == [dest, dest]

    assert materialize_lance_subset(source, dest, txid=txid, columns=("a",)) == dest


def test_materialize_interrupted_publish_rerun_recovers(
    two_version_source: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sidecar write interruption leaves no destination and is restartable.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param monkeypatch: Pytest fixture injecting the interrupted sidecar write.
    """
    source, txid = two_version_source
    dest = tmp_path / "out" / "train.lance"
    original_write_text = Path.write_text

    def interrupt_sidecar_write(
        path: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        if path.name == sidecar_path(dest).name:
            raise OSError("injected sidecar write interruption")
        return original_write_text(
            path, data, encoding=encoding, errors=errors, newline=newline
        )

    with monkeypatch.context() as context:
        context.setattr(Path, "write_text", interrupt_sidecar_write)
        with pytest.raises(OSError, match="injected"):
            materialize_lance_subset(source, dest, txid=txid, columns=("a",))

    assert not dest.exists()
    assert not list(dest.parent.glob(f".{dest.name}.*.partial"))
    assert materialize_lance_subset(source, dest, txid=txid, columns=("a",)) == dest


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


def test_materialize_legacy_pinned_sidecar_without_resolved_txid_reuses_cache(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """A pinned cache remains valid when its legacy sidecar lacks resolved_txid.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, txid = two_version_source
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(source, dest, txid=txid, columns=("a",))
    payload = json.loads(sidecar_path(dest).read_text(encoding="utf-8"))
    del payload["resolved_txid"]
    sidecar_path(dest).write_text(json.dumps(payload), encoding="utf-8")

    assert materialize_lance_subset(source, dest, txid=txid, columns=("a",)) == dest


def test_materialize_tampered_sidecar_resolved_txid_raises(
    two_version_source: tuple[str, str], tmp_path: Path
) -> None:
    """A pinned sidecar cannot claim a different resolved transaction.

    :param two_version_source: Local two-version source dataset and its version-1 txid.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    source, txid = two_version_source
    dest = tmp_path / "out" / "train.lance"
    materialize_lance_subset(source, dest, txid=txid, columns=("a",))
    payload = json.loads(sidecar_path(dest).read_text(encoding="utf-8"))
    payload["resolved_txid"] = "different-transaction"
    sidecar_path(dest).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hash"):
        materialize_lance_subset(source, dest, txid=txid, columns=("a",))


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
