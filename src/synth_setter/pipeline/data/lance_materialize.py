"""Rematerialize a Lance column/row subset to local disk.

Streams a projected scan of one source snapshot into a fresh local Lance
dataset, so hydration transfers only the columns and rows a training run
reads instead of the whole dataset directory. A manifest inside the
materialized directory records the request and gates cache reuse: a rerun with
the same request reuses the local copy; any drift fails loudly.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from uuid import uuid4

import lance
import structlog
from pydantic import BaseModel, ConfigDict, ValidationError
from tenacity import (
    RetryCallState,
    RetryError,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.constants import DATASET_COMPLETE_FILENAME
from synth_setter.pipeline.data.lance_shard import LANCE_DATA_STORAGE_VERSION
from synth_setter.pipeline.file_uri import file_uri_to_path, is_file_uri

logger = structlog.get_logger(__name__)

_SIDECAR_FILENAME = "_materialize.json"
# Long enough that two live subsets under one root never collide, short enough
# to stay readable next to the prefix.
_DIRNAME_DIGEST_CHARS = 8
_DIRNAME_PREFIX_CHARS = 8
_MAX_LANCE_READ_ATTEMPTS = 3
_LANCE_READ_BACKOFF_INITIAL_SECONDS = 0.25
_LANCE_READ_BACKOFF_MAX_SECONDS = 2.0
_MATERIALIZE_BATCH_SIZE = 8192
_MATERIALIZE_IO_BUFFER_SIZE = 32 * 1024**3
_MATERIALIZE_FRAGMENT_READAHEAD = 128
_MATERIALIZE_BATCH_READAHEAD = 8
_MATERIALIZE_MAX_ROWS_PER_GROUP = 4096
_MATERIALIZE_MAX_BYTES_PER_FILE = 256 * 1024**3
_RETRYABLE_LANCE_IO_MARKERS = (
    "408 request timeout",
    "429 too many requests",
    "500 internal server error",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway timeout",
    "connection closed",
    "connection refused",
    "connection reset",
    "error sending request",
    "request timeout",
    "temporarily unavailable",
    "timed out",
)


def _is_retryable_lance_read_error(error: BaseException) -> bool:
    """Return whether a Lance read failed on transient object-store transport.

    :param error: Exception raised by ``lance.dataset`` or ``read_transaction``.
    :returns: Whether retrying the same idempotent metadata read is safe.
    """
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    # pylance 7.0 exposes object-store failures as ValueError without structured status fields.
    message = str(error).casefold()
    return "lanceerror(io)" in message and any(
        marker in message for marker in _RETRYABLE_LANCE_IO_MARKERS
    )


def _retry_lance_read[ReadResult](
    operation_name: str, read: Callable[[], ReadResult]
) -> ReadResult:
    """Run one idempotent Lance metadata read under the bounded retry policy.

    :param operation_name: Secret-free operation label included in retry logs.
    :param read: Zero-argument Lance read operation.
    :returns: The successful read result.
    :raises RuntimeError: All transient attempts failed; third-party details are suppressed.
    """

    def log_failed_attempt(retry_state: RetryCallState) -> None:
        logger.warning(
            "lance_read_attempt_failed",
            operation=operation_name,
            attempt=retry_state.attempt_number,
            max_attempts=_MAX_LANCE_READ_ATTEMPTS,
        )

    retrying = Retrying(
        after=log_failed_attempt,
        retry=retry_if_exception(_is_retryable_lance_read_error),
        stop=stop_after_attempt(_MAX_LANCE_READ_ATTEMPTS),
        wait=wait_exponential(
            multiplier=_LANCE_READ_BACKOFF_INITIAL_SECONDS,
            max=_LANCE_READ_BACKOFF_MAX_SECONDS,
        ),
    )
    try:
        return retrying(read)
    except RetryError:
        raise RuntimeError(
            f"Lance {operation_name} failed after "
            f"{_MAX_LANCE_READ_ATTEMPTS} transient attempts"
        ) from None


class MaterializeManifest(BaseModel):
    """Sidecar record of one materialization request (trust boundary: read back from disk).

    .. attribute :: model_config

        Strict parsing configuration for the on-disk JSON.

    .. attribute :: source_uri

        Source dataset URI exactly as the caller passed it (``r2://`` or local path).

    .. attribute :: txid

        Transaction uuid pinning the source snapshot, or ``None`` for latest.

    .. attribute :: resolved_version

        Source dataset version selected at materialization time.

    .. attribute :: resolved_txid

        Transaction uuid identifying the selected source snapshot when recorded.

    .. attribute :: materialized_txid

        Transaction uuid identifying the local materialized dataset.

    .. attribute :: columns

        Projected column names, in scan order.

    .. attribute :: limit

        First-N row cap, or ``None`` for all rows.

    .. attribute :: request_hash

        :func:`request_hash` over the other five fields; gates cache reuse.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    source_uri: str
    txid: str | None
    resolved_version: int
    resolved_txid: str | None = None
    materialized_txid: str | None = None
    columns: tuple[str, ...]
    limit: int | None
    request_hash: str


def _digest(payload: Mapping[str, object]) -> str:
    """Hash a request payload through one canonical JSON encoding.

    :param payload: JSON-encodable request fields.
    :returns: sha256 hex digest, stable across dict ordering.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def request_hash(
    source_uri: str,
    txid: str | None,
    resolved_version: int,
    columns: tuple[str, ...],
    limit: int | None,
) -> str:
    """Hash one materialization request for sidecar-gated cache reuse.

    :param source_uri: Source dataset URI as the caller passed it.
    :param txid: Transaction uuid pinning the source snapshot, or ``None`` for latest.
    :param resolved_version: Source version selected for materialization.
    :param columns: Projected column names, in scan order.
    :param limit: First-N row cap, or ``None`` for all rows.
    :returns: sha256 hex digest over the canonical JSON encoding of the fields.
    """
    return _digest(
        {
            "source_uri": source_uri,
            "txid": txid,
            "resolved_version": resolved_version,
            "columns": list(columns),
            "limit": limit,
        }
    )


def subset_dirname(
    prefix: str,
    source_root_uri: str,
    *,
    txids: Mapping[str, str] | None,
    projection: Mapping[str, Sequence[str]],
    row_limit: int | None,
) -> str:
    """Name the directory addressing one whole-root materialization request.

    Derived from configuration alone so every rank computes the same directory
    without a source read — ``prepare_data`` runs on one rank, ``setup`` on all.
    The resolved source version is deliberately excluded: drift within a
    directory is the sidecar manifest's job, not the path's.

    :param prefix: Readable leading token, truncated to keep the name short.
    :param source_root_uri: Hydration root as the caller passed it.
    :param txids: Per-split transaction pins, or ``None`` for latest snapshots.
    :param projection: Columns to materialize per split.
    :param row_limit: First-N row cap per split, or ``None`` for all rows.
    :returns: ``<prefix>-<digest>`` directory name.
    """
    digest = _digest(
        {
            "source_root_uri": source_root_uri,
            "txids": dict(txids) if txids is not None else None,
            "projection": {split: list(columns) for split, columns in projection.items()},
            "row_limit": row_limit,
        }
    )
    return f"{prefix[:_DIRNAME_PREFIX_CHARS]}-{digest[:_DIRNAME_DIGEST_CHARS]}"


def sidecar_path(dest_path: Path) -> Path:
    """Return the manifest path inside a materialized dataset directory.

    Keeping the identity record inside the staged directory makes publication
    atomic when the directory is renamed into place.

    :param dest_path: Materialized Lance dataset directory.
    :returns: Materialization manifest within ``dest_path``.
    """
    return dest_path / _SIDECAR_FILENAME


def resolve_txid_version(ds: lance.LanceDataset, txid: str) -> int:
    """Resolve a transaction uuid to the dataset version it committed.

    Linear scan over live versions — O(versions) with one small object read
    each; callers cache the result via the sidecar manifest.

    :param ds: Open Lance dataset to scan.
    :param txid: Transaction uuid to look up.
    :returns: The matching version number.
    :raises LookupError: No live version's transaction matches ``txid`` — the
        pin was cleaned up by ``cleanup_old_versions()`` or never existed.
    """
    versions = _retry_lance_read("version_list", ds.versions)
    for entry in versions:
        version = entry["version"]
        transaction = _retry_lance_read(
            "transaction_read", lambda: ds.read_transaction(version)
        )
        if transaction is not None and transaction.uuid == txid:
            return version
    raise LookupError(
        f"txid {txid!r} matches no live version of {ds.uri} — the pinned "
        "version was cleaned up or the txid never existed"
    )


def _open_source(source_uri: str) -> lance.LanceDataset:
    """Open a local, file-URI, or R2 Lance source.

    :param source_uri: Source dataset location.
    :returns: Open source dataset at its current version.
    """
    if r2_io.is_r2_uri(source_uri):
        open_uri, storage_options = r2_io.lance_target(source_uri)
    elif is_file_uri(source_uri):
        open_uri, storage_options = file_uri_to_path(source_uri).as_uri(), None
    else:
        open_uri, storage_options = source_uri, None
    return _retry_lance_read(
        "source_open", lambda: lance.dataset(open_uri, storage_options=storage_options)
    )


def _transaction_uuid(ds: lance.LanceDataset, version: int) -> str:
    """Return the transaction uuid identifying a source version.

    :param ds: Open source dataset.
    :param version: Source version to identify.
    :returns: Transaction uuid for ``version``.
    :raises ValueError: The source version has no transaction record.
    """
    transaction = _retry_lance_read(
        "transaction_read", lambda: ds.read_transaction(version)
    )
    if transaction is None:
        raise ValueError(f"source version {version} has no transaction record")
    return transaction.uuid


def _read_manifest(manifest_path: Path) -> MaterializeManifest:
    """Parse an existing sidecar manifest, failing loudly on any damage.

    :param manifest_path: Sidecar JSON path (must exist).
    :returns: The parsed manifest.
    :raises ValueError: The sidecar is unreadable or fails strict validation.
    """
    try:
        return MaterializeManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise ValueError(f"unparsable materialize sidecar {manifest_path}: {exc}") from exc


def _manifest_matches_request(
    manifest: MaterializeManifest,
    requested_hash: str,
    resolved_txid: str | None,
) -> bool:
    """Return whether a sidecar covers the request and selected source identity.

    :param manifest: Parsed sidecar manifest.
    :param requested_hash: Hash of the current materialization request.
    :param resolved_txid: Current source transaction for an unpinned request.
    :returns: Whether the manifest is internally sound and current.
    """
    stored_hash = request_hash(
        manifest.source_uri,
        manifest.txid,
        manifest.resolved_version,
        manifest.columns,
        manifest.limit,
    )
    if manifest.txid is not None:
        source_matches = manifest.resolved_txid in (None, manifest.txid)
    else:
        source_matches = manifest.resolved_txid == resolved_txid
    return (
        manifest.request_hash == stored_hash
        and manifest.request_hash == requested_hash
        and source_matches
    )


def _validate_materialized_destination(
    dest_path: Path, manifest: MaterializeManifest
) -> None:
    """Verify that a cache still names the Lance dataset originally published.

    :param dest_path: Existing local materialized dataset.
    :param manifest: Parsed cache sidecar.
    :raises ValueError: The sidecar lacks destination identity, the dataset was replaced, or Lance
        reports structural corruption.
    """
    if manifest.materialized_txid is None:
        raise ValueError(
            f"materialized dataset {dest_path} sidecar has no destination identity; "
            "delete the dataset and re-materialize"
        )
    try:
        destination = _retry_lance_read(
            "destination_open", lambda: lance.dataset(str(dest_path))
        )
        transaction = _retry_lance_read(
            "destination_transaction_read",
            lambda: destination.read_transaction(destination.version),
        )
        _retry_lance_read("destination_validate", destination.validate)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"materialized dataset {dest_path} failed Lance validation; "
            "delete the dataset and re-materialize"
        ) from exc
    if transaction is None or transaction.uuid != manifest.materialized_txid:
        raise ValueError(
            f"materialized dataset {dest_path} identity differs from its sidecar; "
            "delete the dataset and re-materialize"
        )


def _reuse_or_raise(
    dest_path: Path,
    *,
    source_uri: str,
    txid: str | None,
    columns: tuple[str, ...],
    limit: int | None,
    resolved_version: int | None = None,
    resolved_txid: str | None = None,
) -> Path:
    """Validate an existing destination against the current request.

    Recomputes the hash from the current request. Pinned requests retain the
    stored resolved version; unpinned requests include the source's current
    version so an advanced source rejects the stale cache.

    :param dest_path: Existing materialized dataset directory.
    :param source_uri: Current request's source URI.
    :param txid: Current request's transaction uuid, or ``None`` for latest.
    :param columns: Current request's projected columns.
    :param limit: Current request's row cap.
    :param resolved_version: Current latest version for an unpinned request.
    :param resolved_txid: Current source transaction uuid for an unpinned request.
    :returns: ``dest_path`` on a cache hit.
    :raises ValueError: The sidecar or local Lance dataset is invalid, or the
        request diverges from the recorded source and destination identities.
    """
    manifest_path = sidecar_path(dest_path)
    if not manifest_path.is_file():
        raise ValueError(
            f"materialized dataset {dest_path} has no sidecar manifest "
            f"({manifest_path}); delete the dataset and re-materialize"
        )
    manifest = _read_manifest(manifest_path)
    requested_version = (
        manifest.resolved_version if resolved_version is None else resolved_version
    )
    requested_hash = request_hash(source_uri, txid, requested_version, columns, limit)
    if not _manifest_matches_request(manifest, requested_hash, resolved_txid):
        raise ValueError(
            f"materialize request hash mismatch for {dest_path}: sidecar was written "
            f"for source={manifest.source_uri!r} txid={manifest.txid!r} "
            f"columns={manifest.columns} limit={manifest.limit}; current request is "
            f"source={source_uri!r} txid={txid!r} columns={columns} limit={limit} — "
            "delete the dataset and re-materialize"
        )
    _validate_materialized_destination(dest_path, manifest)
    logger.info(
        "lance_materialize.cache_hit",
        dest_path=str(dest_path),
        txid=txid,
        resolved_version=manifest.resolved_version,
    )
    _evict_lance_data_cache(dest_path)
    return dest_path


def _evict_lance_data_cache(dataset_path: Path) -> None:
    """Release clean pages for a completed local Lance dataset.

    :param dataset_path: Published dataset whose data-file pages should be evicted.
    """
    advise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is None or dontneed is None:
        return
    for data_path in (dataset_path / "data").rglob("*"):
        if not data_path.is_file():
            continue
        try:
            fd = os.open(data_path, os.O_RDWR)
        except OSError as error:
            logger.warning(
                "lance_materialize.cache_flush_open_failed",
                path=str(data_path),
                errno=error.errno,
            )
        else:
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        try:
            with data_path.open("rb", buffering=0) as stream:
                advise(stream.fileno(), 0, 0, dontneed)
        except OSError as error:
            logger.warning(
                "lance_materialize.cache_evict_failed",
                path=str(data_path),
                errno=error.errno,
            )


def _write_materialized_snapshot(
    snapshot: lance.LanceDataset,
    *,
    dest_path: Path,
    manifest: MaterializeManifest,
    batch_size: int,
) -> Path:
    """Write one selected source snapshot and its cache manifest.

    :param snapshot: Selected source dataset snapshot.
    :param dest_path: Local destination dataset directory.
    :param manifest: Validated request and source identity to persist.
    :param batch_size: Scan batch size in rows.
    :returns: ``dest_path``.
    :raises OSError: Manifest writing or atomic publication fails without a winner.
    :raises ValueError: The written dataset has no transaction identity.
    """
    scanner = snapshot.scanner(
        columns=list(manifest.columns),
        limit=manifest.limit,
        batch_size=batch_size,
        io_buffer_size=_MATERIALIZE_IO_BUFFER_SIZE,
        fragment_readahead=_MATERIALIZE_FRAGMENT_READAHEAD,
        batch_readahead=_MATERIALIZE_BATCH_READAHEAD,
    )
    logger.info(
        "lance_materialize.start",
        source_uri=manifest.source_uri,
        dest_path=str(dest_path),
        txid=manifest.txid,
        resolved_version=manifest.resolved_version,
        columns=manifest.columns,
        limit=manifest.limit,
    )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = dest_path.parent / f".{dest_path.name}.{uuid4().hex}.partial"
    transaction_properties = (
        {"cloned_from_txn": manifest.txid}
        if manifest.txid is not None
        else {"cloned_from_version": str(manifest.resolved_version)}
    )
    written = lance.write_dataset(
        scanner.to_batches(),
        str(staging_path),
        schema=scanner.projected_schema,
        transaction_properties=transaction_properties,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        max_rows_per_group=_MATERIALIZE_MAX_ROWS_PER_GROUP,
        max_bytes_per_file=_MATERIALIZE_MAX_BYTES_PER_FILE,
    )
    row_count = written.count_rows()
    transaction = written.read_transaction(written.version)
    if transaction is None:
        raise ValueError(f"materialized dataset {staging_path} has no transaction record")
    published_manifest = manifest.model_copy(
        update={"materialized_txid": transaction.uuid}
    )
    try:
        sidecar_path(staging_path).write_text(
            published_manifest.model_dump_json(), encoding="utf-8"
        )
        staging_path.replace(dest_path)
    except OSError:
        winner_exists = dest_path.exists()
        shutil.rmtree(staging_path, ignore_errors=True)
        if not winner_exists:
            raise
        return _reuse_or_raise(
            dest_path,
            source_uri=manifest.source_uri,
            txid=manifest.txid,
            columns=manifest.columns,
            limit=manifest.limit,
            resolved_version=(
                manifest.resolved_version if manifest.txid is None else None
            ),
            resolved_txid=manifest.resolved_txid,
        )
    _evict_lance_data_cache(dest_path)
    logger.info(
        "lance_materialize.done",
        dest_path=str(dest_path),
        rows=row_count,
    )
    return dest_path


# DOC502: the documented LookupError/ValueError propagate from
# resolve_txid_version, _reuse_or_raise, and _transaction_uuid.
def materialize_lance_subset(  # noqa: DOC502
    source_uri: str,
    dest_path: Path,
    *,
    txid: str | None,
    columns: Sequence[str],
    limit: int | None = None,
    batch_size: int = _MATERIALIZE_BATCH_SIZE,
) -> Path:
    """Stream a projected source snapshot scan into ``dest_path``.

    Peak memory is ~one batch; transferred bytes scale with the subset, not
    the source. A txid pins the source snapshot when supplied; otherwise the
    latest version at hydration time is used.

    :param source_uri: Source dataset — ``r2://`` URI (resolved via
        :func:`synth_setter.pipeline.r2_io.lance_target`) or local path.
    :param dest_path: Local destination dataset directory; must not hold an
        unrelated dataset.
    :param txid: Transaction uuid pinning the source snapshot, or ``None`` for latest.
    :param columns: Columns to project, in scan order.
    :param limit: First-N row cap, or ``None`` for all rows.
    :param batch_size: Scan batch size in rows — the streaming memory unit.
    :returns: ``dest_path``.
    :raises LookupError: ``txid`` matches no live source version.
    :raises RuntimeError: A transient source read exhausts the retry budget.
    :raises ValueError: ``dest_path`` exists with a missing/unparsable
        sidecar or a sidecar whose request hash differs from this request.
    """
    dest_path = Path(dest_path)
    requested_columns = tuple(columns)
    if dest_path.exists() and txid is not None:
        return _reuse_or_raise(
            dest_path,
            source_uri=source_uri,
            txid=txid,
            columns=requested_columns,
            limit=limit,
        )
    ds = _open_source(source_uri)
    resolved_version = ds.version if txid is None else resolve_txid_version(ds, txid)
    resolved_txid = _transaction_uuid(ds, resolved_version)
    if dest_path.exists():
        return _reuse_or_raise(
            dest_path,
            source_uri=source_uri,
            txid=txid,
            columns=requested_columns,
            limit=limit,
            resolved_version=resolved_version,
            resolved_txid=resolved_txid,
        )
    snapshot = ds.checkout_version(resolved_version)
    manifest = MaterializeManifest(
        source_uri=source_uri,
        txid=txid,
        resolved_version=resolved_version,
        resolved_txid=resolved_txid,
        columns=requested_columns,
        limit=limit,
        request_hash=request_hash(source_uri, txid, resolved_version, requested_columns, limit),
    )
    return _write_materialized_snapshot(
        snapshot,
        dest_path=dest_path,
        manifest=manifest,
        batch_size=batch_size,
    )


def _require_dataset_complete(source_root_uri: str) -> None:
    """Require the finalize marker before reading any split from a dataset root.

    :param source_root_uri: R2 URI, file URI, or local dataset root.
    :raises FileNotFoundError: The dataset completion marker is absent.
    """
    if r2_io.is_r2_uri(source_root_uri):
        marker = f"{source_root_uri.rstrip('/')}/{DATASET_COMPLETE_FILENAME}"
        marker_exists = r2_io.object_size(marker) is not None
    else:
        source_root = (
            file_uri_to_path(source_root_uri)
            if is_file_uri(source_root_uri)
            else Path(source_root_uri)
        )
        marker_path = source_root / DATASET_COMPLETE_FILENAME
        marker = str(marker_path)
        marker_exists = marker_path.is_file()
    if not marker_exists:
        raise FileNotFoundError(
            f"dataset completion marker {marker} is missing; finalize the dataset before hydration"
        )


def materialize_splits(
    source_root_uri: str,
    dest_root: Path,
    *,
    txids: Mapping[str, str] | None,
    projection: Mapping[str, Sequence[str]],
    row_limit: int | None,
    shard_suffix: str,
) -> None:
    """Materialize each split under a root, then rclone non-Lance sidecars.

    :param source_root_uri: Hydration root (``r2://``, ``file://``, or local path)
        holding the split datasets.
    :param dest_root: Local destination root; each split lands at
        ``dest_root / f"{split}{shard_suffix}"``.
    :param txids: Per-split transaction uuids, or ``None`` to use latest snapshots.
    :param projection: Columns to materialize per split.
    :param row_limit: First-N row cap per split, or ``None`` for all rows.
    :param shard_suffix: Split dataset suffix, e.g. ``.lance``.
    """
    _require_dataset_complete(source_root_uri)
    for split, columns in projection.items():
        name = f"{split}{shard_suffix}"
        materialize_lance_subset(
            f"{source_root_uri.rstrip('/')}/{name}",
            dest_root / name,
            txid=txids[split] if txids is not None else None,
            columns=columns,
            limit=row_limit,
        )
    # Non-Lance sidecars (stats.npz, dataset.json) still hydrate via rclone;
    # split datasets and pipeline-internal metadata/ never feed the loaders.
    r2_io.download_dir_no_overwrite(
        source_root_uri,
        dest_root,
        exclude=f"{{*{shard_suffix}/**,metadata/**}}",
    )
