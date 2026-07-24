"""Rematerialize a Lance column/row subset to local disk.

Streams a projected scan of one source snapshot into a fresh local Lance
dataset, so hydration transfers only the columns and rows a training run
reads instead of the whole dataset directory. A sidecar manifest beside
the destination records the request and gates cache reuse: a rerun with the
same request reuses the local copy; any drift fails loudly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import lance
import structlog
from pydantic import BaseModel, ConfigDict, ValidationError

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.file_uri import file_uri_to_path, is_file_uri

logger = structlog.get_logger(__name__)

_SIDECAR_SUFFIX = ".materialize.json"


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
    columns: tuple[str, ...]
    limit: int | None
    request_hash: str


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
    payload = json.dumps(
        {
            "source_uri": source_uri,
            "txid": txid,
            "resolved_version": resolved_version,
            "columns": list(columns),
            "limit": limit,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sidecar_path(dest_path: Path) -> Path:
    """Return the manifest path beside a materialized dataset directory.

    :param dest_path: Materialized Lance dataset directory.
    :returns: ``<dest>.materialize.json`` in the same parent directory.
    """
    return dest_path.parent / (dest_path.name + _SIDECAR_SUFFIX)


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
    for entry in ds.versions():
        transaction = ds.read_transaction(entry["version"])
        if transaction is not None and transaction.uuid == txid:
            return entry["version"]
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
        open_uri, storage_options = str(file_uri_to_path(source_uri)), None
    else:
        open_uri, storage_options = source_uri, None
    return lance.dataset(open_uri, storage_options=storage_options)


def _transaction_uuid(ds: lance.LanceDataset, version: int) -> str:
    """Return the transaction uuid identifying a source version.

    :param ds: Open source dataset.
    :param version: Source version to identify.
    :returns: Transaction uuid for ``version``.
    :raises ValueError: The source version has no transaction record.
    """
    transaction = ds.read_transaction(version)
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
    :raises ValueError: The sidecar is missing/unparsable, its stored hash
        does not cover its own fields, or the request diverges from it —
        never silently reuse a stale local subset.
    """
    manifest_path = sidecar_path(dest_path)
    if not manifest_path.is_file():
        raise ValueError(
            f"materialized dataset {dest_path} has no sidecar manifest "
            f"({manifest_path}); delete the dataset and re-materialize"
        )
    manifest = _read_manifest(manifest_path)
    stored_hash = request_hash(
        manifest.source_uri,
        manifest.txid,
        manifest.resolved_version,
        manifest.columns,
        manifest.limit,
    )
    requested_version = (
        manifest.resolved_version if resolved_version is None else resolved_version
    )
    requested_hash = request_hash(source_uri, txid, requested_version, columns, limit)
    expected_resolved_txid = txid if txid is not None else resolved_txid
    source_changed = manifest.resolved_txid != expected_resolved_txid
    if (
        manifest.request_hash != stored_hash
        or manifest.request_hash != requested_hash
        or source_changed
    ):
        raise ValueError(
            f"materialize request hash mismatch for {dest_path}: sidecar was written "
            f"for source={manifest.source_uri!r} txid={manifest.txid!r} "
            f"columns={manifest.columns} limit={manifest.limit}; current request is "
            f"source={source_uri!r} txid={txid!r} columns={columns} limit={limit} — "
            "delete the dataset and re-materialize"
        )
    logger.info(
        "lance_materialize.cache_hit",
        dest_path=str(dest_path),
        txid=txid,
        resolved_version=manifest.resolved_version,
    )
    return dest_path


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
    """
    scanner = snapshot.scanner(
        columns=list(manifest.columns), limit=manifest.limit, batch_size=batch_size
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
    transaction_properties = (
        {"cloned_from_txn": manifest.txid}
        if manifest.txid is not None
        else {"cloned_from_version": str(manifest.resolved_version)}
    )
    written = lance.write_dataset(
        scanner.to_batches(),
        str(dest_path),
        schema=scanner.projected_schema,
        transaction_properties=transaction_properties,
    )
    sidecar_path(dest_path).write_text(manifest.model_dump_json(), encoding="utf-8")
    logger.info(
        "lance_materialize.done",
        dest_path=str(dest_path),
        rows=written.count_rows(),
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
    batch_size: int = 512,
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


def materialize_splits(
    source_root_uri: str,
    dest_root: Path,
    *,
    txids: Mapping[str, str] | None,
    columns_for: Callable[[str], Sequence[str]],
    row_limit: int | None,
    shard_suffix: str,
) -> None:
    """Materialize each split under a root, then rclone non-Lance sidecars.

    :param source_root_uri: Hydration root (``r2://``, ``file://``, or local path)
        holding the split datasets.
    :param dest_root: Local destination root; each split lands at
        ``dest_root / f"{split}{shard_suffix}"``.
    :param txids: Per-split transaction uuids, or ``None`` to use latest snapshots.
    :param columns_for: Callback returning the columns to project for a given split name.
    :param row_limit: First-N row cap per split, or ``None`` for all rows.
    :param shard_suffix: Split dataset suffix, e.g. ``.lance``.
    """
    split_txids = (
        txids.items()
        if txids is not None
        else ((split, None) for split in ("train", "val", "test"))
    )
    for split, txid in split_txids:
        name = f"{split}{shard_suffix}"
        materialize_lance_subset(
            f"{source_root_uri.rstrip('/')}/{name}",
            dest_root / name,
            txid=txid,
            columns=columns_for(split),
            limit=row_limit,
        )
    # Non-Lance sidecars (stats.npz, dataset.json) still hydrate via rclone;
    # split datasets and pipeline-internal metadata/ never feed the loaders.
    r2_io.download_dir_no_overwrite(
        source_root_uri,
        dest_root,
        exclude=f"{{*{shard_suffix}/**,metadata/**}}",
    )
