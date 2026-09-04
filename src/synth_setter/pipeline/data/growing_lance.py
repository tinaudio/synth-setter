"""Append-only native Lance train growth and exact local activation."""

from __future__ import annotations

import fcntl
import json
import shutil
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Iterator

import lance
import numpy as np
import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from synth_setter.data.vst.shapes import MEL_SPEC_FIELD, dataset_field_shapes
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_materialize import retry_lance_read
from synth_setter.pipeline.data.lance_shard import (
    LANCE_DATA_STORAGE_VERSION,
    LANCE_MAX_BYTES_PER_FILE,
    fragment_schema_matches,
)
from synth_setter.pipeline.data.stats import (
    WelfordState,
    finalize,
    load_welford,
    merge_welford,
    save_welford,
)

if TYPE_CHECKING:
    from synth_setter.pipeline.schemas.spec import DatasetSpec, ShardSpec

_GROWING_PENDING_PROPERTY = "synth_setter.growing_pending_identity"
_LOCAL_SOURCE_PROPERTY = "synth_setter.growing_remote_identity"


class ActiveGrowingSnapshot(BaseModel):
    """Strict identity joining one remote snapshot to one local Lance version.

    .. attribute :: model_config

        Strict frozen model configuration.
    .. attribute :: branch

        Native remote branch name.
    .. attribute :: remote_version

        Exact ready remote version.
    .. attribute :: remote_transaction

        Exact ready remote transaction.
    .. attribute :: local_version

        Exact shared local version.
    .. attribute :: local_transaction

        Exact shared local transaction.
    .. attribute :: dataset_path

        Shared local train dataset path.
    .. attribute :: version_stats_path

        Version-bound local metadata directory.
    .. attribute :: dataset_spec_fingerprint

        Frozen producer specification digest.
    .. attribute :: row_count

        Expected cumulative rows.
    .. attribute :: fragment_count

        Expected cumulative fragments.
    .. attribute :: schema_fingerprint

        Projected local schema digest.
    .. attribute :: stats_sha256

        Derived statistics digest.
    .. attribute :: welford_sha256

        Cumulative Welford digest.
    .. attribute :: high_watermark

        Next direct train position.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    branch: str
    remote_version: int
    remote_transaction: str
    local_version: int
    local_transaction: str
    dataset_path: str
    version_stats_path: str
    dataset_spec_fingerprint: str
    row_count: int
    fragment_count: int
    schema_fingerprint: str
    stats_sha256: str
    welford_sha256: str
    high_watermark: int


class GrowingSnapshot(BaseModel):
    """Strict identity of one ready append-only native branch version.

    .. attribute :: model_config

        Strict frozen model configuration.
    .. attribute :: branch

        Native branch name.
    .. attribute :: branch_uri

        Native branch URI.
    .. attribute :: version

        Ready native version.
    .. attribute :: baseline_version

        Frozen baseline version.
    .. attribute :: baseline_transaction

        Frozen baseline transaction.
    .. attribute :: transaction

        Ready native transaction.
    .. attribute :: baseline_train_shards

        Baseline train fragment count.
    .. attribute :: max_train_shards

        Total train fragment capacity.
    .. attribute :: num_extra_shards

        Requested fragments per refresh.
    .. attribute :: high_watermark

        Next direct train position.
    .. attribute :: dataset_spec_fingerprint

        Frozen producer specification digest.
    .. attribute :: row_count

        Expected cumulative rows.
    .. attribute :: fragment_count

        Expected cumulative fragments.
    .. attribute :: schema_fingerprint

        Remote schema digest.
    .. attribute :: stats_sha256

        Derived statistics digest.
    .. attribute :: welford_sha256

        Cumulative Welford digest.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    branch: str
    branch_uri: str
    version: int
    baseline_version: int
    baseline_transaction: str
    transaction: str
    baseline_train_shards: int
    max_train_shards: int
    num_extra_shards: int
    high_watermark: int
    dataset_spec_fingerprint: str
    row_count: int
    fragment_count: int
    schema_fingerprint: str
    stats_sha256: str
    welford_sha256: str


class PendingRefreshRequest(BaseModel):
    """Durable bounded shard range bound to one exact ready snapshot.

    .. attribute :: model_config

        Strict frozen model configuration.
    .. attribute :: branch

        Native branch name.
    .. attribute :: source_version

        Ready source version.
    .. attribute :: source_transaction

        Ready source transaction.
    .. attribute :: dataset_spec_fingerprint

        Frozen producer specification digest.
    .. attribute :: enqueue_shard_ids

        Direct train positions to render.
    .. attribute :: next_high_watermark

        High-watermark after publication.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    branch: str
    source_version: int
    source_transaction: str
    dataset_spec_fingerprint: str
    enqueue_shard_ids: tuple[int, ...]
    next_high_watermark: int


@dataclass(frozen=True)
class PendingRefresh:
    """One non-empty bounded range selected from the ready high-watermark.

    .. attribute :: enqueue_shard_ids

        Direct train positions in the range.
    .. attribute :: next_high_watermark

        Exclusive end of the range.
    """

    enqueue_shard_ids: tuple[int, ...]
    next_high_watermark: int


@dataclass(frozen=True)
class GrowingPlan:
    """Immutable bounds for append-only train-shard growth.

    .. attribute :: baseline_train_shards

        Frozen baseline fragment count.
    .. attribute :: max_train_shards

        Total fragment capacity.
    .. attribute :: num_extra_shards

        Requested fragments per refresh.
    .. attribute :: high_watermark

        Next direct train position.
    """

    baseline_train_shards: int
    max_train_shards: int
    num_extra_shards: int
    high_watermark: int

    def __post_init__(self) -> None:
        if self.baseline_train_shards < 1:
            raise ValueError("baseline_train_shards must be positive")
        if self.max_train_shards < self.baseline_train_shards:
            raise ValueError("max_train_shards must include every baseline train shard")
        if self.num_extra_shards < 1:
            raise ValueError("num_extra_shards must be positive")
        if not self.baseline_train_shards <= self.high_watermark <= self.max_train_shards:
            raise ValueError("high_watermark must lie between baseline and maximum")

    def next_refresh(self) -> PendingRefresh | None:
        """Return the next bounded range.

        :returns: Next range, or ``None`` at capacity.
        """
        if self.high_watermark == self.max_train_shards:
            return None
        next_high_watermark = min(
            self.high_watermark + self.num_extra_shards, self.max_train_shards
        )
        return PendingRefresh(
            enqueue_shard_ids=tuple(range(self.high_watermark, next_high_watermark)),
            next_high_watermark=next_high_watermark,
        )

    def advance(self, refresh: PendingRefresh) -> GrowingPlan:
        """Advance only after the selected range is durably ready.

        :param refresh: Exact next range.
        :returns: Plan advanced to the range end.
        :raises ValueError: The range is not the next bounded range.
        """
        expected = self.next_refresh()
        if expected is None or refresh != expected:
            raise ValueError("refresh does not match the next bounded range")
        return GrowingPlan(
            self.baseline_train_shards,
            self.max_train_shards,
            self.num_extra_shards,
            refresh.next_high_watermark,
        )

    def extra_shard(self, spec: DatasetSpec, shard_id: int) -> ShardSpec:
        """Map a direct train position beyond all frozen baseline samples.

        :param spec: Frozen producer specification.
        :param shard_id: Direct train position.
        :returns: Deterministic shard specification.
        :raises ValueError: The position lies outside the growing range.
        """
        from synth_setter.pipeline.schemas.spec import ShardSpec

        if not self.baseline_train_shards <= shard_id < self.max_train_shards:
            raise ValueError(
                f"growing shard ID must be in [{self.baseline_train_shards}, "
                f"{self.max_train_shards})"
            )
        extra_index = shard_id - self.baseline_train_shards
        sample_offset = sum(spec.train_val_test_sizes) + (
            extra_index * spec.render.samples_per_shard
        )
        return ShardSpec(
            shard_id=shard_id,
            filename=f"shard-{shard_id:06d}{spec.output_format.extension}",
            seed=spec.base_seed,
            sample_offset=sample_offset,
        )


def pending_refresh_request(current: GrowingSnapshot) -> PendingRefreshRequest | None:
    """Freeze the next bounded range against one exact ready snapshot.

    :param current: Exact ready snapshot.
    :returns: Bound request, or ``None`` at capacity.
    """
    refresh = _plan(current).next_refresh()
    if refresh is None:
        return None
    return PendingRefreshRequest(
        branch=current.branch,
        source_version=current.version,
        source_transaction=current.transaction,
        dataset_spec_fingerprint=current.dataset_spec_fingerprint,
        enqueue_shard_ids=refresh.enqueue_shard_ids,
        next_high_watermark=refresh.next_high_watermark,
    )


def dataset_spec_fingerprint(spec: DatasetSpec) -> str:
    """Return the stable digest bound into growing records.

    :param spec: Producer specification.
    :returns: SHA-256 hexadecimal digest.
    """
    return sha256(spec.model_dump_json().encode()).hexdigest()


def _plan(snapshot: GrowingSnapshot) -> GrowingPlan:
    return GrowingPlan(
        snapshot.baseline_train_shards,
        snapshot.max_train_shards,
        snapshot.num_extra_shards,
        snapshot.high_watermark,
    )


def _lance_target(uri: Path | str) -> tuple[str, dict[str, str] | None]:
    value = str(uri)
    if r2_io.is_r2_uri(value):
        return r2_io.lance_target(value)
    return value, r2_io.r2_storage_options() if value.startswith("s3://") else None


def _open_train(train_uri: Path | str) -> lance.LanceDataset:
    target, storage_options = _lance_target(train_uri)
    return retry_lance_read(
        "growing_train_open",
        lambda: lance.dataset(target, storage_options=storage_options),
    )


def _transaction(dataset: lance.LanceDataset, version: int) -> lance.Transaction:
    transaction = retry_lance_read(
        "growing_transaction_read", lambda: dataset.read_transaction(version)
    )
    if transaction is None:
        raise ValueError(f"Lance version {version} has no transaction")
    return transaction


def _schema_fingerprint(dataset: lance.LanceDataset) -> str:
    return sha256(dataset.schema.serialize().to_pybytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _expected_mel_shape(spec: DatasetSpec) -> tuple[int, ...]:
    return dataset_field_shapes(spec.render, spec.num_params)[MEL_SPEC_FIELD][1:]


def _version_dir(metadata_root: Path, version: int) -> Path:
    return metadata_root / "versions" / str(version)


def _write_snapshot(metadata_root: Path, snapshot: GrowingSnapshot) -> None:
    version_dir = _version_dir(metadata_root, snapshot.version)
    version_dir.mkdir(parents=True, exist_ok=True)
    destination = version_dir / "snapshot.json"
    staging = destination.with_suffix(".json.tmp")
    staging.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
    staging.replace(destination)


def _create_or_resume_branch(
    dataset: lance.LanceDataset,
    branch: str,
    baseline_version: int,
    contract: dict[str, str],
) -> lance.LanceDataset:
    branch_info = dataset.branches.list().get(branch)
    if branch_info is None:
        checked_out = dataset.create_branch(branch, baseline_version)
        dataset.branches.replace_metadata(branch, contract)
        return checked_out
    if (
        branch_info.get("parent_version") != baseline_version
        or branch_info.get("metadata") != contract
    ):
        raise ValueError(f"branch {branch!r} already exists with a different growing contract")
    checked_out = dataset.checkout_version((branch, None))
    if checked_out.version != baseline_version:
        raise ValueError(f"branch {branch!r} already advanced to version {checked_out.version}")
    return checked_out


def _set_ready_tag(dataset: lance.LanceDataset, branch: str, version: int) -> None:
    name = f"{branch}-ready"
    try:
        ready_version = retry_lance_read(
            "growing_ready_tag_read", lambda: dataset.tags.get_version(name)
        )
    except ValueError:
        ready_version = None
    if ready_version is None:
        dataset.tags.create(name, (branch, version))
    elif ready_version != version:
        dataset.tags.update(name, (branch, version))


def initialize_growing_branch(
    train_uri: Path | str,
    *,
    spec: DatasetSpec,
    branch: str,
    baseline_version: int,
    metadata_root: Path,
    max_train_shards: int,
    num_extra_shards: int,
    publish_metadata: Callable[[GrowingSnapshot, Path], None] | None = None,
) -> GrowingSnapshot:
    """Create an append-only branch pinned to a finalized baseline.

    :param train_uri: Baseline train dataset URI.
    :param spec: Frozen producer specification.
    :param branch: Native branch name.
    :param baseline_version: Complete baseline version.
    :param metadata_root: Local metadata workspace.
    :param max_train_shards: Total fragment capacity including baseline.
    :param num_extra_shards: Maximum fragments per refresh.
    :param publish_metadata: Optional durable metadata publisher.
    :returns: Initialized ready snapshot.
    :raises ValueError: Baseline artifacts or immutable contract are invalid.
    """
    dataset = _open_train(train_uri)
    baseline = dataset.checkout_version(baseline_version)
    baseline_transaction = _transaction(dataset, baseline_version)
    baseline_train_shards = len(baseline.get_fragments())
    expected_train_shards = spec.train_val_test_sizes[0] // spec.render.samples_per_shard
    if baseline_train_shards != expected_train_shards:
        raise ValueError(
            f"baseline train fragments {baseline_train_shards} do not match spec "
            f"{expected_train_shards}"
        )
    if baseline.count_rows() != spec.train_val_test_sizes[0]:
        raise ValueError("baseline train row count does not match the frozen specification")
    GrowingPlan(
        baseline_train_shards,
        max_train_shards,
        num_extra_shards,
        baseline_train_shards,
    )
    baseline_dir = _version_dir(metadata_root, baseline_version)
    stats_path = baseline_dir / "stats.npz"
    welford_path = baseline_dir / "welford.npz"
    if not stats_path.is_file() or not welford_path.is_file():
        raise ValueError("baseline stats.npz and welford.npz are required")
    load_welford(welford_path, expected_shape=_expected_mel_shape(spec))
    fingerprint = dataset_spec_fingerprint(spec)
    contract = {
        "synth_setter.growing_baseline_train_shards": str(baseline_train_shards),
        "synth_setter.growing_baseline_transaction": baseline_transaction.uuid,
        "synth_setter.growing_baseline_version": str(baseline_version),
        "synth_setter.growing_dataset_spec_fingerprint": fingerprint,
        "synth_setter.growing_max_train_shards": str(max_train_shards),
        "synth_setter.growing_num_extra_shards": str(num_extra_shards),
    }
    checked_out = _create_or_resume_branch(dataset, branch, baseline_version, contract)
    snapshot = GrowingSnapshot(
        branch=branch,
        branch_uri=str(checked_out.uri),
        version=checked_out.version,
        baseline_version=baseline_version,
        baseline_transaction=baseline_transaction.uuid,
        transaction=baseline_transaction.uuid,
        baseline_train_shards=baseline_train_shards,
        max_train_shards=max_train_shards,
        num_extra_shards=num_extra_shards,
        high_watermark=baseline_train_shards,
        dataset_spec_fingerprint=fingerprint,
        row_count=baseline.count_rows(),
        fragment_count=baseline_train_shards,
        schema_fingerprint=_schema_fingerprint(baseline),
        stats_sha256=_file_sha256(stats_path),
        welford_sha256=_file_sha256(welford_path),
    )
    _write_snapshot(metadata_root, snapshot)
    if publish_metadata is not None:
        publish_metadata(snapshot, baseline_dir)
    _set_ready_tag(dataset, branch, checked_out.version)
    return snapshot


def _fragment_storage_identity(
    fragment: lance.fragment.FragmentMetadata,
) -> tuple[tuple[str, ...], int]:
    return tuple(data_file.path for data_file in fragment.files), fragment.physical_rows


def _validate_new_fragment_files(
    dataset: lance.LanceDataset,
    fragments: Sequence[lance.fragment.FragmentMetadata],
    expected_rows: int,
    *,
    storage_options: dict[str, str] | None,
) -> None:
    from lance.file import LanceFileReader

    existing = {
        data_file.path
        for fragment in dataset.get_fragments()
        for data_file in fragment.metadata.files
    }
    for fragment in fragments:
        if len(fragment.files) != 1 or fragment.physical_rows != expected_rows:
            raise ValueError(f"each new fragment must contain exactly {expected_rows} rows")
        data_file = fragment.files[0]
        path = PurePosixPath(data_file.path)
        if not data_file.path or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe new fragment data path {data_file.path!r}")
        if data_file.path in existing:
            raise ValueError("append fragment reuses baseline storage identity")
        reader = LanceFileReader(
            f"{dataset.uri}/data/{data_file.path}", storage_options=storage_options
        )
        if reader.num_rows() != expected_rows:
            raise ValueError("new fragment physical row count differs from metadata")
        if not fragment_schema_matches(reader.metadata().schema, dataset.schema):
            raise ValueError("new fragment schema does not match growing branch")


def _pending_identity(
    current: GrowingSnapshot,
    pending: PendingRefreshRequest,
    fragments: Sequence[lance.fragment.FragmentMetadata],
    states: Sequence[WelfordState],
) -> str:
    digest = sha256()
    digest.update(current.transaction.encode())
    digest.update(pending.model_dump_json().encode())
    for fragment, state in zip(fragments, states, strict=True):
        digest.update(json.dumps(fragment.to_json(), sort_keys=True).encode())
        digest.update(np.asarray(state[0], dtype=np.int64).tobytes())
        digest.update(np.asarray(state[1]).tobytes())
        digest.update(np.asarray(state[2]).tobytes())
    return digest.hexdigest()


def _commit_append(
    dataset: lance.LanceDataset,
    fragments: Sequence[lance.fragment.FragmentMetadata],
    identity: str,
) -> lance.LanceDataset:
    transaction = lance.Transaction(
        read_version=dataset.version,
        operation=lance.LanceOperation.Append(list(fragments)),
        transaction_properties={_GROWING_PENDING_PROPERTY: identity},
    )
    return lance.LanceDataset.commit(dataset, transaction)


def _publish_cumulative_stats(
    spec: DatasetSpec,
    current: GrowingSnapshot,
    states: Sequence[WelfordState],
    metadata_root: Path,
    version: int,
) -> tuple[str, str]:
    current_path = _version_dir(metadata_root, current.version) / "welford.npz"
    if _file_sha256(current_path) != current.welford_sha256:
        raise ValueError("prior cumulative Welford digest does not match snapshot")
    cumulative = load_welford(current_path, expected_shape=_expected_mel_shape(spec))
    for state in states:
        cumulative = merge_welford(cumulative, state)
    version_dir = _version_dir(metadata_root, version)
    version_dir.mkdir(parents=True, exist_ok=True)
    welford_path = version_dir / "welford.npz"
    save_welford(welford_path, cumulative, expected_shape=_expected_mel_shape(spec))
    mean, std = finalize(cumulative, mask_degenerate=spec.mask_degenerate_bins)
    stats_path = version_dir / "stats.npz"
    np.savez(stats_path, mean=mean, std=std)
    return _file_sha256(stats_path), _file_sha256(welford_path)


def publish_growing_branch(
    train_uri: Path | str,
    *,
    spec: DatasetSpec,
    current: GrowingSnapshot,
    fragments: Sequence[lance.fragment.FragmentMetadata],
    welford: Sequence[WelfordState],
    metadata_root: Path,
    publish_metadata: Callable[[GrowingSnapshot, Path], None] | None = None,
) -> GrowingSnapshot:
    """Append one exact bounded range and expose readiness last.

    :param train_uri: Baseline train dataset URI.
    :param spec: Frozen producer specification.
    :param current: Exact ready source snapshot.
    :param fragments: New fragment metadata in direct-position order.
    :param welford: Per-fragment Welford states in matching order.
    :param metadata_root: Local metadata workspace.
    :param publish_metadata: Optional durable metadata publisher.
    :returns: Newly ready snapshot, or ``current`` at capacity.
    :raises ValueError: Input identity or append invariants fail validation.
    """
    if dataset_spec_fingerprint(spec) != current.dataset_spec_fingerprint:
        raise ValueError("dataset spec fingerprint does not match growing baseline")
    pending = pending_refresh_request(current)
    if pending is None:
        if fragments or welford:
            raise ValueError("growing branch is at capacity; refusing additional shards")
        return current
    expected_count = len(pending.enqueue_shard_ids)
    if len(fragments) != expected_count or len(welford) != expected_count:
        raise ValueError(f"publication requires exactly {expected_count} new fragments and states")
    if any(state[0] != spec.render.samples_per_shard for state in welford):
        raise ValueError("each new Welford state must cover one complete shard")

    branch_target, storage_options = _lance_target(current.branch_uri)
    source = retry_lance_read(
        "growing_source_version_open",
        lambda: lance.dataset(branch_target, storage_options=storage_options).checkout_version(
            current.version
        ),
    )
    old_identities = tuple(
        _fragment_storage_identity(fragment.metadata) for fragment in source.get_fragments()
    )
    transaction_source = (
        _open_train(train_uri) if current.version == current.baseline_version else source
    )
    source_transaction = _transaction(transaction_source, source.version)
    if (
        source_transaction.uuid != current.transaction
        or source.count_rows() != current.row_count
        or len(old_identities) != current.fragment_count
        or _schema_fingerprint(source) != current.schema_fingerprint
    ):
        raise ValueError("ready snapshot does not match its native branch version")
    _validate_new_fragment_files(
        source,
        fragments,
        spec.render.samples_per_shard,
        storage_options=storage_options,
    )
    identity = _pending_identity(current, pending, fragments, welford)
    latest = _open_train(train_uri).checkout_version((current.branch, None))
    if latest.version == current.version:
        published = _commit_append(source, fragments, identity)
    else:
        transaction = _transaction(latest, latest.version)
        properties = transaction.transaction_properties or {}
        if (
            latest.version != current.version + 1
            or properties.get(_GROWING_PENDING_PROPERTY) != identity
        ):
            raise ValueError(
                f"growing branch advanced unexpectedly from {current.version} to "
                f"{latest.version}"
            )
        published = latest

    identities = tuple(
        _fragment_storage_identity(fragment.metadata) for fragment in published.get_fragments()
    )
    if identities[: len(old_identities)] != old_identities:
        raise ValueError("published fragment storage identities do not preserve the old prefix")
    if len(identities) != len(old_identities) + expected_count:
        raise ValueError("published fragment count did not increase exactly")
    expected_rows = current.row_count + expected_count * spec.render.samples_per_shard
    if published.count_rows() != expected_rows:
        raise ValueError("published row count did not increase exactly")
    stats_sha256, welford_sha256 = _publish_cumulative_stats(
        spec, current, welford, metadata_root, published.version
    )
    transaction = _transaction(published, published.version)
    snapshot = GrowingSnapshot(
        branch=current.branch,
        branch_uri=current.branch_uri,
        version=published.version,
        baseline_version=current.baseline_version,
        baseline_transaction=current.baseline_transaction,
        transaction=transaction.uuid,
        baseline_train_shards=current.baseline_train_shards,
        max_train_shards=current.max_train_shards,
        num_extra_shards=current.num_extra_shards,
        high_watermark=pending.next_high_watermark,
        dataset_spec_fingerprint=current.dataset_spec_fingerprint,
        row_count=expected_rows,
        fragment_count=len(identities),
        schema_fingerprint=_schema_fingerprint(published),
        stats_sha256=stats_sha256,
        welford_sha256=welford_sha256,
    )
    _write_snapshot(metadata_root, snapshot)
    if publish_metadata is not None:
        publish_metadata(snapshot, _version_dir(metadata_root, published.version))
    retry_lance_read("growing_published_validate", published.validate)
    _set_ready_tag(_open_train(train_uri), current.branch, published.version)
    return snapshot


@contextmanager
def _materialize_lock(local_root: Path) -> Iterator[None]:
    local_root.mkdir(parents=True, exist_ok=True)
    with (local_root / ".materialize.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _read_active(path: Path) -> ActiveGrowingSnapshot | None:
    if not path.is_file():
        return None
    return ActiveGrowingSnapshot.model_validate_json(path.read_bytes())


def _local_transaction(dataset: lance.LanceDataset) -> lance.Transaction:
    return _transaction(dataset, dataset.version)


def _write_local_fragments(
    source: lance.LanceDataset,
    destination: Path,
    columns: Sequence[str],
    start_fragment: int,
) -> list[lance.fragment.FragmentMetadata]:
    fragments = source.get_fragments()[start_fragment:]
    written: list[lance.fragment.FragmentMetadata] = []
    for fragment in fragments:
        scanner = source.scanner(columns=list(columns), fragments=[fragment])
        created = lance.fragment.write_fragments(
            scanner.to_batches(),
            str(destination),
            schema=scanner.projected_schema,
            mode="append",
            data_storage_version=LANCE_DATA_STORAGE_VERSION,
            max_bytes_per_file=LANCE_MAX_BYTES_PER_FILE,
        )
        if len(created) != 1:
            raise ValueError("each remote fragment must map to one local fragment")
        written.append(created[0])
    return written


def _commit_local_initial(
    destination: Path,
    schema: pa.Schema,
    fragments: Sequence[lance.fragment.FragmentMetadata],
    identity: str,
) -> lance.LanceDataset:
    transaction = lance.Transaction(
        read_version=0,
        operation=lance.LanceOperation.Overwrite(schema, list(fragments)),
        transaction_properties={_LOCAL_SOURCE_PROPERTY: identity},
    )
    return lance.LanceDataset.commit(str(destination), transaction)


def _commit_local_append(
    dataset: lance.LanceDataset,
    fragments: Sequence[lance.fragment.FragmentMetadata],
    identity: str,
) -> lance.LanceDataset:
    transaction = lance.Transaction(
        read_version=dataset.version,
        operation=lance.LanceOperation.Append(list(fragments)),
        transaction_properties={_LOCAL_SOURCE_PROPERTY: identity},
    )
    return lance.LanceDataset.commit(dataset, transaction)


def _copy_version_metadata(
    snapshot: GrowingSnapshot, metadata_root: Path, local_root: Path
) -> Path:
    source = _version_dir(metadata_root, snapshot.version)
    destination = _version_dir(local_root, snapshot.version)
    destination.mkdir(parents=True, exist_ok=True)
    for name, digest in (
        ("stats.npz", snapshot.stats_sha256),
        ("welford.npz", snapshot.welford_sha256),
    ):
        source_path = source / name
        if _file_sha256(source_path) != digest:
            raise ValueError(f"{name} digest does not match remote snapshot")
        shutil.copyfile(source_path, destination / name)
    (destination / "snapshot.json").write_text(
        snapshot.model_dump_json(indent=2), encoding="utf-8"
    )
    return destination


def _activate(
    snapshot: GrowingSnapshot,
    local: lance.LanceDataset,
    dataset_path: Path,
    version_stats_path: Path,
    active_path: Path,
) -> ActiveGrowingSnapshot:
    transaction = _local_transaction(local)
    active = ActiveGrowingSnapshot(
        branch=snapshot.branch,
        remote_version=snapshot.version,
        remote_transaction=snapshot.transaction,
        local_version=local.version,
        local_transaction=transaction.uuid,
        dataset_path=str(dataset_path),
        version_stats_path=str(version_stats_path),
        dataset_spec_fingerprint=snapshot.dataset_spec_fingerprint,
        row_count=snapshot.row_count,
        fragment_count=snapshot.fragment_count,
        schema_fingerprint=_schema_fingerprint(local),
        stats_sha256=snapshot.stats_sha256,
        welford_sha256=snapshot.welford_sha256,
        high_watermark=snapshot.high_watermark,
    )
    staging = active_path.with_suffix(".json.tmp")
    staging.write_text(active.model_dump_json(indent=2), encoding="utf-8")
    staging.replace(active_path)
    return active


def materialize_and_activate(
    train_uri: Path | str,
    *,
    snapshot: GrowingSnapshot,
    metadata_root: Path,
    local_root: Path,
    columns: Sequence[str],
) -> ActiveGrowingSnapshot:
    """Increment one shared local dataset and atomically activate exact identity.

    :param train_uri: Baseline train dataset URI.
    :param snapshot: Exact remote snapshot to hydrate.
    :param metadata_root: Downloaded metadata workspace.
    :param local_root: Shared local growing root.
    :param columns: Projected dataset columns.
    :returns: Activated remote-to-local identity.
    :raises ValueError: Remote, local, or metadata identity validation fails.
    """
    with _materialize_lock(local_root):
        active_path = local_root / "active.json"
        active = _read_active(active_path)
        if active is not None:
            if (
                active.branch != snapshot.branch
                or active.dataset_spec_fingerprint != snapshot.dataset_spec_fingerprint
            ):
                raise ValueError("local growing root belongs to an incompatible branch or spec")
            if active.remote_version > snapshot.version:
                return active
            if active.remote_version == snapshot.version:
                if active.remote_transaction != snapshot.transaction:
                    raise ValueError("same remote version has a different transaction")
                return active

        source = _open_train(train_uri).checkout_version((snapshot.branch, snapshot.version))
        transaction_source = (
            _open_train(train_uri)
            if snapshot.version == snapshot.baseline_version
            else source
        )
        if _transaction(transaction_source, source.version).uuid != snapshot.transaction:
            raise ValueError("remote transaction does not match snapshot")
        if source.count_rows() != snapshot.row_count:
            raise ValueError("remote row count does not match snapshot")
        if len(source.get_fragments()) != snapshot.fragment_count:
            raise ValueError("remote fragment count does not match snapshot")
        if _schema_fingerprint(source) != snapshot.schema_fingerprint:
            raise ValueError("remote schema does not match snapshot")

        dataset_path = local_root / "train.lance"
        start_fragment = 0 if active is None else active.fragment_count
        identity = f"{snapshot.branch}:{snapshot.version}:{snapshot.transaction}"
        projected_schema = source.scanner(columns=list(columns)).projected_schema
        if active is None and dataset_path.exists():
            try:
                local = lance.dataset(str(dataset_path))
            except (OSError, ValueError) as exc:
                raise ValueError("unrecorded local Lance dataset is not readable") from exc
            properties = _local_transaction(local).transaction_properties or {}
            if properties.get(_LOCAL_SOURCE_PROPERTY) != identity:
                raise ValueError("unrecorded local Lance dataset belongs to a different remote snapshot")
        elif active is None:
            new_fragments = _write_local_fragments(source, dataset_path, columns, 0)
            local = _commit_local_initial(
                dataset_path, projected_schema, new_fragments, identity
            )
        else:
            local = lance.dataset(str(dataset_path))
            if local.version != active.local_version:
                latest_transaction = _local_transaction(local)
                properties = latest_transaction.transaction_properties or {}
                if properties.get(_LOCAL_SOURCE_PROPERTY) != identity:
                    raise ValueError("local Lance dataset advanced outside materialization")
            else:
                new_fragments = _write_local_fragments(
                    source, dataset_path, columns, start_fragment
                )
                local = _commit_local_append(local, new_fragments, identity)
        if local.count_rows() != snapshot.row_count:
            raise ValueError("local row count does not match remote snapshot")
        if len(local.get_fragments()) != snapshot.fragment_count:
            raise ValueError("local fragment count does not match remote snapshot")
        if not local.schema.equals(projected_schema, check_metadata=True):
            raise ValueError("local schema does not match projected remote schema")
        version_stats_path = _copy_version_metadata(snapshot, metadata_root, local_root)
        return _activate(
            snapshot, local, dataset_path, version_stats_path, active_path
        )


def generate_pending_shards(
    spec: DatasetSpec,
    snapshot: GrowingSnapshot,
    pending: PendingRefreshRequest,
    *,
    work_dir: Path,
) -> int:
    """Drain one exact bounded range through its branch-isolated claim queue.

    :param spec: Frozen producer specification.
    :param snapshot: Exact ready source snapshot.
    :param pending: Bound request to generate.
    :param work_dir: Local render workspace.
    :returns: Number of claims rendered.
    :raises ValueError: Request or claimed position is unsafe.
    """
    from synth_setter.cli.generate_dataset import render_and_upload_shard
    from synth_setter.pipeline.shard_claims import ShardClaims

    expected = pending_refresh_request(snapshot)
    if expected is None:
        return 0
    if pending != expected:
        raise ValueError("pending refresh does not match the ready source snapshot")
    plan = _plan(snapshot)
    shards = {
        shard_id: plan.extra_shard(spec, shard_id)
        for shard_id in pending.enqueue_shard_ids
    }
    claims = ShardClaims.for_run(
        *r2_io.lance_target(spec.r2.growing_shard_claims_uri(snapshot.branch))
    )
    claims.populate(shards)
    work_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    while (claim := claims.claim()) is not None:
        shard = shards.get(claim.shard_id)
        if shard is None or shard.shard_id >= snapshot.max_train_shards:
            raise ValueError(f"growing queue returned unsafe shard {claim.shard_id}")
        render_and_upload_shard(
            spec,
            shard,
            work_dir,
            loggers=[],
            target_lance_uri=snapshot.branch_uri,
            attempt_staging_dir_uri=spec.r2.growing_shard_staging_dir_uri(
                snapshot.branch, shard.shard_id
            ),
        )
        claims.complete(claim)
        rendered += 1
    return rendered


def finalize_staged_refresh(
    train_uri: Path | str,
    *,
    spec: DatasetSpec,
    current: GrowingSnapshot,
    pending: PendingRefreshRequest,
    metadata_root: Path,
    publish_metadata: Callable[[GrowingSnapshot, Path], None] | None = None,
) -> GrowingSnapshot:
    """Select exactly one winner per pending direct train position and append it.

    :param train_uri: Baseline train dataset URI.
    :param spec: Frozen producer specification.
    :param current: Exact ready source snapshot.
    :param pending: Bound request to finalize.
    :param metadata_root: Local metadata workspace.
    :param publish_metadata: Optional durable metadata publisher.
    :returns: Newly ready snapshot, or ``current`` at capacity.
    :raises ValueError: Request or staged winners are incomplete or invalid.
    """
    from synth_setter.pipeline.data.lance_finalize import (
        _load_fragment_metadata,
        _load_welford_state,
        select_winner,
        staged_complete_attempts,
    )

    expected = pending_refresh_request(current)
    if expected is None:
        return current
    if pending != expected:
        raise ValueError("pending refresh does not match the ready source snapshot")
    attempts = staged_complete_attempts(
        spec, root_uri=spec.r2.growing_workers_shards_root_uri(current.branch)
    )
    fragments: list[lance.fragment.FragmentMetadata] = []
    states: list[WelfordState] = []
    for shard_id in pending.enqueue_shard_ids:
        candidates = attempts.get(shard_id)
        if not candidates:
            raise ValueError(f"growing shard-{shard_id:06d} has no staged-valid attempt")
        winner = select_winner(candidates)
        staging_dir = spec.r2.growing_shard_staging_dir_uri(current.branch, shard_id)
        fragments.append(
            _load_fragment_metadata(spec, winner, staging_dir_uri=staging_dir)
        )
        states.append(_load_welford_state(spec, winner, staging_dir_uri=staging_dir))
    return publish_growing_branch(
        train_uri,
        spec=spec,
        current=current,
        fragments=fragments,
        welford=states,
        metadata_root=metadata_root,
        publish_metadata=publish_metadata,
    )
