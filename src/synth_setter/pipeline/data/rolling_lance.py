"""Native Lance rolling-window identity and publication primitives."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import lance
import numpy as np
from pydantic import BaseModel, ConfigDict

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_materialize import materialize_lance_subset
from synth_setter.pipeline.data.lance_shard import commit_lance_branch, fragment_schema_matches
from synth_setter.pipeline.data.stats import WelfordState, merge_welford
from synth_setter.pipeline.data.stats import finalize as finalize_welford

if TYPE_CHECKING:
    from synth_setter.pipeline.data.lance_finalize import StagedLanceAttempt
    from synth_setter.pipeline.schemas.spec import DatasetSpec, ShardSpec

_ROLLING_PENDING_PROPERTY = "synth_setter.rolling_pending_identity"


class ActiveRollingSnapshot(BaseModel):
    """Strict local pointer to one fully materialized branch version.

    .. attribute :: model_config

        Pydantic model configuration.

    .. attribute :: branch

        Native Lance branch name.

    .. attribute :: version

        Activated branch version.

    .. attribute :: transaction

        Source transaction UUID.

    .. attribute :: dataset_path

        Immutable local train dataset path.

    .. attribute :: dataset_spec_fingerprint

        Frozen producer-spec digest.

    .. attribute :: row_count

        Expected local row count.

    .. attribute :: schema_fingerprint

        Source Arrow schema digest.

    .. attribute :: stats_sha256

        Activated statistics digest.

    .. attribute :: high_watermark

        Exclusive generated train-shard position.

    .. attribute :: membership_relative_ids

        Ordered logical shards in the rolling window.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    branch: str
    version: int
    transaction: str
    dataset_path: str
    dataset_spec_fingerprint: str
    row_count: int
    schema_fingerprint: str
    stats_sha256: str
    high_watermark: int
    membership_relative_ids: tuple[int, ...]


class RollingSnapshot(BaseModel):
    """Strict identity of one ready native Lance branch version.

    .. attribute :: model_config

        Pydantic model configuration.

    .. attribute :: branch

        Native Lance branch name.

    .. attribute :: branch_uri

        Lance-managed branch URI.

    .. attribute :: version

        Ready branch version.

    .. attribute :: baseline_version

        Pinned main-branch version.

    .. attribute :: baseline_transaction

        Pinned main-branch transaction UUID.

    .. attribute :: transaction

        Ready branch transaction UUID.

    .. attribute :: window_size

        Baseline-derived train-shard count.

    .. attribute :: num_extra_shards

        Shards generated per refresh.

    .. attribute :: high_watermark

        Exclusive generated train-shard position.

    .. attribute :: membership_relative_ids

        Ordered logical shards in this version.

    .. attribute :: dataset_spec_fingerprint

        Frozen producer-spec digest.

    .. attribute :: row_count

        Expected rows in this version.

    .. attribute :: schema_fingerprint

        Arrow schema digest.

    .. attribute :: stats_sha256

        Version-bound statistics digest.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    branch: str
    branch_uri: str
    version: int
    baseline_version: int
    baseline_transaction: str
    transaction: str
    window_size: int
    num_extra_shards: int
    high_watermark: int
    membership_relative_ids: tuple[int, ...]
    dataset_spec_fingerprint: str
    row_count: int
    schema_fingerprint: str
    stats_sha256: str


class PendingRefreshRequest(BaseModel):
    """Bind one durable refresh request to its ready source snapshot.

    .. attribute :: model_config

        Pydantic model configuration.

    .. attribute :: branch

        Native Lance branch name.

    .. attribute :: source_version

        Ready source branch version.

    .. attribute :: source_transaction

        Ready source transaction UUID.

    .. attribute :: dataset_spec_fingerprint

        Frozen producer-spec digest.

    .. attribute :: enqueue_relative_ids

        Logical shards added by this request.

    .. attribute :: membership_relative_ids

        Ordered logical shards in the resulting window.

    .. attribute :: next_high_watermark

        Exclusive generated position after publication.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    branch: str
    source_version: int
    source_transaction: str
    dataset_spec_fingerprint: str
    enqueue_relative_ids: tuple[int, ...]
    membership_relative_ids: tuple[int, ...]
    next_high_watermark: int


@dataclass(frozen=True)
class PendingRefresh:
    """Relative shard identities selected for one fixed-size refresh.

    .. attribute :: enqueue_relative_ids

        Logical shards added by this refresh.

    .. attribute :: membership_relative_ids

        Ordered logical shards in the resulting window.

    .. attribute :: next_high_watermark

        Exclusive generated position after publication.
    """

    enqueue_relative_ids: tuple[int, ...]
    membership_relative_ids: tuple[int, ...]
    next_high_watermark: int


def pending_refresh_request(current: RollingSnapshot) -> PendingRefreshRequest:
    """Freeze the next range against one exact ready snapshot.

    :param current: Ready snapshot observed by the enqueue operator.
    :returns: Strict request consumed by generators and the finalizer.
    """
    refresh = RollingWindow(
        current.window_size,
        current.num_extra_shards,
        current.high_watermark,
    ).next_refresh()
    return PendingRefreshRequest(
        branch=current.branch,
        source_version=current.version,
        source_transaction=current.transaction,
        dataset_spec_fingerprint=current.dataset_spec_fingerprint,
        enqueue_relative_ids=refresh.enqueue_relative_ids,
        membership_relative_ids=refresh.membership_relative_ids,
        next_high_watermark=refresh.next_high_watermark,
    )


def _lance_target(uri: Path | str) -> tuple[str, dict[str, str] | None]:
    value = str(uri)
    if r2_io.is_r2_uri(value):
        return r2_io.lance_target(value)
    storage_options = r2_io.r2_storage_options() if value.startswith("s3://") else None
    return value, storage_options


def _open_train(train_uri: Path | str) -> lance.LanceDataset:
    target, storage_options = _lance_target(train_uri)
    return lance.dataset(target, storage_options=storage_options)


def dataset_spec_fingerprint(spec: DatasetSpec) -> str:
    """Return the stable fingerprint bound into rolling records.

    :param spec: Frozen dataset specification.
    :returns: SHA-256 hexadecimal digest of its canonical JSON.
    """
    return sha256(spec.model_dump_json().encode()).hexdigest()


def _schema_fingerprint(dataset: lance.LanceDataset) -> str:
    return sha256(dataset.schema.serialize().to_pybytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_snapshot(metadata_root: Path, snapshot: RollingSnapshot) -> None:
    version_dir = metadata_root / "versions" / str(snapshot.version)
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
    if branch_info.get("parent_version") != baseline_version:
        raise ValueError(
            f"branch {branch!r} already exists from baseline "
            f"{branch_info.get('parent_version')}, not {baseline_version}"
        )
    if branch_info.get("metadata") != contract:
        raise ValueError(f"branch {branch!r} already exists with a different rolling contract")
    checked_out = dataset.checkout_version((branch, None))
    if checked_out.version != baseline_version:
        raise ValueError(
            f"branch {branch!r} already advanced to version {checked_out.version}"
        )
    return checked_out


def initialize_rolling_branch(
    train_uri: Path | str,
    *,
    spec: DatasetSpec,
    branch: str,
    baseline_version: int,
    metadata_root: Path,
    num_extra_shards: int,
    publish_metadata: Callable[[RollingSnapshot, Path], None] | None = None,
) -> RollingSnapshot:
    """Create and publish a branch pinned to one finalized train version.

    :param train_uri: Finalized baseline ``train.lance`` URI.
    :param spec: Frozen baseline dataset specification.
    :param branch: Native Lance branch name.
    :param baseline_version: Explicit main-branch version to pin.
    :param metadata_root: Version-bound snapshot metadata directory.
    :param num_extra_shards: Shards generated by each refresh.
    :param publish_metadata: Optional durable metadata publisher called before readiness.
    :returns: Ready baseline branch snapshot.
    :raises ValueError: The pinned version has no transaction or refresh size is invalid.
    """
    dataset = _open_train(train_uri)
    transaction = dataset.read_transaction(baseline_version)
    if transaction is None:
        raise ValueError(f"baseline version {baseline_version} has no transaction")
    window = RollingWindow.from_spec(spec, num_extra_shards=num_extra_shards)
    baseline = dataset.checkout_version(baseline_version)
    expected_rows = spec.train_val_test_sizes[0]
    baseline_rows = baseline.count_rows()
    if baseline_rows != expected_rows:
        raise ValueError(f"baseline train rows {baseline_rows} do not match spec {expected_rows}")
    baseline_fragment_count = len(baseline.get_fragments())
    if baseline_fragment_count != window.size:
        raise ValueError(
            f"baseline train fragments {baseline_fragment_count} do not match window {window.size}"
        )
    stats_path = metadata_root / "versions" / str(baseline_version) / "stats.npz"
    if not stats_path.is_file():
        raise ValueError("baseline statistics are required before branch initialization")
    spec_fingerprint = dataset_spec_fingerprint(spec)
    contract = {
        "synth_setter.rolling_baseline_transaction": transaction.uuid,
        "synth_setter.rolling_baseline_version": str(baseline_version),
        "synth_setter.rolling_dataset_spec_fingerprint": spec_fingerprint,
        "synth_setter.rolling_num_extra_shards": str(num_extra_shards),
        "synth_setter.rolling_window_size": str(window.size),
    }
    checked_out = _create_or_resume_branch(
        dataset, branch, baseline_version, contract
    )
    snapshot = RollingSnapshot(
        branch=branch,
        branch_uri=str(checked_out.uri),
        version=checked_out.version,
        baseline_version=baseline_version,
        baseline_transaction=transaction.uuid,
        transaction=transaction.uuid,
        window_size=window.size,
        num_extra_shards=num_extra_shards,
        high_watermark=window.high_watermark,
        membership_relative_ids=tuple(range(window.size)),
        dataset_spec_fingerprint=spec_fingerprint,
        row_count=baseline.count_rows(),
        schema_fingerprint=_schema_fingerprint(baseline),
        stats_sha256=_file_sha256(stats_path),
    )
    _write_snapshot(metadata_root, snapshot)
    if publish_metadata is not None:
        publish_metadata(snapshot, metadata_root / "versions" / str(snapshot.version))
    ready_tag = f"{branch}-ready"
    try:
        ready_version = dataset.tags.get_version(ready_tag)
    except ValueError:
        ready_version = None
    if ready_version is None:
        dataset.tags.create(ready_tag, (branch, checked_out.version))
    elif ready_version != checked_out.version:
        raise ValueError(
            f"ready tag {ready_tag!r} points to version {ready_version}, "
            f"not {checked_out.version}"
        )
    return snapshot


def materialize_and_activate(
    train_uri: Path | str,
    *,
    snapshot: RollingSnapshot,
    metadata_root: Path,
    local_root: Path,
    columns: Sequence[str],
) -> ActiveRollingSnapshot:
    """Materialize an exact ready version and atomically activate its record.

    :param train_uri: Baseline ``train.lance`` URI.
    :param snapshot: Ready branch/version identity to materialize.
    :param metadata_root: Version-bound source metadata directory.
    :param local_root: Local immutable snapshot and active-record root.
    :param columns: Dataset columns required by the training reader.
    :returns: Atomically activated local snapshot identity.
    :raises ValueError: Materialized rows do not match the fixed window.
    """
    source = _open_train(train_uri).checkout_version(
        (snapshot.branch, snapshot.version)
    )
    version_root = local_root / "versions" / str(snapshot.version)
    dataset_path = version_root / "train.lance"
    materialize_lance_subset(
        snapshot.branch_uri,
        dataset_path,
        txid=None,
        columns=columns,
        version=snapshot.version,
        branch=snapshot.branch,
        source_base_uri=str(train_uri),
    )
    materialized = lance.dataset(str(dataset_path))
    expected_rows = source.count_rows()
    if expected_rows != snapshot.row_count:
        raise ValueError("source row count does not match snapshot metadata")
    if _schema_fingerprint(source) != snapshot.schema_fingerprint:
        raise ValueError("source schema does not match snapshot metadata")
    source_fragment_count = len(source.get_fragments())
    if source_fragment_count != snapshot.window_size:
        raise ValueError(
            f"source version {snapshot.version} has {source_fragment_count} fragments; "
            f"expected {snapshot.window_size}"
        )
    if materialized.count_rows() != expected_rows:
        raise ValueError(
            f"materialized version {snapshot.version} has {materialized.count_rows()} rows; "
            f"expected {expected_rows}"
        )
    if tuple(materialized.schema.names) != tuple(columns):
        raise ValueError("materialized schema does not match requested columns")
    materialized.scanner(limit=1).to_table()
    stats_source = metadata_root / "versions" / str(snapshot.version) / "stats.npz"
    if _file_sha256(stats_source) != snapshot.stats_sha256:
        raise ValueError("statistics do not match snapshot metadata")
    version_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(stats_source, version_root / "stats.npz")
    active = ActiveRollingSnapshot(
        branch=snapshot.branch,
        version=snapshot.version,
        transaction=snapshot.transaction,
        dataset_path=str(dataset_path),
        dataset_spec_fingerprint=snapshot.dataset_spec_fingerprint,
        row_count=snapshot.row_count,
        schema_fingerprint=snapshot.schema_fingerprint,
        stats_sha256=snapshot.stats_sha256,
        high_watermark=snapshot.high_watermark,
        membership_relative_ids=snapshot.membership_relative_ids,
    )
    local_root.mkdir(parents=True, exist_ok=True)
    destination = local_root / "active.json"
    staging = destination.with_suffix(".json.tmp")
    staging.write_text(active.model_dump_json(indent=2), encoding="utf-8")
    staging.replace(destination)
    return active


def _validate_new_fragment_files(
    dataset: lance.LanceDataset,
    fragments: Sequence[lance.fragment.FragmentMetadata],
    expected_rows: int,
    *,
    storage_options: dict[str, str] | None,
) -> None:
    from lance.file import LanceFileReader

    existing_paths = {
        data_file.path
        for fragment in dataset.get_fragments()
        for data_file in fragment.metadata.files
    }
    for fragment in fragments:
        if len(fragment.files) != 1 or fragment.physical_rows != expected_rows:
            raise ValueError(
                f"each fragment must contain one data file and {expected_rows} rows"
            )
        data_file = fragment.files[0]
        data_path = PurePosixPath(data_file.path)
        if (
            not data_file.path
            or data_path.is_absolute()
            or ".." in data_path.parts
            or data_path.as_posix() != data_file.path
        ):
            raise ValueError(f"unsafe new fragment data path {data_file.path!r}")
        if data_file.path in existing_paths:
            continue
        try:
            reader = LanceFileReader(
                f"{dataset.uri}/data/{data_file.path}", storage_options=storage_options
            )
            physical_schema = reader.metadata().schema
            physical_rows = reader.num_rows()
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"new fragment data file {data_file.path!r} is missing or unreadable"
            ) from exc
        if physical_rows != fragment.physical_rows:
            raise ValueError(
                f"new fragment data file {data_file.path!r} has {physical_rows} rows; "
                f"metadata declares {fragment.physical_rows}"
            )
        if not fragment_schema_matches(physical_schema, dataset.schema):
            raise ValueError(
                f"new fragment data file {data_file.path!r} schema does not match branch"
            )


def publish_rolling_branch(
    train_uri: Path | str,
    *,
    spec: DatasetSpec,
    current: RollingSnapshot,
    fragments: Sequence[lance.fragment.FragmentMetadata],
    welford: Sequence[WelfordState],
    metadata_root: Path,
    publish_metadata: Callable[[RollingSnapshot, Path], None] | None = None,
) -> RollingSnapshot:
    """Publish one fixed-size branch overwrite and advance readiness last.

    :param train_uri: Baseline ``train.lance`` URI.
    :param spec: Frozen baseline dataset specification.
    :param current: Currently ready snapshot.
    :param fragments: Ordered fragment membership for the next window.
    :param welford: Per-fragment statistics in the same order.
    :param metadata_root: Version-bound metadata directory.
    :param publish_metadata: Optional durable metadata publisher called before readiness.
    :returns: Newly ready branch snapshot.
    :raises ValueError: Membership, statistics, or baseline identity is incompatible.
    """
    if dataset_spec_fingerprint(spec) != current.dataset_spec_fingerprint:
        raise ValueError("dataset spec fingerprint does not match rolling baseline")
    if len(fragments) != current.window_size or len(welford) != current.window_size:
        raise ValueError(f"publication requires exactly {current.window_size} fragments and stats")
    if any(state[0] != spec.render.samples_per_shard for state in welford):
        raise ValueError(
            f"each publication statistic must cover {spec.render.samples_per_shard} rows"
        )
    window = RollingWindow(
        size=current.window_size,
        num_extra_shards=current.num_extra_shards,
        high_watermark=current.high_watermark,
    )
    pending = window.next_refresh()
    branch_target, storage_options = _lance_target(current.branch_uri)
    dataset = lance.dataset(branch_target, storage_options=storage_options).checkout_version(
        current.version
    )
    _validate_new_fragment_files(
        dataset,
        fragments,
        spec.render.samples_per_shard,
        storage_options=storage_options,
    )
    identity = sha256()
    identity.update(current.transaction.encode())
    identity.update(repr(pending.membership_relative_ids).encode())
    identity.update(current.dataset_spec_fingerprint.encode())
    for fragment, shard_state in zip(fragments, welford, strict=True):
        identity.update(json.dumps(fragment.to_json(), sort_keys=True).encode())
        identity.update(str(shard_state[0]).encode())
        for value in shard_state[1:]:
            array = np.asarray(value)
            identity.update(array.dtype.str.encode())
            identity.update(repr(array.shape).encode())
            identity.update(array.tobytes())
    pending_identity = identity.hexdigest()
    latest = _open_train(train_uri).checkout_version((current.branch, None))
    if latest.version == current.version:
        published = commit_lance_branch(
            dataset,
            dataset.schema,
            fragments,
            transaction_properties={_ROLLING_PENDING_PROPERTY: pending_identity},
        )
    else:
        transaction = latest.read_transaction(latest.version)
        properties = (
            transaction.transaction_properties or {} if transaction is not None else {}
        )
        if (
            latest.version != current.version + 1
            or properties.get(_ROLLING_PENDING_PROPERTY) != pending_identity
        ):
            raise ValueError(
                f"rolling branch advanced unexpectedly from version {current.version} "
                f"to {latest.version}"
            )
        published = latest
    state: WelfordState = (0, 0, 0)
    for shard_state in welford:
        state = merge_welford(state, shard_state)
    mean, std = finalize_welford(state, mask_degenerate=spec.mask_degenerate_bins)
    version_dir = metadata_root / "versions" / str(published.version)
    version_dir.mkdir(parents=True, exist_ok=True)
    np.savez(version_dir / "stats.npz", mean=mean, std=std)
    transaction = published.read_transaction(published.version)
    if transaction is None:
        raise ValueError(f"published branch version {published.version} has no transaction")
    snapshot = RollingSnapshot(
        branch=current.branch,
        branch_uri=current.branch_uri,
        version=published.version,
        baseline_version=current.baseline_version,
        baseline_transaction=current.baseline_transaction,
        transaction=transaction.uuid,
        window_size=current.window_size,
        num_extra_shards=current.num_extra_shards,
        high_watermark=pending.next_high_watermark,
        membership_relative_ids=pending.membership_relative_ids,
        dataset_spec_fingerprint=current.dataset_spec_fingerprint,
        row_count=published.count_rows(),
        schema_fingerprint=_schema_fingerprint(published),
        stats_sha256=_file_sha256(version_dir / "stats.npz"),
    )
    _write_snapshot(metadata_root, snapshot)
    if publish_metadata is not None:
        publish_metadata(snapshot, version_dir)
    published.scanner(limit=1).to_table()
    _open_train(train_uri).tags.update(
        f"{current.branch}-ready", (current.branch, published.version)
    )
    return snapshot


@dataclass(frozen=True)
class RollingWindow:
    """Fixed baseline-derived rolling shard window.

    .. attribute :: size

        Baseline train-shard count retained in every version.

    .. attribute :: num_extra_shards

        Shards replaced by each refresh.

    .. attribute :: high_watermark

        Exclusive generated train-shard position.
    """

    size: int
    num_extra_shards: int
    high_watermark: int

    @classmethod
    def from_spec(cls, spec: DatasetSpec, *, num_extra_shards: int) -> RollingWindow:
        """Build a rolling window from the frozen baseline train size.

        :param spec: Baseline dataset specification.
        :param num_extra_shards: Shards generated by each refresh.
        :returns: Window beginning after the baseline shard range.
        :raises ValueError: ``num_extra_shards`` falls outside ``[1, size]``.
        """
        size = spec.train_val_test_sizes[0] // spec.render.samples_per_shard
        if not 1 <= num_extra_shards <= size:
            raise ValueError(
                f"num_extra_shards must be between 1 and baseline window size {size}"
            )
        return cls(
            size=size,
            num_extra_shards=num_extra_shards,
            high_watermark=size,
        )

    def advance(self, refresh: PendingRefresh) -> RollingWindow:
        """Return the window state after a successful publication.

        :param refresh: Refresh whose metadata is durably ready.
        :returns: Window advanced to ``refresh.next_high_watermark``.
        """
        return RollingWindow(
            size=self.size,
            num_extra_shards=self.num_extra_shards,
            high_watermark=refresh.next_high_watermark,
        )

    def extra_shard(self, spec: DatasetSpec, relative_id: int) -> ShardSpec:
        """Map a relative extra ID beyond every frozen baseline identity.

        :param spec: Baseline dataset specification.
        :param relative_id: Non-negative index in the extra train stream.
        :returns: Deterministic shard identity and globally disjoint sample offset.
        :raises ValueError: ``relative_id`` is negative.
        """
        from synth_setter.pipeline.schemas.spec import ShardSpec

        if relative_id < self.size:
            raise ValueError(
                f"relative extra shard ID must be at least baseline size {self.size}"
            )
        extra_index = relative_id - self.size
        shard_id = spec.num_shards + extra_index
        sample_offset = sum(spec.train_val_test_sizes) + (
            extra_index * spec.render.samples_per_shard
        )
        return ShardSpec(
            shard_id=shard_id,
            filename=f"shard-{shard_id:06d}{spec.output_format.extension}",
            seed=spec.base_seed,
            sample_offset=sample_offset,
        )

    def next_refresh(self) -> PendingRefresh:
        """Select the next enqueue range and fixed-size membership.

        :returns: Relative IDs and next high-watermark for one refresh.
        """
        next_high_watermark = self.high_watermark + self.num_extra_shards
        return PendingRefresh(
            enqueue_relative_ids=tuple(range(self.high_watermark, next_high_watermark)),
            membership_relative_ids=tuple(
                range(next_high_watermark - self.size, next_high_watermark)
            ),
            next_high_watermark=next_high_watermark,
        )


def _physical_shard_id(spec: DatasetSpec, window_size: int, relative_id: int) -> int:
    if relative_id < window_size:
        return relative_id
    return spec.num_shards + (relative_id - window_size)


def generate_pending_shards(
    spec: DatasetSpec,
    snapshot: RollingSnapshot,
    pending: PendingRefreshRequest,
    *,
    work_dir: Path,
) -> int:
    """Render the pending relative range through the R2 Lance claim queue.

    :param spec: Frozen baseline dataset specification.
    :param snapshot: Ready rolling identity bound by ``pending``.
    :param pending: Durably frozen range to drain.
    :param work_dir: Local renderer scratch directory.
    :returns: Number of claims rendered by this process.
    :raises ValueError: The pending request does not match the ready source snapshot.
    """
    from synth_setter.cli.generate_dataset import render_and_upload_shard
    from synth_setter.pipeline.shard_claims import ShardClaims

    expected = pending_refresh_request(snapshot)
    if pending != expected:
        raise ValueError("pending refresh does not match the ready source snapshot")
    window = RollingWindow(
        snapshot.window_size, snapshot.num_extra_shards, snapshot.high_watermark
    )
    shards = {}
    for relative_id in pending.enqueue_relative_ids:
        shard = window.extra_shard(spec, relative_id)
        shards[shard.shard_id] = shard
    claims = ShardClaims.for_run(
        *r2_io.lance_target(spec.r2.rolling_shard_claims_uri(snapshot.branch))
    )
    claims.populate(shards)
    work_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    while (claim := claims.claim()) is not None:
        shard = shards.get(claim.shard_id)
        if shard is None:
            raise ValueError(f"rolling queue returned unexpected shard {claim.shard_id}")
        render_and_upload_shard(
            spec,
            shard,
            work_dir,
            loggers=[],
            target_lance_uri=snapshot.branch_uri,
        )
        claims.complete(claim)
        rendered += 1
    return rendered


def _fragment_storage_identity(
    fragment: lance.fragment.FragmentMetadata,
) -> tuple[tuple[str, ...], int]:
    """Identify immutable fragment content despite commit-enriched metadata.

    :param fragment: Worker or committed Lance fragment metadata.
    :returns: Exact underlying file paths and physical row count.
    """
    return tuple(data_file.path for data_file in fragment.files), fragment.physical_rows


def _retained_attempt_for_fragment(
    spec: DatasetSpec,
    candidates: Sequence[StagedLanceAttempt],
    fragment: lance.fragment.FragmentMetadata,
) -> StagedLanceAttempt:
    from synth_setter.pipeline.data.lance_finalize import _load_fragment_metadata

    expected = _fragment_storage_identity(fragment)
    candidate_fragments = [
        (attempt, _load_fragment_metadata(spec, attempt)) for attempt in candidates
    ]
    matches = [
        attempt
        for attempt, candidate in candidate_fragments
        if _fragment_storage_identity(candidate) == expected
    ]
    if len(matches) != 1:
        candidate_paths = [
            [data_file.path for data_file in candidate.files]
            for _, candidate in candidate_fragments
        ]
        expected_paths = [data_file.path for data_file in fragment.files]
        raise ValueError(
            "retained fragment must map to exactly one staged attempt; "
            f"found {len(matches)} for files {expected_paths}, candidates {candidate_paths}"
        )
    return matches[0]


def finalize_staged_refresh(
    train_uri: Path | str,
    *,
    spec: DatasetSpec,
    current: RollingSnapshot,
    pending: PendingRefreshRequest,
    metadata_root: Path,
    publish_metadata: Callable[[RollingSnapshot, Path], None] | None = None,
) -> RollingSnapshot:
    """Finalize one cutoff of staged attempts into the next branch version.

    :param train_uri: Finalized baseline ``train.lance`` URI.
    :param spec: Frozen baseline dataset specification.
    :param current: Ready branch snapshot bound by ``pending``.
    :param pending: Durably frozen refresh request.
    :param metadata_root: Local metadata staging directory.
    :param publish_metadata: Optional durable publisher invoked before the ready tag.
    :returns: Newly ready branch snapshot.
    :raises ValueError: Any selected member lacks a complete valid attempt.
    """
    from synth_setter.pipeline.data.lance_finalize import (
        _load_fragment_metadata,
        _load_welford_state,
        select_winner,
        staged_complete_attempts,
    )

    if pending != pending_refresh_request(current):
        raise ValueError("pending refresh does not match the ready source snapshot")
    membership = pending.membership_relative_ids
    cutoff = staged_complete_attempts(spec)
    current_dataset = _open_train(train_uri).checkout_version(
        (current.branch, current.version)
    )
    current_fragments = {
        relative_id: fragment.metadata
        for relative_id, fragment in zip(
            current.membership_relative_ids,
            current_dataset.get_fragments(),
            strict=True,
        )
    }
    fragments: list[lance.fragment.FragmentMetadata] = []
    states: list[WelfordState] = []
    for relative_id in membership:
        shard_id = _physical_shard_id(spec, current.window_size, relative_id)
        candidates = cutoff.get(shard_id)
        if not candidates:
            raise ValueError(f"shard-{shard_id:06d} has no staged-valid attempt at cutoff")
        retained = current_fragments.get(relative_id)
        attempt = (
            _retained_attempt_for_fragment(spec, candidates, retained)
            if retained is not None
            else select_winner(candidates)
        )
        states.append(_load_welford_state(spec, attempt))
        fragments.append(retained or _load_fragment_metadata(spec, attempt))
    return publish_rolling_branch(
        train_uri,
        spec=spec,
        current=current,
        fragments=fragments,
        welford=states,
        metadata_root=metadata_root,
        publish_metadata=publish_metadata,
    )
