"""Public operator commands for append-only Lance train growth."""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import time
from pathlib import Path

import lance
import structlog

from synth_setter.data.vst.shapes import DATASET_FIELD_NAMES
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.growing_lance import (
    ActiveGrowingSnapshot,
    GrowingSnapshot,
    PendingRefreshRequest,
    dataset_spec_fingerprint,
    finalize_staged_refresh,
    generate_pending_shards,
    initialize_growing_branch,
    materialize_and_activate,
    pending_refresh_request,
)
from synth_setter.pipeline.data.lance_finalize import staged_complete_attempts
from synth_setter.pipeline.data.lance_materialize import retry_lance_read
from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.pipeline.shard_claims import ShardClaims
from synth_setter.pipeline.spec_io import load_spec_from_uri

logger = structlog.get_logger(__name__)

_BRANCH_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _branch(value: str) -> str:
    """Validate a native Lance branch name argument.

    :param value: Raw ``--branch`` argument.
    :returns: The validated branch name.
    :raises argparse.ArgumentTypeError: The name is not branch-safe.
    """
    if not _BRANCH_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError("branch must match [A-Za-z0-9][A-Za-z0-9._-]*")
    return value


def _load_snapshot(uri: str) -> GrowingSnapshot:
    """Download and strictly parse one ready snapshot record.

    :param uri: R2 URI of a ``snapshot.json``.
    :returns: Validated snapshot.
    """
    with r2_io.downloaded_to_tempfile(uri) as path:
        return GrowingSnapshot.model_validate_json(path.read_bytes())


def _load_pending(uri: str) -> PendingRefreshRequest:
    """Download and strictly parse the durable pending request.

    :param uri: R2 URI of ``pending.json``.
    :returns: Validated pending request.
    """
    with r2_io.downloaded_to_tempfile(uri) as path:
        return PendingRefreshRequest.model_validate_json(path.read_bytes())


def _publish_metadata(spec: DatasetSpec, snapshot: GrowingSnapshot, version_dir: Path) -> None:
    """Upload a version's snapshot and statistics sidecars to R2.

    :param spec: Frozen producer specification.
    :param snapshot: Snapshot naming the destination version.
    :param version_dir: Local directory holding the sidecars.
    """
    for name in ("snapshot.json", "stats.npz", "welford.npz"):
        source = version_dir / name
        if source.is_file():
            r2_io.upload(
                source,
                spec.r2.growing_metadata_uri(
                    snapshot.branch, f"versions/{snapshot.version}/{name}"
                ),
            )


def _train_target(spec: DatasetSpec) -> tuple[str, dict[str, str] | None]:
    """Resolve the train split into a Lance-openable target.

    :param spec: Frozen producer specification.
    :returns: ``(uri, storage_options)`` for ``lance.dataset``.
    """
    return r2_io.lance_target(spec.r2.split_lance_uri("train"))


def _ready_version(spec: DatasetSpec, branch: str) -> int:
    """Read the branch's ready-tag version with one cheap remote read.

    :param spec: Frozen producer specification.
    :param branch: Native branch name.
    :returns: The tagged ready version.
    :raises ValueError: The ready tag does not exist.
    """
    target, storage_options = _train_target(spec)
    version = retry_lance_read(
        "growing_ready_tag_read",
        lambda: lance.dataset(target, storage_options=storage_options).tags.get_version(
            f"{branch}-ready"
        ),
    )
    if version is None:
        raise ValueError(f"ready tag for branch {branch!r} does not exist")
    return version


def _ready_snapshot(spec: DatasetSpec, branch: str, version: int | None = None) -> GrowingSnapshot:
    """Load and validate the ready snapshot the tag points at.

    :param spec: Frozen producer specification.
    :param branch: Native branch name.
    :param version: Ready version already read, or ``None`` to read the tag.
    :returns: Validated ready snapshot.
    :raises ValueError: The tag, snapshot, and spec identities disagree.
    """
    if version is None:
        version = _ready_version(spec, branch)
    snapshot = _load_snapshot(
        spec.r2.growing_metadata_uri(branch, f"versions/{version}/snapshot.json")
    )
    if snapshot.branch != branch or snapshot.version != version:
        raise ValueError("ready tag and growing snapshot identity disagree")
    if snapshot.dataset_spec_fingerprint != dataset_spec_fingerprint(spec):
        raise ValueError("ready snapshot dataset specification disagrees with input")
    return snapshot


def _latest_train_version(spec: DatasetSpec) -> int:
    """Return the finalized baseline train version.

    Generation is quiescent behind the completion marker init probes, so the dataset's current
    version IS the finalized baseline manifest.

    :param spec: Frozen producer specification.
    :returns: The train dataset's current main version.
    """
    target, storage_options = _train_target(spec)
    return retry_lance_read(
        "growing_baseline_version_read",
        lambda: lance.dataset(target, storage_options=storage_options).version,
    )


def initialize(args: argparse.Namespace) -> None:
    """Initialize a branch pinned to a complete finalized baseline.

    :param args: Parsed init arguments.
    """
    spec = load_spec_from_uri(args.spec_uri)
    r2_io.ensure_r2_env_loaded()
    with r2_io.downloaded_to_tempfile(spec.r2.dataset_complete_marker_uri()):
        pass
    baseline_version = args.baseline_version
    if baseline_version is None:
        baseline_version = _latest_train_version(spec)
    version_dir = args.work_dir / "versions" / str(baseline_version)
    version_dir.mkdir(parents=True, exist_ok=True)
    for name, uri in (("stats.npz", spec.r2.stats_uri()), ("welford.npz", spec.r2.welford_uri())):
        with r2_io.downloaded_to_tempfile(uri) as source:
            shutil.copyfile(source, version_dir / name)
    initialize_growing_branch(
        spec.r2.split_lance_uri("train"),
        spec=spec,
        branch=args.branch,
        baseline_version=baseline_version,
        metadata_root=args.work_dir,
        max_train_shards=args.max_train_shards,
        num_extra_shards=args.num_extra_shards,
        publish_metadata=lambda snapshot, root: _publish_metadata(spec, snapshot, root),
    )


def _pending_exists(spec: DatasetSpec, branch: str) -> bool:
    """Probe whether the branch has a durable ``pending.json``.

    :param spec: Frozen producer specification.
    :param branch: Native branch name.
    :returns: Whether the object exists.
    """
    return r2_io.object_size(spec.r2.growing_metadata_uri(branch, "pending.json")) is not None


def _download_version_sidecars(
    spec: DatasetSpec, branch: str, version: int, destination: Path
) -> None:
    """Download a version's statistics sidecars into a local directory.

    :param spec: Frozen producer specification.
    :param branch: Native branch name.
    :param version: Ready version whose sidecars are fetched.
    :param destination: Local version directory, created if missing.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("stats.npz", "welford.npz"):
        uri = spec.r2.growing_metadata_uri(branch, f"versions/{version}/{name}")
        with r2_io.downloaded_to_tempfile(uri) as source:
            shutil.copyfile(source, destination / name)


def _enqueue_pending(
    spec: DatasetSpec, branch: str, work_dir: Path, current: GrowingSnapshot
) -> PendingRefreshRequest | None:
    """Persist the next bounded range and seed branch-specific claims.

    :param spec: Frozen producer specification.
    :param branch: Native branch name.
    :param work_dir: Local operator workspace.
    :param current: Exact ready source snapshot.
    :returns: The durable pending request, or ``None`` at capacity.
    :raises ValueError: An incompatible pending request already exists.
    """
    pending = pending_refresh_request(current)
    if pending is None:
        return None
    pending_uri = spec.r2.growing_metadata_uri(branch, "pending.json")
    if _pending_exists(spec, branch):
        if _load_pending(pending_uri) != pending:
            raise ValueError("another pending growing refresh already exists")
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
        local = work_dir / "pending.json"
        local.write_text(pending.model_dump_json(indent=2), encoding="utf-8")
        r2_io.upload(local, pending_uri)
    claims = ShardClaims.for_run(*r2_io.lance_target(spec.r2.growing_shard_claims_uri(branch)))
    claims.populate(pending.enqueue_shard_ids)
    return pending


def enqueue(args: argparse.Namespace) -> None:
    """Persist the next bounded range and seed branch-specific claims.

    :param args: Parsed enqueue arguments.
    """
    spec = load_spec_from_uri(args.spec_uri)
    r2_io.ensure_r2_env_loaded()
    current = _ready_snapshot(spec, args.branch)
    _enqueue_pending(spec, args.branch, args.work_dir, current)


def generate(args: argparse.Namespace) -> None:
    """Drain pending claims without ever crossing the configured maximum.

    With ``--poll-seconds`` the worker keeps polling for the next enqueued
    range until the branch reaches capacity; ``0`` performs a single pass.
    A missing or superseded ``pending.json`` is a wait state, not an error —
    the driver may not have enqueued the next range yet.

    :param args: Parsed generate arguments.
    """
    spec = load_spec_from_uri(args.spec_uri)
    r2_io.ensure_r2_env_loaded()
    pending_uri = spec.r2.growing_metadata_uri(args.branch, "pending.json")
    while True:
        current = _ready_snapshot(spec, args.branch)
        expected = pending_refresh_request(current)
        if expected is None:
            logger.info("growing_branch_at_capacity", branch=args.branch)
            return
        pending = _load_pending(pending_uri) if _pending_exists(spec, args.branch) else None
        if pending is not None and pending == expected:
            generate_pending_shards(spec, current, pending, work_dir=args.work_dir)
        else:
            logger.info("growing_pending_not_ready", branch=args.branch)
        if args.poll_seconds <= 0:
            return
        time.sleep(args.poll_seconds)


def _complete_pending(
    spec: DatasetSpec,
    branch: str,
    pending_uri: str,
    pending: PendingRefreshRequest,
    version: int,
    work_dir: Path,
) -> None:
    """Record a published request durably, then delete the pending marker.

    :param spec: Frozen producer specification.
    :param branch: Native branch name.
    :param pending_uri: R2 URI of the pending marker to clear.
    :param pending: The completed request.
    :param version: Published ready version recording the completion.
    :param work_dir: Local operator workspace.
    """
    completed = work_dir / f"completed-{version}.json"
    completed.parent.mkdir(parents=True, exist_ok=True)
    completed.write_text(pending.model_dump_json(indent=2), encoding="utf-8")
    r2_io.upload(completed, spec.r2.growing_metadata_uri(branch, f"completed/{version}.json"))
    r2_io.delete_file(pending_uri)


def _complete_stale_pending(
    spec: DatasetSpec, branch: str, work_dir: Path, current: GrowingSnapshot
) -> None:
    """Complete a pending request whose range the ready snapshot already covers.

    :param spec: Frozen producer specification.
    :param branch: Native branch name.
    :param work_dir: Local operator workspace.
    :param current: Exact ready snapshot at capacity.
    :raises ValueError: The capacity snapshot disagrees with its stale request.
    """
    pending_uri = spec.r2.growing_metadata_uri(branch, "pending.json")
    if not _pending_exists(spec, branch):
        return
    completed = _load_pending(pending_uri)
    if current.high_watermark != completed.next_high_watermark:
        raise ValueError("capacity snapshot disagrees with pending refresh")
    _complete_pending(spec, branch, pending_uri, completed, current.version, work_dir)


def _finalize_pending(spec: DatasetSpec, branch: str, work_dir: Path) -> None:
    """Append one pending range and clear it only after readiness.

    :param spec: Frozen producer specification.
    :param branch: Native branch name.
    :param work_dir: Local operator workspace.
    """
    current = _ready_snapshot(spec, branch)
    if pending_refresh_request(current) is None:
        _complete_stale_pending(spec, branch, work_dir, current)
        return
    pending_uri = spec.r2.growing_metadata_uri(branch, "pending.json")
    _download_version_sidecars(
        spec, branch, current.version, work_dir / "versions" / str(current.version)
    )
    pending = _load_pending(pending_uri)
    if current.high_watermark == pending.next_high_watermark:
        _complete_pending(spec, branch, pending_uri, pending, current.version, work_dir)
        return
    published = finalize_staged_refresh(
        spec.r2.split_lance_uri("train"),
        spec=spec,
        current=current,
        pending=pending,
        metadata_root=work_dir,
        publish_metadata=lambda snapshot, root: _publish_metadata(spec, snapshot, root),
    )
    _complete_pending(spec, branch, pending_uri, pending, published.version, work_dir)


def finalize(args: argparse.Namespace) -> None:
    """Append one pending range and clear it only after readiness.

    :param args: Parsed finalize arguments.
    """
    spec = load_spec_from_uri(args.spec_uri)
    r2_io.ensure_r2_env_loaded()
    _finalize_pending(spec, args.branch, args.work_dir)


def _pending_fully_staged(spec: DatasetSpec, branch: str, pending: PendingRefreshRequest) -> bool:
    """Probe whether every pending position has a complete staged attempt.

    :param spec: Frozen producer specification.
    :param branch: Native branch name.
    :param pending: Durable pending request.
    :returns: Whether finalization can select a winner for every position.
    """
    attempts = staged_complete_attempts(
        spec, root_uri=spec.r2.growing_workers_shards_root_uri(branch)
    )
    return all(shard_id in attempts for shard_id in pending.enqueue_shard_ids)


def grow(args: argparse.Namespace) -> None:
    """Drive enqueue and finalize cycles until the branch reaches capacity.

    Each cycle persists the next bounded range, waits for generators to stage
    every position, and appends the staged range as the next ready version.
    ``--poll-seconds 0`` drains every already-staged range and returns instead
    of waiting.

    :param args: Parsed grow arguments.
    """
    spec = load_spec_from_uri(args.spec_uri)
    r2_io.ensure_r2_env_loaded()
    while True:
        current = _ready_snapshot(spec, args.branch)
        pending = _enqueue_pending(spec, args.branch, args.work_dir, current)
        if pending is None:
            _complete_stale_pending(spec, args.branch, args.work_dir, current)
            logger.info("growing_branch_at_capacity", branch=args.branch)
            return
        if _pending_fully_staged(spec, args.branch, pending):
            # Re-reads the durable ready snapshot so a concurrent operator's
            # publish is honored over this loop's cached view.
            _finalize_pending(spec, args.branch, args.work_dir)
            continue
        if args.poll_seconds <= 0:
            return
        time.sleep(args.poll_seconds)


def _active_remote_version(local_root: Path) -> int | None:
    """Read the activated remote version leniently for the polling fast path.

    :param local_root: Shared local growing root.
    :returns: The active record's remote version, or ``None`` when unreadable.
    """
    # Lenient on purpose: the fast path must never kill the daemon — a bad
    # record falls through to the full path, which validates under the lock.
    try:
        record = ActiveGrowingSnapshot.model_validate_json(
            (local_root / "active.json").read_bytes()
        )
    except (OSError, ValueError):
        return None
    return record.remote_version


def materialize_ready(args: argparse.Namespace) -> None:
    """Poll and increment one shared local train dataset under a file lock.

    A tick whose active record already matches the ready tag downloads no version metadata —
    polling tightly stays cheap.

    :param args: Parsed materialization arguments.
    """
    spec = load_spec_from_uri(args.spec_uri)
    r2_io.ensure_r2_env_loaded()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    while True:
        ready_version = _ready_version(spec, args.branch)
        if _active_remote_version(args.local_root) == ready_version:
            if args.poll_seconds <= 0:
                return
            time.sleep(args.poll_seconds)
            continue
        snapshot = _ready_snapshot(spec, args.branch, ready_version)
        with tempfile.TemporaryDirectory(dir=args.work_dir) as temporary:
            metadata_root = Path(temporary)
            _download_version_sidecars(
                spec,
                args.branch,
                snapshot.version,
                metadata_root / "versions" / str(snapshot.version),
            )
            materialize_and_activate(
                spec.r2.split_lance_uri("train"),
                snapshot=snapshot,
                metadata_root=metadata_root,
                local_root=args.local_root,
                columns=DATASET_FIELD_NAMES,
            )
        if args.poll_seconds <= 0:
            return
        time.sleep(args.poll_seconds)


def _parser() -> argparse.ArgumentParser:
    """Build the subcommand parser.

    :returns: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="synth-setter-growing-lance",
        description="Append and materialize bounded native Lance train snapshots.",
    )
    subcommands = parser.add_subparsers(required=True)
    init = subcommands.add_parser("init", help="create a growing branch from a baseline")
    init.add_argument("spec_uri", help="R2 URI of the frozen dataset specification")
    init.add_argument("--branch", required=True, type=_branch, help="native Lance branch")
    init.add_argument(
        "--baseline-version",
        default=None,
        type=int,
        help="explicit baseline pin; defaults to the finalized train dataset version",
    )
    init.add_argument("--max-train-shards", required=True, type=int)
    init.add_argument("--num-extra-shards", required=True, type=int)
    init.add_argument("--work-dir", required=True, type=Path)
    init.set_defaults(handler=initialize)

    for name, handler, help_text in (
        ("enqueue", enqueue, "freeze the next bounded range"),
        ("finalize", finalize, "append staged shards as the next ready version"),
        ("generate", generate, "drain pending branch-specific claims"),
        ("grow", grow, "drive enqueue and finalize cycles until capacity"),
    ):
        command = subcommands.add_parser(name, help=help_text)
        command.add_argument("spec_uri")
        command.add_argument("--branch", required=True, type=_branch)
        command.add_argument("--work-dir", required=True, type=Path)
        if name in ("generate", "grow"):
            command.add_argument("--poll-seconds", default=0.0, type=float)
        command.set_defaults(handler=handler)

    materialize = subcommands.add_parser("materialize", help="increment local train.lance")
    materialize.add_argument("spec_uri")
    materialize.add_argument("--branch", required=True, type=_branch)
    materialize.add_argument("--local-root", required=True, type=Path)
    materialize.add_argument("--work-dir", required=True, type=Path)
    materialize.add_argument("--poll-seconds", default=0.0, type=float)
    materialize.set_defaults(handler=materialize_ready)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run one public growing Lance operator command.

    :param argv: Optional command arguments.
    """
    args = _parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
