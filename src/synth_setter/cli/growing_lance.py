"""Public operator commands for append-only Lance train growth."""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import time
from pathlib import Path

import lance

from synth_setter.data.vst.shapes import DATASET_FIELD_NAMES
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.growing_lance import (
    GrowingSnapshot,
    PendingRefreshRequest,
    dataset_spec_fingerprint,
    finalize_staged_refresh,
    generate_pending_shards,
    initialize_growing_branch,
    materialize_and_activate,
    pending_refresh_request,
)
from synth_setter.pipeline.data.lance_materialize import retry_lance_read
from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.pipeline.shard_claims import ShardClaims
from synth_setter.pipeline.spec_io import load_spec_from_uri

_BRANCH_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _branch(value: str) -> str:
    if not _BRANCH_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError("branch must match [A-Za-z0-9][A-Za-z0-9._-]*")
    return value


def _load_snapshot(uri: str) -> GrowingSnapshot:
    with r2_io.downloaded_to_tempfile(uri) as path:
        return GrowingSnapshot.model_validate_json(path.read_bytes())


def _load_pending(uri: str) -> PendingRefreshRequest:
    with r2_io.downloaded_to_tempfile(uri) as path:
        return PendingRefreshRequest.model_validate_json(path.read_bytes())


def _publish_metadata(spec: DatasetSpec, snapshot: GrowingSnapshot, version_dir: Path) -> None:
    for name in ("snapshot.json", "stats.npz", "welford.npz"):
        source = version_dir / name
        if source.is_file():
            r2_io.upload(
                source,
                spec.r2.growing_metadata_uri(
                    snapshot.branch, f"versions/{snapshot.version}/{name}"
                ),
            )


def _ready_snapshot(spec: DatasetSpec, branch: str) -> GrowingSnapshot:
    target, storage_options = r2_io.lance_target(spec.r2.split_lance_uri("train"))
    version = retry_lance_read(
        "growing_ready_tag_read",
        lambda: lance.dataset(target, storage_options=storage_options).tags.get_version(
            f"{branch}-ready"
        ),
    )
    if version is None:
        raise ValueError(f"ready tag for branch {branch!r} does not exist")
    snapshot = _load_snapshot(
        spec.r2.growing_metadata_uri(branch, f"versions/{version}/snapshot.json")
    )
    if snapshot.branch != branch or snapshot.version != version:
        raise ValueError("ready tag and growing snapshot identity disagree")
    if snapshot.dataset_spec_fingerprint != dataset_spec_fingerprint(spec):
        raise ValueError("ready snapshot dataset specification disagrees with input")
    return snapshot


def initialize(args: argparse.Namespace) -> None:
    """Initialize a branch pinned to a complete finalized baseline.

    :param args: Parsed init arguments.
    """
    spec = load_spec_from_uri(args.spec_uri)
    r2_io.ensure_r2_env_loaded()
    with r2_io.downloaded_to_tempfile(spec.r2.dataset_complete_marker_uri()):
        pass
    version_dir = args.work_dir / "versions" / str(args.baseline_version)
    version_dir.mkdir(parents=True, exist_ok=True)
    for name, uri in (("stats.npz", spec.r2.stats_uri()), ("welford.npz", spec.r2.welford_uri())):
        with r2_io.downloaded_to_tempfile(uri) as source:
            shutil.copyfile(source, version_dir / name)
    initialize_growing_branch(
        spec.r2.split_lance_uri("train"),
        spec=spec,
        branch=args.branch,
        baseline_version=args.baseline_version,
        metadata_root=args.work_dir,
        max_train_shards=args.max_train_shards,
        num_extra_shards=args.num_extra_shards,
        publish_metadata=lambda snapshot, root: _publish_metadata(spec, snapshot, root),
    )


def enqueue(args: argparse.Namespace) -> None:
    """Persist the next bounded range and seed branch-specific claims.

    :param args: Parsed enqueue arguments.
    :raises ValueError: An incompatible pending request already exists.
    """
    spec = load_spec_from_uri(args.spec_uri)
    r2_io.ensure_r2_env_loaded()
    current = _ready_snapshot(spec, args.branch)
    pending = pending_refresh_request(current)
    if pending is None:
        return
    pending_uri = spec.r2.growing_metadata_uri(args.branch, "pending.json")
    entries = r2_io.list_entries(spec.r2.growing_metadata_uri(args.branch, ""))
    if any(entry.path == "pending.json" for entry in entries):
        if _load_pending(pending_uri) != pending:
            raise ValueError("another pending growing refresh already exists")
    else:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        local = args.work_dir / "pending.json"
        local.write_text(pending.model_dump_json(indent=2), encoding="utf-8")
        r2_io.upload(local, pending_uri)
    claims = ShardClaims.for_run(
        *r2_io.lance_target(spec.r2.growing_shard_claims_uri(args.branch))
    )
    claims.populate(pending.enqueue_shard_ids)


def generate(args: argparse.Namespace) -> None:
    """Drain pending claims without ever crossing the configured maximum.

    :param args: Parsed generate arguments.
    """
    spec = load_spec_from_uri(args.spec_uri)
    r2_io.ensure_r2_env_loaded()
    current = _ready_snapshot(spec, args.branch)
    if pending_refresh_request(current) is None:
        return
    pending = _load_pending(spec.r2.growing_metadata_uri(args.branch, "pending.json"))
    generate_pending_shards(spec, current, pending, work_dir=args.work_dir)


def _complete_pending(
    spec: DatasetSpec,
    branch: str,
    pending_uri: str,
    pending: PendingRefreshRequest,
    version: int,
    work_dir: Path,
) -> None:
    completed = work_dir / f"completed-{version}.json"
    completed.parent.mkdir(parents=True, exist_ok=True)
    completed.write_text(pending.model_dump_json(indent=2), encoding="utf-8")
    r2_io.upload(completed, spec.r2.growing_metadata_uri(branch, f"completed/{version}.json"))
    r2_io.delete_file(pending_uri)


def finalize(args: argparse.Namespace) -> None:
    """Append one pending range and clear it only after readiness.

    :param args: Parsed finalize arguments.
    :raises ValueError: A capacity snapshot disagrees with its stale request.
    """
    spec = load_spec_from_uri(args.spec_uri)
    r2_io.ensure_r2_env_loaded()
    current = _ready_snapshot(spec, args.branch)
    pending_uri = spec.r2.growing_metadata_uri(args.branch, "pending.json")
    if pending_refresh_request(current) is None:
        entries = r2_io.list_entries(spec.r2.growing_metadata_uri(args.branch, ""))
        if any(entry.path == "pending.json" for entry in entries):
            completed = _load_pending(pending_uri)
            if current.high_watermark != completed.next_high_watermark:
                raise ValueError("capacity snapshot disagrees with pending refresh")
            _complete_pending(
                spec,
                args.branch,
                pending_uri,
                completed,
                current.version,
                args.work_dir,
            )
        return
    current_dir = args.work_dir / "versions" / str(current.version)
    current_dir.mkdir(parents=True, exist_ok=True)
    for name in ("stats.npz", "welford.npz"):
        uri = spec.r2.growing_metadata_uri(args.branch, f"versions/{current.version}/{name}")
        with r2_io.downloaded_to_tempfile(uri) as source:
            shutil.copyfile(source, current_dir / name)
    pending = _load_pending(pending_uri)
    if current.high_watermark == pending.next_high_watermark:
        _complete_pending(spec, args.branch, pending_uri, pending, current.version, args.work_dir)
        return
    published = finalize_staged_refresh(
        spec.r2.split_lance_uri("train"),
        spec=spec,
        current=current,
        pending=pending,
        metadata_root=args.work_dir,
        publish_metadata=lambda snapshot, root: _publish_metadata(spec, snapshot, root),
    )
    _complete_pending(spec, args.branch, pending_uri, pending, published.version, args.work_dir)


def materialize_ready(args: argparse.Namespace) -> None:
    """Poll and increment one shared local train dataset under a file lock.

    :param args: Parsed materialization arguments.
    """
    spec = load_spec_from_uri(args.spec_uri)
    r2_io.ensure_r2_env_loaded()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    while True:
        snapshot = _ready_snapshot(spec, args.branch)
        with tempfile.TemporaryDirectory(dir=args.work_dir) as temporary:
            metadata_root = Path(temporary)
            version_dir = metadata_root / "versions" / str(snapshot.version)
            version_dir.mkdir(parents=True)
            for name in ("stats.npz", "welford.npz"):
                uri = spec.r2.growing_metadata_uri(
                    args.branch, f"versions/{snapshot.version}/{name}"
                )
                with r2_io.downloaded_to_tempfile(uri) as source:
                    shutil.copyfile(source, version_dir / name)
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
    parser = argparse.ArgumentParser(
        prog="synth-setter-growing-lance",
        description="Append and materialize bounded native Lance train snapshots.",
    )
    subcommands = parser.add_subparsers(required=True)
    init = subcommands.add_parser("init", help="create a growing branch from a baseline")
    init.add_argument("spec_uri", help="R2 URI of the frozen dataset specification")
    init.add_argument("--branch", required=True, type=_branch, help="native Lance branch")
    init.add_argument("--baseline-version", required=True, type=int)
    init.add_argument("--max-train-shards", required=True, type=int)
    init.add_argument("--num-extra-shards", required=True, type=int)
    init.add_argument("--work-dir", required=True, type=Path)
    init.set_defaults(handler=initialize)

    for name, handler, help_text in (
        ("enqueue", enqueue, "freeze the next bounded range"),
        ("finalize", finalize, "append staged shards as the next ready version"),
        ("generate", generate, "drain pending branch-specific claims"),
    ):
        command = subcommands.add_parser(name, help=help_text)
        command.add_argument("spec_uri")
        command.add_argument("--branch", required=True, type=_branch)
        command.add_argument("--work-dir", required=True, type=Path)
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
