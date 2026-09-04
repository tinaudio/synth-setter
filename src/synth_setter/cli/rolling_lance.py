"""Public operator commands for native Lance rolling train snapshots."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import lance

from synth_setter.data.vst.shapes import DATASET_FIELD_NAMES
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.rolling_lance import (
    PendingRefreshRequest,
    RollingSnapshot,
    RollingWindow,
    dataset_spec_fingerprint,
    finalize_staged_refresh,
    generate_pending_shards,
    initialize_rolling_branch,
    materialize_and_activate,
    pending_refresh_request,
)
from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.pipeline.shard_claims import ShardClaims
from synth_setter.pipeline.spec_io import load_spec_from_uri

_BRANCH_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _branch(value: str) -> str:
    if not _BRANCH_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError("branch must match [A-Za-z0-9][A-Za-z0-9._-]*")
    return value


def _load_snapshot(uri: str) -> RollingSnapshot:
    with r2_io.downloaded_to_tempfile(uri) as path:
        return RollingSnapshot.model_validate_json(path.read_bytes())


def _load_pending(uri: str) -> PendingRefreshRequest:
    with r2_io.downloaded_to_tempfile(uri) as path:
        return PendingRefreshRequest.model_validate_json(path.read_bytes())


def _publish_metadata(spec: DatasetSpec, snapshot: RollingSnapshot, version_dir: Path) -> None:
    for name in ("stats.npz", "snapshot.json"):
        source = version_dir / name
        if source.is_file():
            r2_io.upload(
                source,
                spec.r2.rolling_metadata_uri(
                    snapshot.branch, f"versions/{snapshot.version}/{name}"
                ),
            )


def _ready_snapshot(spec: DatasetSpec, branch: str) -> RollingSnapshot:
    train_uri = spec.r2.split_lance_uri("train")
    target, storage_options = r2_io.lance_target(train_uri)
    version = lance.dataset(target, storage_options=storage_options).tags.get_version(
        f"{branch}-ready"
    )
    if version is None:
        raise ValueError(f"ready tag for branch {branch!r} does not exist")
    snapshot = _load_snapshot(
        spec.r2.rolling_metadata_uri(branch, f"versions/{version}/snapshot.json")
    )
    if snapshot.branch != branch or snapshot.version != version:
        raise ValueError("ready tag and snapshot identity disagree")
    if snapshot.dataset_spec_fingerprint != dataset_spec_fingerprint(spec):
        raise ValueError("ready snapshot dataset specification disagrees with input")
    return snapshot


def initialize(args: argparse.Namespace) -> None:
    """Initialize a branch pinned to a complete finalized baseline.

    :param args: Parsed ``init`` command arguments.
    """
    spec = load_spec_from_uri(args.spec_uri)
    r2_io.ensure_r2_env_loaded()
    with r2_io.downloaded_to_tempfile(spec.r2.dataset_complete_marker_uri()):
        pass
    version_dir = args.work_dir / "versions" / str(args.baseline_version)
    version_dir.mkdir(parents=True, exist_ok=True)
    with r2_io.downloaded_to_tempfile(spec.r2.stats_uri()) as stats:
        shutil.copyfile(stats, version_dir / "stats.npz")
    initialize_rolling_branch(
        spec.r2.split_lance_uri("train"),
        spec=spec,
        branch=args.branch,
        baseline_version=args.baseline_version,
        metadata_root=args.work_dir,
        num_extra_shards=args.num_extra_shards,
        publish_metadata=lambda snapshot, root: _publish_metadata(spec, snapshot, root),
    )


def enqueue(args: argparse.Namespace) -> None:
    """Durably freeze one pending range and seed its claims idempotently.

    :param args: Parsed ``enqueue`` command arguments.
    :raises ValueError: Another refresh is already pending for a different source snapshot.
    """
    spec = load_spec_from_uri(args.spec_uri)
    r2_io.ensure_r2_env_loaded()
    current = _ready_snapshot(spec, args.branch)
    pending = pending_refresh_request(current)
    pending_uri = spec.r2.rolling_metadata_uri(args.branch, "pending.json")
    entries = r2_io.list_entries(spec.r2.rolling_metadata_uri(args.branch, ""))
    if any(entry.path == "pending.json" for entry in entries):
        if _load_pending(pending_uri) != pending:
            raise ValueError("another pending refresh already exists")
    else:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        local = args.work_dir / "pending.json"
        local.write_text(pending.model_dump_json(indent=2), encoding="utf-8")
        r2_io.upload(local, pending_uri)
    window = RollingWindow(current.window_size, current.num_extra_shards, current.high_watermark)
    shard_ids = [window.extra_shard(spec, item).shard_id for item in pending.enqueue_relative_ids]
    claims = ShardClaims.for_run(
        *r2_io.lance_target(spec.r2.rolling_shard_claims_uri(args.branch))
    )
    claims.populate(shard_ids)


def generate(args: argparse.Namespace) -> None:
    """Drain pending claims and stage uncommitted fragments.

    :param args: Parsed ``generate`` command arguments.
    """
    spec = load_spec_from_uri(args.spec_uri)
    r2_io.ensure_r2_env_loaded()
    current = _ready_snapshot(spec, args.branch)
    pending = _load_pending(spec.r2.rolling_metadata_uri(args.branch, "pending.json"))
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
    r2_io.upload(
        completed,
        spec.r2.rolling_metadata_uri(branch, f"completed/{version}.json"),
    )
    subprocess.run(  # noqa: S603 — URI is produced by the validated R2Location.
        [  # noqa: S607 — rclone is a required project executable resolved by PATH.
            "rclone",
            "deletefile",
            r2_io.to_rclone_path(pending_uri),
            "--checksum",
        ],
        check=True,
    )


def finalize(args: argparse.Namespace) -> None:
    """Commit one pending ready snapshot and clear its request after publication.

    :param args: Parsed ``finalize`` command arguments.
    """
    spec = load_spec_from_uri(args.spec_uri)
    r2_io.ensure_r2_env_loaded()
    current = _ready_snapshot(spec, args.branch)
    pending_uri = spec.r2.rolling_metadata_uri(args.branch, "pending.json")
    pending = _load_pending(pending_uri)
    if (
        current.high_watermark == pending.next_high_watermark
        and current.membership_relative_ids == pending.membership_relative_ids
    ):
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
    """Poll and atomically activate exact native ready-tag versions.

    :param args: Parsed ``materialize`` command arguments.
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
            uri = spec.r2.rolling_metadata_uri(
                args.branch, f"versions/{snapshot.version}/stats.npz"
            )
            with r2_io.downloaded_to_tempfile(uri) as source:
                shutil.copyfile(source, version_dir / "stats.npz")
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
        prog="synth-setter-rolling-lance",
        description="Publish and materialize immutable rolling Lance train snapshots.",
    )
    subcommands = parser.add_subparsers(required=True)
    init = subcommands.add_parser(
        "init", help="create a rolling branch from a finalized train version"
    )
    init.add_argument("spec_uri", help="R2 URI of the frozen dataset specification")
    init.add_argument("--branch", required=True, type=_branch, help="native Lance branch name")
    init.add_argument(
        "--baseline-version", required=True, type=int, help="finalized train version to pin"
    )
    init.add_argument(
        "--num-extra-shards", required=True, type=int, help="shards replaced per refresh"
    )
    init.add_argument("--work-dir", required=True, type=Path, help="operator scratch directory")
    init.set_defaults(handler=initialize)

    commands = (
        ("enqueue", enqueue, "freeze the next shard range and seed its claim queue"),
        ("finalize", finalize, "publish staged shards as the next ready version"),
        ("generate", generate, "drain pending claims and stage rendered shards"),
    )
    for name, handler, help_text in commands:
        command = subcommands.add_parser(name, help=help_text)
        command.add_argument("spec_uri", help="R2 URI of the frozen dataset specification")
        command.add_argument("--branch", required=True, type=_branch, help="native Lance branch")
        command.add_argument(
            "--work-dir", required=True, type=Path, help="operator scratch directory"
        )
        command.set_defaults(handler=handler)

    materialize = subcommands.add_parser(
        "materialize", help="activate ready versions in immutable local directories"
    )
    materialize.add_argument("spec_uri", help="R2 URI of the frozen dataset specification")
    materialize.add_argument("--branch", required=True, type=_branch, help="native Lance branch")
    materialize.add_argument(
        "--local-root", required=True, type=Path, help="root containing versions and active.json"
    )
    materialize.add_argument(
        "--work-dir", required=True, type=Path, help="materialization scratch directory"
    )
    materialize.add_argument(
        "--poll-seconds",
        default=0.0,
        type=float,
        help="poll interval; zero materializes once (default: %(default)s)",
    )
    materialize.set_defaults(handler=materialize_ready)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run one public rolling Lance operator command.

    :param argv: Optional argument tail for tests; ``None`` reads process argv.
    """
    args = _parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
