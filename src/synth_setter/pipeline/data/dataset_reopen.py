"""Reopen a finalized dataset as an incomplete one so the pipeline extends it (#2862).

Growing ``train_val_test_sizes[0]`` regenerates every original shard position
unchanged — ``sample_offset`` derives from the split-local shard index — so a
grown spec *is* the extension. Reopen copies the finalized root, rewrites the
spec, and clears the markers that would otherwise short-circuit generation;
the existing resume skip-probe then renders only the missing shards. Design:
``docs/design/dataset-reopen.md``.

Typical use is ``reopen_dataset(source_root_uri, new_sizes)`` followed by the
normal generate + finalize entrypoints against ``plan.dest_root_uri``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.constants import DATASET_COMPLETE_FILENAME
from synth_setter.pipeline.schemas.prefix import make_dataset_wandb_run_id, make_r2_prefix
from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.pipeline.spec_io import load_spec_from_root, upload_spec

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ReopenPlan:
    """Shard-space partition a reopen produces, computed before any write.

    .. attribute :: source_root_uri

        Finalized root the extension reads from; never mutated.

    .. attribute :: dest_root_uri

        Root the reopened, incomplete dataset lands at.

    .. attribute :: dest_spec

        Grown spec addressing ``dest_root_uri``.

    .. attribute :: preserved_shard_ids

        Shards whose staging is kept; the skip-probe short-circuits these.

    .. attribute :: discarded_shard_ids

        Shards whose staging is deleted — the source's val/test, whose ids the
        grown spec reassigns to train.

    .. attribute :: pending_shard_ids

        Shards with no staging left, i.e. everything generation must render.
    """

    source_root_uri: str
    dest_root_uri: str
    dest_spec: DatasetSpec
    preserved_shard_ids: range
    discarded_shard_ids: range
    pending_shard_ids: range


def validate_reopenable(source_spec: DatasetSpec, new_sizes: Sequence[int]) -> None:
    """Reject source specs and target sizes that cannot compose into one coherent stream.

    :param source_spec: Spec of the finalized dataset being extended.
    :param new_sizes: Target ``(train, val, test)`` sizes.
    :raises ValueError: The source derives seeds from shard ids
        (``train_val_test_seeds is None``), train shrinks, val/test sizes
        change, or a size is not a multiple of ``render.samples_per_shard``.
    """
    if source_spec.train_val_test_seeds is None:
        raise ValueError(
            "cannot reopen a spec with train_val_test_seeds=None: its rows are seeded from "
            "base_seed + shard_id, which renumbering would redefine"
        )
    new_train, new_val, new_test = new_sizes
    old_train, old_val, old_test = source_spec.train_val_test_sizes
    if new_train < old_train:
        raise ValueError(f"cannot shrink train from {old_train} to {new_train}")
    if (new_val, new_test) != (old_val, old_test):
        raise ValueError(
            f"val/test sizes must not change on reopen: got ({new_val}, {new_test}), "
            f"source has ({old_val}, {old_test})"
        )
    samples_per_shard = source_spec.render.samples_per_shard
    ragged = [size for size in new_sizes if size % samples_per_shard]
    if ragged:
        raise ValueError(
            f"sizes {ragged} are not a multiple of render.samples_per_shard={samples_per_shard}"
        )


def _grown_spec(
    source_spec: DatasetSpec, new_sizes: Sequence[int], dest_run_id: str
) -> DatasetSpec:
    """Rebuild the spec at the new sizes, addressing a fresh run prefix.

    Rebuilt rather than ``model_copy``-ed: ``shards`` / ``num_shards`` /
    ``split_shard_ranges`` are cached properties, and a copy carries the
    source's stale values through.

    :param source_spec: Spec of the finalized dataset being extended.
    :param new_sizes: Target ``(train, val, test)`` sizes.
    :param dest_run_id: Run id owning the destination prefix.
    :returns: The grown spec with recomputed shard layout.
    :raises ValueError: The destination prefix equals the source's, which would
        make workers stage into the root this operation exists to protect.
    """
    payload = source_spec.model_dump(mode="json")
    payload["train_val_test_sizes"] = list(new_sizes)
    payload["run_id"] = dest_run_id
    payload["r2"]["prefix"] = make_r2_prefix(
        source_spec.task_name, dest_run_id, source_spec.r2.prefix_root
    )
    dest_spec = DatasetSpec(**payload)
    if dest_spec.r2.prefix == source_spec.r2.prefix:
        raise ValueError(
            f"destination prefix {dest_spec.r2.prefix!r} equals the source's; "
            "reopen must never write into the root it extends"
        )
    return dest_spec


# DOC502: the documented ValueErrors propagate from validate_reopenable / _grown_spec.
def plan_reopen(  # noqa: DOC502
    source_spec: DatasetSpec, new_sizes: Sequence[int], *, dest_run_id: str
) -> ReopenPlan:
    """Partition the shard space and build the destination spec, without writing.

    :param source_spec: Spec of the finalized dataset being extended.
    :param new_sizes: Target ``(train, val, test)`` sizes.
    :param dest_run_id: Run id owning the destination prefix.
    :returns: The reopen plan.
    :raises ValueError: ``validate_reopenable`` rejects the request, or the
        destination prefix collides with the source's.
    """
    validate_reopenable(source_spec, new_sizes)
    dest_spec = _grown_spec(source_spec, new_sizes, dest_run_id)
    old_train_shards = source_spec.train_val_test_sizes[0] // source_spec.render.samples_per_shard
    return ReopenPlan(
        source_root_uri=source_spec.r2.dataset_root_uri(),
        dest_root_uri=dest_spec.r2.dataset_root_uri(),
        dest_spec=dest_spec,
        preserved_shard_ids=range(old_train_shards),
        discarded_shard_ids=range(old_train_shards, source_spec.num_shards),
        pending_shard_ids=range(old_train_shards, dest_spec.num_shards),
    )


def _require_source_complete(source_spec: DatasetSpec) -> None:
    """Require the source's finalize marker before extending it.

    :param source_spec: Spec of the dataset being extended.
    :raises FileNotFoundError: The source's completion marker is absent.
    """
    marker = source_spec.r2.dataset_complete_marker_uri()
    if r2_io.object_size(marker) is None:
        raise FileNotFoundError(
            f"{DATASET_COMPLETE_FILENAME} marker {marker} is missing; "
            "only a finalized dataset can be reopened"
        )


def _purge_uri(r2_uri: str) -> None:
    """Recursively delete one ``r2://`` directory prefix.

    :param r2_uri: ``r2://bucket/key/`` prefix to wipe; must end in ``/``.
    """
    bucket, _, key = r2_uri.removeprefix("r2://").partition("/")
    r2_io.purge_prefix(bucket, key)


# DOC502: the documented FileNotFoundError propagates from _require_source_complete.
def reopen_dataset(  # noqa: DOC502
    source_root_uri: str,
    new_sizes: Sequence[int],
    *,
    dest_run_id: str | None = None,
    dry_run: bool = False,
) -> ReopenPlan:
    """Copy a finalized root and reopen the copy so generation can extend it.

    Writes, in order: the copied prefix, the grown spec, then the marker
    deletions. ``dataset.json`` is deliberately kept — its entries pin which
    attempt won for each preserved shard.

    :param source_root_uri: ``r2://`` root of the finalized dataset to extend.
    :param new_sizes: Target ``(train, val, test)`` sizes.
    :param dest_run_id: Run id owning the destination prefix; defaults to a
        fresh timestamped id derived from the source's task name.
    :param dry_run: When true, plan and return without writing anything.
    :returns: The reopen plan describing what was (or would be) done.
    :raises FileNotFoundError: The source is not finalized.
    """
    source_spec = load_spec_from_root(source_root_uri)
    _require_source_complete(source_spec)
    plan = plan_reopen(
        source_spec,
        new_sizes,
        dest_run_id=dest_run_id or make_dataset_wandb_run_id(source_spec.task_name),
    )
    if dry_run:
        logger.info(
            "reopen_dry_run",
            dest_root=plan.dest_root_uri,
            preserved=len(plan.preserved_shard_ids),
            pending=len(plan.pending_shard_ids),
        )
        return plan

    # Excluded from the copy, not deleted after it: a crash mid-reopen would
    # otherwise leave a destination that reads as finalized but holds the
    # source's spec, and hydration trusts that marker.
    r2_io.copy_prefix(
        plan.source_root_uri, plan.dest_root_uri, exclude=DATASET_COMPLETE_FILENAME
    )
    upload_spec(plan.dest_spec)
    # Idempotency guard for a re-run over a destination a previous attempt finalized.
    r2_io.delete_object(plan.dest_spec.r2.dataset_complete_marker_uri())
    for shard_id in plan.discarded_shard_ids:
        _purge_uri(plan.dest_spec.r2.shard_staging_dir_uri(shard_id))
    _purge_uri(f"{plan.dest_spec.r2.shard_claims_uri()}/")
    logger.info(
        "reopened_dataset",
        source_root=plan.source_root_uri,
        dest_root=plan.dest_root_uri,
        preserved=len(plan.preserved_shard_ids),
        discarded=len(plan.discarded_shard_ids),
        pending=len(plan.pending_shard_ids),
    )
    return plan


def main(argv: list[str] | None = None) -> None:
    """Reopen a finalized dataset at a larger train size.

    Plans only unless ``--apply`` is passed: the copy it performs is the size
    of the source dataset, so writing is opt-in.

    :param argv: argv tail (without the program name); ``None`` reads
        ``sys.argv[1:]`` (the console-script path). Injectable for tests.
    """
    parser = argparse.ArgumentParser(
        prog="synth-setter-reopen-dataset",
        description=(
            "Copy a finalized dataset and reopen the copy at a larger train size so the "
            "normal generate + finalize entrypoints extend it without re-rendering."
        ),
    )
    parser.add_argument("--source", required=True, help="r2:// root of the finalized dataset")
    parser.add_argument("--train-size", required=True, type=int, help="target train row count")
    parser.add_argument("--dest-run-id", default=None, help="run id owning the destination prefix")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the copy and rewrite; without it the plan is printed and nothing is written",
    )
    args = parser.parse_args(argv)

    source_spec = load_spec_from_root(args.source)
    _, val_size, test_size = source_spec.train_val_test_sizes
    plan = reopen_dataset(
        args.source,
        (args.train_size, val_size, test_size),
        dest_run_id=args.dest_run_id,
        dry_run=not args.apply,
    )
    logger.info(
        "reopen_plan",
        applied=args.apply,
        dest_root=plan.dest_root_uri,
        preserved=len(plan.preserved_shard_ids),
        discarded=len(plan.discarded_shard_ids),
        pending=len(plan.pending_shard_ids),
    )


if __name__ == "__main__":
    main()
