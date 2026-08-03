"""Reopen a finalized dataset as an incomplete one so the pipeline extends it (#2862).

Growing ``train_val_test_sizes[0]`` regenerates every original shard position
unchanged — ``sample_offset`` derives from the split-local shard index — so a
grown spec *is* the extension. Reopen copies only preserved train staging and
fragment data, then publishes the grown spec; the existing resume skip-probe
renders only the missing shards. Design:
``docs/design/dataset-reopen.md``.

Typical use is ``reopen_dataset(source_root_uri, new_sizes)`` followed by the
normal generate + finalize entrypoints against ``plan.dest_root_uri``.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, ConfigDict

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.constants import DATASET_COMPLETE_FILENAME
from synth_setter.pipeline.data.lance_staging import COMPLETE_ATTEMPT_SUFFIXES
from synth_setter.pipeline.schemas.lance_attempt import LanceDatasetCard, SelectedLanceAttempt
from synth_setter.pipeline.schemas.prefix import make_dataset_wandb_run_id, make_r2_prefix
from synth_setter.pipeline.schemas.spec import DatasetSpec, Split
from synth_setter.pipeline.spec_io import load_spec_from_root, load_spec_from_uri, upload_spec

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

_SPLITS: tuple[Split, ...] = ("train", "val", "test")
_REOPEN_IDENTITY_RELATIVE_PATH = "metadata/reopen.json"


class _ReopenIdentity(BaseModel):
    """Bind resumable destination state to its exact source and destination specs.

    .. attribute :: model_config

        Strict immutable Pydantic model configuration.

    .. attribute :: source_spec

        Exact finalized source identity.

    .. attribute :: dest_spec

        Exact grown destination identity.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    source_spec: DatasetSpec
    dest_spec: DatasetSpec


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


def _upload_reopen_identity(identity: _ReopenIdentity, uri: str) -> None:
    """Publish a reopen identity before any source state is copied.

    :param identity: Exact source/destination identity to publish.
    :param uri: Destination metadata object URI.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as file:
        file.write(identity.model_dump_json(indent=2))
        file.flush()
        r2_io.upload_to_uri(Path(file.name), uri)


def _load_reopen_identity(uri: str) -> _ReopenIdentity:
    """Parse a strict reopen identity from object storage.

    :param uri: Destination metadata object URI.
    :returns: Strictly parsed reopen identity.
    """
    with r2_io.downloaded_to_tempfile(uri) as path:
        return _ReopenIdentity.model_validate_json(path.read_text())


def _prepare_destination_identity(plan: ReopenPlan, source_spec: DatasetSpec) -> None:
    """Publish or verify the identity required for destination mutation.

    :param plan: Reopen plan naming the destination.
    :param source_spec: Exact source spec being extended.
    :raises ValueError: Existing state has no identity or does not exactly match this source and
        destination spec.
    """
    expected = _ReopenIdentity(source_spec=source_spec, dest_spec=plan.dest_spec)
    identity_uri = f"{plan.dest_root_uri}{_REOPEN_IDENTITY_RELATIVE_PATH}"
    if r2_io.object_size(identity_uri) is None:
        if r2_io.r2_directory_exists(plan.dest_root_uri):
            raise ValueError(
                f"destination {plan.dest_root_uri} has state but no verifiable reopen identity"
            )
        _upload_reopen_identity(expected, identity_uri)

    existing_identity = _load_reopen_identity(identity_uri)
    if existing_identity != expected:
        raise ValueError(
            f"destination {plan.dest_root_uri} reopen identity does not match "
            "the requested source and destination spec"
        )

    spec_uri = plan.dest_spec.r2.input_spec_uri()
    if r2_io.object_size(spec_uri) is not None and load_spec_from_uri(spec_uri) != plan.dest_spec:
        raise ValueError(
            f"destination {plan.dest_root_uri} spec does not match its reopen identity"
        )


def _clear_destination_state(plan: ReopenPlan) -> None:
    """Strictly remove destination state that generation and finalize rebuild.

    :param plan: Reopen plan naming the exact destination layout.
    """
    location = plan.dest_spec.r2
    r2_io.delete_prefix(f"{plan.dest_root_uri}metadata/workers/")
    r2_io.delete_prefix(f"{location.shard_claims_uri()}/")
    for split in _SPLITS:
        r2_io.delete_prefix(f"{location.split_lance_uri(split)}/")
    for uri in (
        location.input_spec_uri(),
        location.dataset_card_uri(),
        location.stats_uri(),
        location.config_yaml_uri(),
    ):
        r2_io.delete_object(uri)


def _source_winners(source_spec: DatasetSpec) -> dict[int, SelectedLanceAttempt]:
    """Load the finalized source's canonical attempt for every shard.

    :param source_spec: Exact finalized source spec.
    :returns: Selected attempt keyed by shard id.
    :raises ValueError: The card belongs to another run or does not select exactly one attempt for
        every source shard.
    """
    with r2_io.downloaded_to_tempfile(source_spec.r2.dataset_card_uri()) as card_path:
        card = LanceDatasetCard.model_validate_json(card_path.read_bytes())
    winners = {attempt.shard_id: attempt for attempt in card.selected_attempts}
    expected_ids = {shard.shard_id for shard in source_spec.shards}
    selected_ids = [attempt.shard_id for attempt in card.selected_attempts]
    if card.run_id != source_spec.run_id or set(selected_ids) != expected_ids:
        raise ValueError("source dataset card does not match the finalized source spec")
    if len(selected_ids) != len(winners):
        raise ValueError("source dataset card selects a shard more than once")
    return winners


def _copy_preserved_state(plan: ReopenPlan, source_spec: DatasetSpec) -> None:
    """Copy only canonical source staging and train fragment data.

    :param plan: Reopen plan naming source, destination, and preserved shards.
    :param source_spec: Exact finalized source spec.
    """
    winners = _source_winners(source_spec)
    if plan.preserved_shard_ids:
        r2_io.copy_prefix(
            f"{source_spec.r2.split_lance_uri('train')}/data/",
            f"{plan.dest_spec.r2.split_lance_uri('train')}/data/",
        )
    for shard_id in plan.preserved_shard_ids:
        attempt_name = winners[shard_id].attempt
        source_dir = source_spec.r2.shard_staging_dir_uri(shard_id)
        dest_dir = plan.dest_spec.r2.shard_staging_dir_uri(shard_id)
        for suffix in COMPLETE_ATTEMPT_SUFFIXES:
            r2_io.upload(f"{source_dir}{attempt_name}{suffix}", f"{dest_dir}{attempt_name}{suffix}")


# DOC502: documented errors propagate from the strict storage helpers.
# DOC502/DOC503: documented errors propagate from source loading and its strict storage probe.
def _load_reopen_source(source_root_uri: str) -> DatasetSpec:  # noqa: DOC502, DOC503
    """Load a finalized source whose embedded storage identity matches its requested root.

    :param source_root_uri: Requested finalized dataset root.
    :returns: Strict source spec loaded from that exact root.
    :raises FileNotFoundError: The source is not finalized.
    :raises ValueError: The loaded spec belongs to another root.
    :raises subprocess.CalledProcessError: A required R2 probe fails.
    """
    source_spec = load_spec_from_root(source_root_uri)
    if source_spec.r2.dataset_root_uri() != source_root_uri:
        raise ValueError("source spec dataset root does not match the requested source root")
    _require_source_complete(source_spec)
    return source_spec


# DOC502: documented errors propagate from strict source loading.
def plan_dataset_reopen(  # noqa: DOC502
    source_root_uri: str,
    new_sizes: Sequence[int],
    *,
    dest_run_id: str | None = None,
) -> ReopenPlan:
    """Plan a dataset reopen without writing destination state.

    :param source_root_uri: ``r2://`` root of the finalized dataset to extend.
    :param new_sizes: Target ``(train, val, test)`` sizes.
    :param dest_run_id: Run id owning the destination prefix; defaults to a
        fresh timestamped id derived from the source's task name.
    :returns: The validated reopen plan.
    :raises FileNotFoundError: The source is not finalized.
    :raises ValueError: The source spec belongs to another root or cannot be reopened.
    :raises subprocess.CalledProcessError: A required R2 probe fails.
    """
    source_spec = _load_reopen_source(source_root_uri)
    return plan_reopen(
        source_spec,
        new_sizes,
        dest_run_id=dest_run_id or make_dataset_wandb_run_id(source_spec.task_name),
    )


# DOC502: documented errors propagate from the strict storage helpers.
def reopen_dataset(  # noqa: DOC502
    source_root_uri: str,
    new_sizes: Sequence[int],
    *,
    dest_run_id: str | None = None,
) -> ReopenPlan:
    """Copy preserved train state into an incomplete grown destination.

    A strict identity makes interrupted copies resumable without admitting
    unrelated destination state. Any existing completion marker is removed
    before destination cleanup; the grown spec is published only after every
    required cleanup and copy succeeds.

    :param source_root_uri: ``r2://`` root of the finalized dataset to extend.
    :param new_sizes: Target ``(train, val, test)`` sizes.
    :param dest_run_id: Run id owning the destination prefix; defaults to a
        fresh timestamped id derived from the source's task name.
    :returns: The reopen plan describing the applied operation.
    :raises FileNotFoundError: The source is not finalized.
    :raises ValueError: Existing destination state is not an exact resume.
    :raises subprocess.CalledProcessError: A required R2 operation fails.
    """
    source_spec = _load_reopen_source(source_root_uri)
    plan = plan_reopen(
        source_spec,
        new_sizes,
        dest_run_id=dest_run_id or make_dataset_wandb_run_id(source_spec.task_name),
    )
    _prepare_destination_identity(plan, source_spec)
    r2_io.delete_object(plan.dest_spec.r2.dataset_complete_marker_uri())
    _clear_destination_state(plan)
    _copy_preserved_state(plan, source_spec)
    upload_spec(plan.dest_spec)
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

    source_spec = _load_reopen_source(args.source)
    _, val_size, test_size = source_spec.train_val_test_sizes
    operation = reopen_dataset if args.apply else plan_dataset_reopen
    plan = operation(
        args.source,
        (args.train_size, val_size, test_size),
        dest_run_id=args.dest_run_id,
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
