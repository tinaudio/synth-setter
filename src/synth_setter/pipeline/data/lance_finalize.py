"""Finalize-side Lance fragment commit: staged winners → split manifests (#1776).

Finalize never decodes a row. It reconciles the staging prefix, selects one
winning attempt per shard (earliest ``.valid`` storage ``LastModified``,
tie-broken by full marker key — server-assigned, so a later straggler can never
displace an existing winner), structural-checks each winner, commits the
winners' fragment metadata into each split dataset as one atomic
``Overwrite`` transaction, reduces the winners' Welford sidecars into
``stats.npz``, and records the selection in ``dataset.json``. Design:
``docs/design/data-pipeline.md`` §7.6.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lance
import numpy as np
from loguru import logger
from pydantic import ValidationError

from synth_setter.data.vst.shapes import dataset_field_shapes
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.constants import (
    ATTEMPT_VALID_SUFFIX,
    DATASET_CARD_FILENAME,
    LANCE_FRAGMENT_SIDECAR_SUFFIX,
    LANCE_SHARD_STATS_KEYS,
    LANCE_SHARD_STATS_SUFFIX,
    STATS_NPZ_FILENAME,
)
from synth_setter.pipeline.data.lance_shard import commit_lance_dataset, lance_schema
from synth_setter.pipeline.data.lance_staging import complete_attempt_names, split_for_shard
from synth_setter.pipeline.data.stats import finalize as finalize_welford
from synth_setter.pipeline.data.stats import merge_welford
from synth_setter.pipeline.schemas.lance_attempt import (
    LanceDatasetCard,
    LanceFragmentSidecar,
    SelectedLanceAttempt,
)

if TYPE_CHECKING:
    import pyarrow as pa

    from synth_setter.pipeline.schemas.spec import DatasetSpec


@dataclass(frozen=True)
class StagedLanceAttempt:
    """One complete staged attempt discovered under a shard's staging directory.

    .. attribute :: shard_id

        Logical shard the attempt rendered (derived from the staging path).

    .. attribute :: name

        Attempt name (``{worker_id}-{attempt_uuid}``) from the staging filenames.

    .. attribute :: valid_key

        Full object key of the attempt's ``.valid`` marker — the selection
        tie-breaker and the audit record's provenance pointer.

    .. attribute :: valid_mtime

        Storage-assigned ``LastModified`` of the ``.valid`` marker.
    """

    shard_id: int
    name: str
    valid_key: str
    valid_mtime: datetime


@dataclass(frozen=True)
class CheckedLanceWinner:
    """A winning attempt after structural checks, ready to commit and reduce.

    .. attribute :: attempt

        The selected staged attempt.

    .. attribute :: fragment

        Lance fragment metadata deserialized from the attempt's sidecar.

    .. attribute :: welford

        Welford ``(count, mean, m2)`` state from the attempt's stats sidecar.
    """

    attempt: StagedLanceAttempt
    fragment: lance.fragment.FragmentMetadata
    welford: tuple[int, np.ndarray, np.ndarray]


def staged_complete_attempts(spec: DatasetSpec) -> dict[int, list[StagedLanceAttempt]]:
    """Discover every complete staged attempt in one recursive listing.

    :param spec: Validated dataset spec.
    :returns: Complete attempts grouped by shard id; shards with none are absent.
    """
    root_uri = spec.r2.workers_shards_root_uri()
    root_key = root_uri.removeprefix(f"{r2_io.R2_URI_SCHEME}{spec.r2.bucket}/")
    entries = r2_io.list_entries(root_uri, recursive=True)
    by_shard_dir: dict[str, dict[str, r2_io.RemoteEntry]] = {}
    for entry in entries:
        shard_dir, _, filename = entry.path.partition("/")
        if not filename:
            logger.warning("skipping stray top-level staging object: {}", entry.path)
            continue
        by_shard_dir.setdefault(shard_dir, {})[filename] = entry
    attempts: dict[int, list[StagedLanceAttempt]] = {}
    for shard_dir, files in by_shard_dir.items():
        # Tolerate non-shard entries (stray objects, a future quarantine/ dir):
        # discovery reports what it recognizes rather than aborting finalize.
        if not re.fullmatch(r"shard-\d{6}", shard_dir):
            logger.warning("skipping non-shard staging entry: {}", shard_dir)
            continue
        shard_id = int(shard_dir.removeprefix("shard-"))
        for name in complete_attempt_names(list(files)):
            valid = files[f"{name}{ATTEMPT_VALID_SUFFIX}"]
            attempts.setdefault(shard_id, []).append(
                StagedLanceAttempt(
                    shard_id=shard_id,
                    name=name,
                    valid_key=f"{root_key}{valid.path}",
                    valid_mtime=valid.mtime,
                )
            )
    return attempts


def select_winner(attempts: list[StagedLanceAttempt]) -> StagedLanceAttempt:
    """Select the winning attempt: earliest ``.valid`` mtime, tie-broken by key.

    ``LastModified`` is storage-server-assigned, so the winner is stable — a
    straggler landing later has a strictly greater timestamp and can never
    displace an already-selected winner (what makes finalize re-run safe).
    Effective precision floors at microseconds (listing timestamps parse via
    ``datetime.fromisoformat``) and S3-compatible stores may serve coarser;
    ties resolve deterministically by key, which carries no completion order
    — expect them routinely when workers finish within the same second.

    :param attempts: Non-empty complete attempts for one shard.
    :returns: The winning attempt.
    """
    return min(attempts, key=lambda attempt: (attempt.valid_mtime, attempt.valid_key))


def load_checked_winner(spec: DatasetSpec, attempt: StagedLanceAttempt) -> CheckedLanceWinner:
    """Load a winner's sidecars and run finalize's structural checks.

    Checks (design §7.6 step 5): the sidecar parses as a strict Pydantic model
    and round-trips through Lance's ``FragmentMetadata.from_json``; the stats
    file carries Welford ``count``/``mean``/``m2``; the fragment's row count
    matches the spec and the stats ``count``; and the fragment's data files
    physically exist under the split dataset the spec assigns to this shard
    (a fragment is only readable from the dataset that holds its file).

    :param spec: Validated dataset spec.
    :param attempt: The selected attempt for one shard.
    :returns: The checked winner with parsed fragment and Welford state.
    :raises ValueError: Any structural check fails.
    """
    staging_dir = spec.r2.shard_staging_dir_uri(attempt.shard_id)
    with r2_io.downloaded_to_tempfile(
        f"{staging_dir}{attempt.name}{LANCE_FRAGMENT_SIDECAR_SUFFIX}"
    ) as sidecar_path:
        try:
            sidecar = LanceFragmentSidecar.model_validate_json(sidecar_path.read_text())
        except ValidationError as exc:
            raise ValueError(
                f"shard {attempt.shard_id} attempt {attempt.name}: invalid fragment sidecar: {exc}"
            ) from exc
    try:
        fragment = lance.fragment.FragmentMetadata.from_json(sidecar.fragment_json)
    except Exception as exc:  # noqa: BLE001 — lance raises varied types (KeyError observed) on malformed payloads
        raise ValueError(
            f"shard {attempt.shard_id} attempt {attempt.name}: fragment_json does not "
            f"deserialize as Lance fragment metadata: {type(exc).__name__}: {exc}"
        ) from exc
    with (
        r2_io.downloaded_to_tempfile(
            f"{staging_dir}{attempt.name}{LANCE_SHARD_STATS_SUFFIX}"
        ) as stats_path,
        np.load(stats_path) as stats,
    ):
        missing_keys = [key for key in LANCE_SHARD_STATS_KEYS if key not in stats]
        if missing_keys:
            raise ValueError(
                f"shard {attempt.shard_id} attempt {attempt.name}: shard-stats.npz "
                f"missing arrays {missing_keys}"
            )
        welford = (int(stats["count"]), stats["mean"], stats["m2"])
    rows = fragment.physical_rows
    if rows != spec.render.samples_per_shard or welford[0] != rows:
        raise ValueError(
            f"shard {attempt.shard_id} attempt {attempt.name}: fragment has {rows} rows, "
            f"stats count {welford[0]}; spec expects {spec.render.samples_per_shard}"
        )
    split_uri = spec.r2.split_lance_uri(split_for_shard(spec, attempt.shard_id))
    for data_file in fragment.files:
        # A zero-size object is a truncated upload, not data — treat as absent.
        if not r2_io.object_size(f"{split_uri}/data/{data_file.path}"):
            raise ValueError(
                f"shard {attempt.shard_id} attempt {attempt.name}: fragment data file "
                f"{data_file.path} missing or empty under {split_uri}/data/"
            )
    return CheckedLanceWinner(attempt=attempt, fragment=fragment, welford=welford)


def _split_schema(spec: DatasetSpec, first_shard_id: int) -> pa.Schema:
    """Build a split dataset's Arrow schema from the spec.

    Mirrors the worker writer's construction — shapes from the render config,
    ``ShardMetadata`` (seeded by the split's first shard, matching the schema
    the previous row-streaming finalize inherited from shard 0) embedded in
    schema metadata so consumers keep reading ``sample_rate`` etc. from splits.

    :param spec: Validated dataset spec.
    :param first_shard_id: The split's first shard, whose seed the metadata carries.
    :returns: Arrow schema for the split's ``Overwrite`` commit.
    """
    render = spec.render.model_copy(update={"base_seed": spec.shards[first_shard_id].seed})
    return lance_schema(dataset_field_shapes(render, spec.num_params), render.shard_metadata())


def _select_checked_winners(spec: DatasetSpec) -> dict[int, CheckedLanceWinner]:
    """Reconcile staging, pick one winner per shard, and structural-check each.

    :param spec: Validated dataset spec.
    :returns: Checked winner keyed by shard id, covering every spec shard.
    :raises ValueError: Any spec shard has no staged-valid attempt, or a winner fails a structural
        check.
    """
    attempts = staged_complete_attempts(spec)
    missing = [shard.shard_id for shard in spec.shards if not attempts.get(shard.shard_id)]
    if missing:
        names = ", ".join(f"shard-{shard_id:06d}" for shard_id in missing)
        raise ValueError(
            f"cannot finalize: {len(missing)}/{spec.num_shards} shards have no "
            f"staged-valid attempt: {names}"
        )
    # Winners load serially (two small downloads + a HEAD per shard) — fine at
    # current shard counts; parallelize here if runs grow to thousands.
    return {
        shard.shard_id: load_checked_winner(spec, select_winner(attempts[shard.shard_id]))
        for shard in spec.shards
    }


def _reduce_and_upload_stats(
    spec: DatasetSpec, winners: dict[int, CheckedLanceWinner], work_dir: Path
) -> None:
    """Reduce the train winners' Welford sidecars into ``stats.npz`` and upload it.

    :param spec: Validated dataset spec.
    :param winners: Checked winner per shard id.
    :param work_dir: Scratch directory the ``stats.npz`` is staged in.
    """
    train_lo, train_hi = spec.split_shard_ranges["train"]
    state: tuple[int, Any, Any] = (0, 0, 0)
    for shard_id in range(train_lo, train_hi):
        state = merge_welford(state, winners[shard_id].welford)
    mean, std = finalize_welford(state, mask_degenerate=spec.mask_degenerate_bins)
    stats_npz = work_dir / STATS_NPZ_FILENAME
    np.savez(stats_npz, mean=mean, std=std)
    r2_io.upload(stats_npz, spec.r2.stats_uri())
    logger.info("uploaded stats to {}", spec.r2.stats_uri())


def _write_dataset_card(
    spec: DatasetSpec, winners: dict[int, CheckedLanceWinner], work_dir: Path
) -> None:
    """Record the selected attempts and their winning ``.valid`` keys in ``dataset.json``.

    :param spec: Validated dataset spec.
    :param winners: Checked winner per shard id.
    :param work_dir: Scratch directory the card is staged in.
    """
    card = LanceDatasetCard(
        schema_version=1,
        run_id=spec.run_id,
        finalized_at=datetime.now(UTC).isoformat(),
        selected_attempts=tuple(
            SelectedLanceAttempt(
                shard_id=shard.shard_id,
                attempt=winners[shard.shard_id].attempt.name,
                valid_key=winners[shard.shard_id].attempt.valid_key,
            )
            for shard in spec.shards
        ),
    )
    card_path = work_dir / DATASET_CARD_FILENAME
    card_path.write_text(card.model_dump_json(indent=2))
    r2_io.upload(card_path, spec.r2.dataset_card_uri())
    logger.info("uploaded dataset card to {}", spec.r2.dataset_card_uri())


# DOC502: the documented ValueErrors propagate from _select_checked_winners.
def finalize_lance_fragments(spec: DatasetSpec, work_dir: Path) -> None:  # noqa: DOC502
    """Commit staged winner fragments into split datasets; reduce stats; write the card.

    Each split is one replace-semantics ``Overwrite`` commit over the full
    winner set in shard order — a re-run rebuilds the identical manifest
    instead of appending. Zero rows are decoded. Precondition: the train
    split is non-empty — the entrypoint (``finalize_dataset.finalize_lance``)
    guards it before delegating.

    :param spec: Validated dataset spec (``output_format == "lance"``).
    :param work_dir: Scratch directory for the staged ``stats.npz`` / ``dataset.json``.
    :raises ValueError: Any spec shard has no staged-valid attempt, or a
        winner fails a structural check.
    """
    winners = _select_checked_winners(spec)

    for split, (lo, hi) in spec.split_shard_ranges.items():
        if lo >= hi:
            continue
        target, storage_options = r2_io.lance_target(spec.r2.split_lance_uri(split))
        commit_lance_dataset(
            target,
            _split_schema(spec, lo),
            [winners[shard_id].fragment for shard_id in range(lo, hi)],
            storage_options=storage_options,
        )
        logger.info("committed {} winner fragments into {} split", hi - lo, split)

    _reduce_and_upload_stats(spec, winners, work_dir)
    _write_dataset_card(spec, winners, work_dir)
