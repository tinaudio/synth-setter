"""Finalize-side Lance fragment commit: staged winners → split manifests (#1776).

Finalize never decodes a row. It reconciles the staging prefix, selects one
winning attempt per shard (earliest ``.valid`` storage ``LastModified``,
tie-broken by full marker key on the first run and pinned by ``dataset.json``
afterward), structural-checks each winner, commits the
winners' fragment metadata into each split dataset as one atomic
``Overwrite`` transaction, reduces the winners' Welford sidecars into
``stats.npz``, and records the selection in ``dataset.json``. Design:
``docs/design/data-pipeline.md`` §7.6.

Typical use is ``finalize_lance_fragments(spec, work_dir)`` after every shard
has published a staged-valid attempt.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from zipfile import BadZipFile

import lance
import numpy as np
import structlog
from pydantic import ValidationError

from synth_setter.data.vst.shapes import MEL_SPEC_FIELD, dataset_field_shapes
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
from synth_setter.pipeline.data.lance_staging import (
    complete_attempt_names,
    invalidate_staged_attempt,
    split_for_shard,
)
from synth_setter.pipeline.data.stats import WelfordState, merge_welford
from synth_setter.pipeline.data.stats import finalize as finalize_welford
from synth_setter.pipeline.schemas.lance_attempt import (
    LanceDatasetCard,
    LanceFragmentSidecar,
    SelectedLanceAttempt,
)
from synth_setter.pipeline.schemas.r2_location import parse_shard_staging_dir

logger = structlog.get_logger(__name__)

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
    welford: WelfordState


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
            logger.warning("skipping_stray_top_level_staging_object", path=entry.path)
            continue
        # Nested dirs under a shard (e.g. a future quarantine/) never hold
        # staged attempts — only a shard dir's direct children count.
        if "/" in filename:
            continue
        by_shard_dir.setdefault(shard_dir, {})[filename] = entry
    attempts: dict[int, list[StagedLanceAttempt]] = {}
    for shard_dir, files in by_shard_dir.items():
        # Tolerate non-shard entries (stray objects, a future quarantine/ dir):
        # discovery reports what it recognizes rather than aborting finalize.
        shard_id = parse_shard_staging_dir(shard_dir)
        if shard_id is None:
            logger.warning("skipping_non_shard_staging_entry", shard_dir=shard_dir)
            continue
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


def select_winner(attempts: Sequence[StagedLanceAttempt]) -> StagedLanceAttempt:
    """Select the winning attempt: earliest ``.valid`` mtime, tie-broken by key.

    Storage timestamps establish first-run order and full keys break coarse
    timestamp ties deterministically. Post-card reruns preserve the recorded
    winner in :func:`_select_checked_winners`, so a tied later arrival cannot
    displace an already-published selection.

    :param attempts: Non-empty complete attempts for one shard.
    :returns: The winning attempt.
    """
    return min(attempts, key=lambda attempt: (attempt.valid_mtime, attempt.valid_key))


def _load_fragment_metadata(
    spec: DatasetSpec, attempt: StagedLanceAttempt
) -> lance.fragment.FragmentMetadata:
    """Parse one strict sidecar into Lance-owned fragment metadata.

    :param spec: Validated dataset spec.
    :param attempt: Staged attempt whose sidecar is loaded.
    :returns: Deserialized Lance fragment metadata.
    :raises ValueError: The sidecar or nested Lance metadata is invalid.
    """
    uri = (
        f"{spec.r2.shard_staging_dir_uri(attempt.shard_id)}"
        f"{attempt.name}{LANCE_FRAGMENT_SIDECAR_SUFFIX}"
    )
    with r2_io.downloaded_to_tempfile(uri) as sidecar_path:
        try:
            sidecar = LanceFragmentSidecar.model_validate_json(sidecar_path.read_bytes())
        except ValidationError as exc:
            raise ValueError(
                f"shard {attempt.shard_id} attempt {attempt.name}: invalid fragment sidecar: {exc}"
            ) from exc
    try:
        return lance.fragment.FragmentMetadata.from_json(sidecar.fragment_json)
    except Exception as exc:  # noqa: BLE001 — Lance raises varied errors for malformed payloads.
        raise ValueError(
            f"shard {attempt.shard_id} attempt {attempt.name}: fragment_json does not "
            f"deserialize as Lance fragment metadata: {type(exc).__name__}: {exc}"
        ) from exc


def _load_welford_state(spec: DatasetSpec, attempt: StagedLanceAttempt) -> WelfordState:
    """Load and validate one staged Welford archive.

    :param spec: Validated dataset spec defining the expected mel shape.
    :param attempt: Staged attempt whose statistics are loaded.
    :returns: Validated ``(count, mean, m2)`` state.
    :raises ValueError: The archive contract is malformed or numerically invalid.
    """
    uri = (
        f"{spec.r2.shard_staging_dir_uri(attempt.shard_id)}"
        f"{attempt.name}{LANCE_SHARD_STATS_SUFFIX}"
    )
    with r2_io.downloaded_to_tempfile(uri) as stats_path:
        try:
            stats_archive = np.load(stats_path)
        except (BadZipFile, EOFError, OSError, ValueError) as exc:
            raise ValueError(
                f"shard {attempt.shard_id} attempt {attempt.name}: invalid "
                f"shard-stats.npz: {type(exc).__name__}: {exc}"
            ) from exc
        with stats_archive as stats:
            missing_keys = [key for key in LANCE_SHARD_STATS_KEYS if key not in stats]
            if missing_keys:
                raise ValueError(
                    f"shard {attempt.shard_id} attempt {attempt.name}: shard-stats.npz "
                    f"missing arrays {missing_keys}"
                )
            count, mean, m2 = stats["count"], stats["mean"], stats["m2"]
            expected_shape = dataset_field_shapes(spec.render, spec.num_params)[MEL_SPEC_FIELD][1:]
            prefix = f"shard {attempt.shard_id} attempt {attempt.name}: shard-stats.npz"
            if count.shape != () or count.dtype != np.dtype(np.int64):
                raise ValueError(f"{prefix} count must be a scalar int64")
            for name, value in (("mean", mean), ("m2", m2)):
                if value.dtype != np.dtype(np.float32):
                    raise ValueError(f"{prefix} {name} must have dtype float32")
                if value.shape != expected_shape:
                    raise ValueError(
                        f"{prefix} {name} must have shape {expected_shape}, got {value.shape}"
                    )
                if not np.isfinite(value).all():
                    raise ValueError(f"{prefix} {name} must contain only finite values")
            if np.any(m2 < 0):
                raise ValueError(f"{prefix} m2 must be non-negative")
            return int(count), np.array(mean, copy=True), np.array(m2, copy=True)


def _validate_fragment_files(
    spec: DatasetSpec,
    attempt: StagedLanceAttempt,
    fragment: lance.fragment.FragmentMetadata,
) -> None:
    """Validate fragment paths, object presence, and physical Arrow schemas.

    :param spec: Validated dataset spec.
    :param attempt: Staged attempt being checked.
    :param fragment: Deserialized metadata naming the fragment files.
    :raises ValueError: A path escapes the split or a file is absent, empty, or schema-drifted.
    """
    from lance.file import LanceFileReader

    split_uri = spec.r2.split_lance_uri(split_for_shard(spec, attempt.shard_id))
    split_target, storage_options = r2_io.lance_target(split_uri)
    expected_schema = _shard_schema(spec, attempt.shard_id)
    if not fragment.files:
        raise ValueError(
            f"shard {attempt.shard_id} attempt {attempt.name}: fragment has no data files"
        )
    for data_file in fragment.files:
        data_path = PurePosixPath(data_file.path)
        if (
            not data_file.path
            or data_path.is_absolute()
            or ".." in data_path.parts
            or data_path.as_posix() != data_file.path
        ):
            raise ValueError(
                f"shard {attempt.shard_id} attempt {attempt.name}: unsafe fragment data path "
                f"{data_file.path!r}"
            )
        if not r2_io.object_size(f"{split_uri}/data/{data_file.path}"):
            raise ValueError(
                f"shard {attempt.shard_id} attempt {attempt.name}: fragment data file "
                f"{data_file.path} missing or empty under {split_uri}/data/"
            )
        try:
            physical_schema = LanceFileReader(
                f"{split_target}/data/{data_file.path}", storage_options=storage_options
            ).metadata().schema
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"shard {attempt.shard_id} attempt {attempt.name}: fragment data file "
                f"{data_file.path} is not a readable Lance file: {type(exc).__name__}: {exc}"
            ) from exc
        if not physical_schema.equals(expected_schema, check_metadata=True):
            raise ValueError(
                f"shard {attempt.shard_id} attempt {attempt.name}: fragment physical schema "
                "does not match spec-derived shard schema"
            )


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
    fragment = _load_fragment_metadata(spec, attempt)
    welford = _load_welford_state(spec, attempt)
    rows = fragment.physical_rows
    if rows != spec.render.samples_per_shard or welford[0] != rows:
        raise ValueError(
            f"shard {attempt.shard_id} attempt {attempt.name}: fragment has {rows} rows, "
            f"stats count {welford[0]}; spec expects {spec.render.samples_per_shard}"
        )
    _validate_fragment_files(spec, attempt, fragment)
    return CheckedLanceWinner(attempt=attempt, fragment=fragment, welford=welford)


def _shard_schema(spec: DatasetSpec, shard_id: int) -> pa.Schema:
    """Build the exact Arrow schema a worker must write for one shard.

    :param spec: Validated dataset spec.
    :param shard_id: Logical shard whose seed appears in schema metadata.
    :returns: Spec-derived physical Arrow schema for the shard.
    """
    render = spec.render.model_copy(update={"base_seed": spec.shards[shard_id].seed})
    return lance_schema(dataset_field_shapes(render, spec.num_params), render.shard_metadata())


def _split_schema(spec: DatasetSpec, first_shard_id: int) -> pa.Schema:
    """Build a split dataset's Arrow schema from the spec.

    Mirrors the worker writer's construction: shapes come from the render
    config, and ``ShardMetadata`` is seeded by the split's first shard so
    consumers keep reading ``sample_rate`` etc. from splits.

    :param spec: Validated dataset spec.
    :param first_shard_id: The split's first shard, whose seed the metadata carries.
    :returns: Arrow schema for the split's ``Overwrite`` commit.
    """
    return _shard_schema(spec, first_shard_id)


def _recorded_attempt_names(spec: DatasetSpec) -> dict[int, str]:
    """Load prior winner names so a post-card rerun remains monotonic.

    :param spec: Validated dataset spec.
    :returns: Previously selected attempt name keyed by shard id, or an empty mapping.
    :raises ValueError: The existing dataset card is invalid or belongs to another run.
    """
    if r2_io.object_size(spec.r2.dataset_card_uri()) is None:
        return {}
    with r2_io.downloaded_to_tempfile(spec.r2.dataset_card_uri()) as card_path:
        try:
            card = LanceDatasetCard.model_validate_json(card_path.read_bytes())
        except ValidationError as exc:
            raise ValueError(f"invalid existing dataset card: {exc}") from exc
    if card.run_id != spec.run_id:
        raise ValueError(
            f"existing dataset card run_id {card.run_id!r} does not match {spec.run_id!r}"
        )
    return {selected.shard_id: selected.attempt for selected in card.selected_attempts}


def select_checked_winner(
    spec: DatasetSpec,
    candidates: Sequence[StagedLanceAttempt],
    *,
    preferred_name: str | None = None,
) -> CheckedLanceWinner:
    """Select the first healthy candidate using finalize's reconciliation contract.

    Structurally invalid candidates are marked invalid and skipped so validation and finalization
    agree on which attempt can become canonical.

    :param spec: Validated dataset spec.
    :param candidates: Complete staged attempts for one logical shard.
    :param preferred_name: Previously recorded winner to try before timestamp order.
    :returns: The first structurally healthy checked winner.
    :raises ValueError: No candidate passes structural validation.
    """
    ordered = sorted(candidates, key=lambda attempt: (attempt.valid_mtime, attempt.valid_key))
    if preferred_name is not None:
        ordered.sort(key=lambda attempt: attempt.name != preferred_name)
    failures: list[str] = []
    for candidate in ordered:
        try:
            return load_checked_winner(spec, candidate)
        except ValueError as exc:
            failures.append(str(exc))
            invalidate_staged_attempt(spec, candidate.shard_id, candidate.name)
            logger.warning(
                "invalidated_staged_attempt",
                attempt=candidate.name,
                reason=str(exc),
                shard_id=candidate.shard_id,
            )
    shard_id = candidates[0].shard_id if candidates else "unknown"
    raise ValueError(
        f"shard {shard_id} has no healthy staged-valid attempt: " + "; ".join(failures)
    )


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
    recorded = _recorded_attempt_names(spec)
    winners: dict[int, CheckedLanceWinner] = {}
    for shard in spec.shards:
        winners[shard.shard_id] = select_checked_winner(
            spec,
            attempts[shard.shard_id],
            preferred_name=recorded.get(shard.shard_id),
        )
    return winners


def _reduce_and_upload_stats(
    spec: DatasetSpec, winners: dict[int, CheckedLanceWinner], work_dir: Path
) -> None:
    """Reduce the train winners' Welford sidecars into ``stats.npz`` and upload it.

    :param spec: Validated dataset spec.
    :param winners: Checked winner per shard id.
    :param work_dir: Scratch directory the ``stats.npz`` is staged in.
    """
    train_lo, train_hi = spec.split_shard_ranges["train"]
    state: WelfordState = (0, 0, 0)
    for shard_id in range(train_lo, train_hi):
        state = merge_welford(state, winners[shard_id].welford)
    mean, std = finalize_welford(state, mask_degenerate=spec.mask_degenerate_bins)
    stats_npz = work_dir / STATS_NPZ_FILENAME
    np.savez(stats_npz, mean=mean, std=std)
    r2_io.upload(stats_npz, spec.r2.stats_uri())
    logger.info("uploaded_stats", uri=spec.r2.stats_uri())


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
    logger.info("uploaded_dataset_card", uri=spec.r2.dataset_card_uri())


# DOC502: the documented ValueErrors propagate from _select_checked_winners.
def finalize_lance_fragments(spec: DatasetSpec, work_dir: Path) -> None:  # noqa: DOC502
    """Commit staged winner fragments into split datasets; reduce stats; write the card.

    Each split is one replace-semantics ``Overwrite`` commit over the full
    winner set in shard order — a re-run rebuilds the identical manifest
    instead of appending. Zero rows are decoded. Preconditions: generation
    for this run prefix is quiescent and the train split is non-empty. The
    standard workflow enforces the generation barrier, and the entrypoint
    (``finalize_dataset.finalize_lance``) guards the split before delegating.

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
        logger.info("committed_winner_fragments", fragment_count=hi - lo, split=split)

    _reduce_and_upload_stats(spec, winners, work_dir)
    _write_dataset_card(spec, winners, work_dir)
