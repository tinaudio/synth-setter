"""Types and helpers for dataset-run R2 prefixes and W&B run IDs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import NewType

from synth_setter.run_id import make_wandb_run_id

DatasetConfigId = NewType("DatasetConfigId", str)
DatasetRunId = NewType("DatasetRunId", str)
R2Prefix = NewType("R2Prefix", str)

DEFAULT_R2_PREFIX_ROOT = "data"


def _utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    :returns: A timezone-aware ``datetime`` in UTC.
    """
    return datetime.now(UTC)


def make_dataset_wandb_run_id(
    dataset_config_id: DatasetConfigId | str,
    timestamp: datetime | None = None,
) -> DatasetRunId:
    """Build a dataset run ID via the shared ``{config_id}-{timestamp}`` convention.

    :param dataset_config_id: The dataset config identifier (e.g. filename stem).
    :param timestamp: Optional UTC datetime; defaults to now.
    :returns: A string like ``<config_id>-<YYYYMMDD>T<HHMMSSsss>Z`` where ``sss`` is
        a zero-padded 3-digit millisecond field.
    """
    return DatasetRunId(make_wandb_run_id(dataset_config_id, timestamp or _utc_now()))


def make_r2_prefix(
    dataset_config_id: DatasetConfigId | str,
    dataset_wandb_run_id: DatasetRunId | str,
    prefix_root: str = DEFAULT_R2_PREFIX_ROOT,
) -> R2Prefix:
    """Build the R2 object prefix for a dataset generation run.

    :param dataset_config_id: The dataset config identifier.
    :param dataset_wandb_run_id: The W&B run ID for this generation run.
    :param prefix_root: Root path component (default ``"data"``). Leading/trailing
        slashes are stripped so callers passing ``"data/"`` or ``"/data"`` don't
        produce a double-slashed prefix pointing at a different R2 keyspace.
    :returns: A prefix string like ``<prefix_root>/<config_id>/<run_id>/``.
    :raises ValueError: If ``prefix_root`` is empty after stripping slashes.
    """
    normalized_root = prefix_root.strip("/")
    if not normalized_root:
        raise ValueError(f"prefix_root must not be empty or slash-only (got {prefix_root!r})")
    return R2Prefix(f"{normalized_root}/{dataset_config_id}/{dataset_wandb_run_id}/")


def assert_r2_prefix_matches(
    prefix: R2Prefix | str,
    dataset_config_id: DatasetConfigId | str,
    dataset_wandb_run_id: DatasetRunId | str,
    prefix_root: str = DEFAULT_R2_PREFIX_ROOT,
) -> None:
    """Assert a materialized R2 prefix matches the canonical value for the given IDs.

    A mismatch means the spec's config/run IDs diverge from the prefix it
    carries. Callers decide whether that is fatal: finalize treats it as
    advisory (logs and proceeds, since custom prefixes are legitimate), while
    a strict caller may let the ``ValueError`` propagate.

    :param prefix: The materialized prefix to check (from ``R2Location.prefix``).
    :param dataset_config_id: The dataset config identifier (e.g. ``spec.task_name``).
    :param dataset_wandb_run_id: The W&B run ID (e.g. ``spec.run_id``).
    :param prefix_root: Root path component (default ``"data"``); must match the
        root used when the prefix was originally built.
    :raises ValueError: If ``prefix`` differs from
        ``make_r2_prefix(dataset_config_id, dataset_wandb_run_id, prefix_root)``.
    """
    expected = make_r2_prefix(dataset_config_id, dataset_wandb_run_id, prefix_root)
    if prefix != expected:
        raise ValueError(
            f"R2 prefix mismatch: got {prefix!r}, expected {expected!r} "
            f"(config_id={dataset_config_id!r}, run_id={dataset_wandb_run_id!r}, "
            f"prefix_root={prefix_root!r})"
        )
