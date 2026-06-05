"""Types and helpers for dataset-run R2 prefixes and W&B run IDs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import NewType

DatasetConfigId = NewType("DatasetConfigId", str)
DatasetRunId = NewType("DatasetRunId", str)
R2Prefix = NewType("R2Prefix", str)

DEFAULT_R2_PREFIX_ROOT = "data"


def _utcnow() -> datetime:
    """Return the current UTC time as a seam tests can patch.

    :returns: A timezone-aware ``datetime`` in UTC.
    """
    return datetime.now(timezone.utc)


def make_dataset_wandb_run_id(
    dataset_config_id: DatasetConfigId | str,
    timestamp: datetime | None = None,
) -> DatasetRunId:
    """Build a unique run ID from a config ID and a UTC timestamp.

    :param dataset_config_id: The dataset config identifier (e.g. filename stem).
    :param timestamp: Optional UTC datetime; defaults to now.
    :returns: A string like ``<config_id>-<YYYYMMDD>T<HHMMSSsss>Z`` where ``sss`` is
        a zero-padded 3-digit millisecond field.
    """
    if timestamp is None:
        timestamp = _utcnow()
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware (got naive datetime)")
    offset = timestamp.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError("timestamp must be UTC")
    millis = timestamp.microsecond // 1000
    formatted = timestamp.strftime("%Y%m%dT%H%M%S") + f"{millis:03d}Z"
    return DatasetRunId(f"{dataset_config_id}-{formatted}")


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
