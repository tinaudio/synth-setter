from __future__ import annotations

from datetime import datetime, timezone
from typing import NewType

DatasetConfigId = NewType("DatasetConfigId", str)
DatasetRunId = NewType("DatasetRunId", str)
R2Prefix = NewType("R2Prefix", str)

DEFAULT_R2_PREFIX_ROOT = "data"


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
        timestamp = datetime.now(timezone.utc)
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
    :param prefix_root: Root path component (default ``"data"``).
    :returns: A prefix string like ``<prefix_root>/<config_id>/<run_id>/``.
    """
    return R2Prefix(f"{prefix_root}/{dataset_config_id}/{dataset_wandb_run_id}/")
