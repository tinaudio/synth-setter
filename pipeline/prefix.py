from __future__ import annotations

from datetime import datetime, timezone
from typing import NewType

DatasetConfigId = NewType("DatasetConfigId", str)
DatasetRunId = NewType("DatasetRunId", str)
R2Prefix = NewType("R2Prefix", str)


def make_dataset_wandb_run_id(
    dataset_config_id: DatasetConfigId,
    timestamp: datetime | None = None,
) -> DatasetRunId:
    """Build a unique run ID from a config ID and a UTC timestamp.

    :param dataset_config_id: The dataset config identifier (e.g. filename stem).
    :param timestamp: Optional UTC datetime; defaults to now.
    :returns: A string like ``<config_id>-<YYYYMMDD>T<HHMMSS>Z``.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware (got naive datetime)")
    offset = timestamp.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError("timestamp must be UTC")
    formatted = timestamp.strftime("%Y%m%dT%H%M%SZ")
    return DatasetRunId(f"{dataset_config_id}-{formatted}")


def make_r2_prefix(
    dataset_config_id: DatasetConfigId, dataset_wandb_run_id: DatasetRunId
) -> R2Prefix:
    """Build the R2 object prefix for a dataset generation run.

    :param dataset_config_id: The dataset config identifier.
    :param dataset_wandb_run_id: The W&B run ID for this generation run.
    :returns: A prefix string like ``data/<config_id>/<run_id>/``.
    """
    return R2Prefix(f"data/{dataset_config_id}/{dataset_wandb_run_id}/")
