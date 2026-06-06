"""Canonical W&B run-id convention, shared across data, training, and eval runs.

Stdlib-only so the launcher-pure ``pipeline.schemas`` layer can import it without
pulling ``omegaconf``/``hydra`` (see ``test_bare_spec_import_does_not_pull_omegaconf``).
"""

from __future__ import annotations

from datetime import datetime, timezone


def make_wandb_run_id(config_id: str, timestamp: datetime | None = None) -> str:
    """Build the canonical ``{config_id}-{timestamp}`` W&B run id.

    One source of truth for the run-id format in storage-provenance-spec.md §1, so
    data-generation, training, and eval runs stay reproducible and lineage-linkable.

    :param config_id: Config identifier (e.g. dataset config stem or experiment name).
    :param timestamp: UTC, timezone-aware datetime; defaults to now.
    :returns: ``<config_id>-<YYYYMMDD>T<HHMMSSsss>Z`` with a zero-padded 3-digit
        millisecond field.
    :raises ValueError: If ``timestamp`` is naive or not UTC.
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
    return f"{config_id}-{formatted}"
