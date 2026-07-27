"""Dependency-light progress-event contract for dataset finalization (#1843).

``shard_processed`` marks a selected Lance winner that passed structural
validation. ``artifact_uploaded`` marks a finalized object landed in R2. These
are progress signals; ``dataset.complete`` remains the completion authority.

Typical usage::

    events: list[FinalizeProgressEvent] = []
    report_finalize_progress(events.append, "shard_processed")
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias

FinalizeProgressEvent: TypeAlias = Literal["shard_processed", "artifact_uploaded"]
FinalizeProgressCallback: TypeAlias = Callable[[FinalizeProgressEvent], None]


def report_finalize_progress(
    callback: FinalizeProgressCallback | None, event: FinalizeProgressEvent
) -> None:
    """Forward one completed finalization event to ``callback`` when present.

    :param callback: Optional progress sink supplied by the standalone entrypoint;
        ``None`` for every wandb-free caller (inline finalize, most tests).
    :param event: Event reported only after its shard fold or artifact upload completes.
    """
    if callback is not None:
        callback(event)
