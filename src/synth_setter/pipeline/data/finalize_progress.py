"""Shared progress-event contract for dataset finalization (#1843).

The standalone ``synth-setter-finalize-dataset`` entrypoint threads an optional
callback through the per-format finalize functions so a W&B run can watch
finalize advance live. This module holds only the event vocabulary and the
None-guarded dispatch helper — deliberately free of ``lance``/``numpy`` imports
so both the cli entrypoint and the lance fragment finalize can import it without
pulling a heavy dependency into module-load time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias

# ``shard_processed`` fires once per source shard the format actually consumes;
# ``artifact_uploaded`` fires once per finalized object landed in R2.
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
