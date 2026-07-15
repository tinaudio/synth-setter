"""Shared progress-event contract for dataset finalization (#1843).

The standalone ``synth-setter-finalize-dataset`` entrypoint threads an optional
callback through the per-format finalize functions so a W&B run can watch
finalize advance live. This module holds only the event vocabulary and the
None-guarded dispatch helper — deliberately free of ``lance``/``numpy`` imports
so both the cli entrypoint and the lance fragment finalize can import it without
pulling a heavy dependency into module-load time.

``shard_processed`` marks one source shard's per-shard finalization work done;
that unit is format-specific — a folded Welford pass (wds), a downloaded shard
awaiting the bulk reshard (hdf5), or a validated winner (lance) — so it is a
progress signal, never a completion guarantee (the ``dataset.complete`` marker
is authoritative). ``artifact_uploaded`` marks one finalized object landed in R2.

Usage::

    def on_event(event: FinalizeProgressEvent) -> None:
        wandb.log({event: 1})

    report_finalize_progress(on_event, "shard_processed")
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
