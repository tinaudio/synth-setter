"""Static range partitioning for SkyPilot dataset workers.

Each worker computes the contiguous slice of shard IDs it owns from
``(total_shards, rank, world)``. The partition is stable across restarts
(no leases, no claim files), so an N-node launch finishes ~N× faster than
a single-node run with no coordination overhead.

Pure functions only — neither helper reads environment variables. Callers
(``pipeline.entrypoints.generate_dataset``, ``pipeline.entrypoints.verify_skypilot_env``)
are the imperative shell that converts ``SKYPILOT_NODE_RANK`` /
``SKYPILOT_NUM_NODES`` into integers and passes them in. This split keeps
the partitioning logic trivially testable and prevents the "silent default"
smell where a missing env var would map to a single-worker run inside the
helper itself.
"""

from __future__ import annotations


def validate_rank_world(rank: int, world: int) -> None:
    """Raise ``ValueError`` unless ``world >= 1`` and ``0 <= rank < world``.

    Shared by the partitioner and the SkyPilot env verifier so the two surfaces can't drift in
    their definition of "valid".
    """
    if world < 1 or rank < 0 or rank >= world:
        raise ValueError(
            f"invalid rank/world: rank={rank} world={world} "
            "(require world >= 1 and 0 <= rank < world)"
        )


def get_my_shards(total_shards: int, rank: int, world: int) -> range:
    """Return the contiguous range of shard IDs owned by ``rank`` of ``world``.

    When ``total_shards`` does not divide evenly by ``world``, the first
    ``total_shards % world`` workers each get one extra shard so the
    imbalance between any two workers is at most one shard.

    Args:
        total_shards: Total number of shards across the run.
        rank: This worker's index in ``[0, world)``.
        world: Total number of workers.

    Returns:
        A ``range`` over the shard IDs owned by this worker. Empty
        (``len() == 0``) when ``world > total_shards`` and ``rank`` is past
        the last shard.

    Raises:
        ValueError: If ``world < 1``, ``rank < 0``, or ``rank >= world``.
    """
    validate_rank_world(rank, world)
    base, extra = divmod(total_shards, world)
    start = rank * base + min(rank, extra)
    end = start + base + (1 if rank < extra else 0)
    return range(start, end)
