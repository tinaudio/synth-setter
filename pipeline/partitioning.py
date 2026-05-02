"""Static range partitioning for SkyPilot dataset workers.

Each worker computes the contiguous slice of shard IDs it owns from
``(total_shards, rank, world)``. The partition is stable across restarts
(no leases, no claim files), so an N-node launch finishes ~N× faster than
a single-node run with no coordination overhead.

``get_my_shards`` and ``validate_rank_world`` are pure — they don't read
the environment. ``read_rank_world_from_env`` is the imperative shell
that pulls ``OVERRIDE_SKYPILOT_NODE_RANK`` / ``OVERRIDE_SKYPILOT_NUM_NODES``
from ``os.environ`` and fails loudly if they're missing or invalid;
``generate_dataset.run`` calls it before any R2 work so a worker
without partition env can't silently default to a single-worker
partition that would make every node render every shard.
"""

from __future__ import annotations

import os

# OVERRIDE_-prefixed because SkyPilot itself reserves SKYPILOT_NODE_RANK and
# overrides our injection at runtime to the cluster-native value (0 on a
# single-node cluster). We launch N independent single-node clusters per
# fan-out (RunPod's backend doesn't support num_nodes>1), so the launcher
# injects synthetic rank/world under non-reserved names that SkyPilot
# leaves alone. The worker reads these and partitions accordingly.
RANK_ENV_VAR = "OVERRIDE_SKYPILOT_NODE_RANK"
WORLD_ENV_VAR = "OVERRIDE_SKYPILOT_NUM_NODES"


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


def read_rank_world_from_env() -> tuple[int, int]:
    """Read OVERRIDE_SKYPILOT_NODE_RANK / OVERRIDE_SKYPILOT_NUM_NODES from the environment.

    No defaults — if either env var is missing, malformed, or out-of-bounds,
    raise ``ValueError`` with a message naming the offending var(s). The
    silent-default behavior (rank=0/world=1 on missing env) is intentionally
    refused: a worker invoked with a multi-shard spec but no partition env
    would otherwise duplicate every shard across every node, which silently
    burns rendering work at multi-worker scale.

    Returns:
        ``(rank, world)`` as integers, validated against ``validate_rank_world``.

    Raises:
        ValueError: If either env var is missing, can't parse as int, or
            fails the rank/world bounds check.
    """
    missing = [name for name in (RANK_ENV_VAR, WORLD_ENV_VAR) if name not in os.environ]
    if missing:
        raise ValueError(f"missing SkyPilot env vars: {', '.join(missing)}")
    rank_raw = os.environ[RANK_ENV_VAR]
    world_raw = os.environ[WORLD_ENV_VAR]
    try:
        rank = int(rank_raw)
    except ValueError as e:
        raise ValueError(f"{RANK_ENV_VAR} is not an integer: {rank_raw!r}") from e
    try:
        world = int(world_raw)
    except ValueError as e:
        raise ValueError(f"{WORLD_ENV_VAR} is not an integer: {world_raw!r}") from e
    validate_rank_world(rank, world)
    return rank, world


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
