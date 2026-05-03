"""Static range partitioning for SkyPilot dataset workers.

Each worker owns a contiguous, deterministic slice of shard IDs computed from
``(total_shards, rank, world)``. Pure helpers (``get_my_shards``,
``validate_rank_world``) and an env-reading shell (``read_rank_world_from_env``)
that fails loudly on missing partition env — see #763.

Env vars are ``SYNTH_SETTER_``-prefixed to dodge SkyPilot's reserved
``SKYPILOT_NODE_RANK`` (clobbered to 0 per single-node cluster) and to avoid
generic ``WORKER_RANK`` / ``NUM_WORKERS`` collisions with multiprocessing
toolkits.
"""

from __future__ import annotations

import os

WORKER_RANK_ENV_VAR = "SYNTH_SETTER_WORKER_RANK"
NUM_WORKERS_ENV_VAR = "SYNTH_SETTER_NUM_WORKERS"


def validate_rank_world(rank: int, world: int) -> None:
    """Raise ``ValueError`` unless ``world >= 1`` and ``0 <= rank < world``."""
    if world < 1 or rank < 0 or rank >= world:
        raise ValueError(
            f"invalid rank/world: rank={rank} world={world} "
            "(require world >= 1 and 0 <= rank < world)"
        )


def read_rank_world_from_env() -> tuple[int, int]:
    """Read SYNTH_SETTER_WORKER_RANK / SYNTH_SETTER_NUM_WORKERS as ints, no defaults.

    Silent defaults are refused so a misconfigured worker can't duplicate every
    shard across every node — see #763.

    Returns:
        ``(rank, world)`` as integers, validated against ``validate_rank_world``.

    Raises:
        ValueError: If either env var is missing, can't parse as int, or fails the
            rank/world bounds check.
    """
    missing = [
        name for name in (WORKER_RANK_ENV_VAR, NUM_WORKERS_ENV_VAR) if name not in os.environ
    ]
    if missing:
        raise ValueError(f"missing partition env vars: {', '.join(missing)}")
    rank_raw = os.environ[WORKER_RANK_ENV_VAR]
    world_raw = os.environ[NUM_WORKERS_ENV_VAR]
    try:
        rank = int(rank_raw)
    except ValueError as e:
        raise ValueError(f"{WORKER_RANK_ENV_VAR} is not an integer: {rank_raw!r}") from e
    try:
        world = int(world_raw)
    except ValueError as e:
        raise ValueError(f"{NUM_WORKERS_ENV_VAR} is not an integer: {world_raw!r}") from e
    validate_rank_world(rank, world)
    return rank, world


def get_my_shards(total_shards: int, rank: int, world: int) -> range:
    """Contiguous shard-ID range owned by ``rank`` of ``world``; balanced ±1 on uneven splits.

    Args:
        total_shards: Total number of shards across the run. Must be ``>= 0``.
        rank: This worker's index in ``[0, world)``.
        world: Total number of workers.

    Returns:
        A ``range`` over the shard IDs owned by this worker. Empty (``len() == 0``)
        when ``world > total_shards`` and ``rank`` is past the last shard.

    Raises:
        ValueError: If ``total_shards < 0``, ``world < 1``, ``rank < 0``, or ``rank >= world``.
    """
    if total_shards < 0:
        raise ValueError(f"total_shards must be >= 0, got {total_shards}")
    validate_rank_world(rank, world)
    base, extra = divmod(total_shards, world)
    start = rank * base + min(rank, extra)
    end = start + base + (1 if rank < extra else 0)
    return range(start, end)
