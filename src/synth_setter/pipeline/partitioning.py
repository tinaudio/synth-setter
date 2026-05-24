"""Static range partitioning for SkyPilot dataset workers.

Each worker owns a contiguous, deterministic slice of shard IDs computed from
``(total_shards, rank, world)``. Pure helpers (``get_my_shards``,
``validate_rank_world``) and an env-reading shell (``read_rank_world_from_env``)
that defaults to single-worker local mode when both partition env vars are
absent, and rejects any partial / malformed config — see #763.

Env vars are ``SYNTH_SETTER_``-prefixed to dodge SkyPilot's reserved
``SKYPILOT_NODE_RANK`` (clobbered to 0 per single-node cluster) and to avoid
generic ``WORKER_RANK`` / ``NUM_WORKERS`` collisions with multiprocessing
toolkits.
"""

from __future__ import annotations

import os

WORKER_RANK_ENV_VAR = "SYNTH_SETTER_WORKER_RANK"
NUM_WORKERS_ENV_VAR = "SYNTH_SETTER_NUM_WORKERS"


def available_cpus() -> int:
    """Return the count of CPUs usable by the current process.

    Linux respects ``taskset`` and cgroup pinning via ``os.sched_getaffinity``;
    other platforms fall back to ``os.cpu_count() or 1``.

    :returns: Number of CPUs usable by this process; always ``>= 1``.
    """
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        return len(sched_getaffinity(0))
    return os.cpu_count() or 1


def validate_rank_world(rank: int, world: int) -> None:
    """Raise ``ValueError`` unless ``world >= 1`` and ``0 <= rank < world``."""
    if world < 1 or rank < 0 or rank >= world:
        raise ValueError(
            f"invalid rank/world: rank={rank} world={world} "
            "(require world >= 1 and 0 <= rank < world)"
        )


def read_rank_world_from_env() -> tuple[int, int]:
    """Read SYNTH_SETTER_WORKER_RANK / SYNTH_SETTER_NUM_WORKERS as ints.

    With both env vars absent, returns ``(0, 1)`` — the local single-worker
    default that makes ``generate_dataset`` usable without manually exporting
    partition env. A partial config (only one set) still raises: that pattern
    almost always means a launcher dropped half its env injection, and silently
    treating it as single-worker would duplicate every shard across every
    node — see #763.

    :return: ``(rank, world)`` as integers, validated against ``validate_rank_world``.
    :raises ValueError: If exactly one env var is set, either can't parse as int,
        or the resulting rank/world fails the bounds check.
    """
    rank_present = WORKER_RANK_ENV_VAR in os.environ
    world_present = NUM_WORKERS_ENV_VAR in os.environ
    if not rank_present and not world_present:
        return 0, 1
    if rank_present != world_present:
        missing = WORKER_RANK_ENV_VAR if not rank_present else NUM_WORKERS_ENV_VAR
        present = NUM_WORKERS_ENV_VAR if not rank_present else WORKER_RANK_ENV_VAR
        raise ValueError(
            f"partial partition env: {present} set but {missing} missing "
            "(set both or neither — neither defaults to single-worker local mode)"
        )
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

    :param total_shards: Total number of shards across the run. Must be ``>= 0``.
    :param rank: This worker's index in ``[0, world)``.
    :param world: Total number of workers.
    :return: A ``range`` over the shard IDs owned by this worker. Empty (``len() == 0``)
        when ``world > total_shards`` and ``rank`` is past the last shard.
    :raises ValueError: If ``total_shards < 0``, ``world < 1``, ``rank < 0``, or ``rank >= world``.
    """
    if total_shards < 0:
        raise ValueError(f"total_shards must be >= 0, got {total_shards}")
    validate_rank_world(rank, world)
    base, extra = divmod(total_shards, world)
    start = rank * base + min(rank, extra)
    end = start + base + (1 if rank < extra else 0)
    return range(start, end)
