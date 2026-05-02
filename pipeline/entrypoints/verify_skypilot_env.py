"""Deployment guard for SkyPilot dataset workers.

Run as ``python -m pipeline.entrypoints.verify_skypilot_env`` ahead of
``generate_dataset`` in the SkyPilot task YAML's ``run:`` block. If the
SkyPilot rank/world env vars are missing or malformed the script exits
non-zero and prints a clear error to stderr — the surrounding ``set -e``
short-circuits before ``generate_dataset`` runs, so a misconfigured launch
fails fast instead of silently letting every worker render every shard.

The bounds check (``0 <= rank < world``) is delegated to
``pipeline.partitioning.validate_rank_world`` so this script and the
partitioner can't drift in their definition of "valid".
"""

from __future__ import annotations

import os
import sys

from loguru import logger

from pipeline.partitioning import validate_rank_world

_RANK_ENV = "SKYPILOT_NODE_RANK"
_WORLD_ENV = "SKYPILOT_NUM_NODES"


def verify_env() -> None:
    """Raise ``ValueError`` if SkyPilot rank/world env vars are missing or invalid."""
    missing = [name for name in (_RANK_ENV, _WORLD_ENV) if name not in os.environ]
    if missing:
        raise ValueError(
            f"missing SkyPilot env vars: {', '.join(missing)} "
            "(this script must run inside a SkyPilot job)"
        )
    rank_raw = os.environ[_RANK_ENV]
    world_raw = os.environ[_WORLD_ENV]
    try:
        rank = int(rank_raw)
    except ValueError as e:
        raise ValueError(f"{_RANK_ENV} is not an integer: {rank_raw!r}") from e
    try:
        world = int(world_raw)
    except ValueError as e:
        raise ValueError(f"{_WORLD_ENV} is not an integer: {world_raw!r}") from e
    validate_rank_world(rank, world)


def main() -> None:
    """Entrypoint: validate env, exit 1 with stderr message on failure."""
    try:
        verify_env()
    except ValueError as e:
        sys.stderr.write(f"verify_skypilot_env: FAIL: {e}\n")
        raise SystemExit(1) from e
    logger.info(
        f"verify_skypilot_env: OK (rank={os.environ[_RANK_ENV]}, world={os.environ[_WORLD_ENV]})"
    )


if __name__ == "__main__":
    main()
