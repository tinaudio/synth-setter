"""Deployment guard for SkyPilot dataset workers.

Run as ``python -m pipeline.entrypoints.verify_skypilot_env`` ahead of
``generate_dataset`` in the SkyPilot task YAML's ``run:`` block. If the
SkyPilot rank/world env vars are missing or malformed the script exits
non-zero and prints a clear error to stderr — the surrounding ``set -e``
short-circuits before ``generate_dataset`` runs, so a misconfigured launch
fails fast instead of silently letting every worker render every shard.

The env-reading and bounds-checking logic lives in
``pipeline.partitioning.read_rank_world_from_env`` so this script and
``generate_dataset.run`` can't drift on what counts as a valid partition.
"""

from __future__ import annotations

import os
import sys

from loguru import logger

from pipeline.partitioning import read_rank_world_from_env

_RANK_ENV = "SKYPILOT_NODE_RANK"
_WORLD_ENV = "SKYPILOT_NUM_NODES"


def main() -> None:
    """Entrypoint: validate env, exit 1 with stderr message on failure."""
    try:
        read_rank_world_from_env()
    except ValueError as e:
        sys.stderr.write(f"verify_skypilot_env: FAIL: {e}\n")
        raise SystemExit(1) from e
    logger.info(
        f"verify_skypilot_env: OK (rank={os.environ[_RANK_ENV]}, world={os.environ[_WORLD_ENV]})"
    )


if __name__ == "__main__":
    main()
