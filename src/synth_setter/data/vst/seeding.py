"""Deterministic per-sample seed derivation for reproducible datagen (#884).

Each row seed is a pure SHA-256 function of ``(master_seed, sample_idx, attempt)``,
so worker count, row order, and sharding cannot perturb parameter draws. The
encoding is golden-pinned because changing it reseeds existing datasets. The
accepted attempt is deterministic for a fixed ``min_loudness``; per-row attempt
provenance remains a follow-up.
"""

import hashlib

import numpy as np

_SEED_BYTES = 8  # 64-bit seed; np.random.default_rng accepts [0, 2**64).


def seed_for_sample(master_seed: int, sample_idx: int, attempt: int = 0) -> int:
    """Derive a stable 64-bit seed from ``(master_seed, sample_idx, attempt)``.

    :param master_seed: Per-shard master seed (``ShardSpec.seed``).
    :param sample_idx: Absolute row index within the shard.
    :param attempt: Retry attempt, folded into the seed so retries stay deterministic.
    :returns: A seed in ``[0, 2**64)``.
    """
    # Wire format: colon-separated decimal integers, hashed; the golden-value test
    # pins this exact encoding (changing it reseeds every existing dataset).
    digest = hashlib.sha256(f"{master_seed}:{sample_idx}:{attempt}".encode()).digest()
    return int.from_bytes(digest[:_SEED_BYTES], "big")


def rng_for_sample(master_seed: int, sample_idx: int, attempt: int = 0) -> np.random.Generator:
    """Build a ``numpy`` ``Generator`` seeded by :func:`seed_for_sample`.

    :param master_seed: Per-shard master seed.
    :param sample_idx: Absolute row index within the shard.
    :param attempt: Retry attempt for the row.
    :returns: A fresh generator deterministic in ``(master_seed, sample_idx, attempt)``.
    """
    return np.random.default_rng(seed_for_sample(master_seed, sample_idx, attempt))
