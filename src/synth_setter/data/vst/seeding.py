"""Deterministic per-sample seed derivation for reproducible datagen (#884).

Every sample's RNG is a pure function of ``(master_seed, sample_idx, attempt)``,
so sample ``N`` gets the same seed regardless of which worker renders it, the
order workers process rows in, whether earlier rows were retried, or how the run
is sharded. A single advancing global RNG cannot give this — its state depends on
consumption order. The derivation (SHA-256 of the three inputs) is load-bearing:
changing it silently invalidates every existing dataset, so it is pinned by a
golden-value test.

The accepted ``attempt`` is deterministic given a fixed ``min_loudness``, so
cross-run reproducibility holds without persisting it — a per-row audit trail in
the artifact is a #884 follow-up.
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

    :param master_seed: Per-shard master seed (``ShardSpec.seed``).
    :param sample_idx: Absolute row index within the shard.
    :param attempt: Retry attempt for the row.
    :returns: A fresh generator deterministic in the three inputs.
    """
    return np.random.default_rng(seed_for_sample(master_seed, sample_idx, attempt))
