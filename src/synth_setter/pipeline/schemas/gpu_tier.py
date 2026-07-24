"""Cumulative GPU tier classification for SkyPilot compute pools."""

from __future__ import annotations

from collections.abc import Collection
from enum import StrEnum


class GpuTier(StrEnum):
    """Select the maximum GPU class allowed in a compute pool.

    .. attribute :: LOW

        Consumer RTX 30/40 cards only.

    .. attribute :: MID

        Adds workstation cards (A40, RTX A-series/Ada).

    .. attribute :: HIGH

        Adds datacenter cards (L40S, RTX 6000 Ada, H100/H200, B200).

    .. attribute :: ANY

        No filter; every classified SKU is allowed.
    """

    LOW = "low"
    MID = "mid"
    HIGH = "high"
    ANY = "any"


_MINIMUM_TIER_BY_SKU: dict[str, GpuTier] = {
    "A40": GpuTier.MID,
    "B200": GpuTier.HIGH,
    "H100-SXM": GpuTier.HIGH,
    "H200-SXM": GpuTier.HIGH,
    "L40S": GpuTier.HIGH,
    "RTX3070": GpuTier.LOW,
    "RTX3080": GpuTier.LOW,
    "RTX3090": GpuTier.LOW,
    "RTX4000Ada": GpuTier.MID,
    "RTX4090": GpuTier.LOW,
    "RTX6000-Ada": GpuTier.HIGH,
    "RTXA4000": GpuTier.MID,
}

_TIER_RANK: dict[GpuTier, int] = {
    GpuTier.LOW: 0,
    GpuTier.MID: 1,
    GpuTier.HIGH: 2,
    GpuTier.ANY: 3,
}


def allowed_gpu_skus(tier: GpuTier) -> frozenset[str]:
    """Return every classified SKU allowed by a cumulative tier.

    :param tier: Maximum GPU class to allow.
    :returns: Classified accelerator SKUs available at or below ``tier``.
    """
    maximum_rank = _TIER_RANK[tier]
    return frozenset(
        sku
        for sku, minimum_tier in _MINIMUM_TIER_BY_SKU.items()
        if _TIER_RANK[minimum_tier] <= maximum_rank
    )


def filter_gpu_skus(skus: Collection[str], tier: GpuTier) -> frozenset[str]:
    """Filter accelerator SKUs through a tier with anti-drift validation.

    :param skus: Accelerator SKU names from one compute resource alternative.
    :param tier: Maximum GPU class to allow.
    :returns: Input SKUs allowed by ``tier``.
    :raises ValueError: A non-``any`` tier encounters an unclassified SKU.
    """
    if tier is GpuTier.ANY:
        return frozenset(skus)

    unknown = sorted(set(skus) - _MINIMUM_TIER_BY_SKU.keys())
    if unknown:
        raise ValueError(
            f"GPU tier {tier.value!r} has no classification for accelerator "
            f"SKU(s): {', '.join(unknown)}"
        )
    return frozenset(skus) & allowed_gpu_skus(tier)
