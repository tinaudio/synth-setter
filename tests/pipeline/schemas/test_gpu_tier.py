"""Tests for cumulative SkyPilot GPU tier classification."""

from __future__ import annotations

import pytest

from synth_setter.pipeline.schemas.gpu_tier import (
    GpuTier,
    allowed_gpu_skus,
    filter_gpu_skus,
)


@pytest.mark.parametrize(
    ("tier", "expected"),
    [
        (
            GpuTier.LOW,
            {"RTX3070", "RTX3080", "RTX3090", "RTX4090"},
        ),
        (
            GpuTier.MID,
            {
                "A40",
                "RTX3070",
                "RTX3080",
                "RTX3090",
                "RTX4000Ada",
                "RTX4090",
                "RTXA4000",
            },
        ),
        (
            GpuTier.HIGH,
            {
                "A40",
                "B200",
                "H100-SXM",
                "H200-SXM",
                "L40S",
                "RTX3070",
                "RTX3080",
                "RTX3090",
                "RTX4000Ada",
                "RTX4090",
                "RTX6000-Ada",
                "RTXA4000",
            },
        ),
        (
            GpuTier.ANY,
            {
                "A40",
                "B200",
                "H100-SXM",
                "H200-SXM",
                "L40S",
                "RTX3070",
                "RTX3080",
                "RTX3090",
                "RTX4000Ada",
                "RTX4090",
                "RTX6000-Ada",
                "RTXA4000",
            },
        ),
    ],
)
def test_allowed_gpu_skus_each_tier_returns_expected_set(
    tier: GpuTier, expected: set[str]
) -> None:
    """Each tier returns its cumulative set of classified accelerator SKUs.

    :param tier: Tier whose allowed set is under test.
    :param expected: Exact cumulative SKU set for the tier.
    """
    assert allowed_gpu_skus(tier) == expected


def test_allowed_gpu_skus_ordered_tiers_are_cumulatively_nested() -> None:
    """Ordered tiers form strict cumulative supersets."""
    assert allowed_gpu_skus(GpuTier.LOW) < allowed_gpu_skus(GpuTier.MID)
    assert allowed_gpu_skus(GpuTier.MID) < allowed_gpu_skus(GpuTier.HIGH)
    assert allowed_gpu_skus(GpuTier.HIGH) == allowed_gpu_skus(GpuTier.ANY)


def test_filter_gpu_skus_unknown_sku_for_non_any_tier_raises() -> None:
    """A non-passthrough tier rejects unclassified pool accelerators."""
    with pytest.raises(ValueError, match="FutureGPU"):
        filter_gpu_skus({"RTX3090", "FutureGPU"}, GpuTier.LOW)
