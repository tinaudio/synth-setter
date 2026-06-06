"""Tests for the _cgroup_aware_cpu_count helper in tests/conftest.py."""

import io
import os
from collections.abc import Callable
from unittest.mock import patch

import pytest

from tests.conftest import _cgroup_aware_cpu_count

_V2_PATH = "/sys/fs/cgroup/cpu.max"
_V1_QUOTA_PATH = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
_V1_PERIOD_PATH = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"


def _open_v2(quota: int, period: int) -> Callable[..., io.StringIO]:
    """Return an ``open()`` side-effect that serves one cgroup v2 ``cpu.max`` file.

    :param quota: Raw CPU quota microseconds to embed in the fake file content.
    :param period: Raw CPU period microseconds to embed in the fake file content.
    :returns: Callable suitable for use as ``patch("builtins.open", side_effect=...)``.
    """

    def _side(path: str, *_args: object, **_kwargs: object) -> io.StringIO:
        if path == _V2_PATH:
            return io.StringIO(f"{quota} {period}")
        raise OSError(f"not found: {path}")

    return _side


def _open_v1(quota_us: int, period_us: int) -> Callable[..., io.StringIO]:
    """Return an ``open()`` side-effect that rejects cgroup v2 and serves v1 files.

    :param quota_us: ``cfs_quota_us`` value to write into the fake v1 quota file.
    :param period_us: ``cfs_period_us`` value to write into the fake v1 period file.
    :returns: Callable suitable for use as ``patch("builtins.open", side_effect=...)``.
    """

    def _side(path: str, *_args: object, **_kwargs: object) -> io.StringIO:
        if path == _V2_PATH:
            raise OSError("no v2")
        if path == _V1_QUOTA_PATH:
            return io.StringIO(str(quota_us))
        if path == _V1_PERIOD_PATH:
            return io.StringIO(str(period_us))
        raise OSError(f"not found: {path}")

    return _side


def _open_none(path: str) -> io.StringIO:
    """Simulate a host with no cgroup filesystem at all.

    :param path: The path that was requested (included in the error message).
    :raises OSError: Always — no cgroup files exist on this simulated host.
    :returns: Never returns; always raises.
    """
    raise OSError(f"not found: {path}")


class TestCgroupAwareCpuCount:
    """Unit tests for _cgroup_aware_cpu_count covering every branch."""

    def test_no_cgroup_returns_affinity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No cgroup files → falls back to affinity count.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
        with patch("builtins.open", side_effect=_open_none):
            assert _cgroup_aware_cpu_count() == 4

    def test_cgroupv2_quota_clamps_below_affinity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """V2 quota < affinity → returns quota (200ms/100ms = 2 CPUs).

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(8)))
        with patch("builtins.open", side_effect=_open_v2(200_000, 100_000)):
            assert _cgroup_aware_cpu_count() == 2

    def test_cgroupv2_quota_above_affinity_uses_affinity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """V2 quota > affinity → returns affinity (min wins).

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1})
        with patch("builtins.open", side_effect=_open_v2(800_000, 100_000)):
            assert _cgroup_aware_cpu_count() == 2

    def test_cgroupv2_max_sentinel_uses_affinity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """V2 ``max <period>`` means no quota → uses affinity.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2})

        def _max_open(path: str, *_a: object, **_k: object) -> io.StringIO:
            if path == _V2_PATH:
                return io.StringIO("max 100000")
            raise OSError(path)

        with patch("builtins.open", side_effect=_max_open):
            assert _cgroup_aware_cpu_count() == 3

    def test_cgroupv1_quota_clamps_below_affinity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """V1 quota_us / period_us = 1 CPU < 4 affinity → returns 1.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(4)))
        with patch("builtins.open", side_effect=_open_v1(100_000, 100_000)):
            assert _cgroup_aware_cpu_count() == 1

    def test_cgroupv1_negative_quota_uses_affinity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """V1 quota_us = -1 (unlimited sentinel) → uses affinity.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2})
        with patch("builtins.open", side_effect=_open_v1(-1, 100_000)):
            assert _cgroup_aware_cpu_count() == 3

    def test_fractional_quota_floors_to_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Quota of 0.5 CPUs → max(1, int(0.5)) = 1, not 0.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(4)))
        with patch("builtins.open", side_effect=_open_v2(50_000, 100_000)):
            assert _cgroup_aware_cpu_count() == 1

    def test_malformed_cgroupv2_empty_uses_affinity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty v2 file → len(parts) < 2 guard skips quota, uses affinity.

        An empty ``cpu.max`` has no usable quota token; the ``len(parts) >= 2``
        guard treats it as no-limit rather than raising ``IndexError``.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(8)))

        def _empty_v2(path: str, *_a: object, **_k: object) -> io.StringIO:
            if path == _V2_PATH:
                return io.StringIO("")
            raise OSError(path)

        with patch("builtins.open", side_effect=_empty_v2):
            assert _cgroup_aware_cpu_count() == 8

    def test_invalid_token_cgroupv2_falls_back_to_v1(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-integer v2 quota token → ValueError caught, falls through to v1.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(8)))

        def _bad_v2_good_v1(path: str, *_a: object, **_k: object) -> io.StringIO:
            if path == _V2_PATH:
                return io.StringIO("broken 100000")  # int("broken") → ValueError
            if path == _V1_QUOTA_PATH:
                return io.StringIO("200000")
            if path == _V1_PERIOD_PATH:
                return io.StringIO("100000")
            raise OSError(path)

        with patch("builtins.open", side_effect=_bad_v2_good_v1):
            assert _cgroup_aware_cpu_count() == 2

    def test_malformed_cgroupv1_falls_back_to_affinity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Malformed v1 file (non-integer) → ValueError caught, uses affinity.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2})

        def _bad_all(path: str, *_a: object, **_k: object) -> io.StringIO:
            if path == _V2_PATH:
                raise OSError("no v2")
            if path == _V1_QUOTA_PATH:
                return io.StringIO("not-a-number")
            raise OSError(path)

        with patch("builtins.open", side_effect=_bad_all):
            assert _cgroup_aware_cpu_count() == 3
