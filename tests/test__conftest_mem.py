"""Tests for the memory-aware xdist worker helpers in tests/conftest.py."""

import io
import os
from collections.abc import Callable
from typing import cast
from unittest.mock import patch

import pytest

from tests.conftest import (
    _available_memory_bytes,
    _memory_aware_worker_count,
    _swap_in_use_bytes,
    pytest_xdist_auto_num_workers,
)

_MEMINFO = "/proc/meminfo"
_V2_MEM = "/sys/fs/cgroup/memory.max"
_V1_MEM = "/sys/fs/cgroup/memory/memory.limit_in_bytes"

_GIB = 1024**3


def _make_open(files: dict[str, str]) -> Callable[..., io.StringIO]:
    """Return an ``open()`` side-effect serving in-memory content for known paths.

    :param files: Mapping of absolute path to the file body it should yield.
    :returns: Callable suitable for use as ``patch("builtins.open", side_effect=...)``.
    """

    def _side(path: str, *_args: object, **_kwargs: object) -> io.StringIO:
        if path in files:
            return io.StringIO(files[path])
        raise OSError(f"not found: {path}")

    return _side


def _meminfo(
    avail_kb: int,
    *,
    swap_total_kb: int | None = None,
    swap_free_kb: int | None = None,
) -> str:
    """Render a minimal ``/proc/meminfo`` body carrying a ``MemAvailable`` line.

    :param avail_kb: Value to report for ``MemAvailable`` (kibibytes).
    :param swap_total_kb: ``SwapTotal`` (kibibytes); the line is omitted when None.
    :param swap_free_kb: ``SwapFree`` (kibibytes); the line is omitted when None.
    :returns: Multi-line meminfo text with realistic surrounding fields.
    """
    body = (
        f"MemTotal:       32000000 kB\nMemAvailable:   {avail_kb} kB\nBuffers:          100 kB\n"
    )
    if swap_total_kb is not None:
        body += f"SwapTotal:      {swap_total_kb} kB\n"
    if swap_free_kb is not None:
        body += f"SwapFree:       {swap_free_kb} kB\n"
    return body


class TestAvailableMemoryBytes:
    """Unit tests for _available_memory_bytes covering the min(host, cgroup) logic."""

    def test_host_only_no_cgroup_limit_returns_host_avail(self) -> None:
        """V2 ``max`` sentinel means no cgroup cap → host MemAvailable wins."""
        files = {_MEMINFO: _meminfo(20 * 1024 * 1024), _V2_MEM: "max\n"}
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _available_memory_bytes() == 20 * _GIB

    def test_cgroup_below_host_returns_cgroup(self) -> None:
        """V2 cgroup limit < host avail → min picks the cgroup limit."""
        files = {_MEMINFO: _meminfo(20 * 1024 * 1024), _V2_MEM: str(4 * _GIB)}
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _available_memory_bytes() == 4 * _GIB

    def test_cgroup_above_host_returns_host(self) -> None:
        """V2 cgroup limit > host avail → min picks the host figure."""
        files = {_MEMINFO: _meminfo(6 * 1024 * 1024), _V2_MEM: str(64 * _GIB)}
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _available_memory_bytes() == 6 * _GIB

    def test_v1_unlimited_sentinel_ignored(self) -> None:
        """V1 huge sentinel (effectively unlimited) is dropped → host avail wins."""
        files = {_MEMINFO: _meminfo(8 * 1024 * 1024), _V1_MEM: "9223372036854771712\n"}
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _available_memory_bytes() == 8 * _GIB

    def test_v1_limit_below_host_returns_v1(self) -> None:
        """V2 absent, V1 limit < host avail → min picks the V1 limit."""
        files = {_MEMINFO: _meminfo(20 * 1024 * 1024), _V1_MEM: str(2 * _GIB)}
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _available_memory_bytes() == 2 * _GIB

    def test_v2_limit_only_no_meminfo_returns_v2(self) -> None:
        """Meminfo unreadable, V2 limit set → the cgroup limit is the sole signal."""
        files = {_V2_MEM: str(3 * _GIB)}
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _available_memory_bytes() == 3 * _GIB

    def test_invalid_v2_falls_back_to_v1(self) -> None:
        """Non-integer V2 token → ValueError caught, falls through to the V1 limit."""
        files = {_V2_MEM: "garbage", _V1_MEM: str(2 * _GIB)}
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _available_memory_bytes() == 2 * _GIB

    def test_v1_zero_limit_treated_as_unlimited(self) -> None:
        """V1 limit of 0 fails the ``0 < limit`` guard → no cgroup signal (None)."""
        files = {_V1_MEM: "0"}
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _available_memory_bytes() is None

    def test_no_signal_returns_none(self) -> None:
        """Neither meminfo nor cgroup readable → no memory signal (None)."""
        with patch("builtins.open", side_effect=_make_open({})):
            assert _available_memory_bytes() is None

    def test_malformed_meminfo_no_cgroup_returns_none(self) -> None:
        """Meminfo missing MemAvailable and no cgroup cap → None, not a crash."""
        files = {_MEMINFO: "MemTotal: 32000000 kB\n", _V2_MEM: "max\n"}
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _available_memory_bytes() is None

    def test_swap_in_use_debited_from_host_avail(self) -> None:
        """20 GiB avail with 5 GiB swapped → host figure debited to 15 GiB."""
        files = {
            _MEMINFO: _meminfo(
                20 * 1024 * 1024, swap_total_kb=8 * 1024 * 1024, swap_free_kb=3 * 1024 * 1024
            ),
            _V2_MEM: "max\n",
        }
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _available_memory_bytes() == 15 * _GIB

    def test_no_swap_used_leaves_host_avail_unchanged(self) -> None:
        """SwapFree == SwapTotal → nothing swapped → host figure unchanged (no-op)."""
        files = {
            _MEMINFO: _meminfo(
                20 * 1024 * 1024, swap_total_kb=8 * 1024 * 1024, swap_free_kb=8 * 1024 * 1024
            ),
            _V2_MEM: "max\n",
        }
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _available_memory_bytes() == 20 * _GIB

    def test_swap_exceeds_avail_floors_host_to_zero(self) -> None:
        """Swap-in-use larger than MemAvailable → host floored to 0, not negative."""
        files = {
            _MEMINFO: _meminfo(
                4 * 1024 * 1024, swap_total_kb=8 * 1024 * 1024, swap_free_kb=1 * 1024 * 1024
            ),
            _V2_MEM: "max\n",
        }
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _available_memory_bytes() == 0

    def test_swap_discount_applies_to_host_term_only(self) -> None:
        """Discount debits the host figure; the cgroup cap is left untouched in the min."""
        files = {
            _MEMINFO: _meminfo(
                20 * 1024 * 1024, swap_total_kb=8 * 1024 * 1024, swap_free_kb=3 * 1024 * 1024
            ),
            _V2_MEM: str(4 * _GIB),
        }
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _available_memory_bytes() == 4 * _GIB

    def test_unreadable_host_avail_with_swap_present_skips_discount(self) -> None:
        """MemAvailable absent but swap present → discount skipped, no crash → None."""
        body = (
            "MemTotal:       32000000 kB\nSwapTotal:      8000000 kB\nSwapFree:       1000000 kB\n"
        )
        with patch("builtins.open", side_effect=_make_open({_MEMINFO: body, _V2_MEM: "max\n"})):
            assert _available_memory_bytes() is None


class TestSwapInUseBytes:
    """Unit tests for _swap_in_use_bytes covering SwapTotal - SwapFree."""

    def test_partial_swap_returns_total_minus_free(self) -> None:
        """SwapTotal 8 GiB, SwapFree 3 GiB → 5 GiB in use."""
        files = {
            _MEMINFO: _meminfo(
                20 * 1024 * 1024, swap_total_kb=8 * 1024 * 1024, swap_free_kb=3 * 1024 * 1024
            )
        }
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _swap_in_use_bytes() == 5 * _GIB

    def test_fully_free_swap_returns_zero(self) -> None:
        """SwapFree == SwapTotal → 0 bytes in use."""
        files = {
            _MEMINFO: _meminfo(
                20 * 1024 * 1024, swap_total_kb=8 * 1024 * 1024, swap_free_kb=8 * 1024 * 1024
            )
        }
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _swap_in_use_bytes() == 0

    def test_missing_swap_fields_returns_none(self) -> None:
        """No SwapTotal/SwapFree lines → None so the caller skips the discount."""
        files = {_MEMINFO: _meminfo(20 * 1024 * 1024)}
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _swap_in_use_bytes() is None

    def test_swaptotal_without_swapfree_returns_none(self) -> None:
        """Only SwapTotal present → None, not a partial figure."""
        files = {_MEMINFO: _meminfo(20 * 1024 * 1024, swap_total_kb=8 * 1024 * 1024)}
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _swap_in_use_bytes() is None

    def test_swapfree_without_swaptotal_returns_none(self) -> None:
        """Only SwapFree present → None (the symmetric half of the both-required guard)."""
        files = {_MEMINFO: _meminfo(20 * 1024 * 1024, swap_free_kb=3 * 1024 * 1024)}
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _swap_in_use_bytes() is None

    def test_free_exceeding_total_floors_to_zero(self) -> None:
        """SwapFree > SwapTotal (transient kernel skew) → clamped to 0, never negative."""
        files = {
            _MEMINFO: _meminfo(
                20 * 1024 * 1024, swap_total_kb=2 * 1024 * 1024, swap_free_kb=3 * 1024 * 1024
            )
        }
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _swap_in_use_bytes() == 0

    def test_malformed_swap_line_returns_none(self) -> None:
        """Non-integer SwapTotal value → ValueError caught → None, not a crash."""
        body = "MemAvailable:   20000000 kB\nSwapTotal:      garbage kB\nSwapFree:       0 kB\n"
        with patch("builtins.open", side_effect=_make_open({_MEMINFO: body})):
            assert _swap_in_use_bytes() is None

    def test_valueless_swap_line_returns_none(self) -> None:
        """SwapTotal line with no value → IndexError caught → None, not a crash."""
        body = "SwapTotal:\nSwapFree:       0 kB\n"
        with patch("builtins.open", side_effect=_make_open({_MEMINFO: body})):
            assert _swap_in_use_bytes() is None

    def test_unreadable_meminfo_returns_none(self) -> None:
        """/proc/meminfo unreadable → OSError caught → None, not a crash."""
        with patch("builtins.open", side_effect=_make_open({})):
            assert _swap_in_use_bytes() is None


class TestMemoryAwareWorkerCount:
    """Unit tests for _memory_aware_worker_count covering budget division."""

    def test_divides_avail_by_default_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """20 GiB avail / 2 GiB default budget → 10 workers.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.delenv("PYTEST_XDIST_WORKER_MEM_MB", raising=False)
        files = {_MEMINFO: _meminfo(20 * 1024 * 1024), _V2_MEM: "max\n"}
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _memory_aware_worker_count() == 10

    def test_swap_pressure_reduces_worker_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """20 GiB avail but 10 GiB swapped → 10 GiB / 2 GiB = 5 workers, not 10.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.delenv("PYTEST_XDIST_WORKER_MEM_MB", raising=False)
        files = {
            _MEMINFO: _meminfo(20 * 1024 * 1024, swap_total_kb=10 * 1024 * 1024, swap_free_kb=0),
            _V2_MEM: "max\n",
        }
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _memory_aware_worker_count() == 5

    def test_fractional_floors_to_at_least_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """1.5 GiB avail / 2 GiB budget → 0.75 floors to 0, then lifted to 1.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.delenv("PYTEST_XDIST_WORKER_MEM_MB", raising=False)
        files = {_MEMINFO: _meminfo(int(1.5 * 1024 * 1024)), _V2_MEM: "max\n"}
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _memory_aware_worker_count() == 1

    def test_env_budget_override_changes_divisor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PYTEST_XDIST_WORKER_MEM_MB=4096 → 16 GiB / 4 GiB = 4 workers.

        The override is deliberately distinct from the 2 GiB default so the test fails if the env
        var is ignored and the default divisor is used instead.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setenv("PYTEST_XDIST_WORKER_MEM_MB", "4096")
        files = {_MEMINFO: _meminfo(16 * 1024 * 1024), _V2_MEM: "max\n"}
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _memory_aware_worker_count() == 4

    def test_invalid_env_budget_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-integer budget env → default 2 GiB divisor, not a crash.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setenv("PYTEST_XDIST_WORKER_MEM_MB", "not-a-number")
        files = {_MEMINFO: _meminfo(8 * 1024 * 1024), _V2_MEM: "max\n"}
        with patch("builtins.open", side_effect=_make_open(files)):
            assert _memory_aware_worker_count() == 4

    def test_no_memory_signal_returns_none(self) -> None:
        """No readable memory signal → None so the caller falls back to CPU count."""
        with patch("builtins.open", side_effect=_make_open({})):
            assert _memory_aware_worker_count() is None


class TestHookCombinesCpuAndMemory:
    """The hook returns min(cpu, memory), with the env var as a hard override."""

    def test_memory_clamps_below_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Plenty of CPUs but tight memory → memory count wins the min.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)
        monkeypatch.delenv("PYTEST_XDIST_WORKER_MEM_MB", raising=False)
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(32)), raising=False)
        files = {_MEMINFO: _meminfo(4 * 1024 * 1024), _V2_MEM: "max\n"}
        with patch("builtins.open", side_effect=_make_open(files)):
            assert pytest_xdist_auto_num_workers(cast("pytest.Config", None)) == 2

    def test_cpu_clamps_below_memory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Plenty of memory but few CPUs → cpu count wins the min.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)
        monkeypatch.delenv("PYTEST_XDIST_WORKER_MEM_MB", raising=False)
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1}, raising=False)
        files = {_MEMINFO: _meminfo(64 * 1024 * 1024), _V2_MEM: "max\n"}
        with patch("builtins.open", side_effect=_make_open(files)):
            assert pytest_xdist_auto_num_workers(cast("pytest.Config", None)) == 2

    def test_no_memory_signal_uses_cpu_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No memory signal → hook falls back to the CPU count alone.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2}, raising=False)
        with patch("builtins.open", side_effect=_make_open({})):
            assert pytest_xdist_auto_num_workers(cast("pytest.Config", None)) == 3

    def test_env_override_wins_over_clamps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit PYTEST_XDIST_AUTO_NUM_WORKERS short-circuits both clamps.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", "7")
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1}, raising=False)
        files = {_MEMINFO: _meminfo(1 * 1024 * 1024), _V2_MEM: "max\n"}
        with patch("builtins.open", side_effect=_make_open(files)):
            assert pytest_xdist_auto_num_workers(cast("pytest.Config", None)) == 7

    def test_env_override_zero_floors_to_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 0/negative env override floors to 1 so xdist never gets ``-n 0``.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", "0")
        assert pytest_xdist_auto_num_workers(cast("pytest.Config", None)) == 1

    def test_env_override_non_integer_falls_back_to_clamps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-integer env override is ignored, not fatal — clamps still apply.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", "auto")
        monkeypatch.delenv("PYTEST_XDIST_WORKER_MEM_MB", raising=False)
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2}, raising=False)
        with patch("builtins.open", side_effect=_make_open({})):
            assert pytest_xdist_auto_num_workers(cast("pytest.Config", None)) == 3

    def test_env_override_empty_string_falls_back_to_clamps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty env override means "unset" → fall through to the clamps.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", "")
        monkeypatch.delenv("PYTEST_XDIST_WORKER_MEM_MB", raising=False)
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3}, raising=False)
        with patch("builtins.open", side_effect=_make_open({})):
            assert pytest_xdist_auto_num_workers(cast("pytest.Config", None)) == 4
