"""Unit tests for the offline-wandb history decoder's flush-race polling.

The offline writer serializes history records to the ``run-*.wandb`` binary
asynchronously, so a single-shot read can land before the records flush
(observed as a conda-only 0-rows flake). ``read_history_rows(until=...)``
polls until the caller's predicate holds; these tests pin that retry contract
by stubbing the per-attempt scan and freezing the clock so no real datastore
binary or wall-clock wait is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import wandb_offline


def test_until_predicate_retries_until_rows_materialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty early scans are retried — one ``sleep`` per gap — until the predicate holds.

    :param monkeypatch: Replaces the per-attempt scan and counts ``sleep`` so
        the flush lag is simulated without a real binary or wall-clock wait.
    """
    scans = iter([[], [], [{"shard/bytes": "1024"}]])
    scan_count = 0
    sleep_count = 0

    def _scan(_path: Path) -> list[dict[str, str]]:
        nonlocal scan_count
        scan_count += 1
        return next(scans)

    def _sleep(_s: float) -> None:
        nonlocal sleep_count
        sleep_count += 1

    monkeypatch.setattr(wandb_offline, "_scan_history_rows", _scan)
    monkeypatch.setattr(wandb_offline.time, "sleep", _sleep)

    rows = wandb_offline.read_history_rows(
        Path("ignored.wandb"),
        until=lambda scanned: any("shard/bytes" in r for r in scanned),
    )

    assert rows == [{"shard/bytes": "1024"}]
    assert scan_count == 3
    assert sleep_count == 2


def test_until_predicate_polls_to_deadline_then_returns_last_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A never-satisfied predicate keeps polling until the deadline, then returns the last scan.

    Drives a fake monotonic clock that advances one second per reading; with
    ``timeout_s=2.5`` the deadline is crossed only after several polls, so this
    pins the loop's deadline exit (not a first-iteration short-circuit) and
    that the unsatisfied caller still receives rows to assert against.

    :param monkeypatch: Stubs the scan to stay empty, advances a fake clock,
        and freezes ``sleep`` so the timeout is reached without real elapsed
        time.
    """
    ticks = iter(range(100))
    scan_count = 0

    def _scan(_path: Path) -> list[dict[str, str]]:
        nonlocal scan_count
        scan_count += 1
        return []

    monkeypatch.setattr(wandb_offline, "_scan_history_rows", _scan)
    monkeypatch.setattr(wandb_offline.time, "monotonic", lambda: float(next(ticks)))
    monkeypatch.setattr(wandb_offline.time, "sleep", lambda _s: None)

    rows = wandb_offline.read_history_rows(
        Path("ignored.wandb"),
        until=lambda scanned: len(scanned) > 0,
        timeout_s=2.5,
    )

    assert rows == []
    assert scan_count > 1


def test_without_predicate_reads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting ``until`` keeps the single-shot contract — one scan, never sleeps.

    :param monkeypatch: Counts scan invocations and fails ``sleep`` to prove no
        retry loop runs.
    """
    scan_count = 0

    def _scan(_path: Path) -> list[dict[str, str]]:
        nonlocal scan_count
        scan_count += 1
        return []

    monkeypatch.setattr(wandb_offline, "_scan_history_rows", _scan)
    monkeypatch.setattr(
        wandb_offline.time, "sleep", lambda _s: pytest.fail("single-shot read must not sleep")
    )

    rows = wandb_offline.read_history_rows(Path("ignored.wandb"))

    assert rows == []
    assert scan_count == 1
