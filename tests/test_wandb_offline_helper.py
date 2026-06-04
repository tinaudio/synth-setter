"""Unit tests for the offline-wandb readers' flush-race polling.

The offline writer serializes records to the ``run-*.wandb`` binary
asynchronously, so a single-shot read can land before they flush (observed as a
conda-only 0-records flake). Both ``read_history_rows`` (decoded rows) and
``read_run_binary`` (raw bytes) poll until the caller's predicate holds via the
shared ``_poll_until`` loop; these tests pin that retry contract by stubbing the
per-attempt read and freezing the clock so no real datastore binary or
wall-clock wait is needed.
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


def test_run_binary_until_predicate_retries_until_record_flushes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Header-only early reads are retried — one ``sleep`` per gap — until the marker lands.

    Mirrors the history-row retry contract for the raw-bytes ``read_run_binary``
    path, which assertions on artifact records (absent from the decoded rows)
    rely on.

    :param monkeypatch: Replaces ``Path.read_bytes`` and counts ``sleep`` so the
        flush lag is simulated without a real binary or wall-clock wait.
    """
    reads = iter([b":W&B", b":W&B", b":W&B...my-artifact...dataset-spec"])
    read_count = 0
    sleep_count = 0

    def _read(_self: Path) -> bytes:
        nonlocal read_count
        read_count += 1
        return next(reads)

    def _sleep(_s: float) -> None:
        nonlocal sleep_count
        sleep_count += 1

    monkeypatch.setattr(Path, "read_bytes", _read)
    monkeypatch.setattr(wandb_offline.time, "sleep", _sleep)

    payload = wandb_offline.read_run_binary(
        Path("ignored.wandb"),
        until=lambda data: b"my-artifact" in data and b"dataset-spec" in data,
    )

    assert payload == b":W&B...my-artifact...dataset-spec"
    assert read_count == 3
    assert sleep_count == 2


def test_run_binary_until_predicate_polls_to_deadline_then_returns_last_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A never-satisfied predicate polls to the deadline, then returns the last (header-only) read.

    Pins that a genuine never-flushed record returns bytes for the caller's own
    assertion to fail on, rather than hanging or raising.

    :param monkeypatch: Stubs ``read_bytes`` to stay header-only, advances a fake
        clock, and freezes ``sleep`` so the timeout is reached instantly.
    """
    ticks = iter(range(100))
    read_count = 0

    def _read(_self: Path) -> bytes:
        nonlocal read_count
        read_count += 1
        return b":W&B"

    monkeypatch.setattr(Path, "read_bytes", _read)
    monkeypatch.setattr(wandb_offline.time, "monotonic", lambda: float(next(ticks)))
    monkeypatch.setattr(wandb_offline.time, "sleep", lambda _s: None)

    payload = wandb_offline.read_run_binary(
        Path("ignored.wandb"),
        until=lambda data: b"never" in data,
        timeout_s=2.5,
    )

    assert payload == b":W&B"
    # Pins the deadline-exit branch, not a first-iteration short-circuit.
    assert read_count > 1


def test_run_binary_without_predicate_reads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting ``until`` keeps the single-shot contract — one read, never sleeps.

    :param monkeypatch: Counts ``read_bytes`` invocations and fails ``sleep`` to
        prove no retry loop runs.
    """
    read_count = 0

    def _read(_self: Path) -> bytes:
        nonlocal read_count
        read_count += 1
        return b":W&B"

    monkeypatch.setattr(Path, "read_bytes", _read)
    monkeypatch.setattr(
        wandb_offline.time, "sleep", lambda _s: pytest.fail("single-shot read must not sleep")
    )

    payload = wandb_offline.read_run_binary(Path("ignored.wandb"))

    assert payload == b":W&B"
    assert read_count == 1
