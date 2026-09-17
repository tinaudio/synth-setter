"""Tests for the progress-bounded wait used by fresh-process integration tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers.subprocess_progress import await_ready_signal

_STALL_TIMEOUT_SECONDS = 0.3
_HARD_CAP_SECONDS = 20.0

_SIGNAL_READY = """
import os, sys
print("starting", flush=True)
fd = os.open(sys.argv[1], os.O_WRONLY)
os.write(fd, b"1")
import time; time.sleep(30)
"""

_BURN_CPU_THEN_SIGNAL_READY = """
import os, sys, time
deadline = time.monotonic() + float(sys.argv[2])
while time.monotonic() < deadline:
    pass
fd = os.open(sys.argv[1], os.O_WRONLY)
os.write(fd, b"1")
time.sleep(30)
"""

_SLEEP_SILENTLY = """
import time
time.sleep(30)
"""

_EXIT_WITHOUT_SIGNALLING = """
import sys
print("giving up", flush=True)
sys.exit(3)
"""

_SPIN_FOREVER = """
while True:
    pass
"""


@pytest.fixture(name="ready_fifo")
def _ready_fifo(tmp_path: Path) -> Path:
    """Create the readiness FIFO the child writes one byte to.

    :param tmp_path: Isolated directory holding the FIFO.
    :returns: Path of the created FIFO.
    """
    fifo = tmp_path / "ready.fifo"
    os.mkfifo(fifo)
    return fifo


def _run_child(script: str, ready_fifo: Path, *args: str) -> subprocess.Popen[str]:
    """Start a fresh interpreter running ``script`` against ``ready_fifo``.

    :param script: Python source executed with ``-c``.
    :param ready_fifo: FIFO path passed as the child's first argument.
    :param *args: Extra arguments appended after the FIFO path.
    :returns: Started child with stdout and stderr merged into one pipe.
    """
    return subprocess.Popen(  # noqa: S603 — fixed interpreter plus test-owned source
        [sys.executable, "-c", script, str(ready_fifo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _await(process: subprocess.Popen[str], ready_fifo: Path, *, stall_s: float) -> str:
    """Wait for ``process`` to report readiness, always reaping it afterwards.

    :param process: Child under test.
    :param ready_fifo: FIFO the child signals on.
    :param stall_s: Seconds of no progress that count as stuck.
    :returns: Output the child emitted before readiness.
    """
    read_fd = os.open(ready_fifo, os.O_RDONLY | os.O_NONBLOCK)
    write_fd = os.open(ready_fifo, os.O_WRONLY | os.O_NONBLOCK)
    try:
        return await_ready_signal(
            process, read_fd, stall_timeout_s=stall_s, hard_cap_s=_HARD_CAP_SECONDS
        )
    finally:
        os.close(write_fd)
        os.close(read_fd)
        process.kill()
        process.wait(timeout=10)


def test_await_ready_signal_when_the_child_signals_returns_its_earlier_output(
    ready_fifo: Path,
) -> None:
    """Output emitted before readiness reaches the caller for failure messages.

    :param ready_fifo: FIFO the child signals on.
    """
    process = _run_child(_SIGNAL_READY, ready_fifo)

    output = _await(process, ready_fifo, stall_s=_STALL_TIMEOUT_SECONDS)

    assert "starting" in output


def test_await_ready_signal_when_the_child_is_silent_but_busy_keeps_waiting(
    ready_fifo: Path,
) -> None:
    """A slow runner is not a hang: CPU time advancing counts as progress.

    :param ready_fifo: FIFO the child signals on.
    """
    busy_seconds = _STALL_TIMEOUT_SECONDS * 4
    process = _run_child(_BURN_CPU_THEN_SIGNAL_READY, ready_fifo, str(busy_seconds))

    output = _await(process, ready_fifo, stall_s=_STALL_TIMEOUT_SECONDS)

    assert output == ""


def test_await_ready_signal_when_the_child_is_silent_and_idle_reports_no_progress(
    ready_fifo: Path,
) -> None:
    """A child consuming neither output nor CPU is stuck and fails fast.

    :param ready_fifo: FIFO the child signals on.
    """
    process = _run_child(_SLEEP_SILENTLY, ready_fifo)

    with pytest.raises(AssertionError, match="made no progress"):
        _await(process, ready_fifo, stall_s=_STALL_TIMEOUT_SECONDS)


def test_await_ready_signal_when_the_child_exits_first_reports_the_exit(
    ready_fifo: Path,
) -> None:
    """An exit before readiness is reported as an exit, with the child's output.

    :param ready_fifo: FIFO the child signals on.
    """
    process = _run_child(_EXIT_WITHOUT_SIGNALLING, ready_fifo)

    with pytest.raises(AssertionError, match="exited before signalling readiness"):
        _await(process, ready_fifo, stall_s=_STALL_TIMEOUT_SECONDS)


def test_await_ready_signal_when_the_child_spins_forever_stops_at_the_hard_cap(
    ready_fifo: Path,
) -> None:
    """A spinning child reports progress forever, so an absolute cap still binds.

    :param ready_fifo: FIFO the child signals on.
    """
    process = _run_child(_SPIN_FOREVER, ready_fifo)
    read_fd = os.open(ready_fifo, os.O_RDONLY | os.O_NONBLOCK)
    write_fd = os.open(ready_fifo, os.O_WRONLY | os.O_NONBLOCK)
    try:
        with pytest.raises(AssertionError, match="did not signal readiness within"):
            await_ready_signal(
                process, read_fd, stall_timeout_s=_STALL_TIMEOUT_SECONDS, hard_cap_s=0.5
            )
    finally:
        os.close(write_fd)
        os.close(read_fd)
        process.kill()
        process.wait(timeout=10)
