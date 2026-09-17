"""Wait on a fresh-process child without mistaking a slow runner for a hang.

A child that must import torch, Lightning and Hydra before the behavior under test begins spends
most of its wall clock in silent startup, so a single whole-process deadline is a contention
threshold rather than a hang bound (#3501; same shape as #3455/#3500). The wait here rejects a
child only once it stops making progress — emitting output or consuming CPU — and keeps an absolute
cap so a spinning child is still killed rather than waited out.
"""

from __future__ import annotations

import contextlib
import os
import select
import subprocess
import time
from collections.abc import Callable

import psutil

_STALL_TIMEOUT_SECONDS = 60.0
_HARD_CAP_SECONDS = 900.0
_READY_BYTE = b"1"
# Bounded so CPU time is sampled several times within one stall window.
_MAX_POLL_SECONDS = 1.0


def accumulated_cpu_seconds(pid: int, last: float) -> float:
    """Return the CPU time of the child's whole tree, or ``last`` once it is gone.

    Descendants are included because a parent blocked in ``wait()`` consumes nothing while its
    own child does the work; ``children_*`` covers the ones already reaped.

    :param pid: Process id of the child.
    :param last: Reading to keep when the process can no longer be sampled.
    :returns: User plus system CPU seconds across the process tree.
    """
    try:
        process = psutil.Process(pid)
        times = process.cpu_times()
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        return last
    total = times.user + times.system + times.children_user + times.children_system
    # Suppressed around the walk as well: one unreadable process must not discard the rest.
    with contextlib.suppress(psutil.Error):
        for descendant in process.children(recursive=True):
            with contextlib.suppress(psutil.Error):
                total += sum(descendant.cpu_times()[:2])
    return total


def await_progress(
    condition: Callable[[], bool],
    pid: int,
    *,
    description: str,
    stall_timeout_s: float = _STALL_TIMEOUT_SECONDS,
    hard_cap_s: float = _HARD_CAP_SECONDS,
) -> None:
    """Wait for ``condition`` while the tree under ``pid`` keeps making progress.

    :param condition: Checked between polls; the wait ends once it returns true.
    :param pid: Process whose tree is sampled for CPU time.
    :param description: What is being awaited, quoted in the failure message.
    :param stall_timeout_s: Seconds without CPU time that count as stuck.
    :param hard_cap_s: Absolute bound, so a spinning tree still fails.
    :raises AssertionError: If the tree stalls or the wait outlasts ``hard_cap_s``.
    """
    cpu_seconds = accumulated_cpu_seconds(pid, 0.0)
    last_progress = time.monotonic()
    hard_deadline = last_progress + hard_cap_s
    poll_s = min(_MAX_POLL_SECONDS, stall_timeout_s / 4)
    while not condition():
        if time.monotonic() >= hard_deadline:
            raise AssertionError(f"waited for {description} for over {hard_cap_s:g}s")
        time.sleep(min(poll_s, max(0.0, hard_deadline - time.monotonic())))
        sampled = accumulated_cpu_seconds(pid, cpu_seconds)
        if sampled > cpu_seconds:
            cpu_seconds = sampled
            last_progress = time.monotonic()
        elif time.monotonic() - last_progress >= stall_timeout_s:
            raise AssertionError(
                f"no progress for {stall_timeout_s:g}s (no CPU time) while waiting "
                f"for {description}"
            )


def await_ready_signal(
    process: subprocess.Popen[str],
    ready_fd: int,
    *,
    stall_timeout_s: float = _STALL_TIMEOUT_SECONDS,
    hard_cap_s: float = _HARD_CAP_SECONDS,
) -> str:
    """Wait until ``process`` writes one byte to ``ready_fd``, retaining its output.

    :param process: Child started with stdout piped in text mode.
    :param ready_fd: Non-blocking read end of the child's readiness FIFO.
    :param stall_timeout_s: Seconds without output or CPU time that count as stuck.
    :param hard_cap_s: Absolute bound, so a child that spins without signalling still fails.
    :returns: Output the child emitted before signalling readiness.
    :raises AssertionError: If the child stalls, exits early, or outlasts ``hard_cap_s``.
    """
    assert process.stdout is not None, "child must be started with stdout=subprocess.PIPE"
    output: list[str] = []
    cpu_seconds = accumulated_cpu_seconds(process.pid, 0.0)
    last_progress = time.monotonic()
    hard_deadline = last_progress + hard_cap_s
    poll_s = min(_MAX_POLL_SECONDS, stall_timeout_s / 4)
    # Dropped on EOF, so a closed pipe cannot spin the loop by staying readable.
    watched: list[object] = [ready_fd, process.stdout]

    while True:
        remaining_s = hard_deadline - time.monotonic()
        if remaining_s <= 0:
            raise AssertionError(
                f"child did not signal readiness within {hard_cap_s:g}s:\n{''.join(output)}"
            )
        readable, _, _ = select.select(watched, [], [], min(poll_s, remaining_s))
        if process.stdout in readable:
            line = process.stdout.readline()
            if line:
                output.append(line)
                last_progress = time.monotonic()
                continue
            watched.remove(process.stdout)
        # Checked after stdout so a child that signals and exits keeps its output.
        if ready_fd in readable and os.read(ready_fd, 1) == _READY_BYTE:
            return "".join(output)
        if process.stdout not in watched and process.poll() is not None:
            raise AssertionError(f"child exited before signalling readiness:\n{''.join(output)}")
        sampled = accumulated_cpu_seconds(process.pid, cpu_seconds)
        if sampled > cpu_seconds:
            cpu_seconds = sampled
            last_progress = time.monotonic()
        elif time.monotonic() - last_progress >= stall_timeout_s:
            raise AssertionError(
                f"child made no progress for {stall_timeout_s:g}s "
                f"(no output, no CPU time):\n{''.join(output)}"
            )
