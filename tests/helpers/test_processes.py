"""Process-lifecycle tests for spawned integration-test workers."""

from __future__ import annotations

import multiprocessing
import os
import signal
from multiprocessing.queues import Queue
from pathlib import Path

import pytest

from tests.helpers.processes import _collect_results, collect_process_results

_WORKER_FAILURE_TIMEOUT_SECONDS = 10.0


def _report_pid_then_stall(pid_file: str, out: Queue[str]) -> None:
    """Publish a result, then model a worker stuck during interpreter shutdown.

    :param pid_file: Path receiving the worker PID.
    :param out: Multiprocessing queue receiving the worker result.
    """
    path = Path(pid_file)
    path.write_text(str(os.getpid()))
    out.put("reported")
    signal.pause()


def _fail_before_reporting(_out: Queue[object]) -> None:
    """Exit before publishing a result.

    :param _out: Unused result queue supplied by the process collector.
    :raises RuntimeError: Always, to model a worker-side failure.
    """
    raise RuntimeError("worker failed")


def test_collect_results_expired_deadline_reports_known_worker_failure() -> None:
    """A known worker failure takes precedence over an expired result deadline."""
    context = multiprocessing.get_context("spawn")
    out = context.Queue()
    process = context.Process(target=_fail_before_reporting, args=(out,))
    process.start()
    process.join()

    try:
        with pytest.raises(RuntimeError, match="worker processes failed"):
            _collect_results(out, [process], timeout_s=0.0)
    finally:
        out.close()
        out.join_thread()
        process.close()


def test_collect_process_results_worker_failure_before_result_raises_promptly() -> None:
    """A worker failure is reported without waiting for the result budget."""
    with pytest.raises(RuntimeError, match="worker processes failed"):
        collect_process_results(
            _fail_before_reporting,
            [()],
            result_timeout_s=_WORKER_FAILURE_TIMEOUT_SECONDS,
        )


def test_collect_process_results_workers_stall_after_results_terminates_processes(
    tmp_path: Path,
) -> None:
    """A worker that reports success but never exits is terminated and reaped.

    :param tmp_path: Scratch location receiving the worker PID.
    """
    pid_files = [tmp_path / "worker-0.pid", tmp_path / "worker-1.pid"]

    results = collect_process_results(
        _report_pid_then_stall,
        [(str(pid_file),) for pid_file in pid_files],
        exit_timeout_s=0.1,
    )

    assert results == ["reported", "reported"]
    for pid_file in pid_files:
        worker_pid = int(pid_file.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)
