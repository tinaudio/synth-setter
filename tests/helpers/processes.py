"""Lifecycle helpers for multiprocessing integration tests."""

from __future__ import annotations

import multiprocessing
import queue
import time
from collections.abc import Callable, Sequence
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue

_PROCESS_EXIT_TIMEOUT_SECONDS = 120.0
_PROCESS_STATUS_POLL_SECONDS = 0.1
_RESULT_TIMEOUT_SECONDS = 600.0
_TERMINATE_TIMEOUT_SECONDS = 5.0


def _terminate_and_reap(process: BaseProcess) -> None:
    """Stop a live process and wait until the OS reaps it.

    :param process: Process handle that must be closed by the caller after reaping.
    :raises RuntimeError: If the worker survives SIGKILL.
    """
    if not process.is_alive():
        process.join()
        return

    process.terminate()
    process.join(timeout=_TERMINATE_TIMEOUT_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(timeout=_TERMINATE_TIMEOUT_SECONDS)
    if process.is_alive():
        raise RuntimeError(f"worker pid={process.pid} survived SIGKILL")


def _collect_results(
    out: Queue[object],
    processes: Sequence[BaseProcess],
    timeout_s: float,
) -> list[object]:
    """Collect one result per process while watching for early exits.

    :param out: Shared result queue.
    :param processes: Started workers expected to publish once each.
    :param timeout_s: Total seconds allowed for all results.
    :returns: Results in queue arrival order.
    :raises RuntimeError: If collection times out or a worker exits before reporting.
    """
    results = []
    result_deadline = time.monotonic() + timeout_s
    while len(results) < len(processes):
        remaining_s = result_deadline - time.monotonic()
        if remaining_s <= 0:
            raise RuntimeError("worker result collection timed out")
        try:
            results.append(out.get(timeout=min(_PROCESS_STATUS_POLL_SECONDS, remaining_s)))
        except queue.Empty:
            failed = [
                (process.pid, process.exitcode)
                for process in processes
                if process.exitcode not in (None, 0)
            ]
            if failed:
                raise RuntimeError(f"worker processes failed: {failed}") from None
            exited = sum(process.exitcode == 0 for process in processes)
            if exited > len(results):
                raise RuntimeError("worker exited without publishing a result") from None
    return results


def _join_and_validate(processes: Sequence[BaseProcess], exit_timeout_s: float) -> None:
    """Wait for workers and reject non-zero exits.

    :param processes: Workers that published their final result.
    :param exit_timeout_s: Seconds allowed for each worker to exit.
    :raises RuntimeError: If a worker exits with a failure status.
    """
    for process in processes:
        process.join(timeout=exit_timeout_s)
    failed = [
        (process.pid, process.exitcode)
        for process in processes
        if not process.is_alive() and process.exitcode != 0
    ]
    if failed:
        raise RuntimeError(f"worker processes failed: {failed}")


def _cleanup_processes_and_queue(processes: Sequence[BaseProcess], out: Queue[object]) -> None:
    """Attempt every process and queue cleanup before reporting failures.

    :param processes: All workers that started successfully.
    :param out: Result queue to close after process cleanup.
    :raises RuntimeError: If any resource cannot be cleaned up.
    """
    errors = []
    for process in processes:
        try:
            _terminate_and_reap(process)
        except (AssertionError, RuntimeError, ValueError) as exc:
            errors.append(str(exc))
    try:
        out.close()
        out.join_thread()
    except (AssertionError, OSError, ValueError) as exc:
        errors.append(str(exc))
    for process in processes:
        try:
            process.close()
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError(f"worker cleanup failed: {'; '.join(errors)}")


def collect_process_results(
    target: Callable[..., None],
    worker_args: Sequence[tuple[object, ...]],
    *,
    exit_timeout_s: float = _PROCESS_EXIT_TIMEOUT_SECONDS,
    result_timeout_s: float = _RESULT_TIMEOUT_SECONDS,
) -> list[object]:
    """Spawn workers, collect one result each, and leave no live children.

    The result queue is appended to each worker's argument tuple. A worker that publishes its final
    result but does not exit is terminated during cleanup.

    :param target: Pickleable worker callable that puts one result on its final argument.
    :param worker_args: Positional argument tuples for each worker, excluding the queue.
    :param exit_timeout_s: Seconds allowed for each worker to exit after results arrive.
    :param result_timeout_s: Total seconds allowed for all worker results to arrive.
    :returns: Results in queue arrival order.
    """
    context = multiprocessing.get_context("spawn")
    out = context.Queue()
    processes = [context.Process(target=target, args=(*args, out)) for args in worker_args]
    started: list[BaseProcess] = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        results = _collect_results(out, started, result_timeout_s)
        _join_and_validate(started, exit_timeout_s)
        return results
    finally:
        _cleanup_processes_and_queue(processes, out)
