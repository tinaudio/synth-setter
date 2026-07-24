"""Lifecycle helpers for multiprocessing integration tests."""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Sequence
from multiprocessing.process import BaseProcess

_PROCESS_EXIT_TIMEOUT_SECONDS = 120.0
_RESULT_TIMEOUT_SECONDS = 600.0
_TERMINATE_TIMEOUT_SECONDS = 5.0


def _terminate_and_reap(process: BaseProcess) -> None:
    """Stop a live process and wait until the OS reaps it.

    :param process: Spawned worker to stop.
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


def collect_process_results(
    target: Callable[..., None],
    worker_args: Sequence[tuple[object, ...]],
    *,
    exit_timeout_s: float = _PROCESS_EXIT_TIMEOUT_SECONDS,
) -> list[object]:
    """Spawn workers, collect one result each, and leave no live children.

    The result queue is appended to each worker's argument tuple. A worker that publishes its final
    result but does not exit is terminated during cleanup.

    :param target: Pickleable worker callable that puts one result on its final argument.
    :param worker_args: Positional argument tuples for each worker, excluding the queue.
    :param exit_timeout_s: Seconds allowed for each worker to exit after results arrive.
    :returns: Results in queue arrival order.
    :raises RuntimeError: If a worker exits with a failure status.
    """
    context = multiprocessing.get_context("spawn")
    out = context.Queue()
    processes = [context.Process(target=target, args=(*args, out)) for args in worker_args]
    started: list[BaseProcess] = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        results = [out.get(timeout=_RESULT_TIMEOUT_SECONDS) for _ in started]

        for process in processes:
            process.join(timeout=exit_timeout_s)
        failed = [
            (process.pid, process.exitcode)
            for process in processes
            if not process.is_alive() and process.exitcode != 0
        ]
        if failed:
            raise RuntimeError(f"worker processes failed: {failed}")
        return results
    finally:
        for process in started:
            _terminate_and_reap(process)
        out.close()
        out.join_thread()
        for process in processes:
            process.close()
