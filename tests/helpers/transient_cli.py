"""Bounded retry and external-service classification for live-registry CLI tests."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable

import pytest

_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2.0

# Node reports a reset or DNS interruption as a bare ``TypeError: fetch failed`` and prints the
# socket code only in the cause chain, so both halves have to be recognised (#3041).
_TRANSIENT_OUTPUT_MARKERS = (
    "fetch failed",
    "ECONNRESET",
    "ETIMEDOUT",
    "ENOTFOUND",
    "EAI_AGAIN",
    "socket hang up",
)


def is_transient_registry_failure(completed: subprocess.CompletedProcess[str]) -> bool:
    """Report whether a CLI run failed on a transient registry or network fault.

    :param completed: Finished CLI process with captured text output.
    :returns: True when the run failed and its output names a transient network fault.
    """
    if completed.returncode == 0:
        return False
    return any(
        marker in completed.stdout + completed.stderr for marker in _TRANSIENT_OUTPUT_MARKERS
    )


def run_until_stable(
    run: Callable[[], subprocess.CompletedProcess[str]],
    *,
    attempts: int = _RETRY_ATTEMPTS,
    backoff_s: float = _RETRY_BACKOFF_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Re-run a CLI invocation while it fails on a transient network fault.

    :param run: Callable invoking the CLI once and returning its finished process.
    :param attempts: Maximum invocations, including the first.
    :param backoff_s: Base linear backoff multiplied by the attempt number.
    :returns: The first non-transient result, or the last attempt's result.
    :raises ValueError: ``attempts`` is below one.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be at least one: {attempts}")
    completed = run()
    for attempt in range(1, attempts):
        if not is_transient_registry_failure(completed):
            return completed
        time.sleep(backoff_s * attempt)
        completed = run()
    return completed


def skip_on_transient_registry_failure(completed: subprocess.CompletedProcess[str]) -> None:
    """Classify an exhausted transient network fault as an external-service skip.

    :param completed: Finished CLI process whose retries are already exhausted.
    """
    if is_transient_registry_failure(completed):
        pytest.skip(f"registry unreachable after bounded retries: {completed.stderr.strip()}")
