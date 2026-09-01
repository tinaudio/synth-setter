"""Centralized bounded retry policy for Lance object-store operations."""

from __future__ import annotations

from collections.abc import Callable

import structlog
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)
_MAX_ATTEMPTS = 3
_BACKOFF_INITIAL_SECONDS = 0.25
_BACKOFF_MAX_SECONDS = 2.0
_RETRYABLE_IO_MARKERS = (
    "408 request timeout",
    "429 too many requests",
    "500 internal server error",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway timeout",
    "connection closed",
    "connection refused",
    "connection reset",
    "error sending request",
    "request timeout",
    "temporarily unavailable",
    "timed out",
)


def is_retryable_lance_io_error(error: BaseException) -> bool:
    """Return whether Lance exposed a transient object-store transport failure.

    :param error: Exception raised by a Lance or object-store operation.
    :returns: Whether retrying an idempotent or publication-recovering operation is safe.
    """
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    message = str(error).casefold()
    return "lanceerror(io)" in message and any(
        marker in message for marker in _RETRYABLE_IO_MARKERS
    )


def retry_lance_io[ResultT](operation_name: str, operation: Callable[[], ResultT]) -> ResultT:
    """Run one operation under the centralized transient Lance retry policy.

    :param operation_name: Secret-free operation label included in retry logs.
    :param operation: Idempotent or publication-recovering operation.
    :returns: The successful operation result.
    """

    def log_failed_attempt(retry_state: RetryCallState) -> None:
        logger.warning(
            "lance_io_attempt_failed",
            operation=operation_name,
            attempt=retry_state.attempt_number,
            max_attempts=_MAX_ATTEMPTS,
        )

    retrying = Retrying(
        after=log_failed_attempt,
        retry=retry_if_exception(is_retryable_lance_io_error),
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        wait=wait_exponential(
            multiplier=_BACKOFF_INITIAL_SECONDS,
            max=_BACKOFF_MAX_SECONDS,
        ),
        reraise=True,
    )
    return retrying(operation)
