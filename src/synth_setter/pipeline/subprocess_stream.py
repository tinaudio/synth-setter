"""Exit-keyed streamed subprocess execution for long-running pipeline children.

Runs a child with merged stdout+stderr, teeing its output in real time
through the parent's ``sys.stderr`` — the only channel wandb
``console=wrap`` captures (#1465) — while keeping a captured copy. The
defining design choice: completion is keyed on child *exit*, never pipe
EOF. Grandchildren that inherit the merged pipe and outlive the child
(the headless-VST X11 daemon tree, #1634) therefore cannot stall
completion or falsify exit codes; wall-clock ``timeout`` stays a pure
per-call-site policy bound around the exit wait. Linux/macOS only.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import os
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence

__all__ = ["check_call_streamed", "check_call_streamed_async"]

# SIGTERM -> SIGKILL escalation grace, seconds. TERM first so children's EXIT
# traps run (the headless-VST wrapper reaps its X11 tree, #1634); KILL keeps
# the timeout real for a wedged child.
_TERM_TO_KILL_GRACE_SECONDS = 5.0

# Post-exit bound on draining output still buffered in the pipe, seconds. A
# surviving pipe-holder writes under this grace, then the reader is abandoned
# — the alternative is keying completion on EOF, the #1617 hang/misreport.
_POST_EXIT_DRAIN_SECONDS = 2.0

_READ_CHUNK_BYTES = 8192


def check_call_streamed(
    cmd: Sequence[str],
    *,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
) -> bytes:
    """Blocking facade over :func:`check_call_streamed_async`.

    Propagates that coroutine's ``CalledProcessError`` / ``TimeoutExpired``
    contract unchanged. Safe off the main thread (each call runs its own
    event loop), which is how ``generate_dataset``'s shard worker threads
    use it.

    :param cmd: Child argv, run unquoted with no shell, so callers pre-validate it.
    :param timeout: Per-call-site policy bound in seconds; ``None`` means no limit.
    :param env: Child environment; ``None`` inherits the parent's.
    :returns: The child's captured merged stdout+stderr bytes.
    """
    return asyncio.run(check_call_streamed_async(cmd, timeout=timeout, env=env))


async def check_call_streamed_async(
    cmd: Sequence[str],
    *,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
) -> bytes:
    """Run ``cmd``, teeing merged stdout+stderr through the wandb-wrapped stream.

    The child leads its own session/process group; on return (any path) the
    group receives a SIGTERM sweep so in-group descendants outliving the
    child are reaped. ``setsid()``/double-forked escapees survive — accepted
    gap until a Linux cgroup scope lands.

    :param cmd: Child argv, run unquoted with no shell, so callers pre-validate it.
    :param timeout: Per-call-site policy bound in seconds on the child's own
        runtime; on expiry the group gets SIGTERM, then SIGKILL after a grace.
        ``None`` means no limit — liveness never depends on this.
    :param env: Child environment; ``None`` inherits the parent's. Either way
        ``PYTHONUNBUFFERED=1`` is pinned so a hung child's last lines are
        already visible (the #735 diagnosis property).
    :returns: The child's captured merged stdout+stderr bytes.
    :raises subprocess.CalledProcessError: Child exited non-zero; ``output``
        carries the bytes captured before death.
    :raises subprocess.TimeoutExpired: The child itself overran ``timeout`` —
        never raised for post-exit pipe loitering.
    """
    child_env = {**(os.environ if env is None else env), "PYTHONUNBUFFERED": "1"}
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
        env=child_env,
    )
    # Saved before the leader is reaped: getpgid on a reaped pid raises even
    # while grandchildren still populate the group.
    pgid = os.getpgid(proc.pid)
    captured: list[bytes] = []
    decoder = codecs.getincrementaldecoder("utf-8")("replace")

    async def _pump() -> None:
        # console=wrap patches ``write`` in place on the parent's *text*
        # stream; the tee must go through it — ``sys.stderr.buffer`` would
        # slip underneath the patch and re-empty the run's Logs tab (#1465).
        # Chunked reads (not readline) so a line longer than the
        # StreamReader limit can't raise mid-pump.
        assert proc.stdout is not None  # noqa: S101 — guaranteed by stdout=PIPE
        while chunk := await proc.stdout.read(_READ_CHUNK_BYTES):
            captured.append(chunk)
            sys.stderr.write(decoder.decode(chunk))

    pump_task = asyncio.create_task(_pump())
    timed_out = False
    try:
        try:
            returncode = await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            _signal_group(pgid, signal.SIGTERM)
            try:
                returncode = await asyncio.wait_for(
                    proc.wait(), timeout=_TERM_TO_KILL_GRACE_SECONDS
                )
            except asyncio.TimeoutError:
                _signal_group(pgid, signal.SIGKILL)
                returncode = await proc.wait()

        try:
            await asyncio.wait_for(pump_task, timeout=_POST_EXIT_DRAIN_SECONDS)
        except asyncio.TimeoutError:
            # A pipe-holding survivor must not stall completion; abandon the tail.
            pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump_task
    finally:
        # Runs immediately after the wait (or on an abrupt cancel, e.g.
        # KeyboardInterrupt) so the pgid-recycle window stays microseconds; if
        # group members survive, POSIX reserves the pgid and this is safe.
        pump_task.cancel()
        _signal_group(pgid, signal.SIGTERM)

    output = b"".join(captured)
    if timed_out:
        raise subprocess.TimeoutExpired(list(cmd), timeout or 0.0, output=output)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, list(cmd), output=output)
    return output


def _signal_group(pgid: int, sig: signal.Signals) -> None:
    """Signal a process group, tolerating one that is already gone.

    :param pgid: Group id saved at spawn time.
    :param sig: Signal to deliver to every member.
    """
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
