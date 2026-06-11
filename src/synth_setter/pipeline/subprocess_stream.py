"""Exit-keyed streamed subprocess execution for long-running pipeline children.

Runs a child with merged stdout+stderr, teeing its output in real time
through the parent's ``sys.stderr`` — the only channel wandb
``console=wrap`` captures (#1465) — while keeping a captured copy. The
defining design choice: completion is keyed on child *exit*, never pipe
EOF. Grandchildren that inherit the merged pipe and outlive the child
(the headless-VST X11 daemon tree, #1634) therefore cannot stall
completion or falsify exit codes; wall-clock ``timeout`` stays a pure
per-call-site policy bound around the exit wait. Linux/macOS only.

The tee assumes the wrapped ``sys.stderr`` write returns promptly (wandb
buffers in-process); concurrent callers interleave at chunk granularity.
"""

from __future__ import annotations

import asyncio
import codecs
import os
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import cast

import structlog

__all__ = ["check_call_streamed", "check_call_streamed_async"]

_LOG = structlog.get_logger(__name__)

# SIGTERM -> SIGKILL escalation grace, seconds. TERM first so children's EXIT
# traps run (the VST wrapper reaps its X11 tree, #1634); KILL keeps timeouts real.
_TERM_TO_KILL_GRACE_SECONDS = 5.0

# Post-exit bound (seconds) on draining buffered pipe output before the reader
# is abandoned — the alternative, keying on EOF, was the #1617 hang/misreport.
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

    The child leads its own session/process group; on return (any path,
    including cancellation, where no KILL escalation follows) the group
    receives a SIGTERM sweep so in-group descendants outliving the child are
    reaped. ``setsid()``/double-forked escapees survive — accepted gap until
    a Linux cgroup scope lands.

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
    # start_new_session makes the child its own group leader, so pid IS pgid —
    # a getpgid syscall would race the watcher reaping a fast exit.
    pgid = proc.pid
    captured: list[bytes] = []
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    tee_broken = False

    def _tee(text: str) -> None:
        # A broken wrapped stream must not kill the pump (exit codes are the
        # contract, and a dead pump would re-open the #735 full-pipe hang).
        nonlocal tee_broken
        if tee_broken or not text:
            return
        try:
            sys.stderr.write(text)
        except Exception:  # noqa: BLE001 — degrade to capture-only forwarding
            tee_broken = True

    async def _pump() -> None:
        # console=wrap patches ``write`` in place on the parent's *text* stream;
        # ``sys.stderr.buffer`` would slip under the patch, emptying the Logs tab (#1465).
        # Chunked reads (not readline): a long line can't overrun the StreamReader limit.
        stdout = proc.stdout
        if stdout is None:  # pragma: no cover — unreachable with stdout=PIPE
            raise RuntimeError("child spawned without a stdout pipe")
        while chunk := await stdout.read(_READ_CHUNK_BYTES):
            captured.append(chunk)
            _tee(decoder.decode(chunk))
        # EOF: flush a trailing incomplete multibyte sequence into the tee.
        _tee(decoder.decode(b"", final=True))

    pump_task = asyncio.create_task(_pump())
    timed_out_after: float | None = None
    try:
        try:
            returncode = await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # wait_for(timeout=None) cannot time out, so timeout is a float here.
            timed_out_after = cast(float, timeout)
            _signal_group(pgid, signal.SIGTERM)
            try:
                returncode = await asyncio.wait_for(
                    proc.wait(), timeout=_TERM_TO_KILL_GRACE_SECONDS
                )
            except asyncio.TimeoutError:
                _LOG.warning(
                    "subprocess_term_to_kill_escalation",
                    cmd=cmd[0],
                    grace_seconds=_TERM_TO_KILL_GRACE_SECONDS,
                )
                _signal_group(pgid, signal.SIGKILL)
                returncode = await proc.wait()

        try:
            await asyncio.wait_for(pump_task, timeout=_POST_EXIT_DRAIN_SECONDS)
        except asyncio.TimeoutError:
            # A pipe-holding survivor must not stall completion; abandon the tail.
            _LOG.warning(
                "subprocess_output_drain_abandoned",
                cmd=cmd[0],
                drain_seconds=_POST_EXIT_DRAIN_SECONDS,
            )
            pump_task.cancel()
            # return_exceptions swallows the pump's CancelledError without
            # masking an external cancellation of this coroutine.
            await asyncio.gather(pump_task, return_exceptions=True)

        if tee_broken:
            # Logged here, where a sink failure can no longer kill the pump.
            _LOG.warning("subprocess_tee_degraded", cmd=cmd[0])
    finally:
        # Sweep right after the wait so the pgid-recycle window stays
        # microseconds; surviving members keep the pgid reserved, so it's safe.
        pump_task.cancel()
        _signal_group(pgid, signal.SIGTERM)

    output = b"".join(captured)
    if timed_out_after is not None:
        raise subprocess.TimeoutExpired(list(cmd), timed_out_after, output=output)
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
