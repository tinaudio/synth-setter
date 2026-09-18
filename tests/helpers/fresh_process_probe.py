"""Run a fresh-interpreter probe with dependency startup bounded apart from behavior.

A probe that imports torch, torchsynth and FLAMO spends most of its wall clock before the behavior
under test begins, so a single whole-process timeout turns a saturated CI runner into a false "the
loader hung" report (#3455 on macOS, #3500 on Ubuntu). Probes here report each startup stage
instead, and a stage is only rejected once it stops making progress — emitting a marker or
consuming CPU, so a stage whose interior is silent (one long import) is not a hang by definition
(#3666).
"""

from __future__ import annotations

import queue
import subprocess
import sys
import tempfile
import threading
import time
from textwrap import dedent
from typing import IO

from tests.helpers.subprocess_progress import accumulated_cpu_seconds

_STARTUP_STALL_TIMEOUT_SECONDS = 30.0
# Launch spans fork/exec and interpreter boot, which report nothing and may be descheduled
# entirely on a saturated runner, so it is bounded apart from the stages it precedes (#3673).
_LAUNCH_TIMEOUT_SECONDS = 60.0
# Absolute bound on startup, so CPU time buys a silent stage time without retiring the hang bound.
_STARTUP_CAP_SECONDS = 300.0
_BEHAVIOR_TIMEOUT_SECONDS = 30.0
_READY_MARKER = "probe-ready"
# Bounded so CPU time is sampled several times within one stall window.
_MAX_POLL_SECONDS = 1.0

# Injected above every probe: progress() marks a completed startup stage and
# ready() hands the remaining budget to the behavior under test.
_PROBE_PRELUDE = f"""
def progress(stage):
    print(stage, flush=True)


def ready():
    progress({_READY_MARKER!r})
"""


def _drain(stream: IO[str], sink: queue.Queue[str | None]) -> None:
    """Forward a child stream line by line and signal end of stream with ``None``.

    :param stream: Child pipe opened in text mode.
    :param sink: Queue receiving each line, then a single ``None``.
    """
    with stream:
        for line in stream:
            sink.put(line.rstrip("\n"))
    sink.put(None)


def _start_stage_reader(stream: IO[str]) -> queue.Queue[str | None]:
    """Drain progress markers off the probe's stdout so it never blocks on a full pipe.

    :param stream: Child stdout pipe opened in text mode.
    :returns: Queue the reader thread publishes markers to.
    """
    sink: queue.Queue[str | None] = queue.Queue()
    threading.Thread(target=_drain, args=(stream, sink), daemon=True).start()
    return sink


def _await_ready(
    stages: queue.Queue[str | None],
    pid: int,
    *,
    launch_timeout_s: float,
    stall_timeout_s: float,
    cap_s: float,
) -> None:
    """Consume startup stages until the probe reports readiness.

    A stage counts as progressing while it emits markers or accumulates CPU time, so one long
    import is bounded by ``cap_s`` rather than read as a stall. The interval before the first
    marker is launch, not a stage, and is bounded and reported as such.

    :param stages: Queue of stdout progress markers.
    :param pid: Probe process id, sampled for CPU time between markers.
    :param launch_timeout_s: Seconds the probe may take to emit its first marker.
    :param stall_timeout_s: Seconds one startup stage may run without marker or CPU time.
    :param cap_s: Absolute seconds startup may take, however busy the probe is.
    :raises AssertionError: If launch reports nothing, a stage stalls, startup outlasts
        ``cap_s``, or the probe exits.
    """
    reached: list[str] = []
    cpu_seconds = accumulated_cpu_seconds(pid, 0.0)
    last_progress = time.monotonic()
    cap_deadline = last_progress + cap_s
    # Sampled within the tighter bound, or launch CPU from the wrapper lands after the window.
    poll_s = min(_MAX_POLL_SECONDS, stall_timeout_s / 4, launch_timeout_s / 4)
    while True:
        remaining_s = cap_deadline - time.monotonic()
        if remaining_s <= 0:
            raise AssertionError(
                f"probe startup did not reach readiness within {cap_s:g}s after stages {reached}"
            )
        try:
            # Clamped to the cap so a spinning probe is rejected at the cap, not a poll later.
            stage = stages.get(timeout=min(poll_s, remaining_s))
        except queue.Empty:
            sampled = accumulated_cpu_seconds(pid, cpu_seconds)
            if sampled > cpu_seconds:
                cpu_seconds = sampled
                last_progress = time.monotonic()
            elif not reached and time.monotonic() - last_progress >= launch_timeout_s:
                raise AssertionError(
                    f"probe launch produced no output within {launch_timeout_s:g}s"
                ) from None
            elif reached and time.monotonic() - last_progress >= stall_timeout_s:
                raise AssertionError(
                    f"probe startup made no progress for {stall_timeout_s}s "
                    f"after stages {reached} (no marker, no CPU time)"
                ) from None
            continue
        if stage is None:
            raise AssertionError(f"probe exited before readiness after stages {reached}")
        if stage == _READY_MARKER:
            return
        reached.append(stage)
        cpu_seconds = accumulated_cpu_seconds(pid, cpu_seconds)
        last_progress = time.monotonic()


def _await_clean_exit(process: subprocess.Popen[str], behavior_timeout_s: float) -> None:
    """Require the behavior under test to finish inside its own budget and exit zero.

    :param process: Probe that has already reported readiness.
    :param behavior_timeout_s: Seconds the behavior may run.
    :raises AssertionError: If the behavior overruns or the probe exits non-zero.
    """
    try:
        process.wait(timeout=behavior_timeout_s)
    except subprocess.TimeoutExpired:
        raise AssertionError(
            f"probe behavior did not finish within {behavior_timeout_s}s"
        ) from None
    if process.returncode != 0:
        raise AssertionError(f"probe exited {process.returncode}")


def run_fresh_process_probe(
    body: str,
    *,
    launch_timeout_s: float = _LAUNCH_TIMEOUT_SECONDS,
    startup_stall_timeout_s: float = _STARTUP_STALL_TIMEOUT_SECONDS,
    startup_cap_s: float = _STARTUP_CAP_SECONDS,
    behavior_timeout_s: float = _BEHAVIOR_TIMEOUT_SECONDS,
) -> None:
    """Run ``body`` in a fresh interpreter and require it to exit zero.

    ``body`` calls the injected ``progress(stage)`` after each heavyweight import
    and ``ready()`` once its dependencies are loaded; the behavior under test
    then runs against its own budget.

    :param body: Probe source, dedented before execution.
    :param launch_timeout_s: Seconds the probe may take to emit its first marker.
    :param startup_stall_timeout_s: Seconds one startup stage may run without marker or CPU time.
    :param startup_cap_s: Absolute seconds startup may take, however busy the probe is.
    :param behavior_timeout_s: Seconds the probe may run after reporting readiness.
    :raises AssertionError: If startup stalls, behavior overruns, or the probe exits non-zero.
    """
    # stderr goes to a file rather than a second drained pipe so the whole child
    # transcript is readable after a kill, without racing the reader thread.
    with tempfile.TemporaryFile("w+", encoding="utf-8") as stderr_file:
        process = subprocess.Popen(  # noqa: S603 - test-owned argv
            [sys.executable, "-c", _PROBE_PRELUDE + dedent(body)],
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
        )
        with process:
            assert process.stdout is not None
            try:
                _await_ready(
                    _start_stage_reader(process.stdout),
                    process.pid,
                    launch_timeout_s=launch_timeout_s,
                    stall_timeout_s=startup_stall_timeout_s,
                    cap_s=startup_cap_s,
                )
                _await_clean_exit(process, behavior_timeout_s)
            except AssertionError as failure:
                process.kill()
                stderr_file.seek(0)
                raise AssertionError(f"{failure}; probe stderr:\n{stderr_file.read()}") from None
