"""Run a fresh-interpreter probe with dependency startup bounded apart from behavior.

A probe that imports torch, torchsynth and FLAMO spends most of its wall clock before the behavior
under test begins, so a single whole-process timeout turns a saturated CI runner into a false "the
loader hung" report (#3455 on macOS, #3500 on Ubuntu). Probes here report each startup stage
instead, and a stage is only rejected once it stops making progress.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import tempfile
import threading
from textwrap import dedent
from typing import IO

_STARTUP_STALL_TIMEOUT_SECONDS = 30.0
_BEHAVIOR_TIMEOUT_SECONDS = 30.0
_READY_MARKER = "probe-ready"

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


def _await_ready(stages: queue.Queue[str | None], stall_timeout_s: float) -> None:
    """Consume startup stages until the probe reports readiness.

    :param stages: Queue of stdout progress markers.
    :param stall_timeout_s: Seconds one startup stage may run without reporting.
    :raises AssertionError: If a stage stalls or the probe exits before readiness.
    """
    reached: list[str] = []
    while True:
        try:
            stage = stages.get(timeout=stall_timeout_s)
        except queue.Empty:
            raise AssertionError(
                f"probe startup made no progress for {stall_timeout_s}s after stages {reached}"
            ) from None
        if stage is None:
            raise AssertionError(f"probe exited before readiness after stages {reached}")
        if stage == _READY_MARKER:
            return
        reached.append(stage)


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
    startup_stall_timeout_s: float = _STARTUP_STALL_TIMEOUT_SECONDS,
    behavior_timeout_s: float = _BEHAVIOR_TIMEOUT_SECONDS,
) -> None:
    """Run ``body`` in a fresh interpreter and require it to exit zero.

    ``body`` calls the injected ``progress(stage)`` after each heavyweight import
    and ``ready()`` once its dependencies are loaded; the behavior under test
    then runs against its own budget.

    :param body: Probe source, dedented before execution.
    :param startup_stall_timeout_s: Seconds one startup stage may run without reporting.
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
                _await_ready(_start_stage_reader(process.stdout), startup_stall_timeout_s)
                _await_clean_exit(process, behavior_timeout_s)
            except AssertionError as failure:
                process.kill()
                stderr_file.seek(0)
                raise AssertionError(f"{failure}; probe stderr:\n{stderr_file.read()}") from None
