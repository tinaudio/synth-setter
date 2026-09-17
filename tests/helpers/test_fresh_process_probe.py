"""Startup/behavior bound semantics of the fresh-process probe runner."""

from __future__ import annotations

import stat
import sys
import time
from pathlib import Path

import pytest

from tests.helpers.fresh_process_probe import run_fresh_process_probe

# Long enough that waiting the probe out instead of killing it is unmistakable.
_STALLED_PROBE_SLEEP_SECONDS = 30.0
# Four reporting stages at a quarter of the bound each: their sum exceeds one
# bound, so a cumulative deadline fails, while a loaded runner would have to
# stretch a single gap fourfold to produce the false red this module exists to
# prevent.
_STAGE_STALL_BOUND_SECONDS = 2.0
_STAGE_GAP_SECONDS = _STAGE_STALL_BOUND_SECONDS / 4
# A silent import that outlasts the stall bound several times over: the gap a
# real dependency opens on a saturated runner (#3666), not a marginal overshoot.
_SILENT_BUSY_SECONDS = 1.5
_BUSY_STALL_BOUND_SECONDS = 0.3
# Work before the first marker: on a loaded runner interpreter startup alone
# outlasts a sub-second bound, and the failure then names no stage at all.
_PRE_MARKER_BUSY_SECONDS = 1.0
# A launch that is descheduled rather than working: it accrues no CPU, so the
# CPU-progress bound cannot tell it from a hang (#3673).
_STARVED_LAUNCH_SECONDS = 1.0


def _starved_launch_interpreter(directory: Path, seconds: float) -> str:
    """Write an interpreter wrapper that sleeps before exec'ing the real one.

    The delay consumes no CPU, so it reproduces a runner that descheduled the probe rather than one
    that is slow because it is working.

    :param directory: Directory the wrapper is written to.
    :param seconds: Seconds to sleep before handing over to the interpreter.
    :returns: Path to the wrapper, usable as an interpreter.
    """
    wrapper = directory / "starved-launch"
    wrapper.write_text(f'#!/bin/sh\nsleep {seconds}\nexec {sys.executable} "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return str(wrapper)


def test_run_fresh_process_probe_steady_startup_progress_outlasts_the_stall_bound() -> None:
    """Startup longer than one stall bound passes while every stage keeps reporting."""
    run_fresh_process_probe(
        f"""
        import time

        for stage in ("torch", "torchsynth", "pyFDN", "flamo"):
            time.sleep({_STAGE_GAP_SECONDS})
            progress(stage)
        ready()
        """,
        startup_stall_timeout_s=_STAGE_STALL_BOUND_SECONDS,
        behavior_timeout_s=_STALLED_PROBE_SLEEP_SECONDS,
    )


def test_run_fresh_process_probe_stalled_startup_stage_fails_naming_the_last_stage() -> None:
    """A startup stage that stops reporting is rejected against the stall bound."""
    started_at = time.monotonic()
    with pytest.raises(AssertionError, match=r"startup.*no progress.*torch") as failure:
        run_fresh_process_probe(
            f"""
            import time

            progress("torch")
            time.sleep({_STALLED_PROBE_SLEEP_SECONDS})
            ready()
            """,
            startup_stall_timeout_s=0.3,
            behavior_timeout_s=_STALLED_PROBE_SLEEP_SECONDS,
        )
    assert "probe-ready" not in str(failure.value)
    # The stalled probe is killed rather than waited out, so the bound is the budget.
    assert time.monotonic() - started_at < _STALLED_PROBE_SLEEP_SECONDS


def test_run_fresh_process_probe_hung_behavior_fails_against_the_behavior_bound() -> None:
    """Work after readiness is bounded on its own, not on the startup allowance."""
    started_at = time.monotonic()
    with pytest.raises(AssertionError, match="behavior"):
        run_fresh_process_probe(
            f"""
            import time

            ready()
            time.sleep({_STALLED_PROBE_SLEEP_SECONDS})
            """,
            startup_stall_timeout_s=_STALLED_PROBE_SLEEP_SECONDS,
            behavior_timeout_s=0.3,
        )
    assert time.monotonic() - started_at < _STALLED_PROBE_SLEEP_SECONDS


def test_run_fresh_process_probe_failing_probe_reports_its_stderr() -> None:
    """A non-zero probe exit surfaces the child traceback instead of a bare exit code."""
    with pytest.raises(AssertionError, match="expected probe failure"):
        run_fresh_process_probe(
            """
            ready()
            raise RuntimeError("expected probe failure")
            """,
            startup_stall_timeout_s=5.0,
            behavior_timeout_s=5.0,
        )


def test_run_fresh_process_probe_silent_busy_stage_counts_cpu_time_as_progress() -> None:
    """A stage burning CPU inside one long import outlasts the stall bound and passes."""
    run_fresh_process_probe(
        f"""
        import time

        progress("torch")
        deadline = time.monotonic() + {_SILENT_BUSY_SECONDS}
        while time.monotonic() < deadline:
            pass
        ready()
        """,
        startup_stall_timeout_s=_BUSY_STALL_BOUND_SECONDS,
        behavior_timeout_s=_STALLED_PROBE_SLEEP_SECONDS,
    )


def test_run_fresh_process_probe_spinning_startup_fails_against_the_absolute_cap() -> None:
    """CPU time buys a stage time, not an unbounded startup: the cap still kills a spinner."""
    started_at = time.monotonic()
    with pytest.raises(AssertionError, match=r"startup did not reach readiness within"):
        run_fresh_process_probe(
            """
            progress("torch")
            while True:
                pass
            """,
            startup_stall_timeout_s=_STALLED_PROBE_SLEEP_SECONDS,
            startup_cap_s=_BUSY_STALL_BOUND_SECONDS,
            behavior_timeout_s=_STALLED_PROBE_SLEEP_SECONDS,
        )
    assert time.monotonic() - started_at < _STALLED_PROBE_SLEEP_SECONDS


def test_run_fresh_process_probe_busy_start_before_the_first_marker_is_not_a_stall() -> None:
    """Startup CPU spent before any marker counts, so an empty stage list is never the verdict."""
    run_fresh_process_probe(
        f"""
        import time

        deadline = time.monotonic() + {_PRE_MARKER_BUSY_SECONDS}
        while time.monotonic() < deadline:
            pass
        progress("torch")
        ready()
        """,
        startup_stall_timeout_s=_BUSY_STALL_BOUND_SECONDS,
        behavior_timeout_s=_STALLED_PROBE_SLEEP_SECONDS,
    )


def test_run_fresh_process_probe_starved_launch_is_not_reported_as_a_stage_stall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Launch outlasting the stage bound without CPU is still launch, and passes.

    :param tmp_path: Directory the interpreter wrapper is written to.
    :param monkeypatch: Fixture redirecting the probe at that wrapper.
    """
    monkeypatch.setattr(
        sys, "executable", _starved_launch_interpreter(tmp_path, _STARVED_LAUNCH_SECONDS)
    )

    run_fresh_process_probe(
        """
        progress("torch")
        ready()
        """,
        startup_stall_timeout_s=_BUSY_STALL_BOUND_SECONDS,
        behavior_timeout_s=_STALLED_PROBE_SLEEP_SECONDS,
    )


def test_run_fresh_process_probe_silent_launch_fails_naming_launch_not_a_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe that reports nothing is rejected against the launch bound, by name.

    :param tmp_path: Directory the interpreter wrapper is written to.
    :param monkeypatch: Fixture redirecting the probe at that wrapper.
    """
    monkeypatch.setattr(
        sys, "executable", _starved_launch_interpreter(tmp_path, _STARVED_LAUNCH_SECONDS)
    )

    with pytest.raises(AssertionError, match=r"launch.*no output") as failure:
        run_fresh_process_probe(
            """
            progress("torch")
            ready()
            """,
            launch_timeout_s=_BUSY_STALL_BOUND_SECONDS,
            startup_stall_timeout_s=_STALLED_PROBE_SLEEP_SECONDS,
            behavior_timeout_s=_STALLED_PROBE_SLEEP_SECONDS,
        )

    # The two verdicts are told apart by the message, not by the empty stage list.
    assert "no progress" not in str(failure.value)
