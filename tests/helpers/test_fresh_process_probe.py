"""Startup/behavior bound semantics of the fresh-process probe runner."""

from __future__ import annotations

import time

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
