"""Behavioural tests for ``_check_call_streamed``.

The helper exists so child-process output flows through the parent's
``sys.stderr`` — the only channel wandb ``console=wrap`` captures (#1465,
#1506). Tests drive real child processes (``sys.executable -c``) and assert
on the streamed output via ``capsys``; no part of the SUT is mocked.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from synth_setter.cli.generate_dataset import _check_call_streamed


class TestCheckCallStreamed:
    """Run a child process and tee its merged output through ``sys.stderr``."""

    def test_child_stdout_and_stderr_stream_to_parent_stderr(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Both child streams land on the parent's ``sys.stderr``.

        :param capsys: Captures the parent-process ``sys.stderr`` writes.
        """
        _check_call_streamed(
            [
                sys.executable,
                "-c",
                "import sys; print('CHILD_STDOUT'); sys.stderr.write('CHILD_STDERR\\n')",
            ]
        )

        err = capsys.readouterr().err
        assert "CHILD_STDOUT" in err
        assert "CHILD_STDERR" in err

    def test_zero_exit_returns_none(self) -> None:
        """A clean child exit returns without raising."""
        assert _check_call_streamed([sys.executable, "-c", "pass"]) is None

    def test_nonzero_exit_raises_calledprocesserror_with_returncode(self) -> None:
        """A non-zero child exit raises ``CalledProcessError`` carrying the code."""
        argv = [sys.executable, "-c", "import sys; sys.exit(3)"]

        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            _check_call_streamed(argv)

        assert excinfo.value.returncode == 3
        assert excinfo.value.cmd == argv

    def test_output_before_failure_is_still_streamed(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Output emitted before a failing exit reaches the parent's stderr.

        :param capsys: Captures the parent-process ``sys.stderr`` writes.
        """
        with pytest.raises(subprocess.CalledProcessError):
            _check_call_streamed(
                [sys.executable, "-c", "import sys; print('BEFORE_CRASH'); sys.exit(1)"]
            )

        assert "BEFORE_CRASH" in capsys.readouterr().err

    def test_timeout_kills_child_and_raises_timeoutexpired(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A child outliving ``timeout`` is killed; pre-hang output was streamed.

        The pre-hang assertion pins the #735 diagnosis property: a hung child's
        last lines must already be on the parent's stderr when it is killed.

        :param capsys: Captures the parent-process ``sys.stderr`` writes.
        """
        argv = [
            sys.executable,
            "-c",
            "print('BEFORE_HANG', flush=True); import time; time.sleep(60)",
        ]

        # 2s leaves the child interpreter startup margin on loaded CI runners;
        # a kill before the print would fail the BEFORE_HANG assertion below.
        with pytest.raises(subprocess.TimeoutExpired):
            _check_call_streamed(argv, timeout=2.0)

        assert "BEFORE_HANG" in capsys.readouterr().err

    def test_zero_exit_under_timeout_returns_none(self) -> None:
        """A child finishing well inside ``timeout`` returns cleanly (timer cancelled)."""
        assert _check_call_streamed([sys.executable, "-c", "pass"], timeout=30) is None

    def test_child_env_keeps_parent_vars_and_sets_unbuffered(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The child sees the parent env plus ``PYTHONUNBUFFERED=1``.

        Callers depend on inherited env (``PATH``, ``WANDB_*`` for rclone/eval
        children) and on the unbuffered pin for live hang diagnosis (#735).

        :param capsys: Captures the parent-process ``sys.stderr`` writes.
        :param monkeypatch: Sets a sentinel var in the parent env.
        """
        monkeypatch.setenv("STREAMED_TEST_VAR", "carried")

        _check_call_streamed(
            [
                sys.executable,
                "-c",
                "import os; print(os.environ['PYTHONUNBUFFERED'], "
                "os.environ['STREAMED_TEST_VAR'])",
            ]
        )

        assert "1 carried" in capsys.readouterr().err

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork (POSIX)")
    def test_timeout_reaps_pipe_holding_grandchild_and_raises(self) -> None:
        """A pipe-holding grandchild raises ``TimeoutExpired``, not a hang.

        Direct child exits 0; the process-group kill reaps the grandchild so the read loop
        unblocks. The timer firing means the wall-clock budget was exceeded, so the call surfaces a
        timeout rather than masking it as success. The elapsed bound turns a kill regression (which
        would otherwise stall until the grandchild's 60s sleep ends) into a failure; no pytest-
        level timeout exists to catch the stall.
        """
        argv = [
            sys.executable,
            "-c",
            "import os, time, sys\n"
            "if os.fork() == 0:\n"
            "    time.sleep(60)\n"  # grandchild keeps the inherited pipe open
            "else:\n"
            "    sys.exit(0)\n",  # direct child exits at once
        ]

        start = time.monotonic()
        # 3s gives the child startup margin to fork-and-exit before the kill.
        with pytest.raises(subprocess.TimeoutExpired):
            _check_call_streamed(argv, timeout=3.0)
        elapsed = time.monotonic() - start

        assert elapsed < 30, f"group kill did not unblock the read loop ({elapsed:.1f}s)"

    def test_timeout_sigterms_group_so_child_can_clean_up_before_kill(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The timeout kill sends SIGTERM first so a child cleanup handler runs.

        The headless-VST wrapper reaps its Xvfb/dbus tree and removes the X
        socket from a ``trap cleanup`` that only fires on SIGTERM (not SIGKILL);
        a bare SIGKILL would orphan that tree and leak the socket. A child that
        traps SIGTERM, emits a sentinel, and exits stands in for that wrapper.

        :param capsys: Captures the parent-process ``sys.stderr`` writes.
        """
        argv = [
            sys.executable,
            "-c",
            "import signal, sys, time\n"
            "def _bye(*_):\n"
            "    print('GRACEFUL_CLEANUP', flush=True)\n"
            "    sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, _bye)\n"
            "print('READY', flush=True)\n"
            "time.sleep(60)\n",
        ]

        with pytest.raises(subprocess.TimeoutExpired):
            _check_call_streamed(argv, timeout=2.0)

        assert "GRACEFUL_CLEANUP" in capsys.readouterr().err

    def test_timeout_sigkills_group_that_ignores_sigterm(self) -> None:
        """A SIGTERM-ignoring child is still hard-killed within the grace bound.

        The escalation is what keeps the timeout real: without the SIGKILL
        fallback a child that ignores SIGTERM would block the read loop until
        its own 60s sleep ends. The elapsed bound turns a missing fallback into
        a failure rather than a 60s stall.
        """
        argv = [
            sys.executable,
            "-c",
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "print('IGNORING_SIGTERM', flush=True)\n"
            "time.sleep(60)\n",
        ]

        start = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired):
            _check_call_streamed(argv, timeout=2.0)
        elapsed = time.monotonic() - start

        # Worst case is timeout(2) + grace(5) + kill ≈ 7s; 15s catches a grace
        # regression without risking flakiness on a loaded runner.
        assert elapsed < 15, f"SIGKILL fallback did not unblock the read loop ({elapsed:.1f}s)"

    def test_non_utf8_child_output_does_not_crash(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Invalid UTF-8 from the child is replaced, not fatal.

        :param capsys: Captures the parent-process ``sys.stderr`` writes.
        """
        _check_call_streamed(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'BAD\\xff\\xfeBYTES\\n')",
            ]
        )

        err = capsys.readouterr().err
        assert "BAD" in err
        assert "BYTES" in err
