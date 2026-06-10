"""Behavioural tests for ``_check_call_streamed``.

The helper exists so child-process output flows through the parent's
``sys.stderr`` — the only channel wandb ``console=wrap`` captures (#1465,
#1506). Tests drive real child processes (``sys.executable -c``) and assert
on the streamed output via ``capsys``; no part of the SUT is mocked.
"""

from __future__ import annotations

import subprocess
import sys

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

    def test_timeout_kills_child_and_raises_timeoutexpired(self) -> None:
        """A child outliving ``timeout`` is killed and ``TimeoutExpired`` raised."""
        argv = [sys.executable, "-c", "import time; time.sleep(60)"]

        with pytest.raises(subprocess.TimeoutExpired):
            _check_call_streamed(argv, timeout=0.5)

    def test_timeout_reaps_pipe_holding_grandchild_without_hanging(self) -> None:
        """A grandchild inheriting the pipe can't hang the call past the timeout.

        The direct child exits 0 but forks a grandchild that holds the merged stdout pipe open for
        60s. Reading to EOF would block far past the timeout; the process-group kill reaps the
        grandchild so the read loop unblocks and the call returns promptly. The direct child
        succeeded, so this is a clean return — the regression guarded here is the *hang*, which
        would trip the test harness's own wall-clock timeout instead.
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

        assert _check_call_streamed(argv, timeout=1.0) is None

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
