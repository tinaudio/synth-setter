"""Adversarial behavioural tests for ``check_call_streamed``.

The runner exists so child-process output flows through the parent's
``sys.stderr`` — the only channel wandb ``console=wrap`` captures (#1465,
#1506) — while completion stays keyed on child *exit*, never pipe EOF
(#1634: the headless-VST X11 daemon tree inherits the merged pipe and holds
it open past child exit). Tests drive real children (``sys.executable -c``)
and ``os.fork`` real grandchildren; no part of the SUT is mocked. Every
liveness claim carries an elapsed-time bound so a hang regression fails
fast instead of passing slowly, and an outer ``asyncio.wait_for`` backstop
turns a true wedge into a test failure rather than a stalled CI job.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from synth_setter.pipeline.subprocess_stream import (
    check_call_streamed,
    check_call_streamed_async,
)

# Hard wedge backstop for every SUT invocation: no test scenario legitimately
# runs this long, so hitting it means the exit-keyed wait itself regressed.
_BACKSTOP_SECONDS = 30.0

# Liveness bound for scenarios that should return almost immediately; sized
# for loaded CI runners (the #1617 lesson: tight margins flake — 7070906).
_PROMPT_SECONDS = 10.0

# Timeout passed to the SUT where a test exercises the timeout path; leaves
# the child interpreter startup margin on loaded CI runners.
_POLICY_TIMEOUT_SECONDS = 2.0


def _streamed(
    cmd: list[str], *, timeout: float | None = None, env: dict[str, str] | None = None
) -> bytes:
    """Run the async SUT under the wedge backstop.

    :param cmd: Child argv.
    :param timeout: Policy timeout forwarded to the SUT.
    :param env: Child env forwarded to the SUT.
    :returns: The SUT's captured merged output.
    """
    return asyncio.run(
        asyncio.wait_for(
            check_call_streamed_async(cmd, timeout=timeout, env=env), _BACKSTOP_SECONDS
        )
    )


def _child(script: str, marker: str) -> list[str]:
    """Build a hermetic child argv stamped with the leak marker.

    :param script: Python source for ``-c``.
    :param marker: Leak-sweep stamp appended as inert argv (inherited by forks).
    :returns: argv for the SUT.
    """
    return [sys.executable, "-c", script, marker]


def _pid_alive(pid: int) -> bool:
    """Probe liveness via signal 0; a zombie counts as dead once reaped.

    :param pid: Process id to probe.
    :returns: True when the pid still exists.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_file(path: Path, deadline_seconds: float = 5.0) -> str:
    """Poll for a sentinel/pid file the child promised to write.

    :param path: File the child writes.
    :param deadline_seconds: Bound on the poll.
    :returns: The file's text.
    :raises AssertionError: The file never appeared within the deadline.
    """
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        if path.is_file() and path.read_text():
            return path.read_text()
        time.sleep(0.05)
    raise AssertionError(f"child never wrote {path}")


@pytest.fixture(autouse=True)
def leak_marker() -> Iterator[str]:
    """Stamp every child argv; fail the test if any stamped process survives.

    The sweep waits briefly first: the SUT's post-exit SIGTERM is delivered
    asynchronously, so an immediately-run ``pgrep`` would race the kill.

    :yields str: Unique argv marker for this test's children.
    """
    marker = f"sps-leak-{uuid.uuid4().hex}"
    yield marker
    deadline = time.monotonic() + 3.0
    survivors: list[int] = []
    while time.monotonic() < deadline:
        result = subprocess.run(  # noqa: S603
            ["pgrep", "-f", marker],  # noqa: S607
            capture_output=True,
            text=True,
        )
        survivors = [int(pid) for pid in result.stdout.split()]
        if not survivors:
            break
        time.sleep(0.1)
    for pid in survivors:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    assert not survivors, f"leaked stamped processes: {survivors}"


class TestExitSemantics:
    """The ``check_call`` contract: truthful exit codes and captured output."""

    def test_clean_exit_returns_merged_output(self, leak_marker: str) -> None:
        """Both child streams come back merged in the captured bytes.

        :param leak_marker: argv stamp from the autouse sweep fixture.
        """
        out = _streamed(
            _child(
                "import sys; print('TO_STDOUT'); print('TO_STDERR', file=sys.stderr)", leak_marker
            )
        )

        assert b"TO_STDOUT" in out
        assert b"TO_STDERR" in out

    def test_nonzero_exit_raises_with_code_and_precrash_output(self, leak_marker: str) -> None:
        """Exit 7 surfaces as ``CalledProcessError(7)`` carrying pre-crash output.

        :param leak_marker: argv stamp from the autouse sweep fixture.
        """
        argv = _child("import sys; print('BEFORE_CRASH'); sys.exit(7)", leak_marker)

        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            _streamed(argv)

        assert excinfo.value.returncode == 7
        assert excinfo.value.cmd == argv
        assert b"BEFORE_CRASH" in excinfo.value.output

    def test_signal_death_raises_with_negative_returncode(self, leak_marker: str) -> None:
        """A child killed by SIGKILL reports ``returncode == -9``.

        :param leak_marker: argv stamp from the autouse sweep fixture.
        """
        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            _streamed(
                _child("import os, signal; os.kill(os.getpid(), signal.SIGKILL)", leak_marker)
            )

        assert excinfo.value.returncode == -signal.SIGKILL


class TestPipeDrain:
    """Problem 1/2: concurrent drain and unbuffered child output."""

    def test_large_interleaved_output_no_deadlock_all_bytes_captured(
        self, leak_marker: str
    ) -> None:
        """~2MB interleaved across both streams completes promptly and fully.

        Well past the ~64KB pipe buffer: any refactor that stops draining concurrently with the
        exit wait turns this into a hang, and the elapsed bound converts that hang into a failure.

        :param leak_marker: argv stamp from the autouse sweep fixture.
        """
        script = (
            "import sys\n"
            "for i in range(1000):\n"
            "    sys.stdout.write('o' * 1024 + '\\n')\n"
            "    sys.stderr.write('e' * 1024 + '\\n')\n"
        )

        start = time.monotonic()
        out = _streamed(_child(script, leak_marker))
        elapsed = time.monotonic() - start

        assert len(out) == 2 * 1000 * 1025
        assert elapsed < _PROMPT_SECONDS, f"drain stalled ({elapsed:.1f}s)"

    def test_unflushed_pre_hang_output_visible_on_timeout(self, leak_marker: str) -> None:
        """A hung child's last unflushed line is already captured when killed.

        The #735 diagnosis property: ``PYTHONUNBUFFERED=1`` must reach the
        child, else its pre-hang ``print`` (no explicit flush) sits in a
        full block buffer and the hang looks output-less.

        :param leak_marker: argv stamp from the autouse sweep fixture.
        """
        argv = _child("import time; print('BEFORE_HANG'); time.sleep(60)", leak_marker)

        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            _streamed(argv, timeout=_POLICY_TIMEOUT_SECONDS)

        assert b"BEFORE_HANG" in excinfo.value.output

    def test_env_none_inherits_parent_and_pins_unbuffered(
        self, leak_marker: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``env=None`` inherits the parent env plus ``PYTHONUNBUFFERED=1``.

        :param leak_marker: argv stamp from the autouse sweep fixture.
        :param monkeypatch: Sets a sentinel var in the parent env.
        """
        monkeypatch.setenv("STREAMED_TEST_VAR", "inherited")

        out = _streamed(
            _child(
                "import os; print(os.environ['PYTHONUNBUFFERED'], os.environ['STREAMED_TEST_VAR'])",
                leak_marker,
            )
        )

        assert b"1 inherited" in out

    def test_env_explicit_mapping_used_and_pins_unbuffered(self, leak_marker: str) -> None:
        """An explicit ``env`` replaces inheritance but still pins unbuffered.

        :param leak_marker: argv stamp from the autouse sweep fixture.
        """
        out = _streamed(
            _child(
                "import os; print(os.environ['PYTHONUNBUFFERED'], os.environ['STREAMED_TEST_VAR'], "
                "os.environ.get('STREAMED_ABSENT_VAR', 'absent'))",
                leak_marker,
            ),
            env={"STREAMED_TEST_VAR": "explicit"},
        )

        assert b"1 explicit absent" in out


class TestWrapTee:
    """Problem 3: the tee must go through the wrap-patched text stream."""

    def test_tee_reaches_wrap_patched_stderr_write(
        self, capsys: pytest.CaptureFixture[str], leak_marker: str
    ) -> None:
        """Forwarded lines hit a ``write`` patched in place on ``sys.stderr``.

        Simulates exactly what wandb ``console=wrap`` does: patch ``write``
        on the existing stream object, not swap the object. A pump that
        writes via ``sys.stderr.buffer``, a saved stream reference's
        ``.buffer``, or ``os.write(2, …)`` bypasses the patched method and
        silently re-empties the run's Logs tab — this is the regression pin.

        :param capsys: Keeps ``sys.stderr`` a Python-level patchable object.
        :param leak_marker: argv stamp from the autouse sweep fixture.
        """
        seen: list[str] = []
        orig = sys.stderr.write

        def _recording_write(s: str) -> int:
            seen.append(s)
            return orig(s)

        sys.stderr.write = _recording_write  # type: ignore[method-assign]
        try:
            out = _streamed(
                _child(
                    "import sys; print('TO_STDOUT'); print('TO_STDERR', file=sys.stderr)",
                    leak_marker,
                )
            )
        finally:
            sys.stderr.write = orig  # type: ignore[method-assign]

        joined = "".join(seen)
        assert "TO_STDOUT" in joined
        assert "TO_STDERR" in joined
        assert b"TO_STDOUT" in out

    def test_non_utf8_output_replaced_in_tee_and_raw_in_capture(
        self, capsys: pytest.CaptureFixture[str], leak_marker: str
    ) -> None:
        """Invalid UTF-8 is replacement-charred on the tee, raw in the capture.

        :param capsys: Captures the parent-process ``sys.stderr`` writes.
        :param leak_marker: argv stamp from the autouse sweep fixture.
        """
        out = _streamed(
            _child("import sys; sys.stdout.buffer.write(b'BAD\\xff\\xfeBYTES\\n')", leak_marker)
        )

        assert b"BAD\xff\xfeBYTES\n" in out
        err = capsys.readouterr().err
        assert "BAD" in err
        assert "BYTES" in err
        assert "�" in err


# Child forks a grandchild that inherits and holds the merged-stdout pipe
# (the X11-daemon stand-in, #1634); the direct child finishes immediately.
_PIPE_HOLDING_GRANDCHILD = """
import os, sys, time
if os.fork() == 0:
    time.sleep(60)
    os._exit(0)
print("CHILD_DONE", flush=True)
{child_exit}
"""


class TestExitKeyedLiveness:
    """Problem 4a: completion keys on child exit, never on pipe EOF."""

    def test_pipe_holding_grandchild_exit_zero_returns_promptly(self, leak_marker: str) -> None:
        """A successful child is reported as success despite a pipe-holding daemon.

        Pins the aa8e03e regression both ways: the EOF-keyed design either
        hung here (no timeout to force EOF) or misreported the successful
        child as ``TimeoutExpired``. No timeout is passed on purpose.

        :param leak_marker: argv stamp from the autouse sweep fixture.
        """
        script = _PIPE_HOLDING_GRANDCHILD.format(child_exit="")

        start = time.monotonic()
        out = _streamed(_child(script, leak_marker))
        elapsed = time.monotonic() - start

        assert b"CHILD_DONE" in out
        assert elapsed < _PROMPT_SECONDS, f"exit-keyed wait stalled on EOF ({elapsed:.1f}s)"

    def test_pipe_holding_grandchild_nonzero_exit_truthful_error(self, leak_marker: str) -> None:
        """Exit codes survive daemon loitering in the failure direction too.

        :param leak_marker: argv stamp from the autouse sweep fixture.
        """
        script = _PIPE_HOLDING_GRANDCHILD.format(child_exit="sys.exit(3)")

        start = time.monotonic()
        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            _streamed(_child(script, leak_marker))
        elapsed = time.monotonic() - start

        assert excinfo.value.returncode == 3
        assert elapsed < _PROMPT_SECONDS, f"exit-keyed wait stalled on EOF ({elapsed:.1f}s)"

    def test_long_runner_without_timeout_not_cut_short(self, leak_marker: str) -> None:
        """Exit-keying is not impatience: a slow child runs to completion.

        The post-exit drain grace must never apply to a live child; all six
        lines emitted over ~3s must come back.

        :param leak_marker: argv stamp from the autouse sweep fixture.
        """
        script = (
            "import time\n"
            "for i in range(6):\n"
            "    print(f'TICK_{i}', flush=True)\n"
            "    time.sleep(0.5)\n"
        )

        out = _streamed(_child(script, leak_marker))

        assert all(f"TICK_{i}".encode() in out for i in range(6))


class TestTimeoutEscalation:
    """Policy timeouts: SIGTERM first (#1634), SIGKILL only for the wedged."""

    def test_timeout_sigterms_child_cleanup_runs_before_kill(self, leak_marker: str) -> None:
        """The child's SIGTERM handler runs and its output is still drained.

        The trap-sentinel property from #1634: a bare group SIGKILL skipped the headless-VST
        wrapper's EXIT trap and leaked X11 sockets that corrupted *later* renders.

        :param leak_marker: argv stamp from the autouse sweep fixture.
        """
        script = (
            "import signal, sys, time\n"
            "def bye(sig, frame):\n"
            "    print('CLEANUP_RAN', flush=True)\n"
            "    sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, bye)\n"
            "print('READY', flush=True)\n"
            "time.sleep(30)\n"
        )

        start = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            _streamed(_child(script, leak_marker), timeout=_POLICY_TIMEOUT_SECONDS)
        elapsed = time.monotonic() - start

        assert b"CLEANUP_RAN" in excinfo.value.output
        assert elapsed < _PROMPT_SECONDS, f"TERM path overran ({elapsed:.1f}s)"

    def test_timeout_sigterm_ignoring_child_escalates_to_sigkill(self, leak_marker: str) -> None:
        """A TERM-ignoring child is SIGKILLed after the escalation grace.

        :param leak_marker: argv stamp from the autouse sweep fixture.
        """
        script = (
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "print('READY', flush=True)\n"
            "time.sleep(60)\n"
        )

        start = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired):
            _streamed(_child(script, leak_marker), timeout=_POLICY_TIMEOUT_SECONDS)
        elapsed = time.monotonic() - start

        # timeout (2s) + grace (5s) + CI margin.
        assert elapsed < 12.0, f"SIGKILL escalation overran ({elapsed:.1f}s)"

    def test_timeout_with_sigterm_ignoring_pipe_holding_grandchild(
        self, tmp_path: Path, leak_marker: str
    ) -> None:
        """9760b71's regression shape: escalation is gated on exit, not drain.

        The grandchild ignores SIGTERM *and* holds the pipe; the child
        sleeps past the timeout. ``TimeoutExpired`` must arrive within the
        escalation bound regardless of the undrainable pipe. The surviving
        TERM-immune grandchild is a documented 4b limitation, so the test
        reaps it itself via pidfile.

        :param tmp_path: Holds the grandchild pidfile.
        :param leak_marker: argv stamp from the autouse sweep fixture.
        """
        pidfile = tmp_path / "grandchild.pid"
        script = (
            "import os, signal, time\n"
            "if os.fork() == 0:\n"
            "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"    open({str(pidfile)!r}, 'w').write(str(os.getpid()))\n"
            "    time.sleep(60)\n"
            "    os._exit(0)\n"
            "time.sleep(60)\n"
        )

        try:
            start = time.monotonic()
            with pytest.raises(subprocess.TimeoutExpired):
                _streamed(_child(script, leak_marker), timeout=_POLICY_TIMEOUT_SECONDS)
            elapsed = time.monotonic() - start

            assert elapsed < 12.0, f"escalation gated on pipe drain, not exit ({elapsed:.1f}s)"
        finally:
            if pidfile.is_file():
                with contextlib.suppress(ProcessLookupError, ValueError):
                    os.kill(int(pidfile.read_text()), signal.SIGKILL)

    def test_timeout_group_kill_stops_ingroup_grandchild_heartbeat(
        self, tmp_path: Path, leak_marker: str
    ) -> None:
        """The timeout kill reaches in-group descendants, not just the leader.

        Distinct from "didn't hang": a heartbeat file that keeps growing after the raise proves a
        leaked worker.

        :param tmp_path: Holds the grandchild heartbeat file.
        :param leak_marker: argv stamp from the autouse sweep fixture.
        """
        heartbeat = tmp_path / "heartbeat"
        script = (
            "import os, time\n"
            "if os.fork() == 0:\n"
            "    while True:\n"
            f"        open({str(heartbeat)!r}, 'a').write('x')\n"
            "        time.sleep(0.2)\n"
            "time.sleep(60)\n"
        )

        with pytest.raises(subprocess.TimeoutExpired):
            _streamed(_child(script, leak_marker), timeout=_POLICY_TIMEOUT_SECONDS)

        _wait_for_file(heartbeat)
        time.sleep(1.0)
        size_after_kill = heartbeat.stat().st_size
        time.sleep(1.0)
        assert heartbeat.stat().st_size == size_after_kill, "grandchild still heartbeating"


class TestPostExitSweep:
    """Problem 4b: the unconditional post-exit in-group SIGTERM sweep."""

    def test_clean_exit_sweep_reaps_ingroup_grandchild(
        self, tmp_path: Path, leak_marker: str
    ) -> None:
        """An in-group grandchild outliving a *successful* child is swept.

        Nothing on the timeout path exercises this: the sweep after a clean
        exit is the only thing standing between a daemonized descendant and
        an infinite leak.

        :param tmp_path: Holds the grandchild pidfile.
        :param leak_marker: argv stamp from the autouse sweep fixture.
        """
        pidfile = tmp_path / "grandchild.pid"
        script = (
            "import os, time\n"
            "if os.fork() == 0:\n"
            f"    open({str(pidfile)!r}, 'w').write(str(os.getpid()))\n"
            "    time.sleep(60)\n"
            "    os._exit(0)\n"
        )

        _streamed(_child(script, leak_marker))

        grandchild_pid = int(_wait_for_file(pidfile))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and _pid_alive(grandchild_pid):
            time.sleep(0.05)
        assert not _pid_alive(grandchild_pid), "post-exit sweep missed the in-group grandchild"

    @pytest.mark.xfail(
        reason="known 4b escape gap: a setsid() grandchild leaves the killable "
        "group; flips when a Linux cgroup scope lands"
    )
    def test_escaped_setsid_grandchild_is_reaped(self, tmp_path: Path, leak_marker: str) -> None:
        """A grandchild that escapes the group via ``setsid()`` should die too.

        Encodes the accepted limitation as xfail so the eventual cgroup fix flips it deliberately
        and visibly.

        :param tmp_path: Holds the grandchild pidfile.
        :param leak_marker: argv stamp from the autouse sweep fixture.
        """
        pidfile = tmp_path / "grandchild.pid"
        script = (
            "import os, time\n"
            "if os.fork() == 0:\n"
            "    os.setsid()\n"
            f"    open({str(pidfile)!r}, 'w').write(str(os.getpid()))\n"
            "    time.sleep(60)\n"
            "    os._exit(0)\n"
        )

        try:
            _streamed(_child(script, leak_marker))
            grandchild_pid = int(_wait_for_file(pidfile))
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and _pid_alive(grandchild_pid):
                time.sleep(0.05)
            assert not _pid_alive(grandchild_pid), "setsid grandchild escaped the sweep"
        finally:
            if pidfile.is_file():
                with contextlib.suppress(ProcessLookupError, ValueError):
                    os.kill(int(pidfile.read_text()), signal.SIGKILL)


class TestSyncFacade:
    """The blocking wrapper production call sites use."""

    def test_sync_facade_returns_output_and_raises_like_async(self, leak_marker: str) -> None:
        """The facade mirrors the async contract: capture and raise semantics.

        :param leak_marker: argv stamp from the autouse sweep fixture.
        """
        out = check_call_streamed(_child("print('VIA_FACADE')", leak_marker))
        assert b"VIA_FACADE" in out

        with pytest.raises(subprocess.CalledProcessError):
            check_call_streamed(_child("import sys; sys.exit(5)", leak_marker))

    def test_sync_facade_runs_from_worker_thread(self, leak_marker: str) -> None:
        """The facade works off the main thread.

        Production runs it inside ``_render_and_upload_shard`` worker
        threads, where asyncio's child watcher must not depend on the main
        thread's signal handling.

        :param leak_marker: argv stamp from the autouse sweep fixture.
        """
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(check_call_streamed, _child("print('FROM_THREAD')", leak_marker))
            out = future.result(timeout=_BACKSTOP_SECONDS)

        assert b"FROM_THREAD" in out
