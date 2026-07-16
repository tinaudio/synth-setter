"""Bootstrap-retry tests for ``run-linux-vst-headless.sh`` (#2035).

Concurrent shard renders each spawn their own Xvfb; under startup contention
an instance can lose the display-lock race or miss the readiness window, and
the wrapper must retry the bootstrap instead of failing the whole renderer
subprocess. These tests run the real script with stub X binaries on ``PATH``
that simulate those losses deterministically, so they stay in the fast suite
on any platform.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.conftest import VST_HEADLESS_WRAPPER

# The wrapper always passes ``-displayfd 3``; stubs write the display there.
_XVFB_STUB = """\
#!/usr/bin/env bash
n=$(cat "$XVFB_STUB_DIR/xvfb_calls" 2>/dev/null || echo 0)
n=$((n + 1))
echo "$n" > "$XVFB_STUB_DIR/xvfb_calls"
echo $$ >> "$XVFB_STUB_DIR/xvfb_pids"
if [ "$n" -le "${XVFB_STUB_FAILS:-0}" ]; then
  echo "stub: lost display lock race" >&2
  exit 1
fi
echo 99 >&3
exec sleep 600
"""

# Succeeds only once Xvfb has been invoked at least MIN times, so a test can
# force a readiness timeout on the first bootstrap attempt only.
_XDPYINFO_STUB = """\
#!/usr/bin/env bash
n=$(cat "$XVFB_STUB_DIR/xvfb_calls" 2>/dev/null || echo 0)
[ "$n" -ge "${XDPYINFO_STUB_MIN_XVFB_CALLS:-0}" ] || exit 1
exit 0
"""

_DAEMON_STUB = """\
#!/usr/bin/env bash
exec sleep 600
"""

_DBUS_STUB = """\
#!/usr/bin/env bash
[ "$1" = "--" ] && shift
exec "$@"
"""


@pytest.fixture
def stub_env(tmp_path: Path) -> dict[str, str]:
    """Build stub X binaries on PATH and the state dir the stubs share.

    :param tmp_path: Per-test dir for the stub bin/ and state files.
    :returns: Environment for running the wrapper against the stubs.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stubs = {
        "Xvfb": _XVFB_STUB,
        "xdpyinfo": _XDPYINFO_STUB,
        "xsettingsd": _DAEMON_STUB,
        "openbox-session": _DAEMON_STUB,
        "dbus-run-session": _DBUS_STUB,
    }
    for name, body in stubs.items():
        stub = bin_dir / name
        stub.write_text(body)
        stub.chmod(0o755)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["XVFB_STUB_DIR"] = str(state_dir)
    # Stubs fail deterministically; jitter only slows the suite down.
    env["XVFB_RETRY_JITTER_MAX"] = "0"
    return env


def _run_wrapper(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run the real wrapper around a command that reports its DISPLAY.

    :param env: Environment prepared by the ``stub_env`` fixture.
    :returns: Completed process with captured stdout/stderr.
    """
    return subprocess.run(  # noqa: S603 — argv is a fixed list of test-owned paths
        [VST_HEADLESS_WRAPPER, "bash", "-c", 'echo "ran-ok DISPLAY=$DISPLAY"'],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _xvfb_calls(env: dict[str, str]) -> int:
    """Read how many times the Xvfb stub was invoked.

    :param env: Environment carrying ``XVFB_STUB_DIR``.
    :returns: Invocation count recorded by the stub.
    """
    return int((Path(env["XVFB_STUB_DIR"]) / "xvfb_calls").read_text())


def _assert_stub_xvfb_pids_dead(env: dict[str, str]) -> None:
    """Assert every stub Xvfb spawned during the run has exited.

    :param env: Environment carrying ``XVFB_STUB_DIR``.
    """
    pids_file = Path(env["XVFB_STUB_DIR"]) / "xvfb_pids"
    for line in pids_file.read_text().splitlines():
        pid = int(line)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_bootstrap_first_attempt_succeeds_runs_command_under_display(
    stub_env: dict[str, str],
) -> None:
    """Happy path: one Xvfb, command runs under the exported DISPLAY.

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    """
    result = _run_wrapper(stub_env)
    assert result.returncode == 0, result.stderr
    assert "ran-ok DISPLAY=:99" in result.stdout
    assert _xvfb_calls(stub_env) == 1


def test_bootstrap_xvfb_dies_once_retries_and_runs_command(
    stub_env: dict[str, str],
) -> None:
    """A single startup death is retried and the command still runs.

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    """
    stub_env["XVFB_STUB_FAILS"] = "1"
    result = _run_wrapper(stub_env)
    assert result.returncode == 0, result.stderr
    assert "ran-ok DISPLAY=:99" in result.stdout
    assert _xvfb_calls(stub_env) == 2


def test_bootstrap_xvfb_dies_every_attempt_fails_without_running_command(
    stub_env: dict[str, str],
) -> None:
    """Exhausting the retry budget fails loudly without running the command.

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    """
    stub_env["XVFB_STUB_FAILS"] = "99"
    result = _run_wrapper(stub_env)
    assert result.returncode != 0
    assert "ran-ok" not in result.stdout
    assert _xvfb_calls(stub_env) == 3
    assert "lost display lock race" in result.stderr


def test_bootstrap_attempts_env_overrides_retry_budget(
    stub_env: dict[str, str],
) -> None:
    """XVFB_BOOTSTRAP_ATTEMPTS=1 restores single-attempt fail-fast.

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    """
    stub_env["XVFB_STUB_FAILS"] = "99"
    stub_env["XVFB_BOOTSTRAP_ATTEMPTS"] = "1"
    result = _run_wrapper(stub_env)
    assert result.returncode != 0
    assert _xvfb_calls(stub_env) == 1


def test_bootstrap_readiness_timeout_retries_and_reaps_stale_xvfb(
    stub_env: dict[str, str],
) -> None:
    """A readiness timeout retries and kills the stale first-attempt Xvfb.

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    """
    stub_env["XDPYINFO_STUB_MIN_XVFB_CALLS"] = "2"
    stub_env["XVFB_READY_PROBES"] = "3"
    result = _run_wrapper(stub_env)
    assert result.returncode == 0, result.stderr
    assert "ran-ok DISPLAY=:99" in result.stdout
    assert _xvfb_calls(stub_env) == 2
    _assert_stub_xvfb_pids_dead(stub_env)
