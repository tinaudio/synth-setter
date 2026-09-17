"""Bootstrap-retry tests for ``run-linux-vst-headless.sh`` (#2035).

Runs the real script with stub X binaries on ``PATH`` that simulate startup
contention losses deterministically, so the tests stay in the fast suite on
any platform.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.conftest import VST_HEADLESS_WRAPPER

# The wrapper always passes ``-displayfd 3``; stubs write the display there.
_XVFB_STUB = """\
#!/bin/bash
set -euo pipefail
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
#!/bin/bash
set -euo pipefail
n=$(cat "$XVFB_STUB_DIR/xvfb_calls" 2>/dev/null || echo 0)
if [[ "$n" -lt "${XDPYINFO_STUB_MIN_XVFB_CALLS:-0}" ]]; then
  exit 1
fi
exit 0
"""

_DAEMON_STUB = """\
#!/bin/bash
set -euo pipefail
exec sleep 600
"""

# Mirrors xsettingsd: it announces the selection it took on its own output, and
# exits when it cannot reach the display. ``XSETTINGSD_STUB_SILENT`` keeps it
# alive without ever announcing ownership.
_XSETTINGSD_STUB = """\
#!/bin/bash
set -euo pipefail
n=$(cat "$XVFB_STUB_DIR/xsettingsd_calls" 2>/dev/null || echo 0)
n=$((n + 1))
echo "$n" > "$XVFB_STUB_DIR/xsettingsd_calls"
echo "xsettingsd: Loaded 0 settings from /dev/null"
if [ "$n" -le "${XSETTINGSD_STUB_FAILS:-0}" ]; then
  echo "xsettingsd: Unable to open connection to X server"
  exit 1
fi
if [ "${XSETTINGSD_STUB_SILENT:-0}" != "1" ]; then
  echo "xsettingsd: Took ownership of selection _XSETTINGS_S0"
fi
exec sleep 600
"""

_DBUS_STUB = """\
#!/bin/bash
set -euo pipefail
if [[ "${1-}" == "--" ]]; then
  shift
fi
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
        "xsettingsd": _XSETTINGSD_STUB,
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


def _xsettingsd_calls(env: dict[str, str]) -> int:
    """Read how many times the xsettingsd stub was invoked.

    :param env: Environment carrying ``XVFB_STUB_DIR``.
    :returns: Invocation count recorded by the stub.
    """
    return int((Path(env["XVFB_STUB_DIR"]) / "xsettingsd_calls").read_text())


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
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            pytest.fail(f"stub Xvfb pid={pid} still exists (PermissionError)")
        pytest.fail(f"stub Xvfb pid={pid} still exists")


def _path_without_xsettingsd(env: dict[str, str], sysbin: Path) -> str:
    """Mirror the environment's PATH into one directory, minus ``xsettingsd``.

    A developer host has the real daemon installed, so deleting the stub only uncovers it; masking
    the name needs a PATH that resolves everything else.

    :param env: Environment whose PATH is mirrored.
    :param sysbin: Directory to populate with the mirrored executables.
    :returns: PATH holding the stub directory and the xsettingsd-free mirror.
    """
    sysbin.mkdir()
    stub_dir, *system_dirs = env["PATH"].split(os.pathsep)
    for directory in system_dirs:
        for entry in Path(directory).glob("*"):
            if entry.name == "xsettingsd" or (sysbin / entry.name).exists():
                continue
            (sysbin / entry.name).symlink_to(entry)
    return os.pathsep.join([stub_dir, str(sysbin)])


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
    """A single-attempt override restores fail-fast behavior.

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    """
    stub_env["XVFB_STUB_FAILS"] = "99"
    stub_env["XVFB_BOOTSTRAP_ATTEMPTS"] = "1"
    result = _run_wrapper(stub_env)
    assert result.returncode != 0
    assert _xvfb_calls(stub_env) == 1


def test_bootstrap_succeeds_on_final_attempt_of_budget(
    stub_env: dict[str, str],
) -> None:
    """Success on the last permitted attempt still runs the command.

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    """
    stub_env["XVFB_STUB_FAILS"] = "2"
    result = _run_wrapper(stub_env)
    assert result.returncode == 0, result.stderr
    assert "ran-ok DISPLAY=:99" in result.stdout
    assert _xvfb_calls(stub_env) == 3


def test_bootstrap_non_numeric_attempts_falls_back_to_default(
    stub_env: dict[str, str],
) -> None:
    """A malformed retry-budget override degrades to the script's default.

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    """
    stub_env["XVFB_STUB_FAILS"] = "99"
    stub_env["XVFB_BOOTSTRAP_ATTEMPTS"] = "not-a-number"
    result = _run_wrapper(stub_env)
    assert result.returncode != 0
    assert _xvfb_calls(stub_env) == 3


@pytest.mark.parametrize("override", ["00", "000"])
def test_bootstrap_zero_padded_zero_attempts_falls_back_to_default(
    stub_env: dict[str, str], override: str
) -> None:
    """A zero-padded zero retry budget degrades to the script's default.

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    :param override: Zero-padded retry-budget override.
    """
    stub_env["XVFB_STUB_FAILS"] = "99"
    stub_env["XVFB_BOOTSTRAP_ATTEMPTS"] = override
    result = _run_wrapper(stub_env)
    assert result.returncode != 0
    assert _xvfb_calls(stub_env) == 3


@pytest.mark.parametrize("override", ["00", "000"])
def test_bootstrap_zero_padded_zero_ready_probes_falls_back_to_default(
    stub_env: dict[str, str], override: str
) -> None:
    """A zero-padded zero probe budget degrades to the script's default.

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    :param override: Zero-padded readiness-probe override.
    """
    stub_env["XVFB_READY_PROBES"] = override
    result = _run_wrapper(stub_env)
    assert result.returncode == 0, result.stderr
    assert "ran-ok DISPLAY=:99" in result.stdout
    assert _xvfb_calls(stub_env) == 1


def test_bootstrap_non_numeric_ready_probes_falls_back_to_default(
    stub_env: dict[str, str],
) -> None:
    """A malformed readiness-probe override degrades to the default.

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    """
    stub_env["XVFB_READY_PROBES"] = "not-a-number"
    result = _run_wrapper(stub_env)
    assert result.returncode == 0, result.stderr
    assert "ran-ok DISPLAY=:99" in result.stdout


@pytest.mark.parametrize("override", ["10", "99999999999999999999"])
def test_bootstrap_attempts_above_max_clamps_to_bound(
    stub_env: dict[str, str], override: str
) -> None:
    """An excessive retry budget remains bounded by the script's maximum.

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    :param override: Retry-budget override exceeding the supported maximum.
    """
    stub_env["XVFB_STUB_FAILS"] = "99"
    stub_env["XVFB_BOOTSTRAP_ATTEMPTS"] = override
    result = _run_wrapper(stub_env)
    assert result.returncode != 0
    assert _xvfb_calls(stub_env) == 9
    assert "clamping XVFB_BOOTSTRAP_ATTEMPTS to 9" in result.stderr


def test_bootstrap_ready_probes_above_max_reports_clamp(
    stub_env: dict[str, str],
) -> None:
    """An excessive readiness budget reports the enforced maximum.

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    """
    stub_env["XVFB_READY_PROBES"] = "101"
    result = _run_wrapper(stub_env)
    assert result.returncode == 0, result.stderr
    assert "clamping XVFB_READY_PROBES to 100" in result.stderr


def test_bootstrap_jitter_above_max_reports_clamp(
    stub_env: dict[str, str],
) -> None:
    """An excessive jitter override reports its sub-second maximum.

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    """
    stub_env["XVFB_RETRY_JITTER_MAX"] = "10"
    result = _run_wrapper(stub_env)
    assert result.returncode == 0, result.stderr
    assert "clamping XVFB_RETRY_JITTER_MAX to 9" in result.stderr


def test_bootstrap_zero_padded_jitter_override_recovers(
    stub_env: dict[str, str],
) -> None:
    """A zero-padded jitter override is read as decimal, not invalid octal.

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    """
    stub_env["XVFB_RETRY_JITTER_MAX"] = "09"
    stub_env["XVFB_STUB_FAILS"] = "1"
    result = _run_wrapper(stub_env)
    assert result.returncode == 0, result.stderr
    assert "ran-ok DISPLAY=:99" in result.stdout
    assert _xvfb_calls(stub_env) == 2


def test_bootstrap_retry_with_default_jitter_recovers(
    stub_env: dict[str, str],
) -> None:
    """The production jitter path (non-zero max) still retries to success.

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    """
    del stub_env["XVFB_RETRY_JITTER_MAX"]
    stub_env["XVFB_STUB_FAILS"] = "1"
    result = _run_wrapper(stub_env)
    assert result.returncode == 0, result.stderr
    assert "ran-ok DISPLAY=:99" in result.stdout
    assert _xvfb_calls(stub_env) == 2


@pytest.mark.requires_vst
def test_bootstrap_real_x_stack_accepts_client_connection() -> None:
    """The wrapper's real Xvfb display accepts an X client connection."""
    result = subprocess.run(  # noqa: S603 — argv is test-owned
        [VST_HEADLESS_WRAPPER, "xdpyinfo"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "name of display:" in result.stdout


def test_bootstrap_xdpyinfo_never_confirms_trusts_displayfd_and_runs_command(
    stub_env: dict[str, str],
) -> None:
    """An unusable xdpyinfo probe does not fail a server -displayfd declared ready.

    Reproduces the bare-runner failure: Xvfb starts and writes its display via
    ``-displayfd`` (server listening), but ``xdpyinfo`` never confirms — because
    ``x11-utils`` is absent or the probe is refused under contention. The wrapper
    trusts the ``-displayfd`` readiness signal and runs the command on the first
    attempt rather than exhausting the retry budget and failing (#2320).

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    """
    # A threshold no bootstrap attempt can reach keeps xdpyinfo failing forever.
    stub_env["XDPYINFO_STUB_MIN_XVFB_CALLS"] = "99"
    stub_env["XVFB_READY_PROBES"] = "3"
    result = _run_wrapper(stub_env)
    assert result.returncode == 0, result.stderr
    assert "ran-ok DISPLAY=:99" in result.stdout
    assert _xvfb_calls(stub_env) == 1
    _assert_stub_xvfb_pids_dead(stub_env)


def test_bootstrap_xsettingsd_dies_once_retries_and_runs_command(
    stub_env: dict[str, str],
) -> None:
    """A manager that loses the display once is restarted, and the command runs.

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    """
    stub_env["XSETTINGSD_STUB_FAILS"] = "1"
    result = _run_wrapper(stub_env)
    assert result.returncode == 0, result.stderr
    assert "ran-ok DISPLAY=:99" in result.stdout
    assert _xsettingsd_calls(stub_env) == 2


def test_bootstrap_xsettingsd_dies_every_attempt_fails_with_its_log(
    stub_env: dict[str, str],
) -> None:
    """No XSETTINGS manager means no render: fail loudly instead of into a fatal X error.

    Without an owner of ``_XSETTINGS_S0`` the plugin queries window 0 and X
    terminates it with ``BadWindow`` (#3152), so the wrapper must not run the
    command — and must show why, since the daemon's log is otherwise discarded.

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    """
    stub_env["XSETTINGSD_STUB_FAILS"] = "99"
    result = _run_wrapper(stub_env)
    assert result.returncode != 0
    assert "ran-ok" not in result.stdout
    assert _xsettingsd_calls(stub_env) == 3
    assert "Unable to open connection to X server" in result.stderr


def test_bootstrap_xsettingsd_never_takes_selection_fails_without_running_command(
    stub_env: dict[str, str],
) -> None:
    """A live manager that never takes the selection is still no manager.

    Liveness alone is not the invariant: the plugin reads the selection, so a
    daemon that stays up without owning it leaves the same BadWindow exposure.

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    """
    stub_env["XSETTINGSD_STUB_SILENT"] = "1"
    stub_env["XSETTINGS_READY_PROBES"] = "2"
    result = _run_wrapper(stub_env)
    assert result.returncode != 0
    assert "ran-ok" not in result.stdout
    assert _xsettingsd_calls(stub_env) == 3


@pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("Xvfb", "xsettingsd", "dbus-run-session")),
    reason="needs the real X stack the wrapper bootstraps",
)
def test_bootstrap_real_x_stack_hands_the_command_an_xsettings_owner(tmp_path: Path) -> None:
    """A second real xsettingsd finds the wrapper's manager already owning the selection.

    The only client that can report a selection owner is an XSettings manager itself, so the
    command is one. It never exits on its own and buffers its output when piped, so the probe logs
    to a file and stops the daemon itself.

    :param tmp_path: Holds the probe daemon's log.
    """
    probe_log = tmp_path / "probe.log"
    probe = f'''
xsettingsd --config /dev/null > "{probe_log}" 2>&1 &
daemon=$!
for _ in $(seq 50); do
  grep -q "Took ownership of selection" "{probe_log}" && break
  sleep 0.1
done
kill "$daemon" 2>/dev/null || true
wait "$daemon" 2>/dev/null || true
'''
    subprocess.run(  # noqa: S603 — argv is test-owned
        [VST_HEADLESS_WRAPPER, "bash", "-c", probe],
        capture_output=True,
        text=True,
        timeout=60,
    )

    owner_lines = [
        line
        for line in probe_log.read_text().splitlines()
        if line.startswith("xsettingsd: Selection _XSETTINGS_S0 is owned by ")
    ]
    assert owner_lines, probe_log.read_text()
    assert not owner_lines[0].endswith(" 0x0")


def test_bootstrap_without_xsettingsd_installed_still_runs_command(
    stub_env: dict[str, str], tmp_path: Path
) -> None:
    """A host with no xsettingsd runs the command instead of failing the bootstrap.

    GitHub's Ubuntu runners have no ``xsettingsd``, and the wrapper is used
    there to compose Hydra configs with no plugin in sight — so an absent
    binary must stay advisory (#3152).

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    :param tmp_path: Per-test dir for the xsettingsd-free system PATH.
    """
    stub_env["PATH"] = _path_without_xsettingsd(stub_env, tmp_path / "sysbin")
    (Path(stub_env["PATH"].split(os.pathsep)[0]) / "xsettingsd").unlink()

    result = _run_wrapper(stub_env)

    assert result.returncode == 0, result.stderr
    assert "ran-ok DISPLAY=:99" in result.stdout
    assert "xsettingsd is not installed" in result.stderr


def test_bootstrap_with_installed_xsettingsd_that_never_owns_fails(
    stub_env: dict[str, str],
) -> None:
    """An installed daemon that never takes the selection still fails loudly.

    :param stub_env: Wrapper environment with stub X binaries on PATH.
    """
    stub_env["XSETTINGSD_STUB_FAILS"] = "99"

    result = _run_wrapper(stub_env)

    assert result.returncode != 0
    assert "ran-ok" not in result.stdout
    assert _xsettingsd_calls(stub_env) == 3
