"""The SkyPilot-local lane survives a transient apt index failure.

Two required generation-smoke jobs red-lined an unrelated PR when a third-party package index
served a `Hash Sum mismatch` during `apt-get update` (#3331). The failure is remote, transient, and
happens before any repository code runs, so the setup step must neither translate it into a review-
blocking red nor install blind when it does not clear.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _PROJECT_ROOT / ".github/workflows/test-skypilot-local.yml"
_STEP_NAME = "Install socat (kubectl port-forward dependency)"
_SOURCEPARTS_OPT = "Dir::Etc::sourceparts=-"


def _socat_step_script() -> str:
    """Return the checked-in shell of the socat install step.

    :returns: The step's ``run`` script.
    """
    workflow = yaml.safe_load(_WORKFLOW.read_text())
    steps = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == _STEP_NAME
    ]
    assert len(steps) == 1, f"expected exactly one {_STEP_NAME!r} step, got {len(steps)}"
    return steps[0]["run"]


@dataclass(frozen=True)
class _StepRun:
    """One run of the step's shell, with every stubbed command's argv.

    .. attribute :: process

        The completed bash process.

    .. attribute :: calls

        Argument lines of each stubbed command, keyed by command name, in call order.
    """

    process: subprocess.CompletedProcess[str]
    calls: dict[str, list[str]]


def _write_logging_stub(path: Path, body: str = "") -> None:
    """Install an executable at `path` that appends its argv to a per-name log.

    :param path: Where to write the stub; its name selects the log file.
    :param body: Extra shell appended after the logging line.
    """
    path.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$*" >> "$STUB_LOG_DIR/{path.name}.log"\n{body}exit 0\n'
    )
    path.chmod(0o755)


def _run_step(tmp_path: Path, failing_updates: int) -> _StepRun:
    """Run the step's own shell against stub `apt-get`, `sleep` and `rm`.

    `sleep` and `rm` are stubbed rather than real so the retry path neither spends wall-clock
    time nor deletes the host's apt lists; every command is logged so the test asserts on what
    the step did instead of on how long it took.

    :param tmp_path: Directory holding the stub executables and their logs.
    :param failing_updates: Number of leading ``update`` calls that fail the way a remote index
        hash mismatch does.
    :returns: What the step did.
    """
    logs = tmp_path / "logs"
    logs.mkdir()
    counter = tmp_path / "update-count"
    counter.write_text("0")

    apt = tmp_path / "apt-get"
    apt.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$STUB_LOG_DIR/apt-get.log"\n'
        'case "$*" in\n'
        "  *update*)\n"
        '    seen=$(($(cat "$APT_UPDATE_COUNT") + 1))\n'
        '    printf "%s" "$seen" > "$APT_UPDATE_COUNT"\n'
        '    if [ "$seen" -le "$APT_FAILING_UPDATES" ]; then\n'
        '      echo "E: Failed to fetch https://example.invalid/dists/stable/main/'
        'binary-amd64/Packages.gz  Hash Sum mismatch" >&2\n'
        "      exit 100\n"
        "    fi\n"
        "    ;;\n"
        "esac\n"
        "exit 0\n"
    )
    apt.chmod(0o755)
    _write_logging_stub(tmp_path / "sleep")
    _write_logging_stub(tmp_path / "rm")
    _write_logging_stub(tmp_path / "sudo", body='exec "$@"\n')

    completed = subprocess.run(  # noqa: S603 — bash executes the checked-in workflow step
        ["/bin/bash", "-c", _socat_step_script()],
        capture_output=True,
        env={
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "STUB_LOG_DIR": str(logs),
            "APT_UPDATE_COUNT": str(counter),
            "APT_FAILING_UPDATES": str(failing_updates),
        },
        text=True,
        timeout=60,
        check=False,
    )

    return _StepRun(
        process=completed,
        calls={log.stem: log.read_text().splitlines() for log in logs.glob("*.log")},
    )


def test_socat_install_without_index_failure_installs_once(tmp_path: Path) -> None:
    """A clean index refresh installs socat without retrying.

    :param tmp_path: Directory holding the stub executables and their logs.
    """
    run = _run_step(tmp_path, failing_updates=0)

    assert run.process.returncode == 0, run.process.stdout + run.process.stderr
    assert [call for call in run.calls["apt-get"] if "install" in call]
    assert "sleep" not in run.calls


def test_socat_install_with_transient_index_failure_recovers(tmp_path: Path) -> None:
    """One hash-sum mismatch is retried, and socat still installs.

    :param tmp_path: Directory holding the stub executables and their logs.
    """
    run = _run_step(tmp_path, failing_updates=1)

    assert run.process.returncode == 0, run.process.stdout + run.process.stderr
    assert [call for call in run.calls["apt-get"] if "install" in call]


def test_socat_install_with_persistent_index_failure_fails(tmp_path: Path) -> None:
    """A mismatch that never clears fails the step rather than installing blind.

    :param tmp_path: Directory holding the stub executables and their logs.
    """
    run = _run_step(tmp_path, failing_updates=99)

    assert run.process.returncode != 0
    assert not [call for call in run.calls["apt-get"] if "install" in call]


def test_socat_install_retry_backoff_is_bounded_and_increasing(tmp_path: Path) -> None:
    """The retry gives up after three refreshes, backing off longer each time.

    :param tmp_path: Directory holding the stub executables and their logs.
    """
    run = _run_step(tmp_path, failing_updates=99)

    assert len([call for call in run.calls["apt-get"] if "update" in call]) == 3
    assert run.calls["sleep"] == ["5", "10"]


def test_socat_install_retry_discards_the_stale_index(tmp_path: Path) -> None:
    """Each retry drops the partially-fetched lists instead of re-reading them.

    A mismatch leaves a corrupt list behind, so refreshing over it reproduces the same failure.

    :param tmp_path: Directory holding the stub executables and their logs.
    """
    run = _run_step(tmp_path, failing_updates=1)

    # The step's own shell expands the glob, so the logged argv varies with the host's lists.
    assert len(run.calls["rm"]) == 1
    assert run.calls["rm"][0].startswith("-rf /var/lib/apt/lists/")


def test_socat_index_refresh_skips_third_party_lists(tmp_path: Path) -> None:
    """The refresh reads only the distro archive, where socat lives.

    The mismatch in #3331 came from a third-party list this lane never installs from, so
    excluding ``sources.list.d`` removes the failure mode rather than waiting it out.

    :param tmp_path: Directory holding the stub executables and their logs.
    """
    run = _run_step(tmp_path, failing_updates=0)

    update_calls = [call for call in run.calls["apt-get"] if "update" in call]
    assert update_calls, "step never refreshed the package index"
    assert all(_SOURCEPARTS_OPT in call for call in update_calls)
