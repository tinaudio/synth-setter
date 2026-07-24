"""Real-process behavior tests for detached Pi review aftercare."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest
import sh

from agent._shared.run_pi_review_aftercare import (
    _MAX_LOG_BYTES,
    AftercareResult,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "agent/_shared/run_pi_review_aftercare.py"


def _manifest(tmp_path: Path, *, output_path: Path | None = None) -> Path:
    review_dir = tmp_path / ".agent-reviews"
    review_dir.mkdir()
    deferred_pass: dict[str, object] = {
        "skill": "correctness-review",
        "pass_name": "free-pool",
        "origin": "primary",
        "model": "kimi-coding/k3",
        "verification_model": "openai-codex/gpt-5.6-sol",
        "thinking": "high",
    }
    if output_path is not None:
        deferred_pass.update(agent_id="agent-foreground", output_path=str(output_path))
    manifest = review_dir / "aftercare.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "mode": "no-comments",
                "repo": "tinaudio/synth-setter",
                "pr_number": 2174,
                "base_sha": "a" * 40,
                "head_sha": "b" * 40,
                "target": "PR #2174",
                "deferred_passes": [deferred_pass],
                "foreground_fingerprints": [],
            }
        )
    )
    return manifest


def _result_path(manifest: Path) -> Path:
    return Path(f"{manifest}.result.json")


def _log_path(manifest: Path) -> Path:
    return Path(f"{manifest}.aftercare.log")


def _valid_result(*, status: str = "complete") -> str:
    return json.dumps(
        {
            "status": status,
            "attempts": [
                {
                    "skill": "correctness-review",
                    "pass_name": "free-pool",
                    "model": "kimi-coding/k3",
                    "status": "success",
                    "agent_id": "agent-aftercare",
                    "output_path": ".pi/output/agent-aftercare.jsonl",
                    "detail": "validated report",
                }
            ],
            "diagnostics": [],
            "late_findings": [],
            "posted_review_url": None,
            "child_exit_code": None,
            "log_tail": "",
            "completed_at": "2026-07-24T00:00:00Z",
        }
    )


def _fake_pi(tmp_path: Path) -> Path:
    pi = tmp_path / "pi"
    pi.write_text(
        "#!/usr/bin/env python3\n"
        "import os, signal, sys, time\n"
        "from pathlib import Path\n"
        "runtime = Path(os.environ['PI_REVIEW_AFTERCARE_RUNTIME_MANIFEST'])\n"
        "result = Path(f'{runtime}.result.json')\n"
        "mode = os.environ['FAKE_PI_MODE']\n"
        "expected = os.environ.get('FAKE_PI_EXPECT_ADOPTED')\n"
        "if expected and expected not in ' '.join(sys.argv):\n"
        "    raise SystemExit(9)\n"
        "print(os.environ.get('FAKE_PI_LOG', 'child-log'), flush=True)\n"
        "if mode == 'valid':\n"
        "    result.write_text(os.environ['FAKE_PI_RESULT'])\n"
        "elif mode == 'invalid':\n"
        "    result.write_text('{not-json')\n"
        "elif mode == 'nonzero':\n"
        "    raise SystemExit(7)\n"
        "elif mode == 'wait-for-kill':\n"
        "    Path(os.environ['FAKE_PI_PID_FILE']).write_text(str(os.getpid()))\n"
        "    time.sleep(30)\n"
        "elif mode == 'atomic':\n"
        "    payload = os.environ['FAKE_PI_RESULT']\n"
        "    with result.open('w') as file:\n"
        "        file.write(payload[:len(payload) // 2])\n"
        "        file.flush()\n"
        "        Path(os.environ['FAKE_PI_PARTIAL_MARKER']).touch()\n"
        "        while not Path(os.environ['FAKE_PI_RELEASE']).exists():\n"
        "            time.sleep(0.01)\n"
        "        file.write(payload[len(payload) // 2:])\n"
    )
    pi.chmod(0o755)
    return pi


def _environment(tmp_path: Path, *, mode: str, foreground_stopped: bool = True) -> dict[str, str]:
    environment = {
        **os.environ,
        "FAKE_PI_MODE": mode,
        "FAKE_PI_RESULT": _valid_result(),
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
    }
    if foreground_stopped:
        environment["SYNTH_SETTER_PI_REVIEW_FOREGROUND_STOPPED"] = "1"
    else:
        environment.pop("SYNTH_SETTER_PI_REVIEW_FOREGROUND_STOPPED", None)
    return environment


def _run_supervisor(manifest: Path, environment: dict[str, str]) -> int:
    try:
        sh.Command(sys.executable)(
            SCRIPT,
            "--supervise",
            manifest,
            _cwd=manifest.parents[1],
            _env=environment,
            _timeout=5,
        )
    except sh.ErrorReturnCode as error:
        return error.exit_code
    return 0


def _read_result(manifest: Path) -> AftercareResult:
    return AftercareResult.model_validate_json(_result_path(manifest).read_text())


def _wait_for_path(path: Path, *, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {path}")
        time.sleep(0.01)


def _wait_for_text(path: Path, expected: str, *, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists() or expected not in path.read_text():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {expected!r} in {path}")
        time.sleep(0.01)


def test_supervisor_valid_child_result_publishes_strict_atomic_result(tmp_path: Path) -> None:
    """Publish a valid model result through the supervisor-owned result path.

    :param tmp_path: Temporary review root and fake Pi executable.
    """
    _fake_pi(tmp_path)
    manifest = _manifest(tmp_path)

    completed = _run_supervisor(manifest, _environment(tmp_path, mode="valid"))

    assert completed == 0
    result = _read_result(manifest)
    assert result.status == "complete"
    assert result.child_exit_code == 0
    assert "child-log" in _log_path(manifest).read_text()


@pytest.mark.parametrize(
    ("mode", "expected_category", "expected_exit_code"),
    [
        ("missing", "missing-result", 0),
        ("invalid", "invalid-result", 0),
        ("nonzero", "child-exit", 7),
    ],
)
def test_supervisor_bad_child_outcome_writes_failed_result(
    tmp_path: Path,
    mode: str,
    expected_category: str,
    expected_exit_code: int,
) -> None:
    """Convert missing, malformed, and nonzero child outcomes into strict failures.

    :param tmp_path: Temporary review root and fake Pi executable.
    :param mode: Fake child behavior.
    :param expected_category: Required supervisor diagnostic category.
    :param expected_exit_code: Required child exit code.
    """
    _fake_pi(tmp_path)
    manifest = _manifest(tmp_path)

    completed = _run_supervisor(manifest, _environment(tmp_path, mode=mode))

    assert completed == 1
    result = _read_result(manifest)
    assert result.status == "failed"
    assert result.child_exit_code == expected_exit_code
    assert expected_category in {diagnostic.category for diagnostic in result.diagnostics}
    assert any(attempt.status == "failed" for attempt in result.attempts)


def test_supervisor_missing_child_result_rejects_stale_runtime_sidecar(tmp_path: Path) -> None:
    """Never accept a runtime result left by an earlier supervisor attempt.

    :param tmp_path: Temporary review root and fake Pi executable.
    """
    _fake_pi(tmp_path)
    manifest = _manifest(tmp_path)
    runtime_result = Path(f"{manifest}.supervised.json.result.json")
    runtime_result.write_text(_valid_result())

    completed = _run_supervisor(manifest, _environment(tmp_path, mode="missing"))

    assert completed == 1
    result = _read_result(manifest)
    assert result.status == "failed"
    assert "missing-result" in {diagnostic.category for diagnostic in result.diagnostics}


def test_supervisor_missing_pi_executable_writes_supervisor_failure(tmp_path: Path) -> None:
    """Publish a strict result when child startup raises before an exit code exists.

    :param tmp_path: Temporary review root without a Pi executable.
    """
    manifest = _manifest(tmp_path)
    environment = _environment(tmp_path, mode="missing")
    environment["PATH"] = str(tmp_path)

    completed = _run_supervisor(manifest, environment)

    assert completed == 1
    result = _read_result(manifest)
    assert result.status == "failed"
    assert result.child_exit_code is None
    assert "supervisor-error" in {diagnostic.category for diagnostic in result.diagnostics}


def test_supervisor_killed_child_writes_failed_result_with_signal_code(tmp_path: Path) -> None:
    """Retain diagnostics when the supervised Pi child is killed.

    :param tmp_path: Temporary review root and fake Pi executable.
    """
    _fake_pi(tmp_path)
    manifest = _manifest(tmp_path)
    child_pid_file = tmp_path / "child.pid"
    environment = _environment(tmp_path, mode="wait-for-kill")
    environment["FAKE_PI_PID_FILE"] = str(child_pid_file)
    supervisor = cast(
        Any,
        sh.Command(sys.executable)(
            SCRIPT,
            "--supervise",
            manifest,
            _bg=True,
            _cwd=tmp_path,
            _env=environment,
        ),
    )
    try:
        _wait_for_path(child_pid_file)
        _wait_for_text(_log_path(manifest), "child-log")
        os.kill(int(child_pid_file.read_text()), signal.SIGKILL)
        with pytest.raises(sh.ErrorReturnCode) as error:
            supervisor.wait(timeout=5)
        assert error.value.exit_code == 1
    finally:
        if child_pid_file.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(child_pid_file.read_text()), signal.SIGKILL)
        with contextlib.suppress(Exception):
            supervisor.kill()

    result = _read_result(manifest)
    assert result.status == "failed"
    assert result.child_exit_code == -signal.SIGKILL
    assert "child-exit" in {diagnostic.category for diagnostic in result.diagnostics}


def test_supervisor_capacity_marker_records_distinct_bounded_diagnostic(tmp_path: Path) -> None:
    """Classify capacity failures and bound the persisted child log.

    :param tmp_path: Temporary review root and fake Pi executable.
    """
    _fake_pi(tmp_path)
    manifest = _manifest(tmp_path)
    environment = _environment(tmp_path, mode="nonzero")
    environment["FAKE_PI_LOG"] = "x" * (_MAX_LOG_BYTES + 1000) + " ResourceExhausted worker limit"

    _run_supervisor(manifest, environment)

    result = _read_result(manifest)
    assert "capacity" in {diagnostic.category for diagnostic in result.diagnostics}
    assert "ResourceExhausted worker limit" in result.log_tail
    assert _log_path(manifest).stat().st_size <= _MAX_LOG_BYTES


def test_supervisor_result_path_never_exposes_partial_child_output(tmp_path: Path) -> None:
    """Keep the canonical result absent until a complete payload can replace it.

    :param tmp_path: Temporary review root and fake Pi executable.
    """
    _fake_pi(tmp_path)
    manifest = _manifest(tmp_path)
    partial_marker = tmp_path / "partial"
    release = tmp_path / "release"
    environment = _environment(tmp_path, mode="atomic")
    environment["FAKE_PI_PARTIAL_MARKER"] = str(partial_marker)
    environment["FAKE_PI_RELEASE"] = str(release)
    supervisor = cast(
        Any,
        sh.Command(sys.executable)(
            SCRIPT,
            "--supervise",
            manifest,
            _bg=True,
            _cwd=tmp_path,
            _env=environment,
        ),
    )
    try:
        _wait_for_path(partial_marker)
        assert not _result_path(manifest).exists()
        release.touch()
        supervisor.wait(timeout=5)
    finally:
        with contextlib.suppress(Exception):
            supervisor.kill()

    assert _read_result(manifest).status == "complete"


def test_supervisor_adopts_valid_foreground_output_without_duplicate_launch(
    tmp_path: Path,
) -> None:
    """Adopt a completed foreground report instead of launching another owner.

    :param tmp_path: Temporary review root and fake Pi executable.
    """
    launch_marker = tmp_path / "pi-launched"
    pi = _fake_pi(tmp_path)
    pi.write_text(
        pi.read_text().replace("runtime =", f"Path({str(launch_marker)!r}).touch()\nruntime =")
    )
    report = {
        "skill": "correctness-review",
        "target": "PR #2174",
        "findings": [],
        "what_looks_good": ["Ownership remained singular."],
    }
    foreground_output = tmp_path / "foreground.jsonl"
    foreground_output.write_text(
        json.dumps({"message": {"role": "assistant", "content": json.dumps(report)}}) + "\n"
    )
    manifest = _manifest(tmp_path, output_path=foreground_output)

    completed = _run_supervisor(
        manifest,
        _environment(tmp_path, mode="valid", foreground_stopped=False),
    )

    assert completed == 0
    assert not launch_marker.exists()
    result = _read_result(manifest)
    assert result.status == "complete"
    assert [attempt.status for attempt in result.attempts] == ["adopted-foreground-result"]


def test_supervisor_adopted_findings_reach_child_with_no_relaunch_instruction(
    tmp_path: Path,
) -> None:
    """Pass adopted findings to aftercare with an explicit no-relaunch contract.

    :param tmp_path: Temporary review root and fake Pi executable.
    """
    _fake_pi(tmp_path)
    report = {
        "skill": "correctness-review",
        "target": "PR #2174",
        "findings": [
            {
                "severity": "warn",
                "path": "agent/example.py",
                "line": 42,
                "description": "Late finding.",
            }
        ],
        "what_looks_good": ["The report validates."],
    }
    foreground_output = tmp_path / "foreground-with-finding.jsonl"
    foreground_output.write_text(
        json.dumps({"message": {"role": "assistant", "content": json.dumps(report)}}) + "\n"
    )
    manifest = _manifest(tmp_path, output_path=foreground_output)
    environment = _environment(tmp_path, mode="valid", foreground_stopped=False)
    environment["FAKE_PI_EXPECT_ADOPTED"] = (
        "Use each row's output_path and do not launch those passes again."
    )

    completed = _run_supervisor(manifest, environment)

    assert completed == 0
    statuses = [attempt.status for attempt in _read_result(manifest).attempts]
    assert statuses == ["adopted-foreground-result", "success"]


def test_supervisor_mixed_ownership_reports_one_terminal_status_per_pass(tmp_path: Path) -> None:
    """Keep adopted passes successful when another foreground owner blocks aftercare.

    :param tmp_path: Temporary review root.
    """
    report = {
        "skill": "correctness-review",
        "target": "PR #2174",
        "findings": [],
        "what_looks_good": ["The foreground report validates."],
    }
    foreground_output = tmp_path / "completed.jsonl"
    foreground_output.write_text(
        json.dumps({"message": {"role": "assistant", "content": json.dumps(report)}}) + "\n"
    )
    manifest = _manifest(tmp_path, output_path=foreground_output)
    payload = json.loads(manifest.read_text())
    payload["deferred_passes"].append(
        {
            "skill": "code-health",
            "pass_name": "free-pool",
            "origin": "primary",
            "model": "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
            "verification_model": "openai-codex/gpt-5.6-terra",
            "thinking": "medium",
            "agent_id": "agent-still-running",
            "output_path": str(tmp_path / "unfinished.jsonl"),
        }
    )
    manifest.write_text(json.dumps(payload))

    completed = _run_supervisor(manifest, _environment(tmp_path, mode="missing"))

    assert completed == 1
    attempts = [(row.skill, row.status) for row in _read_result(manifest).attempts]
    assert attempts == [
        ("correctness-review", "adopted-foreground-result"),
        ("code-health", "failed"),
    ]


def test_supervisor_unstoppable_foreground_owner_fails_closed(tmp_path: Path) -> None:
    """Refuse a duplicate launch when foreground termination is not guaranteed.

    :param tmp_path: Temporary review root and fake Pi executable.
    """
    launch_marker = tmp_path / "pi-launched"
    pi = _fake_pi(tmp_path)
    pi.write_text(
        pi.read_text().replace("runtime =", f"Path({str(launch_marker)!r}).touch()\nruntime =")
    )
    manifest = _manifest(tmp_path, output_path=tmp_path / "unfinished.jsonl")

    completed = _run_supervisor(
        manifest,
        _environment(tmp_path, mode="valid", foreground_stopped=False),
    )

    assert completed == 1
    assert not launch_marker.exists()
    result = _read_result(manifest)
    assert result.status == "failed"
    assert "ownership" in {diagnostic.category for diagnostic in result.diagnostics}


def test_supervisor_host_exit_does_not_authorize_duplicate_launch(tmp_path: Path) -> None:
    """Fail closed because foreground host exit does not stop its workers.

    :param tmp_path: Temporary review root and fake Pi executable.
    """
    launch_marker = tmp_path / "pi-launched"
    pi = _fake_pi(tmp_path)
    pi.write_text(
        pi.read_text().replace("runtime =", f"Path({str(launch_marker)!r}).touch()\nruntime =")
    )
    manifest = _manifest(tmp_path, output_path=tmp_path / "unfinished.jsonl")

    completed = _run_supervisor(manifest, _environment(tmp_path, mode="valid"))

    assert completed == 1
    assert not launch_marker.exists()
    result = _read_result(manifest)
    assert [attempt.status for attempt in result.attempts] == ["failed"]
    assert "ownership" in {diagnostic.category for diagnostic in result.diagnostics}


def test_aftercare_result_failed_status_requires_diagnostic() -> None:
    """Reject a failed model result without actionable evidence."""
    payload = json.loads(_valid_result(status="failed"))

    with pytest.raises(ValueError, match="failed result requires a diagnostic"):
        AftercareResult.model_validate_json(json.dumps(payload))


def test_aftercare_result_rejects_model_written_extra_fields() -> None:
    """Apply the same strict result model to model-written JSON."""
    payload = json.loads(_valid_result())
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        AftercareResult.model_validate_json(json.dumps(payload))
