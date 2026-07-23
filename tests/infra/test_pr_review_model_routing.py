"""Contract tests for PR-review model routing across supported agent harnesses."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import re
import runpy
import shutil
import signal
import sys
import time
import tomllib
from pathlib import Path
from unittest import mock

import pytest
import yaml

from agent._shared.run_pi_review_aftercare import AftercareManifest
from tests.helpers.package_available import _SH_AVAILABLE

REPO_ROOT = Path(__file__).resolve().parents[2]


def _process_state(pid: int) -> str | None:
    """Return Linux scheduler state; elsewhere distinguish only PID presence.

    :param pid: Must identify a process created by the current test.
    :returns: Scheduler state, ``?`` for non-Linux presence, or ``None`` when absent.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return "?"

    if not Path("/proc").is_dir():
        return "?"
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    return stat.rpartition(")")[2].split()[0]


def _assert_process_terminated(pid: int, *, timeout: float = 1) -> None:
    """Wait until a process is absent or zombie, then fail if it remains live.

    :param pid: Process ID expected to stop executing.
    :param timeout: Maximum seconds to wait for termination.
    :raises AssertionError: If the process remains live through the timeout.
    """
    deadline = time.monotonic() + timeout
    while (state := _process_state(pid)) not in (None, "Z"):
        if time.monotonic() >= deadline:
            raise AssertionError(f"process {pid} is still running (state {state})")
        time.sleep(0.05)


def test_process_state_permission_denied_reports_pid_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat an inaccessible PID as present rather than terminated.

    :param monkeypatch: Replaces the process probe with its permission-denied result.
    """
    monkeypatch.setattr(os, "kill", mock.Mock(side_effect=PermissionError))

    assert _process_state(123) == "?"


def test_assert_process_terminated_live_pid_fails() -> None:
    """Reject a descendant that is still executing."""
    child_pid = os.fork()
    if child_pid == 0:
        time.sleep(30)
        os._exit(0)
    try:
        with pytest.raises(AssertionError, match="still running"):
            _assert_process_terminated(child_pid, timeout=0)
    finally:
        os.kill(child_pid, signal.SIGKILL)
        os.waitpid(child_pid, 0)


def test_assert_process_terminated_nonexistent_pid_passes() -> None:
    """Accept a PID after its process has been reaped."""
    child_pid = os.fork()
    if child_pid == 0:
        os._exit(0)
    os.waitpid(child_pid, 0)

    _assert_process_terminated(child_pid, timeout=0)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="requires Linux process states")
def test_assert_process_terminated_zombie_pid_passes() -> None:
    """Accept a terminated child before its parent reaps it."""
    child_pid = os.fork()
    if child_pid == 0:
        os._exit(0)
    try:
        _assert_process_terminated(child_pid, timeout=1)
    finally:
        os.waitpid(child_pid, 0)


_ROLE_MODELS = {
    "pr-review-orchestrator": {
        "claude": ("haiku", "medium"),
        "codex": ("gpt-5.6-terra", "medium"),
    },
    "pr-review-worker-deep": {
        "claude": ("sonnet", "high"),
        "codex": ("gpt-5.6-sol", "high"),
        "opencode": ("opencode-go/kimi-k2.7-code", "high"),
    },
    "pr-review-worker-fast": {
        "claude": ("sonnet", "medium"),
        "codex": ("gpt-5.6-terra", "medium"),
        "opencode": ("opencode-go/glm-5.2", None),
    },
}


def test_review_roles_pin_provider_models_and_effort() -> None:
    """Guard cross-provider routing against silent model or effort drift."""
    for role, providers in _ROLE_MODELS.items():
        text = (REPO_ROOT / ".claude" / "agents" / f"{role}.md").read_text()
        _, frontmatter, _ = text.split("---", 2)
        claude = yaml.safe_load(frontmatter)
        assert isinstance(claude, dict)
        assert claude["name"] == role
        assert claude["description"]
        assert (claude["model"], claude["effort"]) == providers["claude"]

        with (REPO_ROOT / ".codex" / "agents" / f"{role}.toml").open("rb") as file:
            codex = tomllib.load(file)
        assert codex["name"] == role
        assert codex["description"]
        assert codex["developer_instructions"]
        assert (codex["model"], codex["model_reasoning_effort"]) == providers["codex"]
        if role == "pr-review-orchestrator":
            assert "pi-review-host-contract.md" in codex["developer_instructions"]
        else:
            assert (
                "always return the requested structured report" in codex["developer_instructions"]
            )

        if "opencode" in providers:
            with (REPO_ROOT / ".opencode" / "agents" / f"{role}.toml").open("rb") as file:
                opencode = tomllib.load(file)
            assert opencode["name"] == role
            assert opencode["description"]
            assert (opencode["model"], opencode.get("variant")) == providers["opencode"]
            assert (
                "always return the requested structured report"
                in opencode["developer_instructions"]
            )


def test_codex_review_roles_are_registered_with_nested_fanout() -> None:
    """Ensure Codex can spawn the configured orchestrator-to-worker hierarchy."""
    with (REPO_ROOT / ".codex" / "config.toml").open("rb") as file:
        config = tomllib.load(file)

    assert config["agents"]["max_depth"] == 2
    for role in _ROLE_MODELS:
        registered = config["agents"][role]
        assert registered["config_file"] == f"agents/{role}.toml"


def test_full_review_skills_route_external_harnesses_through_pi() -> None:
    """Keep Claude and Codex on the same Pi-native review implementation."""
    contract_path = "agent/_shared/pi-review-host-contract.md"
    for skill in ("repo-review-full", "repo-review-full-no-comments"):
        text = (REPO_ROOT / "agent" / "skills" / skill / "SKILL.md").read_text()
        assert "SYNTH_SETTER_PI_REVIEW" in text
        assert contract_path in text
        assert "run_codex_review_agent.sh" not in text

    contract = " ".join((REPO_ROOT / contract_path).read_text().split())
    assert contract.count("agent/_shared/run_pi_review.sh") == 1
    assert "run_in_background" in contract
    assert "TaskOutput" in contract
    assert "600000" in contract
    assert "until Pi exits" in contract


def test_native_review_orchestrators_delegate_to_pi() -> None:
    """Require both host entrypoints to launch exactly one blocking Pi process.

    Authenticated host completion is an L1 `/pr-checkbox` check because CI cannot invoke the
    installed Claude and Codex accounts.
    """
    orchestrators = (
        REPO_ROOT / ".claude" / "agents" / "pr-review-orchestrator.md",
        REPO_ROOT / ".codex" / "agents" / "pr-review-orchestrator.toml",
    )

    contract_path = "agent/_shared/pi-review-host-contract.md"
    for orchestrator in orchestrators:
        text = " ".join(orchestrator.read_text().split())
        assert contract_path in text
        assert "native review worker" not in text

    contract = " ".join((REPO_ROOT / contract_path).read_text().split())
    assert 'append `--target "$n"`' in contract.lower()
    assert "[1-9][0-9]*" in contract
    assert '--target "$N"' in contract
    assert "Run exactly one command" in contract
    assert "foreground blocking call" in contract
    assert "yields a shell session" in contract


def test_review_fanout_promotes_deep_checklists() -> None:
    """Keep high thinking pinned for correctness-sensitive checklists."""
    routing = (REPO_ROOT / "agent" / "_shared" / "pi_review_routing.py").read_text()

    assert 'REPO_LOCAL_SKILLS = frozenset({"correctness-review", "lance-review"})' in routing
    assert "HIGH_THINKING_SKILLS = REPO_LOCAL_SKILLS" in routing
    assert 'return "high", "deep checklist"' in routing


def test_pi_review_worker_allows_dynamic_model_routing() -> None:
    """Ensure policy, rather than the agent definition, selects Pi worker models."""
    text = (REPO_ROOT / ".pi" / "agents" / "pr-review-worker.md").read_text()
    _, frontmatter, prompt = text.split("---", 2)
    worker = yaml.safe_load(frontmatter)

    assert worker["description"]
    assert worker["prompt_mode"] == "append"
    assert isinstance(worker["tools"], str), "Pi worker tools must use comma-separated syntax"
    assert set(worker["tools"].split(", ")) == {"bash", "grep", "read"}
    assert "max_turns" not in worker
    assert "model" not in worker
    assert "thinking" not in worker
    prompt_flat = " ".join(prompt.split())
    assert "exactly one JSON object" in prompt_flat
    assert '"severity": "block or warn"' in prompt
    assert '"line": 42' in prompt
    assert '"what_looks_good"' in prompt
    assert "no Markdown fence or surrounding prose" in prompt_flat
    assert "Never run `find`" in prompt
    assert "60-second timeout" in prompt
    assert "changed paths" in prompt


def test_pi_project_settings_pin_review_pool_providers_only() -> None:
    """Keep project-local Pi defaults scoped to the review pool, away from Anthropic."""
    settings = json.loads((REPO_ROOT / ".pi" / "settings.json").read_text())

    assert settings["defaultProvider"] == "openai-codex"
    assert settings["defaultModel"] == "gpt-5.6-sol"
    assert settings["enabledModels"]
    assert all("anthropic" not in pattern.lower() for pattern in settings["enabledModels"])
    assert any(pattern.startswith("openai-codex/") for pattern in settings["enabledModels"])
    assert "kimi-coding/k3" in settings["enabledModels"]
    assert any(pattern.startswith("openrouter/") for pattern in settings["enabledModels"])


def test_pi_project_subagents_enforce_model_scope_with_twenty_workers() -> None:
    """Keep Tintin workers concurrent while rejecting models outside Pi's scope."""
    settings = json.loads((REPO_ROOT / ".pi" / "subagents.json").read_text())

    assert settings == {"maxConcurrent": 20, "scopeModels": True}


def test_pi_project_explore_agent_is_disabled() -> None:
    """Keep Tintin's unbounded built-in Explore agent unavailable in this repo."""
    text = (REPO_ROOT / ".pi" / "agents" / "Explore.md").read_text()
    _, frontmatter, _ = text.split("---", 2)

    assert yaml.safe_load(frontmatter) == {"enabled": False}


def test_pi_project_append_system_forbids_anthropic_agents() -> None:
    """Tell Pi sessions and project subagents not to select Anthropic models."""
    text = (REPO_ROOT / ".pi" / "APPEND_SYSTEM.md").read_text()

    assert "Anthropic" in text
    assert "Do not select Anthropic providers or models" in text
    assert "Do not launch subagents" in text
    assert "openai-codex" in text
    assert "kimi-coding" in text
    assert "openrouter" in text


def test_pi_review_policy_wires_routing_and_audit_helpers() -> None:
    """Keep natural-language orchestration connected to tested routing behavior."""
    text = (
        REPO_ROOT / "agent" / "skills" / "_shared" / "repo-review-full-analysis.md"
    ).read_text()

    assert "pi_review_routing.py plan" in text
    assert "pi_review_routing.py extract-report" in text
    assert "pi_review_routing.py validate-report" in text
    assert "pi_review_routing.py transcript-stats" in text
    assert "pi_review_routing.py provenance" in text
    assert "extract a unique worker JSON object from harmless surrounding prose" in text
    assert '"severity": "block"' in text
    assert "The worker does not render Markdown or attach provenance" in text
    assert text.count("./.venv/bin/python agent/_shared/pi_review_routing.py") == 6
    assert '"${PI_REVIEW_PYTHON}" agent/_shared/pi_review_routing.py' in text
    assert "./.venv/bin/python agent/_shared/review_failure.py deliver" in text
    assert "python3 agent/_shared/pi_review_routing.py" not in text
    assert "Insert a `## PR health` section after the `## Provider incidents`" in text
    assert "Prepend a `## PR health` section" not in text
    assert "run_in_background: true" in text
    assert "${PI_REVIEW_AFTERCARE_MANIFEST%.json}.assignments" in text
    assert re.search(r"never\s+put a glob in a worker prompt", text)
    assert re.search(r"never repair assignment paths\s+with", text)
    assert "480-second foreground deadline" in text
    assert "one validated report per selected skill" in text
    assert "resume the same worker once" in text
    assert "Do not repeat the review" in text
    assert "unfinished second pass to aftercare" in text
    assert re.search(r"late Codex-verified\s+findings", text)
    assert "Output file:" in text
    assert "get_subagent_result(wait: true)" in text
    assert "get_subagent_result(wait: false)" in text
    assert re.search(r"poll them a\s+second time in the foreground", text)
    assert "at most 360 seconds" in text
    assert "at most 6 turns per" in text
    assert "parallel Codex verification wave" in text
    assert "Record its audit status as `deferred`" in text
    assert "free-pool-only findings never enter aggregation directly" in text
    assert re.search(r"successful Codex\s+pass's effective model", text)
    assert re.search(r"successful Codex pass's\s+`max_turns`", text)
    assert "`openai-codex/gpt-5.6-sol` and `high` thinking" not in text
    assert "max_turns: <plan.max_turns>" in text
    assert "Model tiers are fixed by checklist" in text
    assert "Smart model tier" in text
    assert "Mechanical model tier" in text
    assert "| Skill | Model tier | Pass | Model | Thinking | Max turns | Status |" in text
    assert re.search(
        r"Gracefully wrapped `steered` attempts proceed to report\s+validation",
        text,
    )
    assert re.search(r"hard-aborted\s+turn-budget result is a failed attempt", text)
    assert "review_failure.py deliver" in text
    assert re.search(r"every terminal failure.*delivery helper", text, re.DOTALL)
    assert re.search(r"never merely print the audit\s+and stop", text)
    assert re.search(r"both Codex\s+and the selected free-pool tier pass provider", text)
    assert "fallback_candidates" in text
    assert "skip remaining candidates from that provider" in text
    assert "authentication never triggers Codex fallback" in text
    assert "Codex fallback" in text
    assert "Free-pool review failed; only Codex ran." in text
    assert "## Provider incidents" in text
    assert re.search(
        r"authentication.*quota/capacity.*before every\s+other `review_body` section",
        text,
        re.DOTALL,
    )
    assert re.search(r"one bullet per\s+affected attempt", text)
    assert re.search(r"exact model selector and diagnostic", text)
    assert re.search(r"successful Codex pass's effective\s+model to the end", text)
    assert "claude -p --dangerously-skip-permissions" in text
    assert "codex exec --dangerously-bypass-approvals-and-sandbox" in text


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_pi_review_launcher_runs_one_targeted_skill_to_completion(tmp_path: Path) -> None:
    """Persist a live JSON transcript and return only Pi's final response.

    :param tmp_path: Temporary directory containing the fake Pi executable.
    """
    sh = importlib.import_module("sh")
    launcher = REPO_ROOT / "agent" / "_shared" / "run_pi_review.sh"
    pi = tmp_path / "pi"
    pi.write_text(
        "#!/bin/bash\n"
        '[[ "${SYNTH_SETTER_PI_REVIEW:-}" == 1 ]]\n'
        "[[ \" $* \" == *' --mode json '* ]]\n"
        "[[ \" $* \" == *' --no-session '* ]]\n"
        "[[ \" $* \" == *'PR #2052'* ]]\n"
        'printf \'{"type":"message_start","message":{"role":"assistant",\'\n'
        'printf \'"content":[],"provider":"openai-codex",\'\n'
        'printf \'"model":"gpt-5.6-terra"}}\\n\'\n'
        'printf \'{"type":"message_end","message":{"role":"assistant",\'\n'
        'printf \'"content":[{"type":"text","text":"pi-complete"}]}}\\n\'\n'
    )
    pi.chmod(0o755)

    stderr = io.BytesIO()
    result = sh.Command(str(launcher))(
        "repo-review-full",
        "--target",
        "2052",
        _cwd=REPO_ROOT,
        _env={"PATH": f"{tmp_path}:{os.environ['PATH']}"},
        _err=stderr,
    )
    assert str(result).strip() == "pi-complete"
    stderr_text = stderr.getvalue().decode()
    match = re.search(r"Live Pi transcript: (.+\.jsonl)", stderr_text)
    assert match is not None
    transcript = Path(match.group(1))
    assert transcript.is_absolute()
    try:
        assert transcript.read_text().count("message_") == 2
        assert "openai-codex/gpt-5.6-terra started" in stderr_text
    finally:
        transcript.unlink(missing_ok=True)


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_pi_review_launcher_manifest_starts_detached_aftercare(tmp_path: Path) -> None:
    """Drive foreground completion through the real detached-handoff path.

    :param tmp_path: Temporary fake Pi executable and aftercare marker.
    """
    sh = importlib.import_module("sh")
    launcher = REPO_ROOT / "agent/_shared/run_pi_review.sh"
    marker = tmp_path / "aftercare-ran"
    manifest_path_file = tmp_path / "manifest-path"
    pi = tmp_path / "pi"
    pi.write_text(
        "#!/bin/bash\n"
        'if [[ "${SYNTH_SETTER_PI_REVIEW_AFTERCARE:-}" == 1 ]]; then\n'
        '  touch "${AFTERCARE_MARKER}"\n'
        "  exit 0\n"
        "fi\n"
        'printf \'%s\\n\' "${PI_REVIEW_AFTERCARE_MANIFEST}" > "${MANIFEST_PATH_FILE}"\n'
        "cat > \"${PI_REVIEW_AFTERCARE_MANIFEST}\" <<'JSON'\n"
        '{"version":1,"mode":"no-comments","repo":"tinaudio/synth-setter",'
        '"pr_number":2174,"base_sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        '"head_sha":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","target":"PR #2174",'
        '"deferred_passes":[{"skill":"correctness-review","pass_name":"free-pool",'
        '"origin":"primary","model":"kimi-coding/k3",'
        '"verification_model":"openai-codex/gpt-5.6-sol","thinking":"high"}],'
        '"foreground_fingerprints":[]}\n'
        "JSON\n"
        'printf \'%s\\n\' \'{"type":"message_end","message":{"role":"assistant",'
        '"content":"foreground-complete"}}\'\n'
    )
    pi.chmod(0o755)

    stderr = io.BytesIO()
    result = sh.Command(str(launcher))(
        "repo-review-full-no-comments",
        "--target",
        "2174",
        _cwd=REPO_ROOT,
        _env={
            **os.environ,
            "AFTERCARE_MARKER": str(marker),
            "MANIFEST_PATH_FILE": str(manifest_path_file),
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
        },
        _err=stderr,
    )
    manifest = Path(manifest_path_file.read_text().strip())
    transcript_match = re.search(r"Live Pi transcript: (.+\.jsonl)", stderr.getvalue().decode())
    assert transcript_match is not None
    transcript = Path(transcript_match.group(1))
    try:
        deadline = time.monotonic() + 2
        while not marker.exists() and time.monotonic() < deadline:
            pass
        assert str(result).strip() == "foreground-complete"
        assert marker.exists()
    finally:
        manifest.unlink(missing_ok=True)
        manifest.with_suffix(".log").unlink(missing_ok=True)
        transcript.unlink(missing_ok=True)


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_pi_review_launcher_falls_back_to_path_python_without_repo_venv(
    tmp_path: Path,
) -> None:
    """Use the active PATH interpreter when the repo copy has no local ``.venv``.

    :param tmp_path: Temporary repo copy and fake Pi executable directory.
    """
    sh = importlib.import_module("sh")
    repo_root = tmp_path / "repo"
    launcher = repo_root / "agent" / "_shared" / "run_pi_review.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text((REPO_ROOT / "agent" / "_shared" / "run_pi_review.sh").read_text())
    launcher.chmod(0o755)
    routing = repo_root / "agent" / "_shared" / "pi_review_routing.py"
    routing.write_text((REPO_ROOT / "agent" / "_shared" / "pi_review_routing.py").read_text())
    pi = tmp_path / "pi"
    pi.write_text(
        "#!/bin/bash\n"
        '[[ "${SYNTH_SETTER_PI_REVIEW:-}" == 1 ]]\n'
        'printf \'{"type":"message_start","message":{"role":"assistant",\'\n'
        'printf \'"content":[],"provider":"openai-codex",\'\n'
        'printf \'"model":"gpt-5.6-terra"}}\\n\'\n'
        'printf \'{"type":"message_end","message":{"role":"assistant",\'\n'
        'printf \'"content":[{"type":"text","text":"pi-complete"}]}}\\n\'\n'
    )
    pi.chmod(0o755)

    stderr = io.BytesIO()
    result = sh.Command(str(launcher))(
        "repo-review-full",
        "--target",
        "2052",
        _cwd=repo_root,
        _env={"PATH": f"{tmp_path}:{os.environ['PATH']}"},
        _err=stderr,
    )
    assert str(result).strip() == "pi-complete"
    stderr_text = stderr.getvalue().decode()
    match = re.search(r"Live Pi transcript: (.+\.jsonl)", stderr_text)
    assert match is not None
    transcript = repo_root / match.group(1)
    try:
        assert transcript.read_text().count("message_") == 2
        assert "openai-codex/gpt-5.6-terra started" in stderr_text
    finally:
        transcript.unlink(missing_ok=True)


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_pi_review_launcher_nonzero_exit_withholds_intermediate_text(tmp_path: Path) -> None:
    """Keep a failed Pi run from publishing an intermediate assistant message.

    :param tmp_path: Temporary directory containing the failing fake Pi executable.
    """
    sh = importlib.import_module("sh")
    launcher = REPO_ROOT / "agent" / "_shared" / "run_pi_review.sh"
    pi = tmp_path / "pi"
    pi.write_text(
        "#!/bin/bash\n"
        'echo \'{"type":"message_end","message":{"role":"assistant",'
        '"content":"intermediate"}}\'\n'
        "exit 7\n"
    )
    pi.chmod(0o755)
    stdout = io.BytesIO()
    stderr = io.BytesIO()

    with pytest.raises(sh.ErrorReturnCode):
        sh.Command(str(launcher))(
            "repo-review-full-no-comments",
            _cwd=REPO_ROOT,
            _env={"PATH": f"{tmp_path}:{os.environ['PATH']}"},
            _out=stdout,
            _err=stderr,
        )

    assert stdout.getvalue() == b""
    assert "Pi review host failed; inspect live transcript:" in stderr.getvalue().decode()


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_pi_review_launcher_rejects_nested_session(tmp_path: Path) -> None:
    """Stop a child Pi session from recursively launching the harness.

    :param tmp_path: Temporary directory placed first on PATH.
    """
    sh = importlib.import_module("sh")
    launcher = REPO_ROOT / "agent" / "_shared" / "run_pi_review.sh"

    with pytest.raises(sh.ErrorReturnCode_2):
        sh.Command(str(launcher))(
            "repo-review-full-no-comments",
            _cwd=REPO_ROOT,
            _env={
                "PATH": f"{tmp_path}:{os.environ['PATH']}",
                "SYNTH_SETTER_PI_REVIEW": "1",
            },
        )


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_pi_review_launcher_rejects_zero_pr_number(tmp_path: Path) -> None:
    """Reject PR zero before invoking Pi.

    :param tmp_path: Temporary directory placed first on PATH.
    """
    sh = importlib.import_module("sh")
    launcher = REPO_ROOT / "agent" / "_shared" / "run_pi_review.sh"

    with pytest.raises(sh.ErrorReturnCode_2):
        sh.Command(str(launcher))(
            "repo-review-full-no-comments",
            "--target",
            "0",
            _cwd=REPO_ROOT,
            _env={"PATH": f"{tmp_path}:{os.environ['PATH']}"},
        )


def test_aftercare_manifest_free_pool_codex_requires_fallback_origin() -> None:
    """Distinguish independent free-pool coverage from an explicit Codex fallback."""
    base = {
        "version": 1,
        "mode": "no-comments",
        "repo": "tinaudio/synth-setter",
        "pr_number": 2174,
        "base_sha": "a" * 40,
        "head_sha": "b" * 40,
        "target": "PR #2174",
        "foreground_fingerprints": [],
    }
    primary = {
        "skill": "correctness-review",
        "pass_name": "free-pool",
        "origin": "primary",
        "model": "openai-codex/gpt-5.6-sol",
        "verification_model": "openai-codex/gpt-5.6-sol",
        "thinking": "high",
    }

    with pytest.raises(ValueError, match="origin does not match"):
        AftercareManifest.model_validate_json(json.dumps({**base, "deferred_passes": [primary]}))

    fallback = {**primary, "origin": "codex-fallback"}
    manifest = AftercareManifest.model_validate_json(
        json.dumps({**base, "deferred_passes": [fallback]})
    )
    assert manifest.deferred_passes[0].origin == "codex-fallback"


def test_pi_review_launcher_declares_detached_aftercare_contract() -> None:
    """Keep deferred second passes auditable after the foreground host exits."""
    launcher = (REPO_ROOT / "agent/_shared/run_pi_review.sh").read_text()
    aftercare = (REPO_ROOT / "agent/_shared/run_pi_review_aftercare.py").read_text()

    assert "PI_REVIEW_AFTERCARE_MANIFEST" in launcher
    assert 'export PI_REVIEW_PYTHON="${review_python}"' in launcher
    assert '"${review_python}" agent/_shared/run_pi_review_aftercare.py' in launcher
    assert "start_new_session" in aftercare
    assert "openai-codex" in aftercare
    assert "gpt-5.6-terra" in aftercare
    assert "anthropic" not in aftercare.lower()
    analysis = (REPO_ROOT / "agent/skills/_shared/repo-review-full-analysis.md").read_text()
    assert '"${PI_REVIEW_PYTHON}" agent/_shared/pi_review_routing.py worker-prompt' in analysis
    assert '`pass_name: "codex"`' in analysis


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_pi_review_aftercare_launcher_runs_detached_pinned_process(tmp_path: Path) -> None:
    """Drive the real aftercare entrypoint through manifest validation and process launch.

    :param tmp_path: Temporary directory containing the fake Pi executable.
    """
    launcher = REPO_ROOT / "agent/_shared/run_pi_review_aftercare.py"
    manifest = REPO_ROOT / ".agent-reviews/test-aftercare-manifest.json"
    marker = tmp_path / "aftercare-ran"
    pi = tmp_path / "pi"
    pi.write_text(
        "#!/bin/bash\n"
        '[[ "${SYNTH_SETTER_PI_REVIEW_AFTERCARE:-}" == 1 ]]\n'
        '[[ -z "${SYNTH_SETTER_PI_REVIEW:-}" ]]\n'
        f"touch {marker}\n"
    )
    pi.chmod(0o755)
    manifest.parent.mkdir(exist_ok=True)
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
                "deferred_passes": [
                    {
                        "skill": "correctness-review",
                        "pass_name": "free-pool",
                        "origin": "primary",
                        "model": "kimi-coding/k3",
                        "verification_model": "openai-codex/gpt-5.6-sol",
                        "thinking": "high",
                    }
                ],
                "foreground_fingerprints": [],
            }
        )
    )
    try:
        sh = importlib.import_module("sh")
        dry_run = sh.Command(sys.executable)(
            launcher,
            manifest,
            "--dry-run",
            _cwd=REPO_ROOT,
            _env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        )
        command = json.loads(str(dry_run))
        assert str(manifest.resolve()) in command[-1]
        assert "agent/skills/_shared/repo-review-aftercare.md" in command[-1]

        result = sh.Command(sys.executable)(
            launcher,
            manifest,
            _cwd=REPO_ROOT,
            _env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        )
        pid = int(str(result))
        deadline = time.monotonic() + 2
        while not marker.exists() and time.monotonic() < deadline:
            pass
        assert marker.exists()
        _assert_process_terminated(pid, timeout=2)
    finally:
        manifest.unlink(missing_ok=True)
        manifest.with_suffix(".log").unlink(missing_ok=True)


def test_no_comments_review_uses_isolated_findings_path() -> None:
    """Prevent concurrent reviews from sharing one global findings file."""
    text = (REPO_ROOT / "agent/skills/repo-review-full-no-comments/SKILL.md").read_text()

    fixed_findings_path = Path("/").joinpath("tmp", "repo-review-full-no-comments-findings.json")
    assert "review_sentinel.py findings" in text
    assert str(fixed_findings_path) not in text
    assert "exact printed path" in text
    assert "pi_review_render.py" in text
    assert "--remove-payload" in text


def test_no_comments_review_reserves_pass_short_form_for_zero_diff() -> None:
    """Keep non-empty reviews from discarding the Pi audit on PASS."""
    text = (REPO_ROOT / "agent/skills/repo-review-full-no-comments/SKILL.md").read_text()

    assert "If `is_zero_diff == true`" in text
    assert "A non-zero diff with no findings" in text
    assert "pi_review_render.py" in text
    assert "complete `## Pi review audit`" in text


def test_full_review_skills_define_flat_pi_orchestration() -> None:
    """Avoid unsupported nested Tintin fan-out while preserving the pipeline."""
    for skill in ("repo-review-full", "repo-review-full-no-comments"):
        text = (REPO_ROOT / "agent" / "skills" / skill / "SKILL.md").read_text()
        assert "Tintin" in text
        assert "pr-review-worker" in text
        assert "flat" in text.lower()
        assert "Agent" in text
        assert "allocation, fallback, merge" in text
        assert "SYNTH_SETTER_PI_REVIEW=1" in text
        if skill == "repo-review-full-no-comments":
            assert "Pi PASS report" in text


def test_headless_hook_review_defaults_match_pinned_tier() -> None:
    """Pin the headless hook launcher's review model to the #1906 tier.

    The doc-drift PostToolUse hook launches ``run_agent_prompt`` in
    ``agent/hooks/_lib.sh``; its default model must match the non-correctness
    review tier pinned in the ``pr-review-worker-fast`` agent files so a
    headless advisory never silently falls back to the session default.
    """
    lib = (REPO_ROOT / "agent" / "hooks" / "_lib.sh").read_text()
    claude_default = re.search(r"CLAUDE_REVIEW_MODEL:-([^}]+)", lib)
    codex_default = re.search(r"CODEX_REVIEW_MODEL:-([^}]+)", lib)
    assert claude_default is not None, "run_agent_prompt must default CLAUDE_REVIEW_MODEL"
    assert codex_default is not None, "run_agent_prompt must default CODEX_REVIEW_MODEL"

    _, claude_frontmatter, _ = (
        (REPO_ROOT / ".claude" / "agents" / "pr-review-worker-fast.md").read_text().split("---", 2)
    )
    assert yaml.safe_load(claude_frontmatter)["model"] == claude_default.group(1)

    with (REPO_ROOT / ".codex" / "agents" / "pr-review-worker-fast.toml").open("rb") as file:
        assert tomllib.load(file)["model"] == codex_default.group(1)


def test_codex_review_launcher_resolves_runtime_model_policy() -> None:
    """Exercise the Codex launcher entry point without spending inference tokens."""
    launcher = REPO_ROOT / "agent" / "_shared" / "run_codex_review_agent.py"

    for role, providers in _ROLE_MODELS.items():
        stdout = io.StringIO()
        original_argv = sys.argv
        prompt_args = ["--prompt", "routing probe"]
        if role == "pr-review-orchestrator":
            prompt_args = [
                "--skill-brief",
                str(REPO_ROOT / "agent" / "skills" / "repo-review-full-no-comments" / "SKILL.md"),
                "--target",
                "1234",
            ]
        sys.argv = [
            str(launcher),
            role,
            *prompt_args,
            "--dry-run",
        ]
        try:
            with contextlib.redirect_stdout(stdout), pytest.raises(SystemExit) as exit_info:
                runpy.run_path(str(launcher), run_name="__main__")
        finally:
            sys.argv = original_argv

        assert exit_info.value.code == 0
        resolved = json.loads(stdout.getvalue())
        command = resolved["command"]
        assert command[0:2] == ["codex", "exec"]
        assert command[command.index("--model") + 1] == providers["codex"][0]
        assert f'model_reasoning_effort="{providers["codex"][1]}"' in command
        expected_timeout = 900 if role == "pr-review-orchestrator" else 720
        assert resolved["timeout_s"] == expected_timeout
        if role == "pr-review-orchestrator":
            brief = resolved["prompt"].split("\n\n", 1)[1]
            assert brief.startswith("## Orchestrator agent brief\n")
            assert "PR #1234" in brief
            assert "<N>" not in brief


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_codex_review_python_launcher_executes_resolved_command(tmp_path: Path) -> None:
    """Protect direct-entrypoint execution parity with the shell wrapper.

    :param tmp_path: Directory for the fake Codex executable.
    """
    sh = importlib.import_module("sh")
    launcher = REPO_ROOT / "agent" / "_shared" / "run_codex_review_agent.py"
    codex = tmp_path / "codex"
    codex.write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' "
        '\'{"type":"item.completed","item":{"type":"agent_message",'
        '"text":"structured report"}}\'\n'
    )
    codex.chmod(0o755)
    shadowed_python = tmp_path / "python3"
    shadowed_python.write_text("#!/bin/bash\nexit 1\n")
    shadowed_python.chmod(0o755)

    result = sh.Command(sys.executable)(
        str(launcher),
        "pr-review-worker-fast",
        "--prompt",
        "routing probe",
        _cwd=REPO_ROOT,
        _env={"PATH": f"{tmp_path}:{os.environ['PATH']}"},
    )

    assert str(result) == "structured report"


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_codex_review_python_launcher_ignores_blank_ndjson_lines(tmp_path: Path) -> None:
    """Preserve valid reports around blank NDJSON records.

    :param tmp_path: Directory for the fake Codex executable.
    """
    sh = importlib.import_module("sh")
    launcher = REPO_ROOT / "agent" / "_shared" / "run_codex_review_agent.py"
    codex = tmp_path / "codex"
    codex.write_text(
        "#!/bin/bash\n"
        "printf '\\n'\n"
        "printf '%s\\n' "
        '\'{"type":"item.completed","item":{"type":"agent_message",'
        '"text":"structured report"}}\'\n'
        "printf '  \\n'\n"
    )
    codex.chmod(0o755)

    result = sh.Command(sys.executable)(
        str(launcher),
        "pr-review-worker-fast",
        "--prompt",
        "routing probe",
        _cwd=REPO_ROOT,
        _env={"PATH": f"{tmp_path}:{os.environ['PATH']}"},
    )

    assert str(result) == "structured report"


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_codex_review_orchestrator_default_timeout_launches(tmp_path: Path) -> None:
    """Protect the orchestrator's default-deadline execution path.

    :param tmp_path: Directory for the fake Codex executable.
    """
    sh = importlib.import_module("sh")
    launcher = REPO_ROOT / "agent" / "_shared" / "run_codex_review_agent.sh"
    codex = tmp_path / "codex"
    codex.write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' "
        '\'{"type":"item.completed","item":{"type":"agent_message",'
        '"text":"orchestrator report"}}\'\n'
    )
    codex.chmod(0o755)

    result = sh.Command(str(launcher))(
        "pr-review-orchestrator",
        "--prompt",
        "routing probe",
        _cwd=REPO_ROOT,
        _env={"PATH": f"{tmp_path}:{os.environ['PATH']}"},
    )

    assert str(result) == "orchestrator report"


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_codex_review_shell_launcher_dry_run_e2e() -> None:
    """Run the user-facing shell launcher without starting model inference."""
    sh = importlib.import_module("sh")
    launcher = REPO_ROOT / "agent" / "_shared" / "run_codex_review_agent.sh"
    result = sh.Command(str(launcher))(
        "pr-review-worker-deep",
        "--prompt",
        "routing probe",
        "--dry-run",
        _cwd=REPO_ROOT,
    )

    resolved = json.loads(str(result))
    assert resolved["command"][3] == "gpt-5.6-sol"
    assert 'model_reasoning_effort="high"' in resolved["command"]
    assert resolved["prompt"].endswith("routing probe")


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_codex_review_shell_launcher_reads_prompt_file(tmp_path: Path) -> None:
    """Exercise the production prompt-file path through the shell launcher.

    :param tmp_path: Temporary directory for the worker prompt file.
    """
    sh = importlib.import_module("sh")
    launcher = REPO_ROOT / "agent" / "_shared" / "run_codex_review_agent.sh"
    prompt_file = tmp_path / "worker-prompt.txt"
    prompt_file.write_text("prompt-file routing probe")

    result = sh.Command(str(launcher))(
        "pr-review-worker-fast",
        "--prompt-file",
        prompt_file,
        "--dry-run",
        _cwd=REPO_ROOT,
    )

    resolved = json.loads(str(result))
    assert resolved["command"][3] == "gpt-5.6-terra"
    assert resolved["prompt"].endswith("prompt-file routing probe")


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_codex_review_shell_launcher_captures_only_agent_message(tmp_path: Path) -> None:
    """Exercise the normal shell path with a deterministic Codex executable.

    :param tmp_path: Temporary directory containing the fake Codex executable.
    """
    sh = importlib.import_module("sh")
    launcher = REPO_ROOT / "agent" / "_shared" / "run_codex_review_agent.sh"
    codex = tmp_path / "codex"
    codex.write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' "
        '\'{"type":"item.completed","item":{"type":"agent_message",'
        '"text":"structured report"}}\'\n'
    )
    codex.chmod(0o755)

    result = sh.Command(str(launcher))(
        "pr-review-worker-fast",
        "--prompt",
        "routing probe",
        _cwd=REPO_ROOT,
        _env={"PATH": f"{tmp_path}:{os.environ['PATH']}"},
    )

    assert str(result) == "structured report"


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_codex_review_shell_launcher_withholds_caller_stdin(tmp_path: Path) -> None:
    """Keep the caller's stdin out of the worker prompt.

    ``codex exec`` appends piped stdin to the prompt and blocks until EOF, so a
    caller holding stdin open would stall the review gate.

    :param tmp_path: Temporary directory containing the fake Codex executable.
    """
    sh = importlib.import_module("sh")
    launcher = REPO_ROOT / "agent" / "_shared" / "run_codex_review_agent.sh"
    codex = tmp_path / "codex"
    codex.write_text(
        "#!/bin/bash\n"
        "leaked=$(cat)\n"
        'jq -cn --arg text "stdin=[${leaked}]" \\\n'
        '  \'{type: "item.completed", item: {type: "agent_message", text: $text}}\'\n'
    )
    codex.chmod(0o755)

    result = sh.Command(str(launcher))(
        "pr-review-worker-fast",
        "--prompt",
        "routing probe",
        _cwd=REPO_ROOT,
        _env={"PATH": f"{tmp_path}:{os.environ['PATH']}"},
        _in="caller-owned stdin payload",
    )

    assert str(result) == "stdin=[]"


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_codex_review_shell_launcher_timeout_kills_hung_run(tmp_path: Path) -> None:
    """Ensure timeout diagnostics include the active deadline.

    :param tmp_path: Directory for the fake Codex executable.
    """
    sh = importlib.import_module("sh")
    launcher = REPO_ROOT / "agent" / "_shared" / "run_codex_review_agent.sh"
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/bash\nexec sleep 5\n")
    codex.chmod(0o755)

    with pytest.raises(sh.ErrorReturnCode) as exc_info:
        sh.Command(str(launcher))(
            "pr-review-worker-fast",
            "--prompt",
            "routing probe",
            _cwd=REPO_ROOT,
            _env={
                "PATH": f"{tmp_path}:{os.environ['PATH']}",
                "CODEX_REVIEW_TIMEOUT": "1",
            },
        )

    assert exc_info.value.exit_code != 0
    assert b"codex exec timed out after 1s" in exc_info.value.stderr


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
@pytest.mark.parametrize(
    ("timeout", "expected_error"),
    [
        ("-1", b"must be a positive integer"),
        ("9" * 400, b"must be between 1 and 86400"),
    ],
)
def test_codex_review_shell_launcher_invalid_timeout_rejected_before_launch(
    tmp_path: Path,
    timeout: str,
    expected_error: bytes,
) -> None:
    """Ensure invalid deadlines prevent subprocess startup.

    :param tmp_path: Directory for the fake Codex executable.
    :param timeout: Invalid timeout override under test.
    :param expected_error: Diagnostic fragment required on standard error.
    """
    sh = importlib.import_module("sh")
    launcher = REPO_ROOT / "agent" / "_shared" / "run_codex_review_agent.sh"
    launched = tmp_path / "launched"
    codex = tmp_path / "codex"
    codex.write_text(f"#!/bin/bash\ntouch {launched}\n")
    codex.chmod(0o755)

    with pytest.raises(sh.ErrorReturnCode) as exc_info:
        sh.Command(str(launcher))(
            "pr-review-worker-fast",
            "--prompt",
            "routing probe",
            _cwd=REPO_ROOT,
            _env={
                "PATH": f"{tmp_path}:{os.environ['PATH']}",
                "CODEX_REVIEW_TIMEOUT": timeout,
            },
        )

    assert expected_error in exc_info.value.stderr
    assert not launched.exists()


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_codex_review_shell_launcher_success_starts_no_watchdog_timer(tmp_path: Path) -> None:
    """Guard against reintroducing an external watchdog process.

    :param tmp_path: Directory for the fake Codex and sleep executables.
    """
    sh = importlib.import_module("sh")
    launcher = REPO_ROOT / "agent" / "_shared" / "run_codex_review_agent.sh"
    sleep_pid_file = tmp_path / "sleep.pid"
    sleep = tmp_path / "sleep"
    sleep.write_text(
        '#!/bin/bash\necho "$$" > "${WATCHDOG_SLEEP_PID_FILE}"\nexec /bin/sleep "$@"\n'
    )
    sleep.chmod(0o755)
    codex = tmp_path / "codex"
    codex.write_text(
        "#!/bin/bash\n"
        "/bin/sleep 0.2\n"
        "printf '%s\\n' "
        '\'{"type":"item.completed","item":{"type":"agent_message",'
        '"text":"structured report"}}\'\n'
    )
    codex.chmod(0o755)

    result = sh.Command(str(launcher))(
        "pr-review-worker-fast",
        "--prompt",
        "routing probe",
        _cwd=REPO_ROOT,
        _env={
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "WATCHDOG_SLEEP_PID_FILE": str(sleep_pid_file),
        },
    )

    assert str(result) == "structured report"
    assert not sleep_pid_file.exists()


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
@pytest.mark.parametrize("parent_term_trap", ["trap '' TERM\n", ""])
def test_codex_review_shell_launcher_timeout_force_kills_process_group(
    tmp_path: Path,
    parent_term_trap: str,
) -> None:
    """Ensure cleanup reaches descendants that ignore SIGTERM.

    :param tmp_path: Directory for the fake Codex executable.
    :param parent_term_trap: Whether the process-group leader ignores SIGTERM.
    """
    sh = importlib.import_module("sh")
    launcher = REPO_ROOT / "agent" / "_shared" / "run_codex_review_agent.sh"
    child_pid_file = tmp_path / "child.pid"
    codex = tmp_path / "codex"
    codex.write_text(
        "#!/bin/bash\n"
        f"{parent_term_trap}"
        "bash -c \"trap '' TERM; exec /bin/sleep 5\" &\n"
        'echo "$!" > "${CODEX_CHILD_PID_FILE}"\n'
        "wait\n"
    )
    codex.chmod(0o755)

    with pytest.raises(sh.ErrorReturnCode) as exc_info:
        sh.Command(str(launcher))(
            "pr-review-worker-fast",
            "--prompt",
            "routing probe",
            _cwd=REPO_ROOT,
            _env={
                "PATH": f"{tmp_path}:{os.environ['PATH']}",
                "CODEX_CHILD_PID_FILE": str(child_pid_file),
                "CODEX_REVIEW_TIMEOUT": "1",
            },
        )

    assert b"codex exec timed out after 1s" in exc_info.value.stderr
    child_pid = int(child_pid_file.read_text())
    _assert_process_terminated(child_pid)


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_codex_review_python_launcher_signal_terminates_process_group(tmp_path: Path) -> None:
    """Ensure launcher termination reaps signal-resistant descendants.

    :param tmp_path: Directory for the fake Codex executable.
    """
    sh = importlib.import_module("sh")
    launcher = REPO_ROOT / "agent" / "_shared" / "run_codex_review_agent.py"
    child_pid_file = tmp_path / "child.pid"
    codex = tmp_path / "codex"
    codex.write_text(
        "#!/bin/bash\n"
        "trap '' TERM\n"
        "bash -c \"trap '' TERM; exec /bin/sleep 30\" &\n"
        'echo "$!" > "${CODEX_CHILD_PID_FILE}"\n'
        "wait\n"
    )
    codex.chmod(0o755)

    process = sh.Command(sys.executable)(
        str(launcher),
        "pr-review-worker-fast",
        "--prompt",
        "routing probe",
        _bg=True,
        _cwd=REPO_ROOT,
        _env={
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "CODEX_CHILD_PID_FILE": str(child_pid_file),
        },
    )
    deadline = time.monotonic() + 5
    while not child_pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert child_pid_file.exists()

    process.signal(signal.SIGTERM)
    with pytest.raises(sh.ErrorReturnCode):
        process.wait()

    child_pid = int(child_pid_file.read_text())
    _assert_process_terminated(child_pid)


_OPENCODE_LAUNCHER_PY = REPO_ROOT / "agent" / "_shared" / "run_opencode_review_agent.py"
_OPENCODE_LAUNCHER_SH = REPO_ROOT / "agent" / "_shared" / "run_opencode_review_agent.sh"


def _path_without_opencode(tmp_path: Path) -> str:
    """Expose the shell launcher's dependencies on PATH without any opencode binary.

    :param tmp_path: Temporary directory to host the restricted bin directory.
    :returns: PATH string containing jq and uv plus the system directories.
    """
    bin_dir = tmp_path / "restricted-bin"
    bin_dir.mkdir()
    for tool in ("jq", "uv"):
        target = shutil.which(tool)
        if target is None:
            pytest.fail(f"{tool} is required to run the shell launcher")
        (bin_dir / tool).symlink_to(target)
    return f"{bin_dir}:/usr/bin:/bin"


def _run_opencode_launcher_py(argv: list[str]) -> tuple[int, str]:
    """Drive the python launcher exactly as the shell wrapper does.

    :param argv: Arguments after the script path.
    :returns: Exit code and captured stdout.
    """
    stdout = io.StringIO()
    # patch.object restores sys.argv even when the launcher raises, so runs
    # stay isolated from each other.
    with (
        mock.patch.object(sys, "argv", [str(_OPENCODE_LAUNCHER_PY), *argv]),
        contextlib.redirect_stdout(stdout),
        contextlib.redirect_stderr(io.StringIO()),
        pytest.raises(SystemExit) as exit_info,
    ):
        runpy.run_path(str(_OPENCODE_LAUNCHER_PY), run_name="__main__")
    code = exit_info.value.code
    return (code if isinstance(code, int) else 1), stdout.getvalue()


def test_opencode_role_pins_orchestrator_role_has_no_toml() -> None:
    """Keep the opencode cross-model pass scoped to the review workers."""
    assert not (REPO_ROOT / ".opencode" / "agents" / "pr-review-orchestrator.toml").exists()


def test_opencode_launcher_dry_run_emits_pinned_command() -> None:
    """Exercise the opencode launcher entry point without spending inference tokens."""
    for role, providers in _ROLE_MODELS.items():
        if "opencode" not in providers:
            continue
        model, variant = providers["opencode"]

        code, output = _run_opencode_launcher_py([role, "--prompt", "routing probe", "--dry-run"])

        assert code == 0
        resolved = json.loads(output)
        command = resolved["command"]
        assert command[0:2] == ["opencode", "run"]
        assert command[command.index("-m") + 1] == model
        assert command[command.index("--agent") + 1] == "pr-reviewer"
        assert command[command.index("--format") + 1] == "json"
        if variant is None:
            assert "--variant" not in command
        else:
            assert command[command.index("--variant") + 1] == variant
        assert resolved["prompt"].endswith("routing probe")


def test_opencode_launcher_prompt_file_read_into_prompt(tmp_path: Path) -> None:
    """Read the worker prompt from a file when ``--prompt-file`` is selected.

    :param tmp_path: Temporary directory for the prompt file.
    """
    prompt_file = tmp_path / "worker-prompt.txt"
    prompt_file.write_text("prompt-file routing probe")

    code, output = _run_opencode_launcher_py(
        ["pr-review-worker-fast", "--prompt-file", str(prompt_file), "--dry-run"]
    )

    assert code == 0
    assert json.loads(output)["prompt"].endswith("prompt-file routing probe")


def test_opencode_launcher_orchestrator_role_rejected() -> None:
    """Refuse roles that have no opencode execution policy."""
    code, _ = _run_opencode_launcher_py(
        ["pr-review-orchestrator", "--prompt", "routing probe", "--dry-run"]
    )

    assert code != 0


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_opencode_shell_launcher_dry_run_without_binary_round_trips_json(tmp_path: Path) -> None:
    """Keep the binary pre-flight after the dry-run branch so opencode-less CI can dry-run.

    :param tmp_path: Temporary directory for the restricted PATH.
    """
    sh = importlib.import_module("sh")

    result = sh.Command(str(_OPENCODE_LAUNCHER_SH))(
        "pr-review-worker-deep",
        "--prompt",
        "routing probe",
        "--dry-run",
        _cwd=REPO_ROOT,
        _env={"PATH": _path_without_opencode(tmp_path), "HOME": os.environ["HOME"]},
    )

    resolved = json.loads(str(result))
    assert resolved["command"][resolved["command"].index("-m") + 1] == "opencode-go/kimi-k2.7-code"
    assert resolved["prompt"].endswith("routing probe")


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_opencode_shell_launcher_reads_prompt_file(tmp_path: Path) -> None:
    """Exercise the production prompt-file path through the shell launcher.

    :param tmp_path: Temporary directory for the worker prompt file.
    """
    sh = importlib.import_module("sh")
    prompt_file = tmp_path / "worker-prompt.txt"
    prompt_file.write_text("prompt-file routing probe")

    result = sh.Command(str(_OPENCODE_LAUNCHER_SH))(
        "pr-review-worker-fast",
        "--prompt-file",
        prompt_file,
        "--dry-run",
        _cwd=REPO_ROOT,
    )

    resolved = json.loads(str(result))
    assert resolved["command"][resolved["command"].index("-m") + 1] == "opencode-go/glm-5.2"
    assert resolved["prompt"].endswith("prompt-file routing probe")


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_opencode_shell_launcher_extracts_last_message_final_text(tmp_path: Path) -> None:
    """Return only the final state of the last message's text parts.

    :param tmp_path: Temporary directory containing the fake opencode executable.
    """
    sh = importlib.import_module("sh")
    # Final-message part ids sort lexically ("prt_10" < "prt_2") in the reverse
    # of their emission order, pinning that extraction preserves stream order.
    events = [
        {"type": "step_start", "part": {"type": "step-start", "messageID": "msg_1"}},
        {
            "type": "text",
            "part": {"id": "prt_1", "messageID": "msg_1", "type": "text", "text": "earlier draft"},
        },
        {
            "type": "text",
            "part": {"id": "prt_2", "messageID": "msg_2", "type": "text", "text": "## first"},
        },
        {
            "type": "text",
            "part": {"id": "prt_10", "messageID": "msg_2", "type": "text", "text": "partial"},
        },
        {
            "type": "text",
            "part": {"id": "prt_10", "messageID": "msg_2", "type": "text", "text": "## second"},
        },
        {"type": "step_finish", "part": {"type": "step-finish", "messageID": "msg_2"}},
    ]
    fake = tmp_path / "opencode"
    lines = "\n".join(json.dumps(event) for event in events)
    fake.write_text(f"#!/bin/bash\ncat <<'EOF'\n{lines}\nEOF\n")
    fake.chmod(0o755)

    result = sh.Command(str(_OPENCODE_LAUNCHER_SH))(
        "pr-review-worker-fast",
        "--prompt",
        "routing probe",
        _cwd=REPO_ROOT,
        _env={"PATH": f"{tmp_path}:{os.environ['PATH']}"},
    )

    assert str(result) == "## first\n\n## second"


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_opencode_shell_launcher_withholds_caller_stdin(tmp_path: Path) -> None:
    """Keep the caller's stdin out of the opencode run.

    :param tmp_path: Temporary directory containing the fake opencode executable.
    """
    sh = importlib.import_module("sh")
    fake = tmp_path / "opencode"
    fake.write_text(
        "#!/bin/bash\n"
        "leaked=$(cat)\n"
        'jq -cn --arg text "stdin=[${leaked}]" \\\n'
        '  \'{type: "text", part: {id: "prt_1", messageID: "msg_1",'
        ' type: "text", text: $text}}\'\n'
    )
    fake.chmod(0o755)

    result = sh.Command(str(_OPENCODE_LAUNCHER_SH))(
        "pr-review-worker-fast",
        "--prompt",
        "routing probe",
        _cwd=REPO_ROOT,
        _env={"PATH": f"{tmp_path}:{os.environ['PATH']}"},
        _in="caller-owned stdin payload",
    )

    assert str(result) == "stdin=[]"


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_opencode_shell_launcher_missing_binary_exits_nonzero(tmp_path: Path) -> None:
    """Signal degrade-and-note to the caller when no opencode CLI is installed.

    :param tmp_path: Temporary directory for the restricted PATH.
    """
    sh = importlib.import_module("sh")

    with pytest.raises(sh.ErrorReturnCode) as exc_info:
        sh.Command(str(_OPENCODE_LAUNCHER_SH))(
            "pr-review-worker-fast",
            "--prompt",
            "routing probe",
            _cwd=REPO_ROOT,
            _env={"PATH": _path_without_opencode(tmp_path), "HOME": os.environ["HOME"]},
        )

    assert exc_info.value.exit_code == 3
    assert b"opencode" in exc_info.value.stderr


@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
def test_opencode_shell_launcher_timeout_kills_hung_run(tmp_path: Path) -> None:
    """Bound a hung opencode run instead of stalling the review worker.

    :param tmp_path: Temporary directory containing the fake opencode executable.
    """
    sh = importlib.import_module("sh")
    fake = tmp_path / "opencode"
    fake.write_text("#!/bin/bash\nsleep 30\n")
    fake.chmod(0o755)

    with pytest.raises(sh.ErrorReturnCode) as exc_info:
        sh.Command(str(_OPENCODE_LAUNCHER_SH))(
            "pr-review-worker-fast",
            "--prompt",
            "routing probe",
            _cwd=REPO_ROOT,
            _env={
                "PATH": f"{tmp_path}:{os.environ['PATH']}",
                "OPENCODE_REVIEW_TIMEOUT": "1",
            },
        )

    assert exc_info.value.exit_code != 0
    assert b"timed out" in exc_info.value.stderr


@pytest.mark.slow
@pytest.mark.skipif(not _SH_AVAILABLE, reason="requires the sh package")
@pytest.mark.skipif(
    shutil.which("opencode") is None,
    reason=(
        "needs the opencode CLI + auth; run manually: "
        "agent/_shared/run_opencode_review_agent.sh pr-review-worker-fast "
        "--prompt 'Reply with exactly the word OK and nothing else.'"
    ),
)
def test_opencode_shell_launcher_real_cli_returns_model_text() -> None:
    """Drive the real opencode CLI once to prove auth, event parsing, and exit flow."""
    sh = importlib.import_module("sh")

    result = sh.Command(str(_OPENCODE_LAUNCHER_SH))(
        "pr-review-worker-fast",
        "--prompt",
        "Reply with exactly the word OK and nothing else.",
        _cwd=REPO_ROOT,
    )

    assert "OK" in str(result)


def _prepare_review_host_l1_project(tmp_path: Path) -> None:
    """Install the real host contracts beside a delayed fake Pi launcher.

    :param tmp_path: Temporary project to populate.
    """
    launcher = tmp_path / "agent" / "_shared" / "run_pi_review.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" >> .host-invocations\n"
        "[[ \"$*\" == 'repo-review-full-no-comments --target 2125' ]]\n"
        "sleep 2\n"
        "printf 'PI_HOST_L1_COMPLETE\\n'\n"
    )
    launcher.chmod(0o755)
    for relative_path in (
        Path("agent/_shared/pi-review-host-contract.md"),
        Path(".claude/agents/pr-review-orchestrator.md"),
        Path(".codex/agents/pr-review-orchestrator.toml"),
    ):
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative_path, destination)


def _run_review_host_l1(host: str, tmp_path: Path) -> str:
    """Run one real host with its committed orchestrator definition.

    :param host: Authenticated CLI host under test.
    :param tmp_path: Prepared temporary project.
    :returns: Host CLI output.
    """
    sh = importlib.import_module("sh")
    prompt = (
        "Run repo-review-full-no-comments for PR 2125. Follow the configured "
        "pr-review-orchestrator exactly and return only its deliverable."
    )
    command = sh.Command(shutil.which(host))
    if host == "claude":
        return str(
            command(
                "-p",
                "--dangerously-skip-permissions",
                "--no-session-persistence",
                "--agent",
                "pr-review-orchestrator",
                "--output-format",
                "text",
                prompt,
                _cwd=tmp_path,
                _timeout=180,
            )
        )

    config = tomllib.loads((tmp_path / ".codex/agents/pr-review-orchestrator.toml").read_text())
    return str(
        command(
            "exec",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "--model",
            config["model"],
            "--config",
            f"developer_instructions={json.dumps(config['developer_instructions'])}",
            prompt,
            _cwd=tmp_path,
            _timeout=180,
        )
    )


@pytest.mark.slow
@pytest.mark.parametrize("host", ("claude", "codex"))
def test_review_host_real_cli_waits_for_delayed_pi(
    host: str,
    tmp_path: Path,
) -> None:
    """Prove each configured host returns after exactly one delayed Pi command.

    :param host: Authenticated CLI host under test.
    :param tmp_path: Temporary project containing the delayed fake Pi launcher.
    """
    if os.environ.get("RUN_REVIEW_HOST_L1") != "1":
        pytest.skip("set RUN_REVIEW_HOST_L1=1 for the authenticated L1 host eval")
    if shutil.which(host) is None:
        pytest.skip(f"{host} CLI is unavailable")

    _prepare_review_host_l1_project(tmp_path)
    started_at = time.monotonic()
    result = _run_review_host_l1(host, tmp_path)

    assert time.monotonic() - started_at >= 2
    assert "PI_HOST_L1_COMPLETE" in result
    assert (tmp_path / ".host-invocations").read_text().splitlines() == [
        "repo-review-full-no-comments --target 2125"
    ]


def test_opencode_config_reviewer_agent_denies_mutations() -> None:
    """Keep the shared opencode review agent read-only."""
    config = json.loads((REPO_ROOT / "opencode.json").read_text())

    reviewer = config["agent"]["pr-reviewer"]
    assert reviewer["description"]
    assert reviewer["permission"]["edit"] == "deny"
    assert reviewer["permission"]["task"] == "deny"
    assert reviewer["permission"]["bash"]["*"] == "deny"
    assert reviewer["permission"]["bash"]["git diff*"] == "allow"


def test_review_fanout_analysis_uses_pi_instead_of_legacy_launchers() -> None:
    """Keep the active fan-out independent of host-specific worker launchers."""
    text = (
        REPO_ROOT / "agent" / "skills" / "_shared" / "repo-review-full-analysis.md"
    ).read_text()

    assert "run_pi_review.sh" in text
    assert "run_opencode_review_agent.sh" not in text
    assert "run_codex_review_agent.sh" not in text


def test_worker_agent_briefs_permit_only_opencode_launcher() -> None:
    """Give workers the launcher carve-out without opening the orchestrator to it."""
    for provider_dir, suffix in ((".claude", ".md"), (".codex", ".toml")):
        for role in ("pr-review-worker-deep", "pr-review-worker-fast"):
            text = (REPO_ROOT / provider_dir / "agents" / f"{role}{suffix}").read_text()
            assert "run_opencode_review_agent.sh" in text
        orchestrator = (
            REPO_ROOT / provider_dir / "agents" / f"pr-review-orchestrator{suffix}"
        ).read_text()
        assert "run_opencode_review_agent.sh" not in orchestrator
