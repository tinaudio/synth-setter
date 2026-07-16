"""Contract tests for PR-review model routing across Claude, Codex, and OpenCode."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import re
import runpy
import shutil
import sys
import tomllib
from pathlib import Path
from unittest import mock

import pytest
import yaml

from tests.helpers.package_available import _SH_AVAILABLE

REPO_ROOT = Path(__file__).resolve().parents[2]

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
            assert "never invoke the top-level review skill" in codex["developer_instructions"]
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


def test_full_review_skills_require_the_pinned_orchestrator() -> None:
    """Prevent review gates from bypassing project-scoped agent configuration."""
    for skill in ("repo-review-full", "repo-review-full-no-comments"):
        text = (REPO_ROOT / "agent" / "skills" / skill / "SKILL.md").read_text()
        assert "`pr-review-orchestrator`" in text
        assert "CLAUDE_CODE_SUBAGENT_MODEL" in text
        assert "run_codex_review_agent.sh" in text
        assert "general-purpose" not in text


def test_review_fanout_promotes_only_correctness() -> None:
    """Keep expensive reasoning reserved for the correctness pass."""
    text = (
        REPO_ROOT / "agent" / "skills" / "_shared" / "repo-review-full-analysis.md"
    ).read_text()

    assert re.search(r"`correctness-review` uses\s+`pr-review-worker-deep`", text)
    assert re.search(r"all other selected skills use\s+`pr-review-worker-fast`", text)
    assert "general-purpose" not in text
    assert "do not fall back" in text.lower()


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

    result = sh.Command(str(launcher))(
        "pr-review-worker-fast",
        "--prompt",
        "routing probe",
        _cwd=REPO_ROOT,
        _env={"PATH": f"{tmp_path}:{os.environ['PATH']}"},
    )

    assert str(result) == "structured report"


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
def test_codex_review_shell_launcher_invalid_timeout_rejected_before_launch(
    tmp_path: Path,
) -> None:
    """Ensure invalid deadlines prevent subprocess startup.

    :param tmp_path: Directory for the fake Codex executable.
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
                "CODEX_REVIEW_TIMEOUT": "-1",
            },
        )

    assert b"CODEX_REVIEW_TIMEOUT must be a positive integer" in exc_info.value.stderr
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
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


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


def test_opencode_config_reviewer_agent_denies_mutations() -> None:
    """Keep the shared opencode review agent read-only."""
    config = json.loads((REPO_ROOT / "opencode.json").read_text())

    reviewer = config["agent"]["pr-reviewer"]
    assert reviewer["description"]
    assert reviewer["permission"]["edit"] == "deny"
    assert reviewer["permission"]["task"] == "deny"
    assert reviewer["permission"]["bash"]["*"] == "deny"
    assert reviewer["permission"]["bash"]["git diff*"] == "allow"


def test_review_fanout_analysis_requires_opencode_pass_and_degrade() -> None:
    """Pin the cross-model pass and its degrade contract in the fan-out spec."""
    text = (
        REPO_ROOT / "agent" / "skills" / "_shared" / "repo-review-full-analysis.md"
    ).read_text()

    assert "run_opencode_review_agent.sh" in text
    assert "flagged by:" in text
    assert "opencode pass skipped/failed" in text


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
