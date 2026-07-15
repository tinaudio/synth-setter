"""Contract tests for PR-review model routing across Claude and Codex."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import re
import runpy
import sys
import tomllib
from pathlib import Path

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
    },
    "pr-review-worker-fast": {
        "claude": ("sonnet", "medium"),
        "codex": ("gpt-5.6-terra", "medium"),
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
