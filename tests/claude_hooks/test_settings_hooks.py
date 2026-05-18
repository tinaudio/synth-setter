"""Tests that ``.claude/settings.json`` hooks are scoped to the intended commands.

Regression coverage for the bug where ``if:`` was placed on the matcher-entry
object (sibling of ``matcher``/``hooks``/``description``). Claude Code silently
ignores ``if:`` at that level, so a hook intended to gate only ``gh pr create``
fired on every Bash tool call — blocking even ``gh pr view`` and ``ls``.

The fix moves ``if:`` inside each hook handler (sibling of ``type`` and
``command``) and uses Claude Code permission-rule syntax such as ``Bash(gh pr
create *)``. Schema reference: https://code.claude.com/docs/en/hooks.md.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_PATH = _REPO_ROOT / ".claude" / "settings.json"


def _load_settings() -> dict[str, Any]:
    """Load ``.claude/settings.json``.

    :returns: The parsed settings dict.
    :rtype: dict[str, Any]
    """
    return json.loads(_SETTINGS_PATH.read_text())


def _matcher_entries() -> list[dict[str, Any]]:
    """Flatten every PreToolUse and PostToolUse matcher-entry object into a single list.

    :returns: The matcher-entry objects from all hook events, in source order.
    :rtype: list[dict[str, Any]]
    """
    return [
        entry
        for event_entries in _load_settings().get("hooks", {}).values()
        for entry in event_entries
    ]


def _find_handler(description_substring: str) -> dict[str, Any]:
    """Look up the lone hook handler whose matcher-entry description contains a substring.

    :param description_substring: Substring identifying the matcher-entry's ``description`` field.
    :returns: The single handler dict from the matcher-entry's ``hooks`` list.
    :rtype: dict[str, Any]
    :raises AssertionError: If zero or >1 matcher entries match, or the matched entry has !=1 handler.
    """
    matches = [
        entry
        for entry in _matcher_entries()
        if description_substring in entry.get("description", "")
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one matcher entry with description containing "
            f"{description_substring!r}, found {len(matches)}"
        )
    handlers = matches[0].get("hooks", [])
    if len(handlers) != 1:
        raise AssertionError(
            f"expected exactly one handler under matcher entry "
            f"matching {description_substring!r}, got {len(handlers)}"
        )
    return handlers[0]


def _run_hook_command(
    command_body: str, stdin_payload: dict[str, Any]
) -> subprocess.CompletedProcess[str]:
    """Run a hook ``command`` body the same way Claude Code does.

    Claude Code invokes the body via the shell and pipes the tool-call JSON to stdin. Tests
    reproduce that contract. The body is sometimes an inline shell snippet and sometimes a
    one-line ``bash …/some-hook.sh`` wrapper — either form runs through this helper.

    :param command_body: The shell command body from a hook handler.
    :param stdin_payload: JSON payload sent to the hook on stdin.
    :returns: The completed subprocess result, with captured stdout/stderr and exit code.
    :rtype: subprocess.CompletedProcess[str]
    """
    # Trust boundary: command_body is read from the repo's own checked-in settings.json,
    # which is maintainer-controlled. Do not reuse this helper against untrusted paths.
    return subprocess.run(  # noqa: S603
        ["bash", "-c", command_body],  # noqa: S607 — bash is a hard requirement of the harness
        input=json.dumps(stdin_payload),
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# Schema regression — the bug this PR fixes
# ---------------------------------------------------------------------------


def test_no_if_at_matcher_entry_level() -> None:
    """Matcher-entry objects must not carry a top-level ``if:`` field.

    Claude Code's hooks schema places ``if:`` inside each hook handler, not on
    the matcher entry. A misplaced ``if:`` is silently ignored, so a Bash hook
    that intended to scope itself to one command runs on every Bash tool call.
    """
    offenders = [
        {"matcher": entry.get("matcher"), "description": entry.get("description", "")[:60]}
        for entry in _matcher_entries()
        if "if" in entry
    ]
    assert offenders == [], (
        "Matcher-entry-level `if:` fields are silently dropped by Claude Code. "
        f"Move each `if:` inside its hook handler (sibling of `type`/`command`). "
        f"Offenders: {offenders}"
    )


_PERMISSION_RULE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*\(.+\)$")


def test_handler_if_values_use_permission_rule_syntax() -> None:
    """Hook-handler ``if:`` values must look like ``Tool(pattern)`` permission rules.

    Shell expressions such as ``jq … | grep -qE …`` were the previous (broken)
    style of "scoping". The handler-level ``if:`` is parsed as a permission rule
    (e.g. ``Bash(gh pr create *)``), not as a shell command — only ``Tool(...)``
    forms are honored.
    """
    bad: list[tuple[str, Any]] = []
    for entry in _matcher_entries():
        for handler in entry.get("hooks", []):
            value = handler.get("if")
            if value is None:
                continue
            if not (isinstance(value, str) and _PERMISSION_RULE_RE.match(value)):
                bad.append((entry.get("description", "")[:60], value))
    assert bad == [], (
        "Hook-handler `if:` values must be Claude Code permission-rule syntax such as "
        f"`Bash(gh pr create *)` — not shell expressions. Offenders: {bad}"
    )


# ---------------------------------------------------------------------------
# Per-handler scope assertions — the specific gates that drove this bug
# ---------------------------------------------------------------------------


_EXPECTED_HANDLER_SCOPES: list[tuple[str, str]] = [
    ("Branch safety", "Bash(git commit *)"),
    ("Pre-PR review gate", "Bash(gh pr create *)"),
    ("Doc-drift advisory review", "Bash(gh pr create *)"),
    ("PR review resolver", "Bash(git push *)"),
]


@pytest.mark.parametrize(("description_substring", "expected_if"), _EXPECTED_HANDLER_SCOPES)
def test_named_handlers_carry_expected_if_scope(
    description_substring: str, expected_if: str
) -> None:
    """Every Bash hook that needs command-specific scoping declares the right ``if:`` rule.

    :param description_substring: Substring identifying the matcher-entry's description.
    :param expected_if: The permission-rule value the inner hook handler must declare.
    """
    handler = _find_handler(description_substring)
    assert handler.get("if") == expected_if, (
        f"handler under {description_substring!r} expected `if: {expected_if}`, "
        f"got {handler.get('if')!r} (missing or wrong scope means this hook fires on "
        "every Bash call)"
    )


# ---------------------------------------------------------------------------
# Behavioural tests for the pre-PR review gate body
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pre_pr_gate_command() -> str:
    """Yield the shell ``command`` body of the gh-pr-create PreToolUse gate.

    Currently a one-line wrapper that invokes ``.claude/hooks/pre-pr-review-gate.sh``;
    the helper runs it via ``bash -c`` so the wrapper re-enters the script transparently.

    :returns: The shell command string from the gate handler.
    :rtype: str
    """
    return _find_handler("Pre-PR review gate")["command"]


def test_pre_pr_gate_blocks_when_token_absent(pre_pr_gate_command: str) -> None:
    """Gate exits 2 with ``BLOCKED`` in stderr when ``REVIEW_FULL_DONE=1`` is missing.

    :param pre_pr_gate_command: The shell command body from the pre-PR review gate handler.
    """
    result = _run_hook_command(
        pre_pr_gate_command,
        {"tool_input": {"command": "gh pr create --title foo --body bar"}},
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


def test_pre_pr_gate_allows_when_token_in_trailing_comment(pre_pr_gate_command: str) -> None:
    """Gate exits 0 when ``REVIEW_FULL_DONE=1`` is present (typically as a trailing comment).

    :param pre_pr_gate_command: The shell command body from the pre-PR review gate handler.
    """
    result = _run_hook_command(
        pre_pr_gate_command,
        {"tool_input": {"command": "gh pr create --title foo --body bar  # REVIEW_FULL_DONE=1"}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert "BLOCKED" not in result.stderr


# ---------------------------------------------------------------------------
# Behavioural test for the branch-print hook
# ---------------------------------------------------------------------------


def test_branch_print_hook_announces_current_branch() -> None:
    """Branch-print hook writes ``Committing to branch: <name>`` to stderr without failing.

    Stdin is irrelevant to this hook (it shells out to ``git branch
    --show-current``), but the test still pipes a representative payload to
    mirror how Claude Code invokes it.
    """
    handler = _find_handler("Branch safety")
    result = _run_hook_command(
        handler["command"],
        {"tool_input": {"command": "git commit -m test"}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert "Committing to branch:" in result.stderr
