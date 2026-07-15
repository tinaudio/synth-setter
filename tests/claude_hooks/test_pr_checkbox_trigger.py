"""Regression tests for the PR-checkbox-trigger PostToolUse hook command.

The hook detects ``gh pr create`` commands and prompts the agent to invoke
the ``/pr-checkbox`` skill. The original inline body matched the phrase with a
plain ``grep -q 'gh pr create'`` on the whole command, so any Bash command
whose quoted *arguments* contained the phrase (a ``gh issue create`` body
quoting the recipe, an ``echo`` printing it) fired the reminder even though no
PR was created. The hook now routes through
``agent/_shared/pr_command_classifier.py`` so only real ``gh pr create``
invocations classify as PR-creation.

Regression: [#1942](https://github.com/tinaudio/synth-setter/issues/1942).
"""

from __future__ import annotations

import json

from tests.claude_hooks.test_settings_hooks import _find_handler, _run_hook_command

_REMINDER_SUBSTRING = "You MUST now invoke the /pr-checkbox skill"


def _hook_command() -> str:
    """Return the shell ``command`` body of the PR-checkbox-trigger hook.

    :returns: The ``command`` string declared in ``.claude/settings.json`` for the
        ``PR checkbox trigger`` matcher entry.
    """
    return _find_handler("PR checkbox trigger")["command"]


def _parse_reminder(stdout: str) -> str:
    """Return the ``additionalContext`` payload the hook wrote on stdout, or "".

    :param stdout: Raw hook stdout (empty when the reminder did not fire).
    :returns: The additionalContext string, or "" when stdout was empty.
    """
    if not stdout.strip():
        return ""
    payload = json.loads(stdout)
    return payload.get("hookSpecificOutput", {}).get("additionalContext", "")


def test_pr_checkbox_trigger_fires_on_direct_pr_create() -> None:
    """A direct ``gh pr create`` invocation emits the ``/pr-checkbox`` reminder.

    Pins the kept-behavior half of the fix: a real PR creation still triggers
    the prompt.
    """
    result = _run_hook_command(
        _hook_command(),
        {"tool_input": {"command": "gh pr create --title 'fix: x' --body 'body'"}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert _REMINDER_SUBSTRING in _parse_reminder(result.stdout)


def test_pr_checkbox_trigger_does_not_fire_on_issue_create_body_quoting_recipe() -> None:
    """A ``gh issue create`` whose ``--body`` argument quotes the recipe does NOT fire.

    Reproduces [#1942](https://github.com/tinaudio/synth-setter/issues/1942):
    filing an issue whose body mentions ``gh pr create`` triggered the
    spurious reminder under the old substring grep.
    """
    result = _run_hook_command(
        _hook_command(),
        {
            "tool_input": {
                "command": (
                    "gh issue create --title 'bug' "
                    "--body 'observed: run `gh pr create` to file the fix'"
                ),
            },
        },
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert _parse_reminder(result.stdout) == ""


def test_pr_checkbox_trigger_does_not_fire_on_echo_quoting_recipe() -> None:
    """An ``echo`` quoting the recipe does NOT fire the reminder."""
    result = _run_hook_command(
        _hook_command(),
        {"tool_input": {"command": "echo 'gh pr create --title x --body y'"}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert _parse_reminder(result.stdout) == ""


def test_pr_checkbox_trigger_does_not_fire_on_unrelated_command() -> None:
    """An unrelated Bash command (no ``gh pr create`` invocation) does NOT fire."""
    result = _run_hook_command(
        _hook_command(),
        {"tool_input": {"command": "git status"}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert _parse_reminder(result.stdout) == ""


def test_pr_checkbox_trigger_fires_on_classifier_wrapped_create() -> None:
    """A shell-wrapped ``gh pr create`` (classifier ``wrapped``) still fires.

    The classifier returns ``wrapped`` for ``bash -c 'gh pr create ...'``; the
    reminder fires on every non-empty classifier mode (``direct`` /
    ``wrapped`` / ``unparsable``), matching the original substring grep's
    coverage of mentioning commands while excluding quoted prose.
    """
    result = _run_hook_command(
        _hook_command(),
        {"tool_input": {"command": "bash -c 'gh pr create --title x --body y'"}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert _REMINDER_SUBSTRING in _parse_reminder(result.stdout)