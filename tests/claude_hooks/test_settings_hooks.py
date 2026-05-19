"""Tests pinning Claude hook ``settings.json`` scoping and runtime behaviour.

Invariant: each Bash hook handler that scopes itself uses Claude Code
permission-rule ``if:`` syntax (e.g. ``Bash(gh pr create *)``) at the handler
level, and the body of each hook honours its documented exit contract.
Schema reference: https://code.claude.com/docs/en/hooks.md.
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
    """Return parsed ``.claude/settings.json``.

    :returns: Settings dict.
    """
    return json.loads(_SETTINGS_PATH.read_text())


def _matcher_entries() -> list[dict[str, Any]]:
    """Return every PreToolUse/PostToolUse matcher-entry in source order.

    :returns: Flat list of matcher-entry objects.
    """
    return [
        entry
        for event_entries in _load_settings().get("hooks", {}).values()
        for entry in event_entries
    ]


def _find_handler(description_substring: str) -> dict[str, Any]:
    """Return the lone hook handler whose matcher-entry description contains a substring.

    :param description_substring: Substring identifying the matcher-entry's ``description`` field.
    :returns: The single handler dict under the matched entry.
    :raises AssertionError: If zero or >1 matcher entries match, or the entry has !=1 handler.
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
    """Run a hook ``command`` body the way Claude Code does (shell + JSON on stdin).

    :param command_body: The shell command body from a hook handler.
    :param stdin_payload: JSON payload sent to the hook on stdin.
    :returns: Completed subprocess result with captured stdout/stderr and exit code.
    """
    # Trust boundary: command_body comes from the repo's checked-in settings.json
    # (maintainer-controlled). Do not reuse this helper against untrusted paths.
    return subprocess.run(  # noqa: S603
        ["bash", "-c", command_body],  # noqa: S607 — bash is a hard requirement of the harness
        input=json.dumps(stdin_payload),
        capture_output=True,
        text=True,
        check=False,
    )


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


_EXPECTED_HANDLER_SCOPES: tuple[tuple[str, str], ...] = (
    ("Branch safety", "Bash(git commit *)"),
    ("Git-commit-trailer-check", "Bash(git commit *)"),
    ("Pre-PR review gate", "Bash(gh pr create *)"),
    ("Doc-drift advisory review", "Bash(gh pr create *)"),
    ("PR review resolver", "Bash(git push *)"),
)

_EXPECTED_SHARED_HOOK_COMMANDS: tuple[tuple[str, str], ...] = (
    ("Credential protection", "bash agent/hooks/edit-write.sh credential-protect"),
    ("No-baseline-additions", "bash agent/hooks/no-baseline-additions.sh"),
    ("No-yaml-run-comments", "bash agent/hooks/no-yaml-run-comments.sh"),
    ("Git-commit-trailer-check", "bash agent/hooks/git-commit-trailer-check.sh"),
    ("Auto-format", "bash agent/hooks/edit-write.sh format"),
    ("Auto-test", "bash agent/hooks/edit-write.sh test"),
    ("Taxonomy verification", "bash agent/hooks/verify-gh-taxonomy.sh"),
    ("Doc-drift advisory review", "bash agent/hooks/doc-drift.sh"),
    ("PR review resolver", "bash agent/hooks/pr-review-resolver.sh"),
)


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


@pytest.mark.parametrize(
    ("description_substring", "expected_command"), _EXPECTED_SHARED_HOOK_COMMANDS
)
def test_named_handlers_use_shared_agent_hook_paths(
    description_substring: str, expected_command: str
) -> None:
    """Claude settings invoke shared agent hook implementations where available.

    :param description_substring: Substring identifying the matcher-entry's description.
    :param expected_command: The shared hook command the handler must execute.
    """
    handler = _find_handler(description_substring)
    assert handler.get("command") == expected_command


def test_credential_guard_uses_tool_input_file_path_not_embedded_text() -> None:
    """Credential guard keys off ``.tool_input.file_path`` only.

    This covers quoted ``file_path`` text inside edit payload content, which broke
    the old grep/head/sed extraction.
    """
    command = _find_handler("Credential protection")["command"]
    payload = {
        "tool_input": {
            "file_path": "src/example.py",
            "old_string": '"file_path": ".env"',
            "new_string": '"file_path": "secrets.pem"',
        }
    }

    result = _run_hook_command(command, payload)

    assert result.returncode == 0, result.stderr


def test_credential_guard_blocks_secret_file_path() -> None:
    """Credential guard still blocks the actual ``.tool_input.file_path`` target."""
    command = _find_handler("Credential protection")["command"]
    result = _run_hook_command(command, {"tool_input": {"file_path": ".env.local"}})

    assert result.returncode == 1
    assert "BLOCKED" in result.stderr


@pytest.fixture(scope="module")
def pre_pr_gate_command() -> str:
    """Return the shell ``command`` body of the gh-pr-create PreToolUse gate.

    :returns: The shell command string.
    """
    return _find_handler("Pre-PR review gate")["command"]


def test_pre_pr_gate_blocks_when_token_absent(pre_pr_gate_command: str) -> None:
    """Gate exits 2 with ``BLOCKED`` in stderr when ``REVIEW_FULL_DONE=1`` is missing.

    :param pre_pr_gate_command: Hook command body fixture.
    """
    result = _run_hook_command(
        pre_pr_gate_command,
        {"tool_input": {"command": "gh pr create --title foo --body bar"}},
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


def test_pre_pr_gate_allows_when_token_in_trailing_comment(pre_pr_gate_command: str) -> None:
    """Gate exits 0 when ``REVIEW_FULL_DONE=1`` is present (typically as a trailing comment).

    :param pre_pr_gate_command: Hook command body fixture.
    """
    result = _run_hook_command(
        pre_pr_gate_command,
        {"tool_input": {"command": "gh pr create --title foo --body bar  # REVIEW_FULL_DONE=1"}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert "BLOCKED" not in result.stderr


def test_branch_print_hook_announces_current_branch() -> None:
    """Branch-print hook writes ``Committing to branch: <name>`` to stderr without failing."""
    handler = _find_handler("Branch safety")
    result = _run_hook_command(
        handler["command"],
        {"tool_input": {"command": "git commit -m test"}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert "Committing to branch:" in result.stderr


@pytest.fixture(scope="module")
def baseline_hook_command() -> str:
    """Return the shell ``command`` body of the no-baseline-additions hook.

    :returns: The shell command string.
    """
    return _find_handler("No-baseline-additions")["command"]


def test_baseline_hook_passes_unrelated_file(baseline_hook_command: str) -> None:
    """The hook fast-paths Edit/Write on files that are not the pydoclint baseline.

    :param baseline_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        baseline_hook_command,
        {"tool_input": {"file_path": "src/synth_setter/whatever.py", "new_string": "x"}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_baseline_hook_blocks_addition(baseline_hook_command: str, tmp_path: Path) -> None:
    """An Edit that increases baseline row count is blocked with exit 2.

    :param baseline_hook_command: Hook command body fixture.
    :param tmp_path: pytest tmp_path.
    """
    baseline = tmp_path / ".pydoclint-baseline.txt"
    baseline.write_text("path/a.py:1:1: DOC101\n")
    result = _run_hook_command(
        baseline_hook_command,
        {
            "tool_name": "Edit",
            "tool_input": {
                "file_path": str(baseline),
                "old_string": "path/a.py:1:1: DOC101",
                "new_string": "path/a.py:1:1: DOC101\npath/b.py:2:2: DOC102",
            },
        },
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr
    assert "append-frozen" in result.stderr


def test_baseline_hook_allows_removal(baseline_hook_command: str, tmp_path: Path) -> None:
    """An Edit that removes a row (graduation) is allowed.

    :param baseline_hook_command: Hook command body fixture.
    :param tmp_path: pytest tmp_path.
    """
    baseline = tmp_path / ".pydoclint-baseline.txt"
    baseline.write_text("path/a.py:1:1: DOC101\npath/b.py:2:2: DOC102\n")
    result = _run_hook_command(
        baseline_hook_command,
        {
            "tool_name": "Edit",
            "tool_input": {
                "file_path": str(baseline),
                "old_string": "path/a.py:1:1: DOC101\n",
                "new_string": "",
            },
        },
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


@pytest.fixture(scope="module")
def yaml_run_hook_command() -> str:
    """Return the shell ``command`` body of the no-yaml-run-comments hook.

    :returns: The shell command string.
    """
    return _find_handler("No-yaml-run-comments")["command"]


def test_yaml_run_hook_passes_unrelated_file(yaml_run_hook_command: str) -> None:
    """Edits to files outside workflows/ and configs/compute/ fast-path through.

    :param yaml_run_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        yaml_run_hook_command,
        {"tool_input": {"file_path": "src/synth_setter/x.py", "content": "# comment\nx = 1\n"}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_yaml_run_hook_blocks_comment_inside_run_block(yaml_run_hook_command: str) -> None:
    """A `#`-comment inside a ``run: |`` block is blocked.

    :param yaml_run_hook_command: Hook command body fixture.
    """
    content = (
        "name: test\n"
        "on: push\n"
        "jobs:\n"
        "  test:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - name: bad\n"
        "        run: |\n"
        "          # this comment is inside the run block — BAD\n"
        "          echo hi\n"
    )
    result = _run_hook_command(
        yaml_run_hook_command,
        {
            "tool_name": "Write",
            "tool_input": {"file_path": ".github/workflows/test.yml", "content": content},
        },
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr
    assert "block scalar" in result.stderr


def test_yaml_run_hook_allows_comment_above_step(yaml_run_hook_command: str) -> None:
    """A `#`-comment above the step (outside the block scalar) is allowed.

    :param yaml_run_hook_command: Hook command body fixture.
    """
    content = (
        "name: test\n"
        "on: push\n"
        "jobs:\n"
        "  test:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      # this comment is correctly above the step\n"
        "      - name: good\n"
        "        run: |\n"
        "          echo hi\n"
    )
    result = _run_hook_command(
        yaml_run_hook_command,
        {
            "tool_name": "Write",
            "tool_input": {"file_path": ".github/workflows/test.yml", "content": content},
        },
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_yaml_run_hook_allows_shebang_inside_run_block(yaml_run_hook_command: str) -> None:
    """A shebang line (``#!``) inside the block scalar is allowed.

    :param yaml_run_hook_command: Hook command body fixture.
    """
    content = (
        "jobs:\n"
        "  test:\n"
        "    steps:\n"
        "      - run: |\n"
        "          #!/usr/bin/env bash\n"
        "          echo hi\n"
    )
    result = _run_hook_command(
        yaml_run_hook_command,
        {
            "tool_name": "Write",
            "tool_input": {"file_path": ".github/workflows/test.yml", "content": content},
        },
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


@pytest.fixture(scope="module")
def trailer_hook_command() -> str:
    """Return the shell ``command`` body of the git-commit-trailer-check hook.

    :returns: The shell command string.
    """
    return _find_handler("Git-commit-trailer-check")["command"]


def test_trailer_hook_passes_non_git_commit(trailer_hook_command: str) -> None:
    """Commands that aren't ``git commit`` (e.g. ``ls``) fast-path through.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": "ls -la"}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_trailer_hook_passes_clean_commit(trailer_hook_command: str) -> None:
    """A clean conventional-commit message is allowed.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": 'git commit -m "feat(scope): clean message"'}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_trailer_hook_blocks_no_verify_long_form(trailer_hook_command: str) -> None:
    """``git commit --no-verify`` is blocked.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": 'git commit --no-verify -m "msg"'}},
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr
    assert "--no-verify" in result.stderr


def test_trailer_hook_blocks_no_verify_short_form(trailer_hook_command: str) -> None:
    """``git commit -n`` (short form of ``--no-verify``) is blocked.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": 'git commit -n -m "msg"'}},
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr
    assert "-n" in result.stderr


def test_trailer_hook_does_not_false_positive_on_chained_grep_n(
    trailer_hook_command: str,
) -> None:
    """A downstream ``grep -n`` after ``git commit && …`` is NOT mistaken for ``commit -n``.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": 'git commit -m "msg" && grep -n bar file.txt'}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_trailer_hook_blocks_co_authored_by_trailer(trailer_hook_command: str) -> None:
    """A ``Co-Authored-By:`` trailer in the ``-m`` body is blocked.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {
            "tool_input": {
                "command": (
                    'git commit -m "feat: x\n\nCo-Authored-By: Claude <noreply@anthropic.com>"'
                )
            }
        },
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr
    assert "Co-Authored-By" in result.stderr


def test_trailer_hook_blocks_generated_with_footer(trailer_hook_command: str) -> None:
    """A ``Generated with …`` agent-attribution footer is blocked.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": 'git commit -m "feat: x\n\nGenerated with Claude Code"'}},
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


def test_trailer_hook_blocks_bundled_short_flag(trailer_hook_command: str) -> None:
    """``git commit -nm "msg"`` (bundled ``-n``) is blocked the same as ``-n -m``.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": 'git commit -nm "msg"'}},
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr
    assert "-n" in result.stderr


def test_trailer_hook_blocks_anm_bundled_short_flag(trailer_hook_command: str) -> None:
    """``git commit -anm "msg"`` (``-a -n -m`` bundled) is blocked.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": 'git commit -anm "msg"'}},
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


def test_trailer_hook_allows_bundled_short_flag_without_n(trailer_hook_command: str) -> None:
    """``git commit -am "msg"`` (no ``n`` in the bundle) is allowed.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": 'git commit -am "msg"'}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_trailer_hook_allows_claude_model_name_in_subject(trailer_hook_command: str) -> None:
    """A subject naming a Claude model (no trailer context) is NOT flagged.

    Verifies the over-broad-regex fix: ``feat(eval): tokeniser for Claude Sonnet
    4.5`` is a legitimate subject. Only attribution-shaped trailers should fire.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": 'git commit -m "feat(eval): tokeniser for Claude Sonnet 4.5"'}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_trailer_hook_blocks_noreply_email(trailer_hook_command: str) -> None:
    """A ``noreply@anthropic.com`` email anywhere in the body is blocked.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": 'git commit -m "feat: x via noreply@anthropic.com"'}},
    )
    assert result.returncode == 2, (result.returncode, result.stderr)


def test_trailer_hook_blocks_dash_F_file_with_trailer(
    trailer_hook_command: str, tmp_path: Path
) -> None:
    """``git commit -F <file>`` whose file contains a forbidden trailer is blocked.

    :param trailer_hook_command: Hook command body fixture.
    :param tmp_path: pytest tmp_path.
    """
    msg_file = tmp_path / "msg.txt"
    msg_file.write_text("feat: x\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n")
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": f"git commit -F {msg_file}"}},
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


def test_trailer_hook_fails_closed_on_malformed_json(trailer_hook_command: str) -> None:
    """A malformed-JSON stdin payload is blocked, not silently let through.

    :param trailer_hook_command: Hook command body fixture.
    """
    # _run_hook_command always JSON-encodes its payload, so bypass it here and
    # send raw invalid JSON directly on stdin.
    result = subprocess.run(  # noqa: S603
        ["bash", "-c", trailer_hook_command],  # noqa: S607
        input="not valid json",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


@pytest.mark.parametrize("metachar", ["&&", "||", ";", "|", "&"])
def test_trailer_hook_does_not_false_positive_on_chained_grep_n_metachars(
    trailer_hook_command: str, metachar: str
) -> None:
    """Every shell metachar terminating ``git commit``'s argv slice is honored.

    Regression coverage: a future tokenizer that drops one of the five
    metachars from the slicing set would silently start mis-attributing a
    downstream ``grep -n`` as a ``commit -n`` violation.

    :param trailer_hook_command: Hook command body fixture.
    :param metachar: Shell metachar to chain after the commit.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": f'git commit -m "msg" {metachar} grep -n bar file.txt'}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


@pytest.mark.parametrize("metachar", ["&&", "||", ";", "|", "&"])
def test_trailer_hook_handles_no_whitespace_metachar_chain(
    trailer_hook_command: str, metachar: str
) -> None:
    """Regression: ``git commit ...METACHARgrep -n ...`` (no spaces) must not mis-attribute.

    Before c1e8120 / 54aeb2b the parser used plain ``shlex.split()`` which only
    splits metachars when whitespace-separated, so ``msg&&grep`` tokenized as a
    single token and ``-n`` from a downstream ``grep -n`` landed inside
    ``git commit``'s argv slice. The fix uses ``shlex.shlex(..., punctuation_chars=True)``;
    this test pins the no-whitespace case.

    :param trailer_hook_command: Hook command body fixture.
    :param metachar: Shell metachar (concatenated with no surrounding whitespace).
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": f'git commit -m "msg"{metachar}grep -n bar file.txt'}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_baseline_hook_blocks_write_addition(baseline_hook_command: str, tmp_path: Path) -> None:
    """A Write that increases baseline row count is blocked (Write path coverage).

    :param baseline_hook_command: Hook command body fixture.
    :param tmp_path: pytest tmp_path.
    """
    baseline = tmp_path / ".pydoclint-baseline.txt"
    baseline.write_text("path/a.py:1:1: DOC101\n")
    result = _run_hook_command(
        baseline_hook_command,
        {
            "tool_name": "Write",
            "tool_input": {
                "file_path": str(baseline),
                "content": "path/a.py:1:1: DOC101\npath/b.py:2:2: DOC102\n",
            },
        },
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


def test_baseline_hook_blocks_write_to_missing_file(
    baseline_hook_command: str, tmp_path: Path
) -> None:
    """A Write that creates a non-existent baseline with rows is blocked.

    Regression: previously the hook fast-pathed exit-0 when the file didn't
    exist, creating a delete+recreate bypass. The fix treats a missing file as
    ``OLD_COUNT=0`` and blocks any Write whose content carries baseline rows.

    :param baseline_hook_command: Hook command body fixture.
    :param tmp_path: pytest tmp_path.
    """
    baseline = tmp_path / ".pydoclint-baseline.txt"
    assert not baseline.exists()
    result = _run_hook_command(
        baseline_hook_command,
        {
            "tool_name": "Write",
            "tool_input": {
                "file_path": str(baseline),
                "content": "path/a.py:1:1: DOC101\n",
            },
        },
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


def test_baseline_hook_allows_write_truncation(baseline_hook_command: str, tmp_path: Path) -> None:
    """A Write that truncates the baseline (graduation) is allowed.

    :param baseline_hook_command: Hook command body fixture.
    :param tmp_path: pytest tmp_path.
    """
    baseline = tmp_path / ".pydoclint-baseline.txt"
    baseline.write_text("path/a.py:1:1: DOC101\npath/b.py:2:2: DOC102\n")
    result = _run_hook_command(
        baseline_hook_command,
        {
            "tool_name": "Write",
            "tool_input": {
                "file_path": str(baseline),
                "content": "path/a.py:1:1: DOC101\n",
            },
        },
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_baseline_hook_fails_closed_on_malformed_json(baseline_hook_command: str) -> None:
    """A malformed-JSON stdin payload is blocked, not silently let through.

    :param baseline_hook_command: Hook command body fixture.
    """
    result = subprocess.run(  # noqa: S603
        ["bash", "-c", baseline_hook_command],  # noqa: S607
        input="not valid json",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


def test_yaml_run_hook_blocks_edit_introducing_comment(
    yaml_run_hook_command: str, tmp_path: Path
) -> None:
    """An Edit that introduces a comment inside a ``run: |`` block is blocked (Edit path).

    :param yaml_run_hook_command: Hook command body fixture.
    :param tmp_path: pytest tmp_path.
    """
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    workflow = workflow_dir / "test.yml"
    workflow.write_text("jobs:\n  test:\n    steps:\n      - run: |\n          echo hi\n")
    result = _run_hook_command(
        yaml_run_hook_command,
        {
            "tool_name": "Edit",
            "tool_input": {
                "file_path": str(workflow),
                "old_string": "          echo hi\n",
                "new_string": "          # injected bad comment\n          echo hi\n",
            },
        },
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


@pytest.mark.parametrize(
    "header",
    [
        "      - run: |-\n",
        "      - run: |+\n",
        "      - run: >\n",
        "      - run: >-\n",
        "      - run: >+\n",
        "      - setup: |\n",
        "      - setup: >-\n",
    ],
)
def test_yaml_run_hook_blocks_comment_in_chomping_variants(
    yaml_run_hook_command: str, header: str
) -> None:
    """Chomping (``|-`` / ``|+``) and folded (``>``) scalars are also scanned.

    :param yaml_run_hook_command: Hook command body fixture.
    :param header: ``run:`` / ``setup:`` header line with a chomping indicator.
    """
    content = (
        "jobs:\n"
        "  test:\n"
        "    steps:\n" + header + "          # bad comment inside the block\n"
        "          echo hi\n"
    )
    result = _run_hook_command(
        yaml_run_hook_command,
        {
            "tool_name": "Write",
            "tool_input": {"file_path": ".github/workflows/test.yml", "content": content},
        },
    )
    assert result.returncode == 2, (result.returncode, result.stderr)


def test_yaml_run_hook_fails_closed_on_malformed_json(yaml_run_hook_command: str) -> None:
    """A malformed-JSON stdin payload is blocked, not silently let through.

    :param yaml_run_hook_command: Hook command body fixture.
    """
    result = subprocess.run(  # noqa: S603
        ["bash", "-c", yaml_run_hook_command],  # noqa: S607
        input="not valid json",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr
