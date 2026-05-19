"""Invariant: each Bash hook handler that scopes itself uses Claude Code permission-rule ``if:`` syntax (e.g. ``Bash(gh pr create *)``) at the handler level, and the body of each hook honours its documented exit contract.

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
    # Trust boundary: command_body comes from maintainer-controlled settings.json.
    return subprocess.run(  # noqa: S603
        ["bash", "-c", command_body],  # noqa: S607 — bash is a hard requirement of the harness
        input=json.dumps(stdin_payload),
        capture_output=True,
        text=True,
        check=False,
    )


def _run_hook_command_raw(command_body: str, raw_stdin: str) -> subprocess.CompletedProcess[str]:
    """Run a hook ``command`` body with raw (un-JSON-encoded) stdin.

    :param command_body: The shell command body from a hook handler.
    :param raw_stdin: Raw text written verbatim to the hook's stdin.
    :returns: Completed subprocess result with captured stdout/stderr and exit code.
    """
    # Trust boundary: command_body comes from maintainer-controlled settings.json.
    return subprocess.run(  # noqa: S603
        ["bash", "-c", command_body],  # noqa: S607 — bash is a hard requirement of the harness
        input=raw_stdin,
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
    ("Doc-drift advisory review", "Bash(gh pr create *)"),
    # `Bash(git*)` so the harness still fires the hook on `git -c X=Y commit
    # ...` (git-level options before the subcommand); the wrapper defers to
    # the Python scanner, which re-scopes to actual commit invocations.
    ("Git-commit-trailer-check", "Bash(git*)"),
    ("PR review resolver", "Bash(git push *)"),
    ("Pre-PR review gate", "Bash(gh pr create *)"),
)

_EXPECTED_SHARED_HOOK_COMMANDS: tuple[tuple[str, str], ...] = (
    ("Auto-format", "bash agent/hooks/edit-write.sh format"),
    ("Auto-test", "bash agent/hooks/edit-write.sh test"),
    ("Credential protection", "bash agent/hooks/edit-write.sh credential-protect"),
    ("Doc-drift advisory review", "bash agent/hooks/doc-drift.sh"),
    ("Git-commit-trailer-check", "bash agent/hooks/git-commit-trailer-check.sh"),
    ("No-baseline-additions", "bash agent/hooks/no-baseline-additions.sh"),
    ("No-yaml-run-comments", "bash agent/hooks/no-yaml-run-comments.sh"),
    ("PR review resolver", "bash agent/hooks/pr-review-resolver.sh"),
    ("Taxonomy verification", "bash agent/hooks/verify-gh-taxonomy.sh"),
)


def _workflow_with_run_body(body: str) -> str:
    """Wrap step body lines in minimal workflow scaffolding.

    :param body: Lines (each terminated with ``\\n``) to inject under ``steps:``.
        Callers supply the step (including the ``- name:``/``run: |`` header) and
        any block-scalar body, so the helper can serve both inside-run-block and
        above-step test cases.
    :returns: A complete workflow YAML string suitable for the yaml-run hook.
    """
    return "name: test\non: push\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n" + body


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
    content = _workflow_with_run_body(
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
    content = _workflow_with_run_body(
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
    content = _workflow_with_run_body(
        "      - run: |\n          #!/usr/bin/env bash\n          echo hi\n"
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


@pytest.mark.parametrize(
    ("command_str", "expected_substring"),
    [
        pytest.param(
            'git commit -m "feat: x\n\nCo-Authored-By: Claude <noreply@anthropic.com>"',
            "Co-Authored-By",
            id="co-authored-by-trailer",
        ),
        pytest.param(
            'git commit -m "feat: x via noreply@anthropic.com"',
            "BLOCKED",
            id="noreply-email",
        ),
    ],
)
def test_trailer_hook_blocks_forbidden_trailer_in_message_body(
    trailer_hook_command: str, command_str: str, expected_substring: str
) -> None:
    """Forbidden trailers inside a ``-m`` body are blocked.

    :param trailer_hook_command: Hook command body fixture.
    :param command_str: ``git commit`` command whose body carries the offender.
    :param expected_substring: Substring that must appear in stderr.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": command_str}},
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr
    assert expected_substring in result.stderr


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
    """A subject that names a Claude model (with no trailer key) is not flagged.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": 'git commit -m "feat(eval): tokeniser for Claude Sonnet 4.5"'}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


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
    result = _run_hook_command_raw(trailer_hook_command, "not valid json")
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


_METACHAR_PARAMS = [
    pytest.param("&&", id="and"),
    pytest.param("||", id="or"),
    pytest.param(";", id="semicolon"),
    pytest.param("|", id="pipe"),
    pytest.param("&", id="ampersand"),
]


@pytest.mark.parametrize("metachar", _METACHAR_PARAMS)
def test_trailer_hook_does_not_false_positive_on_chained_grep_n_metachars(
    trailer_hook_command: str, metachar: str
) -> None:
    """Every shell metachar terminating ``git commit``'s argv slice is honoured.

    :param trailer_hook_command: Hook command body fixture.
    :param metachar: Shell metachar to chain after the commit.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": f'git commit -m "msg" {metachar} grep -n bar file.txt'}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


@pytest.mark.parametrize("metachar", _METACHAR_PARAMS)
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
    result = _run_hook_command_raw(baseline_hook_command, "not valid json")
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
    result = _run_hook_command_raw(yaml_run_hook_command, "not valid json")
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


def test_trailer_hook_blocks_message_equals_form(trailer_hook_command: str) -> None:
    """``git commit --message=<body>`` (= form, not space-separated) is blocked.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {
            "tool_input": {
                "command": (
                    'git commit --message="feat: x\n\nCo-Authored-By: Claude <noreply@anthropic.com>"'
                )
            }
        },
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


def test_trailer_hook_blocks_file_equals_form(trailer_hook_command: str, tmp_path: Path) -> None:
    """``git commit --file=<path>`` whose file contains a forbidden trailer is blocked.

    :param trailer_hook_command: Hook command body fixture.
    :param tmp_path: pytest tmp_path.
    """
    msg_file = tmp_path / "msg.txt"
    msg_file.write_text("feat: x\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n")
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": f"git commit --file={msg_file}"}},
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


def test_trailer_hook_blocks_heredoc_trailer_in_raw_command(trailer_hook_command: str) -> None:
    """Heredoc-bodied commits fall back to raw-command scanning and still block.

    When no ``-m``/``-F`` body is recoverable from argv (heredoc /
    ``$(cat msg.txt)``), the hook scans the raw command via anchored trailer
    regexes — a ``Co-Authored-By:`` line in the heredoc body must still trip.

    :param trailer_hook_command: Hook command body fixture.
    """
    cmd = "git commit <<EOF\nfeat: x\n\nCo-Authored-By: Claude\nEOF"
    result = _run_hook_command(trailer_hook_command, {"tool_input": {"command": cmd}})
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


def test_trailer_hook_allows_positional_n_after_double_dash(trailer_hook_command: str) -> None:
    """``git commit -- -n`` (positional ``-n`` after ``--``) is not ``--no-verify``.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": "git commit -- -n"}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_trailer_hook_handles_unbalanced_quotes_fallback(trailer_hook_command: str) -> None:
    """An unterminated-quote command triggers the shlex fallback path cleanly.

    With no trailer-shaped substring in the (raw) command, the fallback must not mis-attribute and
    the hook should exit 0.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": 'git commit -m "feat: x'}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_trailer_hook_blocks_lowercase_generated_with_in_trailer_position(
    trailer_hook_command: str,
) -> None:
    """Lowercase ``generated with`` in trailer position is blocked (IGNORECASE).

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": 'git commit -m "feat: x\n\ngenerated with claude code"'}},
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


def test_trailer_hook_does_not_block_generated_with_in_subject(
    trailer_hook_command: str,
) -> None:
    """``generated with`` in the subject line (no preceding newline) is not flagged.

    Pins the anchoring fix: trailer regexes require ``(?:^|\\n|\\\\n)\\s*`` before
    the key, so ``fix: docs generated with sphinx-build`` is a clean subject.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": 'git commit -m "fix: docs generated with sphinx-build"'}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_trailer_hook_blocks_direct_claude_attribution_trailer(
    trailer_hook_command: str,
) -> None:
    """Direct positive test for the ``[A-Za-z-]+:\\s*Claude\\s+(Code|Opus|Sonnet|Haiku)`` regex.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": 'git commit -m "feat: x\n\nAuthor: Claude Code"'}},
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


def test_baseline_hook_allows_equal_count_edit(baseline_hook_command: str, tmp_path: Path) -> None:
    """An Edit that swaps rows (NEW_COUNT == OLD_COUNT) is allowed — strict ``>`` boundary.

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
                "old_string": "path/a.py:1:1: DOC101\npath/b.py:2:2: DOC102\n",
                "new_string": "path/c.py:3:3: DOC103\npath/d.py:4:4: DOC104\n",
            },
        },
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_yaml_run_hook_blocks_compute_config_path(yaml_run_hook_command: str) -> None:
    """A `#`-comment inside a ``run: |`` block in ``configs/compute/*.yaml`` is blocked.

    :param yaml_run_hook_command: Hook command body fixture.
    """
    content = _workflow_with_run_body(
        "      - name: bad\n"
        "        run: |\n"
        "          # bad inline comment inside run block\n"
        "          echo hi\n"
    )
    result = _run_hook_command(
        yaml_run_hook_command,
        {
            "tool_name": "Write",
            "tool_input": {"file_path": "configs/compute/foo.yaml", "content": content},
        },
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


def test_yaml_run_hook_blocks_comment_in_map_key_run_block(yaml_run_hook_command: str) -> None:
    """Map-key ``run: |`` (no leading ``- ``) with a body comment is blocked.

    Only the list-marker form (``- run: |``) was previously exercised; this
    pins the bare map-key form where the body is indented deeper than the
    ``run:`` key column.

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
        "          # this is a bad comment inside a map-key run block\n"
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


def test_yaml_run_hook_allows_unrelated_edit_to_file_with_preexisting_comment(
    yaml_run_hook_command: str, tmp_path: Path
) -> None:
    """Unrelated edits don't trip on pre-existing block-scalar comments.

    The scanner diffs pre- vs. post-edit violation multisets so legacy comments (or comments
    inserted by another tool) do not lock the file against unrelated edits.

    :param yaml_run_hook_command: Hook command body fixture.
    :param tmp_path: pytest tmp_path.
    """
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    workflow = workflow_dir / "test.yml"
    workflow.write_text(
        "jobs:\n  test:\n    steps:\n      - run: |\n"
        "          # legacy comment\n          echo hi\n"
    )
    result = _run_hook_command(
        yaml_run_hook_command,
        {
            "tool_name": "Edit",
            "tool_input": {
                "file_path": str(workflow),
                "old_string": "echo hi",
                "new_string": "echo hello",
            },
        },
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_yaml_run_hook_allows_edit_removing_comment(
    yaml_run_hook_command: str, tmp_path: Path
) -> None:
    """An Edit that *removes* a pre-existing block-scalar comment is allowed.

    Exercises the ``replace_all=true`` path: every matched occurrence must
    be subtracted from the post-edit violation set before the diff is taken.

    :param yaml_run_hook_command: Hook command body fixture.
    :param tmp_path: pytest tmp_path.
    """
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    workflow = workflow_dir / "test.yml"
    workflow.write_text(
        "jobs:\n  test:\n    steps:\n      - run: |\n"
        "          # stale comment\n          echo a\n"
        "      - run: |\n"
        "          # stale comment\n          echo b\n"
    )
    result = _run_hook_command(
        yaml_run_hook_command,
        {
            "tool_name": "Edit",
            "tool_input": {
                "file_path": str(workflow),
                "old_string": "          # stale comment\n",
                "new_string": "",
                "replace_all": True,
            },
        },
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_trailer_hook_catches_dash_c_option_before_commit(trailer_hook_command: str) -> None:
    """``git -c <opt>=<val> commit -n`` is blocked despite the option between git and commit.

    The Python slicer walks past git-level option tokens (``-c``, ``--git-dir``,
    ``--work-tree``, ``--namespace``) before locating the ``commit`` subcommand.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": 'git -c gpg.sign=false commit -n -m "x"'}},
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


def test_trailer_hook_catches_dash_c_with_co_authored_by(trailer_hook_command: str) -> None:
    """``git -c X=Y commit -m "...Co-Authored-By: ..."`` is blocked.

    The trailer-scan path runs once the slicer skips git-level options to
    find the ``commit`` subcommand.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {
            "tool_input": {
                "command": (
                    "git -c gpg.sign=false commit "
                    '-m "feat: x\n\nCo-Authored-By: Claude <noreply@anthropic.com>"'
                )
            }
        },
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


def test_trailer_hook_allows_dash_S_keyid_with_n(trailer_hook_command: str) -> None:
    """``git commit -S<keyid> -m "..."`` with an alpha keyid containing ``n`` is allowed.

    The inline gpg-sign keyid is the *value* of ``-S``, not a bundle of
    additional flags.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": 'git commit -Sandykey -m "feat: x"'}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_trailer_hook_still_blocks_bundled_n_after_value_attaching_flag(
    trailer_hook_command: str,
) -> None:
    """``-anSkey`` (bundled ``-a -n -S<keyid>``) still trips the no-verify check.

    ``-S`` only consumes the rest of a cluster when it sits at the cluster
    start; here the leading ``-a -n`` keep their flag meaning.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": 'git commit -anSkey -m "feat: x"'}},
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


def test_trailer_hook_allows_generated_with_non_agent_tool(trailer_hook_command: str) -> None:
    """A body line ``Generated with sphinx-build manually`` is allowed.

    The attribution rule requires an agent-product name (``Claude``,
    ``Copilot``, ...) on the same line within ~80 chars.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {
            "tool_input": {
                "command": (
                    'git commit -m "feat: docs\n\nThis page is auto-generated.\n'
                    'Generated with sphinx-build manually."'
                )
            }
        },
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


@pytest.mark.parametrize(
    "agent_keyword",
    ["Claude Code", "Anthropic Claude", "ChatGPT", "Copilot", "Cursor", "Gemini"],
)
def test_trailer_hook_blocks_generated_with_any_agent_keyword(
    trailer_hook_command: str, agent_keyword: str
) -> None:
    """``Generated with <agent>`` for known agent-product keywords is still blocked.

    :param trailer_hook_command: Hook command body fixture.
    :param agent_keyword: Agent-product name appended after ``Generated with``.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": f'git commit -m "feat: x\n\nGenerated with {agent_keyword}"'}},
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


def test_baseline_hook_honours_replace_all(baseline_hook_command: str, tmp_path: Path) -> None:
    """An Edit with ``replace_all: true`` is row-counted at every occurrence.

    The synthesised post-edit content reflects all replacements so the BLOCKED message reports the
    actual proposed row count.

    :param baseline_hook_command: Hook command body fixture.
    :param tmp_path: pytest tmp_path.
    """
    baseline = tmp_path / ".pydoclint-baseline.txt"
    baseline.write_text("a DOC101\nb DOC101\nc DOC101\n")
    result = _run_hook_command(
        baseline_hook_command,
        {
            "tool_name": "Edit",
            "tool_input": {
                "file_path": str(baseline),
                "old_string": "DOC101",
                "new_string": "DOC101\nEXTRA",
                "replace_all": True,
            },
        },
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "Proposed rows: 6" in result.stderr
    assert "Current rows: 3" in result.stderr


def test_trailer_hook_passes_non_commit_git_command(trailer_hook_command: str) -> None:
    """``git status`` (and other non-commit git commands) fast-paths through.

    The handler ``if:`` is broadened to ``Bash(git*)``; the hook must exit 0
    for any git invocation that is not a commit.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": "git status"}},
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


@pytest.mark.parametrize(
    "command",
    [
        pytest.param('git --git-dir=.git commit -n -m "x"', id="git-dir-equals"),
        pytest.param('git --work-tree /tmp commit -n -m "x"', id="work-tree-space"),
        pytest.param('git -c user.name="Foo Bar" commit -n -m "x"', id="dash-c-quoted-value"),
    ],
)
def test_trailer_hook_catches_no_verify_through_various_git_level_options(
    trailer_hook_command: str, command: str
) -> None:
    """Other git-level option forms (``--git-dir``, ``--work-tree``, quoted ``-c``) also gate.

    Pins the Python slicer's option-skipping logic across the value-bearing
    long options and the shlex-quoted short option that an ERE alone cannot
    parse.

    :param trailer_hook_command: Hook command body fixture.
    :param command: ``git`` invocation carrying the option form under test.
    """
    result = _run_hook_command(trailer_hook_command, {"tool_input": {"command": command}})
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


@pytest.mark.parametrize(
    "agent_keyword",
    ["Codex", "Bard", "GPT-4", "GPT4"],
)
def test_trailer_hook_blocks_generated_with_additional_agent_keywords(
    trailer_hook_command: str, agent_keyword: str
) -> None:
    """Additional agents in the ``Generated with`` alternation are blocked.

    :param trailer_hook_command: Hook command body fixture.
    :param agent_keyword: Agent-product name appended after ``Generated with``.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": f'git commit -m "feat: x\n\nGenerated with {agent_keyword}"'}},
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


@pytest.mark.parametrize(
    "cluster",
    [pytest.param("-un", id="dash-u-n"), pytest.param("-tn", id="dash-t-n")],
)
def test_trailer_hook_blocks_un_and_tn_bundles(trailer_hook_command: str, cluster: str) -> None:
    """``-un`` / ``-tn`` are read as ``-u -n`` / ``-t -n``, not as value-attaching forms.

    ``-u`` and ``-t`` are excluded from the value-attaching short-flag set
    because their bare form is the common usage; treating their tail as a
    value would silently bypass ``-n`` detection.

    :param trailer_hook_command: Hook command body fixture.
    :param cluster: Short-flag cluster under test.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {"tool_input": {"command": f'git commit {cluster} -m "feat: x"'}},
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


@pytest.mark.parametrize(
    "command",
    [
        pytest.param('git log --grep="Co-Authored-By: anyone"', id="git-log-grep-coauthor"),
        pytest.param(
            'git log --grep="\\nGenerated with Claude Code"',
            id="git-log-grep-generated-with-claude",
        ),
        pytest.param('git log --grep="noreply@anthropic.com"', id="git-log-grep-noreply"),
        pytest.param(
            'git show HEAD --format="%B" | grep "Co-Authored-By:"',
            id="git-show-piped-grep-coauthor",
        ),
        pytest.param(
            'echo "msg\\nCo-Authored-By: x" > /tmp/note',
            id="non-git-bash-with-trailer-substring",
        ),
    ],
)
def test_trailer_hook_does_not_false_positive_on_non_commit_with_trailer_substring(
    trailer_hook_command: str, command: str
) -> None:
    """Non-commit invocations that mention a trailer-shaped substring pass through.

    The scanner short-circuits to "no findings" when no ``git commit`` argv
    slice is present, so the raw-command fallback (used only inside a real
    commit invocation with a heredoc body) cannot misfire on diagnostic
    commands like ``git log --grep="Co-Authored-By: ..."`` or on unrelated
    Bash invocations whose command string happens to contain the substring.

    :param trailer_hook_command: Hook command body fixture.
    :param command: Non-commit invocation under test.
    """
    result = _run_hook_command(trailer_hook_command, {"tool_input": {"command": command}})
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_trailer_hook_still_blocks_heredoc_commit_with_trailer(
    trailer_hook_command: str,
) -> None:
    """Heredoc-bodied ``git commit`` invocations still fall back to raw-command scanning.

    Pins that the raw-command fallback remains active when a commit slice IS
    found but no ``-m``/``-F`` body is recoverable — removing the fallback
    entirely would create a different bypass.

    :param trailer_hook_command: Hook command body fixture.
    """
    cmd = "git commit <<EOF\nfeat: x\n\nCo-Authored-By: Claude\nEOF"
    result = _run_hook_command(trailer_hook_command, {"tool_input": {"command": cmd}})
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr


def test_yaml_run_hook_blocks_new_comment_when_legacy_one_is_preserved(
    yaml_run_hook_command: str, tmp_path: Path
) -> None:
    """An Edit that preserves a legacy comment AND introduces a new one blocks on the new one.

    Verifies the multiset diff: subtracting one legacy entry must not also
    consume an unrelated new entry.

    :param yaml_run_hook_command: Hook command body fixture.
    :param tmp_path: pytest tmp_path.
    """
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    workflow = workflow_dir / "test.yml"
    workflow.write_text(
        "jobs:\n  test:\n    steps:\n      - run: |\n"
        "          # legacy comment\n          echo hi\n"
    )
    result = _run_hook_command(
        yaml_run_hook_command,
        {
            "tool_name": "Edit",
            "tool_input": {
                "file_path": str(workflow),
                "old_string": "          echo hi\n",
                "new_string": "          # newly introduced\n          echo hi\n",
            },
        },
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "BLOCKED" in result.stderr
    assert "newly introduced" in result.stderr


def test_baseline_hook_does_not_pollute_stderr_on_allowed_edit(
    baseline_hook_command: str, tmp_path: Path
) -> None:
    """The ``note: Edit old_string not found`` line must not leak to stderr.

    The Edit tool itself rejects a missing old_string. Routing the diagnostic
    through ``log()`` (file log) keeps user-visible stderr reserved for BLOCKED
    messages — any chatter on allowed edits is noise.

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
                "old_string": "path/never-existed.py:9:9: DOC999",
                "new_string": "anything",
            },
        },
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    # The Copilot finding is specifically about the "note:" / "Edit old_string"
    # message; pre-existing _lib.sh log-dir noise is out of scope.
    assert "note:" not in result.stderr, result.stderr
    assert "Edit old_string" not in result.stderr, result.stderr


def test_trailer_hook_finding_does_not_contain_embedded_newlines(
    trailer_hook_command: str,
) -> None:
    """Each Findings entry is one tight line — no leaked ``\\n`` from regex anchors.

    Trailer regexes anchor on ``(?:^|\\n|\\\\n)\\s*``, so ``m.group(0)`` includes
    the leading newline / whitespace context. The scanner must collapse those
    before emitting the finding or the BLOCKED Findings block renders garbled.

    :param trailer_hook_command: Hook command body fixture.
    """
    result = _run_hook_command(
        trailer_hook_command,
        {
            "tool_input": {
                "command": 'git commit -m "feat: x\n\nCo-Authored-By: Claude <noreply@anthropic.com>"',
            },
        },
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    # Precondition asserts give a focused failure when the BLOCKED message is
    # restructured, instead of an IndexError stack trace from the split below.
    assert "Findings:" in result.stderr, result.stderr
    assert "Rules" in result.stderr, result.stderr
    findings_block = result.stderr.split("Findings:", 1)[1].split("Rules", 1)[0]
    finding_lines = [line for line in findings_block.splitlines() if line.strip()]
    for line in finding_lines:
        assert "\\n" not in line, f"escaped newline leaked into finding: {line!r}"
        assert "\n" not in line.rstrip(), f"raw newline leaked into finding: {line!r}"


def test_yaml_run_hook_description_documents_both_extensions() -> None:
    """The matcher description must mention both ``.yml`` and ``.yaml`` extensions.

    The hook's ``in_scope`` accepts ``.github/workflows/*.{yml,yaml}`` and
    ``configs/compute/*.{yml,yaml}``. A description naming only one extension
    per directory misleads users into surprise when the other fires.
    """
    # _find_handler enforces "exactly one matcher entry" — call it first so a
    # description rename trips a focused AssertionError there, not a silent
    # StopIteration here.
    _find_handler("No-yaml-run-comments")
    matcher_entry = next(
        entry
        for entry in _matcher_entries()
        if "No-yaml-run-comments" in entry.get("description", "")
    )
    desc = matcher_entry["description"]
    assert "yml" in desc and "yaml" in desc, desc
    assert "workflows" in desc and "compute" in desc, desc
    # Each scoped directory must reference both extensions, not just one.
    assert "workflows/*.{yml,yaml}" in desc or (
        "workflows/*.yml" in desc and "workflows/*.yaml" in desc
    ), desc
    assert "compute/*.{yml,yaml}" in desc or (
        "compute/*.yml" in desc and "compute/*.yaml" in desc
    ), desc
