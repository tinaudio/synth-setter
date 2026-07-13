"""Unit tests for ``agent/_shared/pr_command_classifier.py``.

The classifier decides whether a Bash tool command really creates a PR
(``direct``), smuggles the creation through a shell wrapper (``wrapped``),
mentions ``gh pr create`` but cannot be parsed (``unparsable``), or is
unrelated (empty string). These tests pin the contract the pre-PR review gate
relies on: every classification error must fail toward blocking, never toward
letting an unreviewed PR through.
"""

from __future__ import annotations

import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HELPER_PATH = _REPO_ROOT / "agent" / "_shared" / "pr_command_classifier.py"


def _load_module() -> ModuleType:
    """Import ``pr_command_classifier`` by path so agent/ needn't be on sys.path.

    :returns: The loaded module object.
    """
    spec = importlib.util.spec_from_file_location("pr_command_classifier", _HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", name="classifier")
def classifier_fixture() -> ModuleType:
    """Load ``pr_command_classifier`` once per module via path-based import.

    :returns: The loaded module object.
    """
    return _load_module()


# A subset of these commands is exercised end-to-end through the gate hook in
# test_pre_pr_review_gate_worktree.py — keep new bypass variants in sync there.
@pytest.mark.parametrize(
    "command",
    [
        "gh pr create --title x --body y",
        "  gh pr create --title x --body y",
        "echo preflight\ngh pr create --title x --body y",
        "! gh pr create --title x --body y",
        "if gh pr create --title x --body y; then :; fi",
        "if false; then :; else gh pr create --title x --body y; fi",
        "if false; then :; elif gh pr create --title x --body y; then :; fi",
        "until gh pr create --title x --body y; do :; done",
        # Heredoc bodies split into ordinary command segments, so the smuggled
        # command classifies as the direct invocation it is.
        "bash <<EOF\ngh pr create --title x --body y\nEOF",
        "bash - <<'EOF'\ngh pr create --title x --body y\nEOF",
        # Over-block pin: `-I{}` shreds into brace segments, stranding a
        # `gh pr create` segment. Wrong reading of xargs, safe direction.
        "xargs -I{} gh pr create --title x --body y",
        "/usr/bin/gh pr create --title x --body y",
        "./gh pr create --title x --body y",
        "gh \\\npr create --title x --body y",
        "cd /repo && gh pr create --title x --body y",
        "echo preflight; gh pr create --title x --body y",
        "FOO=bar gh pr create --title x --body y",
        "env gh pr create --title x --body y",
        "env FOO=bar gh pr create --title x --body y",
        "env =x gh pr create --title x --body y",
        "/usr/bin/env gh pr create --title x --body y",
        "sudo gh pr create --title x --body y",
        "sudo -u root gh pr create --title x --body y",
        "sudo -- gh pr create --title x --body y",
        "/usr/bin/sudo gh pr create --title x --body y",
        "nice -n 5 gh pr create --title x --body y",
        "/usr/bin/time gh pr create --title x --body y",
        "time -o /dev/null gh pr create --title x --body y",
        "exec gh pr create --title x --body y",
        "exec -a fake gh pr create --title x --body y",
        "command gh pr create --title x --body y",
        "builtin exec gh pr create --title x --body y",
        "builtin command gh pr create --title x --body y",
        "eval gh pr create --title x --body y",
        "`gh pr create --title x --body y`",
        "OUT=`gh pr create --title x --body y`",
        "OUT=$(gh pr create --title x --body y)",
    ],
)
def test_classify_direct_invocation_returns_direct(classifier: ModuleType, command: str) -> None:
    """A real, unwrapped PR creation is classified ``direct``.

    :param classifier: The loaded classifier module.
    :param command: Bash command that executes ``gh pr create`` directly.
    """
    assert classifier.classify(command) == "direct"


# A subset of these commands is exercised end-to-end through the gate hook in
# test_pre_pr_review_gate_worktree.py — keep new bypass variants in sync there.
@pytest.mark.parametrize(
    "command",
    [
        "bash -c 'gh pr create --title x --body y'",
        "bash -lc 'gh pr create --title x --body y'",
        "bash -c 'echo hi\ngh pr create --title x --body y'",
        # Over-block pins: a `-c`-bearing flag cluster (not valid bash) and a
        # `-c` value that itself starts with a dash both read as wrapping.
        # Wrong readings of the argv, but they fail toward blocking.
        "bash -norc 'gh pr create --title x --body y'",
        "bash -c '-x value with a c' 'gh pr create --title x --body y'",
        "bash -c -- 'gh pr create --title x --body y'",
        "bash -c $'gh pr create --title x --body y'",
        "bash --command 'gh pr create --title x --body y'",
        "bash --command='gh pr create --title x --body y'",
        "sh -c 'gh pr create --title x --body y'",
        "dash -c 'gh pr create --title x --body y'",
        "ksh -c 'gh pr create --title x --body y'",
        "zsh -c 'gh pr create --title x --body y'",
        "env bash -c 'gh pr create --title x --body y'",
        "sudo bash -c 'gh pr create --title x --body y'",
        "env -S \"bash -c 'gh pr create --title x --body y'\"",
        'bash <<< "gh pr create --title x --body y"',
        "bash <<<'gh pr create --title x --body y'",
    ],
)
def test_classify_shell_wrapped_invocation_returns_wrapped(
    classifier: ModuleType, command: str
) -> None:
    """PR creation smuggled through a shell wrapper is classified ``wrapped``.

    :param classifier: The loaded classifier module.
    :param command: Bash command that runs ``gh pr create`` via a shell.
    """
    assert classifier.classify(command) == "wrapped"


@pytest.mark.parametrize(
    "command",
    [
        "echo 'docs mention gh pr create somewhere'",
        'git commit -m "see gh pr create above"',
        "grep -n 'gh pr create' agent/hooks/pre-pr-review-gate.sh",
        "echo 'use `gh pr create` to open it'",
        "printf 'gh pr create --help\\n'",
        "gh pr view 123",
        "gh pr list",
        "ls -la",
        "",
        # Documented non-goal: non-shell interpreters can execute anything;
        # the classifier only models the sh-family wrappers (see module doc).
        "python3 -c \"import os; os.system('gh pr create -t x -b y')\"",
        "perl -e 'system(qq{gh pr create -t x -b y})'",
    ],
)
def test_classify_unrelated_command_returns_empty(classifier: ModuleType, command: str) -> None:
    """Prose mentions and unrelated commands classify as empty (not gated).

    :param classifier: The loaded classifier module.
    :param command: Bash command that does not execute ``gh pr create``.
    """
    assert classifier.classify(command) == ""


def test_classify_unparsable_command_mentioning_pr_create_fails_closed(
    classifier: ModuleType,
) -> None:
    """An unlexable command that mentions ``gh pr create`` is ``unparsable``.

    Parse failures on the gated surface must block, never fall open.

    :param classifier: The loaded classifier module.
    """
    assert classifier.classify('gh pr create --title "unterminated') == "unparsable"


def test_classify_unparsable_command_without_mention_returns_empty(
    classifier: ModuleType,
) -> None:
    """An unlexable command with no ``gh pr create`` mention is not gated.

    Blocking every hard-to-lex Bash command would brick unrelated tool use.

    :param classifier: The loaded classifier module.
    """
    assert classifier.classify('echo "unterminated') == ""


def test_classify_ansi_c_quoted_escapes_stay_quoted(classifier: ModuleType) -> None:
    """ANSI-C ``$'...'`` payloads with escapes don't unbalance the lexer.

    An escaped quote inside ``$'...'`` previously unbalanced the rewritten
    string and crashed the parser — which the old hook swallowed into a silent
    allow.

    :param classifier: The loaded classifier module.
    """
    assert classifier.classify("bash -c $'gh pr create -t \\'x\\''") == "wrapped"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("gh pr create -t x -b y", "direct"),
        ("bash -c 'gh pr create -t x -b y'", "wrapped"),
        ('gh pr create --title "unterminated', "unparsable"),
        ("ls -la", ""),
    ],
)
def test_cli_prints_mode_and_exits_zero(
    classifier: ModuleType, command: str, expected: str
) -> None:
    """The CLI prints the classification for argv[1] and returns 0.

    :param classifier: The loaded classifier module.
    :param command: Bash command passed as the CLI's single argument.
    :param expected: Mode string the CLI must print for it.
    """
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        rc = classifier._main(["pr_command_classifier.py", command])
    assert rc == 0
    assert buffer.getvalue().strip() == expected


def test_cli_without_argument_exits_two(classifier: ModuleType) -> None:
    """The CLI exits 2 on usage errors so the gate can fail closed.

    :param classifier: The loaded classifier module.
    """
    assert classifier._main(["pr_command_classifier.py"]) == 2
