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
    """Module-scoped wrapper around :func:`_load_module` for test injection.

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
        "gh -R tinaudio/synth-setter pr create --title x --body y",
        "gh --repo=tinaudio/synth-setter pr create --title x --body y",
        "echo preflight\ngh pr create --title x --body y",
        "source .venv/bin/activate && gh pr create --title x --body y",
        "nohup gh pr create --title x --body y",
        "setsid gh pr create --title x --body y",
        "stdbuf -o0 gh pr create --title x --body y",
        "timeout 30 gh pr create --title x --body y",
        "xargs gh pr create --title x --body y",
        "2>/dev/null gh pr create --title x --body y",
        ">/tmp/out gh pr create --title x --body y",
        "</dev/null gh pr create --title x --body y",
        "sudo 2>/dev/null gh pr create --title x --body y",
        "2>&1 gh pr create --title x --body y",
        "1>&2 gh pr create --title x --body y",
        "2> /tmp/errlog gh pr create --title x --body y",
        "< /dev/null gh pr create --title x --body y",
        ">> /tmp/log gh pr create --title x --body y",
        # A fd-dup substring inside a quoted arg must not swallow the quote.
        'gh pr "create" --title "release notes 2>&1" --body "desc"',
        'gh pr create --title "quoted 2>&1" --body y',
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
        # Over-block pins: a `-c`-bearing flag cluster and a dash-led `-c`
        # value both misread as wrapping — wrong, but fails toward blocking.
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
        # GNU env -S appends the remaining argv to the split string, and the
        # split string hides its own words from top-level tokenization.
        "env -S 'gh pr create --title x --body y'",
        'env -S "gh" pr create --title x --body y',
        "env -S gh pr create --title x --body y",
        'env -S "gh pr" create --title x --body y',
        "sudo env -S 'gh pr create --title x --body y'",
        "2>/dev/null env -S 'gh pr create --title x --body y'",
        "nice -n 5 env -S 'gh pr create --title x --body y'",
        "timeout 30 env -S 'gh pr create --title x --body y'",
        # GNU env -S also accepts the getopt-fused spellings.
        "env --split-string='gh pr create --title x --body y'",
        "env -S'gh pr create --title x --body y'",
        "env -Sgh pr create --title x --body y",
        "sudo env --split-string='gh pr create --title x --body y'",
        # -S bundled behind no-value short flags (-v debug, -i ignore-env).
        "env -vSgh pr create --title x --body y",
        "env -iSgh pr create --title x --body y",
        "env -vS 'gh pr create --title x --body y'",
        'bash <<< "gh pr create --title x --body y"',
        "bash <<<'gh pr create --title x --body y'",
        # A single-line heredoc keeps the body in the shell's own segment.
        "bash <<EOF gh pr create --title x --body y",
        # eval re-parses its argument string, hiding the real argv.
        "eval gh pr create --title x --body y",
        'eval "gh pr create --title x --body y"',
        "builtin eval 'gh pr create --title x --body y'",
        # A bare shell executing piped text whose upstream carries the recipe.
        "echo 'gh pr create --title x --body y' | bash",
        "printf 'gh pr create --title x --body y\\n' | sh",
        "echo 'gh pr create --title x --body y' |& bash",
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
        # A quoted fd-dup mention must not desync the lexer into a false block.
        'git commit -m "silence gh pr create noise 2>&1"',
        # Documented non-goals (see module docstring): non-shell interpreters,
        # process substitution, and unmentioned pipes into a shell.
        "python3 -c \"import os; os.system('gh pr create -t x -b y')\"",
        "perl -e 'system(qq{gh pr create -t x -b y})'",
        "source <(echo 'gh pr create -t x -b y')",
        "cat script.sh | bash",
        # env -S argv reconstruction must not gate unrelated split strings.
        "env -S 'python3 script.py' arg1 arg2",
        "env --split-string='python3 script.py'",
        # -u consumes the rest of the cluster as the var to unset, so `Sgh`
        # is a variable name and env runs the `pr` command, not gh.
        "env -uSgh pr create --title x --body y",
        # `gh` here is the redirection target; bash runs `pr create ...`.
        "> gh pr create --title x --body y",
    ],
)
def test_classify_unrelated_command_returns_empty(classifier: ModuleType, command: str) -> None:
    """Prose mentions and unrelated commands classify as empty (not gated).

    :param classifier: The loaded classifier module.
    :param command: Bash command that does not execute ``gh pr create``.
    """
    assert classifier.classify(command) == ""


@pytest.mark.parametrize(
    "command",
    [
        'gh pr create --title "unterminated',
        "env -S 'gh pr create --title \"unterminated'",
    ],
)
def test_classify_unparsable_command_mentioning_pr_create_fails_closed(
    classifier: ModuleType, command: str
) -> None:
    """An unlexable command that mentions ``gh pr create`` is ``unparsable``.

    Parse failures on the gated surface must block, never fall open — also
    when the broken quoting hides inside an ``env -S`` payload.

    :param classifier: The loaded classifier module.
    :param command: Command whose quoting defeats the lexer.
    """
    assert classifier.classify(command) == "unparsable"


def test_classify_unparsable_command_without_mention_returns_empty(
    classifier: ModuleType,
) -> None:
    """An unlexable command with no ``gh pr create`` mention is not gated.

    Blocking every hard-to-lex Bash command would brick unrelated tool use.

    :param classifier: The loaded classifier module.
    """
    assert classifier.classify('echo "unterminated') == ""


def test_classify_ansi_c_quoted_escapes_stay_quoted(classifier: ModuleType) -> None:
    """ANSI-C ``$'...'`` escapes stay balanced even when they contain an escaped quote.

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
