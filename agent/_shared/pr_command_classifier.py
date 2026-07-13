"""Classify how a Bash command relates to ``gh pr create`` for the pre-PR gate.

``agent/hooks/pre-pr-review-gate.sh`` calls the CLI on every Bash tool command
to decide whether the review gate applies. The classification contract:

- ``direct`` — the command executes ``gh pr create`` as its own argv, possibly
  behind benign prefixes (``env``, ``sudo``, ``nice``, ``time``, ``exec``,
  ``command``, ``builtin``, ``eval``, leading ``VAR=val`` assignments) — the
  gate's sentinel checks apply.
- ``wrapped`` — the creation is smuggled through a shell (``bash -c``,
  here-string/heredoc stdin, ``env -S``) — the gate blocks outright.
- ``unparsable`` — the command mentions ``gh pr create`` but cannot be lexed
  — the gate blocks (fail closed on the gated surface, never fall open).
- ``""`` — anything else, including quoted prose that merely mentions
  ``gh pr create`` — not gated.

Stdlib-only so the bash gate can run ``python3 pr_command_classifier.py
"<command>"`` without project deps on PATH.
"""

from __future__ import annotations

import codecs
import os
import re
import shlex
import sys
from collections.abc import Sequence

SHELLS = frozenset({"sh", "bash", "dash", "ksh", "zsh"})
PREFIXES = frozenset({"builtin", "command", "env", "eval", "exec", "nice", "sudo", "time"})
OPTIONS_WITH_VALUES = {
    "env": frozenset({"-C", "--chdir", "-S", "--split-string", "-u", "--unset"}),
    "exec": frozenset({"-a"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "sudo": frozenset(
        {
            "-C",
            "-D",
            "-g",
            "-h",
            "-p",
            "-r",
            "-R",
            "-t",
            "-T",
            "-u",
            "--chdir",
            "--close-from",
            "--command-timeout",
            "--group",
            "--host",
            "--other-user",
            "--prompt",
            "--role",
            "--type",
            "--user",
        }
    ),
    "time": frozenset({"-f", "--format", "-o", "--output"}),
}
# Backtick included: legacy command substitution executes its content, so it
# opens a segment exactly like `;` or `(` — quoted backticks stay in-token.
_PUNCTUATION = ";|&(){}`"
_ANSI_C_QUOTE_RE = re.compile(r"\$'((?:\\.|[^'\\])*)'")
_MENTIONS_PR_CREATE_RE = re.compile(r"gh\s+pr\s+create")


def _normalize_ansi_c_quotes(command: str) -> str:
    """Rewrite ``$'...'`` spans as plain quoted strings the lexer understands.

    Escapes are decoded (``\\'`` no longer unbalances the rewritten string)
    and the result re-quoted with :func:`shlex.quote`.

    :param command: Raw Bash command text.
    :returns: The command with each ANSI-C span replaced by a safe equivalent.
    """
    return _ANSI_C_QUOTE_RE.sub(
        lambda match: shlex.quote(codecs.decode(match.group(1), "unicode_escape")),
        command,
    )


def _command_segments(command: str) -> list[list[str]]:  # noqa: DOC502
    """Split a command into token lists, one per shell control segment.

    :param command: Raw Bash command text.
    :returns: Token lists split on ``;``, ``|``, ``&``, ``()``/``{}``, and backticks; comments
        dropped.
    :raises ValueError: If the command cannot be lexed (unbalanced quoting).
    """
    command = command.replace("\\\n", "")
    command = _normalize_ansi_c_quotes(command)
    lexer = shlex.shlex(command, posix=True, punctuation_chars=_PUNCTUATION)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    segments: list[list[str]] = [[]]
    for token in lexer:
        if all(char in _PUNCTUATION for char in token):
            segments.append([])
        else:
            segments[-1].append(token)
    return segments


def _skip_assignments(tokens: list[str], index: int = 0, *, empty_name_ok: bool = False) -> int:
    """Return the index of the first token that is not a ``VAR=val`` assignment.

    :param tokens: One segment's tokens.
    :param index: Position to start scanning from.
    :param empty_name_ok: Accept ``=val`` tokens too. Bash rejects an empty
        assignment name (the token becomes a command), but GNU ``env`` accepts
        one and still execs its trailing command.
    :returns: Index of the first non-assignment token (may be ``len(tokens)``).
    """
    while (
        index < len(tokens)
        and "=" in tokens[index]
        and (empty_name_ok or not tokens[index].startswith("="))
    ):
        index += 1
    return index


def _option_end(tokens: list[str], index: int, prefix: str) -> int:
    """Return the index just past ``prefix``'s options (and their values).

    :param tokens: One segment's tokens.
    :param index: Position of the first candidate option token.
    :param prefix: The prefix command the options belong to (``sudo``, ...).
    :returns: Index of the first non-option token.
    """
    takes_value = OPTIONS_WITH_VALUES.get(prefix, frozenset())
    while index < len(tokens) and tokens[index].startswith("-"):
        option = tokens[index]
        index += 1
        if option == "--":
            break
        if option in takes_value:
            index += 1
    return index


def _executable_index(tokens: list[str]) -> int:
    """Return the index of the token that names the executable actually run.

    Skips leading assignments, benign prefix commands with their options and
    (for ``env``) its own ``VAR=val`` arguments, and ``then``/``do`` keywords.

    :param tokens: One segment's tokens.
    :returns: Index of the effective executable (may be ``len(tokens)``).
    """
    index = _skip_assignments(tokens)
    while index < len(tokens) and os.path.basename(tokens[index]) in PREFIXES:
        prefix = os.path.basename(tokens[index])
        index += 1
        if prefix == "builtin":
            continue
        index = _option_end(tokens, index, prefix)
        if prefix == "env":
            index = _skip_assignments(tokens, index, empty_name_ok=True)
    while index < len(tokens) and tokens[index] in {"then", "do"}:
        index += 1
    return index


def _env_split_string(tokens: list[str]) -> str | None:
    """Return the payload of an ``env -S/--split-string`` invocation, if any.

    :param tokens: One segment's tokens.
    :returns: The split-string payload, or ``None`` when the segment is not an
        ``env -S`` call.
    """
    index = _skip_assignments(tokens)
    if index >= len(tokens) or os.path.basename(tokens[index]) != "env":
        return None
    index += 1
    while index < len(tokens) and tokens[index].startswith("-"):
        option = tokens[index]
        index += 1
        if option == "--":
            return None
        if option in {"-S", "--split-string"} and index < len(tokens):
            return tokens[index]
        if option in OPTIONS_WITH_VALUES["env"]:
            index += 1
    return None


def _shell_script_payloads(arguments: list[str]) -> list[str]:
    """Collect script text a shell invocation would execute.

    Covers ``-c``/``--command`` (separate, ``--``-separated, or ``=``-fused
    values) plus stdin feeds: ``<<<`` here-strings and ``<<`` heredocs, whose
    body words trail in the same segment.

    :param arguments: Tokens following the shell executable in its segment.
    :returns: Every candidate script payload found (possibly empty).
    """
    payloads = []
    for position, word in enumerate(arguments):
        if word == "--":
            continue
        if word.startswith("--command="):
            payloads.append(word[len("--command=") :])
        elif word == "--command" or (
            word.startswith("-") and not word.startswith("--") and "c" in word[1:]
        ):
            value_index = position + 1
            if value_index < len(arguments) and arguments[value_index] == "--":
                value_index += 1
            if value_index < len(arguments):
                payloads.append(arguments[value_index])
        elif word.startswith("<<<"):
            here_string = word[len("<<<") :]
            if not here_string and position + 1 < len(arguments):
                here_string = arguments[position + 1]
            payloads.append(here_string)
        elif word.startswith("<<"):
            payloads.append(" ".join(arguments[position + 1 :]))
    return payloads


def _pr_create_mode(command: str) -> str:  # noqa: DOC502
    """Classify one (possibly nested) command string.

    :param command: Bash command text.
    :returns: ``direct``, ``wrapped``, or ``""``.
    :raises ValueError: If the command cannot be lexed.
    """
    for tokens in _command_segments(command):
        split_string = _env_split_string(tokens)
        if split_string is not None:
            nested_mode = _pr_create_mode(split_string)
            if nested_mode:
                return nested_mode
        index = _executable_index(tokens)
        if index >= len(tokens):
            continue
        executable = os.path.basename(tokens[index])
        if executable in SHELLS:
            for payload in _shell_script_payloads(tokens[index + 1 :]):
                if _pr_create_mode(payload):
                    return "wrapped"
        elif executable == "gh" and tokens[index + 1 : index + 3] == ["pr", "create"]:
            return "direct"
    return ""


def classify(command: str) -> str:
    """Classify a Bash command's relationship to ``gh pr create``.

    Never raises: a command that defeats the lexer is ``unparsable`` when it
    mentions ``gh pr create`` (the gate blocks it) and ``""`` otherwise, so a
    parser gap can only over-block the gated surface, never fall open.

    :param command: Raw Bash command text.
    :returns: ``direct``, ``wrapped``, ``unparsable``, or ``""``.
    """
    try:
        return _pr_create_mode(command)
    except (ValueError, RecursionError):
        return "unparsable" if _MENTIONS_PR_CREATE_RE.search(command) else ""


def _main(argv: Sequence[str]) -> int:
    """Tiny CLI so the bash gate can classify without Python imports.

    :param argv: Argument list, normally ``sys.argv``; ``argv[1]`` is the
        command to classify.
    :returns: Process exit code (0 success; 2 usage error).
    """
    if len(argv) != 2:
        sys.stderr.write("usage: pr_command_classifier.py <command>\n")
        return 2
    sys.stdout.write(classify(argv[1]) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
