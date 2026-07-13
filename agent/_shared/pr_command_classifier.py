"""Classify how a Bash command relates to ``gh pr create`` for the pre-PR gate.

``agent/hooks/pre-pr-review-gate.sh`` calls the CLI on every Bash tool command
to decide whether the review gate applies. The classification contract:

- ``direct`` — the command executes ``gh pr create`` as its own argv, possibly
  behind a benign prefix (:data:`_PREFIXES`: ``sudo``, ``env``, ``timeout``,
  ...), leading ``VAR=val`` assignments or redirections, or ``gh`` global
  flags — the gate's sentinel checks apply. Heredoc bodies split into
  ordinary command lines, so a smuggled heredoc invocation also lands here.
- ``wrapped`` — the creation is smuggled through a re-parsing layer the gate
  cannot see the argv of: ``bash -c``, ``eval``, ``env -S``, a here-string, or
  piped stdin into a bare shell — the gate blocks outright.
- ``unparsable`` — the command mentions ``gh pr create`` but cannot be lexed
  — the gate blocks (fail closed on the gated surface, never fall open).
- ``""`` — anything else, including quoted prose that merely mentions
  ``gh pr create`` — not gated.

Explicit non-goals: non-shell interpreters (``python3 -c``, ``perl -e``,
``find -exec``, ...), ``source``/``.`` of files or process substitutions, and
pipes into a shell whose upstream text never mentions the recipe — all can
execute anything; the gate is an honor-system floor over the sh-family
wrappers, not a sandbox.

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

_SHELLS = frozenset({"sh", "bash", "dash", "ksh", "zsh"})
_PREFIXES = frozenset(
    {
        "builtin",
        "command",
        "env",
        "exec",
        "nice",
        "nohup",
        "setsid",
        "stdbuf",
        "sudo",
        "time",
        "timeout",
        "xargs",
    }
)
# Reserved words (plus pipeline negation) that may precede the executable at
# command position: `if gh pr create`, `! gh pr create`, `else gh pr create`.
_RESERVED_WORDS = frozenset({"if", "then", "elif", "else", "while", "until", "do", "!"})
_OPTIONS_WITH_VALUES = {
    "env": frozenset({"-C", "--chdir", "-S", "--split-string", "-u", "--unset"}),
    "exec": frozenset({"-a"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "stdbuf": frozenset({"-i", "--input", "-o", "--output", "-e", "--error"}),
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
    "timeout": frozenset({"-k", "--kill-after", "-s", "--signal"}),
    "xargs": frozenset({"-a", "-d", "-E", "-I", "-L", "-n", "-P", "-s"}),
}
# Backtick included: legacy command substitution executes its content, so it
# opens a segment exactly like `;` or `(` — quoted backticks stay in-token.
_PUNCTUATION = ";|&(){}`"
_ANSI_C_QUOTE_RE = re.compile(r"\$'((?:\\.|[^'\\])*)'")
# Keep in sync with the bash fallback grep in agent/hooks/pre-pr-review-gate.sh.
_MENTIONS_PR_CREATE_RE = re.compile(r"gh\s+pr\s+create")
# A redirection word at command position (`2>/dev/null`, `>out`, `<in`).
_REDIRECTION_RE = re.compile(r"^\d*[<>]")
# A bare redirection operator (`>`, `2>`, `>>`) whose target is the NEXT word.
_BARE_REDIRECTION_RE = re.compile(r"^\d*[<>]+$")
# Fused fd-duplication redirections (`2>&1`, `>&2`, `0<&3`): the embedded `&`
# would otherwise open a bogus segment, hiding the executable that follows.
# The target class stops at quotes so blanking a `2>&1` that sits inside a
# quoted string can't swallow the closing quote and desync the lexer.
_FD_DUP_RE = re.compile(r"""\d*[<>]&[^\s;|&()<>`{}'"]*""")


def _normalize_ansi_c_quotes(command: str) -> str:
    """Rewrite ``$'...'`` spans as plain quoted strings the lexer understands.

    Escapes are decoded so an embedded quote can't unbalance the rewritten
    string, then the result is re-quoted with :func:`shlex.quote`.

    :param command: Bash command text that may carry ``$'...'`` spans.
    :returns: The command with each ANSI-C span replaced by a safe equivalent.
    """
    return _ANSI_C_QUOTE_RE.sub(
        lambda match: shlex.quote(codecs.decode(match.group(1), "unicode_escape")),
        command,
    )


def _command_segments(command: str) -> list[tuple[str, list[str]]]:  # noqa: DOC502
    """Split a command into (separator, tokens) pairs, one per control segment.

    :param command: Bash command text (may span multiple lines).
    :returns: One pair per segment split on ``;``, ``|``, ``&``, ``()``/``{}``,
        backticks, and unescaped newlines; ``separator`` is the punctuation run
        that opened the segment (``""`` for the first). Comments dropped.
    :raises ValueError: If the command cannot be lexed (unbalanced quoting).
    """
    command = command.replace("\\\n", "")
    command = _normalize_ansi_c_quotes(command)
    command = _FD_DUP_RE.sub(" ", command)
    # An unescaped newline separates commands like `;`. The real newline is
    # kept (comments must still end at it); the `;` splits the segment.
    command = command.replace("\n", "\n;")
    lexer = shlex.shlex(command, posix=True, punctuation_chars=_PUNCTUATION)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    segments: list[tuple[str, list[str]]] = [("", [])]
    for token in lexer:
        if all(char in _PUNCTUATION for char in token):
            segments.append((token, []))
        else:
            segments[-1][1].append(token)
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
    takes_value = _OPTIONS_WITH_VALUES.get(prefix, frozenset())
    while index < len(tokens) and tokens[index].startswith("-"):
        option = tokens[index]
        index += 1
        if option == "--":
            break
        if option in takes_value:
            index += 1
    return index


def _executable_index(tokens: list[str], *, stop_at: str = "") -> int:
    """Return the index of the token that names the executable actually run.

    Skips, in any order: reserved words at command position (``if``, ``else``,
    ``!``, ...), ``VAR=val`` assignments, redirections, and benign prefix
    commands with their options and (for ``env``) its own ``VAR=val``
    arguments.

    :param tokens: One segment's tokens.
    :param stop_at: Prefix basename to halt on instead of skipping past, so a
        caller can locate that prefix itself behind other prefixes.
    :returns: Index of the effective executable (may be ``len(tokens)``).
    """
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _RESERVED_WORDS:
            index += 1
        elif "=" in token and not token.startswith("="):
            index += 1
        elif _REDIRECTION_RE.match(token):
            index += 1
            if _BARE_REDIRECTION_RE.match(token):
                index += 1  # the spaced redirect-target word
        elif os.path.basename(token) in _PREFIXES:
            prefix = os.path.basename(token)
            if prefix == stop_at:
                break
            index += 1
            if prefix == "builtin":
                continue
            index = _option_end(tokens, index, prefix)
            if prefix == "env":
                index = _skip_assignments(tokens, index, empty_name_ok=True)
            elif prefix == "timeout" and index < len(tokens):
                index += 1  # the positional DURATION argument
        else:
            break
    return index


def _env_split_string(tokens: list[str]) -> str | None:
    """Return the command an ``env -S/--split-string`` invocation would exec.

    GNU ``env -S`` splits the string into words and appends the remaining
    command-line arguments, so the trailing tokens are folded into the
    returned payload.

    :param tokens: One segment's tokens.
    :returns: The reconstructed command text, or ``None`` when the segment is
        not an ``env -S`` call (possibly behind other benign prefixes).
    """
    index = _executable_index(tokens, stop_at="env")
    if index >= len(tokens) or os.path.basename(tokens[index]) != "env":
        return None
    index += 1
    while index < len(tokens) and tokens[index].startswith("-"):
        option = tokens[index]
        index += 1
        if option == "--":
            return None
        if option in {"-S", "--split-string"} and index < len(tokens):
            return " ".join(tokens[index:])
        if option in _OPTIONS_WITH_VALUES["env"]:
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
        # A `-c` anywhere in a single-dash flag cluster (`-lc`, `-norc`).
        is_short_flag_with_c = (
            word.startswith("-") and not word.startswith("--") and "c" in word[1:]
        )
        if word == "--":
            continue
        if word.startswith("--command="):
            payloads.append(word[len("--command=") :])
        elif word == "--command" or is_short_flag_with_c:
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


def _has_pr_create_subcommand(rest: list[str]) -> bool:
    """Report whether ``gh``'s trailing tokens invoke the ``pr create`` subcommand.

    An adjacency scan rather than a flag-table walk, so ``gh`` global flags
    (``-R owner/repo``, ``--repo=...``) before the subcommand still match.
    Quoted prose stays a single token and can never form the adjacent pair.

    :param rest: Tokens following the ``gh`` executable in its segment.
    :returns: True when ``pr`` and ``create`` appear as adjacent bare tokens.
    """
    return any(
        rest[position] == "pr" and rest[position + 1] == "create"
        for position in range(len(rest) - 1)
    )


def _shell_reads_hidden_script(
    segments: list[tuple[str, list[str]]], position: int, index: int
) -> bool:
    """Report whether a shell segment executes script text hidden from the argv.

    Checks explicit payloads (``-c``, here-strings, heredoc remainders) and,
    for a bare shell after a pipe, whether any upstream token mentions the
    recipe — the piped text itself is statically unknowable, so fail closed.

    :param segments: All (separator, tokens) pairs of the command.
    :param position: Index of the shell's segment within ``segments``.
    :param index: Index of the shell executable within its segment's tokens.
    :returns: True when the shell would run smuggled ``gh pr create`` text.
    """
    separator, tokens = segments[position]
    payloads = _shell_script_payloads(tokens[index + 1 :])
    for payload in payloads:
        if _pr_create_mode(payload):
            return True
    if not payloads and separator in {"|", "|&"}:
        upstream_tokens: list[str] = []
        for _, upstream_segment in segments[:position]:
            upstream_tokens.extend(upstream_segment)
        return bool(_MENTIONS_PR_CREATE_RE.search(" ".join(upstream_tokens)))
    return False


def _pr_create_mode(command: str) -> str:  # noqa: DOC502
    """Classify one (possibly nested) command string.

    :param command: Bash command text — the full tool call or an extracted
        wrapper payload being re-classified recursively.
    :returns: ``direct``, ``wrapped``, or ``""``.
    :raises ValueError: If the command cannot be lexed.
    """
    segments = _command_segments(command)
    for position, (_, tokens) in enumerate(segments):
        # env -S re-splits its string argument, hiding words from top-level
        # tokenization — a re-parsing wrapper like eval.
        split_string = _env_split_string(tokens)
        if split_string is not None and _pr_create_mode(split_string):
            return "wrapped"
        index = _executable_index(tokens)
        if index >= len(tokens):
            continue
        executable = os.path.basename(tokens[index])
        if executable in _SHELLS:
            if _shell_reads_hidden_script(segments, position, index):
                return "wrapped"
        elif executable == "eval":
            # eval re-parses its (joined) arguments as shell code.
            if _pr_create_mode(" ".join(tokens[index + 1 :])):
                return "wrapped"
        elif executable == "gh" and _has_pr_create_subcommand(tokens[index + 1 :]):
            return "direct"
    return ""


def classify(command: str) -> str:
    """Classify a Bash command's relationship to ``gh pr create``.

    Never raises: a command that defeats the lexer is ``unparsable`` when it
    mentions ``gh pr create`` (the gate blocks it) and ``""`` otherwise, so a
    parser gap can only over-block the gated surface, never fall open.

    :param command: Full, untrusted Bash tool-call text (may be multi-line).
    :returns: ``direct``, ``wrapped``, ``unparsable``, or ``""``.
    """
    try:
        return _pr_create_mode(command)
    except (ValueError, RecursionError) as exc:
        sys.stderr.write(f"pr_command_classifier: cannot lex command: {exc}\n")
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
