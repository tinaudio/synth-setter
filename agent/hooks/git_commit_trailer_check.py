"""Forbidden-flag / trailer scanner for the git-commit-trailer-check hook.

Reads a single ``git commit`` command on stdin; prints one
``<label>\\t<match>`` line per forbidden finding (``--no-verify`` / bundled
``-n`` short flags, ``Co-Authored-By:`` trailers, agent-attribution footers).
Empty stdout means clean.

The .sh wrapper handles JSON parsing, re-scope, and the user-facing BLOCKED
message; this module is the pure-Python parser so unit-test-style argv slicing
can avoid confusing ``-n`` on ``git commit`` with a downstream ``grep -n``.
"""

from __future__ import annotations

import pathlib
import re
import shlex
import sys
from collections.abc import Iterator

# Trailer regexes are anchored to trailer context — a line-start (real ``\n``
# or shlex-escaped literal ``\\n``) followed by optional whitespace — so a
# subject like "feat: docs generated with sphinx-build" does not match. Group
# 1 captures the trailer key for clean reporting.
_CO_AUTHORED_BY = re.compile(r"(?:^|\\n|\n)\s*(Co-Authored-By:)", re.IGNORECASE)
_GENERATED_WITH = re.compile(r"(?:^|\\n|\n)\s*(Generated with)", re.IGNORECASE)
_CLAUDE_ATTRIBUTION = re.compile(
    r"(?:^|\\n|\n)\s*[A-Za-z-]+:\s*Claude\s+(?:Code|Opus|Sonnet|Haiku)\b"
)
_NOREPLY_EMAIL = re.compile(r"\bnoreply@anthropic\.com\b")

_TRAILER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Co-Authored-By trailer", _CO_AUTHORED_BY),
    ("agent-attribution footer", _GENERATED_WITH),
    ("agent-attribution footer", _CLAUDE_ATTRIBUTION),
    ("agent-attribution footer", _NOREPLY_EMAIL),
)

_METACHARS = frozenset({"&&", "||", ";", "|", "&"})


def _tokenize(cmd: str) -> list[str]:
    """Tokenize ``cmd`` like a POSIX shell, splitting on metachars.

    :param cmd: Raw shell command.
    :returns: Token list. On unbalanced quotes, falls back to a metachar-aware
        ``re.split`` so downstream argv slicing still terminates at ``&&`` etc.
    """
    try:
        lexer = shlex.shlex(cmd, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except ValueError:
        return [
            token for token in re.split(r"(\&\&|\|\||[;|&])", cmd) if token and not token.isspace()
        ]


def _commit_arg_slices(tokens: list[str]) -> Iterator[tuple[int, int]]:
    """Yield each ``[start, end)`` slice of tokens belonging to a ``git commit`` argv.

    A slice ends at a shell metachar token (``&&`` / ``||`` / ``;`` / ``|`` /
    ``&``) or at end-of-tokens, so a downstream ``grep -n`` after ``&&`` is not
    mistaken for ``git commit -n``.

    :param tokens: Tokenized command.
    :yields: ``(start, end)`` index pairs into ``tokens``.
    :ytype: tuple[int, int]
    """
    i = 0
    while i < len(tokens) - 1:
        if tokens[i] == "git" and tokens[i + 1] == "commit":
            start = i + 2
            j = start
            while j < len(tokens) and tokens[j] not in _METACHARS:
                j += 1
            yield start, j
            i = j
        else:
            i += 1


def _iter_commit_argvs(tokens: list[str]) -> Iterator[list[str]]:
    """Yield each argv (token sub-list) belonging to a ``git commit`` invocation.

    :param tokens: Tokenized command.
    :yields: Each ``git commit`` argv slice.
    :ytype: list[str]
    """
    for start, end in _commit_arg_slices(tokens):
        yield tokens[start:end]


def _has_no_verify_short_flag(argv: list[str]) -> bool:
    """Return True if ``argv`` has ``-n`` or any bundled short flag containing ``n``.

    Short flags cluster: ``git commit -nm "msg"`` tokenizes as
    ``["-nm", "msg"]``, so a bare ``"-n" in argv`` misses it. Stops scanning at
    the ``--`` end-of-options marker, and skips long flags (``--no-foo``) and
    positional args.

    :param argv: ``git commit`` argv slice.
    :returns: True if a no-verify-equivalent short flag is present.
    """
    for tok in argv:
        if tok == "--":
            return False
        if tok.startswith("--"):
            continue
        if len(tok) >= 2 and tok[0] == "-" and "n" in tok[1:] and tok[1:].isalpha():
            return True
    return False


def _collect_message_texts(argv: list[str]) -> list[str]:
    """Extract every commit-message body from one ``git commit`` argv slice.

    Handles ``-m``/``--message`` and ``-F``/``--file`` in both space-separated
    and ``=``-form (e.g. ``--message=value``). ``-F`` paths are read from disk;
    OSError is silently swallowed (the offender is presumed already inspected
    via the raw-command fallback).

    :param argv: ``git commit`` argv slice.
    :returns: List of message-body strings extracted from ``argv``.
    """
    texts: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("-m", "--message") and i + 1 < len(argv):
            texts.append(argv[i + 1])
            i += 2
            continue
        if tok.startswith("--message="):
            texts.append(tok.split("=", 1)[1])
            i += 1
            continue
        if tok in ("-F", "--file") and i + 1 < len(argv):
            try:
                texts.append(pathlib.Path(argv[i + 1]).read_text())
            except OSError:
                pass
            i += 2
            continue
        if tok.startswith("--file="):
            try:
                texts.append(pathlib.Path(tok.split("=", 1)[1]).read_text())
            except OSError:
                pass
            i += 1
            continue
        i += 1
    return texts


def _scan(cmd: str) -> list[tuple[str, str]]:
    """Return de-duplicated ``(label, match)`` findings for ``cmd``.

    :param cmd: Raw ``git commit`` shell command.
    :returns: Findings in discovery order; empty if clean.
    """
    tokens = _tokenize(cmd)
    findings: list[tuple[str, str]] = []

    # --no-verify / -n (incl. bundled `-nm` / `-anm` / …) on `git commit` itself.
    # Iterate all argvs so chained invocations are all checked, but record only
    # the first hit per kind to keep the output tight.
    for argv in _iter_commit_argvs(tokens):
        if "--no-verify" in argv:
            findings.append(("--no-verify flag", "--no-verify"))
            break
    for argv in _iter_commit_argvs(tokens):
        if _has_no_verify_short_flag(argv):
            findings.append(("-n flag (== --no-verify)", "-n"))
            break

    # Forbidden trailers inside -m / -F / heredoc bodies. When we cannot find
    # any explicit message body (heredoc / shell-substituted), fall back to the
    # raw command — but with anchored trailer regexes only, since substring
    # matches against arbitrary command text produce false positives. The
    # raw_command_fallback flag makes that trust-level shift explicit.
    texts: list[str] = []
    for argv in _iter_commit_argvs(tokens):
        texts.extend(_collect_message_texts(argv))
    raw_command_fallback = not texts
    if raw_command_fallback:
        texts = [cmd]

    for text in texts:
        for label, pattern in _TRAILER_PATTERNS:
            m = pattern.search(text)
            if m:
                findings.append((label, m.group(0)))
                break

    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    for label, match in findings:
        key = (label, match)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((label, match))
    return deduped


def main() -> None:
    """Read the command on stdin and print one ``<label>\\t<match>`` line per finding."""
    cmd = sys.stdin.read()
    for label, match in _scan(cmd):
        print(f"{label}\t{match}")  # noqa: T201 -- stdout is the contract with the shell wrapper


if __name__ == "__main__":
    main()
