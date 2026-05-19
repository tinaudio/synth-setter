"""Forbidden-flag / trailer scanner for the git-commit-trailer-check hook.

Reads a shell command on stdin (any ``git ...`` invocation under the
``Bash(git*)`` matcher); prints one ``<label>\\t<match>`` line per forbidden
finding (``--no-verify`` / bundled ``-n`` short flags, ``Co-Authored-By:``
trailers, agent-attribution footers). Empty stdout means clean — including
non-``git commit`` invocations, which reach the scanner because the wrapper
no longer pre-filters.

The .sh wrapper handles JSON parsing and the user-facing BLOCKED message;
this module is the pure-Python parser and the authoritative scope filter
(``_find_commit_subcommand`` yields nothing for non-commit ``git`` calls and
shlex tokenization distinguishes ``-n`` on ``git commit`` from a downstream
``grep -n``).
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
# Match only when an agent-product name follows ``Generated with`` on the
# same line, so bodies like "Generated with sphinx-build" stay legitimate.
_GENERATED_WITH = re.compile(
    r"(?:^|\\n|\n)\s*(Generated with)[^\r\n]{0,80}?"
    r"\b(Claude|Anthropic|Cursor|Copilot|ChatGPT|GPT-?\d|Codex|Gemini|Bard)\b",
    re.IGNORECASE,
)
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


_GIT_LEVEL_OPTION_RE = re.compile(r"^(-[a-zA-Z]|--[a-zA-Z][a-zA-Z0-9_-]*)(=.*)?$")
_GIT_LEVEL_VALUE_OPTS = frozenset({"-c", "-C", "--git-dir", "--work-tree", "--namespace"})


def _find_commit_subcommand(tokens: list[str], git_idx: int) -> int | None:
    """Return the index of the ``commit`` token following ``git`` at ``git_idx``.

    Skips ``git``-level options between ``git`` and the subcommand (``-c key=val``,
    ``--git-dir=.git``, ``--work-tree path``, etc.) so ``git -c gpg.sign=false commit``
    is still recognised. Returns ``None`` if no ``commit`` subcommand is found
    before a metachar or end-of-tokens.

    :param tokens: Tokenized command.
    :param git_idx: Index of the ``git`` token.
    :returns: Index of the ``commit`` subcommand, or ``None``.
    """
    j = git_idx + 1
    while j < len(tokens) and tokens[j] not in _METACHARS:
        tok = tokens[j]
        if tok == "commit":
            return j
        if not _GIT_LEVEL_OPTION_RE.match(tok):
            return None
        if tok in _GIT_LEVEL_VALUE_OPTS and j + 1 < len(tokens):
            # Space-separated value form (e.g. `-c key=val`, `--git-dir .git`).
            j += 2
            continue
        j += 1
    return None


def _commit_arg_slices(tokens: list[str]) -> Iterator[tuple[int, int]]:
    """Yield each ``[start, end)`` slice of tokens belonging to a ``git commit`` argv.

    A slice ends at a shell metachar token (``&&`` / ``||`` / ``;`` / ``|`` /
    ``&``) or at end-of-tokens, so a downstream ``grep -n`` after ``&&`` is not
    mistaken for ``git commit -n``. ``git`` options between ``git`` and
    ``commit`` (e.g. ``git -c gpg.sign=false commit``) are skipped.

    :param tokens: Tokenized command.
    :yields: ``(start, end)`` index pairs into ``tokens``.
    :ytype: tuple[int, int]
    """
    i = 0
    while i < len(tokens):
        if tokens[i] == "git":
            commit_idx = _find_commit_subcommand(tokens, i)
            if commit_idx is not None:
                start = commit_idx + 1
                j = start
                while j < len(tokens) and tokens[j] not in _METACHARS:
                    j += 1
                yield start, j
                i = j
                continue
        i += 1


def _iter_commit_argvs(tokens: list[str]) -> Iterator[list[str]]:
    """Yield each argv (token sub-list) belonging to a ``git commit`` invocation.

    :param tokens: Tokenized command.
    :yields: Each ``git commit`` argv slice.
    :ytype: list[str]
    """
    for start, end in _commit_arg_slices(tokens):
        yield tokens[start:end]


# ``git commit`` short flags whose value attaches inline (``-S<keyid>``,
# ``-C<commit>``, ``-c<commit>``, ``-F<file>``, ``-m<msg>``). When the cluster
# starts with one of these, the remainder is the value — not additional
# bundled flags — so ``n`` in the value (``-Sandykey``) is not a ``-n``.
# ``-u``/``-t`` are intentionally excluded: their value is optional and they
# are overwhelmingly used bare (``-u`` == ``--untracked-files``), so a
# hand-typed ``-un`` is far more likely to mean ``-u -n`` than ``-u`` with
# value ``n`` — better to keep the fail-closed default and block.
_VALUE_ATTACHING_SHORT_FLAGS = frozenset({"S", "C", "c", "F", "m"})


def _has_no_verify_short_flag(argv: list[str]) -> bool:
    """Return True if ``argv`` has ``-n`` or any bundled short flag containing ``n``.

    Short flags cluster: ``git commit -nm "msg"`` tokenizes as
    ``["-nm", "msg"]``, so a bare ``"-n" in argv`` misses it. Stops scanning at
    the ``--`` end-of-options marker, and skips long flags (``--no-foo``) and
    positional args. When a cluster begins with a value-attaching short flag
    (``-S<keyid>``, ``-C<commit>``, etc.), only the first letter is treated as
    a flag — the rest is its value.

    :param argv: ``git commit`` argv slice.
    :returns: True if a no-verify-equivalent short flag is present.
    """
    for tok in argv:
        if tok == "--":
            return False
        if tok.startswith("--"):
            continue
        if len(tok) < 2 or tok[0] != "-":
            continue
        cluster = tok[1:]
        if not cluster.isalpha():
            continue
        if cluster[0] in _VALUE_ATTACHING_SHORT_FLAGS:
            cluster = cluster[0]
        if "n" in cluster:
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

    commit_slices = list(_iter_commit_argvs(tokens))
    if not commit_slices:
        # No ``git commit`` invocation in this command — nothing to gate.
        # We must NOT raw-command-fallback the trailer regexes here:
        # `git log --grep="Co-Authored-By: …"` (and similar diagnostic
        # invocations) embed trailer-shaped substrings as flag values, not
        # as outgoing commit body content.
        return findings

    # --no-verify / -n (incl. bundled `-nm` / `-anm` / …) on `git commit` itself.
    # Iterate all argvs so chained invocations are all checked, but record only
    # the first hit per kind to keep the output tight.
    for argv in commit_slices:
        if "--no-verify" in argv:
            findings.append(("--no-verify flag", "--no-verify"))
            break
    for argv in commit_slices:
        if _has_no_verify_short_flag(argv):
            findings.append(("-n flag (== --no-verify)", "-n"))
            break

    # Forbidden trailers inside -m / -F / heredoc bodies. When a commit slice
    # exists but no explicit message body is recoverable (heredoc /
    # shell-substituted), fall back to scanning the raw command — anchored
    # trailer regexes still rule out substring false positives.
    texts: list[str] = []
    for argv in commit_slices:
        texts.extend(_collect_message_texts(argv))
    if not texts:
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
