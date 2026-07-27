"""Title checker for the pr-title-guard hook.

Two CLI modes, stdlib-only so hooks can run it without project deps:

- ``--command``: reads a shell command on stdin, extracts inline
  ``--title``/``-t`` values from direct ``gh pr create`` / ``gh pr edit``
  invocations, and prints one finding per line (empty stdout = clean).
  Checks both the ``.gitlint`` type vocabulary and the release reservation.
- ``--commit-msg-file <path>``: native commit-msg hook entry. Checks only the
  release reservation — gitlint enforces the vocabulary at that stage.

Release-triggering types are read from ``pyproject.toml``'s semantic-release
``minor_tags``/``patch_tags``; the vocabulary from ``.gitlint``. Explicit
release intent is signaled via ``RELEASE_INTENT=1`` — as an env-assignment
prefix on the gated command or in the process environment.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shlex
import sys
import tomllib

# <type>(<optional-scope>)<optional-!>: <description> — type group only.
_TYPE_RE = re.compile(r"^([A-Za-z][A-Za-z-]*)(?:\([^)]*\))?!?: ")
_METACHARS = frozenset({"&&", "||", ";", "|", "&", "(", ")", "\n"})
# Benign single-token prefixes that keep `gh` the effective command word.
_PREFIXES = frozenset({"command", "env", "exec", "nice", "nohup", "setsid", "sudo", "time"})
# Regex-level mention check for the fail-closed path when shlex cannot lex.
_MENTIONS_GATED_RE = re.compile(
    r"\bgh\b.*\bpr\b\s+(create|edit)\b.*(--title|(^|\s)-t\b)", re.DOTALL
)


def _repo_root() -> pathlib.Path:
    """Return the repo root, resolved relative to this file.

    :returns: Path two levels above ``agent/hooks/``.
    """
    return pathlib.Path(__file__).resolve().parents[2]


def load_vocabulary(root: pathlib.Path) -> frozenset[str]:
    """Parse the allowed conventional-commit types from ``.gitlint``.

    :param root: Repo root containing ``.gitlint``.
    :returns: The ``types=`` set of the contrib-title-conventional-commits rule.
    :raises ValueError: If no ``types=`` line is found.
    """
    for line in (root / ".gitlint").read_text().splitlines():
        if line.startswith("types="):
            return frozenset(
                t.strip() for t in line.removeprefix("types=").split(",") if t.strip()
            )
    raise ValueError(".gitlint has no types= line")


def load_release_types(root: pathlib.Path) -> frozenset[str]:
    """Parse the release-triggering types from semantic-release config.

    :param root: Repo root containing ``pyproject.toml``.
    :returns: Union of ``minor_tags`` and ``patch_tags``.
    """
    with (root / "pyproject.toml").open("rb") as fh:
        config = tomllib.load(fh)
    options = config["tool"]["semantic_release"]["commit_parser_options"]
    return frozenset(options["minor_tags"]) | frozenset(options["patch_tags"])


def check_title(
    title: str,
    *,
    vocabulary: frozenset[str],
    release_types: frozenset[str],
    release_intent: bool,
    enforce_vocabulary: bool = True,
) -> list[str]:
    """Check one title/subject; return human-readable findings.

    :param title: The PR title or commit subject.
    :param vocabulary: Allowed conventional-commit types.
    :param release_types: Types that trigger a semantic-release version bump.
    :param release_intent: True when RELEASE_INTENT=1 was signaled.
    :param enforce_vocabulary: False at the commit-msg stage, where gitlint already owns the
        vocabulary check.
    :returns: Findings; empty when the title is clean.
    """
    match = _TYPE_RE.match(title)
    if match is None:
        if enforce_vocabulary:
            return [
                f"title {title!r} is not a conventional commit (<type>(<scope>)?: <description>)"
            ]
        return []
    conv_type = match.group(1)
    if enforce_vocabulary and conv_type not in vocabulary:
        allowed = ", ".join(sorted(vocabulary))
        return [f"off-vocabulary type {conv_type!r} in title {title!r}; allowed: {allowed}"]
    if conv_type in release_types and not release_intent:
        return [
            f"release-triggering type {conv_type!r} in title {title!r}: it cuts a "
            "semantic-release version bump on merge. Use internal-feat:/internal-fix: "
            "for logic PRs, or prefix the command with RELEASE_INTENT=1 if a release "
            "is genuinely intended."
        ]
    return []


def _segments(command: str) -> list[list[str]]:  # noqa: DOC502 -- ValueError raised by shlex iteration
    """Split ``command`` into per-command token segments.

    :param command: Raw shell command text.
    :returns: Token lists, one per simple command.
    :raises ValueError: When the text cannot be lexed (unbalanced quotes).
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    segments: list[list[str]] = [[]]
    for token in lexer:
        if token in _METACHARS:
            segments.append([])
        else:
            segments[-1].append(token)
    return [seg for seg in segments if seg]


def _gated_titles_in_segment(tokens: list[str]) -> tuple[list[str], bool]:
    """Extract gated titles and inline release intent from one command segment.

    :param tokens: shlex tokens of a simple command.
    :returns: ``(titles, release_intent)``; titles is empty when the segment is
        not a direct ``gh pr create|edit``.
    """
    intent = False
    index = 0
    # Assignments and benign prefixes interleave (`env RELEASE_INTENT=1 gh …`),
    # so skip both in one loop rather than two ordered passes.
    while index < len(tokens):
        token = tokens[index]
        if "=" in token and not token.startswith("-"):
            name, _, value = token.partition("=")
            if name == "RELEASE_INTENT" and value == "1":
                intent = True
        elif token not in _PREFIXES:
            break
        index += 1
    if index >= len(tokens) or tokens[index] != "gh":
        return [], intent
    rest = tokens[index + 1 :]
    is_gated = any(
        tok == "pr" and idx + 1 < len(rest) and rest[idx + 1] in ("create", "edit")
        for idx, tok in enumerate(rest)
    )
    if not is_gated:
        return [], intent
    titles: list[str] = []
    for idx, tok in enumerate(rest):
        if tok in ("--title", "-t") and idx + 1 < len(rest):
            titles.append(rest[idx + 1])
        elif tok.startswith("--title="):
            titles.append(tok.removeprefix("--title="))
    return titles, intent


def check_command(command: str, root: pathlib.Path) -> list[str]:
    """Check a shell command's gated PR titles; return findings.

    :param command: Raw shell command text from the tool payload.
    :param root: Repo root for config lookups.
    :returns: Findings; empty when clean or not a gated invocation.
    """
    try:
        segments = _segments(command)
    except ValueError:
        if _MENTIONS_GATED_RE.search(command):
            return ["unparsable command mentions gh pr create/edit with a title — run it directly"]
        return []
    ambient_intent = os.environ.get("RELEASE_INTENT") == "1"
    findings: list[str] = []
    # Config loads are deferred until a gated title is actually found, so the
    # common non-gated command skips the .gitlint/pyproject reads entirely.
    vocabulary: frozenset[str] | None = None
    release_types: frozenset[str] | None = None
    for tokens in segments:
        titles, inline_intent = _gated_titles_in_segment(tokens)
        for title in titles:
            if vocabulary is None or release_types is None:
                vocabulary = load_vocabulary(root)
                release_types = load_release_types(root)
            findings.extend(
                check_title(
                    title,
                    vocabulary=vocabulary,
                    release_types=release_types,
                    release_intent=ambient_intent or inline_intent,
                )
            )
    return findings


def check_commit_msg(path: pathlib.Path, root: pathlib.Path) -> list[str]:
    """Check a commit-message file's subject for the release reservation.

    :param path: The commit-message file (``.git/COMMIT_EDITMSG``).
    :param root: Repo root for config lookups.
    :returns: Findings; empty when clean.
    """
    subject = next(
        (
            line
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ),
        "",
    )
    if not subject:
        return []
    # Vocabulary is unread when enforce_vocabulary=False (gitlint owns it at
    # this stage — including malformed subjects like `feat:x` with no space).
    return check_title(
        subject,
        vocabulary=frozenset(),
        release_types=load_release_types(root),
        release_intent=os.environ.get("RELEASE_INTENT") == "1",
        enforce_vocabulary=False,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint; see module docstring for the two modes.

    :param argv: Argument vector; defaults to ``sys.argv[1:]``.
    :returns: 0 when clean; 1 when a commit-msg check fails. Command mode
        always returns 0 — the wrapper decides on stdout findings.
    """
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--command", action="store_true")
    mode.add_argument("--commit-msg-file", type=pathlib.Path)
    args = parser.parse_args(argv)
    root = _repo_root()
    if args.command:
        for finding in check_command(sys.stdin.read(), root):
            print(finding)  # noqa: T201 -- stdout is the contract with the shell wrapper
        return 0
    findings = check_commit_msg(args.commit_msg_file, root)
    for finding in findings:
        print(f"pr-title-guard: {finding}", file=sys.stderr)  # noqa: T201
    if findings:
        print(  # noqa: T201
            "pr-title-guard: re-run with RELEASE_INTENT=1 to confirm a deliberate release.",
            file=sys.stderr,
        )
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
