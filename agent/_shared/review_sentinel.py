"""Manage pre-PR review attempts and the review-gate sentinel filename.

The sentinel encodes the commit SHA the review was performed against directly
in the review file's name, e.g.::

    .agent-reviews/repo-review-full-no-comments.<40-char-sha>.md

The no-comments launcher also claims one of three durable per-branch attempts
here before starting Pi. The renderer and retained PR gate share the filename
helpers so their sentinel contract has one source of truth.

Stdlib-only so shell entrypoints can call it without project dependencies.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

MAX_PRE_PR_REVIEW_ATTEMPTS = 3
REVIEW_DIR = ".agent-reviews"
SKILL_PREFIX = "repo-review-full-no-comments"
_ATTEMPT_PREFIX = f"{SKILL_PREFIX}-attempts"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_FILENAME_RE = re.compile(rf"^{re.escape(SKILL_PREFIX)}\.([0-9a-f]{{40}})\.md$")
_SUBCOMMANDS = frozenset({"claim", "findings", "make", "parse", "path"})
_USAGE = f"usage: review_sentinel.py {{{'|'.join(sorted(_SUBCOMMANDS))}}} <arg>"


def make_review_filename(sha: str) -> str:
    """Return the canonical sentinel filename for a commit SHA.

    :param sha: Full 40-character lowercase-hex git SHA to encode.
    :returns: A basename like ``repo-review-full-no-comments.<sha>.md``.
    :raises ValueError: If ``sha`` is not a 40-char lowercase hex string.
    """
    if not _SHA_RE.match(sha):
        raise ValueError(f"expected 40-char lowercase hex SHA, got {sha!r}")
    return f"{SKILL_PREFIX}.{sha}.md"


def claim_review_attempt(branch: str, base_dir: str = REVIEW_DIR) -> int | None:
    """Claim one sentinel review attempt for a branch.

    :param branch: Non-empty Git branch name.
    :param base_dir: Directory holding local review state.
    :returns: Claimed one-based attempt number, or ``None`` after the limit.
    :raises ValueError: If the branch or prior state is invalid.
    """
    if not branch:
        raise ValueError("cannot track review attempts without a branch")
    digest = hashlib.sha256(f"{branch}\n".encode()).hexdigest()
    state_path = Path(base_dir) / f"{_ATTEMPT_PREFIX}.{digest}.txt"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("a+", encoding="utf-8") as state:
        fcntl.flock(state, fcntl.LOCK_EX)
        state.seek(0)
        raw_count = state.read().strip()
        try:
            count = int(raw_count) if raw_count else 0
        except ValueError as exc:
            raise ValueError(f"invalid review attempt state in {state_path}") from exc
        if count >= MAX_PRE_PR_REVIEW_ATTEMPTS:
            return None
        claimed = count + 1
        state.seek(0)
        state.truncate()
        state.write(f"{claimed}\n")
        state.flush()
        os.fsync(state.fileno())
        return claimed


def make_findings_path(base_dir: str | None = None) -> str:
    """Create an isolated findings JSON path for one review invocation.

    :param base_dir: Optional temporary directory; defaults to the platform temporary directory.
    :returns: Absolute path to a newly created empty JSON file.
    """
    file_descriptor, path = tempfile.mkstemp(
        prefix=f"{SKILL_PREFIX}-findings.",
        suffix=".json",
        dir=base_dir,
    )
    os.close(file_descriptor)
    return path


def parse_review_filename(filename: str) -> str | None:
    """Extract the SHA from a sentinel filename, or ``None`` if it doesn't match.

    Never raises; malformed input returns ``None``. Accepts either a bare
    basename or a full path — non-basename components are stripped before
    matching so callers don't have to remember which form to pass.

    :param filename: Basename or full path of a review file.
    :returns: The encoded 40-char SHA, or ``None`` if the basename does not
        follow the sentinel pattern.
    """
    base = os.path.basename(filename)
    match = _FILENAME_RE.match(base)
    return match.group(1) if match else None


def make_review_path(sha: str, base_dir: str = REVIEW_DIR) -> str:  # noqa: DOC502
    """Return the canonical relative path for a sentinel review file.

    :param sha: Full 40-char lowercase-hex commit SHA.
    :param base_dir: Directory under which review files live; defaults to
        :data:`REVIEW_DIR`.
    :returns: Path of the form ``<base_dir>/repo-review-full-no-comments.<sha>.md``.
    :raises ValueError: If ``sha`` is not a 40-char lowercase hex string
        (delegated to :func:`make_review_filename`).
    """
    return os.path.join(base_dir, make_review_filename(sha))


def _main(argv: Sequence[str]) -> int:
    """Tiny CLI so the bash gate can parse/format filenames without Python imports.

    Subcommands: ``claim <branch>`` persists and prints an allowed sentinel
    review attempt (or exits 3 at the limit); ``make <sha>`` prints the
    filename; ``parse <path>`` prints the encoded SHA (or exits 1 if the path
    is not a sentinel); ``path <sha>`` prints ``<REVIEW_DIR>/<filename>``;
    ``findings <dir>`` creates and prints a unique findings JSON path.

    :param argv: Argument list, normally ``sys.argv``.
    :returns: Process exit code (0 success; 1 parse no-match; 2 invalid input).
    """
    if len(argv) < 3 or argv[1] not in _SUBCOMMANDS:
        sys.stderr.write(f"{_USAGE}\n")
        return 2
    command, arg = argv[1], argv[2]
    try:
        if command == "claim":
            attempt = claim_review_attempt(arg)
            if attempt is None:
                sys.stdout.write(f"{MAX_PRE_PR_REVIEW_ATTEMPTS}\n")
                return 3
            sys.stdout.write(f"{attempt} {MAX_PRE_PR_REVIEW_ATTEMPTS}\n")
        elif command == "findings":
            sys.stdout.write(make_findings_path(arg) + "\n")
        elif command == "make":
            sys.stdout.write(make_review_filename(arg) + "\n")
        elif command == "path":
            sys.stdout.write(make_review_path(arg) + "\n")
        else:
            sha = parse_review_filename(arg)
            if sha is None:
                return 1
            sys.stdout.write(sha + "\n")
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
