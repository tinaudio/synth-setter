"""Shared helpers for the pre-PR review-gate sentinel filename.

The sentinel encodes the commit SHA the review was performed against directly
in the review file's name, e.g.::

    .agent-reviews/repo-review-full-no-comments.<40-char-sha>.md

Both ``/repo-review-full-no-comments`` (when writing the rendered report) and
``agent/hooks/pre-pr-review-gate.sh`` (when validating the path supplied via
``REVIEW_FULL=<path>`` on ``gh pr create``) call into this module so the file
name format has exactly one source of truth.

Stdlib-only so the bash gate can ``python3 review_sentinel.py parse <path>``
without project deps on PATH.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Sequence

REVIEW_DIR = ".agent-reviews"
SKILL_PREFIX = "repo-review-full-no-comments"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_FILENAME_RE = re.compile(rf"^{re.escape(SKILL_PREFIX)}\.([0-9a-f]{{40}})\.md$")
_SUBCOMMANDS = frozenset({"make", "parse", "path"})
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

    Subcommands: ``make <sha>`` prints the filename; ``parse <path>`` prints
    the encoded SHA (or exits 1 if the path is not a sentinel); ``path <sha>``
    prints ``<REVIEW_DIR>/<filename>``.

    :param argv: Argument list, normally ``sys.argv``.
    :returns: Process exit code (0 success; 1 parse no-match; 2 usage/ValueError).
    """
    if len(argv) < 3 or argv[1] not in _SUBCOMMANDS:
        sys.stderr.write(f"{_USAGE}\n")
        return 2
    command, arg = argv[1], argv[2]
    try:
        if command == "make":
            sys.stdout.write(make_review_filename(arg) + "\n")
        elif command == "path":
            sys.stdout.write(make_review_path(arg) + "\n")
        else:
            sha = parse_review_filename(arg)
            if sha is None:
                return 1
            sys.stdout.write(sha + "\n")
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
