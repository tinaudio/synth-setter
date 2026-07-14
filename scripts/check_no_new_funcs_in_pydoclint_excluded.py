"""CI guard against new ``def``/``class`` lines in files skipped by pydoclint's exclude list.

Pydoclint's per-file exclude list closes the lint surface for legacy files,
but it also silently accepts new functions added to those files — the
adversarial probe on #939 documented this as blind spot P6. This guard
makes the exclude list a one-way ratchet: a file stays excluded only
until someone wants to clean it up; nobody can grow it.

Wired into the PR-time ``code-quality-pr`` workflow. Fails the job with a
non-zero exit if the PR diff adds any ``+def``/``+class`` line whose file
matches the pydoclint exclude regex. Only top-level (zero-indent) and
method-level (four-space indent) declarations count — anything more
deeply indented is a nested closure and ignored. The threshold lives in
``DEF_OR_CLASS_PATTERN`` (``{0,4}`` spaces).

See PR for the test suite and #938 for the audit context.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_DEFAULT = REPO_ROOT / "pyproject.toml"

DEF_OR_CLASS_PATTERN = re.compile(
    r"^\+(?P<indent> {0,4})(?:async\s+)?(?:def|class)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)
FILE_HEADER_PATTERN = re.compile(r"^\+\+\+ b/(?P<path>.+)$")
HUNK_HEADER_PATTERN = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,\d+)? @@")


def load_exclude_regex(pyproject_path: Path) -> re.Pattern[str]:  # noqa: DOC502
    """Read ``[tool.pydoclint].exclude`` from ``pyproject_path`` and compile it.

    :param pyproject_path: Path to the pyproject.toml whose pydoclint exclude regex is the
        single source of truth for which files this guard considers excluded.
    :returns: A compiled regex pattern identical to the one pydoclint itself uses.
    :rtype: re.Pattern[str]
    :raises KeyError: If the pydoclint section or its ``exclude`` key is absent. Guarding
        against a silently absent regex matters more than guarding against a typo, because
        an absent regex would let the guard pass-through on every file. (Raised by
        ``tomllib`` via dict indexing — pydoclint DOC502 doesn't see indirect raises.)
    """
    with pyproject_path.open("rb") as fh:
        cfg = tomllib.load(fh)
    return re.compile(cfg["tool"]["pydoclint"]["exclude"])


def find_new_defs_in_excluded(
    diff_text: str, exclude_regex: re.Pattern[str]
) -> list[tuple[str, str, int]]:
    """Scan a unified diff for new ``def``/``class`` declarations in excluded files.

    :param diff_text: Raw unified diff output (the kind ``git diff`` produces).
    :param exclude_regex: Compiled regex against which each touched file is tested.
        Files matching the regex are considered excluded.
    :returns: List of ``(path, name, line_number)`` tuples — one per offending declaration —
        in the order they appear in the diff. Line numbers reference the post-change file.
    :rtype: list[tuple[str, str, int]]
    """
    findings: list[tuple[str, str, int]] = []
    current_path: str | None = None
    current_line: int | None = None
    in_excluded_file = False
    for raw_line in diff_text.splitlines():
        header = FILE_HEADER_PATTERN.match(raw_line)
        if header:
            path = header["path"]
            current_path = path
            in_excluded_file = bool(exclude_regex.search(path))
            current_line = None
            continue
        if not in_excluded_file or current_path is None:
            continue
        hunk = HUNK_HEADER_PATTERN.match(raw_line)
        if hunk:
            current_line = int(hunk["start"])
            continue
        if raw_line.startswith("+++") or raw_line.startswith("---"):
            continue
        if raw_line.startswith("+"):
            match = DEF_OR_CLASS_PATTERN.match(raw_line)
            if match and current_line is not None:
                findings.append((current_path, match["name"], current_line))
            if current_line is not None:
                current_line += 1
            continue
        if raw_line.startswith("-"):
            continue
        # "\ No newline at end of file" is a diff metadata marker, not a real file line —
        # don't count it toward current_line or it would skew the reported line numbers.
        if raw_line.startswith("\\"):
            continue
        if current_line is not None:
            current_line += 1
    return findings


def _read_diff_from_git(base_ref: str) -> str:  # noqa: DOC502
    """Run ``git diff <base>...HEAD`` and return its stdout.

    :param base_ref: Reference to diff against. Typically the PR's base SHA on origin/main.
    :returns: The raw unified diff produced by git.
    :rtype: str
    :raises subprocess.CalledProcessError: If git exits non-zero. Letting the error propagate is
        correct — a broken git invocation is a setup bug, not a clean run. (Raised by
        ``subprocess.run(..., check=True)`` — pydoclint DOC502 doesn't see indirect raises.)
    """
    result = subprocess.run(  # noqa: S603 — args composed from a CLI string in a CI runner
        ["git", "diff", "--unified=0", f"{base_ref}...HEAD"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return result.stdout


def run(
    diff_text: str,
    pyproject_path: Path = PYPROJECT_DEFAULT,
) -> int:
    """Apply the guard to ``diff_text``; return a process exit code.

    :param diff_text: The unified diff to scan. Tests pass synthetic input; the CLI
        invocation derives it from ``git diff``.
    :param pyproject_path: Path to pyproject.toml. The pydoclint exclude regex is read from here.
    :returns: 0 if the diff is clean; 1 if it adds any ``def``/``class`` in an excluded file.
        Findings (when present) are written to stdout in ``path:line: name`` form so a CI
        viewer can click straight to them.
    :rtype: int
    """
    exclude_regex = load_exclude_regex(pyproject_path)
    findings = find_new_defs_in_excluded(diff_text, exclude_regex)
    if not findings:
        return 0
    print(  # noqa: T201 — CLI tool: stdout is its product, not a debug print
        "New top-level def/class added to a pydoclint-excluded file. Either remove the file "
        "from [tool.pydoclint].exclude (preferred) or revert the addition. See #938.\n"
    )
    for path, name, line in findings:
        print(f"{path}:{line}: {name}")  # noqa: T201 — see above
    return 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments.

    :param argv: Argument list, typically ``sys.argv[1:]``.
    :returns: Namespace with ``base`` (the diff base ref) and ``pyproject`` (path override).
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="origin/main",
        help="Git ref to diff against (default: origin/main).",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=PYPROJECT_DEFAULT,
        help="Path to pyproject.toml (default: repo-root pyproject.toml).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Reads the diff from git, runs the guard, returns its exit code.

    :param argv: Argument list, defaults to ``sys.argv[1:]``.
    :returns: Process exit code: 0 for a clean diff, 1 for any finding.
    :rtype: int
    """
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    diff_text = _read_diff_from_git(args.base)
    return run(diff_text=diff_text, pyproject_path=args.pyproject)


if __name__ == "__main__":
    raise SystemExit(main())
