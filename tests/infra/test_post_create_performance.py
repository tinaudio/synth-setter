"""Invariant 4: post-create.sh stays fast (< 30s) because the base image bakes deps.

Primary check: static analysis — the script must not contain slow operations
(apt-get/pip/conda/curl|sh) since those belong in the Dockerfile, not in
per-container post-create. Secondary check: opt-in timing test (skipped
outside the devcontainer).
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

import pytest

POST_CREATE_BUDGET_SECONDS = 30.0

FORBIDDEN_SLOW_OPERATIONS: tuple[tuple[str, str], ...] = (
    (r"\bapt-get\s+install\b", "apt-get install (belongs in Dockerfile)"),
    (r"\bapt\s+install\b", "apt install (belongs in Dockerfile)"),
    (r"(?<!uv\s)\bpip\s+install\b", "pip install (belongs in Dockerfile)"),
    (r"\buv\s+pip\s+install\b", "uv pip install (belongs in Dockerfile)"),
    (r"\bconda\s+install\b", "conda install (belongs in Dockerfile)"),
    (r"\bmamba\s+install\b", "mamba install (belongs in Dockerfile)"),
    (r"\bnpm\s+install\b", "npm install (belongs in Dockerfile)"),
    (r"\bcurl\b[^\n]*\|\s*(sudo\s+)?(bash|sh)\b", "curl ... | sh (belongs in Dockerfile)"),
    (r"\bwget\b[^\n]*\|\s*(sudo\s+)?(bash|sh)\b", "wget ... | sh (belongs in Dockerfile)"),
)


def _resolve_git_hooks_dir(project_root: Path) -> Path | None:
    """Resolve the effective git hooks directory, honoring worktree pointer files."""
    try:
        result = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["git", "rev-parse", "--git-path", "hooks"],  # noqa: S607 — git on PATH
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    hooks_path = Path(result.stdout.strip())
    if not hooks_path.is_absolute():
        hooks_path = project_root / hooks_path
    return hooks_path


def _iter_script_lines_without_comments(text: str) -> list[tuple[int, str]]:
    """Return (line_number, line) pairs with full-line shell comments dropped.

    Only `^\\s*#…` lines are treated as comments. Mid-line `#` is left intact — bash parameter
    expansion uses `${VAR#prefix}` (and post-create.sh's PAT quote-strip relies on it), so a global
    "strip from first #" rule would truncate those lines and let slow-op patterns hide after the
    `#`.
    """
    out: list[tuple[int, str]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if re.match(r"^\s*#", raw) or not raw.strip():
            continue
        out.append((lineno, raw))
    return out


@pytest.mark.infra
def test_post_create_has_no_slow_operations_for_performance(
    post_create_script: Path,
) -> None:
    """Static check: post-create.sh contains no install-heavy commands."""
    text = post_create_script.read_text()
    offenders: list[str] = []
    for lineno, line in _iter_script_lines_without_comments(text):
        for pattern, description in FORBIDDEN_SLOW_OPERATIONS:
            if re.search(pattern, line):
                offenders.append(f"  line {lineno}: {line.strip()!r} → {description}")
    assert not offenders, (
        f"{post_create_script.name} contains slow operations that should live in the "
        f"Dockerfile (image-bake time), not in per-container post-create:\n" + "\n".join(offenders)
    )


@pytest.mark.infra
def test_post_create_completes_under_budget_for_performance(
    post_create_script: Path,
) -> None:
    """Integration timing: opt-in, runs only inside a devcontainer where it's safe.

    Skipped when:
      - not inside a devcontainer (no /.dockerenv and no REMOTE_CONTAINERS)
      - the script has already run successfully (pre-commit hook installed),
        because rerunning side-effects in a populated workspace is destructive
    """
    if not Path("/.dockerenv").exists() and not os.environ.get("REMOTE_CONTAINERS"):
        pytest.skip("not running inside a devcontainer; timing test is opt-in")

    project_root = post_create_script.resolve().parent.parent
    hooks_dir = _resolve_git_hooks_dir(project_root)
    if hooks_dir is not None and (hooks_dir / "pre-commit").exists():
        pytest.skip("post-create.sh already ran (pre-commit hook present); skipping rerun")

    start = time.perf_counter()
    result = subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["bash", str(post_create_script)],  # noqa: S607 — bash on PATH
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=int(POST_CREATE_BUDGET_SECONDS * 2),
        check=False,
    )
    elapsed = time.perf_counter() - start

    assert result.returncode == 0, (
        f"post-create.sh exited with {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert elapsed < POST_CREATE_BUDGET_SECONDS, (
        f"post-create.sh took {elapsed:.1f}s, budget is {POST_CREATE_BUDGET_SECONDS}s"
    )
