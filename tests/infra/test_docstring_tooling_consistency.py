"""Pin the docformatter ↔ ruff D205 invariant so the two cannot deadlock.

docformatter's ``wrap-summaries`` reflows an over-length summary onto a second
physical line. ruff's ``D205`` ("1 blank line required between summary line and
description") then reads that continuation as a description with no preceding
blank line and raises an error that no tool can auto-fix (not even
``--unsafe-fixes``) and that docformatter will not undo. The only stable state
is to keep summaries on one physical line: ``wrap-summaries = 0``. This test
fails fast if either side drifts back into the conflicting combination.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def test_docformatter_disables_summary_wrap_while_ruff_enforces_d205(project_root: Path) -> None:
    """While ruff selects D205, docformatter must not wrap summaries.

    :param project_root: Repo root holding ``pyproject.toml`` (from conftest).
    """
    with (project_root / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)
    ruff_select = pyproject["tool"]["ruff"]["lint"]["select"]
    wrap_summaries = pyproject["tool"]["docformatter"]["wrap-summaries"]

    if "D205" in ruff_select:
        assert wrap_summaries == 0, (
            "ruff's D205 rejects multi-line summaries, but docformatter is configured to "
            f"wrap summaries at {wrap_summaries} columns. A summary longer than that wraps "
            "onto a second physical line that D205 flags and no tool can auto-fix. Set "
            "[tool.docformatter] wrap-summaries = 0, or drop D205 from [tool.ruff.lint] select."
        )
