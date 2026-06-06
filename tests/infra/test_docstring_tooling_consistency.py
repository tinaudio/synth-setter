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

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def _longest_prefix_len(code: str, selectors: list[str]) -> int:
    """Length of the most specific selector that enables ``code``, or -1 if none.

    Mirrors ruff's prefix matching: a selector enables a rule when it is a
    prefix of the rule code (``"D"``, ``"D2"``, ``"D205"``), and ``"ALL"``
    enables every rule. Specificity is the prefix length, so ``"ALL"`` is the
    least specific (0) and an exact code the most.

    :param code: The rule code to test, e.g. ``"D205"``.
    :param selectors: A ruff select/ignore list.
    :returns: The winning selector's specificity (prefix length, 0 for ``"ALL"``), or -1.
    """
    lengths = [
        0 if sel == "ALL" else len(sel)
        for sel in selectors
        if sel == "ALL" or code.startswith(sel)
    ]
    return max(lengths) if lengths else -1


def _rule_enabled(code: str, select: list[str], ignore: list[str]) -> bool:
    """Whether ``code`` is effectively enabled given ruff select/ignore lists.

    Ruff resolves select-vs-ignore conflicts by specificity: the longer
    (more specific) matching prefix wins, and select wins ties.

    :param code: The rule code to test, e.g. ``"D205"``.
    :param select: The effective select list (``select`` + ``extend-select``).
    :param ignore: The effective ignore list (``ignore`` + ``extend-ignore``).
    :returns: True when ``code`` is selected and no more-specific ignore overrides it.
    """
    selected = _longest_prefix_len(code, select)
    return selected >= 0 and selected >= _longest_prefix_len(code, ignore)


@pytest.mark.parametrize(
    ("select", "ignore", "expected"),
    [
        (["D205"], [], True),  # exact code
        (["D"], [], True),  # family prefix
        (["D2"], [], True),  # partial prefix
        (["ALL"], [], True),  # select-all
        (["E", "F"], [], False),  # unrelated codes only
        ([], [], False),  # nothing selected
        (["D"], ["D205"], False),  # ignore is more specific than select
        (["D205"], ["D"], True),  # select is more specific than ignore
        (["ALL"], ["D205"], False),  # specific ignore overrides select-all
    ],
)
def test_rule_enabled_resolves_d205_through_ruff_prefix_semantics(
    select: list[str], ignore: list[str], expected: bool
) -> None:
    """``_rule_enabled`` matches ruff's prefix + specificity resolution for D205.

    :param select: Effective ruff select list for the case.
    :param ignore: Effective ruff ignore list for the case.
    :param expected: Whether D205 should be reported as enabled.
    """
    assert _rule_enabled("D205", select, ignore) is expected


def test_docformatter_disables_summary_wrap_while_ruff_enforces_d205(project_root: Path) -> None:
    """While ruff enforces D205, docformatter must not wrap summaries.

    :param project_root: Repo root holding ``pyproject.toml`` (from conftest).
    """
    with (project_root / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)
    lint = pyproject["tool"]["ruff"]["lint"]
    select = lint.get("select", []) + lint.get("extend-select", [])
    ignore = lint.get("ignore", []) + lint.get("extend-ignore", [])
    wrap_summaries = pyproject["tool"]["docformatter"]["wrap-summaries"]

    if _rule_enabled("D205", select, ignore):
        assert wrap_summaries == 0, (
            "ruff's D205 rejects multi-line summaries, but docformatter is configured to "
            f"wrap summaries at {wrap_summaries} columns. A summary longer than that wraps "
            "onto a second physical line that D205 flags and no tool can auto-fix. Set "
            "[tool.docformatter] wrap-summaries = 0, or drop D205 from [tool.ruff.lint] select."
        )
