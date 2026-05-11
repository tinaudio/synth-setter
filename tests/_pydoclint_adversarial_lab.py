"""Adversarial probe surface for pydoclint 0.8.3.

This module exists to test what the pydoclint pre-commit hook catches and
what slips past it. It is intentionally NOT pytest-collected (the leading
underscore in the filename keeps pytest from collecting it, and it contains
no ``test_*`` functions). It is also NOT in ``[tool.pydoclint].exclude``,
so pydoclint will run on it.

Every function/class below contains a docstring defect that a reviewer
would reject, but that pydoclint 0.8.3 (as configured in
``[tool.pydoclint]``) reports zero violations on. The whole file is meant
to pass pre-commit cleanly — that is the slip surface this PR documents.

The positive controls that prove the rules *do* fire (DOC101/DOC103,
DOC501/DOC503, narrative+params → DOC101/DOC103/DOC201/DOC203) are
verified separately in the PR description and not included here, because
including them would fail the lint and obscure the slip surface.

Do not import or call anything in this file from production code.

See PR #939 (pydoclint adoption) and issue #938 (audit / remediation).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# P1 — Missing docstring entirely.
#
# pydoclint defers "must have a docstring" to pydocstyle / flake8-docstrings
# (see ``pydoclint/visitor.py:210-215``: "We don't check functions without
# docstrings"). Neither is wired into pre-commit; only ``interrogate`` is,
# and interrogate enforces coverage % at the file level, not per-function.
# A new public function with NO docstring at all passes pydoclint without
# a single DOCxxx flag.
# ---------------------------------------------------------------------------
def silent_no_docstring(x: int, y: int) -> int:
    return x + y


# ---------------------------------------------------------------------------
# P2 — Transitive raise (helper raises, caller does not).
#
# ``hasRaiseStatements`` (``pydoclint/utils/return_yield_raise.py:106-112``)
# only walks ``ast.Raise`` nodes in the immediate function body, not into
# called helpers. ``run_workflow`` propagates a ValueError from
# ``_helper_that_raises`` but doesn't declare it. Reviewer expectation: the
# caller's docstring should list ``:raises ValueError:``. Pydoclint gives
# no DOC501 / DOC503 because there is no ``raise`` keyword in the caller.
# ---------------------------------------------------------------------------
def _helper_that_raises() -> None:
    """Raise unconditionally.

    :raises ValueError: always.
    """
    raise ValueError("boom")


def run_workflow() -> None:
    """Run the workflow."""
    _helper_that_raises()


# ---------------------------------------------------------------------------
# P3 — Instance attributes set in ``__init__`` are invisible to the
# class-attribute checker.
#
# ``extractClassAttributesFromNode``
# (``pydoclint/utils/visitor_helper.py:136-227``) only inspects class-level
# ``ast.AnnAssign`` / ``ast.Assign`` statements, not ``self.x = ...``
# assignments inside ``__init__``. The class below has no class-level
# attributes at all — both ``id`` and ``name`` live entirely on instances.
# The class docstring claims ``:ivar id:`` but mentions nothing about
# ``name``. Pydoclint reports nothing.
# ---------------------------------------------------------------------------
class Account:
    """Account record.

    :ivar id: identifier.
    """

    def __init__(self, id: int, name: str) -> None:
        """Initialise.

        :param id: identifier.
        :param name: account holder name.
        """
        self.id = id
        self.name = name


# ---------------------------------------------------------------------------
# P4 — Narrative-only docstring on a function with no params, no return
# value, and no direct raises.
#
# ``_containsSphinxStylePattern`` / ``_containsNumpyStylePattern`` /
# ``_containsGoogleStylePattern`` (``pydoclint/utils/parse_docstring.py:127-171``)
# return False on prose with no section markers; when no style matches,
# DOC003 is suppressed. With no params, no return value, no raise statement,
# pydoclint has nothing to mismatch against — the gate passes a prose
# docstring that says effectively nothing about the function's behaviour.
#
# Note: this only slips when the signature has nothing to check. A
# narrative-only docstring on ``def f(x, y) -> int`` correctly fires
# DOC101/DOC103/DOC201/DOC203 — pydoclint catches that case.
# ---------------------------------------------------------------------------
def vague_void_function() -> None:
    """Do the thing."""


# ---------------------------------------------------------------------------
# P5 — ``# noqa: DOCxxx`` suppression on the docstring-closing line.
#
# The CLI default for ``--native-mode-noqa-location`` is ``"docstring"``
# (``pydoclint/main.py:402-410``), so pydoclint reads suppression comments
# from the line containing the closing ``\"\"\"``. A real DOC101/DOC103
# violation (signature has ``x``, ``y``; docstring documents ``a``, ``b``)
# is silenced because the suppression line carries both codes. The lint
# passes; the docstring is silently allowed to lie about parameter names.
# ---------------------------------------------------------------------------
def with_docstring_line_noqa_suppression(x: int, y: int) -> int:
    """Add.

    :param a: WRONG NAME — actually called ``x``.
    :param b: WRONG NAME — actually called ``y``.
    :return int: result.
    """  # noqa: DOC101, DOC103
    return x + y


# ---------------------------------------------------------------------------
# P5b — companion to P5: the *natural* place to put a noqa (the ``def``
# line, mirroring flake8/ruff convention) does NOT work under the CLI
# default. The suppression below is read by the noqa parser but never
# applied, because the default location is ``"docstring"``, not
# ``"definition"``. The hook does fire here — but the developer who wrote
# this comment thinks it doesn't. Both behaviours surprise: the suppression
# either silently slips a violation (P5) or silently fails (P5b).
#
# To keep the slip surface clean (this file should pass pre-commit), we do
# NOT actually create a violation here — the function below is a docstring
# that already matches its signature. The ``# noqa`` is harmless prose.
# The point is the documented surprise, not a runtime demonstration.
# ---------------------------------------------------------------------------
def def_line_noqa_is_inert(x: int) -> int:  # noqa: DOC101, DOC103
    """Add one.

    :param x: input.
    :return int: result.
    """
    return x + 1
