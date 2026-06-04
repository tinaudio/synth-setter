#!/usr/bin/env python
"""Lint a pull-request body against the repo's canonical section hierarchy.

Pins the section names ``Why`` → ``What changed`` → ``Test plan`` (optional
``Out of scope``) and their order, mapping legacy aliases (see ``ALIASES``) onto
the canonical names so every PR reads the same way. Structural only — it never
inspects prose quality. Findings: ``missing-section``, ``out-of-order``,
``aliased-heading``.

Run in ``warn`` mode (default) to annotate without failing, or ``block`` mode to
exit non-zero on any finding::

    gh api repos/:owner/:repo/pulls/:n --jq .body | python scripts/ci/pr_body_lint.py --mode block
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from typing import Literal, TextIO

import click

#: Canonical required sections, in the order a reader should encounter them.
REQUIRED_SECTIONS: tuple[str, ...] = ("Why", "What changed", "Test plan")

#: Sections that are encouraged but never flagged when absent.
OPTIONAL_SECTIONS: tuple[str, ...] = ("Out of scope",)

#: Legacy heading name (lower-cased) -> the canonical section it stands in for.
ALIASES: Mapping[str, str] = {
    "summary": "What changed",
    "what": "What changed",
    "changes": "What changed",
    "what's in the diff": "What changed",
    "what does this pr do": "What changed",
    "motivation": "Why",
    "rationale": "Why",
    "problem": "Why",
    "context": "Why",
    "verification": "Test plan",
    "validation": "Test plan",
    "tests": "Test plan",
    "testing": "Test plan",
    "how tested": "Test plan",
}

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")
# Up to 3 spaces of indent, then a run of >=3 backticks or tildes (CommonMark).
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


# Plain class, not @dataclass: pydoclint 0.8.3 fires DOC601/603 on class-level
# annotated fields; instance attributes set in __init__ are exempt.
class Finding:
    """A single structural problem with a PR body."""

    def __init__(self, code: str, message: str) -> None:
        """Build a finding from its category code and message.

        :param code: machine-readable finding category.
        :param message: human-readable, fix-oriented description.
        """
        self.code = code
        self.message = message


def extract_headings(body: str) -> list[str]:
    """Return the ATX headings of ``body`` in document order.

    Headings inside fenced code blocks are skipped — a ``#`` line there is shell
    or markup, not a section. A fence closes only on a line of the same character
    that is at least as long as the opener, so a shorter fence nested in a longer
    one does not end it early. Each heading is stripped of a trailing colon,
    question mark (legacy templates used ``?``), and surrounding whitespace.

    :param body: the PR-body markdown.
    :returns: the ordered heading texts (e.g. ``["Why", "What changed"]``).
    """
    headings: list[str] = []
    open_fence: str | None = None
    for line in body.splitlines():
        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if open_fence is None:
                open_fence = marker
            elif marker[0] == open_fence[0] and len(marker) >= len(open_fence):
                open_fence = None
            continue
        if open_fence is not None:
            continue
        match = _HEADING_RE.match(line)
        if match:
            headings.append(match.group(1).rstrip(" :?"))
    return headings


def _canonical_for(heading: str) -> str | None:
    """Return the canonical section a heading denotes, or ``None`` if it is not a section.

    :param heading: a raw heading text from :func:`extract_headings`.
    :returns: a member of :data:`REQUIRED_SECTIONS` / :data:`OPTIONAL_SECTIONS`, or ``None``.
    """
    lowered = heading.lower()
    for section in (*REQUIRED_SECTIONS, *OPTIONAL_SECTIONS):
        if lowered == section.lower():
            return section
    return ALIASES.get(lowered)


def lint_pr_body(body: str) -> list[Finding]:
    """Return the structural findings for a PR ``body``.

    :param body: the PR-body markdown.
    :returns: findings in a stable order — aliases (document order), then missing sections
        (canonical order), then a single ordering finding if needed.
    """
    sections = [(heading, _canonical_for(heading)) for heading in extract_headings(body)]
    # Sections present under their own canonical name — used to suppress a rename
    # suggestion that would just duplicate an already-present canonical heading.
    literal = {c for heading, c in sections if c is not None and heading.lower() == c.lower()}

    alias_findings: list[Finding] = []
    present: dict[str, int] = {}
    for index, (heading, canonical) in enumerate(sections):
        if canonical is None:
            continue
        present.setdefault(canonical, index)
        if heading.lower() != canonical.lower() and canonical not in literal:
            alias_findings.append(
                Finding(
                    code="aliased-heading",
                    message=f"Rename '## {heading}' to the canonical '## {canonical}'",
                )
            )

    missing_findings = [
        Finding(code="missing-section", message=f"Missing required section: ## {section}")
        for section in REQUIRED_SECTIONS
        if section not in present
    ]

    ordering_findings: list[Finding] = []
    positions = [present[s] for s in REQUIRED_SECTIONS if s in present]
    if positions != sorted(positions):
        canonical_order = " → ".join(f"## {s}" for s in REQUIRED_SECTIONS)
        ordering_findings.append(
            Finding(
                code="out-of-order",
                message=f"Required sections are out of order; expected {canonical_order}",
            )
        )

    return alias_findings + missing_findings + ordering_findings


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["warn", "block"]),
    default="warn",
    help="warn: annotate but exit 0. block: exit 1 when any finding exists.",
)
@click.option(
    "--body-file",
    type=click.File("r", encoding="utf-8"),
    default="-",
    help="File to read the PR body from (default: stdin).",
)
def main(mode: Literal["warn", "block"], body_file: TextIO) -> None:
    """Lint a PR body read from ``--body-file`` (or stdin) and annotate findings.

    Emits GitHub Actions annotations (``::warning::`` / ``::error::``) so findings
    surface inline on the workflow run. Exit code is governed by ``--mode``.

    :param mode: ``warn`` (always exit 0) or ``block`` (exit 1 on any finding).
    :param body_file: open text handle to read the PR body from.
    """
    findings = lint_pr_body(body_file.read())
    annotation = "::error::" if mode == "block" else "::warning::"
    for finding in findings:
        click.echo(f"{annotation}[{finding.code}] {finding.message}")

    if not findings:
        click.echo("✓ PR body matches the canonical section hierarchy.")
        return
    if mode == "block":
        sys.exit(1)


if __name__ == "__main__":
    main()
