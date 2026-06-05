"""Tests for scripts/ci/pr_body_lint.py — canonical PR-body section linter.

The linter pins the repo's PR-body hierarchy: a body must carry the canonical
``Why`` → ``What changed`` → ``Test plan`` sections, in that order, and should
prefer those names over the legacy variants (``Summary``, ``Verification``, …)
the project drifted into. The contract under test is "given this body text,
which structural findings are emitted?" — a pure function over a string, so the
tests pass body literals and assert on returned :class:`Finding` codes.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from scripts.ci.pr_body_lint import (
    extract_headings,
    lint_pr_body,
    main,
)

# A body that satisfies every rule — the canonical shape, in order.
CANONICAL_BODY = """\
## Why

Refs #123 — the foo retries were silently dropping 5xx.

## What changed

- Retry R2 puts on 5xx with capped backoff.

## Test plan

- `make test-infra` green; injected a 503 and saw three retries.

## Out of scope

- 4xx handling stays fail-fast.
"""


def _codes(body: str) -> set[str]:
    """Return the set of finding codes emitted for ``body``.

    :param body: the PR-body markdown to lint.
    :returns: the distinct :attr:`Finding.code` values.
    """
    return {f.code for f in lint_pr_body(body)}


# ---------------------------------------------------------------------------
# extract_headings — parsing
# ---------------------------------------------------------------------------


def test_extract_headings_returns_atx_headings_in_order() -> None:
    """Each ``##`` heading round-trips into the returned list, in document order."""
    body = "## Why\n\ntext\n\n## What changed\n\ntext\n"
    assert extract_headings(body) == ["Why", "What changed"]


def test_extract_headings_strips_trailing_colon_and_whitespace() -> None:
    """A heading written ``## Test plan:`` normalizes to ``Test plan``."""
    assert extract_headings("##   Test plan:  \n") == ["Test plan"]


def test_extract_headings_ignores_headings_inside_fenced_code_blocks() -> None:
    """A ``#`` line inside a ``` fence is shell/markup, not a section heading."""
    body = "## Why\n\n```bash\n# not a heading\necho hi\n```\n\n## What changed\n"
    assert extract_headings(body) == ["Why", "What changed"]


def test_extract_headings_ignores_headings_inside_tilde_fenced_blocks() -> None:
    """A ``#`` line inside a ``~~~`` fence is also skipped, like a backtick fence."""
    body = "## Why\n\n~~~\n# not a heading\n~~~\n\n## What changed\n"
    assert extract_headings(body) == ["Why", "What changed"]


def test_extract_headings_strips_trailing_question_mark() -> None:
    """A heading written ``## Why?`` normalizes to ``Why`` (legacy templates used ``?``)."""
    assert extract_headings("## Why?\n") == ["Why"]


def test_extract_headings_longer_fence_is_not_closed_by_shorter_inner_fence() -> None:
    """A 4-backtick fence embedding a 3-backtick line stays open until the matching close.

    A shorter ``` line inside the outer ```` fence must not end it early, or a ``#``
    heading in the embedded block would be misread as a section.
    """
    body = "## Why\n\n````\n```\n# not a heading\n```\n````\n\n## What changed\n"
    assert extract_headings(body) == ["Why", "What changed"]


# ---------------------------------------------------------------------------
# lint_pr_body — missing required sections
# ---------------------------------------------------------------------------


def test_lint_canonical_body_has_no_findings() -> None:
    """The fully canonical body is clean — no findings at all."""
    assert lint_pr_body(CANONICAL_BODY) == []


def test_lint_empty_body_flags_every_required_section_missing() -> None:
    """An empty body is missing all three required sections."""
    findings = lint_pr_body("")
    missing = {f.message for f in findings if f.code == "missing-section"}
    assert missing == {
        "Missing required section: ## Why",
        "Missing required section: ## What changed",
        "Missing required section: ## Test plan",
    }


def test_lint_body_missing_only_why_flags_one_missing_section() -> None:
    """A body with What changed + Test plan but no Why flags exactly Why."""
    body = "## What changed\n\n- x\n\n## Test plan\n\n- y\n"
    missing = [f for f in lint_pr_body(body) if f.code == "missing-section"]
    assert [f.message for f in missing] == ["Missing required section: ## Why"]


def test_lint_optional_out_of_scope_absence_is_not_flagged() -> None:
    """Omitting the optional Out of scope section produces no finding."""
    body = "## Why\n\na\n\n## What changed\n\nb\n\n## Test plan\n\nc\n"
    assert _codes(body) == set()


# ---------------------------------------------------------------------------
# lint_pr_body — ordering
# ---------------------------------------------------------------------------


def test_lint_required_sections_out_of_order_flags_ordering() -> None:
    """Test plan before Why violates the canonical order."""
    body = "## Test plan\n\na\n\n## Why\n\nb\n\n## What changed\n\nc\n"
    assert "out-of-order" in _codes(body)


def test_lint_required_sections_in_order_has_no_ordering_finding() -> None:
    """Canonical order emits no ordering finding even with extra sections between."""
    body = "## Why\n\na\n\n## Notes\n\nx\n\n## What changed\n\nb\n\n## Test plan\n\nc\n"
    assert "out-of-order" not in _codes(body)


# ---------------------------------------------------------------------------
# lint_pr_body — legacy aliases
# ---------------------------------------------------------------------------


def test_lint_summary_heading_flagged_as_alias_for_what_changed() -> None:
    """`## Summary` is the most common legacy name; suggest `What changed`."""
    body = "## Why\n\na\n\n## Summary\n\nb\n\n## Test plan\n\nc\n"
    aliases = [(f.code, f.message) for f in lint_pr_body(body) if f.code == "aliased-heading"]
    assert aliases == [
        ("aliased-heading", "Rename '## Summary' to the canonical '## What changed'")
    ]


def test_lint_verification_heading_flagged_as_alias_for_test_plan() -> None:
    """`## Verification` is a legacy name for the Test plan section."""
    body = "## Why\n\na\n\n## What changed\n\nb\n\n## Verification\n\nc\n"
    messages = [f.message for f in lint_pr_body(body) if f.code == "aliased-heading"]
    assert messages == ["Rename '## Verification' to the canonical '## Test plan'"]


def test_lint_legacy_what_does_this_pr_do_heading_maps_to_what_changed() -> None:
    """The old template's ``## What does this PR do?`` is recognized as the What changed section.

    The trailing ``?`` must not defeat the alias, so a PR still on the old template
    is not told its What changed section is missing.
    """
    body = "## Why\n\na\n\n## What does this PR do?\n\nb\n\n## Test plan\n\nc\n"
    findings = lint_pr_body(body)
    assert "missing-section" not in {f.code for f in findings}
    aliases = [f.message for f in findings if f.code == "aliased-heading"]
    assert aliases == ["Rename '## What does this PR do' to the canonical '## What changed'"]


def test_lint_alias_satisfies_the_missing_section_check() -> None:
    """An alias counts as its canonical section — no missing-section finding for it."""
    body = "## Why\n\na\n\n## Summary\n\nb\n\n## Test plan\n\nc\n"
    assert "missing-section" not in _codes(body)


def test_lint_alias_alongside_its_canonical_heading_is_not_flagged() -> None:
    """When the canonical heading already exists, an alias is not flagged for rename.

    Renaming would otherwise create a duplicate ``## What changed`` heading.
    """
    body = "## Why\n\na\n\n## What changed\n\nb\n\n## Summary\n\nc\n\n## Test plan\n\nx\n"
    assert "aliased-heading" not in _codes(body)


def test_lint_alias_before_its_canonical_heading_is_not_flagged() -> None:
    """Suppression is order-independent: the alias may precede the canonical heading."""
    body = "## Why\n\na\n\n## Summary\n\nb\n\n## What changed\n\nc\n\n## Test plan\n\nx\n"
    assert "aliased-heading" not in _codes(body)


# ---------------------------------------------------------------------------
# main — CLI shell (warn vs block exit codes)
# ---------------------------------------------------------------------------


def test_main_warn_mode_exits_zero_even_with_findings() -> None:
    """Warn mode reports findings but never fails the job."""
    result = CliRunner().invoke(main, ["--mode", "warn"], input="(no sections)\n")
    assert result.exit_code == 0
    assert "::warning::" in result.output


def test_main_block_mode_exits_nonzero_when_findings_exist() -> None:
    """Block mode fails the job when the body violates the contract."""
    result = CliRunner().invoke(main, ["--mode", "block"], input="(no sections)\n")
    assert result.exit_code == 1
    assert "::error::" in result.output


def test_main_block_mode_exits_zero_for_canonical_body() -> None:
    """A canonical body passes even in block mode."""
    result = CliRunner().invoke(main, ["--mode", "block"], input=CANONICAL_BODY)
    assert result.exit_code == 0


def test_main_clean_body_prints_success_line() -> None:
    """A clean body reports the success message rather than annotations."""
    result = CliRunner().invoke(main, input=CANONICAL_BODY)
    assert "✓ PR body matches the canonical section hierarchy." in result.output


def test_main_reads_body_from_body_file_option(tmp_path: Path) -> None:
    """``--body-file`` lints the named file instead of stdin.

    :param tmp_path: pytest temporary-directory fixture.
    """
    body_path = tmp_path / "body.md"
    body_path.write_text("(no sections)\n", encoding="utf-8")
    result = CliRunner().invoke(main, ["--mode", "block", "--body-file", str(body_path)])
    assert result.exit_code == 1
    assert "::error::" in result.output
