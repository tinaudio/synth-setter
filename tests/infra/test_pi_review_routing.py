"""Behavior tests for Pi PR-review model allocation."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent._shared.pi_review_routing import (
    build_review_plan,
    extract_report,
    main,
    parse_available_models,
    provenance_for_model,
    report_is_parseable,
)

AVAILABLE_MODELS = """\
openai-codex  gpt-5.6-sol    372K  128K  yes  yes
openai-codex  gpt-5.6-terra  372K  128K  yes  yes
openrouter    nvidia/nemotron-3-ultra-550b-a55b:free  1000K  128K  yes  yes
openrouter    openrouter/free  200K  128K  yes  yes
openrouter    cohere/north-mini-code:free  256K  128K  yes  yes
"""


def test_parse_available_models_joins_provider_and_model_id() -> None:
    """Parse Pi's column output into canonical model selectors."""
    assert parse_available_models(AVAILABLE_MODELS) == {
        "openai-codex/gpt-5.6-sol",
        "openai-codex/gpt-5.6-terra",
        "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
        "openrouter/openrouter/free",
        "openrouter/cohere/north-mini-code:free",
    }


def test_build_review_plan_allocates_deep_and_mechanical_passes() -> None:
    """Allocate two providers per skill with risk-sensitive thinking."""
    plan = build_review_plan(
        ["correctness-review", "comment-hygiene"],
        changed_lines=120,
        risk_reasons=(),
        available_models=parse_available_models(AVAILABLE_MODELS),
    )

    assert [(item.skill, item.pass_name, item.thinking) for item in plan] == [
        ("correctness-review", "codex", "high"),
        ("correctness-review", "openrouter", "high"),
        ("comment-hygiene", "codex", "low"),
        ("comment-hygiene", "openrouter", "low"),
    ]
    assert plan[1].candidates[0] == "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"
    assert plan[3].candidates[0] == "openrouter/cohere/north-mini-code:free"
    assert all(
        model.startswith(("openai-codex/", "openrouter/"))
        for item in plan
        for model in item.candidates
    )


def test_build_review_plan_promotes_risky_standard_passes() -> None:
    """Promote standard passes when the diff carries a named risk."""
    plan = build_review_plan(
        ["code-health"],
        changed_lines=40,
        risk_reasons=("concurrency",),
        available_models=parse_available_models(AVAILABLE_MODELS),
    )

    assert [item.thinking for item in plan] == ["high", "high"]
    assert [item.reason for item in plan] == ["risk: concurrency", "risk: concurrency"]


def test_build_review_plan_skips_unavailable_candidate() -> None:
    """Advance to an available fallback when a free model is retired."""
    available = parse_available_models(AVAILABLE_MODELS)
    available.remove("openrouter/cohere/north-mini-code:free")

    plan = build_review_plan(
        ["code-health"],
        changed_lines=300,
        risk_reasons=(),
        available_models=available,
    )

    assert plan[1].candidates[0] == "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"
    assert plan[1].unavailable == ("openrouter/cohere/north-mini-code:free",)


def test_build_review_plan_missing_provider_raises_actionable_error() -> None:
    """Reject missing provider authentication before launching workers."""
    available = {
        model
        for model in parse_available_models(AVAILABLE_MODELS)
        if not model.startswith("openrouter/")
    }

    with pytest.raises(ValueError, match=r"openrouter.*credentials required"):
        build_review_plan(
            ["code-health"],
            changed_lines=300,
            risk_reasons=(),
            available_models=available,
        )


@pytest.mark.parametrize(
    ("skill", "changed_lines", "match"),
    [
        ("correctness-reveiw", 10, "Unknown review skill"),
        ("code-health", -1, "changed_lines must be non-negative"),
    ],
)
def test_build_review_plan_invalid_input_raises(
    skill: str, changed_lines: int, match: str
) -> None:
    """Fail closed when routing inputs are invalid.

    :param skill: Candidate checklist name.
    :param changed_lines: Candidate changed-line count.
    :param match: Expected validation error text.
    """
    with pytest.raises(ValueError, match=match):
        build_review_plan(
            [skill],
            changed_lines=changed_lines,
            risk_reasons=(),
            available_models=parse_available_models(AVAILABLE_MODELS),
        )


def test_provenance_for_model_uses_effective_provider() -> None:
    """Attribute fallback findings to the model that produced the report."""
    assert provenance_for_model("openai-codex/gpt-5.6-sol") == "codex"
    assert provenance_for_model("openrouter/openrouter/free") == "openrouter"


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (
            "## code-health review — PR #1\n\n"
            "### BLOCK findings\nNone.\n\n"
            "### WARN findings\nNone.\n\n"
            "### What looks good\n- Clear.",
            True,
        ),
        (
            "## code-health review — PR #1\n\n"
            "### BLOCK findings\n"
            "1. **src/example.py:42** — A concrete defect.\n\n"
            "### WARN findings\nNone.\n\n"
            "### What looks good\n- Clear.",
            True,
        ),
        ("No output.", False),
        ("## Summary\n- No findings.", False),
        (
            "## code-health review — PR #1\n\n"
            "### WARN findings\nNone.\n\n"
            "### BLOCK findings\nNone.\n\n"
            "### What looks good\n- Clear.",
            False,
        ),
        (
            "## code-health review — PR #1\n\n"
            "> ### BLOCK findings\nNone.\n\n"
            "> ### WARN findings\nNone.\n\n"
            "> ### What looks good\n- Clear.",
            False,
        ),
        (
            "## code-health review — PR #1\n\n"
            "### BLOCK findings\n1. Missing path and line.\n\n"
            "### WARN findings\nNone.\n\n"
            "### What looks good\n- Clear.",
            False,
        ),
        (
            "## code-health review — PR #1\n\n"
            "### BLOCK findings\nNone.\n\n"
            "### BLOCK findings\nNone.\n\n"
            "### WARN findings\nNone.\n\n"
            "### What looks good\n- Clear.",
            False,
        ),
        (
            "## code-health review — PR #1\n\n"
            "Unexpected preface.\n\n"
            "### BLOCK findings\nNone.\n\n"
            "### WARN findings\nNone.\n\n"
            "### What looks good\n- Clear.",
            False,
        ),
        (
            "## code-health review — PR #1\n\n"
            "### BLOCK findings\n"
            "1. **src/example.py:42** — A concrete defect.\n"
            "Unstructured continuation.\n\n"
            "### WARN findings\nNone.\n\n"
            "### What looks good\n- Clear.",
            False,
        ),
    ],
)
def test_report_is_parseable_requires_structured_contract(report: str, expected: bool) -> None:
    """Reject empty or structurally incomplete worker reports.

    :param report: Candidate worker report.
    :param expected: Whether the report satisfies the contract.
    """
    assert report_is_parseable(report, expected_skill="code-health") is expected


def test_report_is_parseable_rejects_wrong_skill() -> None:
    """Prevent a valid report for another checklist from entering the merge."""
    report = (
        "## python-style review — PR #1\n\n"
        "### BLOCK findings\nNone.\n\n"
        "### WARN findings\nNone.\n\n"
        "### What looks good\n- Clear."
    )

    assert not report_is_parseable(report, expected_skill="code-health")


def test_extract_report_returns_last_assistant_markdown(tmp_path: Path) -> None:
    """Extract only final assistant text from Tintin JSONL.

    :param tmp_path: Temporary location for a transcript.
    """
    transcript = tmp_path / "worker.output"
    transcript.write_text(
        '{"message":{"role":"assistant","content":[{"type":"text","text":"draft"}]}}\n'
        '{"message":{"role":"toolResult","content":[{"type":"text","text":"noise"}]}}\n'
        '{"message":{"role":"assistant","content":['
        '{"type":"thinking","thinking":"hidden"},'
        '{"type":"text","text":"## code-health review — smoke\\n\\n'
        "### BLOCK findings\\nNone.\\n\\n### WARN findings\\nNone.\\n\\n"
        '### What looks good\\n- Clear."}]}}\n'
    )

    report = extract_report(transcript)

    assert report.startswith("## code-health review — smoke")
    assert "noise" not in report
    assert "hidden" not in report


def test_extract_report_missing_assistant_text_raises(tmp_path: Path) -> None:
    """Reject transcripts without a completed assistant report.

    :param tmp_path: Temporary location for a transcript.
    """
    transcript = tmp_path / "worker.output"
    transcript.write_text('{"message":{"role":"user","content":"prompt"}}\n')

    with pytest.raises(ValueError, match="assistant text"):
        extract_report(transcript)


def test_validate_report_cli_returns_nonzero_for_malformed_output(tmp_path: Path) -> None:
    """Expose report validation to the natural-language orchestrator.

    :param tmp_path: Temporary location for worker output.
    """
    report = tmp_path / "report.md"
    report.write_text("No output.")

    assert main(["validate-report", str(report), "--skill", "code-health"]) == 1
