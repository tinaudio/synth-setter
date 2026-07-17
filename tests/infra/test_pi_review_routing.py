"""Behavior tests for Pi PR-review model allocation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import sh

from agent._shared.pi_review_routing import (
    build_review_plan,
    extract_report,
    main,
    parse_available_models,
    parse_worker_report,
    provenance_for_model,
    report_is_parseable,
    transcript_stats,
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
    assert all(item.max_turns == 12 for item in plan)
    assert plan[1].candidates[0] == "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"
    assert plan[3].candidates[0] == "openrouter/cohere/north-mini-code:free"
    assert all(
        model.startswith("openai-codex/") for item in plan[::2] for model in item.candidates
    )
    assert all(model.startswith("openrouter/") for item in plan[1::2] for model in item.candidates)
    assert all(
        model.startswith("openai-codex/")
        for item in plan[1::2]
        for model in item.fallback_candidates
    )


def test_build_review_plan_keeps_mechanical_passes_bounded_on_risky_diff() -> None:
    """Keep style checklists below high thinking even when the diff is risky."""
    plan = build_review_plan(
        ["python-style"],
        changed_lines=1_000,
        risk_reasons=("concurrency", "authentication"),
        available_models=parse_available_models(AVAILABLE_MODELS),
    )

    assert [item.thinking for item in plan] == ["medium", "medium"]
    assert all(item.reason == "mechanical checklist on diff of 200+ lines" for item in plan)


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


@pytest.mark.parametrize(
    ("skill", "changed_lines", "expected_thinking"),
    [
        ("python-style", 199, "low"),
        ("python-style", 200, "medium"),
        ("code-health", 800, "medium"),
        ("code-health", 801, "high"),
    ],
)
def test_build_review_plan_pins_line_count_boundaries(
    skill: str, changed_lines: int, expected_thinking: str
) -> None:
    """Pin exact thinking-allocation thresholds.

    :param skill: Checklist whose threshold is exercised.
    :param changed_lines: Total diff lines at the boundary.
    :param expected_thinking: Expected allocation at that boundary.
    """
    plan = build_review_plan(
        [skill],
        changed_lines=changed_lines,
        risk_reasons=(),
        available_models=parse_available_models(AVAILABLE_MODELS),
    )

    assert [item.thinking for item in plan] == [expected_thinking, expected_thinking]


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


def test_build_review_plan_empty_skills_raises_actionable_error() -> None:
    """Reject an empty review before provider or worker selection."""
    with pytest.raises(ValueError, match="skills must be non-empty"):
        build_review_plan(
            [],
            changed_lines=10,
            risk_reasons=(),
            available_models=parse_available_models(AVAILABLE_MODELS),
        )


def test_build_review_plan_missing_openrouter_uses_codex_fallback() -> None:
    """Keep both logical passes when OpenRouter has no registered models."""
    available = {
        model
        for model in parse_available_models(AVAILABLE_MODELS)
        if not model.startswith("openrouter/")
    }

    plan = build_review_plan(
        ["code-health"],
        changed_lines=300,
        risk_reasons=(),
        available_models=available,
    )

    assert [item.pass_name for item in plan] == ["codex", "openrouter"]
    assert plan[1].candidates == ()
    assert plan[1].fallback_candidates == (
        "openai-codex/gpt-5.6-sol",
        "openai-codex/gpt-5.6-terra",
    )


def test_build_review_plan_missing_codex_raises_actionable_error() -> None:
    """Reject a plan that cannot run its required Codex passes."""
    available = {
        model
        for model in parse_available_models(AVAILABLE_MODELS)
        if not model.startswith("openai-codex/")
    }

    with pytest.raises(ValueError, match=r"openai-codex.*credentials required"):
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
    assert (
        report_is_parseable(report, expected_skill="code-health", expected_target="PR #1")
        is expected
    )


def test_parse_worker_report_returns_validated_boundary_model() -> None:
    """Return typed report data after structural validation."""
    report = (
        "## code-health review — PR #1\n\n"
        "### BLOCK findings\nNone.\n\n"
        "### WARN findings\nNone.\n\n"
        "### What looks good\n- Clear."
    )

    parsed = parse_worker_report(report, expected_skill="code-health", expected_target="PR #1")

    assert parsed.skill == "code-health"
    assert parsed.target == "PR #1"


def test_report_is_parseable_rejects_wrong_skill_or_target() -> None:
    """Prevent a valid report for another checklist from entering the merge."""
    report = (
        "## python-style review — PR #1\n\n"
        "### BLOCK findings\nNone.\n\n"
        "### WARN findings\nNone.\n\n"
        "### What looks good\n- Clear."
    )

    assert not report_is_parseable(report, expected_skill="code-health", expected_target="PR #1")
    assert not report_is_parseable(
        report.replace("python-style", "code-health"),
        expected_skill="code-health",
        expected_target="PR #2",
    )


def test_transcript_stats_summarizes_runtime_budget(tmp_path: Path) -> None:
    """Expose turns, elapsed time, and cumulative tokens for the audit.

    :param tmp_path: Temporary location for a transcript.
    """
    transcript = tmp_path / "worker.output"
    transcript.write_text(
        '{"type":"assistant","message":{"role":"assistant","content":"draft",'
        '"usage":{"input":10,"output":2,"reasoning":1,"totalTokens":12}},'
        '"timestamp":"2026-07-16T20:00:00Z"}\n'
        '{"type":"toolResult","message":{"role":"toolResult","content":"noise"},'
        '"timestamp":"2026-07-16T20:00:02Z"}\n'
        '{"type":"assistant","message":{"role":"assistant","content":"final",'
        '"usage":{"input":20,"output":3,"reasoning":2,"totalTokens":23}},'
        '"timestamp":"2026-07-16T20:00:05Z"}\n'
    )

    stats = transcript_stats(transcript)

    assert stats.turns == 2
    assert stats.elapsed_seconds == 5
    assert stats.cumulative_tokens == 35


def test_transcript_stats_marks_unavailable_telemetry_unknown(tmp_path: Path) -> None:
    """Distinguish absent telemetry from genuine zero usage.

    :param tmp_path: Temporary location for a transcript.
    """
    transcript = tmp_path / "worker.output"
    transcript.write_text('{"message":{"role":"assistant","content":"final"}}\n')

    stats = transcript_stats(transcript)

    assert stats.elapsed_seconds is None
    assert stats.cumulative_tokens is None


def test_transcript_stats_marks_partial_usage_unknown(tmp_path: Path) -> None:
    """Reject falsely precise totals when one turn omits total tokens.

    :param tmp_path: Temporary location for a transcript.
    """
    transcript = tmp_path / "worker.output"
    transcript.write_text(
        '{"message":{"role":"assistant","content":"draft",'
        '"usage":{"totalTokens":12}},"timestamp":"2026-07-16T20:00:00Z"}\n'
        '{"message":{"role":"assistant","content":"final","usage":{}},'
        '"timestamp":"2026-07-16T20:00:05Z"}\n'
    )

    assert transcript_stats(transcript).cumulative_tokens is None


def test_extract_report_returns_last_assistant_markdown(tmp_path: Path) -> None:
    """Extract only final assistant text from Tintin JSONL.

    :param tmp_path: Temporary location for a transcript.
    """
    transcript = tmp_path / "worker.output"
    transcript.write_text(
        '{"type":"session_start","sessionId":"metadata-only"}\n'
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


def test_extract_report_normalizes_preface_and_trailing_prose(tmp_path: Path) -> None:
    """Keep the structured report when a model wraps it in narration.

    :param tmp_path: Temporary location for a transcript.
    """
    transcript = tmp_path / "worker.output"
    transcript.write_text(
        '{"message":{"role":"assistant","content":"analysis\\n\\n'
        "## code-health review — smoke\\n\\n### BLOCK findings\\nNone.\\n\\n"
        "### WARN findings\\n1. **src/example.py:10-12** — Defect.\\n"
        "```python\\n1. **src/fake.py:99-100** — Example.\\n```\\n"
        "- **src/bullet.py:20-22** — Bullet.\\n\\n"
        '### What looks good\\n- Clear.\\n\\nclosing"}}\n'
    )

    report = extract_report(transcript)

    assert report == (
        "## code-health review — smoke\n\n"
        "### BLOCK findings\nNone.\n\n"
        "### WARN findings\n"
        "1. **src/example.py:10** — [reported range 10-12] Defect. "
        "1. **src/fake.py:99-100** — Example.\n"
        "2. **src/bullet.py:20** — [reported range 20-22] Bullet.\n\n"
        "### What looks good\n- Clear."
    )


def test_extract_report_drops_section_preamble_and_renumbers_mixed_findings(
    tmp_path: Path,
) -> None:
    """Produce a valid contract after a narrated mixed-style findings section.

    :param tmp_path: Temporary location for a transcript.
    """
    transcript = tmp_path / "worker.output"
    transcript.write_text(
        '{"message":{"role":"assistant","content":"'
        "## code-health review — smoke\\n\\n### BLOCK findings\\nNone.\\n\\n"
        "### WARN findings\\nThe following were observed.\\n"
        "- **src/first.py:10** — First finding.\\n"
        "2. **src/second.py:20** — Second finding.\\n\\n"
        '### What looks good\\n- Clear."}}\n'
    )

    report = extract_report(transcript)

    assert report_is_parseable(report, expected_skill="code-health", expected_target="smoke")
    assert "The following were observed." not in report
    assert "1. **src/first.py:10** — First finding." in report
    assert "2. **src/second.py:20** — Second finding." in report


def test_extract_report_cli_normalizes_narrated_mixed_findings(tmp_path: Path) -> None:
    """Write a valid canonical report from a narrated worker transcript.

    :param tmp_path: Temporary location for transcript and report files.
    """
    transcript = tmp_path / "worker.jsonl"
    report = tmp_path / "worker.md"
    transcript.write_text(
        '{"message":{"role":"assistant","content":"'
        "## code-health review — smoke\\n\\n### BLOCK findings\\nNone.\\n\\n"
        "### WARN findings\\nThe following were observed.\\n"
        "- **src/first.py:10** — First finding.\\n"
        "2. **src/second.py:20** — Second finding.\\n\\n"
        '### What looks good\\n- Clear."}}\n'
    )
    script = Path(__file__).resolve().parents[2] / "agent/_shared/pi_review_routing.py"
    python = sh.Command(sys.executable)

    python(script, "extract-report", transcript, "--output", report)
    python(
        script,
        "validate-report",
        report,
        "--skill",
        "code-health",
        "--target",
        "smoke",
    )

    assert report.read_text() == (
        "## code-health review — smoke\n\n"
        "### BLOCK findings\nNone.\n\n"
        "### WARN findings\n"
        "1. **src/first.py:10** — First finding.\n"
        "2. **src/second.py:20** — Second finding.\n\n"
        "### What looks good\n- Clear.\n"
    )


def test_extract_report_rejects_duplicate_heading_after_good_section(
    tmp_path: Path,
) -> None:
    """Keep duplicate headings visible so validation rejects the report.

    :param tmp_path: Temporary location for a transcript.
    """
    transcript = tmp_path / "worker.output"
    transcript.write_text(
        '{"message":{"role":"assistant","content":"'
        "## code-health review — smoke\\n\\n### BLOCK findings\\nNone.\\n\\n"
        "### WARN findings\\nNone.\\n\\n### What looks good\\n- Clear.\\n\\n"
        '### BLOCK findings\\n1. **src/hidden.py:9** — Hidden."}}\n'
    )

    report = extract_report(transcript)

    assert not report_is_parseable(report, expected_skill="code-health", expected_target="smoke")
    assert "src/hidden.py:9" in report


def test_extract_report_requires_final_assistant_report(tmp_path: Path) -> None:
    """Reject an earlier provisional report followed by a final retraction.

    :param tmp_path: Temporary location for a transcript.
    """
    transcript = tmp_path / "worker.output"
    transcript.write_text(
        '{"message":{"role":"assistant","content":"'
        "## code-health review — smoke\\n\\n### BLOCK findings\\nNone.\\n\\n"
        '### WARN findings\\nNone.\\n\\n### What looks good\\n- Clear."}}\n'
        '{"message":{"role":"assistant","content":"Tool failure; retract the report."}}\n'
    )

    with pytest.raises(ValueError, match="final assistant text is not a report"):
        extract_report(transcript)


def test_extract_report_rejects_untyped_metadata_event(tmp_path: Path) -> None:
    """Reject transcript rows with neither a message nor event discriminator.

    :param tmp_path: Temporary location for a transcript.
    """
    transcript = tmp_path / "worker.output"
    transcript.write_text("{}\n")

    with pytest.raises(ValueError, match="message or event type"):
        extract_report(transcript)


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

    assert (
        main(
            [
                "validate-report",
                str(report),
                "--skill",
                "code-health",
                "--target",
                "smoke",
            ]
        )
        == 1
    )


def test_report_cli_real_process_extracts_and_validates_transcript(tmp_path: Path) -> None:
    """Exercise the documented transcript-to-validation command path.

    :param tmp_path: Temporary location for transcript and report files.
    """
    transcript = tmp_path / "worker.jsonl"
    report = tmp_path / "worker.md"
    transcript.write_text(
        '{"message":{"role":"assistant","content":[{"type":"text","text":'
        '"## code-health review — smoke\\n\\n### BLOCK findings\\nNone.\\n\\n'
        '### WARN findings\\nNone.\\n\\n### What looks good\\n- Clear."}]}}\n'
    )
    script = Path(__file__).resolve().parents[2] / "agent/_shared/pi_review_routing.py"
    python = sh.Command(sys.executable)

    python(script, "extract-report", transcript, "--output", report)
    python(
        script,
        "validate-report",
        report,
        "--skill",
        "code-health",
        "--target",
        "smoke",
    )

    stats = json.loads(str(python(script, "transcript-stats", transcript)))
    provenance = str(
        python(script, "provenance", "openrouter/cohere/north-mini-code:free")
    ).strip()

    assert report.read_text().startswith("## code-health review — smoke")
    assert stats["turns"] == 1
    assert provenance == "openrouter"


def test_plan_cli_real_process_surfaces_pi_registry_failure(tmp_path: Path) -> None:
    """Return actionable diagnostics when Pi cannot list models.

    :param tmp_path: Temporary location for the failing executable.
    """
    pi = tmp_path / "pi"
    pi.write_text("#!/bin/sh\necho registry unavailable >&2\nexit 7\n")
    pi.chmod(0o755)
    script = Path(__file__).resolve().parents[2] / "agent/_shared/pi_review_routing.py"

    with pytest.raises(sh.ErrorReturnCode) as error:
        sh.Command(sys.executable)(
            script,
            "plan",
            "--skill",
            "code-health",
            "--changed-lines",
            "20",
            _env={"PATH": f"{tmp_path}:{os.environ['PATH']}"},
        )

    assert b"pi --list-models failed: registry unavailable" in error.value.stderr


def test_plan_cli_real_process_uses_fake_pi_registry(tmp_path: Path) -> None:
    """Exercise the user-facing planner with a deterministic Pi executable.

    :param tmp_path: Temporary location for the fake executable.
    """
    pi = tmp_path / "pi"
    pi.write_text(f"#!/bin/sh\nprintf '%s' '{AVAILABLE_MODELS}'\n")
    pi.chmod(0o755)
    script = Path(__file__).resolve().parents[2] / "agent/_shared/pi_review_routing.py"

    result = sh.Command(sys.executable)(
        script,
        "plan",
        "--skill",
        "code-health",
        "--changed-lines",
        "20",
        _env={"PATH": f"{tmp_path}:{os.environ['PATH']}"},
    )

    payload = json.loads(str(result))
    assert payload[1]["candidates"][0] == "openrouter/cohere/north-mini-code:free"
