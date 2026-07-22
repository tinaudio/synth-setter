"""Behavior tests for Pi PR-review model allocation."""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pytest
import sh

from agent._shared.pi_review_routing import (
    build_review_plan,
    build_worker_prompt,
    extract_report,
    finding_fingerprint,
    main,
    parse_available_models,
    parse_worker_report,
    provenance_for_model,
    report_is_parseable,
    report_repair_prompt,
    stream_host_events,
    transcript_stats,
)

AVAILABLE_MODELS = """\
openai-codex  gpt-5.6-sol    372K  128K  yes  yes
openai-codex  gpt-5.6-terra  372K  128K  yes  yes
kimi-coding   k3  256K  128K  yes  yes
openrouter    nvidia/nemotron-3-ultra-550b-a55b:free  1M  65.5K  yes  no
openrouter    nvidia/nemotron-3-super-120b-a12b:free  262.1K  262.1K  yes  no
openrouter    tencent/hy3:free  262.1K  262.1K  yes  no
"""

OPENROUTER_FREE_MODELS = (
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/tencent/hy3:free",
)
SMART_FREE_POOL_MODELS = ("kimi-coding/k3", *OPENROUTER_FREE_MODELS)


def test_parse_available_models_joins_provider_and_model_id() -> None:
    """Parse Pi's column output into canonical model selectors."""
    assert parse_available_models(AVAILABLE_MODELS) == {
        "openai-codex/gpt-5.6-sol",
        "openai-codex/gpt-5.6-terra",
        "kimi-coding/k3",
        "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
        "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
        "openrouter/tencent/hy3:free",
    }


def test_build_review_plan_allocates_fixed_smart_and_mechanical_model_tiers() -> None:
    """Reserve Sol and K3 for semantic checklists regardless of diff risk."""
    plan = build_review_plan(
        ["correctness-review", "comment-hygiene"],
        changed_lines=120,
        risk_reasons=(),
        available_models=parse_available_models(AVAILABLE_MODELS),
    )

    assert [
        (item.skill, item.model_tier, item.pass_name, item.thinking, item.candidates)
        for item in plan
    ] == [
        (
            "correctness-review",
            "smart",
            "codex",
            "high",
            (
                "openai-codex/gpt-5.6-sol",
                "openai-codex/gpt-5.6-terra",
            ),
        ),
        (
            "correctness-review",
            "smart",
            "free-pool",
            "high",
            SMART_FREE_POOL_MODELS,
        ),
        (
            "comment-hygiene",
            "mechanical",
            "codex",
            "low",
            ("openai-codex/gpt-5.6-terra",),
        ),
        (
            "comment-hygiene",
            "mechanical",
            "free-pool",
            "low",
            OPENROUTER_FREE_MODELS,
        ),
    ]
    assert all(item.max_turns == 12 for item in plan)
    assert plan[1].fallback_candidates == (
        "openai-codex/gpt-5.6-terra",
        "openai-codex/gpt-5.6-sol",
    )
    assert plan[3].fallback_candidates == ("openai-codex/gpt-5.6-terra",)


@pytest.mark.parametrize(
    ("skill", "expected_tier"),
    [
        ("code-health", "mechanical"),
        ("comment-hygiene", "mechanical"),
        ("correctness-review", "smart"),
        ("gha-workflow-validator", "mechanical"),
        ("lance-review", "smart"),
        ("ml-data-pipeline", "smart"),
        ("ml-test", "smart"),
        ("python-style", "mechanical"),
        ("shell-style", "mechanical"),
        ("synth-setter-project-standards", "smart"),
        ("tdd-implementation", "mechanical"),
        ("tdd-refactor", "mechanical"),
    ],
)
def test_build_review_plan_uses_fixed_tier_for_every_skill(skill: str, expected_tier: str) -> None:
    """Keep every supported checklist in its explicitly approved model tier.

    :param skill: Checklist being routed.
    :param expected_tier: Fixed smart or mechanical model tier.
    """
    plan = build_review_plan(
        [skill],
        changed_lines=50,
        risk_reasons=("concurrency",),
        available_models=parse_available_models(AVAILABLE_MODELS),
    )

    assert [item.model_tier for item in plan] == [expected_tier, expected_tier]


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


def test_build_review_plan_risky_mechanical_skill_keeps_lower_model_tier() -> None:
    """Raise thinking without promoting a fixed mechanical route to Sol or K3."""
    plan = build_review_plan(
        ["code-health"],
        changed_lines=40,
        risk_reasons=("concurrency",),
        available_models=parse_available_models(AVAILABLE_MODELS),
    )

    assert [item.thinking for item in plan] == ["high", "high"]
    assert [item.reason for item in plan] == ["risk: concurrency", "risk: concurrency"]
    assert [item.model_tier for item in plan] == ["mechanical", "mechanical"]
    assert plan[0].candidates == ("openai-codex/gpt-5.6-terra",)
    assert plan[1].candidates == OPENROUTER_FREE_MODELS


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


def test_build_review_plan_smart_codex_pass_falls_back_to_terra() -> None:
    """Retain Terra as the bounded availability fallback for smart reviews."""
    available = parse_available_models(AVAILABLE_MODELS)
    available.remove("openai-codex/gpt-5.6-sol")

    codex_pass, free_pool_pass = build_review_plan(
        ["correctness-review"],
        changed_lines=300,
        risk_reasons=(),
        available_models=available,
    )

    assert codex_pass.candidates == ("openai-codex/gpt-5.6-terra",)
    assert free_pool_pass.fallback_candidates == codex_pass.candidates


def test_build_review_plan_mechanical_codex_pass_does_not_fall_back_to_sol() -> None:
    """Fail closed instead of spending Sol on a mechanical checklist."""
    available = parse_available_models(AVAILABLE_MODELS)
    available.remove("openai-codex/gpt-5.6-terra")

    with pytest.raises(ValueError, match=r"code-health/codex"):
        build_review_plan(
            ["code-health"],
            changed_lines=300,
            risk_reasons=(),
            available_models=available,
        )


def test_build_review_plan_smart_pool_skips_unavailable_k3() -> None:
    """Fall back from unavailable K3 to the pinned OpenRouter models."""
    available = parse_available_models(AVAILABLE_MODELS)
    available.remove("kimi-coding/k3")
    available.remove("openrouter/tencent/hy3:free")

    plan = build_review_plan(
        ["correctness-review"],
        changed_lines=300,
        risk_reasons=(),
        available_models=available,
    )

    assert plan[1].candidates == ("openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",)
    assert plan[1].unavailable == (
        "kimi-coding/k3",
        "openrouter/tencent/hy3:free",
    )


def test_build_review_plan_empty_skills_raises_actionable_error() -> None:
    """Reject an empty review before provider or worker selection."""
    with pytest.raises(ValueError, match="skills must be non-empty"):
        build_review_plan(
            [],
            changed_lines=10,
            risk_reasons=(),
            available_models=parse_available_models(AVAILABLE_MODELS),
        )


def test_build_review_plan_missing_free_pool_raises_provider_error() -> None:
    """Reject the plan once when no free-pool model is registered with Pi."""
    available = {
        model
        for model in parse_available_models(AVAILABLE_MODELS)
        if model.startswith("openai-codex/")
    }

    with pytest.raises(ValueError, match=r"free-pool.*credentials required"):
        build_review_plan(
            ["code-health"],
            changed_lines=300,
            risk_reasons=(),
            available_models=available,
        )


def test_build_review_plan_mechanical_pool_requires_openrouter() -> None:
    """Do not substitute K3 when a mechanical checklist lacks free OpenRouter models."""
    available = {
        model
        for model in parse_available_models(AVAILABLE_MODELS)
        if not model.startswith("openrouter/")
    }

    with pytest.raises(ValueError, match=r"free-pool.*code-health") as error:
        build_review_plan(
            ["code-health"],
            changed_lines=300,
            risk_reasons=(),
            available_models=available,
        )

    assert "/login openrouter" in str(error.value)
    assert "kimi-coding" not in str(error.value)


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
    """Attribute pinned review models to the provider that produced the report."""
    assert provenance_for_model("openai-codex/gpt-5.6-sol") == "codex"
    assert (
        provenance_for_model("openrouter/nvidia/nemotron-3-ultra-550b-a55b:free") == "openrouter"
    )
    assert provenance_for_model("kimi-coding/k3") == "kimi-coding"


@pytest.mark.parametrize(
    "model",
    [
        "kimi-coding/other",
        "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
        "openrouter/paid-model",
    ],
)
def test_provenance_for_model_unpinned_free_pool_model_raises(model: str) -> None:
    """Reject selectors outside the exact pinned free-pool policy.

    :param model: Unpinned selector using an otherwise allowed provider.
    """
    with pytest.raises(ValueError, match="Unsupported Pi review model"):
        provenance_for_model(model)


def test_report_is_parseable_accepts_structured_json() -> None:
    """Accept a complete structured worker result."""
    report = json.dumps(
        {
            "skill": "code-health",
            "target": "PR #1",
            "findings": [],
            "what_looks_good": ["Clear data flow."],
        }
    )

    assert report_is_parseable(
        report,
        expected_skill="code-health",
        expected_target="PR #1",
    )


@pytest.mark.parametrize(
    "report",
    [
        "No output.",
        "## code-health review — PR #1",
        '{"skill":"code-health","target":"PR #1","findings":[],"what_looks_good":[]}',
        '{"skill":"code-health","target":"PR #1","findings":[],'
        '"what_looks_good":["Clear."],"unexpected":true}',
        '{"skill":"code-health","target":"PR #1","findings":['
        '{"severity":"BLOCK","path":"src/example.py","line":42,'
        '"description":"Defect."}],"what_looks_good":["Clear."]}',
        '{"skill":"code-health","target":"PR #1","findings":['
        '{"severity":"warn","path":"src/example.py","line":"42",'
        '"description":"Defect."}],"what_looks_good":["Clear."]}',
        '{"skill":"code-health","target":"PR #1","findings":['
        '{"severity":"warn","path":"../example.py","line":42,'
        '"description":"Defect."}],"what_looks_good":["Clear."]}',
        '{"skill":"code-health","target":"PR #1","findings":['
        '{"severity":"warn","path":"./src/example.py","line":42,'
        '"description":"Defect."}],"what_looks_good":["Clear."]}',
        '{"skill":"code-health","target":"PR #1","findings":['
        '{"severity":"warn","path":".","line":42,'
        '"description":"Defect."}],"what_looks_good":["Clear."]}',
        '{"skill":"code-health","target":"PR #1","findings":['
        '{"severity":"warn","path":"src/example.py","line":0,'
        '"description":"Defect."}],"what_looks_good":["Clear."]}',
        '{"skill":"code-health","target":"PR #1","findings":['
        '{"severity":"warn","path":"src/example.py","line":42,'
        '"description":" "}],"what_looks_good":["Clear."]}',
    ],
)
def test_report_is_parseable_rejects_invalid_structured_json(report: str) -> None:
    """Reject non-JSON and structurally invalid worker results.

    :param report: Invalid candidate worker result.
    """
    assert not report_is_parseable(
        report,
        expected_skill="code-health",
        expected_target="PR #1",
    )


def test_finding_fingerprint_normalizes_description_whitespace() -> None:
    """Identify the same late finding despite inconsequential prose spacing."""
    first = finding_fingerprint(
        skill="code-health",
        severity="warn",
        path="src/example.py",
        line=42,
        description="Concern with  extra spacing.",
    )
    second = finding_fingerprint(
        skill="code-health",
        severity="warn",
        path="src/example.py",
        line=42,
        description="Concern with extra spacing.",
    )

    assert first == second
    assert len(first) == 64


def test_parse_worker_report_rejects_duplicate_keys() -> None:
    """Reject last-write-wins JSON that could silently erase findings."""
    report = (
        '{"skill":"code-health","target":"PR #1","findings":['
        '{"severity":"block","path":"src/example.py","line":42,'
        '"description":"Defect."}],"findings":[],"what_looks_good":["Clear."]}'
    )

    with pytest.raises(ValueError, match="Duplicate JSON key: findings"):
        parse_worker_report(report, expected_skill="code-health", expected_target="PR #1")


def test_parse_worker_report_returns_validated_boundary_model() -> None:
    """Return typed report data after structural validation."""
    report = json.dumps(
        {
            "skill": "code-health",
            "target": "PR #1",
            "findings": [
                {
                    "severity": "warn",
                    "path": "src/example.py",
                    "line": 42,
                    "description": "A concrete concern.",
                }
            ],
            "what_looks_good": ["Clear data flow."],
        }
    )

    parsed = parse_worker_report(report, expected_skill="code-health", expected_target="PR #1")

    assert parsed.skill == "code-health"
    assert parsed.target == "PR #1"
    assert parsed.findings[0].line == 42
    assert parsed.findings[0].severity == "warn"


def test_report_is_parseable_rejects_wrong_skill_or_target() -> None:
    """Prevent a valid result for another assignment from entering the merge."""
    report = json.dumps(
        {
            "skill": "python-style",
            "target": "PR #1",
            "findings": [],
            "what_looks_good": ["Clear."],
        }
    )

    assert not report_is_parseable(
        report,
        expected_skill="code-health",
        expected_target="PR #1",
    )
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


def test_stream_host_events_persists_live_json_and_reports_safe_progress(
    tmp_path: Path,
) -> None:
    """Persist host events while exposing progress without tool arguments.

    :param tmp_path: Temporary location for the live host transcript.
    """
    source = io.StringIO(
        "\n".join(
            (
                '{"type":"message_start","message":{"role":"assistant",'
                '"content":[],"provider":"openrouter","model":"free-model"}}',
                '{"type":"tool_execution_start","toolName":"bash",'
                '"args":{"command":"printf secret-value"}}',
                '{"type":"auto_retry_start","attempt":1,"maxAttempts":3,'
                '"delayMs":1000,"errorMessage":"Authorization: Custom secret-token; '
                'API key is backup-secret; token expired: third-secret"}',
                '{"type":"message_end","message":{"role":"assistant",'
                '"content":[{"type":"text","text":"final report"}]}}',
            )
        )
        + "\n"
    )
    transcript = tmp_path / "host.jsonl"
    progress = io.StringIO()

    final_text = stream_host_events(source, transcript, progress)

    assert final_text == "final report"
    assert transcript.read_text() == source.getvalue()
    progress_text = progress.getvalue()
    assert "openrouter/free-model started" in progress_text
    assert "tool bash started" in progress_text
    assert "retry 1/3" in progress_text
    assert "<redacted>" in progress_text
    assert "secret-token" not in progress_text
    assert "backup-secret" not in progress_text
    assert "third-secret" not in progress_text
    assert "secret-value" not in progress_text


def test_stream_host_events_empty_notification_ack_preserves_deliverable(tmp_path: Path) -> None:
    """Ignore empty assistant acknowledgements caused by late background notifications.

    :param tmp_path: Temporary location for the live host transcript.
    """
    source = io.StringIO(
        '{"type":"message_end","message":{"role":"assistant",'
        '"content":"final report\\nSentinel: existing"}}\n'
        '{"type":"message_end","message":{"role":"custom","content":"worker finished"}}\n'
        '{"type":"message_end","message":{"role":"assistant","content":"Sentinel: late"}}\n'
    )

    assert (
        stream_host_events(source, tmp_path / "host.jsonl", io.StringIO())
        == "final report\nSentinel: existing"
    )


def test_stream_host_events_substantive_post_notification_response_replaces_draft(
    tmp_path: Path,
) -> None:
    """Keep a real final report that follows a worker completion notification.

    :param tmp_path: Temporary location for the live host transcript.
    """
    source = io.StringIO(
        '{"type":"message_end","message":{"role":"assistant","content":"draft"}}\n'
        '{"type":"message_end","message":{"role":"custom","content":"worker finished"}}\n'
        '{"type":"message_end","message":{"role":"assistant",'
        '"content":"# repo-review-full-no-comments — final"}}\n'
    )

    result = stream_host_events(source, tmp_path / "host.jsonl", io.StringIO())

    assert result == "# repo-review-full-no-comments — final"


def test_stream_host_events_empty_terminal_assistant_raises(tmp_path: Path) -> None:
    """Reject stale intermediate text when the terminal response is empty.

    :param tmp_path: Temporary location for the live host transcript.
    """
    source = io.StringIO(
        '{"type":"message_end","message":{"role":"assistant",'
        '"content":"intermediate report"}}\n'
        '{"type":"message_end","message":{"role":"assistant","content":[]}}\n'
    )

    with pytest.raises(ValueError, match="no final assistant text"):
        stream_host_events(source, tmp_path / "host.jsonl", io.StringIO())


def test_extract_report_returns_terminal_assistant_text_without_interpretation(
    tmp_path: Path,
) -> None:
    """Extract only final assistant text from Tintin JSONL.

    :param tmp_path: Temporary location for a transcript.
    """
    transcript = tmp_path / "worker.output"
    final_result = json.dumps(
        {
            "skill": "code-health",
            "target": "smoke",
            "findings": [],
            "what_looks_good": ["Clear."],
        }
    )
    transcript.write_text(
        '{"type":"session_start","sessionId":"metadata-only"}\n'
        '{"message":{"role":"assistant","content":[{"type":"text","text":"draft"}]}}\n'
        '{"message":{"role":"toolResult","content":[{"type":"text","text":"noise"}]}}\n'
        + json.dumps(
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hidden"},
                        {"type": "text", "text": final_result},
                    ],
                }
            }
        )
        + "\n"
    )

    assert extract_report(transcript) == final_result


def test_extract_report_empty_terminal_assistant_raises(tmp_path: Path) -> None:
    """Reject earlier output when the terminal assistant message is empty.

    :param tmp_path: Temporary location for a transcript.
    """
    transcript = tmp_path / "worker.output"
    transcript.write_text(
        '{"message":{"role":"assistant","content":"provisional"}}\n'
        '{"message":{"role":"assistant","content":[]}}\n'
    )

    with pytest.raises(ValueError, match="terminal assistant text"):
        extract_report(transcript)


def test_extract_report_selects_unique_json_object_from_narration(tmp_path: Path) -> None:
    """Discard harmless prose around one complete worker report.

    :param tmp_path: Temporary location for a transcript.
    """
    transcript = tmp_path / "worker.output"
    report = {
        "skill": "code-health",
        "target": "smoke",
        "findings": [],
        "what_looks_good": ["Clear."],
    }
    narrated = f"Review complete.\n```json\n{json.dumps(report)}\n```"
    transcript.write_text(
        json.dumps({"message": {"role": "assistant", "content": narrated}}) + "\n"
    )

    extracted = extract_report(transcript)

    assert json.loads(extracted) == report
    assert report_is_parseable(
        extracted,
        expected_skill="code-health",
        expected_target="smoke",
    )


def test_extract_report_multiple_worker_objects_raises(tmp_path: Path) -> None:
    """Reject ambiguous narration containing competing final reports.

    :param tmp_path: Temporary location for a transcript.
    """
    transcript = tmp_path / "worker.output"
    report = json.dumps(
        {
            "skill": "code-health",
            "target": "smoke",
            "findings": [],
            "what_looks_good": ["Clear."],
        }
    )
    transcript.write_text(
        json.dumps({"message": {"role": "assistant", "content": f"{report}\n{report}"}}) + "\n"
    )

    with pytest.raises(ValueError, match="multiple worker JSON objects"):
        extract_report(transcript)


def test_extract_report_returns_final_retraction_for_validation(tmp_path: Path) -> None:
    """Do not reuse an earlier result after the worker retracts it.

    :param tmp_path: Temporary location for a transcript.
    """
    transcript = tmp_path / "worker.output"
    transcript.write_text(
        '{"message":{"role":"assistant","content":"provisional"}}\n'
        '{"message":{"role":"assistant","content":"Tool failure; retract result."}}\n'
    )

    extracted = extract_report(transcript)

    assert extracted == "Tool failure; retract result."
    assert not report_is_parseable(
        extracted,
        expected_skill="code-health",
        expected_target="smoke",
    )


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


def test_report_repair_prompt_preserves_analysis_and_includes_diagnostic() -> None:
    """Ask the same worker for a format-only correction with actionable context."""
    report = '{"skill":"plugin:code-health","target":"PR #1"}'

    prompt = report_repair_prompt(
        report,
        expected_skill="code-health",
        expected_target="PR #1",
    )

    assert "Do not repeat the review" in prompt
    assert "Do not add, remove, or reinterpret findings" in prompt
    assert "Worker report identity does not match" in prompt or "Field required" in prompt
    assert report in prompt


def test_build_worker_prompt_contains_bounded_assignment_without_diff_duplication() -> None:
    """Generate the complete worker packet outside the host LLM."""
    prompt = build_worker_prompt(
        skill="correctness-review",
        target="PR #2174",
        repo="tinaudio/synth-setter",
        base_sha="a" * 40,
        head_sha="b" * 40,
        changed_paths=("agent/_shared/pi_review_routing.py", "tests/infra/test.py"),
    )

    assert "PR #2174" in prompt
    assert "correctness-review" in prompt
    assert "git diff " + "a" * 40 + ".." + "b" * 40 in prompt
    assert "agent/_shared/pi_review_routing.py" in prompt
    assert "exactly one JSON object" in prompt
    assert "Do not recursively discover" in prompt


def test_validate_report_cli_returns_nonzero_for_malformed_output(tmp_path: Path) -> None:
    """Expose report validation to the natural-language orchestrator.

    :param tmp_path: Temporary location for worker output.
    """
    report = tmp_path / "report.json"
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
    report = tmp_path / "worker.json"
    result = {
        "skill": "code-health",
        "target": "smoke",
        "findings": [],
        "what_looks_good": ["Clear."],
    }
    transcript.write_text(
        json.dumps(
            {
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": json.dumps(result)}],
                }
            }
        )
        + "\n"
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
    provenance = str(python(script, "provenance", "kimi-coding/k3")).strip()

    assert json.loads(report.read_text()) == result
    assert stats["turns"] == 1
    assert provenance == "kimi-coding"


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


def test_plan_cli_real_process_missing_free_pool_fails_once(tmp_path: Path) -> None:
    """Stop before expanding model candidates when no free-pool model is registered.

    :param tmp_path: Temporary location for the fake executable.
    """
    pi = tmp_path / "pi"
    codex_models = "openai-codex  gpt-5.6-terra  372K  128K  yes  yes\n"
    pi.write_text(f"#!/bin/sh\nprintf '%s' '{codex_models}'\n")
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

    stderr = error.value.stderr.decode()
    assert stderr.count("No free-pool models available") == 1
    assert "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free" not in stderr


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
    assert payload[1]["candidates"] == list(OPENROUTER_FREE_MODELS)
    assert payload[1]["model_tier"] == "mechanical"
