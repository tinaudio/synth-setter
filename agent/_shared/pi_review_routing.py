"""Build auditable model plans for Pi PR-review workers.

For example, inspect one checklist's available passes with::

    python3 agent/_shared/pi_review_routing.py plan \
      --skill code-health --changed-lines 20
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

import sh
from pydantic import BaseModel, Field, model_validator

DEEP_SKILLS = frozenset({"correctness-review", "lance-review"})
MECHANICAL_SKILLS = frozenset({"comment-hygiene", "python-style", "shell-style"})
SUPPORTED_SKILLS = frozenset(
    {
        "code-health",
        "comment-hygiene",
        "correctness-review",
        "gha-workflow-validator",
        "lance-review",
        "ml-data-pipeline",
        "ml-test",
        "python-style",
        "shell-style",
        "synth-setter-project-standards",
        "tdd-implementation",
        "tdd-refactor",
    }
)
PI_REVIEW_MAX_TURNS = 12
_MECHANICAL_LOW_LINE_LIMIT = 200
_HIGH_RISK_LINE_LIMIT = 800
_CODEX_SETUP = "authenticate with `/login openai-codex`"
_OPENROUTER_SETUP = "authenticate with `/login openrouter`"

_DEEP_CODEX_CANDIDATES = (
    "openai-codex/gpt-5.6-sol",
    "openai-codex/gpt-5.6-terra",
)
_STANDARD_CODEX_CANDIDATES = (
    "openai-codex/gpt-5.6-terra",
    "openai-codex/gpt-5.6-sol",
)
# Keep the routed OpenRouter review pool pinned to exact live free-model IDs.
# These lists intentionally exclude Anthropic-backed aliases and generic routers
# such as ``openrouter/free`` so the audit can report the exact reviewed model.
_DEEP_OPENROUTER_PRIMARY_CANDIDATES = (
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/google/gemma-4-31b-it:free",
    "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "openrouter/openai/gpt-oss-20b:free",
)
_DEEP_OPENROUTER_SECONDARY_FALLBACK_CANDIDATES = (
    "openrouter/google/gemma-4-26b-a4b-it:free",
    "openrouter/poolside/laguna-m.1:free",
    "openrouter/cohere/north-mini-code:free",
    "openrouter/nvidia/nemotron-3-nano-30b-a3b:free",
    "openrouter/tencent/hy3:free",
)
_STANDARD_OPENROUTER_PRIMARY_CANDIDATES = (
    "openrouter/cohere/north-mini-code:free",
    "openrouter/openai/gpt-oss-20b:free",
    "openrouter/google/gemma-4-31b-it:free",
    "openrouter/google/gemma-4-26b-a4b-it:free",
    "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
)
_STANDARD_OPENROUTER_SECONDARY_FALLBACK_CANDIDATES = (
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/poolside/laguna-m.1:free",
    "openrouter/nvidia/nemotron-3-nano-30b-a3b:free",
    "openrouter/tencent/hy3:free",
)
_REQUIRED_REPORT_HEADINGS = (
    "### BLOCK findings",
    "### WARN findings",
    "### What looks good",
)
_REPORT_TITLE = re.compile(r"^## (?P<skill>[a-z0-9-]+) review — (?P<target>.+)$")
_FINDING = re.compile(r"^\d+\. \*\*.+:\d+\*\* — \S.+$")
_RANGED_FINDING = re.compile(r"^(\d+\. \*\*.+:)(\d+)-(\d+)(\*\* — \S.+)$")
_BULLET_FINDING = re.compile(r"^- \*\*(.+:)(\d+)(?:-(\d+))?\*\* — (\S.+)$")


class _TranscriptContentBlock(BaseModel, strict=True, extra="ignore"):
    """One Tintin assistant-content block.

    .. attribute :: type
        :type: str

        Block discriminator.

    .. attribute :: text
        :type: str | None

        Text payload when the block contains report content.
    """

    type: str
    text: str | None = None


class _TranscriptUsage(BaseModel, strict=True, extra="ignore"):
    """Token accounting attached to one assistant turn.

    .. attribute :: total_tokens
        :type: int | None

        Provider-reported processed tokens for the turn.
    """

    total_tokens: int | None = Field(default=None, alias="totalTokens")


class _TranscriptMessage(BaseModel, strict=True, extra="ignore"):
    """Message payload from one Tintin transcript entry.

    .. attribute :: role
        :type: str

        Conversation role.

    .. attribute :: content
        :type: str | list[_TranscriptContentBlock]

        Raw string or structured content blocks.

    .. attribute :: usage
        :type: _TranscriptUsage | None

        Optional provider token accounting.

    .. attribute :: provider
        :type: str | None

        Effective provider for host lifecycle events.

    .. attribute :: model
        :type: str | None

        Effective model for host lifecycle events.
    """

    role: str
    content: str | list[_TranscriptContentBlock]
    usage: _TranscriptUsage | None = None
    provider: str | None = None
    model: str | None = None


class _TranscriptEntry(BaseModel, strict=True, extra="ignore"):
    """Validated trust-boundary shape for one Tintin JSONL row.

    .. attribute :: type
        :type: str | None

        Event discriminator for metadata-only rows.

    .. attribute :: message
        :type: _TranscriptMessage | None

        Conversation message, or ``None`` for typed metadata events.

    .. attribute :: timestamp
        :type: str | None

        Optional ISO-8601 event timestamp.
    """

    type: str | None = None
    message: _TranscriptMessage | None = None
    timestamp: str | None = None

    @model_validator(mode="after")
    def _require_message_or_type(self) -> _TranscriptEntry:
        """Reject rows that cannot be classified as messages or metadata.

        :returns: Validated transcript row.
        :raises ValueError: If both the message and event type are absent.
        """
        if self.message is None and not self.type:
            raise ValueError("Transcript row requires a message or event type")
        return self


class _HostEvent(BaseModel, strict=True, extra="ignore"):
    """Validated trust-boundary shape for one Pi host JSON event.

    .. attribute :: type
        :type: str

        Event discriminator.

    .. attribute :: message
        :type: _TranscriptMessage | None

        Message lifecycle payload.

    .. attribute :: tool_name
        :type: str | None

        Tool lifecycle name.

    .. attribute :: is_error
        :type: bool | None

        Whether tool execution failed.

    .. attribute :: attempt
        :type: int | None

        Current provider retry number.

    .. attribute :: max_attempts
        :type: int | None

        Provider retry limit.

    .. attribute :: error_message
        :type: str | None

        Provider retry diagnostic.
    """

    type: str
    message: _TranscriptMessage | None = None
    tool_name: str | None = Field(default=None, alias="toolName")
    is_error: bool | None = Field(default=None, alias="isError")
    attempt: int | None = None
    max_attempts: int | None = Field(default=None, alias="maxAttempts")
    error_message: str | None = Field(default=None, alias="errorMessage")


class WorkerReport(BaseModel, strict=True, extra="forbid"):
    """Validated worker-report fields consumed by aggregation.

    .. attribute :: skill
        :type: str

        Checklist that produced the report.

    .. attribute :: target
        :type: str

        Assigned PR or branch label.

    .. attribute :: block_findings
        :type: tuple[str, ...]

        Validated blocking-finding rows.

    .. attribute :: warn_findings
        :type: tuple[str, ...]

        Validated warning-finding rows.

    .. attribute :: what_looks_good
        :type: tuple[str, ...]

        Evidence bullets from the worker.
    """

    skill: str
    target: str
    block_findings: tuple[str, ...]
    warn_findings: tuple[str, ...]
    what_looks_good: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TranscriptStats:
    """Audit statistics derived from one Tintin transcript.

    .. attribute :: turns
        :type: int

        Assistant turns recorded in the transcript.

    .. attribute :: elapsed_seconds
        :type: int | None

        Wall-clock span, or ``None`` when timestamps are unavailable.

    .. attribute :: cumulative_tokens
        :type: int | None

        Sum of per-turn counters, or ``None`` when usage is unavailable.
    """

    turns: int
    elapsed_seconds: int | None
    cumulative_tokens: int | None


@dataclass(frozen=True, slots=True)
class ReviewPass:
    """One skill/model-family pass and its available candidate sequence.

    .. attribute :: skill
        :type: str

        Authoritative checklist name.

    .. attribute :: pass_name
        :type: str

        Logical Codex or OpenRouter pass.

    .. attribute :: candidates
        :type: tuple[str, ...]

        Available models in attempt order.

    .. attribute :: unavailable
        :type: tuple[str, ...]

        Configured models absent from Pi's registry.

    .. attribute :: secondary_fallback_candidates
        :type: tuple[str, ...]

        Same-provider OpenRouter free models reserved for the secondary fallback
        tier after the primary free-model tier is exhausted.

    .. attribute :: fallback_candidates
        :type: tuple[str, ...]

        Codex models used only after an OpenRouter pass exhausts both free-model
        tiers. The orchestrator reorders them around the effective Codex-pass model.

    .. attribute :: thinking
        :type: str

        Pi thinking level for every attempt.

    .. attribute :: reason
        :type: str

        Auditable explanation for the thinking allocation.

    .. attribute :: max_turns
        :type: int

        Hard turn budget passed to Tintin for every attempt.
    """

    skill: str
    pass_name: str
    candidates: tuple[str, ...]
    unavailable: tuple[str, ...]
    secondary_fallback_candidates: tuple[str, ...]
    fallback_candidates: tuple[str, ...]
    thinking: str
    reason: str
    max_turns: int


def parse_available_models(output: str) -> set[str]:
    """Parse ``pi --list-models`` output into canonical selectors.

    :param output: Whitespace-delimited Pi model table.
    :returns: Available ``provider/model-id`` selectors.
    """
    models: set[str] = set()
    for line in output.splitlines():
        columns = line.split()
        if len(columns) >= 2 and columns[0] != "provider":
            models.add(f"{columns[0]}/{columns[1]}")
    return models


def _transcript_entries(transcript: Path) -> list[_TranscriptEntry]:
    """Read validated non-empty transcript rows.

    :param transcript: Tintin JSONL output path containing worker events.
    :returns: Validated transcript rows in file order.
    """
    return [
        _TranscriptEntry.model_validate_json(raw_line)
        for raw_line in transcript.read_text().splitlines()
        if raw_line.strip()
    ]


def _message_text(message: _TranscriptMessage) -> str:
    """Return concatenated text blocks from one Pi message.

    :param message: Validated Pi message.
    :returns: Plain text content in block order.
    """
    if isinstance(message.content, str):
        return message.content
    return "".join(block.text or "" for block in message.content if block.type == "text")


def _redact_diagnostic(diagnostic: str) -> str:
    """Remove credential-shaped values from a provider diagnostic.

    :param diagnostic: Raw retry diagnostic emitted by Pi.
    :returns: Diagnostic safe for the terminal progress stream.
    """
    redacted = re.sub(
        r"(?i)\b(authorization)(\s*:\s*)(?:bearer\s+)?\S+",
        r"\1\2<redacted>",
        diagnostic,
    )
    return re.sub(
        r"(?i)\b(bearer|api[-_ ]?key|token)([\s:=\"']+)\S+",
        r"\1\2<redacted>",
        redacted,
    )


def stream_host_events(source: TextIO, transcript: Path, progress: TextIO) -> str:
    """Persist Pi JSON events live and emit a sanitized progress projection.

    :param source: Pi's newline-delimited JSON event stream.
    :param transcript: Destination for the authoritative raw event stream.
    :param progress: Terminal stream for sanitized lifecycle updates.
    :returns: Final assistant text for the host caller.
    :raises ValueError: If an event is malformed or no final response exists.
    """
    final_text = ""
    with transcript.open("w") as transcript_file:
        for raw_line in source:
            transcript_file.write(raw_line)
            transcript_file.flush()
            if not raw_line.strip():
                continue
            event = _HostEvent.model_validate_json(raw_line)
            if event.type == "message_start" and event.message is not None:
                message = event.message
                if message.role == "assistant" and message.provider and message.model:
                    progress.write(f"[pi-review] {message.provider}/{message.model} started\n")
            elif event.type == "tool_execution_start" and event.tool_name:
                progress.write(f"[pi-review] tool {event.tool_name} started\n")
            elif event.type == "tool_execution_end" and event.tool_name:
                outcome = "failed" if event.is_error else "finished"
                progress.write(f"[pi-review] tool {event.tool_name} {outcome}\n")
            elif event.type == "auto_retry_start":
                attempt = event.attempt if event.attempt is not None else "?"
                maximum = event.max_attempts if event.max_attempts is not None else "?"
                diagnostic = _redact_diagnostic(event.error_message or "unknown error")
                progress.write(f"[pi-review] retry {attempt}/{maximum}: {diagnostic}\n")
            elif event.type == "message_end" and event.message is not None:
                if event.message.role == "assistant":
                    text = _message_text(event.message)
                    if text.strip():
                        final_text = text
            progress.flush()
    if not final_text:
        raise ValueError(f"Pi host transcript has no final assistant text: {transcript}")
    return final_text


def _structured_report_markdown(text: str) -> str | None:
    """Remove narration surrounding one structured worker report.

    :param text: Assistant text that may wrap a report in prose.
    :returns: Structured Markdown, or ``None`` when no report title exists.
    """
    lines = text.strip().splitlines()
    start = next(
        (index for index, line in enumerate(lines) if _REPORT_TITLE.fullmatch(line)),
        None,
    )
    if start is None:
        return None
    report_lines = lines[start:]
    if any(report_lines.count(heading) != 1 for heading in _REQUIRED_REPORT_HEADINGS):
        return "\n".join(report_lines).strip()
    indices = [report_lines.index(heading) for heading in _REQUIRED_REPORT_HEADINGS]
    if indices != sorted(indices):
        return "\n".join(report_lines).strip()

    block_lines = _normalized_finding_lines(report_lines[indices[0] + 1 : indices[1]])
    warn_lines = _normalized_finding_lines(report_lines[indices[1] + 1 : indices[2]])
    good_lines = [line for line in report_lines[indices[2] + 1 :] if line.startswith("- ")]
    sections = (
        ("### BLOCK findings", block_lines),
        ("### WARN findings", warn_lines),
        ("### What looks good", good_lines),
    )
    output = [report_lines[0]]
    for heading, content in sections:
        output.extend(("", heading, *content))
    return "\n".join(output).strip()


def _normalized_finding_lines(lines: Sequence[str]) -> list[str]:
    """Keep contract findings and anchor ranges at their first changed line.

    :param lines: Raw lines from a BLOCK or WARN section.
    :returns: Canonical finding lines with surrounding narration removed.
    :raises ValueError: If a finding-like line violates the report contract.
    """
    normalized: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not stripped:
            continue
        if stripped in {"None", "None.", "- None", "- None."}:
            normalized.append("None.")
            continue
        if _FINDING.fullmatch(stripped):
            normalized.append(stripped)
            continue
        ranged = _RANGED_FINDING.fullmatch(stripped)
        if ranged is not None:
            prefix, start, end, suffix = ranged.groups()
            normalized.append(
                f"{prefix}{start}{suffix.replace(' — ', f' — [reported range {start}-{end}] ', 1)}"
            )
            continue
        bullet = _BULLET_FINDING.fullmatch(stripped)
        if bullet is not None:
            path, start, end, description = bullet.groups()
            range_note = f"[reported range {start}-{end}] " if end is not None else ""
            normalized.append(
                f"{len(normalized) + 1}. **{path}{start}** — {range_note}{description}"
            )
            continue
        if "**" in stripped:
            raise ValueError(f"malformed finding-like line: {stripped}")
    return [
        re.sub(r"^\d+\.", f"{index}.", finding) if _FINDING.fullmatch(finding) else finding
        for index, finding in enumerate(normalized, start=1)
    ]


def extract_report(transcript: Path) -> str:
    """Extract the final assistant Markdown from a Tintin transcript.

    :param transcript: Tintin JSONL output path returned by ``Agent``.
    :returns: Final non-empty assistant text.
    :raises ValueError: If no assistant report text exists.
    """
    latest = ""
    saw_assistant_text = False
    for entry in _transcript_entries(transcript):
        if entry.message is None or entry.message.role != "assistant":
            continue
        text = _message_text(entry.message)
        if not text.strip():
            continue
        saw_assistant_text = True
        structured = _structured_report_markdown(text)
        latest = structured or ""
    if not latest:
        detail = (
            "final assistant text is not a report"
            if saw_assistant_text
            else "has no assistant text"
        )
        raise ValueError(f"Transcript {detail}: {transcript}")
    return latest


def transcript_stats(transcript: Path) -> TranscriptStats:
    """Summarize a Tintin transcript for the sentinel audit.

    :param transcript: Tintin JSONL output path returned by ``Agent``.
    :returns: Assistant turns, elapsed seconds, and cumulative processed tokens.
    """
    timestamps: list[datetime] = []
    turns = 0
    cumulative_tokens = 0
    usage_complete = True
    for entry in _transcript_entries(transcript):
        if entry.timestamp is not None:
            timestamps.append(datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00")))
        if entry.message is None or entry.message.role != "assistant":
            continue
        turns += 1
        usage = entry.message.usage
        if usage is None or usage.total_tokens is None:
            usage_complete = False
        else:
            cumulative_tokens += usage.total_tokens
    elapsed_seconds = None
    if len(timestamps) >= 2:
        elapsed_seconds = int((timestamps[-1] - timestamps[0]).total_seconds())
    return TranscriptStats(
        turns=turns,
        elapsed_seconds=elapsed_seconds,
        cumulative_tokens=cumulative_tokens if turns and usage_complete else None,
    )


def provenance_for_model(model: str) -> str:
    """Return finding provenance from the model that produced it.

    :param model: Canonical ``provider/model-id`` selector.
    :returns: ``codex`` or ``openrouter``.
    :raises ValueError: If the provider is outside the review policy.
    """
    provider = model.split("/", 1)[0]
    if provider == "openai-codex":
        return "codex"
    if provider == "openrouter":
        return "openrouter"
    raise ValueError(f"Unsupported Pi review provider: {provider}")


def parse_worker_report(report: str, *, expected_skill: str, expected_target: str) -> WorkerReport:
    """Parse a structurally valid worker report into its boundary model.

    :param report: Worker Markdown output.
    :param expected_skill: Checklist the worker was assigned.
    :param expected_target: PR or branch label the worker was assigned.
    :returns: Validated report data consumed by aggregation.
    :raises ValueError: If identity or report structure is invalid.
    """
    lines = report.strip().splitlines()
    if not lines:
        raise ValueError("Worker report is empty")
    title = _REPORT_TITLE.fullmatch(lines[0])
    if (
        title is None
        or title.group("skill") != expected_skill
        or title.group("target") != expected_target
    ):
        raise ValueError("Worker report identity does not match its assignment")
    if any(lines.count(heading) != 1 for heading in _REQUIRED_REPORT_HEADINGS):
        raise ValueError("Worker report headings are missing or duplicated")
    indices = [lines.index(heading) for heading in _REQUIRED_REPORT_HEADINGS]
    if indices != sorted(indices) or any(line.strip() for line in lines[1 : indices[0]]):
        raise ValueError("Worker report headings are out of order")

    block_lines = lines[indices[0] + 1 : indices[1]]
    warn_lines = lines[indices[1] + 1 : indices[2]]
    good_lines = [line for line in lines[indices[2] + 1 :] if line.strip()]
    if not (
        _findings_section_is_valid(block_lines)
        and _findings_section_is_valid(warn_lines)
        and bool(good_lines)
        and all(line.startswith("- ") and len(line) > 2 for line in good_lines)
    ):
        raise ValueError("Worker report sections are malformed")
    return WorkerReport(
        skill=title.group("skill"),
        target=title.group("target"),
        block_findings=tuple(line for line in block_lines if line.strip()),
        warn_findings=tuple(line for line in warn_lines if line.strip()),
        what_looks_good=tuple(good_lines),
    )


def report_is_parseable(report: str, *, expected_skill: str, expected_target: str) -> bool:
    """Return whether a worker report satisfies the merge contract.

    :param report: Worker Markdown output.
    :param expected_skill: Checklist the worker was assigned.
    :param expected_target: PR or branch label the worker was assigned.
    :returns: Whether identity and ordered sections are structurally valid.
    """
    try:
        parse_worker_report(
            report,
            expected_skill=expected_skill,
            expected_target=expected_target,
        )
    except ValueError:
        return False
    return True


def _findings_section_is_valid(lines: Sequence[str]) -> bool:
    """Return whether a findings section satisfies the worker-report contract.

    :param lines: Raw non-heading lines from one findings section.
    :returns: Whether the section is empty-free and contains valid finding rows.
    """
    content = [line for line in lines if line.strip()]
    if content in (["None."], ["None"], ["- None."], ["- None"]):
        return True
    if not content:
        return False

    return all(_FINDING.fullmatch(line) for line in content)


def _available_and_unavailable(
    configured: Sequence[str],
    available_models: AbstractSet[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split configured model selectors by current Pi registry availability.

    :param configured: Ordered model selectors from the routing policy.
    :param available_models: Canonical selectors returned by Pi's model registry.
    :returns: Ordered available and unavailable selectors.
    """
    available = tuple(model for model in configured if model in available_models)
    unavailable = tuple(model for model in configured if model not in available_models)
    return available, unavailable


def build_review_plan(
    skills: Sequence[str],
    *,
    changed_lines: int,
    risk_reasons: Sequence[str],
    available_models: AbstractSet[str],
) -> list[ReviewPass]:
    """Allocate model candidates and thinking to selected review skills.

    :param skills: Selected authoritative review checklists.
    :param changed_lines: Total added and deleted lines in the diff.
    :param risk_reasons: Named risk signals detected in the diff.
    :param available_models: Canonical selectors from Pi's model registry.
    :returns: Two ordered passes per skill, preserving the supplied skill order.
    :raises ValueError: If no skills are selected, inputs are invalid, or a required provider is
        absent from Pi's registry.
    """
    if not skills:
        raise ValueError("skills must be non-empty")
    if changed_lines < 0:
        raise ValueError("changed_lines must be non-negative")
    unknown = sorted(set(skills) - SUPPORTED_SKILLS)
    if unknown:
        raise ValueError(f"Unknown review skill(s): {', '.join(unknown)}")
    _require_codex(available_models)
    _require_openrouter(available_models)
    plan: list[ReviewPass] = []
    for skill in skills:
        is_deep = skill in DEEP_SKILLS
        codex_configured = _DEEP_CODEX_CANDIDATES if is_deep else _STANDARD_CODEX_CANDIDATES
        openrouter_primary_configured = (
            _DEEP_OPENROUTER_PRIMARY_CANDIDATES
            if is_deep
            else _STANDARD_OPENROUTER_PRIMARY_CANDIDATES
        )
        openrouter_secondary_configured = (
            _DEEP_OPENROUTER_SECONDARY_FALLBACK_CANDIDATES
            if is_deep
            else _STANDARD_OPENROUTER_SECONDARY_FALLBACK_CANDIDATES
        )
        thinking, reason = _thinking_for(
            skill,
            changed_lines=changed_lines,
            risk_reasons=risk_reasons,
        )
        codex_candidates, codex_unavailable = _available_and_unavailable(
            codex_configured,
            available_models,
        )
        if not codex_candidates:
            raise ValueError(f"No available models remain for {skill}/codex")
        codex_label = "codex"
        plan.append(
            ReviewPass(
                skill=skill,
                pass_name=codex_label,
                candidates=codex_candidates,
                unavailable=codex_unavailable,
                secondary_fallback_candidates=(),
                fallback_candidates=(),
                thinking=thinking,
                reason=reason,
                max_turns=PI_REVIEW_MAX_TURNS,
            )
        )
        openrouter_candidates, openrouter_primary_unavailable = _available_and_unavailable(
            openrouter_primary_configured,
            available_models,
        )
        (
            openrouter_secondary_candidates,
            openrouter_secondary_unavailable,
        ) = _available_and_unavailable(
            openrouter_secondary_configured,
            available_models,
        )
        openrouter_label = "openrouter"
        plan.append(
            ReviewPass(
                skill=skill,
                pass_name=openrouter_label,
                candidates=openrouter_candidates,
                unavailable=(
                    *openrouter_primary_unavailable,
                    *openrouter_secondary_unavailable,
                ),
                secondary_fallback_candidates=openrouter_secondary_candidates,
                fallback_candidates=tuple(reversed(codex_candidates)),
                thinking=thinking,
                reason=reason,
                max_turns=PI_REVIEW_MAX_TURNS,
            )
        )
    return plan


def _require_codex(available_models: AbstractSet[str]) -> None:
    """Require a registered Codex model for the always-available fallback.

    :param available_models: Canonical selectors returned by Pi's model registry.
    :raises ValueError: If Codex has no available model.
    """
    if not any(model.startswith("openai-codex/") for model in available_models):
        raise ValueError(f"No openai-codex models available; {_CODEX_SETUP}; credentials required")


def _require_openrouter(available_models: AbstractSet[str]) -> None:
    """Require OpenRouter registration before expanding free-model candidates.

    :param available_models: Canonical selectors returned by Pi's model registry.
    :raises ValueError: If OpenRouter has no registered model.
    """
    if not any(model.startswith("openrouter/") for model in available_models):
        raise ValueError(
            f"No OpenRouter models available; {_OPENROUTER_SETUP}; credentials required"
        )


def _thinking_for(
    skill: str,
    *,
    changed_lines: int,
    risk_reasons: Sequence[str],
) -> tuple[str, str]:
    """Choose a thinking level and auditable reason for one review pass.

    :param skill: Authoritative checklist name.
    :param changed_lines: Total added and deleted lines in the diff.
    :param risk_reasons: Named risk signals detected in the diff.
    :returns: Selected thinking level and its allocation rationale.
    """
    if skill in DEEP_SKILLS:
        return "high", "deep checklist"

    if skill in MECHANICAL_SKILLS:
        if changed_lines < _MECHANICAL_LOW_LINE_LIMIT:
            return "low", f"mechanical checklist on diff under {_MECHANICAL_LOW_LINE_LIMIT} lines"
        return "medium", f"mechanical checklist on diff of {_MECHANICAL_LOW_LINE_LIMIT}+ lines"

    risks = list(risk_reasons)
    if changed_lines > _HIGH_RISK_LINE_LIMIT:
        risks.insert(0, f"diff over {_HIGH_RISK_LINE_LIMIT} lines")
    if risks:
        return "high", f"risk: {', '.join(risks)}"
    return "medium", "standard checklist"


def _build_parser() -> argparse.ArgumentParser:
    """Build the routing command-line parser.

    :returns: Parser for planning, report, audit, and provenance commands.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="print the available review plan as JSON")
    plan.add_argument("--skill", action="append", required=True)
    plan.add_argument("--changed-lines", type=int, required=True)
    plan.add_argument("--risk", action="append", default=[])
    extract = subparsers.add_parser(
        "extract-report", help="write final assistant Markdown from Tintin JSONL"
    )
    extract.add_argument("transcript", type=Path)
    extract.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser(
        "validate-report", help="check a worker report's section contract"
    )
    validate.add_argument("path", type=Path)
    validate.add_argument("--skill", required=True, choices=sorted(SUPPORTED_SKILLS))
    validate.add_argument("--target", required=True)
    stats = subparsers.add_parser(
        "transcript-stats", help="print Tintin runtime-budget statistics as JSON"
    )
    stats.add_argument("transcript", type=Path)
    provenance = subparsers.add_parser(
        "provenance", help="print provenance for an effective model"
    )
    provenance.add_argument("model")
    stream = subparsers.add_parser(
        "stream-host", help="persist Pi host JSON while reporting safe progress"
    )
    stream.add_argument("--transcript", type=Path, required=True)
    return parser


def _print_plan(args: argparse.Namespace) -> None:
    """Build and print a plan from parsed CLI arguments.

    :param args: Parsed ``plan`` arguments.
    :raises RuntimeError: If Pi is missing or cannot list models.
    """
    pi_executable = shutil.which("pi")
    if pi_executable is None:
        raise RuntimeError("pi executable not found on PATH")
    try:
        model_output = str(sh.Command(pi_executable)("--list-models"))
    except sh.ErrorReturnCode as error:
        stderr = error.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"pi --list-models failed: {stderr}") from error
    plan = build_review_plan(
        args.skill,
        changed_lines=args.changed_lines,
        risk_reasons=args.risk,
        available_models=parse_available_models(model_output),
    )
    sys.stdout.write(f"{json.dumps([asdict(item) for item in plan], indent=2)}\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the routing CLI.

    :param argv: Optional command arguments for tests or embedding.
    :returns: Process exit status.
    :raises AssertionError: If argument parsing returns an unknown command.
    """
    args = _build_parser().parse_args(argv)
    if args.command == "plan":
        _print_plan(args)
        return 0
    if args.command == "extract-report":
        args.output.write_text(f"{extract_report(args.transcript)}\n")
        return 0
    if args.command == "validate-report":
        return (
            0
            if report_is_parseable(
                args.path.read_text(),
                expected_skill=args.skill,
                expected_target=args.target,
            )
            else 1
        )
    if args.command == "transcript-stats":
        sys.stdout.write(f"{json.dumps(asdict(transcript_stats(args.transcript)), indent=2)}\n")
        return 0
    if args.command == "provenance":
        sys.stdout.write(f"{provenance_for_model(args.model)}\n")
        return 0
    if args.command == "stream-host":
        sys.stdout.write(f"{stream_host_events(sys.stdin, args.transcript, sys.stderr)}\n")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
