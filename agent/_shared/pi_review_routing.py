"""Build auditable model plans for Pi PR-review workers.

For example, inspect one checklist's available passes with::

    python3 agent/_shared/pi_review_routing.py plan \
      --skill code-health --changed-lines 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, TextIO

import sh
from pydantic import BaseModel, Field, model_validator

_REPORT_KEYS = frozenset({"findings", "skill", "target", "what_looks_good"})
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

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
_FREE_POOL_SETUP = "authenticate with `/login kimi-coding` or `/login openrouter`"

_DEEP_CODEX_CANDIDATES = (
    "openai-codex/gpt-5.6-sol",
    "openai-codex/gpt-5.6-terra",
)
_STANDARD_CODEX_CANDIDATES = (
    "openai-codex/gpt-5.6-terra",
    "openai-codex/gpt-5.6-sol",
)
# Keep this ordered pool pinned so audit provenance records each provider and exact model.
_FREE_POOL_CANDIDATES = (
    "kimi-coding/k3",
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/tencent/hy3:free",
)


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


class WorkerFinding(BaseModel, strict=True, extra="forbid"):
    """One structured finding returned by a review worker.

    .. attribute :: severity
        :type: Literal["block", "warn"]

        Merge severity assigned by the checklist.

    .. attribute :: path
        :type: str

        Repository-relative changed file path.

    .. attribute :: line
        :type: int

        Positive changed-line anchor.

    .. attribute :: description
        :type: str

        Self-contained failure scenario or concern.
    """

    severity: Literal["block", "warn"]
    path: str
    line: int = Field(gt=0)
    description: str

    @model_validator(mode="after")
    def _require_content(self) -> WorkerFinding:
        """Reject findings that cannot be anchored or explained.

        :returns: Validated finding.
        :raises ValueError: If the path or description is empty or unsafe.
        """
        path = Path(self.path)
        is_canonical = path.as_posix() == self.path and "\\" not in self.path
        if (
            not self.path.strip()
            or self.path == "."
            or path.is_absolute()
            or ".." in path.parts
            or not is_canonical
        ):
            raise ValueError("Finding path must be canonical and repository-relative")
        if not self.description.strip():
            raise ValueError("Finding description must be non-empty")
        return self


class WorkerReport(BaseModel, strict=True, extra="forbid"):
    """Structured worker result consumed by aggregation.

    .. attribute :: skill
        :type: str

        Checklist that produced the report.

    .. attribute :: target
        :type: str

        Assigned PR or branch label.

    .. attribute :: findings
        :type: tuple[WorkerFinding, ...]

        Typed blocking and warning findings.

    .. attribute :: what_looks_good
        :type: tuple[str, ...]

        Positive evidence from the reviewed diff.
    """

    skill: str
    target: str
    findings: tuple[WorkerFinding, ...]
    what_looks_good: tuple[str, ...]

    @model_validator(mode="after")
    def _require_content(self) -> WorkerReport:
        """Reject reports without assignment identity or positive evidence.

        :returns: Validated worker report.
        :raises ValueError: If identity or positive evidence is empty.
        """
        if not self.skill.strip() or not self.target.strip():
            raise ValueError("Worker report identity must be non-empty")
        if not self.what_looks_good or any(not item.strip() for item in self.what_looks_good):
            raise ValueError("Worker report requires positive evidence")
        return self


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

        Logical ``codex`` or ``free-pool`` pass.

    .. attribute :: candidates
        :type: tuple[str, ...]

        Available models in attempt order.

    .. attribute :: unavailable
        :type: tuple[str, ...]

        Configured models absent from Pi's registry.

    .. attribute :: fallback_candidates
        :type: tuple[str, ...]

        Codex models used only after a free-pool pass exhausts its candidates.
        The orchestrator reorders them around the effective Codex-pass model.

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
        r"(?i)\b(authorization)(\s*:\s*)[^\r\n;,]+",
        r"\1\2<redacted>",
        diagnostic,
    )
    return re.sub(
        r"(?i)\b(bearer|api[-_ ]?key|token)\b"
        r"((?:\s+(?:is|expired))?\s*[:=\"']*\s*)\S+",
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
    notification_pending = False
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
                if event.message.role == "custom":
                    notification_pending = True
                elif event.message.role == "assistant":
                    assistant_text = _message_text(event.message)
                    has_deliverable = bool(final_text.strip())
                    if not notification_pending or not has_deliverable:
                        final_text = assistant_text
                    notification_pending = False
            progress.flush()
    if not final_text.strip():
        raise ValueError(f"Pi host transcript has no final assistant text: {transcript}")
    return final_text


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build one JSON object while rejecting ambiguous duplicate keys.

    :param pairs: Decoder-preserved object members in source order.
    :returns: Object mapping when every key is unique.
    :raises ValueError: If a key appears more than once.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_loads(value: str) -> object:
    """Decode JSON without last-write-wins duplicate-key behavior.

    :param value: Candidate JSON text.
    :returns: Decoded JSON value.
    :raises ValueError: If syntax or object keys are invalid.
    """
    try:
        return json.loads(value, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid worker JSON: {error.msg}") from error


def _extract_report_envelope(value: str) -> str:
    """Strip harmless text when one report-shaped JSON object is present.

    :param value: Terminal worker text that may contain narration or a Markdown fence.
    :returns: Unique report object, or unchanged text for correction when none is complete.
    :raises ValueError: If competing report objects make the result ambiguous.
    """
    decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_keys)
    candidates: list[str] = []
    for start, character in enumerate(value):
        if character != "{":
            continue
        try:
            decoded, end = decoder.raw_decode(value, start)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(decoded, dict) and frozenset(decoded) == _REPORT_KEYS:
            candidates.append(value[start:end])
    if not candidates:
        return value.strip()
    if len(candidates) > 1:
        raise ValueError("Terminal assistant text contains multiple worker JSON objects")
    return candidates[0]


def extract_report(transcript: Path) -> str:
    """Extract one unambiguous worker JSON object from a Tintin transcript.

    :param transcript: Tintin JSONL output path returned by ``Agent``.
    :returns: Unique JSON object, or raw terminal text for same-session correction.
    :raises ValueError: If the terminal assistant message is empty or contains competing reports.
    """
    latest: str | None = None
    for entry in _transcript_entries(transcript):
        if entry.message is not None and entry.message.role == "assistant":
            latest = _message_text(entry.message).strip()
    if not latest:
        raise ValueError(f"Transcript has no terminal assistant text: {transcript}")
    return _extract_report_envelope(latest)


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


def finding_fingerprint(
    *, skill: str, severity: str, path: str, line: int, description: str
) -> str:
    """Return a stable identity for foreground/aftercare finding deduplication.

    :param skill: Checklist that produced the finding.
    :param severity: Finding severity.
    :param path: Repository-relative finding path.
    :param line: Positive finding anchor.
    :param description: Self-contained finding text.
    :returns: Lowercase SHA-256 digest of normalized finding content.
    """
    normalized = {
        "description": " ".join(description.split()),
        "line": line,
        "path": path,
        "severity": severity,
        "skill": skill,
    }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def provenance_for_model(model: str) -> str:
    """Return finding provenance from the model that produced it.

    :param model: Canonical ``provider/model-id`` selector.
    :returns: ``codex`` for Codex models, else the pinned free-pool provider.
    :raises ValueError: If the model is outside the review policy.
    """
    provider = model.split("/", 1)[0]
    if provider == "openai-codex":
        return "codex"
    if model in _FREE_POOL_CANDIDATES:
        return provider
    raise ValueError(f"Unsupported Pi review model: {model}")


def parse_worker_report(report: str, *, expected_skill: str, expected_target: str) -> WorkerReport:
    """Parse a worker's structured JSON result into its boundary model.

    :param report: Worker JSON output.
    :param expected_skill: Checklist the worker was assigned.
    :param expected_target: PR or branch label the worker was assigned.
    :returns: Validated report data consumed by aggregation.
    :raises ValueError: If JSON, identity, or report fields are invalid.
    """
    strict_json = json.dumps(_strict_json_loads(report))
    parsed = WorkerReport.model_validate_json(strict_json)
    if parsed.skill != expected_skill or parsed.target != expected_target:
        raise ValueError("Worker report identity does not match its assignment")
    return parsed


def report_repair_prompt(report: str, *, expected_skill: str, expected_target: str) -> str:
    """Build a format-only correction prompt for the worker that wrote a bad report.

    :param report: Extracted report text that failed validation.
    :param expected_skill: Checklist assigned to the worker.
    :param expected_target: Target assigned to the worker.
    :returns: Prompt suitable for one same-session resume turn.
    """
    try:
        parse_worker_report(
            report,
            expected_skill=expected_skill,
            expected_target=expected_target,
        )
    except ValueError as error:
        diagnostic = str(error)
    else:
        diagnostic = "The report is already valid; return it unchanged."
    return (
        "Correct only the structured report from your preceding response. "
        "Do not repeat the review or use tools. Do not add, remove, or reinterpret findings. "
        f"The assigned skill is {expected_skill!r} and target is {expected_target!r}. "
        f"Validation diagnostic: {diagnostic}\n"
        "Return exactly one JSON object with no Markdown fence or surrounding prose:\n"
        f"{report.strip()}"
    )


def build_worker_prompt(
    *,
    skill: str,
    target: str,
    repo: str,
    base_sha: str,
    head_sha: str,
    changed_paths: Sequence[str],
) -> str:
    """Build one deterministic, bounded assignment shared by both model passes.

    :param skill: Authoritative checklist name.
    :param target: Assigned PR or branch label.
    :param repo: GitHub repository in ``owner/name`` form.
    :param base_sha: Full base commit SHA.
    :param head_sha: Full reviewed commit SHA.
    :param changed_paths: Repository-relative paths in the reviewed diff.
    :returns: Complete worker prompt stored outside the host model response.
    :raises ValueError: If assignment identity, SHAs, or paths are invalid.
    """
    if skill not in SUPPORTED_SKILLS:
        raise ValueError(f"Unknown review skill: {skill}")
    if not target.strip() or not repo.strip() or not changed_paths:
        raise ValueError("Worker assignment identity and changed paths must be non-empty")
    if not _SHA_RE.fullmatch(base_sha) or not _SHA_RE.fullmatch(head_sha):
        raise ValueError("Worker assignment requires full lowercase commit SHAs")
    for changed_path in changed_paths:
        path = Path(changed_path)
        if path.is_absolute() or path.as_posix() != changed_path or ".." in path.parts:
            raise ValueError("Worker assignment paths must be canonical and repository-relative")
    skill_instruction = (
        f"Invoke the tinaudio-synth-setter-skills:{skill} skill via the Skill tool."
    )
    if skill in DEEP_SKILLS:
        skill_instruction = (
            f"Invoke the repo-local {skill} skill by reading agent/skills/{skill}/SKILL.md."
        )
    paths = "\n".join(f"- {path}" for path in changed_paths)
    return f"""Review assignment
Target: {target}
Repository: {repo}
Base SHA: {base_sha}
Head SHA: {head_sha}
Skill: {skill}

{skill_instruction}
Inspect only `git diff {base_sha}..{head_sha} -- <changed paths>` and explicit checklist paths.
Do not recursively discover files, inspect caches, dependencies, sibling worktrees, or modify state.
Every Bash call has a 60-second timeout.

Changed paths:
{paths}

Return exactly one JSON object and no surrounding prose:
{{"skill":"{skill}","target":"{target}","findings":[{{"severity":"block or warn","path":"repository-relative changed path","line":42,"description":"self-contained concern"}}],"what_looks_good":["positive evidence"]}}
Use an empty findings array when appropriate. Keep what_looks_good non-empty and string values under 1500 words total.
"""


def report_is_parseable(report: str, *, expected_skill: str, expected_target: str) -> bool:
    """Return whether a worker result satisfies the merge contract.

    :param report: Worker JSON output.
    :param expected_skill: Checklist the worker was assigned.
    :param expected_target: PR or branch label the worker was assigned.
    :returns: Whether identity and structured fields satisfy the worker-result schema.
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


def _codex_candidates_for_skill(
    skill: str,
    available_models: AbstractSet[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return available and unavailable Codex candidates for one checklist.

    :param skill: Authoritative checklist name.
    :param available_models: Canonical selectors returned by Pi's model registry.
    :returns: Available and unavailable Codex selectors in policy order.
    :raises ValueError: If no configured Codex candidate is available.
    """
    configured = _DEEP_CODEX_CANDIDATES if skill in DEEP_SKILLS else _STANDARD_CODEX_CANDIDATES
    candidates, unavailable = _available_and_unavailable(configured, available_models)
    if not candidates:
        raise ValueError(f"No available models remain for {skill}/codex")
    return candidates, unavailable


def _review_passes_for_skill(
    skill: str,
    *,
    changed_lines: int,
    risk_reasons: Sequence[str],
    available_models: AbstractSet[str],
    free_pool_candidates: tuple[str, ...],
    free_pool_unavailable: tuple[str, ...],
) -> tuple[ReviewPass, ReviewPass]:
    """Build the paired Codex and free-pool passes for one checklist.

    :param skill: Authoritative checklist name.
    :param changed_lines: Total added and deleted lines in the diff.
    :param risk_reasons: Named risk signals detected in the diff.
    :param available_models: Canonical selectors returned by Pi's model registry.
    :param free_pool_candidates: Registered free-pool selectors in policy order.
    :param free_pool_unavailable: Unregistered free-pool selectors in policy order.
    :returns: Paired Codex and free-pool passes.
    """
    thinking, reason = _thinking_for(
        skill,
        changed_lines=changed_lines,
        risk_reasons=risk_reasons,
    )
    codex_candidates, codex_unavailable = _codex_candidates_for_skill(
        skill,
        available_models,
    )
    # Bind pass names to locals; a string literal on ``pass_name=`` trips ruff S106.
    codex_label = "codex"
    free_pool_label = "free-pool"
    return (
        ReviewPass(
            skill=skill,
            pass_name=codex_label,
            candidates=codex_candidates,
            unavailable=codex_unavailable,
            fallback_candidates=(),
            thinking=thinking,
            reason=reason,
            max_turns=PI_REVIEW_MAX_TURNS,
        ),
        ReviewPass(
            skill=skill,
            pass_name=free_pool_label,
            candidates=free_pool_candidates,
            unavailable=free_pool_unavailable,
            fallback_candidates=tuple(reversed(codex_candidates)),
            thinking=thinking,
            reason=reason,
            max_turns=PI_REVIEW_MAX_TURNS,
        ),
    )


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
    _require_free_pool(available_models)
    # The free pool is a single fixed tuple, so its availability split is invariant
    # across skills; only the Codex candidates vary with each skill's depth.
    free_pool_candidates, free_pool_unavailable = _available_and_unavailable(
        _FREE_POOL_CANDIDATES,
        available_models,
    )
    return [
        review_pass
        for skill in skills
        for review_pass in _review_passes_for_skill(
            skill,
            changed_lines=changed_lines,
            risk_reasons=risk_reasons,
            available_models=available_models,
            free_pool_candidates=free_pool_candidates,
            free_pool_unavailable=free_pool_unavailable,
        )
    ]


def _require_codex(available_models: AbstractSet[str]) -> None:
    """Require a registered Codex model for the always-available fallback.

    :param available_models: Canonical selectors returned by Pi's model registry.
    :raises ValueError: If Codex has no available model.
    """
    if not any(model.startswith("openai-codex/") for model in available_models):
        raise ValueError(f"No openai-codex models available; {_CODEX_SETUP}; credentials required")


def _require_free_pool(available_models: AbstractSet[str]) -> None:
    """Require at least one registered free-pool model before planning the second pass.

    :param available_models: Canonical selectors returned by Pi's model registry.
    :raises ValueError: If no free-pool candidate is registered.
    """
    if not any(model in available_models for model in _FREE_POOL_CANDIDATES):
        raise ValueError(
            f"No free-pool models available; {_FREE_POOL_SETUP}; credentials required"
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
        "extract-report", help="write the unique worker JSON object from Tintin JSONL"
    )
    extract.add_argument("transcript", type=Path)
    extract.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser(
        "validate-report", help="check a worker result's JSON contract"
    )
    validate.add_argument("path", type=Path)
    validate.add_argument("--skill", required=True, choices=sorted(SUPPORTED_SKILLS))
    validate.add_argument("--target", required=True)
    repair = subparsers.add_parser(
        "repair-prompt", help="build one same-session format-correction prompt"
    )
    repair.add_argument("path", type=Path)
    repair.add_argument("--skill", required=True, choices=sorted(SUPPORTED_SKILLS))
    repair.add_argument("--target", required=True)
    worker_prompt = subparsers.add_parser(
        "worker-prompt", help="write a deterministic review-worker assignment"
    )
    worker_prompt.add_argument("--skill", required=True, choices=sorted(SUPPORTED_SKILLS))
    worker_prompt.add_argument("--target", required=True)
    worker_prompt.add_argument("--repo", required=True)
    worker_prompt.add_argument("--base-sha", required=True)
    worker_prompt.add_argument("--head-sha", required=True)
    worker_prompt.add_argument("--changed-path", action="append", required=True)
    worker_prompt.add_argument("--output", type=Path, required=True)
    stats = subparsers.add_parser(
        "transcript-stats", help="print Tintin runtime-budget statistics as JSON"
    )
    stats.add_argument("transcript", type=Path)
    provenance = subparsers.add_parser(
        "provenance", help="print provenance for an effective model"
    )
    provenance.add_argument("model")
    fingerprint = subparsers.add_parser(
        "finding-fingerprint", help="print a stable finding identity"
    )
    fingerprint.add_argument("--skill", required=True, choices=sorted(SUPPORTED_SKILLS))
    fingerprint.add_argument("--severity", required=True, choices=("block", "warn"))
    fingerprint.add_argument("--path", required=True)
    fingerprint.add_argument("--line", required=True, type=int)
    fingerprint.add_argument("--description", required=True)
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
    if args.command == "repair-prompt":
        sys.stdout.write(
            f"{report_repair_prompt(args.path.read_text(), expected_skill=args.skill, expected_target=args.target)}\n"
        )
        return 0
    if args.command == "worker-prompt":
        args.output.write_text(
            build_worker_prompt(
                skill=args.skill,
                target=args.target,
                repo=args.repo,
                base_sha=args.base_sha,
                head_sha=args.head_sha,
                changed_paths=args.changed_path,
            )
        )
        return 0
    if args.command == "transcript-stats":
        sys.stdout.write(f"{json.dumps(asdict(transcript_stats(args.transcript)), indent=2)}\n")
        return 0
    if args.command == "provenance":
        sys.stdout.write(f"{provenance_for_model(args.model)}\n")
        return 0
    if args.command == "finding-fingerprint":
        sys.stdout.write(
            f"{finding_fingerprint(skill=args.skill, severity=args.severity, path=args.path, line=args.line, description=args.description)}\n"
        )
        return 0
    if args.command == "stream-host":
        sys.stdout.write(f"{stream_host_events(sys.stdin, args.transcript, sys.stderr)}\n")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
