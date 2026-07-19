#!/usr/bin/env python3
"""Render Pi no-comments payloads without host-model Markdown reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

if __package__:
    from agent._shared.review_sentinel import make_review_path
else:
    from review_sentinel import make_review_path


class ReviewFinding(BaseModel, strict=True, extra="forbid"):
    """One already-aggregated inline finding.

    .. attribute :: path
        :type: str

        Repository-relative file path.

    .. attribute :: line
        :type: int

        Positive right-side line anchor.

    .. attribute :: body
        :type: str

        Markdown body including skill and severity tag.
    """

    path: str = Field(min_length=1)
    line: int = Field(gt=0)
    body: str = Field(min_length=1)


class ReviewPayload(BaseModel, strict=True, extra="forbid"):
    """Aggregated review data consumed by foreground delivery.

    .. attribute :: pr_number
        :type: int | None

        Pull request number, or ``None`` for local mode.

    .. attribute :: repo
        :type: str

        GitHub repository identity.

    .. attribute :: review_body
        :type: str

        Lead-in, provider incidents, health, and audit Markdown.

    .. attribute :: findings
        :type: tuple[ReviewFinding, ...]

        Aggregated findings in review order.
    """

    pr_number: int | None = Field(default=None, gt=0)
    repo: str = Field(min_length=1)
    review_body: str = Field(min_length=1)
    findings: tuple[ReviewFinding, ...]


@dataclass(frozen=True, slots=True)
class RenderContext:
    """Foreground state included in the deterministic report summary.

    .. attribute :: target
        :type: str

        Human-readable PR or branch label.

    .. attribute :: head_sha
        :type: str

        Full reviewed commit SHA.

    .. attribute :: head_ref
        :type: str

        Reviewed branch name.

    .. attribute :: upstream_sha
        :type: str

        Upstream SHA or ``none``.

    .. attribute :: worktree_state
        :type: str

        ``clean`` or ``dirty``.

    .. attribute :: unchanged_count
        :type: int

        Number of repeated non-progress reviews.

    .. attribute :: skill_count
        :type: int

        Number of selected checklists.

    .. attribute :: next_step
        :type: str

        Caller-specific remediation or posting guidance.
    """

    target: str
    head_sha: str
    head_ref: str
    upstream_sha: str
    worktree_state: str
    unchanged_count: int
    skill_count: int
    next_step: str


def _validated_payload(payload: ReviewPayload | Mapping[str, object]) -> ReviewPayload:
    """Return strict payload data from a model or mapping.

    :param payload: Validated payload or raw mapping.
    :returns: Strict review payload.
    """
    if isinstance(payload, ReviewPayload):
        return payload
    return ReviewPayload.model_validate_json(json.dumps(payload))


def _finding_lines(findings: tuple[ReviewFinding, ...]) -> list[str]:
    """Group findings into canonical path sections.

    :param findings: Aggregated findings in review order.
    :returns: Markdown lines for all inline findings.
    """
    grouped: dict[str, list[ReviewFinding]] = {}
    for finding in findings:
        grouped.setdefault(finding.path, []).append(finding)
    lines: list[str] = []
    for path, path_findings in grouped.items():
        lines.append(f"### `{path}`")
        lines.extend(f"- **L{finding.line}** — {finding.body}" for finding in path_findings)
        lines.append("")
    return lines


def _summary_lines(review: ReviewPayload, context: RenderContext) -> list[str]:
    """Render deterministic severity and progress summary lines.

    :param review: Validated aggregate payload.
    :param context: Reviewed Git and progress state.
    :returns: Markdown lines for the summary section.
    """
    blocks = sum(":block]" in finding.body for finding in review.findings)
    warns = sum(":warn]" in finding.body for finding in review.findings)
    lines = [
        "## Summary",
        "",
        f"- {blocks} BLOCK, {warns} WARN across {context.skill_count} skills",
        f"- Reviewed at: {context.head_sha}",
        "- Progress: "
        f"branch {context.head_ref}; HEAD {context.head_sha}; "
        f"upstream {context.upstream_sha}; worktree {context.worktree_state}; "
        f"unchanged review count {context.unchanged_count}.",
        f"- {context.next_step}",
    ]
    if context.unchanged_count > 0:
        lines.append(
            "- Possible review loop: make coherent remediation durable or report the blocker "
            "before retrying."
        )
    return lines


def render_markdown(
    payload: ReviewPayload | Mapping[str, object], *, context: RenderContext
) -> str:
    """Render the complete no-comments report from structured inputs.

    :param payload: Aggregated review body and findings.
    :param context: Reviewed state and summary metadata.
    :returns: Canonical Markdown sentinel content.
    """
    review = _validated_payload(payload)
    lines = [
        f"# repo-review-full-no-comments — {context.target}",
        "",
        review.review_body,
        "",
        "## Inline findings (would be posted by `/repo-review-full`)",
        "",
        *_finding_lines(review.findings),
        *_summary_lines(review, context),
    ]
    return "\n".join(lines) + "\n"


def _write_report(report: str, output_path: Path) -> None:
    """Atomically persist one rendered review.

    :param report: Complete Markdown deliverable.
    :param output_path: Canonical sentinel destination.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=output_path.parent, prefix=".pi-review-")
    try:
        with os.fdopen(descriptor, "w") as output:
            output.write(report)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, output_path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def render_zero_diff(context: RenderContext) -> str:
    """Render the fixed PASS report for a target with no changed files.

    :param context: Reviewed Git state.
    :returns: Zero-diff Markdown sentinel content.
    """
    return (
        f"# repo-review-full-no-comments — {context.target}\n\n"
        "PASS — no findings across all skills (code-health, correctness, comment-hygiene, "
        "python-style, shell-style, synth-setter, tdd-impl, ml-test).\n\n"
        "## Summary\n\n"
        "- 0 BLOCK, 0 WARN\n"
        f"- Reviewed at: {context.head_sha}\n"
        "- Progress: "
        f"branch {context.head_ref}; HEAD {context.head_sha}; "
        f"upstream {context.upstream_sha}; worktree {context.worktree_state}; "
        "unchanged review count 0.\n"
    )


def render_payload(
    payload_path: Path,
    *,
    output_path: Path,
    context: RenderContext,
    remove_payload: bool,
) -> str:
    """Validate, atomically write, and optionally remove one findings payload.

    :param payload_path: Exact invocation-isolated JSON payload path.
    :param output_path: Canonical sentinel destination.
    :param context: Reviewed state and summary metadata.
    :param remove_payload: Whether to unlink only ``payload_path`` after success.
    :returns: Markdown written to ``output_path``.
    """
    payload = ReviewPayload.model_validate_json(payload_path.read_text())
    report = render_markdown(payload, context=context)
    _write_report(report, output_path)
    if remove_payload:
        payload_path.unlink()
    return report


def _git(*args: str) -> str:
    """Return stripped output from one read-only Git command.

    :param *args: Git arguments.
    :returns: Command standard output.
    :raises RuntimeError: If Git exits nonzero.
    """
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable not found on PATH")
    result = subprocess.run(  # noqa: S603
        [git, *args],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _progress_state(head_sha: str, upstream_sha: str, status: str) -> str:
    """Build the compact progress identity for one worktree state.

    :param head_sha: Current commit SHA.
    :param upstream_sha: Current upstream SHA or ``none``.
    :param status: Git porcelain status.
    :returns: Stable progress-state string.
    """
    status_digest = hashlib.sha256(status.encode()).hexdigest()
    return f"{head_sha}|{upstream_sha}|{status_digest}"


def _read_progress(path: Path) -> tuple[int, str]:
    """Read prior progress, treating malformed advisory state as absent.

    :param path: Per-branch progress path.
    :returns: Prior unchanged count and state identity.
    """
    if not path.exists():
        return 0, ""
    fields = path.read_text().strip().split(" ", 1)
    if len(fields) != 2:
        return 0, ""
    try:
        return int(fields[0]), fields[1]
    except ValueError:
        return 0, ""


def _progress_count(payload: ReviewPayload, head_ref: str, current_state: str) -> int:
    """Persist and return the repeated non-progress review count.

    :param payload: Validated aggregate payload.
    :param head_ref: Current branch name.
    :param current_state: Current progress identity.
    :returns: Updated unchanged review count.
    """
    progress_key = hashlib.sha256(f"{head_ref}\n".encode()).hexdigest()
    progress_path = Path(
        f".agent-reviews/repo-review-full-no-comments-progress.{progress_key}.txt"
    )
    previous_count, previous_state = _read_progress(progress_path)
    is_non_pass = bool(payload.findings) or "[pr-health]" in payload.review_body
    unchanged_count = previous_count + 1 if is_non_pass and current_state == previous_state else 0
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(f"{unchanged_count} {current_state}\n")
    return unchanged_count


def resolve_context(
    payload: ReviewPayload,
    *,
    reviewed_head: str,
    target: str,
    skill_count: int,
    next_step: str,
) -> RenderContext:
    """Derive review progress and reject delivery-time HEAD drift.

    :param payload: Validated aggregate payload.
    :param reviewed_head: Exact commit reviewed by every foreground worker.
    :param target: Human-readable PR or branch label.
    :param skill_count: Number of selected checklists.
    :param next_step: Caller-specific follow-up guidance.
    :returns: Render context persisted in the report summary.
    :raises ValueError: If the worktree HEAD changed after worker assignment.
    """
    head_sha = _git("rev-parse", "HEAD")
    if head_sha != reviewed_head:
        raise ValueError("Reviewed HEAD changed before delivery")
    head_ref = _git("branch", "--show-current")
    try:
        upstream_sha = _git("rev-parse", "@{upstream}")
    except RuntimeError:
        upstream_sha = "none"
    status = _git("status", "--porcelain")
    current_state = _progress_state(head_sha, upstream_sha, status)
    unchanged_count = _progress_count(payload, head_ref, current_state)
    return RenderContext(
        target=target,
        head_sha=head_sha,
        head_ref=head_ref,
        upstream_sha=upstream_sha,
        worktree_state="dirty" if status else "clean",
        unchanged_count=unchanged_count,
        skill_count=skill_count,
        next_step=next_step,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the deterministic renderer CLI parser.

    :returns: Argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--payload", type=Path)
    source.add_argument("--zero-diff", action="store_true")
    parser.add_argument("--target", required=True)
    parser.add_argument("--reviewed-head", required=True)
    parser.add_argument("--skill-count", type=int, default=0)
    parser.add_argument("--next-step", default="No follow-up required.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--remove-payload", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Render one payload and print the report plus sentinel path.

    :param argv: Optional CLI arguments.
    :returns: Process exit status.
    """
    args = _build_parser().parse_args(argv)
    if args.zero_diff:
        payload = ReviewPayload.model_validate_json(
            '{"pr_number":null,"repo":"local","review_body":"PASS — no diff.","findings":[]}'
        )
    else:
        payload = ReviewPayload.model_validate_json(args.payload.read_text())
    context = resolve_context(
        payload,
        reviewed_head=args.reviewed_head,
        target=args.target,
        skill_count=args.skill_count,
        next_step=args.next_step,
    )
    output_path = args.output or Path(make_review_path(context.head_sha))
    if args.zero_diff:
        report = render_zero_diff(context)
        _write_report(report, output_path)
    else:
        report = render_payload(
            args.payload,
            output_path=output_path,
            context=context,
            remove_payload=args.remove_payload,
        )
    sys.stdout.write(f"{report}Sentinel: {output_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
