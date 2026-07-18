"""Persist and deliver terminal Pi review failure audits."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel, Field, model_validator

from agent._shared.review_sentinel import make_review_filename
from agent.skills._shared.post_review import submit_review

FailureMode = Literal["full", "no-comments"]
Submitter = Callable[[str, int, dict[str, object], str], dict[str, object]]


class ProviderIncident(BaseModel, strict=True, extra="forbid"):
    """One authentication or capacity incident from a model attempt.

    .. attribute :: model
        :type: str

        Exact Pi model selector.

    .. attribute :: category
        :type: Literal["authentication", "quota/capacity"]

        Auditable provider failure class.

    .. attribute :: diagnostic
        :type: str

        Exact provider diagnostic.
    """

    model: str = Field(min_length=1)
    category: Literal["authentication", "quota/capacity"]
    diagnostic: str = Field(min_length=1)


class FailureRequest(BaseModel, strict=True, extra="forbid"):
    """Validated terminal review state consumed by both delivery modes.

    .. attribute :: target
        :type: str

        PR or branch label.

    .. attribute :: head
        :type: str

        Full origin HEAD reviewed.

    .. attribute :: stage
        :type: str

        Pipeline stage that failed.

    .. attribute :: diagnostic
        :type: str

        Exact terminal failure diagnostic.

    .. attribute :: repo
        :type: str | None

        GitHub owner/repository for full-review delivery.

    .. attribute :: pr_number
        :type: int | None

        Pull request number for full-review delivery.

    .. attribute :: transcript_paths
        :type: tuple[str, ...]

        Preserved host or worker transcript paths.

    .. attribute :: provider_incidents
        :type: tuple[ProviderIncident, ...]

        Authentication and quota/capacity attempt failures.

    .. attribute :: audit_markdown
        :type: str

        Accumulated attempt audit, possibly empty.

    .. attribute :: partial_findings
        :type: tuple[str, ...]

        Validated findings recovered before failure.
    """

    target: str = Field(min_length=1)
    head: str = Field(pattern=r"^[0-9a-f]{40}$")
    stage: str = Field(min_length=1)
    diagnostic: str = Field(min_length=1)
    repo: str | None = None
    pr_number: int | None = Field(default=None, gt=0)
    transcript_paths: tuple[str, ...] = ()
    provider_incidents: tuple[ProviderIncident, ...] = ()
    audit_markdown: str = ""
    partial_findings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _pair_github_target(self) -> FailureRequest:
        """Require repository and PR number together when either is set.

        :returns: Validated failure request.
        :raises ValueError: If only one GitHub target field is present.
        """
        if (self.repo is None) != (self.pr_number is None):
            raise ValueError("repo and pr_number must be provided together")
        return self


@dataclass(frozen=True, slots=True)
class FailureDelivery:
    """Persisted failure report and optional GitHub review URL.

    .. attribute :: report_path
        :type: Path

        Durable local report path.

    .. attribute :: report
        :type: str

        Exact rendered Markdown.

    .. attribute :: posted_url
        :type: str | None

        GitHub review URL for full mode.
    """

    report_path: Path
    report: str
    posted_url: str | None


class FailurePostError(RuntimeError):
    """GitHub failure-delivery error that retains the local report path.

    .. attribute :: report_path
        :type: Path

        Persisted report containing the original review failure.
    """

    report_path: Path

    def __init__(self, report_path: Path, diagnostic: str) -> None:
        """Initialize the delivery error.

        :param report_path: Persisted failure report.
        :param diagnostic: GitHub delivery diagnostic.
        """
        super().__init__(diagnostic)
        self.report_path = report_path


def render_failure_report(request: FailureRequest, *, mode: FailureMode = "full") -> str:
    """Render the common ordered terminal failure report.

    :param request: Validated terminal review state.
    :param mode: Calling review mode.
    :returns: Failure Markdown with provider incidents first.
    """
    skill = "repo-review-full" if mode == "full" else "repo-review-full-no-comments"
    lines = [f"# {skill} failure — {request.target}", "", "## Provider incidents", ""]
    if request.provider_incidents:
        lines.extend(
            f"- `{incident.model}` — {incident.category}: {incident.diagnostic}"
            for incident in request.provider_incidents
        )
    else:
        lines.append("None.")
    lines.extend(
        (
            "",
            "## Failure summary",
            "",
            "FAIL — review did not complete.",
            "",
            f"- **[{skill}:block]** Terminal review failure: {request.diagnostic}",
            f"- Failed stage: {request.stage}",
            f"- Target: {request.target}",
            f"- Origin HEAD: `{request.head}`",
        )
    )
    lines.append("- Transcripts: " + (", ".join(request.transcript_paths) or "None recorded."))
    lines.extend(("", "## Partial review audit", ""))
    lines.append(request.audit_markdown.strip() or "No completed attempt audit was recovered.")
    lines.extend(("", "### Validated partial findings", ""))
    if request.partial_findings:
        lines.extend(f"- {finding}" for finding in request.partial_findings)
    else:
        lines.append("None.")
    return "\n".join(lines).rstrip() + "\n"


def _report_path(request: FailureRequest, mode: FailureMode, review_dir: Path) -> Path:
    """Return the mode-specific durable report path.

    :param request: Validated terminal review state.
    :param mode: Calling review mode.
    :param review_dir: Local audit directory.
    :returns: Canonical sentinel or full-review failure path.
    """
    if mode == "no-comments":
        return review_dir / make_review_filename(request.head)
    return review_dir / f"repo-review-full.failure.{request.head}.md"


def _write_report(path: Path, report: str) -> None:
    """Write and flush a private failure report before external delivery.

    :param path: Destination report path.
    :param report: Rendered failure Markdown.
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as report_file:
            os.fchmod(report_file.fileno(), 0o600)
            report_file.write(report)
            report_file.flush()
            os.fsync(report_file.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def deliver_failure(
    request: FailureRequest,
    *,
    mode: FailureMode,
    review_dir: Path = Path(".agent-reviews"),
    submitter: Submitter = submit_review,
) -> FailureDelivery:
    """Persist a failure audit, then perform the mode-specific delivery.

    :param request: Validated terminal review state.
    :param mode: Full GitHub review or local no-comments delivery.
    :param review_dir: Local audit directory.
    :param submitter: GitHub review submission boundary.
    :returns: Persisted report and optional posted review URL.
    :raises ValueError: If full mode lacks a GitHub target.
    :raises FailurePostError: If GitHub delivery fails after persistence.
    """
    report = render_failure_report(request, mode=mode)
    report_path = _report_path(request, mode, review_dir)
    _write_report(report_path, report)
    if mode == "no-comments":
        return FailureDelivery(report_path=report_path, report=report, posted_url=None)
    if request.repo is None or request.pr_number is None:
        raise ValueError("full failure delivery requires repo and pr_number")
    payload: dict[str, object] = {
        "body": report,
        "event": "REQUEST_CHANGES",
        "comments": [],
    }
    fallback_banner = (
        "Blocking terminal review failure. GitHub rejected REQUEST_CHANGES on "
        "the reviewer's own PR, so this review falls back to COMMENT; the "
        "[repo-review-full:block] marker remains auditable.\n\n"
    )
    try:
        response = submitter(request.repo, request.pr_number, payload, fallback_banner)
    except SystemExit as error:
        raise FailurePostError(
            report_path, f"GitHub review delivery exited {error.code}"
        ) from error
    except Exception as error:
        raise FailurePostError(report_path, str(error)) from error
    posted_url = response.get("html_url")
    return FailureDelivery(
        report_path=report_path,
        report=report,
        posted_url=posted_url if isinstance(posted_url, str) else None,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the failure-delivery command parser.

    :returns: Parser for the deliver command.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    deliver = subparsers.add_parser("deliver")
    deliver.add_argument("--mode", choices=("full", "no-comments"), required=True)
    deliver.add_argument("--input", type=Path, required=True)
    deliver.add_argument("--review-dir", type=Path, default=Path(".agent-reviews"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Deliver a terminal failure report and retain failure exit status.

    :param argv: Optional command arguments for tests or embedding.
    :returns: Always 1 after a terminal review failure.
    """
    args = _build_parser().parse_args(argv)
    request = FailureRequest.model_validate_json(args.input.read_text())
    try:
        delivery = deliver_failure(request, mode=args.mode, review_dir=args.review_dir)
    except FailurePostError as error:
        sys.stderr.write(f"Original review failure: {request.diagnostic}\n")
        sys.stderr.write(f"Failure report: {error.report_path}\n")
        sys.stderr.write(f"Failure delivery error: {error}\n")
        return 1
    if args.mode == "no-comments":
        sys.stdout.write(delivery.report)
        sys.stdout.write(f"Sentinel: {delivery.report_path}\n")
    else:
        sys.stdout.write(
            f"Posted terminal failure audit: {delivery.posted_url or 'unknown URL'}\n"
        )
        sys.stdout.write(f"Failure report: {delivery.report_path}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
