"""Behavior tests for terminal Pi review failure delivery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from errno import EBADF
from pathlib import Path

import pytest

import agent._shared.review_failure as review_failure
from agent._shared.review_failure import (
    FailurePostError,
    FailureRequest,
    ProviderIncident,
    deliver_failure,
    main,
    render_failure_report,
)

_HEAD = "a" * 40


def _request() -> FailureRequest:
    """Build one representative partial-review failure.

    :returns: Validated failure-delivery request.
    """
    return FailureRequest(
        target="PR #2155",
        head=_HEAD,
        stage="worker aggregation",
        diagnostic="required OpenRouter passes did not produce valid reports",
        repo="tinaudio/synth-setter",
        pr_number=2155,
        transcript_paths=(".agent-reviews/pi-host.jsonl",),
        provider_incidents=(
            ProviderIncident(
                model="openrouter/cohere/north-mini-code:free",
                category="quota/capacity",
                diagnostic="429: free-models-per-min",
            ),
        ),
        audit_markdown="| Skill | Status |\n| --- | --- |\n| code-health | verified |",
        partial_findings=("code-health completed with no findings",),
    )


def test_render_failure_report_puts_provider_incidents_first() -> None:
    """Keep provider failures above every other audit section."""
    report = render_failure_report(_request())

    provider = report.index("## Provider incidents")
    summary = report.index("## Failure summary")
    audit = report.index("## Partial review audit")
    assert provider < summary < audit
    assert "openrouter/cohere/north-mini-code:free" in report
    assert "429: free-models-per-min" in report
    assert "FAIL — review did not complete" in report


def test_deliver_failure_no_comments_writes_blocking_head_sentinel(tmp_path: Path) -> None:
    """Persist a failure sentinel that the pre-PR gate cannot treat as PASS.

    :param tmp_path: Temporary review directory.
    """
    result = deliver_failure(_request(), mode="no-comments", review_dir=tmp_path)

    assert result.posted_url is None
    assert result.report_path.name == f"repo-review-full-no-comments.{_HEAD}.md"
    report = result.report_path.read_text()
    assert "FAIL — review did not complete" in report
    assert "**[repo-review-full-no-comments:block]**" in report


def test_deliver_failure_no_comments_replaces_sentinel_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave the prior canonical sentinel intact if replacement is interrupted.

    :param tmp_path: Temporary review directory.
    :param monkeypatch: Replaces the atomic filesystem boundary.
    """
    sentinel = tmp_path / f"repo-review-full-no-comments.{_HEAD}.md"
    sentinel.write_text("prior complete sentinel")

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        del source, destination
        raise OSError("interrupted before replace")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="interrupted before replace"):
        deliver_failure(_request(), mode="no-comments", review_dir=tmp_path)

    assert sentinel.read_text() == "prior complete sentinel"
    assert list(tmp_path.iterdir()) == [sentinel]


def test_deliver_failure_closes_temporary_descriptor_when_chmod_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close the raw temporary descriptor when permission setup fails.

    :param tmp_path: Temporary review directory.
    :param monkeypatch: Replaces the descriptor permission boundary.
    """
    observed_descriptor: int | None = None

    def fail_fchmod(descriptor: int, mode: int) -> None:
        nonlocal observed_descriptor
        del mode
        observed_descriptor = descriptor
        raise OSError("chmod failed")

    monkeypatch.setattr(os, "fchmod", fail_fchmod)

    with pytest.raises(OSError, match="chmod failed"):
        deliver_failure(_request(), mode="no-comments", review_dir=tmp_path)

    assert observed_descriptor is not None
    with pytest.raises(OSError) as error:
        os.fstat(observed_descriptor)
    assert error.value.errno == EBADF


def test_deliver_failure_full_posts_blocking_review_after_persisting(
    tmp_path: Path,
) -> None:
    """Post a finding-free blocking review from the persisted failure report.

    :param tmp_path: Temporary review directory.
    """
    observed: dict[str, object] = {}

    def submitter(
        repo: str,
        pr_number: int,
        payload: dict[str, object],
        fallback_banner: str,
    ) -> dict[str, object]:
        observed.update(
            repo=repo,
            pr_number=pr_number,
            payload=payload,
            fallback_banner=fallback_banner,
        )
        assert list(tmp_path.glob("repo-review-full.failure.*.md"))
        return {"html_url": "https://github.com/tinaudio/synth-setter/pull/2155#review"}

    result = deliver_failure(
        _request(),
        mode="full",
        review_dir=tmp_path,
        submitter=submitter,
    )

    assert result.posted_url == "https://github.com/tinaudio/synth-setter/pull/2155#review"
    assert observed["repo"] == "tinaudio/synth-setter"
    assert observed["pr_number"] == 2155
    payload = observed["payload"]
    assert isinstance(payload, dict)
    assert payload["event"] == "REQUEST_CHANGES"
    assert payload["comments"] == []
    assert str(payload["body"]).startswith("# repo-review-full failure — PR #2155")
    assert "falls back to COMMENT" in str(observed["fallback_banner"])


def test_deliver_failure_post_error_preserves_original_report(tmp_path: Path) -> None:
    """Retain the local failure audit when GitHub delivery fails.

    :param tmp_path: Temporary review directory.
    """

    def failing_submitter(
        repo: str,
        pr_number: int,
        payload: dict[str, object],
        fallback_banner: str,
    ) -> dict[str, object]:
        del repo, pr_number, payload, fallback_banner
        raise RuntimeError("GitHub unavailable")

    with pytest.raises(FailurePostError, match="GitHub unavailable") as error:
        deliver_failure(
            _request(),
            mode="full",
            review_dir=tmp_path,
            submitter=failing_submitter,
        )

    assert error.value.report_path.read_text().startswith("# repo-review-full failure — PR #2155")


def test_failure_delivery_cli_returns_nonzero_after_writing_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return failure status after durable no-comments delivery.

    :param tmp_path: Temporary request and review paths.
    :param capsys: Pytest output capture fixture.
    """
    request_path = tmp_path / "failure.json"
    request_path.write_text(json.dumps(_request().model_dump(mode="json")))

    status = main(
        [
            "deliver",
            "--mode",
            "no-comments",
            "--input",
            str(request_path),
            "--review-dir",
            str(tmp_path / "reviews"),
        ]
    )

    assert status == 1
    captured = capsys.readouterr()
    assert "FAIL — review did not complete" in captured.out
    assert "Sentinel:" in captured.out


def test_failure_delivery_script_executes_from_repo_root(tmp_path: Path) -> None:
    """Exercise the documented direct script invocation.

    :param tmp_path: Temporary request and review paths.
    """
    request_path = tmp_path / "failure.json"
    request_path.write_text(json.dumps(_request().model_dump(mode="json")))
    script = Path(__file__).resolve().parents[2] / "agent/_shared/review_failure.py"

    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(script),
            "deliver",
            "--mode",
            "no-comments",
            "--input",
            str(request_path),
            "--review-dir",
            str(tmp_path / "reviews"),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=script.parents[2],
    )

    assert result.returncode == 1
    assert "FAIL — review did not complete" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_missing_provider_planner_failure_reaches_blocking_delivery(
    tmp_path: Path,
) -> None:
    """Drive a real planner preflight error through the failure-delivery CLI.

    :param tmp_path: Temporary provider registry, request, and review paths.
    """
    repo_root = Path(__file__).resolve().parents[2]
    routing_script = repo_root / "agent/_shared/pi_review_routing.py"
    failure_script = repo_root / "agent/_shared/review_failure.py"
    pi_executable = tmp_path / "pi"
    pi_executable.write_text(
        "#!/bin/bash\necho 'openai-codex  gpt-5.6-terra  372K  128K  yes  yes'\n"
    )
    pi_executable.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{tmp_path}:{environment['PATH']}"
    planner = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(routing_script),
            "plan",
            "--skill",
            "code-health",
            "--changed-lines",
            "10",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
        env=environment,
    )
    assert planner.returncode != 0
    assert "No free-pool models available" in planner.stderr

    request = _request().model_copy(
        update={
            "stage": "planner/provider preflight",
            "diagnostic": planner.stderr.strip(),
            "repo": None,
            "pr_number": None,
        }
    )
    request_path = tmp_path / "failure.json"
    request_path.write_text(json.dumps(request.model_dump(mode="json")))
    delivery = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(failure_script),
            "deliver",
            "--mode",
            "no-comments",
            "--input",
            str(request_path),
            "--review-dir",
            str(tmp_path / "reviews"),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert delivery.returncode == 1
    sentinel = tmp_path / "reviews" / f"repo-review-full-no-comments.{_HEAD}.md"
    assert "**[repo-review-full-no-comments:block]**" in sentinel.read_text()
    assert "No free-pool models available" in delivery.stdout


def test_failure_delivery_cli_reports_original_and_post_failures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the original diagnostic visible when GitHub delivery also fails.

    :param tmp_path: Temporary request and report paths.
    :param capsys: Pytest output capture fixture.
    :param monkeypatch: Replaces the GitHub delivery boundary.
    """
    request_path = tmp_path / "failure.json"
    request_path.write_text(json.dumps(_request().model_dump(mode="json")))
    report_path = tmp_path / "reviews/failure.md"
    report_path.parent.mkdir()
    report_path.write_text("persisted original review failure")

    def fail_delivery(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise FailurePostError(report_path, "GitHub unavailable")

    monkeypatch.setattr(review_failure, "deliver_failure", fail_delivery)

    status = review_failure.main(["deliver", "--mode", "full", "--input", str(request_path)])

    assert status == 1
    captured = capsys.readouterr()
    assert "required OpenRouter passes did not produce valid reports" in captured.err
    assert str(report_path) in captured.err
    assert "GitHub unavailable" in captured.err
