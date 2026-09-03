"""Behavior contracts for the automatic Pi review limit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path("agent/_shared/should_run_pi_review.py")


def _review(author: str) -> dict[str, object]:
    return {"user": {"login": author}}


def _run_limit(
    project_root: Path, reviews: list[dict[str, object]]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed repository script
        [sys.executable, str(project_root / SCRIPT_PATH)],
        check=False,
        input=json.dumps(reviews),
        capture_output=True,
        text=True,
    )


@pytest.mark.infra
@pytest.mark.parametrize(
    ("automatic_review_count", "expected"),
    [(0, "true"), (1, "true"), (2, "false"), (3, "false")],
)
def test_automatic_review_runs_only_before_two_bot_reviews(
    project_root: Path,
    automatic_review_count: int,
    expected: str,
) -> None:
    """The third automatic review request and later requests are rejected.

    :param project_root: Repository root containing the authorization script.
    :param automatic_review_count: Existing reviews from ``github-actions[bot]``.
    :param expected: Expected workflow-compatible authorization result.
    """
    reviews = [_review("github-actions[bot]")] * automatic_review_count

    result = _run_limit(project_root, reviews)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.infra
def test_review_limit_ignores_reviews_from_other_accounts(project_root: Path) -> None:
    """Human and other-bot reviews do not consume automatic Pi review slots.

    :param project_root: Repository root containing the authorization script.
    """
    reviews = [
        _review("ktinubu"),
        _review("copilot-pull-request-reviewer[bot]"),
        _review("github-actions[bot]"),
    ]

    result = _run_limit(project_root, reviews)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "true"


@pytest.mark.infra
def test_review_limit_fails_closed_for_malformed_api_payload(project_root: Path) -> None:
    """Unexpected GitHub API data cannot silently authorize another review.

    :param project_root: Repository root containing the authorization script.
    """
    result = subprocess.run(  # noqa: S603 - fixed repository script
        [sys.executable, str(project_root / SCRIPT_PATH)],
        check=False,
        input='{"message": "API failure"}',
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert result.stdout == ""
