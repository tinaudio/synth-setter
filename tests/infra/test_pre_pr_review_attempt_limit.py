"""Behavior tests for the pre-PR sentinel review attempt limit."""

from __future__ import annotations

import io
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import sh

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class _ReviewResult:
    """Captured launcher result.

    .. attribute :: returncode
        :type: int

        Process exit status.

    .. attribute :: stderr
        :type: str

        Captured standard error.

    .. attribute :: stdout
        :type: str

        Captured standard output.
    """

    returncode: int
    stderr: str
    stdout: str


def _review_checkout(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    """Create a minimal checkout with a deterministic Pi replacement.

    :param tmp_path: Temporary test directory.
    :returns: Checkout path, launcher environment, and Pi invocation log.
    """
    checkout = tmp_path / "checkout"
    shutil.copytree(_REPO_ROOT / "agent", checkout / "agent")
    sh.Command("git")("init", "-q", "-b", "test-branch", checkout)

    gh = tmp_path / "gh"
    gh.write_text(
        "#!/bin/bash\n"
        'if [[ "${GH_LOOKUP_FAIL:-}" == "1" ]]; then exit 7; fi\n'
        'printf "%s\\n" "${GH_OPEN_PR_NUMBER:-}"\n'
    )
    gh.chmod(0o755)

    invocation_log = tmp_path / "pi-invocations"
    pi = tmp_path / "pi"
    pi.write_text(
        "#!/bin/bash\n"
        'printf "invoked\\n" >>"${PI_INVOCATION_LOG}"\n'
        "printf '%s\\n' "
        '\'{"type":"message_start","message":{"role":"assistant",'
        '"content":[],"provider":"openai-codex","model":"gpt-5.6-terra"}}\'\n'
        "printf '%s\\n' "
        '\'{"type":"message_end","message":{"role":"assistant",'
        '"content":[{"type":"text","text":"review-complete"}]}}\'\n'
    )
    pi.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "PI_INVOCATION_LOG": str(invocation_log),
        "SYNTH_SETTER_PI_REVIEW": "",
    }
    return checkout, env, invocation_log


def _run_review(checkout: Path, env: dict[str, str], request: tuple[str, ...]) -> _ReviewResult:
    """Run the real shared launcher for one review request.

    :param checkout: Minimal review checkout.
    :param env: Launcher environment containing the deterministic Pi executable.
    :param request: Review skill and optional target arguments.
    :returns: Captured launcher process result.
    :raises sh.ErrorReturnCode: If the launcher exits with an unexpected status.
    """
    stderr = io.BytesIO()
    command = sh.Command(str(checkout / "agent/_shared/run_pi_review.sh"))
    try:
        stdout = str(command(*request, _cwd=checkout, _env=env, _err=stderr))
        returncode = 0
    except sh.ErrorReturnCode as exc:
        if exc.exit_code != 2:
            raise
        stdout = ""
        returncode = exc.exit_code
    return _ReviewResult(
        returncode=returncode,
        stderr=stderr.getvalue().decode(),
        stdout=stdout,
    )


def test_pre_pr_sentinel_review_fourth_attempt_refused_before_pi(tmp_path: Path) -> None:
    """Allow three sentinel reviews, then direct the agent to public review.

    :param tmp_path: Temporary checkout and fake external process directory.
    """
    checkout, env, invocation_log = _review_checkout(tmp_path)

    request = ("repo-review-full-no-comments",)
    first = _run_review(checkout, env, request)
    second = _run_review(checkout, env, request)
    third = _run_review(checkout, env, request)
    refused = _run_review(checkout, env, request)

    assert [first.returncode, second.returncode, third.returncode] == [0, 0, 0]
    assert refused.returncode == 2
    assert "Pre-PR sentinel review limit reached after 3 attempts." in refused.stderr
    assert "Open the PR and continue with /repo-review-full" in refused.stderr
    assert "public GitHub review bot" in refused.stderr
    assert invocation_log.read_text().splitlines() == ["invoked", "invoked", "invoked"]


def test_explicit_pr_dry_run_does_not_consume_pre_pr_attempt(tmp_path: Path) -> None:
    """Exclude explicit PR-mode no-comments reviews from the local budget.

    :param tmp_path: Temporary checkout and fake external process directory.
    """
    checkout, env, invocation_log = _review_checkout(tmp_path)

    targeted = _run_review(
        checkout,
        env,
        ("repo-review-full-no-comments", "--target", "3039"),
    )
    local = _run_review(checkout, env, ("repo-review-full-no-comments",))

    assert targeted.returncode == 0
    assert "Pre-PR sentinel review attempt" not in targeted.stderr
    assert "Pre-PR sentinel review attempt 1/3." in local.stderr
    assert invocation_log.read_text().splitlines() == ["invoked", "invoked"]


def test_implicit_open_pr_dry_run_does_not_consume_pre_pr_attempt(tmp_path: Path) -> None:
    """Exclude a resolved open PR from the local pre-PR budget.

    :param tmp_path: Temporary checkout and fake external process directory.
    """
    checkout, env, invocation_log = _review_checkout(tmp_path)
    open_pr_env = {**env, "GH_OPEN_PR_NUMBER": "3039"}
    request = ("repo-review-full-no-comments",)

    first = _run_review(checkout, open_pr_env, request)
    second = _run_review(checkout, open_pr_env, request)
    third = _run_review(checkout, open_pr_env, request)
    fourth = _run_review(checkout, open_pr_env, request)
    local = _run_review(checkout, env, request)

    assert [first.returncode, second.returncode, third.returncode, fourth.returncode] == [
        0,
        0,
        0,
        0,
    ]
    assert "Pre-PR sentinel review attempt" not in fourth.stderr
    assert "Pre-PR sentinel review attempt 1/3." in local.stderr
    assert invocation_log.read_text().splitlines() == [
        "invoked",
        "invoked",
        "invoked",
        "invoked",
        "invoked",
    ]


def test_open_pr_lookup_failure_refused_without_consuming_attempt(tmp_path: Path) -> None:
    """Fail closed on lookup errors without spending the local budget.

    :param tmp_path: Temporary checkout and fake external process directory.
    """
    checkout, env, invocation_log = _review_checkout(tmp_path)
    failing_env = {**env, "GH_LOOKUP_FAIL": "1"}
    request = ("repo-review-full-no-comments",)

    failed = _run_review(checkout, failing_env, request)
    local = _run_review(checkout, env, request)

    assert failed.returncode == 2
    assert "Unable to resolve whether the current branch has an open PR." in failed.stderr
    assert "Pre-PR sentinel review attempt 1/3." in local.stderr
    assert invocation_log.read_text().splitlines() == ["invoked"]


def test_public_pr_review_available_after_sentinel_limit(tmp_path: Path) -> None:
    """Keep the public GitHub review path available after sentinel exhaustion.

    :param tmp_path: Temporary checkout and fake external process directory.
    """
    checkout, env, invocation_log = _review_checkout(tmp_path)
    request = ("repo-review-full-no-comments",)
    first = _run_review(checkout, env, request)
    second = _run_review(checkout, env, request)
    third = _run_review(checkout, env, request)
    assert [first.returncode, second.returncode, third.returncode] == [0, 0, 0]

    public_review = _run_review(checkout, env, ("repo-review-full",))

    assert public_review.returncode == 0
    assert public_review.stdout.strip() == "review-complete"
    assert invocation_log.read_text().splitlines() == [
        "invoked",
        "invoked",
        "invoked",
        "invoked",
    ]
