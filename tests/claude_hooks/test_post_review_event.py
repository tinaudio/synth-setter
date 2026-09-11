"""Unit tests for event derivation and self-review fallback in ``post_review.py``.

The helper backs ``/repo-review-full``: the calling skill sets a top-level
``event`` (REQUEST_CHANGES when any BLOCK finding exists, COMMENT for WARN-only,
APPROVE when clean). GitHub rejects REQUEST_CHANGES/APPROVE on the bot's own PR
with an HTTP 422, so ``submit_review`` must retry once as a plain COMMENT with a
loud banner prepended. These tests pin the payload shape and the fallback.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HELPER_PATH = _REPO_ROOT / "agent" / "skills" / "_shared" / "post_review.py"


def _load_helper() -> ModuleType:
    """Import ``post_review`` directly by path so the test avoids sys.path edits.

    :returns: The imported module.
    :raises RuntimeError: If the helper file can't be located by ``importlib``.
    """
    spec = importlib.util.spec_from_file_location("post_review", _HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not locate {_HELPER_PATH} via importlib")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module's dataclasses can resolve their own __module__.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def helper() -> ModuleType:
    """Load ``post_review`` once per module via path-based import.

    :returns: The imported module.
    """
    return _load_helper()


def test_build_review_payload_event_defaults_to_comment(helper: ModuleType) -> None:
    """The event defaults to COMMENT when the argument is omitted.

    :param helper: The loaded ``post_review`` module.
    """
    payload = helper.build_review_payload("body", [], [])
    assert payload["event"] == "COMMENT"


def test_build_review_payload_request_changes_passes_through(helper: ModuleType) -> None:
    """REQUEST_CHANGES is carried through to the payload event.

    :param helper: The loaded ``post_review`` module.
    """
    payload = helper.build_review_payload("body", [], [], event="REQUEST_CHANGES")
    assert payload["event"] == "REQUEST_CHANGES"


def test_build_review_payload_invalid_event_raises(helper: ModuleType) -> None:
    """An unrecognized event value raises ValueError.

    :param helper: The loaded ``post_review`` module.
    """
    with pytest.raises(ValueError, match="event"):
        helper.build_review_payload("body", [], [], event="LGTM")


def test_build_review_payload_approve_with_no_findings_omits_comments(
    helper: ModuleType,
) -> None:
    """APPROVE with no findings omits the comments key (GitHub rejects it otherwise).

    :param helper: The loaded ``post_review`` module.
    """
    payload = helper.build_review_payload("clean", [], [], event="APPROVE")
    assert payload["event"] == "APPROVE"
    assert "comments" not in payload


def test_build_review_payload_comment_keeps_empty_comments_key(helper: ModuleType) -> None:
    """COMMENT keeps an empty comments list rather than dropping the key.

    :param helper: The loaded ``post_review`` module.
    """
    payload = helper.build_review_payload("body", [], [], event="COMMENT")
    assert payload["comments"] == []


def test_build_review_payload_pi_audit_is_collapsed_by_default(helper: ModuleType) -> None:
    """The verbose Pi audit remains available without obscuring review findings.

    :param helper: The loaded ``post_review`` module.
    """
    payload = helper.build_review_payload(
        "Actionable summary.\n\n## Pi review audit\n\n| Skill | Status |\n| --- | --- |\n| ml-test | success |",
        [],
        [],
    )
    assert payload["body"] == (
        "Actionable summary.\n\n<details>\n<summary>Pi review audit</summary>\n\n"
        "| Skill | Status |\n| --- | --- |\n| ml-test | success |\n\n</details>"
    )


def _fake_run_factory(responses: list[SimpleNamespace]) -> tuple:
    """Build a fake ``subprocess.run`` that returns queued responses in order.

    :param responses: Queued ``CompletedProcess``-like results, consumed FIFO.
    :returns: The fake callable and a list recording each invocation's stdin.
    """
    calls: list[str] = []

    def fake_run(*_args: object, **kwargs: object) -> SimpleNamespace:
        calls.append(str(kwargs.get("input")))
        return responses.pop(0)

    return fake_run, calls


def test_submit_review_self_review_422_falls_back_to_comment(
    helper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A self-review 422 retries once as COMMENT with the banner prepended.

    :param helper: The loaded ``post_review`` module.
    :param monkeypatch: Pytest fixture for patching ``subprocess.run``.
    """
    err_422 = SimpleNamespace(
        returncode=1,
        stdout="",
        stderr="HTTP 422: Can not request changes on your own pull request",
    )
    ok = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"html_url": "https://example/r/1"}),
        stderr="",
    )
    fake_run, calls = _fake_run_factory([err_422, ok])
    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    monkeypatch.setattr(helper, "gh_executable", lambda: "/usr/bin/gh")

    payload = {"body": "Original review.", "event": "REQUEST_CHANGES", "comments": []}
    response = helper.submit_review("o/r", 7, payload, fallback_banner="⛔ BANNER")

    assert response["html_url"] == "https://example/r/1"
    assert len(calls) == 2
    retried = json.loads(calls[1])
    assert retried["event"] == "COMMENT"
    assert retried["body"].startswith("⛔ BANNER")
    assert "Original review." in retried["body"]


def test_submit_review_self_review_422_on_stdout_falls_back_to_comment(
    helper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A self-review 422 whose body lands on stdout still retries as COMMENT.

    :param helper: The loaded ``post_review`` module.
    :param monkeypatch: Pytest fixture for patching ``subprocess.run``.
    """
    err_422 = SimpleNamespace(
        returncode=1,
        stdout=json.dumps(
            {
                "message": "Unprocessable Entity",
                "errors": ["Review Can not request changes on your own pull request"],
            }
        ),
        stderr="gh: Unprocessable Entity (HTTP 422)",
    )
    ok = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"html_url": "https://example/r/4"}),
        stderr="",
    )
    fake_run, calls = _fake_run_factory([err_422, ok])
    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    monkeypatch.setattr(helper, "gh_executable", lambda: "/usr/bin/gh")

    payload = {"body": "Original review.", "event": "REQUEST_CHANGES", "comments": []}
    response = helper.submit_review("o/r", 7, payload, fallback_banner="⛔ BANNER")

    assert response["html_url"] == "https://example/r/4"
    assert len(calls) == 2
    retried = json.loads(calls[1])
    assert retried["event"] == "COMMENT"
    assert retried["body"].startswith("⛔ BANNER")
    assert "Original review." in retried["body"]


def test_submit_review_self_approve_422_falls_back_to_comment(
    helper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A self-APPROVE 422 also falls back to COMMENT (regex covers approve + request-changes).

    :param helper: The loaded ``post_review`` module.
    :param monkeypatch: Pytest fixture for patching ``subprocess.run``.
    """
    err_422 = SimpleNamespace(
        returncode=1,
        stdout="",
        stderr="HTTP 422: Can not approve your own pull request",
    )
    ok = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"html_url": "https://example/r/3"}),
        stderr="",
    )
    fake_run, calls = _fake_run_factory([err_422, ok])
    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    monkeypatch.setattr(helper, "gh_executable", lambda: "/usr/bin/gh")

    payload = {"body": "Clean review.", "event": "APPROVE"}
    response = helper.submit_review("o/r", 7, payload, fallback_banner="✅ No findings")

    assert response["html_url"] == "https://example/r/3"
    assert len(calls) == 2
    retried = json.loads(calls[1])
    assert retried["event"] == "COMMENT"
    assert retried["body"].startswith("✅ No findings")


def test_submit_review_actions_approve_422_falls_back_to_comment(
    helper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zero-finding APPROVE refused for the Actions token retries as COMMENT.

    GitHub Actions refusal lacks self-review wording, so it must also retry as COMMENT (#2922).

    :param helper: The loaded ``post_review`` module.
    :param monkeypatch: Pytest fixture for patching ``subprocess.run``.
    """
    err_422 = SimpleNamespace(
        returncode=1,
        stdout="",
        stderr="HTTP 422: GitHub Actions is not permitted to approve pull requests.",
    )
    ok = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"html_url": "https://example/r/9"}),
        stderr="",
    )
    fake_run, calls = _fake_run_factory([err_422, ok])
    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    monkeypatch.setattr(helper, "gh_executable", lambda: "/usr/bin/gh")

    payload = {"body": "Clean review.", "event": "APPROVE"}
    response = helper.submit_review("o/r", 7, payload, fallback_banner="✅ No findings")

    assert response["html_url"] == "https://example/r/9"
    assert len(calls) == 2
    retried = json.loads(calls[1])
    assert retried["event"] == "COMMENT"
    assert retried["body"].startswith("✅ No findings")


def test_submit_review_reworded_refusal_still_falls_back_to_comment(
    helper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An equivalent refusal wording still retries as COMMENT.

    GitHub's prose is not a stable contract, so "cannot approve" must be recognised
    alongside "can not approve" and "is not permitted to approve".

    :param helper: The loaded ``post_review`` module.
    :param monkeypatch: Pytest fixture for patching ``subprocess.run``.
    """
    err_422 = SimpleNamespace(
        returncode=1,
        stdout="",
        stderr="HTTP 422: GitHub Actions cannot approve pull requests.",
    )
    ok = SimpleNamespace(
        returncode=0, stdout=json.dumps({"html_url": "https://example/r/8"}), stderr=""
    )
    fake_run, calls = _fake_run_factory([err_422, ok])
    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    monkeypatch.setattr(helper, "gh_executable", lambda: "/usr/bin/gh")

    payload = {"body": "Clean review.", "event": "APPROVE"}
    response = helper.submit_review("o/r", 7, payload, fallback_banner="✅ No findings")

    assert response["html_url"] == "https://example/r/8"
    assert json.loads(calls[1])["event"] == "COMMENT"


def test_submit_review_unrelated_422_exits_without_downgrading(
    helper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 422 that is not an event refusal fails loudly instead of retrying.

    A bad inline-comment anchor also answers 422; downgrading it would drop the finding.

    :param helper: The loaded ``post_review`` module.
    :param monkeypatch: Pytest fixture for patching ``subprocess.run``.
    """
    err_422 = SimpleNamespace(
        returncode=1,
        stdout=json.dumps(
            {"message": "Validation Failed", "errors": ["line must be part of the diff"]}
        ),
        stderr="HTTP 422: Validation Failed",
    )
    fake_run, calls = _fake_run_factory([err_422])
    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    monkeypatch.setattr(helper, "gh_executable", lambda: "/usr/bin/gh")

    payload = {"body": "b", "event": "REQUEST_CHANGES", "comments": []}
    with pytest.raises(SystemExit):
        helper.submit_review("o/r", 7, payload, fallback_banner="x")
    assert len(calls) == 1


def _write_stub_gh(tmp_path: Path, calls_dir: Path) -> Path:
    """Write a stub ``gh`` that refuses the first POST and accepts the retry.

    Counts prior payload files rather than lines: BSD ``wc`` pads its output, which
    would corrupt the filename on macOS runners.

    :param tmp_path: Directory the stub executable is written into.
    :param calls_dir: Directory the stub records each request payload into.
    :returns: Path to the executable stub.
    """
    stub = tmp_path / "gh"
    stub.write_text(
        "#!/bin/bash\n"
        f'dir="{calls_dir}"\n'
        'count=$(find "$dir" -name "payload.*" | wc -l | tr -d "[:space:]")\n'
        'cat > "$dir/payload.$count"\n'
        'if [ "$count" -eq 0 ]; then\n'
        '  echo "HTTP 422: GitHub Actions is not permitted to approve pull requests." >&2\n'
        "  exit 1\n"
        "fi\n"
        'echo \'{"html_url": "https://example/r/real"}\'\n'
    )
    stub.chmod(0o755)
    return stub


def test_submit_review_retries_through_the_real_gh_command_boundary(
    helper: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The APPROVE downgrade works through a real subprocess, not a mocked ``subprocess.run``.

    Every other fallback test patches ``subprocess.run``, so none of them exercise argv
    construction, stdin piping, or stdout parsing. This drives a stub ``gh`` executable that
    refuses the first POST exactly as GitHub does and accepts the COMMENT retry.

    :param helper: The loaded ``post_review`` module.
    :param tmp_path: Scratch root for the stub executable and its call log.
    :param monkeypatch: Points ``gh_executable`` at the stub; ``subprocess`` stays real.
    """
    calls_dir = tmp_path / "calls"
    calls_dir.mkdir()
    monkeypatch.setattr(helper, "gh_executable", lambda: str(_write_stub_gh(tmp_path, calls_dir)))

    payload = {"body": "Clean review.", "event": "APPROVE"}
    response = helper.submit_review("o/r", 7, payload, fallback_banner="✅ No findings")

    assert response["html_url"] == "https://example/r/real"
    assert len(list(calls_dir.glob("payload.*"))) == 2
    assert json.loads((calls_dir / "payload.0").read_text())["event"] == "APPROVE"
    retried = json.loads((calls_dir / "payload.1").read_text())
    assert retried["event"] == "COMMENT"
    assert retried["body"].startswith("✅ No findings")


def test_submit_review_success_does_not_retry(
    helper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful first submit does not trigger the fallback retry.

    :param helper: The loaded ``post_review`` module.
    :param monkeypatch: Pytest fixture for patching ``subprocess.run``.
    """
    ok = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"html_url": "https://example/r/2"}),
        stderr="",
    )
    fake_run, calls = _fake_run_factory([ok])
    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    monkeypatch.setattr(helper, "gh_executable", lambda: "/usr/bin/gh")

    payload = {"body": "b", "event": "REQUEST_CHANGES", "comments": []}
    response = helper.submit_review("o/r", 7, payload, fallback_banner="x")

    assert response["html_url"] == "https://example/r/2"
    assert len(calls) == 1


def test_submit_review_non_self_422_exits(
    helper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-self-review API error propagates as SystemExit (no fallback).

    :param helper: The loaded ``post_review`` module.
    :param monkeypatch: Pytest fixture for patching ``subprocess.run``.
    """
    err = SimpleNamespace(returncode=1, stdout="", stderr="HTTP 404: Not Found")
    fake_run, _calls = _fake_run_factory([err])
    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    monkeypatch.setattr(helper, "gh_executable", lambda: "/usr/bin/gh")

    payload = {"body": "b", "event": "REQUEST_CHANGES", "comments": []}
    with pytest.raises(SystemExit):
        helper.submit_review("o/r", 7, payload, fallback_banner="x")
