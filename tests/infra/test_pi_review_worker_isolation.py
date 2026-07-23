"""Real-git behavior tests for isolated Pi review workers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import sh

REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTING_SCRIPT = REPO_ROOT / "agent/_shared/pi_review_routing.py"
AFTERCARE_SCRIPT = REPO_ROOT / "agent/_shared/run_pi_review_aftercare.py"
LAUNCHER = REPO_ROOT / "agent/_shared/run_pi_review.sh"


def _git(repo: Path, *args: str) -> str:
    return str(sh.Command("git")("--no-pager", "-c", "color.ui=false", "-C", repo, *args)).strip()


def _tiny_repo(tmp_path: Path) -> tuple[Path, str]:
    """Create a committed repository with ignored review artifacts.

    :param tmp_path: Temporary parent directory.
    :returns: Repository root and full HEAD SHA.
    """
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "review-test@example.com")
    _git(repo, "config", "user.name", "Review Test")
    (repo / ".gitignore").write_text(".agent-reviews/\n")
    checklist = repo / "agent/skills/correctness-review/SKILL.md"
    checklist.parent.mkdir(parents=True)
    checklist.write_text("# Correctness review\n")
    (repo / "uv.lock").write_text("source-lock\n")
    _git(repo, "add", ".gitignore", "agent", "uv.lock")
    _git(repo, "commit", "--quiet", "-m", "test: seed repository")
    return repo, _git(repo, "rev-parse", "HEAD")


def _prepare_worktree(repo: Path, head_sha: str) -> tuple[Path, Path, dict[str, str]]:
    """Prepare a review worktree through the real routing CLI.

    :param repo: Temporary source checkout.
    :param head_sha: Commit reviewed by workers.
    :returns: Manifest path, disposable worktree path, and strict metadata payload.
    """
    manifest = repo / ".agent-reviews/review.json"
    result = sh.Command(sys.executable)(
        ROUTING_SCRIPT,
        "prepare-review-worktree",
        "--manifest",
        manifest,
        "--head-sha",
        head_sha,
        _cwd=repo,
    )
    metadata = json.loads(str(result))
    return manifest, Path(metadata["path"]), metadata


def _write_manifest(manifest: Path, metadata: dict[str, str], head_sha: str) -> None:
    """Write one strict aftercare manifest containing worktree ownership metadata.

    :param manifest: Foreground manifest path.
    :param metadata: Prepared disposable-worktree metadata.
    :param head_sha: Reviewed commit.
    """
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "mode": "no-comments",
                "repo": "tinaudio/synth-setter",
                "pr_number": 2330,
                "base_sha": head_sha,
                "head_sha": head_sha,
                "target": "PR #2330",
                "deferred_passes": [
                    {
                        "skill": "correctness-review",
                        "pass_name": "free-pool",
                        "origin": "primary",
                        "model": "kimi-coding/k3",
                        "verification_model": "openai-codex/gpt-5.6-sol",
                        "thinking": "high",
                    }
                ],
                "foreground_fingerprints": [],
                "review_worktree": metadata,
            }
        )
    )


def _valid_aftercare_result() -> str:
    """Return a strict successful child result payload.

    :returns: Serialized aftercare result.
    """
    return json.dumps(
        {
            "status": "complete",
            "attempts": [],
            "diagnostics": [],
            "late_findings": [],
            "posted_review_url": None,
            "child_exit_code": None,
            "log_tail": "",
            "completed_at": "2026-07-24T00:00:00Z",
        }
    )


def test_review_worktree_cli_isolates_mutation_and_stash_from_source(tmp_path: Path) -> None:
    """Create, use, and remove a detached review copy without dirtying its source.

    :param tmp_path: Temporary real Git repository parent.
    """
    repo, head_sha = _tiny_repo(tmp_path)
    manifest, review_worktree, _ = _prepare_worktree(repo, head_sha)
    prompt_path = manifest.with_suffix("").with_suffix(".assignments") / "correctness-review.txt"

    sh.Command(sys.executable)(
        ROUTING_SCRIPT,
        "worker-prompt",
        "--manifest",
        manifest,
        "--skill",
        "correctness-review",
        "--target",
        "PR #2330",
        "--repo",
        "tinaudio/synth-setter",
        "--base-sha",
        head_sha,
        "--head-sha",
        head_sha,
        "--changed-path",
        "uv.lock",
        "--output",
        prompt_path,
        _cwd=repo,
    )
    prompt = prompt_path.read_text()
    assert f"Worker cwd: {review_worktree}\n" in prompt
    assert f"Every git and pytest command must run with cwd {review_worktree}." in prompt
    assert (
        f"Read the checklist at `{review_worktree}/agent/skills/correctness-review/SKILL.md`"
        in prompt
    )
    assert str(repo / "agent/skills/correctness-review/SKILL.md") not in prompt
    assert f"Worker cwd: {repo}\n" not in prompt

    (review_worktree / "uv.lock").write_text("worker-lock\n")
    _git(review_worktree, "stash", "push", "--message", "worker mutation")
    (review_worktree / "uv.lock").write_text("conflicting-worker-lock\n")
    _git(review_worktree, "add", "uv.lock")
    _git(review_worktree, "commit", "--quiet", "-m", "test: create worker conflict")
    with pytest.raises(sh.ErrorReturnCode):
        sh.Command("git")("-C", review_worktree, "stash", "pop")

    assert _git(review_worktree, "diff", "--diff-filter=U", "--name-only") == "uv.lock"
    assert (repo / "uv.lock").read_text() == "source-lock\n"
    assert _git(repo, "status", "--short") == ""

    sh.Command(sys.executable)(
        ROUTING_SCRIPT,
        "cleanup-review-worktree",
        "--manifest",
        manifest,
        _cwd=repo,
    )

    assert not review_worktree.exists()
    assert str(review_worktree) not in _git(repo, "worktree", "list", "--porcelain")
    assert _git(repo, "status", "--short") == ""


def test_cleanup_review_worktree_rejects_metadata_outside_assignment_scope(
    tmp_path: Path,
) -> None:
    """Refuse cleanup when persisted metadata is redirected to the source checkout.

    :param tmp_path: Temporary real Git repository parent.
    """
    repo, head_sha = _tiny_repo(tmp_path)
    manifest, review_worktree, metadata = _prepare_worktree(repo, head_sha)
    metadata["path"] = str(repo)
    Path(f"{manifest}.review-worktree.json").write_text(json.dumps(metadata))

    with pytest.raises(sh.ErrorReturnCode):
        sh.Command(sys.executable)(
            ROUTING_SCRIPT,
            "cleanup-review-worktree",
            "--manifest",
            manifest,
            _cwd=repo,
        )

    assert repo.exists()
    assert review_worktree.exists()
    assert _git(repo, "status", "--short") == ""


def test_aftercare_supervisor_routes_prompt_and_cleans_disposable_worktree(
    tmp_path: Path,
) -> None:
    """Keep aftercare workers in the detached copy and remove it on supervisor exit.

    :param tmp_path: Temporary real Git repository and fake Pi parent.
    """
    repo, head_sha = _tiny_repo(tmp_path)
    manifest, review_worktree, metadata = _prepare_worktree(repo, head_sha)
    _write_manifest(manifest, metadata, head_sha)
    (review_worktree / "uv.lock").write_text("aftercare-lock\n")
    _git(review_worktree, "stash", "push", "--message", "aftercare mutation")

    prompt_capture = tmp_path / "aftercare-prompt.txt"
    fake_pi = tmp_path / "pi"
    fake_pi.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, subprocess, sys\n"
        "from pathlib import Path\n"
        "runtime = Path(os.environ['PI_REVIEW_AFTERCARE_RUNTIME_MANIFEST'])\n"
        "payload = json.loads(runtime.read_text())\n"
        "assignment = Path(payload['review_worktree']['assignment_dir']) / "
        "'correctness-review.txt'\n"
        "subprocess.run([sys.executable, os.environ['ROUTING_SCRIPT'], 'worker-prompt', "
        "'--manifest', str(runtime), '--skill', 'correctness-review', '--target', "
        "'PR #2330', '--repo', 'tinaudio/synth-setter', '--base-sha', "
        "os.environ['HEAD_SHA'], '--head-sha', os.environ['HEAD_SHA'], "
        "'--changed-path', 'uv.lock', '--output', str(assignment)], check=True)\n"
        "Path(os.environ['PROMPT_CAPTURE']).write_text(assignment.read_text())\n"
        "Path(f'{runtime}.result.json').write_text(os.environ['FAKE_RESULT'])\n"
    )
    fake_pi.chmod(0o755)
    environment = {
        **os.environ,
        "FAKE_RESULT": _valid_aftercare_result(),
        "HEAD_SHA": head_sha,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "PROMPT_CAPTURE": str(prompt_capture),
        "ROUTING_SCRIPT": str(ROUTING_SCRIPT),
    }

    sh.Command(sys.executable)(
        AFTERCARE_SCRIPT,
        "--supervise",
        manifest,
        _cwd=repo,
        _env=environment,
    )

    prompt = prompt_capture.read_text()
    assert f"Worker cwd: {review_worktree}\n" in prompt
    assert f"Every git and pytest command must run with cwd {review_worktree}." in prompt
    assert not review_worktree.exists()
    assert (repo / "uv.lock").read_text() == "source-lock\n"
    assert _git(repo, "status", "--short") == ""


def test_launcher_without_aftercare_cleans_disposable_worktree_after_delivery(
    tmp_path: Path,
) -> None:
    """Remove the foreground review copy when no aftercare manifest is launched.

    :param tmp_path: Temporary real Git repository and fake Pi parent.
    """
    repo, _ = _tiny_repo(tmp_path)
    shared_dir = repo / "agent/_shared"
    shared_dir.mkdir(parents=True)
    for source in (LAUNCHER, ROUTING_SCRIPT):
        destination = shared_dir / source.name
        destination.write_text(source.read_text())
        destination.chmod(source.stat().st_mode)
    _git(repo, "add", "agent")
    _git(repo, "commit", "--quiet", "-m", "test: add review launcher")

    worktree_path_file = tmp_path / "worktree-path"
    fake_pi = tmp_path / "pi"
    fake_pi.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, subprocess, sys\n"
        "from pathlib import Path\n"
        "head = subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, "
        "capture_output=True, text=True).stdout.strip()\n"
        "result = subprocess.run([os.environ['PI_REVIEW_PYTHON'], "
        "'agent/_shared/pi_review_routing.py', 'prepare-review-worktree', "
        "'--manifest', os.environ['PI_REVIEW_AFTERCARE_MANIFEST'], '--head-sha', head], "
        "check=True, capture_output=True, text=True)\n"
        "worktree = Path(json.loads(result.stdout)['path'])\n"
        "Path(os.environ['WORKTREE_PATH_FILE']).write_text(str(worktree))\n"
        "(worktree / 'uv.lock').write_text('foreground-lock\\n')\n"
        "subprocess.run(['git', '-C', str(worktree), 'stash', 'push', '--message', "
        "'foreground mutation'], check=True, capture_output=True)\n"
        "print(json.dumps({'type': 'message_end', 'message': "
        "{'role': 'assistant', 'content': 'foreground-complete'}}))\n"
    )
    fake_pi.chmod(0o755)

    result = sh.Command(str(shared_dir / "run_pi_review.sh"))(
        "repo-review-full-no-comments",
        "--target",
        "2330",
        _cwd=repo,
        _env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "WORKTREE_PATH_FILE": str(worktree_path_file),
        },
    )

    review_worktree = Path(worktree_path_file.read_text())
    assert str(result).strip() == "foreground-complete"
    assert not review_worktree.exists()
    assert (repo / "uv.lock").read_text() == "source-lock\n"
    assert _git(repo, "status", "--short") == ""
