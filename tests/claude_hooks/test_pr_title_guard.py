"""Tests for the pr-title-guard hook (``agent/hooks/pr_title_guard.py`` + ``.sh``).

The guard blocks ``gh pr create`` / ``gh pr edit`` titles whose conventional
type is off-vocabulary (``.gitlint``) or release-triggering (semantic-release
``minor_tags``/``patch_tags`` in ``pyproject.toml``) without explicit
``RELEASE_INTENT=1``. Command mode enforces both checks; commit-msg mode
enforces only the release reservation (gitlint owns vocabulary at that stage).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GUARD_PY = _REPO_ROOT / "agent" / "hooks" / "pr_title_guard.py"
_GUARD_SH = _REPO_ROOT / "agent" / "hooks" / "pr-title-guard.sh"


def _run_command_mode(
    command: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the checker in ``--command`` mode with ``command`` on stdin.

    :param command: Shell command text the PreToolUse payload would carry.
    :param env: Extra environment entries merged over a scrubbed base env.
    :returns: The completed process (findings on stdout, one per line).
    """
    import os

    base = {k: v for k, v in os.environ.items() if k != "RELEASE_INTENT"}
    if env:
        base.update(env)
    return subprocess.run(  # noqa: S603
        ["python3", str(_GUARD_PY), "--command"],  # noqa: S607
        input=command,
        capture_output=True,
        text=True,
        env=base,
        cwd=_REPO_ROOT,
    )


def _run_commit_msg_mode(
    tmp_path: Path, message: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the checker in ``--commit-msg-file`` mode against ``message``.

    :param tmp_path: Directory to write the commit-message file into.
    :param message: Commit message content.
    :param env: Extra environment entries merged over a scrubbed base env.
    :returns: The completed process.
    """
    import os

    msg_file = tmp_path / "COMMIT_EDITMSG"
    msg_file.write_text(message)
    base = {k: v for k, v in os.environ.items() if k != "RELEASE_INTENT"}
    if env:
        base.update(env)
    return subprocess.run(  # noqa: S603
        ["python3", str(_GUARD_PY), "--commit-msg-file", str(msg_file)],  # noqa: S607
        capture_output=True,
        text=True,
        env=base,
        cwd=_REPO_ROOT,
    )


def _run_hook_sh(
    command: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the ``.sh`` wrapper with a PreToolUse JSON payload for ``command``.

    :param command: Shell command to embed as ``tool_input.command``.
    :param env: Extra environment entries merged over a scrubbed base env.
    :returns: The completed process.
    """
    import os

    base = {k: v for k, v in os.environ.items() if k not in ("RELEASE_INTENT", "PR_TITLE_GUARD")}
    if env:
        base.update(env)
    payload = json.dumps({"tool_input": {"command": command}})
    return subprocess.run(  # noqa: S603
        ["bash", str(_GUARD_SH)],  # noqa: S607
        input=payload,
        capture_output=True,
        text=True,
        env=base,
        cwd=_REPO_ROOT,
    )


class TestCommandModeVocabulary:
    """Cover command mode vocabulary behavior."""

    def test_internal_fix_title_on_pr_create_passes(self) -> None:
        """Check that internal fix title on pr create passes."""
        result = _run_command_mode(
            'gh pr create --title "internal-fix(ci): guard titles" --body "x"'
        )
        assert result.stdout == ""
        assert result.returncode == 0

    def test_off_vocabulary_type_reports_finding(self) -> None:
        """Check that off vocabulary type reports finding."""
        result = _run_command_mode('gh pr create --title "wip: try things" --body "x"')
        assert "off-vocabulary" in result.stdout
        assert "wip" in result.stdout

    def test_non_conventional_title_reports_finding(self) -> None:
        """Check that non conventional title reports finding."""
        result = _run_command_mode('gh pr create --title "just some words" --body "x"')
        assert "conventional commit" in result.stdout

    def test_chore_scoped_title_passes(self) -> None:
        """Check that chore scoped title passes."""
        result = _run_command_mode(
            'gh pr create --title "chore(lint): drop baseline row" --body "x"'
        )
        assert result.stdout == ""


class TestCommandModeReleaseReservation:
    """Cover command mode release reservation behavior."""

    @pytest.mark.parametrize("rtype", ["feat", "fix", "perf", "revert"])
    def test_release_type_without_intent_reports_finding(self, rtype: str) -> None:
        """Check that release type without intent reports finding.

        :param rtype: The release-triggering type under test.
        """
        result = _run_command_mode(f'gh pr create --title "{rtype}: something" --body "x"')
        assert "release-triggering" in result.stdout
        assert "RELEASE_INTENT=1" in result.stdout

    def test_release_type_with_inline_intent_passes(self) -> None:
        """Check that release type with inline intent passes."""
        result = _run_command_mode(
            'RELEASE_INTENT=1 gh pr create --title "feat: ship the gate" --body "x"'
        )
        assert result.stdout == ""

    def test_release_type_with_env_prefixed_inline_intent_passes(self) -> None:
        """Check that release type with env prefixed inline intent passes."""
        result = _run_command_mode(
            'env RELEASE_INTENT=1 gh pr create --title "feat: ship the gate" --body "x"'
        )
        assert result.stdout == ""

    def test_release_type_with_ambient_intent_passes(self) -> None:
        """Check that release type with ambient intent passes."""
        result = _run_command_mode(
            'gh pr create --title "feat: ship the gate" --body "x"',
            env={"RELEASE_INTENT": "1"},
        )
        assert result.stdout == ""

    def test_feat_with_scope_and_bang_reports_finding(self) -> None:
        """Check that feat with scope and bang reports finding."""
        result = _run_command_mode('gh pr create --title "feat(training)!: breaking" --body "x"')
        assert "release-triggering" in result.stdout

    def test_internal_feat_is_not_release_triggering(self) -> None:
        """Check that internal feat is not release triggering."""
        result = _run_command_mode(
            'gh pr create --title "internal-feat(training): logic" --body "x"'
        )
        assert result.stdout == ""


class TestCommandModeScoping:
    """Cover command mode scoping behavior."""

    def test_gh_pr_edit_title_is_gated(self) -> None:
        """Check that gh pr edit title is gated."""
        result = _run_command_mode('gh pr edit 123 --title "fix: sneaky release"')
        assert "release-triggering" in result.stdout

    def test_title_equals_form_is_gated(self) -> None:
        """Check that title equals form is gated."""
        result = _run_command_mode("gh pr create --title='feat: x' --body y")
        assert "release-triggering" in result.stdout

    def test_short_t_flag_is_gated(self) -> None:
        """Check that short t flag is gated."""
        result = _run_command_mode('gh pr create -t "feat: x" -b y')
        assert "release-triggering" in result.stdout

    def test_quoted_prose_in_issue_body_is_not_gated(self) -> None:
        """Check that quoted prose in issue body is not gated."""
        result = _run_command_mode(
            'gh issue create --title "chore: doc" --body "recipe: gh pr create --title \'feat: x\'"'
        )
        assert result.stdout == ""

    def test_echoed_prose_is_not_gated(self) -> None:
        """Check that echoed prose is not gated."""
        result = _run_command_mode('echo gh pr create --title "feat: x"')
        assert result.stdout == ""

    def test_pr_create_without_title_flag_passes(self) -> None:
        """Check that pr create without title flag passes."""
        result = _run_command_mode("gh pr create --fill")
        assert result.stdout == ""

    def test_unrelated_command_passes(self) -> None:
        """Check that unrelated command passes."""
        result = _run_command_mode("git status && ls -la")
        assert result.stdout == ""

    def test_chained_pr_create_is_gated(self) -> None:
        """Check that chained pr create is gated."""
        result = _run_command_mode(
            'git push -u origin b && gh pr create --title "feat: x" --body y'
        )
        assert "release-triggering" in result.stdout

    def test_repo_flag_between_gh_and_pr_is_gated(self) -> None:
        """Check that repo flag between gh and pr is gated."""
        result = _run_command_mode(
            'gh -R tinaudio/synth-setter pr create --title "feat: x" --body y'
        )
        assert "release-triggering" in result.stdout

    def test_unparsable_command_mentioning_gated_surface_reports_finding(self) -> None:
        """Check that unparsable command mentioning gated surface reports finding."""
        result = _run_command_mode('gh pr create --title "feat: x')
        assert "unparsable" in result.stdout


class TestCommitMsgMode:
    """Cover commit msg mode behavior."""

    def test_feat_subject_without_intent_blocks(self, tmp_path: Path) -> None:
        """Check that feat subject without intent blocks.

        :param tmp_path: Where the commit-message file is written.
        """
        result = _run_commit_msg_mode(tmp_path, "feat: cut a release\n\nbody\n")
        assert result.returncode != 0
        assert "release-triggering" in result.stderr

    def test_internal_feat_subject_passes(self, tmp_path: Path) -> None:
        """Check that internal feat subject passes.

        :param tmp_path: Where the commit-message file is written.
        """
        result = _run_commit_msg_mode(tmp_path, "internal-feat(training): logic\n")
        assert result.returncode == 0

    def test_feat_subject_with_intent_env_passes(self, tmp_path: Path) -> None:
        """Check that feat subject with intent env passes.

        :param tmp_path: Where the commit-message file is written.
        """
        result = _run_commit_msg_mode(
            tmp_path, "feat: cut a release\n", env={"RELEASE_INTENT": "1"}
        )
        assert result.returncode == 0

    def test_off_vocabulary_subject_passes_here(self, tmp_path: Path) -> None:
        """Check that off vocabulary subject passes here.

        :param tmp_path: Where the commit-message file is written.
        """
        # Vocabulary at commit-msg stage is gitlint's job; the guard only
        # reserves release-triggering types.
        result = _run_commit_msg_mode(tmp_path, "wip: scratch\n")
        assert result.returncode == 0

    def test_comment_lines_before_subject_are_skipped(self, tmp_path: Path) -> None:
        """Check that comment lines before subject are skipped.

        :param tmp_path: Where the commit-message file is written.
        """
        result = _run_commit_msg_mode(tmp_path, "# hint from template\nfeat: x\n")
        assert result.returncode != 0


class TestCommitMsgWiring:
    """Pin the pre-commit registration that routes commit subjects to the guard."""

    def test_config_registers_guard_at_commit_msg_stage(self) -> None:
        """Check that the release-type-guard hook is wired at the commit-msg stage."""
        config = (_REPO_ROOT / ".pre-commit-config.yaml").read_text()
        hooks = [
            block for block in config.split("- id: ") if block.startswith("release-type-guard")
        ]
        assert len(hooks) == 1, "release-type-guard must be registered exactly once"
        block = hooks[0]
        assert "stages: [commit-msg]" in block
        assert "pr_title_guard.py --commit-msg-file" in block


class TestHookWrapper:
    """Cover hook wrapper behavior."""

    def test_release_title_blocks_with_exit_2(self) -> None:
        """Check that release title blocks with exit 2."""
        result = _run_hook_sh('gh pr create --title "feat: x" --body y')
        assert result.returncode == 2
        assert "BLOCKED" in result.stderr
        assert "internal-feat" in result.stderr

    def test_clean_title_exits_0(self) -> None:
        """Check that clean title exits 0."""
        result = _run_hook_sh('gh pr create --title "internal-fix(ci): x" --body y')
        assert result.returncode == 0

    def test_warn_mode_exits_0_with_warning(self) -> None:
        """Check that warn mode exits 0 with warning."""
        result = _run_hook_sh(
            'gh pr create --title "feat: x" --body y', env={"PR_TITLE_GUARD": "warn"}
        )
        assert result.returncode == 0
        assert "WARNING" in result.stderr

    def test_off_mode_exits_0_silently(self) -> None:
        """Check that off mode exits 0 silently."""
        result = _run_hook_sh(
            'gh pr create --title "feat: x" --body y', env={"PR_TITLE_GUARD": "off"}
        )
        assert result.returncode == 0
        assert result.stderr == ""

    def test_malformed_payload_fails_closed(self) -> None:
        """Check that malformed payload fails closed."""
        import os

        base = {
            k: v for k, v in os.environ.items() if k not in ("RELEASE_INTENT", "PR_TITLE_GUARD")
        }
        result = subprocess.run(  # noqa: S603
            ["bash", str(_GUARD_SH)],  # noqa: S607
            input="not json",
            capture_output=True,
            text=True,
            env=base,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 2

    def test_non_gh_command_exits_0(self) -> None:
        """Check that non gh command exits 0."""
        result = _run_hook_sh("make test-fast")
        assert result.returncode == 0
