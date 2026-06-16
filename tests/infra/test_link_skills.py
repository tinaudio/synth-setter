"""Tests for the ``link-skills`` projection into ``~/.agents/skills``."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LINK_SKILLS = REPO_ROOT / "scripts" / "dev" / "link-skills.sh"
MARKETPLACE_REL = ".claude/plugins/marketplaces/tinaudio-skills"
CODEX_SKILLS_REL = f"{MARKETPLACE_REL}/codex/synth-setter-skills"


def _write_marketplace_skill(home: Path, name: str) -> Path:
    """Create a fake installed marketplace skill under the Claude plugin cache.

    :param home: Fake home directory.
    :param name: Skill name.
    :returns: The created skill directory.
    """
    skill_dir = home / MARKETPLACE_REL / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\n---\n")
    return skill_dir


def _write_codex_marketplace_skill(home: Path, name: str) -> Path:
    """Create a fake installed skill under the Claude cache's Codex projection.

    :param home: Isolated HOME root so projection never reads the real cache.
    :param name: Skill identifier expected to become a ``~/.agents`` link.
    :returns: Directory that the projected link must resolve to.
    """
    skill_dir = home / CODEX_SKILLS_REL / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\n---\n")
    return skill_dir


def _run_link_skills(home: Path, repo: Path) -> subprocess.CompletedProcess[str]:
    """Run ``link-skills.sh`` with an isolated home and repo root.

    :param home: Fake home directory the script reads and writes through.
    :param repo: Fake repo directory.
    :returns: Completed process for assertions.
    """
    env = {**os.environ, "HOME": str(home), "SYNTH_SETTER_REPO_ROOT": str(repo)}
    (repo / "agent" / "skills").mkdir(parents=True, exist_ok=True)
    return subprocess.run(  # noqa: S603
        ["bash", str(LINK_SKILLS)],  # noqa: S607 - bash is required for the shell script
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_link_skills_projects_marketplace_skills_into_agents_skills(tmp_path: Path) -> None:
    """Each marketplace skill dir becomes a resolvable ``~/.agents/skills`` symlink.

    :param tmp_path: Temporary test root.
    """
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    code_health = _write_marketplace_skill(home, "code-health")
    tdd = _write_marketplace_skill(home, "tdd-implementation")

    result = _run_link_skills(home, repo)

    assert result.returncode == 0, result.stderr
    dest = home / ".agents" / "skills"
    for name, source in (("code-health", code_health), ("tdd-implementation", tdd)):
        link = dest / name
        assert link.is_symlink()
        assert link.resolve() == source.resolve()


def test_link_skills_projects_claude_codex_skill_projection(tmp_path: Path) -> None:
    """The Claude marketplace's Codex skill path is projected into ``~/.agents``.

    :param tmp_path: Temporary test root.
    """
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    simplify = _write_codex_marketplace_skill(home, "simplify")

    result = _run_link_skills(home, repo)

    assert result.returncode == 0, result.stderr
    link = home / ".agents" / "skills" / "simplify"
    assert link.is_symlink()
    assert link.resolve() == simplify.resolve()


def test_link_skills_falls_back_when_codex_projection_is_empty(tmp_path: Path) -> None:
    """An empty Codex projection does not mask top-level marketplace skills.

    :param tmp_path: Isolated filesystem root for the fake marketplace cache.
    """
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    code_health = _write_marketplace_skill(home, "code-health")
    (home / CODEX_SKILLS_REL).mkdir(parents=True)

    result = _run_link_skills(home, repo)

    assert result.returncode == 0, result.stderr
    link = home / ".agents" / "skills" / "code-health"
    assert link.is_symlink()
    assert link.resolve() == code_health.resolve()


def test_link_skills_projects_codex_and_top_level_only_skills(tmp_path: Path) -> None:
    """Mixed marketplace layouts project both Codex and top-level-only skills.

    :param tmp_path: Isolated filesystem root for the fake mixed-layout cache.
    """
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    codex_skill = _write_codex_marketplace_skill(home, "code-health")
    top_level_only = _write_marketplace_skill(home, "wiki-content")

    result = _run_link_skills(home, repo)

    assert result.returncode == 0, result.stderr
    dest = home / ".agents" / "skills"
    assert (dest / "code-health").resolve() == codex_skill.resolve()
    assert (dest / "wiki-content").resolve() == top_level_only.resolve()


def test_link_skills_skips_marketplace_dirs_without_a_skill_file(tmp_path: Path) -> None:
    """A marketplace subdir lacking ``SKILL.md`` (e.g. ``_shared``) is not projected.

    :param tmp_path: Temporary test root.
    """
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    _write_marketplace_skill(home, "code-health")
    (home / MARKETPLACE_REL / "_shared").mkdir(parents=True)

    result = _run_link_skills(home, repo)

    assert result.returncode == 0, result.stderr
    assert not (home / ".agents" / "skills" / "_shared").exists()


def test_link_skills_without_marketplace_cache_is_noop(tmp_path: Path) -> None:
    """Absent the marketplace cache the script exits 0 and creates no links.

    :param tmp_path: Temporary test root.
    """
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    home.mkdir()

    result = _run_link_skills(home, repo)

    assert result.returncode == 0, result.stderr
    assert not (home / ".agents" / "skills").exists()


def test_link_skills_is_idempotent(tmp_path: Path) -> None:
    """Re-running over an already-projected home stays green and keeps the link.

    :param tmp_path: Temporary test root.
    """
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    source = _write_marketplace_skill(home, "code-health")

    first = _run_link_skills(home, repo)
    second = _run_link_skills(home, repo)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    link = home / ".agents" / "skills" / "code-health"
    assert link.is_symlink()
    assert link.resolve() == source.resolve()


def test_link_skills_projects_into_workspace_agent_skills(tmp_path: Path) -> None:
    """Marketplace skills not present natively are symlinked into workspace and excluded.

    :param tmp_path: Temporary test root.
    """
    home = tmp_path / "home"
    repo = tmp_path / "repo"

    # Create native repository skills (real directories)
    native_skills_dir = repo / "agent" / "skills"
    native_skills_dir.mkdir(parents=True)
    (native_skills_dir / "pr-readiness").mkdir()
    (native_skills_dir / "pr-readiness" / "SKILL.md").write_text("---\nname: pr-readiness\n---\n")

    # Create marketplace skills
    code_health = _write_marketplace_skill(home, "code-health")
    _write_marketplace_skill(home, "pr-readiness")

    result = _run_link_skills(home, repo)

    assert result.returncode == 0, result.stderr

    # Verify code-health is symlinked into repo's agent/skills
    linked_code_health = native_skills_dir / "code-health"
    assert linked_code_health.is_symlink()
    assert linked_code_health.resolve() == code_health.resolve()

    # Verify native pr-readiness is NOT overwritten or symlinked
    native_pr_readiness = native_skills_dir / "pr-readiness"
    assert not native_pr_readiness.is_symlink()

    # Verify code-health was added to the exclude file
    exclude_file = repo / ".git" / "info" / "exclude"
    assert exclude_file.exists()
    exclude_content = exclude_file.read_text()
    assert "agent/skills/code-health" in exclude_content
    assert "agent/skills/pr-readiness" not in exclude_content


def test_link_skills_exclude_file_is_idempotent(tmp_path: Path) -> None:
    """Re-running the script does not append duplicate entries to git exclude file.

    :param tmp_path: Temporary test root.
    """
    home = tmp_path / "home"
    repo = tmp_path / "repo"

    _write_marketplace_skill(home, "code-health")

    first = _run_link_skills(home, repo)
    second = _run_link_skills(home, repo)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    exclude_file = repo / ".git" / "info" / "exclude"
    assert exclude_file.exists()
    exclude_lines = exclude_file.read_text().splitlines()
    assert exclude_lines.count("agent/skills/code-health") == 1
