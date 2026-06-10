"""Tests for the ``link-skills`` projection into ``~/.agents/skills``."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LINK_SKILLS = REPO_ROOT / "scripts" / "dev" / "link-skills.sh"
MARKETPLACE_REL = ".claude/plugins/marketplaces/tinaudio-skills"


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


def _run_link_skills(home: Path) -> subprocess.CompletedProcess[str]:
    """Run ``link-skills.sh`` with an isolated home.

    :param home: Fake home directory the script reads and writes through.
    :returns: Completed process for assertions.
    """
    env = {**os.environ, "HOME": str(home)}
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
    code_health = _write_marketplace_skill(home, "code-health")
    tdd = _write_marketplace_skill(home, "tdd-implementation")

    result = _run_link_skills(home)

    assert result.returncode == 0, result.stderr
    dest = home / ".agents" / "skills"
    for name, source in (("code-health", code_health), ("tdd-implementation", tdd)):
        link = dest / name
        assert link.is_symlink()
        assert link.resolve() == source.resolve()


def test_link_skills_skips_marketplace_dirs_without_a_skill_file(tmp_path: Path) -> None:
    """A marketplace subdir lacking ``SKILL.md`` (e.g. ``_shared``) is not projected.

    :param tmp_path: Temporary test root.
    """
    home = tmp_path / "home"
    _write_marketplace_skill(home, "code-health")
    (home / MARKETPLACE_REL / "_shared").mkdir(parents=True)

    result = _run_link_skills(home)

    assert result.returncode == 0, result.stderr
    assert not (home / ".agents" / "skills" / "_shared").exists()


def test_link_skills_without_marketplace_cache_is_noop(tmp_path: Path) -> None:
    """Absent the marketplace cache the script exits 0 and creates no links.

    :param tmp_path: Temporary test root.
    """
    home = tmp_path / "home"
    home.mkdir()

    result = _run_link_skills(home)

    assert result.returncode == 0, result.stderr
    assert not (home / ".agents" / "skills").exists()


def test_link_skills_is_idempotent(tmp_path: Path) -> None:
    """Re-running over an already-projected home stays green and keeps the link.

    :param tmp_path: Temporary test root.
    """
    home = tmp_path / "home"
    source = _write_marketplace_skill(home, "code-health")

    first = _run_link_skills(home)
    second = _run_link_skills(home)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    link = home / ".agents" / "skills" / "code-health"
    assert link.is_symlink()
    assert link.resolve() == source.resolve()
