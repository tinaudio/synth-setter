"""Tests for the Codex setup doctor."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCTOR = REPO_ROOT / "scripts" / "dev" / "codex-doctor.sh"
REQUIRED_PLUGIN_SKILLS = (
    "code-health",
    "github-taxonomy",
    "pr-checkbox",
    "pr-review-resolver",
    "simplify",
    "tdd-implementation",
)
MARKETPLACE_CODEX_REL = ".claude/plugins/marketplaces/tinaudio-skills/codex/synth-setter-skills"


def _write_fake_codex(bin_dir: Path) -> None:
    """Create a fake ``codex`` executable on PATH.

    :param bin_dir: Directory that will hold the executable.
    """
    bin_dir.mkdir()
    codex = bin_dir / "codex"
    codex.write_text("#!/usr/bin/env bash\nexit 0\n")
    codex.chmod(codex.stat().st_mode | stat.S_IXUSR)


def _write_user_skill(home: Path, name: str) -> None:
    """Create a fake user-level Codex skill.

    :param home: Fake home directory.
    :param name: Skill name.
    """
    skill_dir = home / ".agents" / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\n---\n")


def _write_claude_marketplace_skill(home: Path, name: str) -> None:
    """Create a fake skill in Claude's installed marketplace Codex projection.

    :param home: Isolated HOME root so the doctor cannot read the real cache.
    :param name: Required skill identifier that ``codex-doctor`` must discover.
    """
    skill_dir = home / MARKETPLACE_CODEX_REL / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\n---\n")


def _run_doctor(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the doctor with an isolated home, PATH, and git repository.

    :param tmp_path: Temporary test root.
    :returns: Completed process for assertions.
    """
    bin_dir = tmp_path / "bin"
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    _write_fake_codex(bin_dir)

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo_dir, check=True)  # noqa: S603, S607

    (repo_dir / "agent" / "skills").mkdir(parents=True)
    (repo_dir / "scripts" / "dev").mkdir(parents=True)

    (repo_dir / "scripts" / "dev" / "codex-doctor.sh").symlink_to(
        REPO_ROOT / "scripts" / "dev" / "codex-doctor.sh"
    )
    (repo_dir / "scripts" / "dev" / "link-skills.sh").symlink_to(
        REPO_ROOT / "scripts" / "dev" / "link-skills.sh"
    )
    (repo_dir / "agent" / "hooks").symlink_to(REPO_ROOT / "agent" / "hooks")
    (repo_dir / "agent" / "_shared").symlink_to(REPO_ROOT / "agent" / "_shared")

    (repo_dir / ".agents" / "plugins").mkdir(parents=True)
    (repo_dir / ".agents" / "plugins" / "marketplace.json").symlink_to(
        REPO_ROOT / ".agents" / "plugins" / "marketplace.json"
    )
    (repo_dir / ".agents" / "skills").symlink_to("../agent/skills")

    (repo_dir / "agent" / "skills" / "pr-readiness").symlink_to(
        REPO_ROOT / "agent" / "skills" / "pr-readiness"
    )

    env = {
        **os.environ,
        "HOME": str(home),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "SYNTH_SETTER_REPO_ROOT": str(repo_dir),
    }
    return subprocess.run(  # noqa: S603
        ["bash", str(repo_dir / "scripts" / "dev" / "codex-doctor.sh")],  # noqa: S607 - bash is required for shell doctor coverage
        cwd=repo_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_codex_doctor_with_required_plugin_skills_passes(tmp_path: Path) -> None:
    """Doctor passes when Codex and required reusable skills are discoverable.

    :param tmp_path: Temporary test root.
    """
    home = tmp_path / "home"
    for skill in REQUIRED_PLUGIN_SKILLS:
        _write_user_skill(home, skill)

    result = _run_doctor(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Codex setup looks ready." in result.stdout


def test_codex_doctor_projects_claude_marketplace_before_checking(
    tmp_path: Path,
) -> None:
    """Doctor self-heals from Claude's marketplace cache before skill checks.

    :param tmp_path: Isolated filesystem root that starts without ``~/.agents`` links.
    """
    home = tmp_path / "home"
    for skill in REQUIRED_PLUGIN_SKILLS:
        _write_claude_marketplace_skill(home, skill)

    result = _run_doctor(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Codex setup looks ready." in result.stdout
    assert (home / ".agents" / "skills" / "code-health").is_symlink()


def test_codex_doctor_missing_plugin_skills_prints_install_command(
    tmp_path: Path,
) -> None:
    """Doctor reports missing reusable skills without installing plugins.

    :param tmp_path: Temporary test root.
    """
    result = _run_doctor(tmp_path)

    assert result.returncode == 1
    assert "codex plugin marketplace add tinaudio/skills" in result.stdout
    assert "If tinaudio plugin skills are missing" in result.stdout
    assert (
        "Repo-local skills (e.g. pr-readiness) come from the repo skill projection"
        in result.stdout
    )
