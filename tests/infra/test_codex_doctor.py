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


def _run_doctor(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the doctor with an isolated home and PATH.

    :param tmp_path: Temporary test root.
    :returns: Completed process for assertions.
    """
    bin_dir = tmp_path / "bin"
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    _write_fake_codex(bin_dir)

    env = {
        **os.environ,
        "HOME": str(home),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
    }
    return subprocess.run(  # noqa: S603
        ["bash", str(DOCTOR)],  # noqa: S607 - bash is required for shell doctor coverage
        cwd=REPO_ROOT,
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


def test_codex_doctor_missing_plugin_skills_prints_install_command(
    tmp_path: Path,
) -> None:
    """Doctor reports missing reusable skills without installing plugins.

    :param tmp_path: Temporary test root.
    """
    result = _run_doctor(tmp_path)

    assert result.returncode == 1
    assert "codex plugin marketplace add tinaudio/skills" in result.stdout
    assert "Install or enable the tinaudio skill plugin" in result.stdout
