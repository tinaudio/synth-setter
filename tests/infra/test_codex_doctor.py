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


def _write_fake_codex(
    bin_dir: Path,
    version: str = "codex-cli 0.144.1",
    *,
    exit_code: int = 0,
) -> Path:
    """Create a fake ``codex`` executable on PATH.

    :param bin_dir: Directory that will hold the executable.
    :param version: Version string printed by ``codex --version``.
    :param exit_code: Process exit code for the fake executable.
    :returns: Path to the executable.
    """
    bin_dir.mkdir(parents=True)
    codex = bin_dir / "codex"
    codex.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' '{version}'\nexit {exit_code}\n")
    codex.chmod(codex.stat().st_mode | stat.S_IXUSR)
    return codex


def _write_user_skill(home: Path, name: str) -> None:
    """Create a fake user-level Codex skill.

    :param home: Fake home directory.
    :param name: Skill name.
    """
    skill_dir = home / ".agents" / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\n---\n")


def _write_required_user_skills(home: Path) -> None:
    """Create every required reusable Codex skill in the fake home.

    :param home: Isolated HOME used by ``has_skill`` during doctor execution.
    """
    for skill in REQUIRED_PLUGIN_SKILLS:
        _write_user_skill(home, skill)


def _write_claude_marketplace_skill(home: Path, name: str) -> None:
    """Create a fake skill in Claude's installed marketplace Codex projection.

    :param home: Isolated HOME root so the doctor cannot read the real cache.
    :param name: Required skill identifier that ``codex-doctor`` must discover.
    """
    skill_dir = home / MARKETPLACE_CODEX_REL / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\n---\n")


def _create_doctor_repo(tmp_path: Path) -> Path:
    """Create an isolated repo fixture that can run the real doctor script.

    :param tmp_path: Test root that owns the isolated repository.
    :returns: Path to the isolated repository.
    """
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo_dir, check=True)  # noqa: S603, S607

    (repo_dir / "agent" / "skills").mkdir(parents=True)
    (repo_dir / "scripts" / "dev").mkdir(parents=True)
    (repo_dir / "scripts" / "dev" / "codex-doctor.sh").symlink_to(DOCTOR)
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
    return repo_dir


def _run_doctor(
    tmp_path: Path,
    *,
    extra_path_entries: tuple[Path, ...] = (),
    codex_exit_code: int = 0,
) -> subprocess.CompletedProcess[str]:
    """Run the doctor against an isolated repo with fake Codex PATH entries.

    :param tmp_path: Test root that owns HOME, PATH entries, and the repository.
    :param extra_path_entries: Additional fake PATH entries after the primary fake Codex.
    :param codex_exit_code: Exit code returned by the primary fake Codex.
    :returns: Completed doctor process for assertions.
    """
    bin_dir = tmp_path / "bin"
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    _write_fake_codex(bin_dir, exit_code=codex_exit_code)

    repo_dir = _create_doctor_repo(tmp_path)
    env = {
        **os.environ,
        "HOME": str(home),
        "PATH": os.pathsep.join(
            str(entry)
            for entry in (
                bin_dir,
                *extra_path_entries,
                Path("/usr/bin"),
                Path("/bin"),
            )
        ),
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
    _write_required_user_skills(home)

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


def test_codex_doctor_rejects_multiple_real_codex_installs(tmp_path: Path) -> None:
    """Doctor fails when PATH exposes separate Codex installations.

    :param tmp_path: Test root for fake HOME, repository, and PATH entries.
    """
    home = tmp_path / "home"
    _write_required_user_skills(home)
    _write_fake_codex(tmp_path / "older-bin", "codex-cli 0.139.0")

    result = _run_doctor(tmp_path, extra_path_entries=(tmp_path / "older-bin",))

    assert result.returncode == 1
    assert "Multiple Codex launchers resolve to different installs" in (
        result.stdout + result.stderr
    )


def test_codex_doctor_rejects_same_version_from_multiple_real_installs(
    tmp_path: Path,
) -> None:
    """Doctor fails when separate Codex installs report the same version.

    :param tmp_path: Test root for fake HOME, repository, and PATH entries.
    """
    home = tmp_path / "home"
    _write_required_user_skills(home)
    _write_fake_codex(tmp_path / "second-bin")

    result = _run_doctor(tmp_path, extra_path_entries=(tmp_path / "second-bin",))

    assert result.returncode == 1
    assert "Multiple Codex launchers resolve to different installs" in (
        result.stdout + result.stderr
    )


def test_codex_doctor_rejects_codex_launcher_without_version(
    tmp_path: Path,
) -> None:
    """Doctor fails when the selected Codex launcher cannot report its version.

    :param tmp_path: Test root for fake HOME, repository, and PATH entries.
    """
    home = tmp_path / "home"
    _write_required_user_skills(home)
    result = _run_doctor(tmp_path, codex_exit_code=1)

    assert result.returncode == 1
    assert "Codex launcher(s) failed to report a version" in result.stdout + result.stderr


def test_codex_doctor_allows_user_path_symlink_to_system_codex(
    tmp_path: Path,
) -> None:
    """Doctor accepts duplicate PATH entries when they point at one Codex install.

    :param tmp_path: Test root for fake HOME, repository, and PATH entries.
    """
    home = tmp_path / "home"
    _write_required_user_skills(home)
    user_bin = tmp_path / "user-bin"
    user_bin.mkdir()
    (user_bin / "codex").symlink_to(tmp_path / "bin" / "codex")

    result = _run_doctor(tmp_path, extra_path_entries=(user_bin,))

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
    assert "If tinaudio plugin skills are missing" in result.stdout
    assert (
        "Repo-local skills (e.g. pr-readiness) come from the repo skill projection"
        in result.stdout
    )
