"""Parity: every shipped skill resolves through both the Claude and Codex discovery globs.

`agent/hooks/_lib.sh`'s ``has_skill`` is the single discovery point used by the headless
`doc-drift` hook. #1558 taught it the Codex plugin-manifest layout alongside the
Claude one; this test pins that the two stay symmetric — for each ``agent/skills/<name>``, a skill
installed only via the Claude marketplace glob and one installed only via the Codex plugin-manifest
glob are both found. The repo-relative ``agent/skills/`` fallback is deliberately sidestepped by
running from a non-repo cwd so the install-path globs — not the source tree — are what's exercised.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIB = _REPO_ROOT / "agent" / "hooks" / "_lib.sh"
_MARKETPLACE = "tinaudio-synth-setter-skills"


def _shipped_skill_names() -> list[str]:
    """Return every ``agent/skills/<name>`` directory that ships a ``SKILL.md``.

    :returns: Sorted skill names (``_shared`` and other SKILL.md-less dirs excluded).
    """
    skills_dir = _REPO_ROOT / "agent" / "skills"
    return sorted(p.parent.name for p in skills_dir.glob("*/SKILL.md"))


_SKILL_NAMES = _shipped_skill_names()


def _claude_skill_path(home: Path, name: str) -> Path:
    """Path a skill occupies when installed via the Claude plugin marketplace.

    :param home: Simulated ``$HOME`` install root.
    :param name: Skill name.
    :returns: The ``SKILL.md`` path under the Claude marketplace layout.
    """
    return home / ".claude" / "plugins" / _MARKETPLACE / "skills" / name / "SKILL.md"


def _codex_skill_path(home: Path, name: str) -> Path:
    """Path a skill occupies when installed via the Codex plugin manifest.

    :param home: Simulated ``$HOME`` install root.
    :param name: Skill name.
    :returns: The ``SKILL.md`` path under the Codex plugin-manifest layout.
    """
    return (
        home
        / ".codex"
        / "plugins"
        / _MARKETPLACE
        / "codex"
        / "synth-setter-skills"
        / name
        / "SKILL.md"
    )


def _install(skill_file: Path) -> None:
    """Create a minimal ``SKILL.md`` at ``skill_file``.

    :param skill_file: Destination ``SKILL.md`` path (parents created).
    """
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    skill_file.write_text("---\nname: stub\n---\n")


def _has_skill(name: str, home: Path, cwd: Path) -> bool:
    """Run ``has_skill <name>`` with ``$HOME`` and cwd overridden.

    cwd is a non-repo directory so ``has_skill``'s repo-relative ``agent/skills/`` lookups cannot
    short-circuit the install-path globs under test.

    :param name: Skill name to resolve.
    :param home: Value for ``$HOME`` (the simulated install root).
    :param cwd: Working directory to run from (outside any git repo).
    :returns: True when ``has_skill`` exits 0.
    """
    result = subprocess.run(  # noqa: S603 — fixed argv, no shell string
        ["bash", "-c", 'source "$1"; has_skill "$2"', "_", str(_LIB), name],  # noqa: S607 — bash on PATH
        cwd=cwd,
        env={**os.environ, "HOME": str(home)},
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def test_skills_are_shipped() -> None:
    """The discovery set is non-empty, so the parity test below is not vacuously green."""
    assert _SKILL_NAMES


def test_uninstalled_skill_is_not_found(tmp_path: Path) -> None:
    """With no install present, ``has_skill`` returns non-zero.

    Guards the parity tests against silently passing because the repo-relative fallback (rather than
    an install glob) matched.

    :param tmp_path: Scratch dir holding an empty fake ``$HOME`` and a non-repo cwd.
    """
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    assert not _has_skill("repo-review", home, cwd)


@pytest.mark.parametrize("name", _SKILL_NAMES)
def test_skill_resolves_via_claude_marketplace_glob(name: str, tmp_path: Path) -> None:
    """A skill installed only through the Claude marketplace glob is discoverable.

    :param name: Shipped skill name under test.
    :param tmp_path: Scratch dir holding both the fake ``$HOME`` and a non-repo cwd.
    """
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _install(_claude_skill_path(home, name))
    assert _has_skill(name, home, cwd)


@pytest.mark.parametrize("name", _SKILL_NAMES)
def test_skill_resolves_via_codex_plugin_manifest_glob(name: str, tmp_path: Path) -> None:
    """A skill installed only through the Codex plugin-manifest glob is discoverable.

    :param name: Shipped skill name under test.
    :param tmp_path: Scratch dir holding both the fake ``$HOME`` and a non-repo cwd.
    """
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _install(_codex_skill_path(home, name))
    assert _has_skill(name, home, cwd)
