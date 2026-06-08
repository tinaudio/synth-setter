"""Run the bash hook suite (`agent/hooks/test.sh`) in CI under a simulated Codex skill layout.

The suite is the authoritative contract for `agent/hooks/*`. Before #1561 only one
discovery-path case ran in CI (via ``test_settings_hooks.py``); this wrapper runs the whole suite
so every hook's exit contract is exercised on each PR, with HOME pointed at a Codex plugin-manifest
install so the run reflects a Codex-shaped environment rather than only the Claude one.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.infra

# The suite spins up sandbox git repos and headless-agent stubs; give it room
# without letting a hung child wedge the parent CI job indefinitely.
_TIMEOUT_S = 300

for _tool in ("bash", "git"):
    if shutil.which(_tool) is None:
        pytest.skip(f"{_tool} not on PATH", allow_module_level=True)


def _simulate_codex_skill_layout(home: Path) -> None:
    """Materialize a Codex plugin-manifest skill install under a throwaway ``home``.

    Mirrors the discovery glob ``has_skill`` matches
    (``~/.codex/plugins/*/codex/synth-setter-skills/<name>/SKILL.md``) so the suite runs as it would
    on a machine onboarded through the Codex CLI.

    :param home: Fake ``$HOME`` to populate.
    """
    skill_dir = (
        home
        / ".codex"
        / "plugins"
        / "tinaudio-synth-setter-skills"
        / "codex"
        / "synth-setter-skills"
        / "simplify"
    )
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: simplify\n---\n")


def test_agent_hooks_bash_suite_passes_under_codex_skill_layout(tmp_path: Path) -> None:
    """`agent/hooks/test.sh` reports zero failures when a Codex skill install is present.

    :param tmp_path: Per-test scratch directory used as the simulated ``$HOME``.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    _simulate_codex_skill_layout(fake_home)

    result = subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["bash", "agent/hooks/test.sh"],  # noqa: S607 — bash on PATH, repo-relative script
        cwd=PROJECT_ROOT,
        env={**os.environ, "HOME": str(fake_home)},
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert result.returncode == 0, f"hook suite failed:\n{result.stdout}\n{result.stderr}"
    # Parse the summary counts rather than substring-matching "FAIL: 0": a suite
    # that registered zero cases also prints "FAIL: 0" and exits 0, so require a
    # positive PASS count too. Coupled to test.sh's summary lines.
    passed = re.search(r"^PASS: (\d+)$", result.stdout, re.MULTILINE)
    failed = re.search(r"^FAIL: (\d+)$", result.stdout, re.MULTILINE)
    assert passed and failed, f"summary lines not found in:\n{result.stdout}"
    assert int(failed.group(1)) == 0, result.stdout
    assert int(passed.group(1)) > 0, f"suite registered no cases:\n{result.stdout}"
