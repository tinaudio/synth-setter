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
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The hooks target the Linux devcontainer/CI environment. The bash suite has
# known macOS-only gaps (no GNU `timeout`/`gtimeout`; `/var`->`/private/var`
# canonicalization breaks the primary-root-from-subdir checks), so gate the
# wrapper to Linux — the ubuntu CI cells still exercise all cases. Making the
# suite itself macOS-clean is separate from this #1561 parity work.
pytestmark = [
    pytest.mark.infra,
    pytest.mark.skipif(sys.platform != "linux", reason="agent/hooks/test.sh is Linux-targeted"),
]

# The suite spins up sandbox git repos and headless-agent stubs; give it room
# without letting a hung child wedge the parent CI job indefinitely.
_TIMEOUT_S = 300

for _tool in ("bash", "git"):
    if shutil.which(_tool) is None:
        pytest.skip(f"{_tool} not on PATH", allow_module_level=True)


def _find_compatible_node() -> Path:
    """Return the first PATH Node that imports the real Pi TypeScript extension.

    :returns: Compatible Node executable path.
    :raises AssertionError: No PATH candidate supports the extension.
    """
    extension_uri = (PROJECT_ROOT / ".pi/extensions/pr-readiness-stop.ts").as_uri()
    for node_dir in os.environ["PATH"].split(os.pathsep):
        candidate = Path(node_dir or ".") / "node"
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        probe = subprocess.run(  # noqa: S603 — PATH candidate, fixed argv
            [
                str(candidate),
                "--experimental-strip-types",
                "--input-type=module",
                "-e",
                f'await import("{extension_uri}");',
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            return candidate
    raise AssertionError("install Node with TypeScript stripping support")


def _write_incompatible_node(bin_dir: Path) -> Path:
    """Create a Node stand-in that rejects TypeScript stripping.

    :param bin_dir: New directory that will contain the stand-in.
    :returns: Executable stand-in path.
    """
    bin_dir.mkdir()
    node = bin_dir / "node"
    node.write_text(
        "#!/bin/bash\nprintf '%s\\n' 'node: bad option: --experimental-strip-types' >&2\nexit 9\n"
    )
    node.chmod(0o755)
    return node


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


def test_agent_hooks_bash_suite_passes_under_codex_skill_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`agent/hooks/test.sh` reports zero failures when a Codex skill install is present.

    :param tmp_path: Per-test scratch directory used as the simulated ``$HOME``.
    :param monkeypatch: Environment isolation fixture.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    _simulate_codex_skill_layout(fake_home)

    compatible_node_bin = tmp_path / "compatible-node-bin"
    compatible_node_bin.mkdir()
    (compatible_node_bin / "node").symlink_to(_find_compatible_node())
    old_node_bin = tmp_path / "old-node-bin"
    _write_incompatible_node(old_node_bin)

    interpreter_bin = str(Path(sys.executable).parent)
    sanitized_path = os.pathsep.join((interpreter_bin, str(old_node_bin), os.defpath))
    monkeypatch.setenv("PATH", sanitized_path)
    assert str(compatible_node_bin) not in os.environ["PATH"].split(os.pathsep)

    result = subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["bash", "agent/hooks/test.sh"],  # noqa: S607 — bash on PATH, repo-relative script
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "HOME": str(fake_home),
            "HOOK_TEST_NODE_PATH": os.pathsep.join((str(old_node_bin), str(compatible_node_bin))),
            "PATH": os.environ["PATH"],
        },
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
    assert "  PASS  Pi readiness: settled blocking report re-prompts once" in result.stdout
    assert "  PASS  Pi readiness: passing result re-arms future nudge" in result.stdout
    assert "  PASS  Pi readiness: print mode does not re-prompt" in result.stdout
    assert "  PASS  Pi readiness: warn mode displays an advisory" in result.stdout
    assert "  PASS  Pi readiness: hook path is repository-absolute" in result.stdout


def test_agent_hooks_bash_suite_without_compatible_node_errors(tmp_path: Path) -> None:
    """`agent/hooks/test.sh` reports how to provide a compatible Node runtime.

    :param tmp_path: Per-test scratch directory for the incompatible executable.
    """
    old_node_bin = tmp_path / "old-node-bin"
    _write_incompatible_node(old_node_bin)

    result = subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["bash", "agent/hooks/test.sh"],  # noqa: S607 — bash on PATH, repo-relative script
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "HOOK_TEST_NODE_PATH": str(old_node_bin),
            "PATH": os.pathsep.join((str(old_node_bin), os.defpath)),
        },
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert result.returncode != 0
    assert "no compatible Node found" in result.stderr
    assert "HOOK_TEST_NODE_PATH" in result.stderr
