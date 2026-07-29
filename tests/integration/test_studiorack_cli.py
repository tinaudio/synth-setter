"""Production-path Studiorack CLI artifact-lock integration coverage."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDIORACK_CLI = PROJECT_ROOT / "node_modules/.bin/studiorack"
STUDIORACK_LOCK = PROJECT_ROOT / "studiorack.lock.json"
DEXED_REFERENCE = "asb2m10/dexed@0.9.8"
DEXED_PLUGIN = Path("VST3/asb2m10/dexed/0.9.8/Dexed.vst3/Contents/x86_64-linux/Dexed.so")

pytestmark = [
    pytest.mark.infra,
    pytest.mark.network,
    pytest.mark.skipif(
        sys.platform != "linux" or platform.machine() not in {"AMD64", "x86_64"},
        reason="Linux x64 Dexed lock is pinned",
    ),
]


@dataclass(frozen=True)
class StudiorackRoot:
    """Paths for one isolated Studiorack CLI test root.

    .. attribute :: home

        Temporary home containing the CLI configuration.

    .. attribute :: lock_path

        Isolated artifact-lock path.

    .. attribute :: plugins_dir

        Isolated plugin installation root.
    """

    home: Path
    lock_path: Path
    plugins_dir: Path


def _run_studiorack(home: Path, args: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    """Invoke the repository-pinned Studiorack CLI under an isolated home.

    :param home: Temporary home containing the CLI configuration.
    :param args: Studiorack command arguments.
    :returns: Completed CLI process with captured text output.
    """
    env = os.environ.copy()
    env["HOME"] = str(home)
    env.pop("SYNTH_SETTER_STUDIORACK_ARTIFACT_LOCK", None)
    return subprocess.run(  # noqa: S603 — fixed repository executable
        [str(STUDIORACK_CLI), *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=180,
    )


@pytest.fixture()
def studiorack_root(tmp_path: Path) -> StudiorackRoot:
    """Create an isolated root configured with the shipped artifact lock.

    :param tmp_path: Per-test filesystem root.
    :returns: Paths consumed by the real Studiorack CLI.
    """
    if not STUDIORACK_CLI.exists():
        pytest.skip("repository-pinned Studiorack CLI missing; run npm ci")
    cli_manifest = json.loads(
        (PROJECT_ROOT / "node_modules/@studiorack/cli/package.json").read_text()
    )
    assert cli_manifest["version"] == "3.0.6"

    root = StudiorackRoot(
        home=tmp_path / "home",
        lock_path=tmp_path / "studiorack.lock.json",
        plugins_dir=tmp_path / "plugins",
    )
    root.lock_path.write_text(STUDIORACK_LOCK.read_text())
    config_dir = root.home / ".local/share/open-audio-stack"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(
        json.dumps(
            {
                "appDir": str(tmp_path / "app"),
                "appsDir": str(tmp_path / "apps"),
                "artifactLockPath": str(root.lock_path),
                "pluginsDir": str(root.plugins_dir),
                "presetsDir": str(tmp_path / "presets"),
                "projectsDir": str(tmp_path / "projects"),
                "registries": [
                    {
                        "name": "Open Audio Registry",
                        "url": "https://open-audio-stack.github.io/open-audio-stack-registry",
                    }
                ],
            }
        )
    )
    return root


def _install_locked_dexed(root: StudiorackRoot) -> subprocess.CompletedProcess[str]:
    """Install locked Dexed and require a nonempty plugin binary.

    :param root: Isolated configured CLI root.
    :returns: Successful CLI process with captured text output.
    """
    installed = _run_studiorack(
        root.home,
        ("plugins", "install", DEXED_REFERENCE, "--json"),
    )
    assert installed.returncode == 0, installed.stderr
    assert (root.plugins_dir / DEXED_PLUGIN).stat().st_size > 0
    return installed


def _replace_dexed_sha(root: StudiorackRoot, sha256: str) -> None:
    """Replace only the Dexed digest in an isolated lock.

    :param root: Isolated configured CLI root.
    :param sha256: Replacement lowercase SHA-256 digest.
    """
    lock = json.loads(root.lock_path.read_text())
    lock[DEXED_REFERENCE]["artifacts"][0]["sha256"] = sha256
    root.lock_path.write_text(json.dumps(lock))


def test_studiorack_cli_config_lock_installs_plugin(
    studiorack_root: StudiorackRoot,
) -> None:
    """The shipped CLI installs Dexed from the configured lock.

    :param studiorack_root: Isolated configured CLI root.
    """
    installed = _install_locked_dexed(studiorack_root)

    assert '"installed": true' in installed.stdout


def test_studiorack_cli_config_lock_rejects_sha_drift(
    studiorack_root: StudiorackRoot,
) -> None:
    """The shipped CLI rejects SHA drift after a locked install.

    :param studiorack_root: Isolated configured CLI root.
    """
    _install_locked_dexed(studiorack_root)
    _replace_dexed_sha(studiorack_root, "0" * 64)

    rejected = _run_studiorack(
        studiorack_root.home,
        ("plugins", "install", DEXED_REFERENCE, "--json"),
    )

    assert rejected.returncode != 0
    output = rejected.stdout + rejected.stderr
    assert f"artifact lock mismatch for {DEXED_REFERENCE}" in output
