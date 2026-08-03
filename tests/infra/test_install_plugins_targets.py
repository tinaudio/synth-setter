"""Studiorack is the source of truth for local and image VST3 provisioning."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from synth_setter.plugin_integrity import package_install_lock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CI_CONDA_WORKFLOW = PROJECT_ROOT / ".github/workflows/test-conda.yml"
CI_TEST_WORKFLOW = PROJECT_ROOT / ".github/workflows/test.yml"
MAKEFILE = PROJECT_ROOT / "Makefile"
DOCKERFILE = PROJECT_ROOT / "docker/ubuntu22_04/Dockerfile"
ARTIFACT_LOCK = PROJECT_ROOT / "studiorack.lock.json"
CARDINAL_ARTIFACT_LOCK = PROJECT_ROOT / "studiorack-cardinal.lock.json"
CARDINAL_MANIFEST = PROJECT_ROOT / "studiorack-cardinal.json"
MANIFEST = PROJECT_ROOT / "studiorack.json"
PACKAGE_JSON = PROJECT_ROOT / "package.json"
PACKAGE_LOCK = PROJECT_ROOT / "package-lock.json"
SETUP_SURGE_ACTION = PROJECT_ROOT / ".github/actions/setup-surge-xt/action.yml"
TART_TEMPLATE = PROJECT_ROOT / "tart/macos.pkr.hcl"

pytestmark = pytest.mark.infra

_EXPECTED_PLUGINS = {
    "asb2m10/dexed": ("0.9.8", "Dexed.vst3"),
    "baconpaul/six-sines": ("1.1.0", "Six Sines.vst3"),
    "kayrockscreenprinting/ultramaster-kr106": ("2.5.13", "Ultramaster KR-106.vst3"),
    "surge-synthesizer/ob-xf": ("1.0.3", "OB-Xf.vst3"),
    "surge-synthesizer/surge": ("1.3.4", "Surge XT.vst3"),
}


def _write_executable(path: Path, body: str) -> None:
    """Write one executable test command.

    :param path: Command path under the isolated fake ``PATH``.
    :param body: Complete command source, including its shebang.
    """
    path.write_text(body)
    path.chmod(0o755)


def _run_setup_surge_script(script: str, fake_bin: Path, managed_root: Path) -> None:
    """Execute one setup action script against isolated command fakes.

    :param script: Composite action Bash body.
    :param fake_bin: Directory containing the fake external commands.
    :param managed_root: Temporary replacement for the system managed root.
    """
    subprocess.run(  # noqa: S603 -- checked-in action script with an isolated command path
        ["/bin/bash", "-c", script],
        cwd=managed_root.parent,
        env={
            "FAKE_MANAGED_ROOT": str(managed_root),
            "PATH": f"{fake_bin}:{os.defpath}",
        },
        check=True,
    )


def _dockerfile_stage_text(stage_name: str) -> str:
    """Return Dockerfile text from ``stage_name`` until the next stage.

    :param stage_name: Docker stage alias.
    :returns: Selected stage text.
    """
    text = DOCKERFILE.read_text()
    match = re.search(rf"^FROM .+ AS {re.escape(stage_name)}\n", text, re.MULTILINE)
    assert match, f"Dockerfile does not define stage {stage_name}"
    next_stage = re.search(r"^FROM ", text[match.end() :], re.MULTILINE)
    end = match.end() + next_stage.start() if next_stage else len(text)
    return text[match.start() : end]


def test_studiorack_manifest_pins_runtime_plugin_set() -> None:
    """The project manifest is the single source for shipped plugin versions."""
    payload = json.loads(MANIFEST.read_text())

    assert payload["type"] == "project"
    assert {
        package: (version, payload["vst3Bundles"][package])
        for package, version in payload["plugins"].items()
    } == _EXPECTED_PLUGINS
    assert payload["vst3Versions"] == {
        **payload["plugins"],
        "asb2m10/dexed": "1.0.0",
    }
    assert payload["vst3PluginNames"] == {"baconpaul/six-sines": "Six Sines"}


def test_cardinal_manifest_pins_optional_plugin() -> None:
    """Cardinal stays installable without joining the runtime image plugin set."""
    payload = json.loads(CARDINAL_MANIFEST.read_text())

    assert payload["plugins"] == {"distrho/cardinal": "2026.2.0"}
    assert payload["vst3Bundles"] == {"distrho/cardinal": "CardinalSynth.vst3"}
    assert payload["vst3Versions"] == {"distrho/cardinal": "0.26.2"}


@pytest.mark.parametrize(
    ("manifest_path", "artifact_lock_path", "expected_hosts"),
    [
        (
            MANIFEST,
            ARTIFACT_LOCK,
            {("linux", "x64"), ("mac", "arm64"), ("mac", "x64")},
        ),
        (
            CARDINAL_MANIFEST,
            CARDINAL_ARTIFACT_LOCK,
            {
                ("linux", "arm64"),
                ("linux", "x64"),
                ("mac", "arm64"),
                ("mac", "x64"),
            },
        ),
    ],
    ids=("runtime", "cardinal"),
)
def test_artifact_lock_exactly_covers_manifest_pins(
    manifest_path: Path,
    artifact_lock_path: Path,
    expected_hosts: set[tuple[str, str]],
) -> None:
    """Every manifest has an exact lock covering its supported POSIX hosts.

    :param manifest_path: Manifest whose exact package references must be locked.
    :param artifact_lock_path: Same-stem repository artifact lock.
    :param expected_hosts: Host identities supported by the manifest's install flow.
    """
    manifest = json.loads(manifest_path.read_text())
    artifact_lock = json.loads(artifact_lock_path.read_text())

    assert set(artifact_lock) == {
        f"{package}@{version}" for package, version in manifest["plugins"].items()
    }
    selected_hosts = {
        (system, architecture)
        for package in artifact_lock.values()
        for artifact in package["artifacts"]
        for system in artifact["systems"]
        for architecture in artifact["architectures"]
    }
    assert selected_hosts == expected_hosts


def test_ci_test_paths_include_every_studiorack_manifest_and_lock() -> None:
    """Push and pull-request triggers cover every manifest artifact identity."""
    workflow = yaml.safe_load(CI_TEST_WORKFLOW.read_text())
    # PyYAML 1.1 resolves GitHub's unquoted ``on`` key to ``True``.
    triggers = workflow[True]
    expected = {
        "studiorack-cardinal.json",
        "studiorack-cardinal.lock.json",
        "studiorack.json",
        "studiorack.lock.json",
    }

    assert expected <= set(triggers["push"]["paths"])
    assert expected <= set(triggers["pull_request"]["paths"])


def test_ci_executes_installed_patched_core_artifact_lock_test() -> None:
    """CI installs the pinned npm graph and executes its real-core test."""
    scripts = json.loads(PACKAGE_JSON.read_text())["scripts"]
    workflow = CI_TEST_WORKFLOW.read_text()

    assert scripts["test"] == "node --test scripts/studiorack/test-artifact-lock.mjs"
    assert "npm ci" in workflow
    assert "npm test" in workflow


def test_conda_ci_installs_patched_studiorack_before_pytest() -> None:
    """Conda CI provisions the Node integration dependency before collection."""
    workflow = CI_CONDA_WORKFLOW.read_text()
    parsed = yaml.safe_load(workflow)

    assert "actions/setup-node@" in workflow
    assert workflow.index("npm ci") < workflow.index("pytest -n auto")
    for event in ("push", "pull_request"):
        assert {"package.json", "package-lock.json"} <= set(parsed[True][event]["paths"])


def test_package_lock_pins_studiorack_cli_and_core() -> None:
    """The npm lock fixes both the CLI and its behavior-defining core version."""
    lock = json.loads(PACKAGE_LOCK.read_text())

    assert lock["packages"]["node_modules/@studiorack/cli"]["version"] == "3.0.6"
    assert lock["packages"]["node_modules/@open-audio-stack/core"]["version"] == "0.1.55"


def test_make_plugin_targets_delegate_to_studiorack_cli() -> None:
    """Every public Make target delegates package installation to Studiorack."""
    makefile = MAKEFILE.read_text()

    assert "install-studiorack:" in makefile
    assert "npm ci" in makefile
    for package in _EXPECTED_PLUGINS:
        assert f"install --plugin {package}" in makefile
    plugin_section = makefile[makefile.index("STUDIORACK :=") : makefile.index("link-thoughts:")]
    assert not re.search(r"\b(curl|wget|git clone|tar -|unzip)\b", plugin_section)


def test_docker_plugin_stage_uses_locked_studiorack_cli() -> None:
    """The image installs plugins through the same locked CLI as local hosts."""
    stage = _dockerfile_stage_text("builder-install-studiorack-plugins")

    assert "COPY --from=synth-setter-src /home/build/synth-setter/package.json" in stage
    assert "COPY --from=synth-setter-src /home/build/synth-setter/package-lock.json" in stage
    assert "npm ci" in stage
    assert "python -m synth_setter.cli.plugins" in stage
    assert "src/synth_setter/plugin_integrity.py" in stage
    assert "src/synth_setter/plugin_runtime.py" in stage
    assert "update-alternatives --install /usr/bin/gcc" in DOCKERFILE.read_text()
    assert "studiorack.json" in stage
    assert "studiorack-cardinal.lock.json" in stage
    assert "studiorack.lock.json" in stage


def test_docker_plugin_stage_provisions_cardinal_at_configured_path() -> None:
    """The image installs and links Cardinal into the checkout plugin namespace."""
    stage = _dockerfile_stage_text("builder-install-studiorack-plugins")

    assert "studiorack-cardinal.json" in stage
    assert "--plugin distrho/cardinal" in stage
    assert '"CardinalSynth|"' in stage


def test_docker_alias_restore_runs_from_mounted_source() -> None:
    """Snapshot images restore aliases without requiring the new console script."""
    helper = (PROJECT_ROOT / "docker/ubuntu22_04/ensure_plugin_symlinks.sh").read_text()

    assert '"PYTHONPATH=${repo_root}/src" python -m synth_setter.cli.plugins' in helper
    assert "adopt \\\n    --plugin surge-synthesizer/surge" in helper


def test_docker_fetched_plugins_have_no_manual_download_stage() -> None:
    """Archive synths no longer have parallel Docker download recipes."""
    dockerfile = DOCKERFILE.read_text()

    assert "AS vst3-synths-fetch" not in dockerfile
    assert "DEXED_SHA256" not in dockerfile
    assert "OBXF_SHA256" not in dockerfile
    assert "SIX_SINES_SHA256" not in dockerfile


def test_macos_provisioners_install_surge_through_studiorack() -> None:
    """CI and Tart use the manifest instead of Homebrew's rolling cask."""
    action = SETUP_SURGE_ACTION.read_text()
    tart = TART_TEMPLATE.read_text()

    assert "npm ci" in action
    assert '"CI="' in action
    assert "synth-setter-plugins" in action
    assert "brew install --cask surge-xt" not in action
    assert "npm ci" in tart
    assert "synth-setter-plugins" in tart
    assert "brew install --cask surge-xt" not in tart


def test_macos_ci_plugin_storage_returns_to_runner_after_elevated_install(
    tmp_path: Path,
) -> None:
    """The unprivileged smoke test can lock the installed package.

    :param tmp_path: Scratch roots and executable command fakes.
    """
    action = yaml.safe_load(SETUP_SURGE_ACTION.read_text())
    run_script = action["runs"]["steps"][0]["run"]
    pre_fix_script = run_script.rstrip().rsplit("\n", maxsplit=1)[0]

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "npm", "#!/bin/bash\nexit 0\n")
    _write_executable(
        fake_bin / "uv",
        """#!/bin/bash
set -euo pipefail
managed_root="${STUDIORACK_PLUGINS_DIR:?}"
lock="${managed_root}/.synth-setter-install-locks/surge-synthesizer/surge/1.3.4.lock"
snapshot="${managed_root}/.synth-setter-runtime-snapshots/installed.snapshot"
mkdir -p "$(dirname "${lock}")" "$(dirname "${snapshot}")"
: > "${lock}"
: > "${snapshot}"
chmod -R a-w "${managed_root}"
""",
    )
    _write_executable(
        fake_bin / "sudo",
        """#!/bin/bash
set -euo pipefail
system_root="/Library/Application Support/synth-setter/studiorack"
mapped_root="${FAKE_MANAGED_ROOT:?}"
if [[ "${1:-}" == "-E" && "${2:-}" == "env" ]]; then
  shift 2
  translated=()
  for argument in "$@"; do
    if [[ "${argument}" == "STUDIORACK_PLUGINS_DIR=${system_root}" ]]; then
      argument="STUDIORACK_PLUGINS_DIR=${mapped_root}"
    fi
    translated+=("${argument}")
  done
  exec env "${translated[@]}"
fi
if [[ "${1:-}" == "chown" && "${2:-}" == "-R" && "${4:-}" == "${system_root}" ]]; then
  chmod -R u+rwX "${mapped_root}"
  exit 0
fi
exit 64
""",
    )

    pre_fix_root = tmp_path / "pre-fix-managed"
    _run_setup_surge_script(pre_fix_script, fake_bin, pre_fix_root)
    with pytest.raises(PermissionError):
        with package_install_lock("surge-synthesizer/surge", "1.3.4", pre_fix_root.resolve()):
            pass

    current_root = tmp_path / "current-managed"
    _run_setup_surge_script(run_script, fake_bin, current_root)
    with package_install_lock("surge-synthesizer/surge", "1.3.4", current_root.resolve()):
        pass


def test_docker_keeps_source_fallback_only_for_incompatible_registry_artifacts() -> None:
    """Source builds remain documented compatibility fallbacks, not package pins."""
    dockerfile = DOCKERFILE.read_text()

    assert "AS builder-install-surge-from-source" in dockerfile
    assert "AS builder-build-ultramaster-kr106" in dockerfile
    assert "open-audio-stack/open-audio-stack-core/issues/82" in dockerfile
