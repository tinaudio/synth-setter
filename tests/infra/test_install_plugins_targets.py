"""Studiorack is the source of truth for local and image VST3 provisioning."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from synth_setter.plugin_manager import PluginManifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CI_TEST_WORKFLOW = PROJECT_ROOT / ".github/workflows/test.yml"
MPS_TEST_WORKFLOW = PROJECT_ROOT / ".github/workflows/test-mps.yml"
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


@dataclass(frozen=True)
class _Kr106InstallFakes:
    """Group external command fakes for the KR-106 Make targets.

    .. attribute :: manager

        Fake plugin-manager executable.

    .. attribute :: tool_log

        Event log shared by the command fakes.
    """

    manager: Path
    tool_log: Path


def _write_fake_uname(fake_bin: Path, os_name: str, architecture: str) -> None:
    """Write a deterministic uname command.

    :param fake_bin: Directory receiving the executable.
    :param os_name: Value emitted for the operating system.
    :param architecture: Value emitted for ``uname -m``.
    """
    _write_executable(
        fake_bin / "uname",
        f"""#!/bin/bash
set -eu
if [[ "${{1:-}}" == "-m" ]]; then
  printf '{architecture}\\n'
else
  printf '{os_name}\\n'
fi
""",
    )


def _write_kr106_install_fakes(checkout: Path) -> _Kr106InstallFakes:
    """Provide offline git, CMake, npm, and plugin-manager boundaries.

    :param checkout: Isolated checkout receiving the fake executables.
    :returns: Plugin-manager executable and shared event log.
    """
    fake_bin = checkout / "bin"
    fake_bin.mkdir()
    tool_log = checkout / "tool.log"
    _write_executable(
        fake_bin / "git",
        """#!/bin/bash
set -eu
workdir="$PWD"
if [[ "$1" == "-C" ]]; then
  workdir="$2"
  shift 2
fi
printf 'git -C %s %s\n' "$workdir" "$*" >> "$TOOL_LOG"
if [[ "$1 ${2:-}" == "rev-parse --git-dir" ]]; then
  [[ -d "$workdir/.git" ]]
  exit
fi
if [[ "$1 ${2:-}" == "remote get-url" || "$1 ${2:-}" == "remote set-url" ]]; then
  [[ -e "$workdir/.git/origin" ]]
  exit
fi
if [[ "$1" == "init" ]]; then
  mkdir -p "$workdir/.git"
elif [[ "$1 ${2:-}" == "remote add" ]]; then
  touch "$workdir/.git/origin"
fi
""",
    )
    _write_executable(
        fake_bin / "cmake",
        """#!/bin/bash
set -eu
printf 'cmake %s\n' "$*" >> "$TOOL_LOG"
if [[ "$1" == "--build" ]]; then
  mkdir -p "$2/KR106_artefacts/Release/VST3/Ultramaster KR-106.vst3/Contents"
fi
""",
    )
    _write_executable(fake_bin / "npm", "#!/bin/bash\nset -eu\n")
    _write_executable(
        fake_bin / "flock",
        "#!/bin/bash\nset -eu\nprintf 'flock %s\\n' \"$*\" >> \"$TOOL_LOG\"\nprintf 'locked\\n' >&9\n",
    )
    _write_fake_uname(fake_bin, "Linux", "x86_64")
    manager = fake_bin / "synth-setter-plugins"
    _write_executable(
        manager,
        """#!/bin/bash
set -eu
printf 'plugins %s\n' "$*" >> "$TOOL_LOG"
command="$1"
shift
case "$command" in
  install)
    [[ "$#" -gt 0 ]]
    ;;
  adopt)
    while [[ "$#" -gt 0 ]]; do
      if [[ "$1" == "--bundle-path" ]]; then
        source_bundle="$2"
        break
      fi
      shift
    done
    [[ -d "$source_bundle" ]]
    managed="$HOME/managed-kr106.vst3"
    if [[ -L "$managed" && "$(readlink "$managed")" == "$source_bundle" ]]; then
      exit 0
    fi
    [[ ! -e "$managed" && ! -L "$managed" ]]
    ln -s "$source_bundle" "$managed"
    ;;
  link)
    mkdir -p plugins
    alias="plugins/Ultramaster KR-106.vst3"
    managed="$HOME/managed-kr106.vst3"
    if [[ -L "$alias" && "$(readlink "$alias")" == "$managed" ]]; then
      exit 0
    fi
    [[ ! -e "$alias" && ! -L "$alias" ]]
    ln -s "$managed" "$alias"
    ;;
  *)
    exit 64
    ;;
esac
""",
    )
    return _Kr106InstallFakes(manager=manager, tool_log=tool_log)


def _run_make_target(
    checkout: Path,
    target: str,
    fakes: _Kr106InstallFakes,
) -> subprocess.CompletedProcess[str]:
    """Run one plugin target against isolated external-command fakes.

    :param checkout: Isolated checkout containing the Makefile.
    :param target: Public Make target to invoke.
    :param fakes: Fake plugin-manager executable and shared event log.
    :returns: Completed Make invocation with captured output.
    """
    env = {
        **os.environ,
        "HOME": str(checkout / "home"),
        "PATH": f"{fakes.manager.parent}:{os.defpath}",
        "TOOL_LOG": str(fakes.tool_log),
    }
    return subprocess.run(  # noqa: S603 -- fixed make target in an isolated checkout
        [
            shutil.which("make", path=os.defpath) or "make",
            target,
            f"STUDIORACK={fakes.manager}",
        ],
        cwd=checkout,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _makefile_variable(name: str) -> str:
    match = re.search(rf"^{name} := (.+)$", MAKEFILE.read_text(), re.MULTILINE)
    assert match, f"Makefile does not define {name}"
    return match.group(1)


def _dockerfile_argument(name: str) -> str:
    match = re.search(rf"^ARG {name}=(.+)$", DOCKERFILE.read_text(), re.MULTILINE)
    assert match, f"Dockerfile does not define ARG {name}"
    return match.group(1)


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
        "baconpaul/six-sines": "1.1.0.43d10b2",
    }
    assert payload["vst3PluginNames"] == {"baconpaul/six-sines": "Six Sines"}


def test_six_sines_manifest_pins_source_qualified_runtime_version() -> None:
    """Six Sines validates the exact runtime identity of its pinned release."""
    plugin = PluginManifest.load(MANIFEST).resolve("baconpaul/six-sines")

    assert plugin.version == "1.1.0"
    assert plugin.renderer_version == "1.1.0.43d10b2"


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


def test_package_lock_pins_studiorack_cli_and_core() -> None:
    """The npm lock fixes both the CLI and its behavior-defining core version."""
    lock = json.loads(PACKAGE_LOCK.read_text())

    assert lock["packages"]["node_modules/@studiorack/cli"]["version"] == "3.0.6"
    assert lock["packages"]["node_modules/@open-audio-stack/core"]["version"] == "0.1.55"


def test_make_registry_plugin_targets_delegate_to_studiorack_cli() -> None:
    """Registry-backed public Make targets delegate installation to Studiorack."""
    makefile = MAKEFILE.read_text()

    assert "install-studiorack:" in makefile
    assert "npm ci" in makefile
    registry_packages = set(_EXPECTED_PLUGINS) - {"kayrockscreenprinting/ultramaster-kr106"}
    for package in registry_packages:
        assert f"install --plugin {package}" in makefile


@pytest.mark.parametrize(
    "name",
    ["ULTRAMASTER_KR106_GIT_REF", "ULTRAMASTER_KR106_VERSION"],
)
def test_makefile_kr106_source_pin_matches_dockerfile(name: str) -> None:
    """Local and image source builds share each KR-106 pin.

    :param name: Build identity variable present in both recipes.
    """
    assert _makefile_variable(name) == _dockerfile_argument(name)


def test_makefile_kr106_source_identity_is_immutable_and_manifest_pinned() -> None:
    """The source fallback names the manifest version and a complete Git SHA."""
    manifest_version = (
        PluginManifest.load(MANIFEST).resolve("kayrockscreenprinting/ultramaster-kr106").version
    )

    assert _makefile_variable("ULTRAMASTER_KR106_VERSION") == f"v{manifest_version}"
    assert re.fullmatch(r"[0-9a-f]{40}", _makefile_variable("ULTRAMASTER_KR106_GIT_REF"))


def test_install_ultramaster_kr106_builds_adopts_and_links_source(tmp_path: Path) -> None:
    """One Make command provisions a checkout alias from the pinned source build.

    :param tmp_path: Isolated checkout and command-fake root.
    """
    shutil.copy(MAKEFILE, tmp_path / "Makefile")
    fakes = _write_kr106_install_fakes(tmp_path)

    result = _run_make_target(tmp_path, "install-ultramaster-kr106", fakes)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "plugins" / "Ultramaster KR-106.vst3").is_dir()
    events = fakes.tool_log.read_text()
    expected_ref = _makefile_variable("ULTRAMASTER_KR106_GIT_REF")
    source_root = (
        tmp_path
        / "home"
        / ".cache"
        / "synth-setter"
        / f"ultramaster-kr106-{_makefile_variable('ULTRAMASTER_KR106_VERSION')}"
    )
    assert "flock 9" in events
    assert (source_root / ".install.lock").read_text() == "locked\n"
    assert f"fetch --depth 1 origin {expected_ref}" in events
    assert "--target KR106_VST3" in events
    assert "plugins adopt --plugin kayrockscreenprinting/ultramaster-kr106" in events
    assert "plugins link --plugin kayrockscreenprinting/ultramaster-kr106" in events
    assert "plugins install --plugin kayrockscreenprinting/ultramaster-kr106" not in events


def test_install_ultramaster_kr106_on_macos_delegates_to_registry(tmp_path: Path) -> None:
    """The supported macOS path retains the locked registry installation.

    :param tmp_path: Isolated checkout and command-fake root.
    """
    shutil.copy(MAKEFILE, tmp_path / "Makefile")
    fakes = _write_kr106_install_fakes(tmp_path)
    _write_fake_uname(fakes.manager.parent, "Darwin", "arm64")

    result = _run_make_target(tmp_path, "install-ultramaster-kr106", fakes)

    assert result.returncode == 0, result.stderr
    events = fakes.tool_log.read_text()
    assert "plugins install --plugin kayrockscreenprinting/ultramaster-kr106" in events
    assert "git -C" not in events
    assert "cmake " not in events


def test_install_ultramaster_kr106_on_unsupported_host_fails(tmp_path: Path) -> None:
    """Unsupported hosts fail before source or package installation.

    :param tmp_path: Isolated checkout and command-fake root.
    """
    shutil.copy(MAKEFILE, tmp_path / "Makefile")
    fakes = _write_kr106_install_fakes(tmp_path)
    _write_fake_uname(fakes.manager.parent, "Linux", "aarch64")

    result = _run_make_target(tmp_path, "install-ultramaster-kr106", fakes)

    assert result.returncode != 0
    assert "supports macOS or Linux x86_64 (host: Linux/aarch64)" in result.stderr
    assert not fakes.tool_log.exists()


def test_install_ultramaster_kr106_partial_cache_reinitializes_checkout(
    tmp_path: Path,
) -> None:
    """A cache interrupted before origin creation is rebuilt automatically.

    :param tmp_path: Isolated checkout and command-fake root.
    """
    shutil.copy(MAKEFILE, tmp_path / "Makefile")
    fakes = _write_kr106_install_fakes(tmp_path)
    version = _makefile_variable("ULTRAMASTER_KR106_VERSION")
    source = tmp_path / "home" / ".cache" / "synth-setter" / f"ultramaster-kr106-{version}" / "src"
    (source / ".git").mkdir(parents=True)

    result = _run_make_target(tmp_path, "install-ultramaster-kr106", fakes)

    assert result.returncode == 0, result.stderr
    assert f"git -C {source} init" in fakes.tool_log.read_text()
    assert (tmp_path / "plugins" / "Ultramaster KR-106.vst3").is_dir()


def test_install_ultramaster_kr106_existing_source_install_succeeds(tmp_path: Path) -> None:
    """Repeated source installs preserve one usable checkout alias.

    :param tmp_path: Isolated checkout and command-fake root.
    """
    shutil.copy(MAKEFILE, tmp_path / "Makefile")
    fakes = _write_kr106_install_fakes(tmp_path)

    first = _run_make_target(tmp_path, "install-ultramaster-kr106", fakes)
    second = _run_make_target(tmp_path, "install-ultramaster-kr106", fakes)
    third = _run_make_target(tmp_path, "install-ultramaster-kr106", fakes)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert third.returncode == 0, third.stderr
    alias = tmp_path / "plugins" / "Ultramaster KR-106.vst3"
    assert alias.is_dir()
    assert not (alias / "Ultramaster KR-106.vst3").exists()
    assert not (alias / "managed-kr106.vst3").exists()


def test_install_plugins_routes_kr106_through_source_fallback(tmp_path: Path) -> None:
    """Aggregate installation avoids the KR-106 registry artifact.

    :param tmp_path: Isolated checkout and command-fake root.
    """
    shutil.copy(MAKEFILE, tmp_path / "Makefile")
    fakes = _write_kr106_install_fakes(tmp_path)

    result = _run_make_target(tmp_path, "install-plugins", fakes)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "plugins" / "Ultramaster KR-106.vst3").is_dir()
    install_events = {
        event
        for event in fakes.tool_log.read_text().splitlines()
        if event.startswith("plugins install")
    }
    assert install_events == {
        "plugins install --plugin asb2m10/dexed",
        "plugins install --plugin baconpaul/six-sines",
        "plugins install --plugin surge-synthesizer/ob-xf",
        "plugins install --plugin surge-synthesizer/surge",
    }


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


def test_docker_dev_base_exposes_pinned_studiorack_graph_to_pytest() -> None:
    """The in-image suite receives the same patched graph used for plugin installs."""
    stage = _dockerfile_stage_text("dev-base")
    graph_link = "ln -s /artifacts/studiorack/node_modules node_modules"

    assert graph_link in stage
    assert stage.index(graph_link) < stage.index('pytest -k "not slow"')


def test_docker_plugin_stage_provisions_cardinal_at_configured_path() -> None:
    """The image installs and links Cardinal through its required headless host."""
    stage = _dockerfile_stage_text("builder-install-studiorack-plugins")
    headless_wrapper = "/artifacts/run-linux-vst-headless.sh"
    cardinal_install = "--plugin distrho/cardinal"

    assert "studiorack-cardinal.json" in stage
    assert headless_wrapper in stage
    assert stage.index(headless_wrapper) < stage.index(cardinal_install)
    normalized_stage = " ".join(stage.replace("\\", "").split())
    assert f"{headless_wrapper} python -m synth_setter.cli.plugins" in normalized_stage
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


def test_mps_workflow_runs_for_surge_setup_changes() -> None:
    """The real macOS lane validates changes to its plugin setup boundary."""
    workflow = yaml.safe_load(MPS_TEST_WORKFLOW.read_text())

    # Both events: a push-only or PR-only filter leaves one lane blind to setup changes.
    for event in ("push", "pull_request"):
        paths = workflow[True][event]["paths"]
        assert ".github/actions/setup-surge-xt/**" in paths, event
        assert "tests/infra/test_install_plugins_targets.py" in paths, event


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


def _write_install_fakes(fake_bin: Path) -> None:
    """Write the npm and uv fakes standing in for the privileged Studiorack install.

    The uv fake reproduces what the real install leaves behind: a lock and a runtime
    snapshot under the managed root, with every path write-protected.

    :param fake_bin: Directory holding the fake commands.
    """
    _write_executable(fake_bin / "npm", "#!/bin/bash\nset -euo pipefail\nexit 0\n")
    _write_executable(
        fake_bin / "uv",
        """#!/bin/bash
set -euo pipefail
readonly managed_root="${STUDIORACK_PLUGINS_DIR:?}"
lock="${managed_root}/.synth-setter-install-locks/surge-synthesizer/surge/1.3.4.lock"
snapshot="${managed_root}/.synth-setter-runtime-snapshots/installed.snapshot"
mkdir -p "$(dirname "${lock}")" "$(dirname "${snapshot}")"
: > "${lock}"
: > "${snapshot}"
chmod -R a-w "${managed_root}"
""",
    )


def _write_sudo_fake(fake_bin: Path) -> None:
    """Write the sudo fake mapping the system managed root onto a scratch root.

    Pins both the chown target and its owner: chowning to anyone but the invoking user
    leaves the unprivileged smoke step unable to write, which is the bug under test.

    :param fake_bin: Directory holding the fake commands.
    """
    _write_executable(
        fake_bin / "sudo",
        """#!/bin/bash
set -euo pipefail
readonly system_root="/Library/Application Support/synth-setter/studiorack"
readonly mapped_root="${FAKE_MANAGED_ROOT:?}"
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
  if [[ "${3:-}" != "$(id -u):$(id -g)" ]]; then
    echo "sudo chown: refusing owner ${3:-} (expected $(id -u):$(id -g))" >&2
    exit 65
  fi
  chmod -R u+rwX "${mapped_root}"
  exit 0
fi
exit 64
""",
    )


def _write_surge_setup_fakes(tmp_path: Path) -> Path:
    """Write every command the setup script shells out to.

    :param tmp_path: Scratch root holding the fake command directory.
    :returns: Directory to prepend to ``PATH``.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_install_fakes(fake_bin)
    _write_sudo_fake(fake_bin)
    return fake_bin


def test_macos_ci_plugin_storage_returns_to_runner_after_elevated_install(
    tmp_path: Path,
) -> None:
    """Privileged setup restores owner write permission before the smoke test.

    :param tmp_path: Scratch roots and executable command fakes.
    """
    action = yaml.safe_load(SETUP_SURGE_ACTION.read_text())
    run_script = action["runs"]["steps"][0]["run"]
    # Select the ownership command explicitly; keying off the last line would silently
    # stop removing it if any command were appended after it.
    pre_fix_script = "\n".join(
        line for line in run_script.splitlines() if not line.strip().startswith("sudo chown")
    )
    assert pre_fix_script != run_script

    fake_bin = _write_surge_setup_fakes(tmp_path)

    pre_fix_root = tmp_path / "pre-fix-managed"
    _run_setup_surge_script(pre_fix_script, fake_bin, pre_fix_root)
    assert not pre_fix_root.stat().st_mode & stat.S_IWUSR

    current_root = tmp_path / "current-managed"
    _run_setup_surge_script(run_script, fake_bin, current_root)
    assert current_root.stat().st_mode & stat.S_IWUSR


def test_docker_keeps_source_fallback_only_for_incompatible_registry_artifacts() -> None:
    """Source builds remain documented compatibility fallbacks, not package pins."""
    dockerfile = DOCKERFILE.read_text()

    assert "AS builder-install-surge-from-source" in dockerfile
    assert "AS builder-build-ultramaster-kr106" in dockerfile
    assert "open-audio-stack/open-audio-stack-core/issues/82" in dockerfile
