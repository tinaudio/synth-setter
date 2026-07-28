"""Studiorack is the source of truth for local and image VST3 provisioning."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = PROJECT_ROOT / "Makefile"
DOCKERFILE = PROJECT_ROOT / "docker/ubuntu22_04/Dockerfile"
MANIFEST = PROJECT_ROOT / "studiorack.json"
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
    assert "update-alternatives --install /usr/bin/gcc" in DOCKERFILE.read_text()
    assert "studiorack.json" in stage


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


def test_docker_keeps_source_fallback_only_for_incompatible_registry_artifacts() -> None:
    """Source builds remain documented compatibility fallbacks, not package pins."""
    dockerfile = DOCKERFILE.read_text()

    assert "AS builder-install-surge-from-source" in dockerfile
    assert "AS builder-build-ultramaster-kr106" in dockerfile
    assert "open-audio-stack/open-audio-stack-core/issues/82" in dockerfile
