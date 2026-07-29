"""Behavior tests for manifest-backed Studiorack package management."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from synth_setter.cli.plugins import main
from synth_setter.plugin_manager import (
    PluginManifest,
    adopt_plugin_bundle,
    default_plugins_dir,
    install_plugins,
    link_plugin,
    resolve_plugin_bundle,
)


def _manifest(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "name": "test-project",
                "type": "project",
                "plugins": {"example/synth": "1.2.3"},
                "vst3Bundles": {"example/synth": "Example Synth.vst3"},
            }
        )
    )
    return path


def test_manifest_load_valid_project_returns_pinned_plugin(tmp_path: Path) -> None:
    """A valid project resolves exact package metadata.

    :param tmp_path: Scratch root for the test manifest and plugin tree.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))

    plugin = manifest.resolve("example/synth")

    assert plugin.package == "example/synth"
    assert plugin.version == "1.2.3"
    assert plugin.bundle == "Example Synth.vst3"
    assert plugin.reference == "example/synth@1.2.3"


def test_manifest_resolve_unknown_package_raises(tmp_path: Path) -> None:
    """Unknown package selection fails at the manifest boundary.

    :param tmp_path: Scratch root for the test manifest.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))

    with pytest.raises(KeyError, match="missing/synth"):
        manifest.resolve("missing/synth")


def test_default_plugins_dir_expands_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The managed storage override accepts a user-relative path.

    :param tmp_path: Scratch home used to expand the override.
    :param monkeypatch: Supplies isolated HOME and storage settings.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("STUDIORACK_PLUGINS_DIR", "~/managed")

    assert default_plugins_dir() == tmp_path / "managed"


def test_install_plugins_missing_executable_raises(tmp_path: Path) -> None:
    """Installation reports how to provision the pinned CLI.

    :param tmp_path: Scratch root without a Studiorack executable.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")

    with pytest.raises(FileNotFoundError, match="npm ci"):
        install_plugins(
            (plugin,),
            plugins_dir=tmp_path / "managed",
            studiorack_executable=tmp_path / "missing-studiorack",
        )


def test_manifest_load_unpinned_version_rejected(tmp_path: Path) -> None:
    """Version ranges are rejected at the manifest boundary.

    :param tmp_path: Scratch root for the test manifest.
    """
    path = _manifest(tmp_path / "studiorack.json")
    payload = json.loads(path.read_text())
    payload["plugins"]["example/synth"] = "^1.2.3"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValidationError, match="exact semantic version"):
        PluginManifest.load(path)


def test_manifest_load_bundle_keys_differ_from_plugins_rejected(tmp_path: Path) -> None:
    """Every package must name its expected VST3 bundle.

    :param tmp_path: Scratch root for the test manifest.
    """
    path = _manifest(tmp_path / "studiorack.json")
    payload = json.loads(path.read_text())
    payload["vst3Bundles"] = {"other/synth": "Example Synth.vst3"}
    path.write_text(json.dumps(payload))

    with pytest.raises(ValidationError, match="same package keys"):
        PluginManifest.load(path)


def test_resolve_plugin_bundle_managed_archive_returns_versioned_bundle(tmp_path: Path) -> None:
    """Archive installs resolve from Studiorack's versioned VST3 tree.

    :param tmp_path: Scratch root for the managed plugin tree.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))
    plugin = manifest.resolve("example/synth")
    bundle = tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3"
    bundle.mkdir(parents=True)

    resolved = resolve_plugin_bundle(plugin, tmp_path / "managed")

    assert resolved == bundle


def test_resolve_plugin_bundle_unmanaged_system_bundle_raises(tmp_path: Path) -> None:
    """A same-named system bundle cannot satisfy an exact package pin.

    :param tmp_path: Scratch root for the system plugin tree.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))
    plugin = manifest.resolve("example/synth")
    (tmp_path / "system-vst3/Example Synth.vst3").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="example/synth@1.2.3"):
        resolve_plugin_bundle(plugin, tmp_path / "managed")


def test_resolve_plugin_bundle_missing_raises_actionable_error(tmp_path: Path) -> None:
    """A missing bundle reports the exact Studiorack install command.

    :param tmp_path: Scratch root without an installed bundle.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))

    with pytest.raises(FileNotFoundError, match="studiorack plugins install example/synth@1.2.3"):
        resolve_plugin_bundle(manifest.resolve("example/synth"), tmp_path / "managed")


def test_link_plugin_managed_bundle_creates_stable_checkout_alias(tmp_path: Path) -> None:
    """Managed bundles receive stable checkout-local aliases.

    :param tmp_path: Scratch root for managed storage and checkout aliases.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))
    plugin = manifest.resolve("example/synth")
    bundle = tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3"
    bundle.mkdir(parents=True)

    alias = link_plugin(
        plugin,
        plugins_dir=tmp_path / "managed",
        links_dir=tmp_path / "checkout/plugins",
    )

    assert alias == tmp_path / "checkout/plugins/Example Synth.vst3"
    assert alias.is_symlink()
    assert alias.resolve() == bundle.resolve()


def test_install_plugins_adopts_native_installer_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful native install enters the exact managed namespace.

    :param tmp_path: Scratch root for managed and system plugin trees.
    :param monkeypatch: Supplies the native bundle path to the fake installer.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))
    plugin = manifest.resolve("example/synth")
    system_dir = tmp_path / "system-vst3"
    bundle = system_dir / plugin.bundle
    executable = tmp_path / "studiorack"
    executable.write_text('#!/usr/bin/env bash\nmkdir -p "$STUDIORACK_TEST_BUNDLE"\n')
    executable.chmod(0o755)
    monkeypatch.setenv("STUDIORACK_TEST_BUNDLE", str(bundle))

    install_plugins(
        (plugin,),
        plugins_dir=tmp_path / "managed",
        studiorack_executable=executable,
        system_dirs=(system_dir,),
    )

    resolved = resolve_plugin_bundle(plugin, tmp_path / "managed")
    assert resolved.is_symlink()
    assert resolved.resolve() == bundle.resolve()


def test_link_plugin_existing_stale_symlink_is_replaced(tmp_path: Path) -> None:
    """Alias refresh replaces only stale symlinks, not real bundles.

    :param tmp_path: Scratch root containing managed and stale bundles.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    managed = tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3"
    managed.mkdir(parents=True)
    stale = tmp_path / "stale/Example Synth.vst3"
    stale.mkdir(parents=True)
    alias = tmp_path / "checkout/plugins/Example Synth.vst3"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(stale, target_is_directory=True)

    first = link_plugin(plugin, plugins_dir=tmp_path / "managed", links_dir=alias.parent)
    second = link_plugin(plugin, plugins_dir=tmp_path / "managed", links_dir=alias.parent)

    assert first == second == alias
    assert alias.resolve() == managed.resolve()


def test_adopt_plugin_bundle_missing_source_raises(tmp_path: Path) -> None:
    """Adoption rejects a fallback path that does not exist.

    :param tmp_path: Scratch root without a source bundle.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")

    with pytest.raises(FileNotFoundError, match="fallback bundle is not installed"):
        adopt_plugin_bundle(
            plugin,
            plugins_dir=tmp_path / "managed",
            bundle=tmp_path / "missing.vst3",
        )


def test_adopt_plugin_bundle_repeated_call_is_idempotent(tmp_path: Path) -> None:
    """Repeated adoption preserves the exact managed fallback alias.

    :param tmp_path: Scratch root for source and managed bundles.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    bundle = tmp_path / "source/Example Synth.vst3"
    bundle.mkdir(parents=True)

    first = adopt_plugin_bundle(plugin, plugins_dir=tmp_path / "managed", bundle=bundle)
    second = adopt_plugin_bundle(plugin, plugins_dir=tmp_path / "managed", bundle=bundle)

    assert first == second


def test_adopt_plugin_bundle_conflicting_managed_path_raises(tmp_path: Path) -> None:
    """Adoption refuses to replace a different managed bundle.

    :param tmp_path: Scratch root containing a managed path conflict.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    bundle = tmp_path / "source/Example Synth.vst3"
    bundle.mkdir(parents=True)
    managed = tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3"
    managed.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="refusing to replace"):
        adopt_plugin_bundle(plugin, plugins_dir=tmp_path / "managed", bundle=bundle)


def test_link_plugin_existing_real_bundle_refuses_to_overwrite(tmp_path: Path) -> None:
    """Alias refresh refuses to overwrite an unrelated real bundle.

    :param tmp_path: Scratch root containing the conflicting bundle.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))
    plugin = manifest.resolve("example/synth")
    managed = tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3"
    managed.mkdir(parents=True)
    existing = tmp_path / "checkout/plugins/Example Synth.vst3"
    existing.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="not a symlink"):
        link_plugin(
            plugin,
            plugins_dir=tmp_path / "managed",
            links_dir=existing.parent,
        )


def test_plugins_cli_adopt_records_source_fallback(tmp_path: Path) -> None:
    """The adopt command records a pinned source build for later linking.

    :param tmp_path: Scratch root for the manifest and fallback bundle.
    """
    manifest_path = _manifest(tmp_path / "studiorack.json")
    bundle = tmp_path / "source/Example Synth.vst3"
    bundle.mkdir(parents=True)

    result = CliRunner().invoke(
        main,
        [
            "--manifest",
            str(manifest_path),
            "--plugins-dir",
            str(tmp_path / "managed"),
            "adopt",
            "--plugin",
            "example/synth",
            "--bundle-path",
            str(bundle),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    managed = tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3"
    assert managed.is_symlink()
    assert managed.resolve() == bundle.resolve()


def test_plugins_cli_resolve_prints_managed_bundle(tmp_path: Path) -> None:
    """The resolve command prints the exact managed bundle path.

    :param tmp_path: Scratch root for the manifest and managed bundle.
    """
    manifest_path = _manifest(tmp_path / "studiorack.json")
    bundle = tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3"
    bundle.mkdir(parents=True)

    result = CliRunner().invoke(
        main,
        [
            "--manifest",
            str(manifest_path),
            "--plugins-dir",
            str(tmp_path / "managed"),
            "resolve",
            "example/synth",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip() == str(bundle)


def test_plugins_cli_link_without_selection_links_only_installed_packages(tmp_path: Path) -> None:
    """Bulk linking skips manifest packages absent from the host.

    :param tmp_path: Scratch root for the CLI invocation.
    """
    manifest_path = _manifest(tmp_path / "studiorack.json")
    payload = json.loads(manifest_path.read_text())
    payload["plugins"]["missing/synth"] = "2.0.0"
    payload["vst3Bundles"]["missing/synth"] = "Missing Synth.vst3"
    manifest_path.write_text(json.dumps(payload))
    managed = tmp_path / "managed"
    bundle = managed / "VST3/example/synth/1.2.3/Example Synth.vst3"
    bundle.mkdir(parents=True)

    result = CliRunner().invoke(
        main,
        [
            "--manifest",
            str(manifest_path),
            "--plugins-dir",
            str(managed),
            "--links-dir",
            str(tmp_path / "checkout/plugins"),
            "link",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "checkout/plugins/Example Synth.vst3").is_symlink()
    assert "Missing Synth.vst3 is not installed" in result.output


def test_plugins_cli_install_configures_studiorack_and_links_bundle(tmp_path: Path) -> None:
    """The install entrypoint configures Studiorack and creates its alias.

    :param tmp_path: Scratch root for the CLI boundary and call log.
    """
    manifest_path = _manifest(tmp_path / "studiorack.json")
    managed = tmp_path / "managed"
    bundle = managed / "VST3/example/synth/1.2.3/Example Synth.vst3"
    bundle.mkdir(parents=True)
    calls = tmp_path / "calls.jsonl"
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['STUDIORACK_TEST_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    executable.chmod(0o755)

    result = CliRunner().invoke(
        main,
        [
            "--manifest",
            str(manifest_path),
            "--plugins-dir",
            str(managed),
            "--links-dir",
            str(tmp_path / "checkout/plugins"),
            "--studiorack-executable",
            str(executable),
            "install",
            "--plugin",
            "example/synth",
        ],
        env={"STUDIORACK_TEST_CALLS": str(calls), **os.environ},
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert [json.loads(line) for line in calls.read_text().splitlines()] == [
        ["config", "set", "pluginsDir", str(managed.resolve())],
        ["plugins", "install", "example/synth@1.2.3"],
    ]
    alias = tmp_path / "checkout/plugins/Example Synth.vst3"
    assert alias.is_symlink()
    assert alias.resolve() == bundle.resolve()
