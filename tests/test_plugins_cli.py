"""Manifest and CLI tests for managed Studiorack plugins."""

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from synth_setter.cli.plugins import main
from synth_setter.plugin_manager import (
    ArtifactLock,
    PluginManifest,
    default_plugins_dir,
    resolve_plugin_bundle,
)
from tests.plugin_manager_test_support import (
    PROJECT_ROOT,
    _adopt_bundle,
    _artifact_lock,
    _binary_path,
    _bundle,
    _example_lock,
    _manifest,
    _seal_bundle,
)
from tests.plugin_manager_test_support import (
    _platform_binary_environment as _platform_binary_environment,
)


def test_manifest_load_valid_project_returns_pinned_plugin(tmp_path: Path) -> None:
    """A valid project resolves exact package metadata.

    :param tmp_path: Scratch root for the test manifest and plugin tree.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))

    plugin = manifest.resolve("example/synth")

    assert plugin.package == "example/synth"
    assert plugin.version == "1.2.3"
    assert plugin.renderer_version == "1.2.3"
    assert plugin.bundle == "Example Synth.vst3"
    assert plugin.reference == "example/synth@1.2.3"
    assert manifest.vst3_versions is None


def test_manifest_explicit_renderer_version_differs_from_package_version(tmp_path: Path) -> None:
    """A package pin may identify a VST3 that reports another exact version.

    :param tmp_path: Scratch root for the test manifest.
    """
    path = _manifest(tmp_path / "studiorack.json")
    payload = json.loads(path.read_text())
    payload["plugins"]["example/synth"] = "2026.2.0"
    payload["vst3Versions"] = {"example/synth": "0.26.2"}
    path.write_text(json.dumps(payload))

    plugin = PluginManifest.load(path).resolve("example/synth")

    assert plugin.version == "2026.2.0"
    assert plugin.renderer_version == "0.26.2"
    assert plugin.reference == "example/synth@2026.2.0"


def test_committed_manifests_match_plugin_manifest_contract() -> None:
    """Repository manifests remain consumable by the plugin installer."""
    for filename in ("studiorack-cardinal.json", "studiorack.json"):
        PluginManifest.load(PROJECT_ROOT / filename)


def test_manifest_vst3_prerelease_with_build_metadata_accepted(tmp_path: Path) -> None:
    """Exact runtime versions may carry prerelease and build metadata.

    :param tmp_path: Scratch root for the test manifest.
    """
    path = _manifest(tmp_path / "studiorack.json", renderer_version="4.5.6-rc.1+build.7")

    plugin = PluginManifest.load(path).resolve("example/synth")

    assert plugin.renderer_version == "4.5.6-rc.1+build.7"


@pytest.mark.parametrize("version", ["4.5.6-rc..1", "4.5.6+build..7", "^4.5.6"])
def test_manifest_invalid_vst3_version_rejected(tmp_path: Path, version: str) -> None:
    """Malformed or ranged runtime versions fail manifest validation.

    :param tmp_path: Scratch root for the test manifest.
    :param version: Invalid runtime version.
    """
    path = _manifest(tmp_path / "studiorack.json", renderer_version=version)

    with pytest.raises(ValidationError, match="VST3 version must be an exact semantic version"):
        PluginManifest.load(path)


def test_plugins_cli_archive_renderer_version_mismatch_creates_no_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Archive output with the wrong renderer version never reaches the stable namespace.

    :param tmp_path: Scratch root for managed state, checkout aliases, and installer.
    :param monkeypatch: Supplies the managed output path to the installer process.
    """
    manifest_path = _manifest(
        tmp_path / "studiorack.json",
        package_version="2026.2.0",
        renderer_version="0.26.2",
    )
    plugin = PluginManifest.load(manifest_path).resolve("example/synth")
    _artifact_lock(tmp_path / "studiorack.lock.json", package_version="2026.2.0")
    bundle = tmp_path / "managed/VST3/example/synth/2026.2.0/Example Synth.vst3"
    links_dir = tmp_path / "plugins"
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    bundle = pathlib.Path(os.environ['STUDIORACK_TEST_BUNDLE'])\n"
        "    binary = bundle / os.environ['STUDIORACK_TEST_BINARY']\n"
        "    binary.parent.mkdir(parents=True)\n"
        "    binary.write_bytes(bytes.fromhex(os.environ['STUDIORACK_TEST_BINARY_MAGIC']) + b'archive')\n"
        "    (bundle / 'Contents/moduleinfo.json').write_text(json.dumps({'Version': '9.9.9'}))\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("STUDIORACK_TEST_BUNDLE", str(bundle))

    result = CliRunner().invoke(
        main,
        [
            "--manifest",
            str(manifest_path),
            "--plugins-dir",
            str(tmp_path / "managed"),
            "--links-dir",
            str(links_dir),
            "--studiorack-executable",
            str(executable),
            "install",
        ],
    )

    assert result.exit_code == 1
    assert "expected 0.26.2" in result.output
    assert not (links_dir / plugin.bundle).exists()
    assert not (bundle.parent / ".synth-setter-complete.json").exists()


def test_manifest_renderer_version_keys_differ_from_plugins_rejected(tmp_path: Path) -> None:
    """An explicit renderer-version map must cover the same package set.

    :param tmp_path: Scratch root for the test manifest.
    """
    path = _manifest(tmp_path / "studiorack.json")
    payload = json.loads(path.read_text())
    payload["vst3Versions"] = {"other/synth": "1.2.3"}
    path.write_text(json.dumps(payload))

    with pytest.raises(ValidationError, match="same package keys"):
        PluginManifest.load(path)


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
    bundle = _bundle(tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3")
    _seal_bundle(bundle, plugin)

    resolved = resolve_plugin_bundle(plugin, tmp_path / "managed", artifact_lock=_example_lock())

    assert resolved == bundle


def test_resolve_plugin_bundle_unmanaged_system_bundle_raises(tmp_path: Path) -> None:
    """A same-named system bundle cannot satisfy an exact package pin.

    :param tmp_path: Scratch root for the system plugin tree.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))
    plugin = manifest.resolve("example/synth")
    (tmp_path / "system-vst3/Example Synth.vst3").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="example/synth@1.2.3"):
        resolve_plugin_bundle(plugin, tmp_path / "managed", artifact_lock=_example_lock())


def test_resolve_plugin_bundle_missing_raises_actionable_error(tmp_path: Path) -> None:
    """A missing bundle reports the exact Studiorack install command.

    :param tmp_path: Scratch root without an installed bundle.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))

    with pytest.raises(FileNotFoundError, match="studiorack plugins install example/synth@1.2.3"):
        resolve_plugin_bundle(
            manifest.resolve("example/synth"),
            tmp_path / "managed",
            artifact_lock=_example_lock(),
        )


def test_plugins_cli_artifact_lock_mismatch_renders_click_error(tmp_path: Path) -> None:
    """Deterministic lock rejection is rendered without a Python traceback.

    :param tmp_path: Scratch root for CLI files and installer.
    """
    manifest_path = _manifest(tmp_path / "studiorack.json")
    _artifact_lock(tmp_path / "studiorack.lock.json")
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    print('artifact lock mismatch for example/synth@1.2.3', file=sys.stderr)\n"
        "    raise SystemExit(7)\n"
    )
    executable.chmod(0o755)

    result = CliRunner().invoke(
        main,
        [
            "--manifest",
            str(manifest_path),
            "--plugins-dir",
            str(tmp_path / "managed"),
            "--studiorack-executable",
            str(executable),
            "install",
            "--plugin",
            "example/synth",
        ],
    )

    assert result.exit_code == 1
    assert "returned non-zero exit status 7" in result.output
    assert "Traceback" not in result.output


def test_plugins_cli_failed_native_postvalidation_creates_no_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Install failure after native execution cannot expose a checkout alias.

    :param tmp_path: Scratch root for managed, native, and checkout state.
    :param monkeypatch: Redirects native discovery and installer output.
    """
    manifest_path = _manifest(tmp_path / "studiorack.json")
    _artifact_lock(tmp_path / "studiorack.lock.json")
    system_dir = tmp_path / "system-vst3"
    bundle = _bundle(system_dir / "Example Synth.vst3", payload=b"before")
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    bundle = pathlib.Path(os.environ['STUDIORACK_TEST_BUNDLE'])\n"
        "    binary = bundle / os.environ['STUDIORACK_TEST_BINARY']\n"
        "    binary.write_bytes(bytes.fromhex(os.environ['STUDIORACK_TEST_BINARY_MAGIC']) + b'changed')\n"
        "    (bundle / 'Contents/moduleinfo.json').write_text(json.dumps({'Version': '9.9.9'}))\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("STUDIORACK_TEST_BUNDLE", str(bundle))
    monkeypatch.setattr(
        "synth_setter.plugin_manager.default_system_vst3_dirs", lambda: (system_dir,)
    )

    result = CliRunner().invoke(
        main,
        [
            "--manifest",
            str(manifest_path),
            "--plugins-dir",
            str(tmp_path / "managed"),
            "--links-dir",
            str(tmp_path / "checkout/plugins"),
            "--studiorack-executable",
            str(executable),
            "install",
            "--plugin",
            "example/synth",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code != 0
    assert "expected 1.2.3" in result.output
    assert not (tmp_path / "checkout/plugins/Example Synth.vst3").exists()


def test_plugins_cli_link_value_error_renders_click_error(tmp_path: Path) -> None:
    """Link validation failures render as Click errors without tracebacks.

    :param tmp_path: Scratch root for manifest, lock, and malformed managed state.
    """
    manifest_path = _manifest(tmp_path / "studiorack.json")
    _artifact_lock(tmp_path / "studiorack.lock.json")
    managed = tmp_path / "managed"
    managed.mkdir()
    lock_escape = managed / ".synth-setter-install-locks"
    lock_escape.symlink_to(tmp_path, target_is_directory=True)

    result = CliRunner().invoke(
        main,
        [
            "--manifest",
            str(manifest_path),
            "--plugins-dir",
            str(managed),
            "link",
            "--plugin",
            "example/synth",
        ],
    )

    assert result.exit_code == 1
    assert "Error: refusing managed parent symlink" in result.output
    assert "Traceback" not in result.output


def test_plugins_cli_custom_manifest_missing_sibling_lock_rejected(tmp_path: Path) -> None:
    """A custom manifest defaults to an artifact lock in the same directory.

    :param tmp_path: Scratch root containing only a manifest.
    """
    manifest_path = _manifest(tmp_path / "config/custom-plugins.json")

    result = CliRunner().invoke(
        main,
        ["--manifest", str(manifest_path), "resolve", "example/synth"],
    )

    assert result.exit_code != 0
    assert str(tmp_path / "config/custom-plugins.lock.json") in result.output


def test_plugins_cli_cardinal_manifest_resolves_with_derived_lock(tmp_path: Path) -> None:
    """The real optional manifest resolves through its own sibling lock.

    :param tmp_path: Scratch root for a sealed Cardinal managed bundle.
    """
    bundle = _bundle(
        tmp_path / "managed/VST3/distrho/cardinal/2026.2.0/CardinalSynth.vst3",
        version="0.26.2",
    )
    cardinal_manifest = PluginManifest.load(PROJECT_ROOT / "studiorack-cardinal.json")
    cardinal_lock = ArtifactLock.load(
        PROJECT_ROOT / "studiorack-cardinal.lock.json",
        cardinal_manifest,
    )
    _seal_bundle(bundle, cardinal_manifest.resolve("distrho/cardinal"), cardinal_lock)

    result = CliRunner().invoke(
        main,
        [
            "--manifest",
            str(PROJECT_ROOT / "studiorack-cardinal.json"),
            "--plugins-dir",
            str(tmp_path / "managed"),
            "resolve",
            "distrho/cardinal",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip() == str(bundle)
    plugin = cardinal_manifest.resolve("distrho/cardinal")
    assert plugin.version == "2026.2.0"
    assert plugin.renderer_version == "0.26.2"
    assert list(cardinal_lock.root) == ["distrho/cardinal@2026.2.0"]


def test_plugins_cli_explicit_artifact_lock_overrides_manifest_sibling(tmp_path: Path) -> None:
    """The global lock option accepts a repository-controlled alternate path.

    :param tmp_path: Scratch root for manifest, lock, and managed bundle.
    """
    manifest_path = _manifest(tmp_path / "config/studiorack.json")
    lock_path = _artifact_lock(tmp_path / "locks/plugins.json")
    bundle = _bundle(tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3")
    _seal_bundle(bundle)

    result = CliRunner().invoke(
        main,
        [
            "--manifest",
            str(manifest_path),
            "--artifact-lock",
            str(lock_path),
            "--plugins-dir",
            str(tmp_path / "managed"),
            "resolve",
            "example/synth",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip() == str(bundle)


def test_plugins_cli_explicit_adoption_rejected_after_lock_rotation(
    tmp_path: Path,
) -> None:
    """Explicit source adoption remains bound to its repository package lock.

    :param tmp_path: Scratch root for source, managed state, and repository lock.
    """
    manifest_path = _manifest(tmp_path / "studiorack.json")
    lock_path = _artifact_lock(tmp_path / "studiorack.lock.json")
    plugin = PluginManifest.load(manifest_path).resolve("example/synth")
    source = _bundle(tmp_path / "source/Example Synth.vst3")
    managed = _adopt_bundle(plugin, plugins_dir=tmp_path / "managed", bundle=source)
    seal = json.loads((managed.parent / ".synth-setter-complete.json").read_text())
    assert seal["source_kind"] == "explicit"
    assert seal["locked_package_sha256"] == (
        "ce68cc987663810c48dcd1de66953f8f278a569dc4aef0747a68b18044502c46"
    )

    payload = json.loads(lock_path.read_text())
    payload["example/synth@1.2.3"]["artifacts"][0]["sha256"] = "b" * 64
    lock_path.write_text(json.dumps(payload))
    result = CliRunner().invoke(
        main,
        [
            "--manifest",
            str(manifest_path),
            "--plugins-dir",
            str(tmp_path / "managed"),
            "resolve",
            plugin.package,
        ],
    )

    assert result.exit_code == 1
    assert "failed managed bundle integrity" in result.output


def test_plugins_cli_same_version_lock_rotation_reinstalls_managed_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new artifact identity at the same version replaces stale managed output.

    :param tmp_path: Scratch root for lock, managed output, and installer state.
    :param monkeypatch: Supplies output paths to the deterministic installer.
    """
    manifest_path = _manifest(tmp_path / "studiorack.json")
    lock_path = _artifact_lock(tmp_path / "studiorack.lock.json")
    managed = tmp_path / "managed"
    links = tmp_path / "links"
    bundle = managed / "VST3/example/synth/1.2.3/Example Synth.vst3"
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    bundle = pathlib.Path(os.environ['STUDIORACK_TEST_BUNDLE'])\n"
        "    calls = pathlib.Path(os.environ['STUDIORACK_TEST_CALLS'])\n"
        "    calls.write_text(str(int(calls.read_text()) + 1) if calls.exists() else '1')\n"
        "    if not bundle.exists():\n"
        "        lock = json.loads(pathlib.Path(os.environ['SYNTH_SETTER_STUDIORACK_ARTIFACT_LOCK']).read_text())\n"
        "        url = lock['example/synth@1.2.3']['artifacts'][0]['url']\n"
        "        binary = bundle / os.environ['STUDIORACK_TEST_BINARY']\n"
        "        binary.parent.mkdir(parents=True)\n"
        "        binary.write_bytes(bytes.fromhex(os.environ['STUDIORACK_TEST_BINARY_MAGIC']) + url.encode())\n"
        "        (bundle / 'Contents/moduleinfo.json').write_text(json.dumps({'Version': '1.2.3'}))\n"
    )
    executable.chmod(0o755)
    calls = tmp_path / "calls"
    monkeypatch.setenv("STUDIORACK_TEST_BUNDLE", str(bundle))
    monkeypatch.setenv("STUDIORACK_TEST_CALLS", str(calls))

    first = CliRunner().invoke(
        main,
        [
            "--manifest",
            str(manifest_path),
            "--plugins-dir",
            str(managed),
            "--links-dir",
            str(links),
            "--studiorack-executable",
            str(executable),
            "install",
            "--plugin",
            "example/synth",
        ],
        catch_exceptions=False,
    )
    assert first.exit_code == 0, first.output
    first_binary = _binary_path(bundle).read_bytes()

    lock_payload = json.loads(lock_path.read_text())
    lock_payload["example/synth@1.2.3"]["artifacts"][0].update(
        sha256="b" * 64,
        url="https://example.test/rotated.zip",
    )
    lock_path.write_text(json.dumps(lock_payload))
    second = CliRunner().invoke(
        main,
        [
            "--manifest",
            str(manifest_path),
            "--plugins-dir",
            str(managed),
            "--links-dir",
            str(links),
            "--studiorack-executable",
            str(executable),
            "install",
            "--plugin",
            "example/synth",
        ],
        catch_exceptions=False,
    )

    assert second.exit_code == 0, second.output
    assert calls.read_text() == "2"
    assert _binary_path(bundle).read_bytes() != first_binary


def test_plugins_cli_stale_artifact_seal_rejected_after_lock_rotation(
    tmp_path: Path,
) -> None:
    """Resolve rejects managed registry output sealed under another lock identity.

    :param tmp_path: Scratch root for manifest, lock, and managed bundle.
    """
    manifest_path = _manifest(tmp_path / "studiorack.json")
    lock_path = _artifact_lock(tmp_path / "studiorack.lock.json")
    plugin = PluginManifest.load(manifest_path).resolve("example/synth")
    bundle = _bundle(tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3")
    _seal_bundle(bundle, plugin)
    lock_payload = json.loads(lock_path.read_text())
    lock_payload["example/synth@1.2.3"]["artifacts"][0]["sha256"] = "b" * 64
    lock_path.write_text(json.dumps(lock_payload))

    result = CliRunner().invoke(
        main,
        [
            "--manifest",
            str(manifest_path),
            "--plugins-dir",
            str(tmp_path / "managed"),
            "resolve",
            plugin.package,
        ],
    )

    assert result.exit_code == 1
    assert "failed managed bundle integrity" in result.output


def test_plugins_cli_adopt_records_source_fallback(tmp_path: Path) -> None:
    """The adopt command records a pinned source build for later linking.

    :param tmp_path: Scratch root for the manifest and fallback bundle.
    """
    manifest_path = _manifest(tmp_path / "studiorack.json")
    _artifact_lock(tmp_path / "studiorack.lock.json")
    bundle = _bundle(tmp_path / "source/Example Synth.vst3")

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
    _artifact_lock(tmp_path / "studiorack.lock.json")
    bundle = _bundle(tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3")
    _seal_bundle(bundle)

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
    lock_path = _artifact_lock(tmp_path / "studiorack.lock.json")
    lock_payload = json.loads(lock_path.read_text())
    lock_payload["missing/synth@2.0.0"] = lock_payload["example/synth@1.2.3"]
    lock_path.write_text(json.dumps(lock_payload))
    managed = tmp_path / "managed"
    bundle = _bundle(managed / "VST3/example/synth/1.2.3/Example Synth.vst3")
    _seal_bundle(bundle)

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
    assert "Missing Synth.vst3 failed managed bundle integrity" in result.output


def test_plugins_cli_install_configures_studiorack_and_links_bundle(tmp_path: Path) -> None:
    """The install entrypoint configures Studiorack and creates its alias.

    :param tmp_path: Scratch root for the CLI boundary and call log.
    """
    manifest_path = _manifest(tmp_path / "studiorack.json")
    artifact_lock = _artifact_lock(tmp_path / "studiorack.lock.json")
    managed = tmp_path / "managed"
    bundle = _bundle(managed / "VST3/example/synth/1.2.3/Example Synth.vst3")
    _seal_bundle(bundle)
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
        [
            "config",
            "set",
            "artifactLockPath",
            str(artifact_lock.resolve()),
        ],
        ["plugins", "install", "example/synth@1.2.3"],
    ]
    alias = tmp_path / "checkout/plugins/Example Synth.vst3"
    assert alias.is_symlink()
    assert alias.resolve() == bundle.resolve()
