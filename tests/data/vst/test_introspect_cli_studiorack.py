from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner
from omegaconf import OmegaConf

import synth_setter
from synth_setter.cli.introspect_plugin import main
from synth_setter.plugin_manager import ArtifactLock, PluginManifest, seal_plugin_bundle
from tests.data.vst._introspect_fakes import IntrospectFakePlugin

_REAL_PKG_DIR = Path(synth_setter.__file__).parent


def _binary_path(bundle: Path) -> tuple[Path, bytes]:
    """Return host executable location and signature for an introspection bundle.

    :param bundle: Test bundle root.
    :returns: Platform executable path and native signature bytes.
    """
    if sys.platform == "darwin":
        return bundle / "Contents/MacOS/Example Synth", b"\xcf\xfa\xed\xfe"
    if sys.platform.startswith("linux"):
        return bundle / "Contents/x86_64-linux/Example Synth.so", b"\x7fELF"
    header = bytearray(68)
    header[:2] = b"MZ"
    header[0x3C:0x40] = (64).to_bytes(4, "little")
    header[64:68] = b"PE\0\0"
    return bundle / "Contents/x86_64-win/Example Synth.vst3", bytes(header)


def _artifact_lock(manifest_path: Path) -> ArtifactLock:
    """Write and parse the lock paired with an introspection manifest.

    :param manifest_path: Manifest requiring exact lock coverage.
    :returns: Parsed sibling artifact lock.
    """
    lock_path = manifest_path.with_suffix(".lock.json")
    lock_path.write_text(
        json.dumps(
            {
                "example/synth@9.9.9": {
                    "artifacts": [
                        {
                            "architectures": ["x64"],
                            "sha256": "a" * 64,
                            "systems": ["linux"],
                            "type": "archive",
                            "url": "https://example.test/plugin.zip",
                        }
                    ]
                }
            }
        )
    )
    return ArtifactLock.load(lock_path, PluginManifest.load(manifest_path))


def _seal(bundle: Path, manifest_path: Path) -> None:
    """Seal a fixture under its manifest's exact package provenance.

    :param bundle: Managed fixture bundle.
    :param manifest_path: Manifest and sibling lock governing the bundle.
    """
    manifest = PluginManifest.load(manifest_path)
    plugin = manifest.resolve("example/synth")
    lock = _artifact_lock(manifest_path)
    seal_plugin_bundle(bundle, plugin, locked_package=lock.package_for(plugin))


def _checkout(tmp_path: Path) -> Path:
    """Create the minimum checkout layout required by registration.

    :param tmp_path: Scratch root receiving the checkout.
    :returns: Skeleton checkout root.
    """
    root = tmp_path / "checkout"
    vst_dir = root / "src/synth_setter/data/vst"
    vst_dir.mkdir(parents=True)
    shutil.copy(_REAL_PKG_DIR / "data/vst/param_spec_registry.py", vst_dir)
    shutil.copy(_REAL_PKG_DIR / "synth_spec.py", vst_dir.parents[1])
    render_dir = root / "src/synth_setter/configs/render"
    render_dir.mkdir(parents=True)
    shutil.copy(_REAL_PKG_DIR / "configs/render/vst.yaml", render_dir)
    return root


def test_register_studiorack_plugin_links_managed_bundle_and_records_alias(
    tmp_path: Path,
    fake_plugin: IntrospectFakePlugin,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed registration records a stable alias instead of a host path.

    :param tmp_path: Scratch root for checkout and managed plugin storage.
    :param fake_plugin: Real introspection surface behind the patched loader boundary.
    :param monkeypatch: Replaces the native plugin loader with the test plugin.
    """
    checkout = _checkout(tmp_path)
    manifest = checkout / "studiorack.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "test-project",
                "type": "project",
                "plugins": {"example/synth": "9.9.9"},
                "vst3Bundles": {"example/synth": "Example Synth.vst3"},
            }
        )
    )
    managed = tmp_path / "managed"
    bundle = managed / "VST3/example/synth/9.9.9/Example Synth.vst3"
    binary, magic = _binary_path(bundle)
    binary.parent.mkdir(parents=True)
    binary.write_bytes(magic + b"plugin")
    (bundle / "Contents/moduleinfo.json").write_text('{"Version": "9.9.9"}')
    _seal(bundle, manifest)
    monkeypatch.setattr(
        "synth_setter.cli.introspect_plugin.load_plugin",
        lambda _path, _name=None: fake_plugin,
    )

    result = CliRunner().invoke(
        main,
        [
            "--studiorack-plugin",
            "example/synth",
            "--studiorack-manifest",
            str(manifest),
            "--studiorack-plugins-dir",
            str(managed),
            "--spec-name",
            "fake_synth",
            "--register",
            "--repo-root",
            str(checkout),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    alias = checkout / "plugins/Example Synth.vst3"
    assert alias.is_symlink()
    assert alias.resolve() == bundle.resolve()
    identity = OmegaConf.load(checkout / "src/synth_setter/configs/synth/fake_synth.yaml")
    assert identity.plugin_path == "plugins/Example Synth.vst3"
    assert identity.synth_version == "9.9.9"


def test_register_studiorack_conflict_preserves_existing_alias(tmp_path: Path) -> None:
    """Registration validation runs before changing a managed alias.

    :param tmp_path: Scratch root for checkout and managed plugin storage.
    """
    checkout = _checkout(tmp_path)
    old_bundle = tmp_path / "old/Surge XT.vst3"
    old_bundle.mkdir(parents=True)
    alias = checkout / "plugins/Surge XT.vst3"
    alias.parent.mkdir()
    alias.symlink_to(old_bundle, target_is_directory=True)
    manifest = checkout / "studiorack.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "test-project",
                "type": "project",
                "plugins": {"example/synth": "9.9.9"},
                "vst3Bundles": {"example/synth": "Surge XT.vst3"},
            }
        )
    )
    new_bundle = tmp_path / "managed/VST3/example/synth/9.9.9/Surge XT.vst3"
    binary, magic = _binary_path(new_bundle)
    binary.parent.mkdir(parents=True)
    binary.write_bytes(magic + b"plugin")
    _seal(new_bundle, manifest)

    result = CliRunner().invoke(
        main,
        [
            "--studiorack-plugin",
            "example/synth",
            "--studiorack-manifest",
            str(manifest),
            "--studiorack-plugins-dir",
            str(tmp_path / "managed"),
            "--spec-name",
            "surge_xt",
            "--register",
            "--repo-root",
            str(checkout),
        ],
    )

    assert result.exit_code != 0
    assert alias.resolve() == old_bundle.resolve()


def test_register_rejects_plugin_path_with_studiorack_plugin(
    tmp_path: Path,
) -> None:
    """Managed and explicit plugin selectors cannot be combined.

    :param tmp_path: Scratch root for the checkout and explicit bundle.
    """
    checkout = _checkout(tmp_path)
    explicit = tmp_path / "explicit.vst3"
    explicit.mkdir()

    result = CliRunner().invoke(
        main,
        [
            "--plugin-path",
            str(explicit),
            "--studiorack-plugin",
            "example/synth",
            "--spec-name",
            "fake_synth",
            "--register",
            "--repo-root",
            str(checkout),
        ],
    )

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output
