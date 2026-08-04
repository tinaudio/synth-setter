"""Shared filesystem fixtures for managed-plugin tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from synth_setter.cli.plugins import main
from synth_setter.plugin_integrity import ArtifactLock
from synth_setter.plugin_manager import (
    ManagedPlugin,
    PluginManifest,
    adopt_plugin_bundle,
    seal_plugin_bundle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _test_binary_relative() -> Path:
    """Return the host VST3 executable path used by test bundles.

    :returns: Platform-relative VST3 executable path.
    """
    if sys.platform == "darwin":
        return Path("Contents/MacOS/Example Synth")
    if sys.platform.startswith("linux"):
        return Path("Contents/x86_64-linux/Example Synth.so")
    return Path("Contents/x86_64-win/Example Synth.vst3")


def _pe_test_binary_magic() -> bytes:
    """Build a minimal PE header with its signature at the declared offset.

    :returns: Minimal valid PE signature bytes.
    """
    header = bytearray(68)
    header[:2] = b"MZ"
    header[0x3C:0x40] = (64).to_bytes(4, "little")
    header[64:68] = b"PE\0\0"
    return bytes(header)


def _test_binary_magic() -> bytes:
    """Return a minimal host-native executable signature.

    :returns: Platform-native executable signature bytes.
    """
    if sys.platform == "darwin":
        return b"\xcf\xfa\xed\xfe"
    if sys.platform.startswith("linux"):
        return b"\x7fELF"
    return _pe_test_binary_magic()


def _binary_path(bundle: Path) -> Path:
    """Locate the host executable in a test bundle.

    :param bundle: Platform-shaped VST3 bundle.
    :returns: Host executable path within the bundle.
    """
    return bundle / _test_binary_relative()


@pytest.fixture(autouse=True)
def _platform_binary_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose host bundle layout to subprocess installer fixtures.

    :param monkeypatch: Sets subprocess fixture environment variables.
    """
    monkeypatch.setenv("STUDIORACK_TEST_BINARY", _test_binary_relative().as_posix())
    monkeypatch.setenv("STUDIORACK_TEST_BINARY_MAGIC", _test_binary_magic().hex())


def _manifest(
    path: Path,
    *,
    package_version: str = "1.2.3",
    renderer_version: str | None = None,
) -> Path:
    """Write an exact example package manifest.

    :param path: Manifest output path.
    :param package_version: Exact example package version.
    :param renderer_version: Optional distinct VST3-reported version.
    :returns: Written manifest path.
    """
    payload = {
        "name": "test-project",
        "type": "project",
        "plugins": {"example/synth": package_version},
        "vst3Bundles": {"example/synth": "Example Synth.vst3"},
    }
    if renderer_version is not None:
        payload["vst3Versions"] = {"example/synth": renderer_version}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _bundle(path: Path, *, version: str = "1.2.3", payload: bytes = b"plugin") -> Path:
    """Write a platform-shaped bundle with static version metadata.

    :param path: VST3 bundle output path.
    :param version: Static renderer version.
    :param payload: Bytes appended to the platform executable signature.
    :returns: Written bundle path.
    """
    contents = path / "Contents"
    binary = _binary_path(path)
    binary.parent.mkdir(parents=True)
    binary.write_bytes(_test_binary_magic() + payload)
    (contents / "moduleinfo.json").write_text(json.dumps({"Version": version}))
    return path


def _run_archive_cli_install(
    tmp_path: Path,
    expected_version: str,
    actual_version: str,
) -> tuple[Result, Path, Path]:
    """Run the plugin CLI against a local archive-producing installer.

    :param tmp_path: Scratch root for the manifest, bundle, and alias.
    :param expected_version: Exact runtime version pinned by the manifest.
    :param actual_version: Runtime version written into the installed bundle.
    :returns: CLI result, versioned bundle path, and checkout alias path.
    """
    package_version = ".".join(expected_version.split(".")[:3])
    manifest_path = _manifest(
        tmp_path / "studiorack.json",
        package_version=package_version,
        renderer_version=expected_version,
    )
    plugin = PluginManifest.load(manifest_path).resolve("example/synth")
    _artifact_lock(tmp_path / "studiorack.lock.json", package_version=package_version)
    managed = tmp_path / "managed"
    bundle = managed / f"VST3/example/synth/{package_version}/Example Synth.vst3"
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
        f"    (bundle / 'Contents/moduleinfo.json').write_text(json.dumps({{'Version': {actual_version!r}}}))\n"
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
            str(links_dir),
            "--studiorack-executable",
            str(executable),
            "install",
        ],
        env={**os.environ, "STUDIORACK_TEST_BUNDLE": str(bundle)},
    )
    return result, bundle, links_dir / plugin.bundle


def _example_lock(*, package_version: str = "1.2.3") -> ArtifactLock:
    """Return the repository identity for the example package.

    :param package_version: Exact package version represented by the lock.
    :returns: Validated example artifact lock.
    """
    return ArtifactLock.model_validate(
        {
            f"example/synth@{package_version}": {
                "artifacts": [
                    {
                        "architectures": ["x64"],
                        "sha256": "a" * 64,
                        "systems": ["linux"],
                        "type": "archive",
                        "url": "https://example.test/synth.zip",
                    }
                ]
            }
        }
    )


def _cross_platform_lock(*, linux_sha256: str = "a" * 64) -> ArtifactLock:
    """Return one package lock containing macOS and Linux artifacts.

    :param linux_sha256: Linux artifact digest used to model lock rotation.
    :returns: Validated cross-platform artifact lock.
    """
    return ArtifactLock.model_validate(
        {
            "example/synth@1.2.3": {
                "artifacts": [
                    {
                        "architectures": ["arm64"],
                        "sha256": "b" * 64,
                        "systems": ["mac"],
                        "type": "archive",
                        "url": "https://example.test/synth-mac.zip",
                    },
                    {
                        "architectures": ["x64"],
                        "sha256": linux_sha256,
                        "systems": ["linux"],
                        "type": "archive",
                        "url": "https://example.test/synth-linux.zip",
                    },
                ]
            }
        }
    )


def _example_plugin(*, bundle_name: str = "Example Synth.vst3") -> ManagedPlugin:
    """Return the exact package identity used by test bundles.

    :param bundle_name: Managed VST3 bundle basename.
    :returns: Exact example managed-plugin identity.
    """
    return ManagedPlugin(
        package="example/synth",
        version="1.2.3",
        renderer_version="1.2.3",
        bundle=bundle_name,
    )


def _adopt_bundle(
    plugin: ManagedPlugin,
    *,
    plugins_dir: Path,
    bundle: Path,
    artifact_lock: ArtifactLock | None = None,
) -> Path:
    """Adopt a test source under an explicit exact package lock.

    :param plugin: Exact package represented by the source bundle.
    :param plugins_dir: Isolated managed storage root.
    :param bundle: Existing source bundle to adopt.
    :param artifact_lock: Lock override, or the default example lock.
    :returns: Managed adopted bundle path.
    """
    lock = artifact_lock or _example_lock(package_version=plugin.version)
    return adopt_plugin_bundle(
        plugin,
        plugins_dir=plugins_dir,
        bundle=bundle,
        locked_package=lock.package_for(plugin),
    )


def _seal_bundle(
    plugin_bundle: Path,
    plugin: ManagedPlugin | None = None,
    artifact_lock: ArtifactLock | None = None,
) -> Path:
    """Seal a test bundle under an exact package lock.

    :param plugin_bundle: Bundle content to seal.
    :param plugin: Package override inferred from the bundle when omitted.
    :param artifact_lock: Lock override, or the default example lock.
    :returns: Written bundle-seal path.
    """
    selected = plugin or _example_plugin(bundle_name=plugin_bundle.name)
    lock = artifact_lock or _example_lock()
    return seal_plugin_bundle(
        plugin_bundle,
        selected,
        locked_package=lock.package_for(selected),
    )


def _artifact_lock(path: Path, *, package_version: str = "1.2.3") -> Path:
    """Write the example lock passed across Python and Node.

    :param path: Artifact-lock output path.
    :param package_version: Exact example package version.
    :returns: Written artifact-lock path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_example_lock(package_version=package_version).model_dump_json())
    return path
