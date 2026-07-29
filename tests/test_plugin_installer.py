"""Behavior tests for manifest-backed Studiorack package management."""

from __future__ import annotations

import errno
import json
import multiprocessing
import subprocess
import traceback
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest

import synth_setter.plugin_integrity as plugin_integrity
import synth_setter.plugin_manager as plugin_manager
from synth_setter.data.vst.core import extract_renderer_version
from synth_setter.plugin_manager import (
    ArtifactLock,
    PluginManifest,
    install_plugins,
    link_plugin,
    managed_plugin_digest,
    resolve_plugin_bundle,
)
from tests.plugin_manager_test_support import (
    _adopt_bundle,
    _artifact_lock,
    _binary_path,
    _bundle,
    _example_lock,
    _manifest,
    _seal_bundle,
    _test_binary_magic,
)
from tests.plugin_manager_test_support import (
    _platform_binary_environment as _platform_binary_environment,
)


def _concurrent_install_worker(
    role: str,
    paths: tuple[Path, Path, Path, Path, Path],
    signals: tuple[Any, Any, Any, Any, Any, Any],
) -> None:
    """Run one process-controlled installation against shared managed state.

    :param role: ``first`` publishes bytes; ``second`` verifies the serialized result.
    :param paths: Manifest, lock, managed root, alias root, and executable paths.
    :param signals: Ordered synchronization events and result queue.
    """
    manifest_path, lock_path, managed, links_dir, executable = paths
    (
        first_started,
        release_first,
        second_attempting,
        release_second,
        second_entered,
        results,
    ) = signals
    try:
        manifest = PluginManifest.load(manifest_path)
        plugin = manifest.resolve("example/synth")
        bundle = managed / "VST3/example/synth/1.2.3/Example Synth.vst3"
        original_install = plugin_manager._install_plugin

        def _install(context: object) -> None:
            if role == "second":
                second_attempting.set()
            original_install(cast("plugin_manager._InstallContext", context))

        def _invoke(argv: list[str], _env: dict[str, str]) -> None:
            if argv[1:3] != ["plugins", "install"]:
                return
            if role == "first":
                _bundle(bundle, payload=b"published-by-first")
                first_started.set()
                if not release_first.wait(10):
                    raise RuntimeError("timed out waiting to release first installer")
                return
            second_entered.set()
            if not release_second.wait(10):
                raise RuntimeError("timed out waiting to release second installer")

        plugin_manager._install_plugin = _install
        plugin_manager._invoke_studiorack = _invoke
        install_plugins(
            (plugin,),
            artifact_lock=lock_path,
            plugins_dir=managed,
            studiorack_executable=executable,
            system_dirs=(),
        )
        link_plugin(
            plugin,
            artifact_lock=ArtifactLock.load(lock_path, manifest),
            plugins_dir=managed,
            links_dir=links_dir,
        )
    except BaseException:  # pragma: no cover - returned to the parent for assertion
        results.put(traceback.format_exc())
    else:
        results.put(None)


def test_install_plugins_same_package_processes_publish_one_valid_bundle(tmp_path: Path) -> None:
    """A package lock prevents a second process from deleting in-flight output.

    :param tmp_path: Scratch root shared by two spawned installer processes.
    """
    manifest_path = _manifest(tmp_path / "studiorack.json")
    lock_path = _artifact_lock(tmp_path / "studiorack.lock.json")
    managed = tmp_path / "managed"
    links_dir = tmp_path / "plugins"
    executable = tmp_path / "studiorack"
    executable.write_text("#!/usr/bin/env python3\n")
    executable.chmod(0o755)
    context = multiprocessing.get_context("spawn")
    first_started = context.Event()
    release_first = context.Event()
    second_attempting = context.Event()
    release_second = context.Event()
    second_entered = context.Event()
    results = context.Queue()
    paths = (manifest_path, lock_path, managed, links_dir, executable)
    signals = (
        first_started,
        release_first,
        second_attempting,
        release_second,
        second_entered,
        results,
    )
    first = context.Process(
        target=_concurrent_install_worker,
        args=("first", paths, signals),
    )
    second = context.Process(
        target=_concurrent_install_worker,
        args=("second", paths, signals),
    )

    first.start()
    assert first_started.wait(10)
    second.start()
    assert second_attempting.wait(10)
    second_entered_before_publication = second_entered.wait(2)
    release_second.set()
    release_first.set()
    first.join(10)
    second.join(10)

    assert not second_entered_before_publication
    assert first.exitcode == 0
    assert second.exitcode == 0
    assert results.get(timeout=2) is None
    assert results.get(timeout=2) is None
    plugin = PluginManifest.load(manifest_path).resolve("example/synth")
    alias = links_dir / plugin.bundle
    resolved = resolve_plugin_bundle(
        plugin,
        managed,
        artifact_lock=ArtifactLock.load(lock_path, PluginManifest.load(manifest_path)),
    )
    assert _binary_path(resolved).read_bytes() == _test_binary_magic() + b"published-by-first"
    assert alias.resolve(strict=True) == resolved.resolve(strict=True)
    assert managed_plugin_digest(alias) is not None
    assert (managed / ".synth-setter-install-locks/example/synth/1.2.3.lock").is_file()


def test_install_plugins_archive_renderer_version_matches_manifest_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Archive output is published when its renderer version matches the distinct pin.

    :param tmp_path: Scratch root for managed state and the fake archive installer.
    :param monkeypatch: Supplies the managed output path to the installer process.
    """
    manifest_path = _manifest(
        tmp_path / "studiorack.json",
        package_version="2026.2.0",
        renderer_version="0.26.2",
    )
    plugin = PluginManifest.load(manifest_path).resolve("example/synth")
    managed = tmp_path / "managed"
    bundle = managed / "VST3/example/synth/2026.2.0/Example Synth.vst3"
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    bundle = pathlib.Path(os.environ['STUDIORACK_TEST_BUNDLE'])\n"
        "    binary = bundle / os.environ['STUDIORACK_TEST_BINARY']\n"
        "    binary.parent.mkdir(parents=True)\n"
        "    binary.write_bytes(bytes.fromhex(os.environ['STUDIORACK_TEST_BINARY_MAGIC']) + b'archive')\n"
        "    (bundle / 'Contents/moduleinfo.json').write_text(json.dumps({'Version': '0.26.2'}))\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("STUDIORACK_TEST_BUNDLE", str(bundle))
    lock_path = _artifact_lock(tmp_path / "studiorack.lock.json", package_version="2026.2.0")

    install_plugins(
        (plugin,),
        artifact_lock=lock_path,
        plugins_dir=managed,
        studiorack_executable=executable,
        system_dirs=(),
    )

    assert (
        resolve_plugin_bundle(
            plugin,
            managed,
            artifact_lock=_example_lock(package_version="2026.2.0"),
        )
        == bundle
    )


def test_install_plugins_missing_executable_raises(tmp_path: Path) -> None:
    """Installation reports how to provision the pinned CLI.

    :param tmp_path: Scratch root without a Studiorack executable.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")

    with pytest.raises(FileNotFoundError, match="npm ci"):
        install_plugins(
            (plugin,),
            artifact_lock=_artifact_lock(tmp_path / "studiorack.lock.json"),
            plugins_dir=tmp_path / "managed",
            studiorack_executable=tmp_path / "missing-studiorack",
        )


def test_install_plugins_seal_permission_error_preserves_managed_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable seal is operational failure, not removable corruption.

    :param tmp_path: Scratch root for sealed managed state.
    :param monkeypatch: Injects the permission failure at the filesystem boundary.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    managed = tmp_path / "managed"
    bundle = _bundle(managed / "VST3/example/synth/1.2.3/Example Synth.vst3")
    _seal_bundle(bundle, plugin)
    binary = _binary_path(bundle)
    original_entries = plugin_integrity.bundle_entries

    def _permission_error(path: Path) -> list[plugin_integrity.BundleEntry]:
        if path == bundle:
            raise PermissionError(errno.EACCES, "permission denied", path)
        return original_entries(path)

    monkeypatch.setattr(plugin_integrity, "bundle_entries", _permission_error)
    executable = tmp_path / "studiorack"
    executable.write_text("#!/usr/bin/env python3\n")
    executable.chmod(0o755)

    with pytest.raises(PermissionError, match="permission denied"):
        install_plugins(
            (plugin,),
            artifact_lock=_artifact_lock(tmp_path / "studiorack.lock.json"),
            plugins_dir=managed,
            studiorack_executable=executable,
            system_dirs=(),
        )

    assert binary.read_bytes() == _test_binary_magic() + b"plugin"
    assert bundle.is_dir()


def test_install_plugins_native_snapshot_io_error_does_not_adopt_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unreadable stale native state cannot be classified as newly created.

    :param tmp_path: Scratch root for managed and native state.
    :param monkeypatch: Injects a transient candidate read failure.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    system_dir = tmp_path / "system-vst3"
    candidate = _bundle(system_dir / plugin.bundle, payload=b"stale")
    original_entries = plugin_integrity.bundle_entries

    def _io_error(path: Path) -> list[plugin_integrity.BundleEntry]:
        if path == candidate:
            raise OSError(errno.EIO, "input/output error", path)
        return original_entries(path)

    monkeypatch.setattr(plugin_integrity, "bundle_entries", _io_error)
    executable = tmp_path / "studiorack"
    executable.write_text("#!/usr/bin/env python3\n")
    executable.chmod(0o755)

    with pytest.raises(OSError, match="input/output error"):
        install_plugins(
            (plugin,),
            artifact_lock=_artifact_lock(tmp_path / "studiorack.lock.json"),
            plugins_dir=tmp_path / "managed",
            studiorack_executable=executable,
            system_dirs=(system_dir,),
        )

    managed_bundle = tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3"
    assert not managed_bundle.exists()
    assert candidate.is_dir()


def test_install_plugins_explicit_adoption_is_replaced_by_locked_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Locked install cannot retain bytes sealed as explicit source adoption.

    :param tmp_path: Scratch root for source and managed plugin state.
    :param monkeypatch: Supplies the managed destination to the fake installer.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    source = _bundle(tmp_path / "source/Example Synth.vst3", payload=b"source-build")
    managed_dir = tmp_path / "managed"
    managed = _adopt_bundle(plugin, plugins_dir=managed_dir, bundle=source)
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    bundle = pathlib.Path(os.environ['STUDIORACK_TEST_BUNDLE'])\n"
        "    if bundle.is_symlink():\n"
        "        raise SystemExit(0)\n"
        "    binary = bundle / os.environ['STUDIORACK_TEST_BINARY']\n"
        "    binary.parent.mkdir(parents=True)\n"
        "    binary.write_bytes(bytes.fromhex(os.environ['STUDIORACK_TEST_BINARY_MAGIC']) + b'locked-artifact')\n"
        "    (bundle / 'Contents/moduleinfo.json').write_text(json.dumps({'Version': '1.2.3'}))\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("STUDIORACK_TEST_BUNDLE", str(managed))

    install_plugins(
        (plugin,),
        artifact_lock=_artifact_lock(tmp_path / "studiorack.lock.json"),
        plugins_dir=managed_dir,
        studiorack_executable=executable,
        system_dirs=(),
    )

    seal = json.loads((managed.parent / ".synth-setter-complete.json").read_text())
    assert not managed.is_symlink()
    assert _binary_path(managed).read_bytes() == _test_binary_magic() + b"locked-artifact"
    assert _binary_path(source).read_bytes() == _test_binary_magic() + b"source-build"
    assert seal["source_kind"] == "artifact-lock"
    assert seal["locked_package_sha256"] == plugin_integrity.locked_package_digest(
        plugin.reference,
        _example_lock().package_for(plugin),
    )


def test_install_plugins_partial_managed_state_reinstalled_and_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt exact-version directory is removed before a clean reinstall.

    :param tmp_path: Scratch root for managed state and the fake installer.
    :param monkeypatch: Supplies the managed destination to the installer process.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    managed = tmp_path / "managed"
    bundle = _bundle(managed / "VST3/example/synth/1.2.3/Example Synth.vst3", payload=b"partial")
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "bundle = pathlib.Path(os.environ['STUDIORACK_TEST_BUNDLE'])\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    if bundle.exists():\n"
        "        raise SystemExit(9)\n"
        "    binary = bundle / os.environ['STUDIORACK_TEST_BINARY']\n"
        "    binary.parent.mkdir(parents=True)\n"
        "    binary.write_bytes(bytes.fromhex(os.environ['STUDIORACK_TEST_BINARY_MAGIC']) + b'reinstalled')\n"
        "    (bundle / 'Contents/moduleinfo.json').write_text(json.dumps({'Version': '1.2.3'}))\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("STUDIORACK_TEST_BUNDLE", str(bundle))

    install_plugins(
        (plugin,),
        artifact_lock=_artifact_lock(tmp_path / "studiorack.lock.json"),
        plugins_dir=managed,
        studiorack_executable=executable,
        system_dirs=(),
    )

    assert resolve_plugin_bundle(plugin, managed, artifact_lock=_example_lock()) == bundle
    assert _binary_path(bundle).read_bytes() == _test_binary_magic() + b"reinstalled"
    assert (bundle.parent / ".synth-setter-complete.json").is_file()


def test_install_plugins_invalid_managed_symlink_repaired_without_deleting_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repair unlinks managed adoption state without traversing its source.

    :param tmp_path: Scratch root for managed and source plugin trees.
    :param monkeypatch: Supplies the managed destination to the installer process.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    source = _bundle(tmp_path / "source/Example Synth.vst3", payload=b"source")
    managed = tmp_path / "managed"
    bundle = managed / "VST3/example/synth/1.2.3/Example Synth.vst3"
    bundle.parent.mkdir(parents=True)
    bundle.symlink_to(source, target_is_directory=True)
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    bundle = pathlib.Path(os.environ['STUDIORACK_TEST_BUNDLE'])\n"
        "    binary = bundle / os.environ['STUDIORACK_TEST_BINARY']\n"
        "    binary.parent.mkdir(parents=True)\n"
        "    binary.write_bytes(bytes.fromhex(os.environ['STUDIORACK_TEST_BINARY_MAGIC']) + b'repaired')\n"
        "    (bundle / 'Contents/moduleinfo.json').write_text(json.dumps({'Version': '1.2.3'}))\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("STUDIORACK_TEST_BUNDLE", str(bundle))

    install_plugins(
        (plugin,),
        artifact_lock=_artifact_lock(tmp_path / "studiorack.lock.json"),
        plugins_dir=managed,
        studiorack_executable=executable,
        system_dirs=(),
    )

    assert _binary_path(source).read_bytes() == _test_binary_magic() + b"source"
    assert not bundle.is_symlink()
    assert resolve_plugin_bundle(plugin, managed, artifact_lock=_example_lock()) == bundle


def test_install_plugins_native_rerun_adopts_fresh_changed_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new invocation adopts only output it changes after a fresh snapshot.

    :param tmp_path: Scratch root for managed, native, and installer state.
    :param monkeypatch: Supplies native paths to the fake installer.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    managed = tmp_path / "managed"
    system_dir = tmp_path / "system-vst3"
    bundle = _bundle(system_dir / plugin.bundle, payload=b"before")
    external = tmp_path / "external.txt"
    external.write_text("outside")
    calls = tmp_path / "calls.txt"
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    bundle = pathlib.Path(os.environ['STUDIORACK_TEST_BUNDLE'])\n"
        "    calls = pathlib.Path(os.environ['STUDIORACK_TEST_CALLS'])\n"
        "    is_rerun = calls.exists()\n"
        "    calls.write_text(calls.read_text() + 'install\\n' if is_rerun else 'install\\n')\n"
        "    binary = bundle / os.environ['STUDIORACK_TEST_BINARY']\n"
        "    payload = b'rerun' if is_rerun else b'changed'\n"
        "    binary.write_bytes(bytes.fromhex(os.environ['STUDIORACK_TEST_BINARY_MAGIC']) + payload)\n"
        "    if not is_rerun:\n"
        "        (bundle / 'Contents/external.txt').symlink_to(os.environ['STUDIORACK_TEST_EXTERNAL'])\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("STUDIORACK_TEST_BUNDLE", str(bundle))
    monkeypatch.setenv("STUDIORACK_TEST_CALLS", str(calls))
    monkeypatch.setenv("STUDIORACK_TEST_EXTERNAL", str(external))
    artifact_lock = _artifact_lock(tmp_path / "studiorack.lock.json")

    with pytest.raises(FileNotFoundError, match="did not create or change"):
        install_plugins(
            (plugin,),
            artifact_lock=artifact_lock,
            plugins_dir=managed,
            studiorack_executable=executable,
            system_dirs=(system_dir,),
        )

    transaction = managed / ".synth-setter-native-install/example/synth/1.2.3.json"
    assert transaction.is_file()
    (bundle / "Contents/external.txt").unlink()

    install_plugins(
        (plugin,),
        artifact_lock=artifact_lock,
        plugins_dir=managed,
        studiorack_executable=executable,
        system_dirs=(system_dir,),
    )

    assert calls.read_text().splitlines() == ["install", "install"]
    assert _binary_path(bundle).read_bytes() == _test_binary_magic() + b"rerun"
    assert (
        resolve_plugin_bundle(plugin, managed, artifact_lock=_example_lock()).resolve()
        == bundle.resolve()
    )
    assert not transaction.exists()


def test_install_plugins_persisted_transaction_does_not_trust_between_process_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rerun rejects same-version bytes not changed by its own installer.

    :param tmp_path: Scratch root for managed, native, and transaction state.
    :param monkeypatch: Supplies the installer call-log path.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    managed = tmp_path / "managed"
    system_dir = tmp_path / "system-vst3"
    bundle = _bundle(system_dir / plugin.bundle, payload=b"original")
    calls = tmp_path / "calls.txt"
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    calls = pathlib.Path(os.environ['STUDIORACK_TEST_CALLS'])\n"
        "    calls.write_text(calls.read_text() + 'install\\n' if calls.exists() else 'install\\n')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("STUDIORACK_TEST_CALLS", str(calls))
    artifact_lock = _artifact_lock(tmp_path / "studiorack.lock.json")

    with pytest.raises(FileNotFoundError, match="did not create or change"):
        install_plugins(
            (plugin,),
            artifact_lock=artifact_lock,
            plugins_dir=managed,
            studiorack_executable=executable,
            system_dirs=(system_dir,),
        )

    transaction = managed / ".synth-setter-native-install/example/synth/1.2.3.json"
    assert transaction.is_file()
    _binary_path(bundle).write_bytes(_test_binary_magic() + b"replacement-between-processes")

    with pytest.raises(FileNotFoundError, match="did not create or change"):
        install_plugins(
            (plugin,),
            artifact_lock=artifact_lock,
            plugins_dir=managed,
            studiorack_executable=executable,
            system_dirs=(system_dir,),
        )

    assert calls.read_text().splitlines() == ["install", "install"]
    assert not (managed / "VST3/example/synth/1.2.3/Example Synth.vst3").is_symlink()


def test_install_plugins_unchanged_native_candidate_rejected(
    tmp_path: Path,
) -> None:
    """A pre-existing same-name native bundle cannot satisfy a new install.

    :param tmp_path: Scratch root for managed and native plugin trees.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    system_dir = tmp_path / "system-vst3"
    _bundle(system_dir / plugin.bundle, version="1.2.3", payload=b"stale")
    executable = tmp_path / "studiorack"
    executable.write_text("#!/usr/bin/env python3\n")
    executable.chmod(0o755)

    with pytest.raises(FileNotFoundError, match="did not create or change"):
        install_plugins(
            (plugin,),
            artifact_lock=_artifact_lock(tmp_path / "studiorack.lock.json"),
            plugins_dir=tmp_path / "managed",
            studiorack_executable=executable,
            system_dirs=(system_dir,),
        )

    assert not (tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3").is_symlink()


def test_install_plugins_native_retry_adopts_change_from_first_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One invocation retains its original native snapshot across transient retries.

    :param tmp_path: Scratch root for managed, native, and installer state.
    :param monkeypatch: Supplies native output and call-count paths to the installer.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    managed = tmp_path / "managed"
    system_dir = tmp_path / "system-vst3"
    bundle = _bundle(system_dir / plugin.bundle, payload=b"before")
    calls = tmp_path / "calls.txt"
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    calls = pathlib.Path(os.environ['STUDIORACK_TEST_CALLS'])\n"
        "    attempt = int(calls.read_text()) + 1 if calls.exists() else 1\n"
        "    calls.write_text(str(attempt))\n"
        "    if attempt == 1:\n"
        "        bundle = pathlib.Path(os.environ['STUDIORACK_TEST_BUNDLE'])\n"
        "        binary = bundle / os.environ['STUDIORACK_TEST_BINARY']\n"
        "        binary.write_bytes(bytes.fromhex(os.environ['STUDIORACK_TEST_BINARY_MAGIC']) + b'changed')\n"
        "        raise SystemExit(8)\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("STUDIORACK_TEST_BUNDLE", str(bundle))
    monkeypatch.setenv("STUDIORACK_TEST_CALLS", str(calls))

    install_plugins(
        (plugin,),
        artifact_lock=_artifact_lock(tmp_path / "studiorack.lock.json"),
        plugins_dir=managed,
        studiorack_executable=executable,
        system_dirs=(system_dir,),
    )

    resolved = resolve_plugin_bundle(plugin, managed, artifact_lock=_example_lock())
    assert calls.read_text() == "2"
    assert resolved.is_symlink()
    assert resolved.resolve() == bundle.resolve()
    assert _binary_path(resolved).read_bytes() == _test_binary_magic() + b"changed"


def test_install_plugins_nonzero_native_output_retried_then_rejected_without_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Complete native output from failed attempts is never adopted or sealed.

    :param tmp_path: Scratch root for managed, native, and installer state.
    :param monkeypatch: Supplies native paths to the fake installer.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    managed = tmp_path / "managed"
    system_dir = tmp_path / "system-vst3"
    bundle = _bundle(system_dir / plugin.bundle, payload=b"before")
    calls = tmp_path / "calls.txt"
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    bundle = pathlib.Path(os.environ['STUDIORACK_TEST_BUNDLE'])\n"
        "    calls = pathlib.Path(os.environ['STUDIORACK_TEST_CALLS'])\n"
        "    calls.write_text(calls.read_text() + 'install\\n' if calls.exists() else 'install\\n')\n"
        "    binary = bundle / os.environ['STUDIORACK_TEST_BINARY']\n"
        "    binary.write_bytes(bytes.fromhex(os.environ['STUDIORACK_TEST_BINARY_MAGIC']) + b'changed')\n"
        "    raise SystemExit(8)\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("STUDIORACK_TEST_BUNDLE", str(bundle))
    monkeypatch.setenv("STUDIORACK_TEST_CALLS", str(calls))

    with pytest.raises(subprocess.CalledProcessError):
        install_plugins(
            (plugin,),
            artifact_lock=_artifact_lock(tmp_path / "studiorack.lock.json"),
            plugins_dir=managed,
            studiorack_executable=executable,
            system_dirs=(system_dir,),
        )

    assert calls.read_text().splitlines() == ["install", "install", "install"]
    version_dir = managed / "VST3/example/synth/1.2.3"
    assert not (version_dir / plugin.bundle).exists()
    assert not (version_dir / ".synth-setter-complete.json").exists()


def test_install_plugins_success_does_not_publish_managed_output_from_failed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later zero exit cannot publish managed output left by an earlier failure.

    :param tmp_path: Scratch root for managed output and installer state.
    :param monkeypatch: Supplies managed paths to the fake installer.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    managed = tmp_path / "managed"
    bundle = managed / "VST3/example/synth/1.2.3/Example Synth.vst3"
    calls = tmp_path / "calls.txt"
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    calls = pathlib.Path(os.environ['STUDIORACK_TEST_CALLS'])\n"
        "    attempt = int(calls.read_text()) + 1 if calls.exists() else 1\n"
        "    calls.write_text(str(attempt))\n"
        "    if attempt == 1:\n"
        "        bundle = pathlib.Path(os.environ['STUDIORACK_TEST_BUNDLE'])\n"
        "        binary = bundle / os.environ['STUDIORACK_TEST_BINARY']\n"
        "        binary.parent.mkdir(parents=True)\n"
        "        binary.write_bytes(bytes.fromhex(os.environ['STUDIORACK_TEST_BINARY_MAGIC']) + b'failed')\n"
        "        (bundle / 'Contents/moduleinfo.json').write_text(json.dumps({'Version': '1.2.3'}))\n"
        "        raise SystemExit(8)\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("STUDIORACK_TEST_BUNDLE", str(bundle))
    monkeypatch.setenv("STUDIORACK_TEST_CALLS", str(calls))

    with pytest.raises(FileNotFoundError, match="did not create or change"):
        install_plugins(
            (plugin,),
            artifact_lock=_artifact_lock(tmp_path / "studiorack.lock.json"),
            plugins_dir=managed,
            studiorack_executable=executable,
            system_dirs=(),
        )

    assert calls.read_text() == "2"
    assert not bundle.exists()
    assert not (bundle.parent / ".synth-setter-complete.json").exists()


def test_install_plugins_success_with_invalid_binary_signature_rejected_without_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful installer cannot seal a nonempty bundle with an invalid host binary.

    :param tmp_path: Scratch root for managed output and the fake installer.
    :param monkeypatch: Supplies the managed destination to the fake installer.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    managed = tmp_path / "managed"
    bundle = managed / "VST3/example/synth/1.2.3/Example Synth.vst3"
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    bundle = pathlib.Path(os.environ['STUDIORACK_TEST_BUNDLE'])\n"
        "    binary = bundle / os.environ['STUDIORACK_TEST_BINARY']\n"
        "    binary.parent.mkdir(parents=True)\n"
        "    binary.write_bytes(b'not-a-native-binary')\n"
        "    (bundle / 'Contents/moduleinfo.json').write_text(json.dumps({'Version': '1.2.3'}))\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("STUDIORACK_TEST_BUNDLE", str(bundle))

    with pytest.raises(ValueError, match="exactly one valid platform VST3 binary"):
        install_plugins(
            (plugin,),
            artifact_lock=_artifact_lock(tmp_path / "studiorack.lock.json"),
            plugins_dir=managed,
            studiorack_executable=executable,
            system_dirs=(),
        )

    assert not (bundle.parent / ".synth-setter-complete.json").exists()
    assert not (tmp_path / "plugins/Example Synth.vst3").exists()


def test_install_plugins_nonzero_truncated_header_valid_binary_rejected_without_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Binary magic and exact static metadata cannot publish failed installer output.

    :param tmp_path: Scratch root for managed and native state.
    :param monkeypatch: Supplies the native output path to the installer.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    managed = tmp_path / "managed"
    system_dir = tmp_path / "system-vst3"
    bundle = _bundle(system_dir / plugin.bundle, payload=b"before")
    calls = tmp_path / "calls.txt"
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    bundle = pathlib.Path(os.environ['STUDIORACK_TEST_BUNDLE'])\n"
        "    calls = pathlib.Path(os.environ['STUDIORACK_TEST_CALLS'])\n"
        "    calls.write_text(calls.read_text() + 'install\\n' if calls.exists() else 'install\\n')\n"
        "    magic = bytes.fromhex(os.environ['STUDIORACK_TEST_BINARY_MAGIC'])\n"
        "    (bundle / os.environ['STUDIORACK_TEST_BINARY']).write_bytes(magic)\n"
        "    (bundle / 'Contents/moduleinfo.json').write_text(json.dumps({'Version': '1.2.3'}))\n"
        "    raise SystemExit(8)\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("STUDIORACK_TEST_BUNDLE", str(bundle))
    monkeypatch.setenv("STUDIORACK_TEST_CALLS", str(calls))

    with pytest.raises(subprocess.CalledProcessError):
        install_plugins(
            (plugin,),
            artifact_lock=_artifact_lock(tmp_path / "studiorack.lock.json"),
            plugins_dir=managed,
            studiorack_executable=executable,
            system_dirs=(system_dir,),
        )

    assert calls.read_text().splitlines() == ["install", "install", "install"]
    version_dir = managed / "VST3/example/synth/1.2.3"
    assert not (version_dir / plugin.bundle).exists()
    assert not (version_dir / ".synth-setter-complete.json").exists()


@pytest.mark.parametrize(
    "stderr",
    [
        "artifact lock mismatch for example/synth@1.2.3 (linux-x64)",
        "artifact lock missing example/synth@1.2.3",
        "Studiorack artifact lock path is required",
        "Error: ENOENT: no such file or directory, open '/repo/studiorack.lock.json'",
        "SyntaxError: Expected property name or '}' in JSON at position 1",
        "SyntaxError: Unexpected token 'b', \"broken\" is not valid JSON",
        "Invalid package slug: example",
        "Invalid package version: latest",
        "Package example/missing not found in registry",
        "Package example/synth version 9.9.9 not found in registry",
        "download denied: 401 Unauthorized",
        "download denied: 403 Forbidden",
        "download failed: HTTP 404",
    ],
)
def test_install_plugins_permanent_cli_failure_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stderr: str
) -> None:
    """Stable Studiorack permanent errors stop after the first attempt.

    :param tmp_path: Scratch root for managed state and the fake Studiorack CLI.
    :param monkeypatch: Supplies a call counter to the fake CLI.
    :param stderr: Stable permanent error emitted by Studiorack or HTTP.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    calls = tmp_path / "calls.txt"
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    calls = pathlib.Path(os.environ['STUDIORACK_TEST_CALLS'])\n"
        "    calls.write_text(calls.read_text() + 'install\\n' if calls.exists() else 'install\\n')\n"
        f"    print({stderr!r}, file=sys.stderr)\n"
        "    raise SystemExit(7)\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("STUDIORACK_TEST_CALLS", str(calls))
    monkeypatch.setattr(plugin_manager.time, "sleep", lambda _seconds: None)

    with pytest.raises(subprocess.CalledProcessError):
        install_plugins(
            (plugin,),
            artifact_lock=_artifact_lock(tmp_path / "studiorack.lock.json"),
            plugins_dir=tmp_path / "managed",
            studiorack_executable=executable,
            system_dirs=(),
        )

    assert calls.read_text().splitlines() == ["install"]


@pytest.mark.parametrize(
    "stderr",
    [
        "fetch failed: ECONNRESET",
        "download failed: HTTP 503",
        "unexpected installer failure",
    ],
)
def test_install_plugins_transient_or_unclassified_cli_failure_retries_to_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stderr: str
) -> None:
    """Network, server, and unclassified failures retain bounded retries.

    :param tmp_path: Scratch root for managed state and the fake Studiorack CLI.
    :param monkeypatch: Supplies a call counter to the fake CLI.
    :param stderr: Retryable or unclassified process error.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    calls = tmp_path / "calls.txt"
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    calls = pathlib.Path(os.environ['STUDIORACK_TEST_CALLS'])\n"
        "    calls.write_text(calls.read_text() + 'install\\n' if calls.exists() else 'install\\n')\n"
        f"    print({stderr!r}, file=sys.stderr)\n"
        "    raise SystemExit(7)\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("STUDIORACK_TEST_CALLS", str(calls))

    with pytest.raises(subprocess.CalledProcessError):
        install_plugins(
            (plugin,),
            artifact_lock=_artifact_lock(tmp_path / "studiorack.lock.json"),
            plugins_dir=tmp_path / "managed",
            studiorack_executable=executable,
            system_dirs=(),
        )

    assert calls.read_text().splitlines() == ["install", "install", "install"]


def test_install_plugins_subprocess_timeout_retries_to_limit_without_sleep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hung Studiorack process uses bounded retries before an actionable error.

    :param tmp_path: Scratch root for manifest and lock files.
    :param monkeypatch: Replaces subprocess and retry sleeping deterministically.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    calls = 0

    def _timeout(*args: object, **kwargs: object) -> NoReturn:
        nonlocal calls
        calls += 1
        command = cast("list[str]", args[0])
        timeout = cast("float", kwargs["timeout"])
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(plugin_manager.subprocess, "run", _timeout)
    monkeypatch.setattr(plugin_manager.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="timed out.*3 attempts"):
        install_plugins(
            (plugin,),
            artifact_lock=_artifact_lock(tmp_path / "studiorack.lock.json"),
            plugins_dir=tmp_path / "managed",
            studiorack_executable=Path("/bin/true"),
            system_dirs=(),
        )

    assert calls == 3


def test_install_plugins_transient_cli_failure_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient installer exit retains bounded retry behavior.

    :param tmp_path: Scratch root for managed state and the fake Studiorack CLI.
    :param monkeypatch: Supplies managed paths to the fake CLI.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    managed = tmp_path / "managed"
    bundle = managed / "VST3/example/synth/1.2.3/Example Synth.vst3"
    calls = tmp_path / "calls.txt"
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    calls = pathlib.Path(os.environ['STUDIORACK_TEST_CALLS'])\n"
        "    attempt = int(calls.read_text()) + 1 if calls.exists() else 1\n"
        "    calls.write_text(str(attempt))\n"
        "    if attempt < 3:\n"
        "        raise SystemExit(8)\n"
        "    bundle = pathlib.Path(os.environ['STUDIORACK_TEST_BUNDLE'])\n"
        "    binary = bundle / os.environ['STUDIORACK_TEST_BINARY']\n"
        "    binary.parent.mkdir(parents=True)\n"
        "    binary.write_bytes(bytes.fromhex(os.environ['STUDIORACK_TEST_BINARY_MAGIC']) + b'plugin')\n"
        "    (bundle / 'Contents/moduleinfo.json').write_text(json.dumps({'Version': '1.2.3'}))\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("STUDIORACK_TEST_BUNDLE", str(bundle))
    monkeypatch.setenv("STUDIORACK_TEST_CALLS", str(calls))

    install_plugins(
        (plugin,),
        artifact_lock=_artifact_lock(tmp_path / "studiorack.lock.json"),
        plugins_dir=managed,
        studiorack_executable=executable,
        system_dirs=(),
    )

    assert calls.read_text() == "3"
    assert resolve_plugin_bundle(plugin, managed, artifact_lock=_example_lock()) == bundle


def test_install_plugins_changed_native_wrong_version_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed native bundle with another version is not adopted.

    :param tmp_path: Scratch root for managed and native plugin trees.
    :param monkeypatch: Supplies the native bundle path to the installer process.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    system_dir = tmp_path / "system-vst3"
    bundle = _bundle(system_dir / plugin.bundle, version="1.2.3", payload=b"before")
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

    with pytest.raises(ValueError, match="expected 1.2.3.*found 9.9.9"):
        install_plugins(
            (plugin,),
            artifact_lock=_artifact_lock(tmp_path / "studiorack.lock.json"),
            plugins_dir=tmp_path / "managed",
            studiorack_executable=executable,
            system_dirs=(system_dir,),
        )

    assert not (tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3").is_symlink()


def test_install_plugins_changed_native_exact_version_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful native install enters the exact managed namespace.

    :param tmp_path: Scratch root for managed and system plugin trees.
    :param monkeypatch: Supplies the native bundle path to the fake installer.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))
    plugin = manifest.resolve("example/synth")
    system_dir = tmp_path / "system-vst3"
    bundle = _bundle(system_dir / plugin.bundle, payload=b"before")
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    bundle = pathlib.Path(os.environ['STUDIORACK_TEST_BUNDLE'])\n"
        "    binary = bundle / os.environ['STUDIORACK_TEST_BINARY']\n"
        "    binary.write_bytes(bytes.fromhex(os.environ['STUDIORACK_TEST_BINARY_MAGIC']) + b'changed')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("STUDIORACK_TEST_BUNDLE", str(bundle))

    install_plugins(
        (plugin,),
        artifact_lock=_artifact_lock(tmp_path / "studiorack.lock.json"),
        plugins_dir=tmp_path / "managed",
        studiorack_executable=executable,
        system_dirs=(system_dir,),
    )

    resolved = resolve_plugin_bundle(
        plugin,
        tmp_path / "managed",
        artifact_lock=_example_lock(),
    )
    assert resolved.is_symlink()
    assert resolved.resolve() == bundle.resolve()


def test_install_plugins_native_adoption_uses_distinct_renderer_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native adoption validates renderer identity while retaining package storage.

    :param tmp_path: Scratch root for managed and system plugin trees.
    :param monkeypatch: Supplies the native bundle path to the fake installer.
    """
    manifest_path = _manifest(tmp_path / "studiorack.json")
    payload = json.loads(manifest_path.read_text())
    payload["vst3Versions"] = {"example/synth": "0.26.2"}
    manifest_path.write_text(json.dumps(payload))
    plugin = PluginManifest.load(manifest_path).resolve("example/synth")
    system_dir = tmp_path / "system-vst3"
    bundle = _bundle(system_dir / plugin.bundle, version="0.26.2", payload=b"before")
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    bundle = pathlib.Path(os.environ['STUDIORACK_TEST_BUNDLE'])\n"
        "    binary = bundle / os.environ['STUDIORACK_TEST_BINARY']\n"
        "    binary.write_bytes(bytes.fromhex(os.environ['STUDIORACK_TEST_BINARY_MAGIC']) + b'changed')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("STUDIORACK_TEST_BUNDLE", str(bundle))

    install_plugins(
        (plugin,),
        artifact_lock=_artifact_lock(tmp_path / "studiorack.lock.json"),
        plugins_dir=tmp_path / "managed",
        studiorack_executable=executable,
        system_dirs=(system_dir,),
    )

    resolved = resolve_plugin_bundle(
        plugin,
        tmp_path / "managed",
        artifact_lock=_example_lock(),
    )
    assert resolved == tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3"
    assert extract_renderer_version(resolved) == "0.26.2"
