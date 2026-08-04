"""Runtime lease and managed-alias transaction tests."""

from __future__ import annotations

import errno
import json
import multiprocessing
import os
import shutil
import stat
import threading
import traceback
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path, PosixPath
from typing import Any, Literal, NoReturn, cast

import pytest

import synth_setter.data.vst.core as vst_core
import synth_setter.plugin_integrity as plugin_integrity
import synth_setter.plugin_manager as plugin_manager
import synth_setter.plugin_runtime as plugin_runtime
from synth_setter.data.vst.core import extract_renderer_version, load_plugin
from synth_setter.plugin_integrity import PluginIntegrityError
from synth_setter.plugin_manager import (
    ArtifactLock,
    ManagedPlugin,
    PluginManifest,
    install_plugins,
    link_plugin,
    managed_plugin_digest,
    resolve_plugin_bundle,
)
from tests._vst import PLUGIN_PATH
from tests.plugin_manager_test_support import (
    PROJECT_ROOT,
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


def _same_path_reinstall_worker(
    paths: tuple[Path, Path, Path],
    signals: tuple[Any, Any, Any],
) -> None:
    """Replace one package version after its consumer lease is released.

    :param paths: Lock, managed storage, and installer executable paths.
    :param signals: Attempt, completion, and result synchronization objects.
    """
    lock_path, managed, executable = paths
    attempting, completed, results = signals
    try:
        manifest = PluginManifest.load(lock_path.with_name("studiorack.json"))
        plugin = manifest.resolve("example/synth")
        original_install = plugin_manager._install_plugin

        def _attempt(context: object) -> None:
            attempting.set()
            original_install(cast("plugin_manager._InstallContext", context))

        plugin_manager._install_plugin = _attempt
        install_plugins(
            (plugin,),
            artifact_lock=lock_path,
            plugins_dir=managed,
            studiorack_executable=executable,
            system_dirs=(),
        )
    except BaseException:  # pragma: no cover - returned to the parent for assertion
        results.put(traceback.format_exc())
    else:
        completed.set()
        results.put(None)


def _readopt_mutated_source_worker(
    paths: tuple[Path, Path, Path],
    result: Any,
) -> None:
    """Re-adopt a changed source already recorded as a managed alias.

    :param paths: Manifest, managed storage, and native source paths.
    :param result: Queue receiving the managed path or traceback.
    """
    manifest_path, managed_root, source = paths
    try:
        plugin = PluginManifest.load(manifest_path).resolve("example/synth")
        managed = _adopt_bundle(plugin, plugins_dir=managed_root, bundle=source)
    except BaseException:  # pragma: no cover - returned to the parent for assertion
        result.put(traceback.format_exc())
    else:
        result.put(str(managed))


def _removed_same_path_reinstall_worker(
    paths: tuple[Path, Path, Path],
    signals: tuple[Any, Any, Any],
) -> None:
    """Hold the package lock while one managed version is temporarily absent.

    :param paths: Manifest, lock, and managed storage paths.
    :param signals: Removal, release, and result synchronization objects.
    :raises RuntimeError: Reinstall release is not signaled before timeout.
    """
    manifest_path, lock_path, managed = paths
    removed, release, results = signals
    try:
        manifest = PluginManifest.load(manifest_path)
        plugin = manifest.resolve("example/synth")
        artifact_lock = ArtifactLock.load(lock_path, manifest)
        with plugin_manager._package_install_lock(plugin, managed.resolve()):
            plugin_manager._remove_managed_version(plugin, managed.resolve())
            removed.set()
            if not release.wait(10):
                raise RuntimeError("timed out waiting to finish reinstall")
            plugin_bundle = _bundle(
                managed / "VST3/example/synth/1.2.3/Example Synth.vst3",
                payload=b"artifact-b",
            )
            plugin_integrity.seal_plugin_bundle(
                plugin_bundle,
                plugin,
                locked_package=artifact_lock.package_for(plugin),
            )
    except BaseException:  # pragma: no cover - returned to the parent for assertion
        results.put(traceback.format_exc())
    else:
        results.put(None)


def _rotated_package(artifact_lock: ArtifactLock, plugin: ManagedPlugin) -> ArtifactLock:
    """Return a lock with changed artifact identity for one plugin.

    :param artifact_lock: Original immutable artifact lock.
    :param plugin: Plugin whose artifact identity changes.
    :returns: Validated lock carrying the changed digest.
    """
    payload = artifact_lock.model_dump(mode="json")
    payload[plugin.reference]["artifacts"][0]["sha256"] = "c" * 64
    return ArtifactLock.model_validate(payload)


def _replace_managed_bundle(
    bundle: Path,
    plugin: ManagedPlugin,
    artifact_lock: ArtifactLock,
) -> None:
    """Replace and reseal the binary bytes in a managed bundle.

    :param bundle: Existing managed bundle.
    :param plugin: Package identity used for sealing.
    :param artifact_lock: Rotated artifact provenance.
    """
    _binary_path(bundle).write_bytes(_test_binary_magic() + b"artifact-b")
    plugin_integrity.seal_plugin_bundle(
        bundle,
        plugin,
        locked_package=artifact_lock.package_for(plugin),
    )


def test_load_plugin_managed_host_value_error_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lease does not misclassify a host-construction error as bad bytes.

    :param tmp_path: Scratch root for one sealed managed bundle.
    :param monkeypatch: Replaces the VST host with a rejecting constructor.
    """
    plugin_bundle = _bundle(tmp_path / "Example Synth.vst3")
    _seal_bundle(plugin_bundle)

    def _reject_plugin_class(path: str, plugin_name: str | None = None) -> NoReturn:
        del path, plugin_name
        raise ValueError("multiple plugin classes")

    monkeypatch.setattr(vst_core, "VST3Plugin", _reject_plugin_class)

    with pytest.raises(ValueError, match="multiple plugin classes"):
        load_plugin(str(plugin_bundle))


def test_validate_plugin_bundle_lease_exit_failure_translated_to_integrity_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public validation translates lock substitution detected during lease exit.

    :param tmp_path: Scratch managed bundle and adopted ownership state.
    :param monkeypatch: Simulates retained lock-directory replacement on exit.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    managed = _adopt_bundle(
        plugin,
        plugins_dir=tmp_path / "managed",
        bundle=_bundle(tmp_path / "source/Example Synth.vst3"),
    )

    @contextmanager
    def _replaced_lease(path: Path) -> Iterator[None]:
        del path
        yield
        raise ValueError("lock directory was replaced")

    monkeypatch.setattr(plugin_integrity, "advisory_file_lease", _replaced_lease)

    with pytest.raises(PluginIntegrityError, match="managed bundle integrity"):
        plugin_manager.validate_plugin_bundle_for_runtime(managed)


def _read_only_managed_alias(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a sealed alias with system-install permission modes.

    :param tmp_path: Scratch root for managed content and its checkout alias.
    :returns: Alias, managed root, and durable package lock paths.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    managed_root = tmp_path / "managed"
    bundle = _bundle(
        managed_root / "VST3/example/synth/1.2.3/Example Synth.vst3",
        payload=b"system-install",
    )
    _seal_bundle(bundle, plugin, _example_lock())
    alias = tmp_path / "plugins/Example Synth.vst3"
    alias.parent.mkdir()
    alias.symlink_to(bundle.absolute(), target_is_directory=True)
    plugin_runtime.record_managed_alias(alias, bundle)
    lock_path = plugin_integrity.package_install_lock_path(
        plugin.package,
        plugin.version,
        managed_root,
    )
    lock_path.parent.mkdir(parents=True)
    lock_path.touch(mode=0o444)
    managed_root.chmod(0o555)
    return alias, managed_root, lock_path


@pytest.mark.skipif(os.name == "nt", reason="POSIX system-install ownership semantics")
def test_load_plugin_read_only_managed_root_uses_existing_install_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime consumers lock a system-managed bundle without writing its install tree.

    :param tmp_path: Scratch root made non-writable after package installation.
    :param monkeypatch: Replaces the native host with a bundle-content consumer.
    """
    alias, managed_root, lock_path = _read_only_managed_alias(tmp_path)
    loaded_payloads: list[bytes] = []

    def _load(path: str, plugin_name: str | None = None) -> object:
        del plugin_name
        loaded_payloads.append(_binary_path(Path(path)).read_bytes())
        return object()

    monkeypatch.setattr(vst_core, "VST3Plugin", _load)
    try:
        load_plugin(str(alias))
    finally:
        managed_root.chmod(0o755)
        lock_path.chmod(0o644)

    assert loaded_payloads == [_test_binary_magic() + b"system-install"]


def test_renderer_construction_before_rotation_consumes_validated_old_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rotation begun after prevalidation cannot replace bytes under construction.

    :param tmp_path: Scratch root shared by the runtime and installer threads.
    :param monkeypatch: Replaces the native host with a synchronized byte consumer.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    artifact_lock = _example_lock()
    rotated_lock = _rotated_package(artifact_lock, plugin)
    managed_root = tmp_path / "managed"
    bundle = _bundle(
        managed_root / "VST3/example/synth/1.2.3/Example Synth.vst3",
        payload=b"artifact-a",
    )
    _seal_bundle(bundle, plugin, artifact_lock)
    expected_digest = managed_plugin_digest(bundle)
    assert expected_digest is not None

    constructing = threading.Event()
    rotation_started = threading.Event()
    consumed: list[bytes] = []

    def _construct(path: str, plugin_name: str | None = None) -> object:
        del plugin_name
        constructing.set()
        if not rotation_started.wait(10):
            raise RuntimeError("timed out waiting for rotation attempt")
        consumed.append(_binary_path(Path(path)).read_bytes())
        return object()

    def _rotate() -> None:
        if not constructing.wait(10):
            raise RuntimeError("timed out waiting for renderer construction")
        rotation_started.set()
        with plugin_integrity.package_install_lock(
            plugin.package,
            plugin.version,
            managed_root.resolve(),
        ):
            _replace_managed_bundle(bundle, plugin, rotated_lock)

    monkeypatch.setattr(vst_core, "VST3Plugin", _construct)
    installer = threading.Thread(target=_rotate)
    installer.start()

    load_plugin(str(bundle))
    installer.join(10)

    assert not installer.is_alive()
    assert consumed == [_test_binary_magic() + b"artifact-a"]
    assert managed_plugin_digest(bundle) != expected_digest


@pytest.mark.parametrize("consumer", ["load", "version"])
def test_validated_bundle_lease_blocks_same_path_reinstall_until_consumer_opens_a(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consumer: Literal["load", "version"],
) -> None:
    """Validation and consumption hold one package lock across same-path replacement.

    :param tmp_path: Scratch root shared by consumer and installer processes.
    :param monkeypatch: Controls consumer timing and the installer payload.
    :param consumer: Managed runtime operation exercised while holding the lease.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))
    plugin = manifest.resolve("example/synth")
    managed = tmp_path / "managed"
    plugin_bundle = _bundle(
        managed / "VST3/example/synth/1.2.3/Example Synth.vst3",
        payload=b"artifact-a",
    )
    _seal_bundle(plugin_bundle, plugin, _example_lock())
    alias = tmp_path / "plugins/Example Synth.vst3"
    alias.parent.mkdir()
    alias.symlink_to(plugin_bundle.absolute(), target_is_directory=True)
    plugin_runtime.record_managed_alias(alias, plugin_bundle)

    lock_path = _artifact_lock(tmp_path / "studiorack.lock.json")
    rotated = ArtifactLock.model_validate_json(lock_path.read_text())
    rotated_payload = rotated.model_dump(mode="json")
    rotated_payload[plugin.reference]["artifacts"][0]["sha256"] = "c" * 64
    lock_path.write_text(ArtifactLock.model_validate(rotated_payload).model_dump_json())
    executable = tmp_path / "studiorack"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "if sys.argv[1:3] == ['plugins', 'install']:\n"
        "    bundle = pathlib.Path(os.environ['STUDIORACK_TEST_BUNDLE'])\n"
        "    binary = bundle / os.environ['STUDIORACK_TEST_BINARY']\n"
        "    binary.parent.mkdir(parents=True)\n"
        "    binary.write_bytes(bytes.fromhex(os.environ['STUDIORACK_TEST_BINARY_MAGIC']) + b'artifact-b')\n"
        "    (bundle / 'Contents/moduleinfo.json').write_text(json.dumps({'Version': '1.2.3'}))\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("STUDIORACK_TEST_BUNDLE", str(plugin_bundle))

    validated = threading.Event()
    release_consumer = threading.Event()
    opened_payloads: list[bytes | str] = []
    consumer_errors: list[BaseException] = []
    real_lease = plugin_runtime.validated_bundle_lease

    @contextmanager
    def _pause_after_validation(path: Path, **_kwargs: object) -> Iterator[Path]:
        with real_lease(path) as resolved:
            validated.set()
            if not release_consumer.wait(10):
                raise RuntimeError("timed out waiting to release consumer")
            yield resolved

    if consumer == "load":
        monkeypatch.setattr(vst_core, "validated_bundle_lease", _pause_after_validation)

        def _open(path: str, plugin_name: str | None = None) -> object:
            del plugin_name
            opened_payloads.append(_binary_path(Path(path)).read_bytes())
            return object()

        monkeypatch.setattr(vst_core, "VST3Plugin", _open)

        def _consume() -> None:
            vst_core.load_plugin(str(alias))

    else:
        monkeypatch.setattr(plugin_runtime, "validated_bundle_lease", _pause_after_validation)

        def _consume() -> None:
            opened_payloads.append(plugin_runtime.plugin_bundle_version(alias))

    def _run_consumer() -> None:
        try:
            _consume()
        except BaseException as exc:  # pragma: no cover - asserted in the parent thread
            consumer_errors.append(exc)

    consumer_thread = threading.Thread(target=_run_consumer)
    consumer_thread.start()
    assert validated.wait(10)

    context = multiprocessing.get_context("spawn")
    attempting = context.Event()
    completed = context.Event()
    results = context.Queue()
    installer = context.Process(
        target=_same_path_reinstall_worker,
        args=((lock_path, managed, executable), (attempting, completed, results)),
    )
    installer.start()
    assert attempting.wait(10)
    assert not completed.wait(0.5)
    assert _binary_path(plugin_bundle).read_bytes() == _test_binary_magic() + b"artifact-a"

    release_consumer.set()
    consumer_thread.join(10)
    installer.join(10)

    assert not consumer_thread.is_alive()
    assert installer.exitcode == 0
    assert results.get(timeout=2) is None
    assert consumer_errors == []
    expected = _test_binary_magic() + b"artifact-a" if consumer == "load" else "1.2.3"
    assert opened_payloads == [expected]
    assert _binary_path(plugin_bundle).read_bytes() == _test_binary_magic() + b"artifact-b"


@pytest.mark.parametrize("consumer", ["load", "version"])
def test_validated_bundle_lease_waits_for_in_progress_same_path_reinstall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consumer: Literal["load", "version"],
) -> None:
    """A consumer entering during replacement waits for the same package lock.

    :param tmp_path: Scratch root shared by consumer and installer processes.
    :param monkeypatch: Replaces plugin construction with a payload reader.
    :param consumer: Managed runtime operation started during replacement.
    """
    manifest_path = _manifest(tmp_path / "studiorack.json")
    manifest = PluginManifest.load(manifest_path)
    plugin = manifest.resolve("example/synth")
    managed = tmp_path / "managed"
    plugin_bundle = _bundle(
        managed / "VST3/example/synth/1.2.3/Example Synth.vst3",
        payload=b"artifact-a",
    )
    _seal_bundle(plugin_bundle, plugin, _example_lock())
    alias = tmp_path / "plugins/Example Synth.vst3"
    alias.parent.mkdir()
    alias.symlink_to(plugin_bundle.absolute(), target_is_directory=True)
    plugin_runtime.record_managed_alias(alias, plugin_bundle)

    lock_path = _artifact_lock(tmp_path / "studiorack.lock.json")
    rotated = ArtifactLock.model_validate_json(lock_path.read_text())
    rotated_payload = rotated.model_dump(mode="json")
    rotated_payload[plugin.reference]["artifacts"][0]["sha256"] = "c" * 64
    lock_path.write_text(ArtifactLock.model_validate(rotated_payload).model_dump_json())

    context = multiprocessing.get_context("spawn")
    removed = context.Event()
    release_installer = context.Event()
    results = context.Queue()
    installer = context.Process(
        target=_removed_same_path_reinstall_worker,
        args=((manifest_path, lock_path, managed), (removed, release_installer, results)),
    )
    installer.start()
    assert removed.wait(10)
    assert not plugin_bundle.exists()

    opened_payloads: list[bytes | str] = []
    consumer_errors: list[BaseException] = []
    consumer_completed = threading.Event()
    if consumer == "load":

        def _open(path: str, plugin_name: str | None = None) -> object:
            del plugin_name
            opened_payloads.append(_binary_path(Path(path)).read_bytes())
            return object()

        monkeypatch.setattr(vst_core, "VST3Plugin", _open)

        def _consume() -> None:
            load_plugin(str(alias))

    else:

        def _consume() -> None:
            opened_payloads.append(plugin_runtime.plugin_bundle_version(alias))

    def _run_consumer() -> None:
        try:
            _consume()
        except BaseException as exc:  # pragma: no cover - asserted in the parent thread
            consumer_errors.append(exc)
        finally:
            consumer_completed.set()

    consumer_thread = threading.Thread(target=_run_consumer)
    consumer_thread.start()
    assert not consumer_completed.wait(0.5)

    release_installer.set()
    consumer_thread.join(10)
    installer.join(10)

    assert installer.exitcode == 0
    assert results.get(timeout=2) is None
    assert not consumer_thread.is_alive()
    assert consumer_errors == []
    expected = _test_binary_magic() + b"artifact-b" if consumer == "load" else "1.2.3"
    assert opened_payloads == [expected]


def test_link_plugin_managed_bundle_creates_stable_checkout_alias(tmp_path: Path) -> None:
    """Managed bundles receive stable checkout-local aliases.

    :param tmp_path: Scratch root for managed storage and checkout aliases.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))
    plugin = manifest.resolve("example/synth")
    bundle = _bundle(tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3")
    _seal_bundle(bundle, plugin)

    alias = link_plugin(
        plugin,
        artifact_lock=_example_lock(),
        plugins_dir=tmp_path / "managed",
        links_dir=tmp_path / "checkout/plugins",
    )

    assert alias == tmp_path / "checkout/plugins/Example Synth.vst3"
    assert alias.is_symlink()
    assert alias.resolve() == bundle.resolve()


def test_publish_runtime_tree_permissions_windows_junction_not_traversed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows permission publication does not mutate junction targets.

    :param tmp_path: Scratch managed tree containing a simulated junction.
    :param monkeypatch: Selects the Windows walker and identifies the junction.
    """
    root = tmp_path / "snapshot"
    junction = root / "junction"
    junction.mkdir(parents=True)
    payload = junction / "external.bin"
    payload.write_bytes(b"external")
    payload.chmod(0o600)
    monkeypatch.setattr(plugin_runtime.os, "name", "nt")
    monkeypatch.setattr(plugin_runtime, "Path", PosixPath)
    monkeypatch.setattr(
        plugin_runtime.os.path,
        "isjunction",
        lambda path: PosixPath(path).name == junction.name,
    )

    plugin_runtime._publish_runtime_tree_permissions(root)

    assert stat.S_IMODE(payload.stat().st_mode) == 0o600


def test_publish_runtime_snapshot_permissions_absent_directory_noop(tmp_path: Path) -> None:
    """Accept a package-version directory without a snapshot directory.

    :param tmp_path: Scratch package-version directory.
    """
    plugin_runtime._publish_runtime_snapshot_permissions(tmp_path / "version")


def test_prepare_managed_bundle_publishes_direct_bundle_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installer preparation makes direct managed bundle contents readable.

    :param tmp_path: Scratch managed bundle hierarchy.
    :param monkeypatch: Rejects mutable-path chmod during POSIX publication.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    plugins_dir = tmp_path / "managed"
    managed = _bundle(plugins_dir / "VST3/example/synth/1.2.3/Example Synth.vst3")
    _seal_bundle(managed, plugin, _example_lock())
    _restrict_runtime_tree_permissions(managed)

    def _reject_path_chmod(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("path chmod followed")

    monkeypatch.setattr(Path, "chmod", _reject_path_chmod)

    plugin_runtime._prepare_managed_bundle_for_runtime(managed, plugins_dir)

    for current, _, files in os.walk(managed):
        assert Path(current).stat().st_mode & 0o055 == 0o055
        assert all((Path(current) / name).stat().st_mode & 0o044 == 0o044 for name in files)


def test_windows_runtime_parent_permissions_publish_real_hierarchy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows publication makes each real managed parent traversable.

    :param tmp_path: Scratch managed hierarchy.
    :param monkeypatch: Replaces native Windows handle acquisition.
    """
    opened: list[Path] = []

    def _retain(path: Path) -> int:
        opened.append(path)
        return id(path)

    monkeypatch.setattr(plugin_integrity, "_windows_open_directory_handle", _retain)
    monkeypatch.setattr(plugin_integrity, "_windows_close_handle", lambda handle: None)
    plugins_dir = tmp_path / "managed"
    version_dir = plugins_dir / "VST3/example/synth/1.2.3"
    version_dir.mkdir(parents=True)
    hierarchy = [plugins_dir]
    for component in version_dir.relative_to(plugins_dir).parts:
        hierarchy.append(hierarchy[-1] / component)
    for path in hierarchy:
        path.chmod(0o700)

    plugin_runtime._publish_windows_runtime_parent_permissions(plugins_dir, version_dir)

    assert opened == hierarchy


def test_windows_runtime_parent_permissions_reject_symlinked_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows publication rejects a linked parent without mutating its target.

    :param tmp_path: Scratch managed hierarchy and outside target.
    :param monkeypatch: Rejects linked directories through the native-handle boundary.
    """

    def _reject_link(path: Path) -> int:
        if path.is_symlink():
            raise OSError(f"runtime hierarchy path is not a real directory: {path}")
        return id(path)

    monkeypatch.setattr(plugin_integrity, "_windows_open_directory_handle", _reject_link)
    monkeypatch.setattr(plugin_integrity, "_windows_close_handle", lambda handle: None)
    plugins_dir = tmp_path / "managed"
    plugins_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    (plugins_dir / "VST3").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError, match="not a real directory"):
        plugin_runtime._publish_windows_runtime_parent_permissions(
            plugins_dir,
            plugins_dir / "VST3/example/synth/1.2.3",
        )

    assert stat.S_IMODE(outside.stat().st_mode) == 0o700


@pytest.mark.skipif(os.name == "nt", reason="POSIX traversal mode regression")
def test_adopt_plugin_bundle_restrictive_umask_publishes_managed_hierarchy(tmp_path: Path) -> None:
    """Privileged adoption publishes traversal through the package version.

    :param tmp_path: Scratch source and managed storage root.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    source = _bundle(tmp_path / "system/Example Synth.vst3")
    plugins_dir = tmp_path / "managed"
    prior_umask = os.umask(0o077)
    try:
        managed = _adopt_bundle(plugin, plugins_dir=plugins_dir, bundle=source)
    finally:
        os.umask(prior_umask)

    relative_version = managed.parent.relative_to(plugins_dir)
    hierarchy = [plugins_dir]
    for component in relative_version.parts:
        hierarchy.append(hierarchy[-1] / component)
    assert all(path.stat().st_mode & 0o055 == 0o055 for path in hierarchy)


def test_adopt_plugin_bundle_prepares_snapshot_before_read_only_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public adoption prepares runtime bytes before storage becomes read-only.

    :param tmp_path: Scratch source and managed storage root.
    :param monkeypatch: Rejects any runtime attempt to create a missing snapshot path.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    source = _bundle(tmp_path / "system/Example Synth.vst3")
    managed = plugin_manager.adopt_plugin_bundle(
        plugin,
        plugins_dir=tmp_path / "managed",
        bundle=source,
        locked_package=_example_lock().package_for(plugin),
    )
    snapshots = managed.parent / ".synth-setter-runtime-snapshots"
    for current, directories, _ in os.walk(tmp_path / "managed"):
        Path(current).chmod(0o555)
        for name in directories:
            (Path(current) / name).chmod(0o555)
    snapshot_bundles = list(snapshots.glob("*/Example Synth.vst3"))
    assert len(snapshot_bundles) == 1
    assert _binary_path(snapshot_bundles[0]).is_file()

    real_mkdir = os.mkdir

    def _deny_snapshot_creation(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        if str(path).startswith(".Example Synth.vst3.tmp-"):
            raise PermissionError("read-only managed storage")
        real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(plugin_runtime.os, "mkdir", _deny_snapshot_creation)

    assert snapshots.is_dir()
    assert plugin_manager.validate_plugin_bundle_for_runtime(managed).is_dir()


def test_package_install_lock_replaces_invalid_snapshot_container(tmp_path: Path) -> None:
    """Installer preparation replaces invalid snapshot-container state.

    :param tmp_path: Scratch managed alias and snapshot path.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    source = _bundle(tmp_path / "system/Example Synth.vst3")
    managed = _adopt_bundle(plugin, plugins_dir=tmp_path / "managed", bundle=source)
    snapshots = managed.parent / ".synth-setter-runtime-snapshots"
    shutil.rmtree(snapshots)
    snapshots.write_text("invalid")

    with plugin_manager._package_install_lock(plugin, tmp_path / "managed"):
        pass

    assert snapshots.is_dir()
    assert plugin_manager.validate_plugin_bundle_for_runtime(managed).is_dir()


def _restrict_runtime_tree_permissions(root: Path) -> None:
    """Restrict every directory and regular file in one runtime tree.

    :param root: Existing bundle or snapshot tree.
    """
    for current, _, files in os.walk(root):
        Path(current).chmod(0o700)
        for name in files:
            (Path(current) / name).chmod(0o600)


def test_link_plugin_native_source_alias_preserves_existing_real_bundle(tmp_path: Path) -> None:
    """Docker native fallback keeps its source but consumes a managed snapshot.

    :param tmp_path: Scratch root reproducing the ``/usr/lib/vst3`` fallback shape.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    system_vst3 = tmp_path / "usr/lib/vst3"
    source = _bundle(system_vst3 / plugin.bundle)
    _restrict_runtime_tree_permissions(source)
    managed = _adopt_bundle(
        plugin,
        plugins_dir=tmp_path / "managed",
        bundle=source,
        artifact_lock=_example_lock(),
    )

    alias = link_plugin(
        plugin,
        artifact_lock=_example_lock(),
        plugins_dir=tmp_path / "managed",
        links_dir=system_vst3,
    )

    assert alias == source
    assert alias.is_dir()
    assert not alias.is_symlink()
    assert managed.is_symlink()
    assert managed.resolve() == alias.resolve()
    initial_snapshot = plugin_manager.validate_plugin_bundle_for_runtime(alias)
    assert stat.S_IMODE(_binary_path(initial_snapshot).stat().st_mode) == 0o644
    snapshots = managed.parent / ".synth-setter-runtime-snapshots"
    restricted_directories = [
        snapshots,
        initial_snapshot.parent,
        initial_snapshot,
        initial_snapshot / "Contents",
    ]
    for directory in restricted_directories:
        directory.chmod(0o700)

    with plugin_manager._package_install_lock(plugin, tmp_path / "managed"):
        pass
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o755 for path in restricted_directories)

    validated = plugin_manager.validate_plugin_bundle_for_runtime(alias)

    assert validated != alias.resolve(strict=True)
    assert stat.S_IMODE(snapshots.stat().st_mode) == 0o755
    assert stat.S_IMODE(validated.parent.stat().st_mode) == 0o755
    assert plugin_integrity.bundle_entries(validated) == plugin_integrity.bundle_entries(alias)


def _shared_lease_observer(
    real_lease: Callable[[Path], AbstractContextManager[None]],
    both_entered: threading.Event,
) -> Callable[[Path], AbstractContextManager[None]]:
    """Build a lease wrapper that signals simultaneous consumer entry.

    :param real_lease: Shared lease implementation under test.
    :param both_entered: Event set once two leases overlap.
    :returns: Instrumented lease callable.
    """
    active_leases = 0
    count_lock = threading.Lock()

    @contextmanager
    def _observe(path: Path) -> Iterator[None]:
        nonlocal active_leases
        with real_lease(path):
            with count_lock:
                active_leases += 1
                if active_leases == 2:
                    both_entered.set()
            try:
                yield
            finally:
                with count_lock:
                    active_leases -= 1

    return _observe


def _pausing_snapshot_publisher(
    managed: Path,
    *,
    started: threading.Event,
    release: threading.Event,
    destinations: list[Path],
) -> Callable[..., None]:
    """Build a publisher that pauses before replacing the managed snapshot.

    :param managed: Managed alias naming the snapshot destination.
    :param started: Event set when publication reaches replacement.
    :param release: Event allowing replacement to continue.
    :param destinations: Collection receiving attempted destination paths.
    :returns: Instrumented runtime snapshot publisher.
    """
    real_publish = plugin_runtime._publish_posix_runtime_snapshot

    def _pause(
        source: Path,
        destination: Path,
        parent_descriptor: int,
        *,
        seal: plugin_integrity.BundleSeal,
    ) -> None:
        if destination.name == managed.name:
            destinations.append(destination)
            started.set()
            if not release.wait(10):
                raise RuntimeError("timed out waiting to publish runtime snapshot")
        real_publish(source, destination, parent_descriptor, seal=seal)

    return _pause


def _start_runtime_snapshot_consumer(
    managed: Path,
    results: list[Path],
    errors: list[BaseException],
    *,
    completed: threading.Event | None = None,
) -> threading.Thread:
    """Start one managed-bundle validation consumer.

    :param managed: Managed alias consumed by the thread.
    :param results: Collection receiving validated paths.
    :param errors: Collection receiving thread failures.
    :param completed: Optional completion event.
    :returns: Started consumer thread.
    """

    def _consume() -> None:
        try:
            results.append(plugin_manager.validate_plugin_bundle_for_runtime(managed))
        except BaseException as exc:  # pragma: no cover - asserted in the parent thread
            errors.append(exc)
        finally:
            if completed is not None:
                completed.set()

    thread = threading.Thread(target=_consume)
    thread.start()
    return thread


def _adopt_bundle_without_runtime_snapshot(tmp_path: Path) -> Path:
    """Create one adopted bundle whose installer snapshot was removed.

    :param tmp_path: Scratch root for the source and managed bundle.
    :returns: Adopted managed bundle path.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    source = _bundle(tmp_path / "system/Example Synth.vst3")
    managed = _adopt_bundle(plugin, plugins_dir=tmp_path / "managed", bundle=source)
    shutil.rmtree(managed.parent / ".synth-setter-runtime-snapshots")
    return managed


def _assert_concurrent_snapshot_results(
    published_destinations: list[Path],
    results: list[Path],
    errors: list[BaseException],
) -> None:
    """Assert both consumers received one identically published snapshot.

    :param published_destinations: Snapshot destinations atomically published.
    :param results: Snapshot paths returned to consumers.
    :param errors: Consumer thread failures.
    """
    assert errors == []
    assert len(published_destinations) == 1
    assert len(results) == 2
    assert results[0] == results[1]


def test_concurrent_runtime_consumers_publish_one_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared package leases serialize one adopted-snapshot publication.

    :param tmp_path: Scratch root for one adopted source and managed alias.
    :param monkeypatch: Pauses the real snapshot copy while another consumer enters.
    """
    managed = _adopt_bundle_without_runtime_snapshot(tmp_path)
    publication_started = threading.Event()
    release_publication = threading.Event()
    second_completed = threading.Event()
    both_leases_entered = threading.Event()
    published_destinations: list[Path] = []
    results: list[Path] = []
    errors: list[BaseException] = []
    observed_lease = _shared_lease_observer(
        plugin_runtime.integrity.advisory_file_lease,
        both_leases_entered,
    )
    paused_publisher = _pausing_snapshot_publisher(
        managed,
        started=publication_started,
        release=release_publication,
        destinations=published_destinations,
    )

    monkeypatch.setattr(plugin_runtime, "_publish_posix_runtime_snapshot", paused_publisher)
    monkeypatch.setattr(plugin_runtime.integrity, "advisory_file_lease", observed_lease)
    first = _start_runtime_snapshot_consumer(managed, results, errors)
    assert publication_started.wait(10)
    second = _start_runtime_snapshot_consumer(
        managed,
        results,
        errors,
        completed=second_completed,
    )
    assert both_leases_entered.wait(10)
    assert not second_completed.wait(0.2)
    assert len(published_destinations) == 1

    release_publication.set()
    first.join(10)
    second.join(10)

    _assert_concurrent_snapshot_results(published_destinations, results, errors)


def test_replace_managed_alias_symlinked_ownership_rejected_before_target_read(
    tmp_path: Path,
) -> None:
    """A symlinked ownership record is rejected without following its target.

    :param tmp_path: Scratch root containing an unreadable record target.
    """
    alias = tmp_path / "plugins/Example Synth.vst3"
    alias.parent.mkdir(parents=True)
    ownership, _ = plugin_runtime.managed_alias_paths(alias)
    sensitive = tmp_path / "sensitive.json"
    sensitive.write_text('{"managed_bundle":"/outside/secret.vst3"}')
    sensitive.chmod(0)
    ownership.symlink_to(sensitive)

    with pytest.raises(FileExistsError, match="not a regular file"):
        plugin_runtime.replace_managed_alias(alias, tmp_path / "managed/Example Synth.vst3")


def test_replace_managed_alias_racing_regular_file_is_not_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file racing temporary symlink creation survives failed publication.

    :param tmp_path: Scratch root for alias publication.
    :param monkeypatch: Injects a deterministic competing file creation.
    """
    alias = tmp_path / "plugins/Example Synth.vst3"
    alias.parent.mkdir(parents=True)
    racing_paths: list[Path] = []
    real_symlink_to = Path.symlink_to

    def _race_symlink(
        path: Path,
        target: Path,
        target_is_directory: bool = False,
    ) -> None:
        if path.parent != alias.parent:
            path.write_text("unrelated")
            racing_paths.append(path)
        real_symlink_to(path, target, target_is_directory=target_is_directory)

    monkeypatch.setattr(Path, "symlink_to", _race_symlink)

    with pytest.raises(FileExistsError):
        plugin_runtime.replace_managed_alias(alias, tmp_path / "managed/Example Synth.vst3")

    assert racing_paths[0].read_text() == "unrelated"


def test_link_plugin_ownership_enospc_exposes_no_new_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ownership publication failure leaves no unowned alias visible.

    :param tmp_path: Scratch root for managed storage and checkout aliases.
    :param monkeypatch: Injects ENOSPC at ownership-record replacement.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    managed = _bundle(tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3")
    _seal_bundle(managed, plugin)
    alias = tmp_path / "checkout/plugins/Example Synth.vst3"
    ownership = alias.parent / ".Example Synth.vst3.synth-setter-managed.json"
    real_replace = os.replace

    def _fail_ownership_replace(source: Path | str, destination: Path | str) -> None:
        if Path(destination) == ownership:
            raise OSError(errno.ENOSPC, "no space left on device", str(destination))
        real_replace(source, destination)

    monkeypatch.setattr(plugin_runtime.os, "replace", _fail_ownership_replace)

    with pytest.raises(OSError) as excinfo:
        link_plugin(
            plugin,
            artifact_lock=_example_lock(),
            plugins_dir=tmp_path / "managed",
            links_dir=alias.parent,
        )

    assert excinfo.value.errno == errno.ENOSPC
    with pytest.raises(FileNotFoundError):
        alias.lstat()
    assert not ownership.exists()


def test_link_plugin_alias_replace_enospc_restores_prior_working_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Alias replacement failure restores the prior ownership publication.

    :param tmp_path: Scratch root for old and new sealed managed bundles.
    :param monkeypatch: Injects ENOSPC at atomic alias replacement.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    managed = _bundle(tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3")
    prior = _bundle(tmp_path / "prior/Example Synth.vst3", payload=b"prior")
    _seal_bundle(managed, plugin)
    _seal_bundle(prior, plugin)
    alias = tmp_path / "checkout/plugins/Example Synth.vst3"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(prior.absolute(), target_is_directory=True)
    plugin_runtime.record_managed_alias(alias, prior)
    ownership = alias.parent / ".Example Synth.vst3.synth-setter-managed.json"
    previous_ownership = ownership.read_text()
    assert plugin_manager.validate_plugin_bundle_for_runtime(alias) == prior.resolve(strict=True)
    real_replace = os.replace

    def _fail_alias_replace(source: Path | str, destination: Path | str) -> None:
        if Path(destination) == alias:
            raise OSError(errno.ENOSPC, "no space left on device", str(destination))
        real_replace(source, destination)

    monkeypatch.setattr(plugin_runtime.os, "replace", _fail_alias_replace)

    with pytest.raises(OSError) as excinfo:
        link_plugin(
            plugin,
            artifact_lock=_example_lock(),
            plugins_dir=tmp_path / "managed",
            links_dir=alias.parent,
        )

    assert excinfo.value.errno == errno.ENOSPC
    assert Path(os.readlink(alias)) == prior.absolute()
    assert ownership.read_text() == previous_ownership
    assert plugin_manager.validate_plugin_bundle_for_runtime(alias) == prior.resolve(strict=True)


@pytest.mark.parametrize(
    ("crash_destination", "expected_after_crash"),
    [
        ("transaction", "prior"),
        ("ownership", "prior"),
        ("alias", "next"),
    ],
)
def test_replace_managed_alias_crash_recovers_complete_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_destination: str,
    expected_after_crash: str,
) -> None:
    """A crash at each publication boundary leaves a recoverable prior or next pair.

    :param tmp_path: Scratch root for two sealed bundles and their stable alias.
    :param monkeypatch: Injects a process-style interruption after one atomic replace.
    :param crash_destination: Publication boundary interrupted after replacement.
    :param expected_after_crash: Pair runtime validation must select after interruption.
    """

    class InjectedCrash(BaseException):
        """Simulate termination without entering OSError rollback."""

    prior = _bundle(tmp_path / "prior/Example Synth.vst3", payload=b"prior")
    next_bundle = _bundle(tmp_path / "next/Example Synth.vst3", payload=b"next")
    _seal_bundle(prior)
    _seal_bundle(next_bundle)
    alias = tmp_path / "plugins/Example Synth.vst3"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(prior.absolute(), target_is_directory=True)
    plugin_runtime.record_managed_alias(alias, prior)
    expected_digests = {
        "prior": managed_plugin_digest(prior),
        "next": managed_plugin_digest(next_bundle),
    }
    destinations = {
        "alias": alias,
        "ownership": alias.parent / ".Example Synth.vst3.synth-setter-managed.json",
        "transaction": alias.parent / ".Example Synth.vst3.synth-setter-publication.json",
    }
    real_replace = os.replace

    def _crash_after_replace(source: Path | str, destination: Path | str) -> None:
        real_replace(source, destination)
        if Path(destination) == destinations[crash_destination]:
            raise InjectedCrash

    with monkeypatch.context() as patch:
        patch.setattr(plugin_runtime.os, "replace", _crash_after_replace)
        with pytest.raises(InjectedCrash):
            plugin_runtime.replace_managed_alias(alias, next_bundle)

    assert managed_plugin_digest(alias) == expected_digests[expected_after_crash]
    assert destinations["transaction"].is_file()

    plugin_runtime.replace_managed_alias(alias, next_bundle)

    assert managed_plugin_digest(alias) == expected_digests["next"]
    assert alias.resolve(strict=True) == next_bundle.resolve(strict=True)
    assert not destinations["transaction"].exists()


def test_replace_managed_alias_concurrent_readers_observe_only_complete_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Readers accept A before the alias swap and B after it without transient failure.

    :param tmp_path: Scratch root for two sealed bundles and their stable alias.
    :param monkeypatch: Pauses publication at both atomic phase boundaries.
    """
    prior = _bundle(tmp_path / "prior/Example Synth.vst3", payload=b"prior")
    next_bundle = _bundle(tmp_path / "next/Example Synth.vst3", payload=b"next")
    _seal_bundle(prior)
    _seal_bundle(next_bundle)
    alias = tmp_path / "plugins/Example Synth.vst3"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(prior.absolute(), target_is_directory=True)
    plugin_runtime.record_managed_alias(alias, prior)
    prior_digest = managed_plugin_digest(prior)
    next_digest = managed_plugin_digest(next_bundle)
    ownership = alias.parent / ".Example Synth.vst3.synth-setter-managed.json"
    ownership_published = threading.Event()
    release_ownership = threading.Event()
    alias_published = threading.Event()
    release_alias = threading.Event()
    publisher_errors: list[BaseException] = []
    real_replace = os.replace

    def _pause_after_replace(source: Path | str, destination: Path | str) -> None:
        real_replace(source, destination)
        if Path(destination) == ownership:
            ownership_published.set()
            if not release_ownership.wait(10):
                raise RuntimeError("timed out waiting to release ownership publication")
        elif Path(destination) == alias:
            alias_published.set()
            if not release_alias.wait(10):
                raise RuntimeError("timed out waiting to release alias publication")

    def _publish() -> None:
        try:
            plugin_runtime.replace_managed_alias(alias, next_bundle)
        except BaseException as exc:  # pragma: no cover - asserted in the parent thread
            publisher_errors.append(exc)

    monkeypatch.setattr(plugin_runtime.os, "replace", _pause_after_replace)
    publisher = threading.Thread(target=_publish)
    publisher.start()
    assert ownership_published.wait(10)
    assert {managed_plugin_digest(alias) for _ in range(10)} == {prior_digest}
    release_ownership.set()
    assert alias_published.wait(10)
    assert {managed_plugin_digest(alias) for _ in range(10)} == {next_digest}
    release_alias.set()
    publisher.join(10)

    assert not publisher.is_alive()
    assert publisher_errors == []
    assert managed_plugin_digest(alias) == next_digest


def test_link_plugin_existing_stale_symlink_is_replaced(tmp_path: Path) -> None:
    """Alias refresh replaces only stale symlinks, not real bundles.

    :param tmp_path: Scratch root containing managed and stale bundles.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    managed = _bundle(tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3")
    _seal_bundle(managed, plugin)
    stale = _bundle(tmp_path / "stale/Example Synth.vst3")
    alias = tmp_path / "checkout/plugins/Example Synth.vst3"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(stale, target_is_directory=True)

    first = link_plugin(
        plugin,
        artifact_lock=_example_lock(),
        plugins_dir=tmp_path / "managed",
        links_dir=alias.parent,
    )
    second = link_plugin(
        plugin,
        artifact_lock=_example_lock(),
        plugins_dir=tmp_path / "managed",
        links_dir=alias.parent,
    )

    assert first == second == alias
    assert alias.resolve() == managed.resolve()


def test_adopt_plugin_bundle_missing_source_raises(tmp_path: Path) -> None:
    """Adoption rejects a fallback path that does not exist.

    :param tmp_path: Scratch root without a source bundle.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")

    with pytest.raises(FileNotFoundError, match="fallback bundle is not installed"):
        _adopt_bundle(
            plugin,
            plugins_dir=tmp_path / "managed",
            bundle=tmp_path / "missing.vst3",
        )


@pytest.mark.parametrize("symlink_parent", ["organization", "package", "version"])
def test_adopt_plugin_bundle_symlinked_managed_parent_rejected_without_outside_mutation(
    tmp_path: Path, symlink_parent: str
) -> None:
    """Adoption refuses managed parent chains that escape the storage root.

    :param tmp_path: Scratch root for source, managed, and external trees.
    :param symlink_parent: Managed path component replaced by an escaping symlink.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    bundle = _bundle(tmp_path / "source/Example Synth.vst3")
    managed = tmp_path / "managed"
    external = tmp_path / "outside"
    external.mkdir()
    parents = {
        "organization": managed / "VST3/example",
        "package": managed / "VST3/example/synth",
        "version": managed / "VST3/example/synth/1.2.3",
    }
    escaping_parent = parents[symlink_parent]
    escaping_parent.parent.mkdir(parents=True)
    escaping_parent.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="managed parent symlink"):
        _adopt_bundle(plugin, plugins_dir=managed, bundle=bundle)

    assert list(external.iterdir()) == []


def test_adopt_plugin_bundle_accepts_renderer_version_distinct_from_package_version(
    tmp_path: Path,
) -> None:
    """Adoption validates VST3 identity without changing package-keyed storage.

    :param tmp_path: Scratch root for source and managed bundles.
    """
    manifest_path = _manifest(tmp_path / "studiorack.json")
    payload = json.loads(manifest_path.read_text())
    payload["plugins"]["example/synth"] = "2026.2.0"
    payload["vst3Versions"] = {"example/synth": "0.26.2"}
    manifest_path.write_text(json.dumps(payload))
    plugin = PluginManifest.load(manifest_path).resolve("example/synth")
    bundle = _bundle(tmp_path / "source/Example Synth.vst3", version="0.26.2")

    managed = _adopt_bundle(plugin, plugins_dir=tmp_path / "managed", bundle=bundle)

    assert managed == tmp_path / "managed/VST3/example/synth/2026.2.0/Example Synth.vst3"


def test_adopt_plugin_bundle_wrong_renderer_version_rejected(tmp_path: Path) -> None:
    """Explicit fallback adoption rejects a wrong VST3-reported version.

    :param tmp_path: Scratch root for source and managed bundles.
    """
    manifest_path = _manifest(tmp_path / "studiorack.json")
    payload = json.loads(manifest_path.read_text())
    payload["plugins"]["example/synth"] = "2026.2.0"
    payload["vst3Versions"] = {"example/synth": "0.26.2"}
    manifest_path.write_text(json.dumps(payload))
    plugin = PluginManifest.load(manifest_path).resolve("example/synth")
    bundle = _bundle(tmp_path / "source/Example Synth.vst3", version="9.9.9")

    with pytest.raises(ValueError, match="expected 0.26.2.*found 9.9.9"):
        _adopt_bundle(plugin, plugins_dir=tmp_path / "managed", bundle=bundle)

    assert not (tmp_path / "managed/VST3/example/synth/2026.2.0/Example Synth.vst3").exists()


def test_adopt_plugin_bundle_repeated_call_is_idempotent(tmp_path: Path) -> None:
    """Repeated adoption preserves the exact managed fallback alias.

    :param tmp_path: Scratch root for source and managed bundles.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    bundle = _bundle(tmp_path / "source/Example Synth.vst3")

    first = _adopt_bundle(plugin, plugins_dir=tmp_path / "managed", bundle=bundle)
    second = _adopt_bundle(plugin, plugins_dir=tmp_path / "managed", bundle=bundle)

    assert first == second


def test_adopt_plugin_bundle_dangling_managed_target_uses_conflict_policy(
    tmp_path: Path,
) -> None:
    """A stale managed symlink is a conflict rather than a leaked target-resolution error.

    :param tmp_path: Scratch root for source and dangling managed state.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    replacement = _bundle(tmp_path / "replacement/Example Synth.vst3")
    missing = tmp_path / "deleted/Example Synth.vst3"
    managed = tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3"
    managed.parent.mkdir(parents=True)
    managed.symlink_to(missing, target_is_directory=True)

    with pytest.raises(FileExistsError, match="refusing to replace managed bundle"):
        _adopt_bundle(plugin, plugins_dir=tmp_path / "managed", bundle=replacement)


def test_adopt_plugin_bundle_changed_source_recorded_as_alias_completes_without_deadlock(
    tmp_path: Path,
) -> None:
    """Re-adoption does not reacquire its package lock through source ownership.

    :param tmp_path: Scratch root shared with the isolated re-adoption process.
    """
    manifest_path = _manifest(tmp_path / "studiorack.json")
    plugin = PluginManifest.load(manifest_path).resolve("example/synth")
    source = _bundle(tmp_path / "system-vst3/Example Synth.vst3", payload=b"original")
    managed_root = tmp_path / "managed"
    _adopt_bundle(plugin, plugins_dir=managed_root, bundle=source)
    link_plugin(
        plugin,
        artifact_lock=_example_lock(),
        plugins_dir=managed_root,
        links_dir=source.parent,
    )
    _binary_path(source).write_bytes(_test_binary_magic() + b"changed")

    context = multiprocessing.get_context("spawn")
    result = context.Queue()
    process = context.Process(
        target=_readopt_mutated_source_worker,
        args=((manifest_path, managed_root, source), result),
    )
    process.start()
    process.join(5)
    timed_out = process.is_alive()
    if timed_out:
        process.terminate()
        process.join(5)

    assert not timed_out, "re-adoption deadlocked while reacquiring its package lock"
    assert process.exitcode == 0
    assert result.get(timeout=2) == str(
        managed_root / "VST3/example/synth/1.2.3/Example Synth.vst3"
    )


def test_adopt_plugin_bundle_source_mutation_invalidates_managed_alias(tmp_path: Path) -> None:
    """The seal makes later mutation of adopted symlink content fail closed.

    :param tmp_path: Scratch root for source and managed bundles.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    source = _bundle(tmp_path / "source/Example Synth.vst3", payload=b"original")
    managed = tmp_path / "managed"
    _adopt_bundle(plugin, plugins_dir=managed, bundle=source)
    _binary_path(source).write_bytes(b"mutated")

    with pytest.raises(FileNotFoundError, match="integrity"):
        resolve_plugin_bundle(plugin, managed, artifact_lock=_example_lock())


def test_load_plugin_source_mutation_after_validation_fails_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed external source cannot change between validation and plugin opening.

    :param tmp_path: Scratch root for source and managed state.
    :param monkeypatch: Mutates the real source immediately after real validation.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    source = _bundle(tmp_path / "source/Example Synth.vst3", payload=b"original")
    managed = _adopt_bundle(plugin, plugins_dir=tmp_path / "managed", bundle=source)
    shutil.rmtree(managed.parent / ".synth-setter-runtime-snapshots")
    real_validate = plugin_integrity.ManagedBundleStorage.validate
    opened: list[Path] = []

    def _validate_then_mutate(
        storage: plugin_integrity.ManagedBundleStorage,
    ) -> tuple[Path, plugin_integrity.BundleSeal]:
        validated = real_validate(storage)
        _binary_path(source).write_bytes(_test_binary_magic() + b"changed")
        return validated

    monkeypatch.setattr(plugin_integrity.ManagedBundleStorage, "validate", _validate_then_mutate)
    monkeypatch.setattr(vst_core, "VST3Plugin", lambda path: opened.append(Path(path)))

    with pytest.raises(PluginIntegrityError, match="managed bundle integrity"):
        load_plugin(str(managed))

    assert opened == []


def test_copy_bundle_absolute_internal_symlink_rejected(tmp_path: Path) -> None:
    """Snapshot copy rejects source-bound absolute symlink targets.

    :param tmp_path: Scratch bundle and retained destination directory.
    """
    source = _bundle(tmp_path / "source/Example Synth.vst3")
    link = source / "absolute-link"
    link.symlink_to(_binary_path(source))
    destination = tmp_path / "destination"
    destination.mkdir()
    descriptor = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError, match="absolute bundle symlink"):
            plugin_runtime._copy_symlink_to_descriptor(link, descriptor, link.name)
    finally:
        os.close(descriptor)


def test_adopt_plugin_bundle_absolute_internal_symlink_rejected(tmp_path: Path) -> None:
    """Public adoption rejects a bundle containing an absolute symlink.

    :param tmp_path: Scratch adopted bundle and managed root.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    source = _bundle(tmp_path / "source/Example Synth.vst3")
    (source / "absolute-link").symlink_to(_binary_path(source))

    with pytest.raises(ValueError, match="bundle symlink"):
        plugin_manager.adopt_plugin_bundle(
            plugin,
            plugins_dir=tmp_path / "managed",
            bundle=source,
            locked_package=_example_lock().package_for(plugin),
        )
    assert not (tmp_path / "managed/example/synth/1.2.3/Example Synth.vst3").exists()


def test_runtime_snapshot_candidate_uses_destination_filesystem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Snapshot candidates are staged beneath the retained destination parent.

    :param tmp_path: Scratch adopted source and managed state.
    :param monkeypatch: Rejects use of the process-default temporary filesystem.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    source = _bundle(tmp_path / "source/Example Synth.vst3")
    managed = _adopt_bundle(plugin, plugins_dir=tmp_path / "managed", bundle=source)
    shutil.rmtree(managed.parent / ".synth-setter-runtime-snapshots")
    real_rename = os.rename
    staged_on_destination = False

    def _require_destination_filesystem(
        source_name: str,
        destination_name: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal staged_on_destination
        if source_name.startswith(".Example Synth.vst3.tmp-"):
            assert src_dir_fd is not None
            assert dst_dir_fd is not None
            if os.fstat(src_dir_fd).st_dev != os.fstat(dst_dir_fd).st_dev:
                raise OSError(errno.EXDEV, "candidate uses another filesystem")
            staged_on_destination = True
        real_rename(
            source_name,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(plugin_runtime.os, "rename", _require_destination_filesystem)

    assert plugin_manager.validate_plugin_bundle_for_runtime(managed).is_dir()
    assert staged_on_destination


def test_runtime_snapshot_parent_substitution_publishes_through_retained_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Snapshot publication never mutates a substituted outside directory.

    :param tmp_path: Scratch managed state and outside substitution target.
    :param monkeypatch: Replaces the publication parent after its lock is acquired.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    source = _bundle(tmp_path / "source/Example Synth.vst3")
    managed = _adopt_bundle(plugin, plugins_dir=tmp_path / "managed", bundle=source)
    published = plugin_manager.validate_plugin_bundle_for_runtime(managed)
    publication_parent = published.parent
    shutil.rmtree(published)
    retained_parent = publication_parent.with_name(f"{publication_parent.name}.retained")
    outside = tmp_path / "outside"
    outside.mkdir()

    def _substitute_parent(
        destination: Path,
        seal: plugin_integrity.BundleSeal,
        parent_descriptor: int | None,
    ) -> bool:
        del destination, seal, parent_descriptor
        publication_parent.rename(retained_parent)
        publication_parent.symlink_to(outside, target_is_directory=True)
        return False

    monkeypatch.setattr(plugin_runtime, "_locked_snapshot_matches", _substitute_parent)

    with pytest.raises(PluginIntegrityError, match="failed managed bundle integrity") as exc_info:
        plugin_manager.validate_plugin_bundle_for_runtime(managed)

    assert "publication directory was replaced" in str(exc_info.value.__cause__)
    assert list(outside.iterdir()) == []
    assert (retained_parent / managed.name).is_dir()


def test_load_plugin_symlinked_snapshot_root_rejected_without_outside_write(
    tmp_path: Path,
) -> None:
    """Runtime snapshot publication cannot escape the managed version directory.

    :param tmp_path: Scratch root for source, managed state, and escape target.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    source = _bundle(tmp_path / "source/Example Synth.vst3")
    managed = _adopt_bundle(plugin, plugins_dir=tmp_path / "managed", bundle=source)
    outside = tmp_path / "outside"
    outside.mkdir()
    snapshots = managed.parent / ".synth-setter-runtime-snapshots"
    shutil.rmtree(snapshots)
    snapshots.symlink_to(outside, target_is_directory=True)

    with pytest.raises(PluginIntegrityError, match="managed bundle integrity"):
        load_plugin(str(managed))

    assert list(outside.iterdir()) == []


def test_managed_symlink_source_mutation_rejected_at_public_boundaries(tmp_path: Path) -> None:
    """Direct managed paths revalidate their sibling seal before resolving the source.

    :param tmp_path: Scratch root for source and managed state.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    source = _bundle(tmp_path / "source/Example Synth.vst3", payload=b"original")
    managed = _adopt_bundle(plugin, plugins_dir=tmp_path / "managed", bundle=source)
    _binary_path(source).write_bytes(_test_binary_magic() + b"mutated")

    with pytest.raises(PluginIntegrityError, match="managed bundle integrity"):
        extract_renderer_version(managed)
    with pytest.raises(PluginIntegrityError, match="managed bundle integrity"):
        load_plugin(str(managed))


def test_managed_symlink_seal_deletion_chains_integrity_cause(tmp_path: Path) -> None:
    """A missing managed seal fails both consumers with its filesystem cause preserved.

    :param tmp_path: Scratch root for source and managed state.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    source = _bundle(tmp_path / "source/Example Synth.vst3")
    managed = _adopt_bundle(plugin, plugins_dir=tmp_path / "managed", bundle=source)
    (managed.parent / ".synth-setter-complete.json").unlink()

    with pytest.raises(PluginIntegrityError, match="managed bundle integrity") as excinfo:
        extract_renderer_version(managed)
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)
    assert ".synth-setter-complete.json" in str(excinfo.value.__cause__)
    with pytest.raises(PluginIntegrityError, match="managed bundle integrity"):
        load_plugin(str(managed))


def test_managed_directory_seal_deletion_rejected_at_public_boundaries(
    tmp_path: Path,
) -> None:
    """Durable ownership keeps a directly installed bundle managed without its seal.

    :param tmp_path: Scratch root for directly installed managed state.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    bundle = _bundle(tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3")
    marker = _seal_bundle(bundle, plugin)
    marker.unlink()

    with pytest.raises(PluginIntegrityError, match="managed bundle integrity"):
        extract_renderer_version(bundle)
    with pytest.raises(PluginIntegrityError, match="managed bundle integrity"):
        load_plugin(str(bundle))


def test_managed_alias_source_mutation_rejected_at_public_version_boundary(
    tmp_path: Path,
) -> None:
    """Runtime version extraction revalidates manager-owned alias content.

    :param tmp_path: Scratch root for source, managed state, and stable alias.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    source = _bundle(tmp_path / "source/Example Synth.vst3", payload=b"original")
    managed = tmp_path / "managed"
    _adopt_bundle(plugin, plugins_dir=managed, bundle=source)
    alias = link_plugin(
        plugin,
        artifact_lock=_example_lock(),
        plugins_dir=managed,
        links_dir=tmp_path / "plugins",
    )
    _binary_path(source).write_bytes(b"mutated")

    with pytest.raises(FileNotFoundError, match="managed bundle integrity"):
        extract_renderer_version(alias)
    with pytest.raises(FileNotFoundError, match="managed bundle integrity"):
        load_plugin(str(alias))


def test_plugin_bundle_version_uses_validated_bundle_during_alias_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Version extraction reads validated A after its stable alias switches to B.

    :param tmp_path: Scratch root for two sealed bundles and their stable alias.
    :param monkeypatch: Switches the alias immediately after runtime validation.
    """
    first = _bundle(tmp_path / "first/Example Synth.vst3", version="1.2.3", payload=b"first")
    second = _bundle(tmp_path / "second/Example Synth.vst3", version="9.9.9", payload=b"second")
    _seal_bundle(first)
    _seal_bundle(second)
    alias = tmp_path / "plugins/Example Synth.vst3"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(first.absolute(), target_is_directory=True)
    plugin_runtime.record_managed_alias(alias, first)
    real_lease = plugin_runtime.validated_bundle_lease

    @contextmanager
    def _lease_then_swap(bundle: Path, **_kwargs: object) -> Iterator[Path]:
        with real_lease(bundle) as validated:
            plugin_runtime.replace_managed_alias(alias, second)
            yield validated

    monkeypatch.setattr(plugin_runtime, "validated_bundle_lease", _lease_then_swap)

    assert plugin_runtime.plugin_bundle_version(alias) == "1.2.3"


def test_load_plugin_uses_validated_bundle_during_alias_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VST loading opens validated A after its stable alias switches to B.

    :param tmp_path: Scratch root for two sealed bundles and their stable alias.
    :param monkeypatch: Switches the alias immediately after runtime validation.
    """
    first = _bundle(tmp_path / "first/Example Synth.vst3", payload=b"first")
    second = _bundle(tmp_path / "second/Example Synth.vst3", payload=b"second")
    _seal_bundle(first)
    _seal_bundle(second)
    alias = tmp_path / "plugins/Example Synth.vst3"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(first.absolute(), target_is_directory=True)
    plugin_runtime.record_managed_alias(alias, first)
    real_lease = plugin_runtime.validated_bundle_lease
    loaded_paths: list[Path] = []

    @contextmanager
    def _lease_then_swap(bundle: Path, **_kwargs: object) -> Iterator[Path]:
        with real_lease(bundle) as validated:
            plugin_runtime.replace_managed_alias(alias, second)
            yield validated

    def _load(path: str, plugin_name: str | None = None) -> object:
        loaded_paths.append(Path(path))
        return object()

    monkeypatch.setattr(vst_core, "validated_bundle_lease", _lease_then_swap)
    monkeypatch.setattr("synth_setter.data.vst.core.VST3Plugin", _load)

    load_plugin(str(alias))

    assert loaded_paths == [first.resolve(strict=True)]


def test_managed_alias_retargeted_with_stale_ownership_rejected_at_public_load_boundary(
    tmp_path: Path,
) -> None:
    """Runtime loading rejects an alias no longer pointing at its owned bundle.

    :param tmp_path: Scratch root for managed state, stable alias, and retarget.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    source = _bundle(tmp_path / "source/Example Synth.vst3")
    managed = tmp_path / "managed"
    _adopt_bundle(plugin, plugins_dir=managed, bundle=source)
    alias = link_plugin(
        plugin,
        artifact_lock=_example_lock(),
        plugins_dir=managed,
        links_dir=tmp_path / "plugins",
    )
    replacement = _bundle(tmp_path / "replacement/Example Synth.vst3")
    alias.unlink()
    alias.symlink_to(replacement, target_is_directory=True)

    with pytest.raises(FileNotFoundError, match="managed bundle integrity"):
        load_plugin(str(alias))


def test_managed_alias_seal_deletion_rejected_at_public_version_boundary(
    tmp_path: Path,
) -> None:
    """Stable ownership prevents deleting the target seal from bypassing checks.

    :param tmp_path: Scratch root for source, managed state, and stable alias.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    source = _bundle(tmp_path / "source/Example Synth.vst3")
    managed = tmp_path / "managed"
    managed_bundle = _adopt_bundle(plugin, plugins_dir=managed, bundle=source)
    alias = link_plugin(
        plugin,
        artifact_lock=_example_lock(),
        plugins_dir=managed,
        links_dir=tmp_path / "plugins",
    )
    (managed_bundle.parent / ".synth-setter-complete.json").unlink()

    with pytest.raises(FileNotFoundError, match="managed bundle integrity"):
        extract_renderer_version(alias)


def test_adopt_plugin_bundle_conflicting_managed_path_raises(tmp_path: Path) -> None:
    """Adoption refuses to replace a different managed bundle.

    :param tmp_path: Scratch root containing a managed path conflict.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    bundle = _bundle(tmp_path / "source/Example Synth.vst3")
    managed = tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3"
    managed.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="refusing to replace"):
        _adopt_bundle(plugin, plugins_dir=tmp_path / "managed", bundle=bundle)


def test_link_plugin_existing_real_bundle_refuses_to_overwrite(tmp_path: Path) -> None:
    """Alias refresh refuses to overwrite an unrelated real bundle.

    :param tmp_path: Scratch root containing the conflicting bundle.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))
    plugin = manifest.resolve("example/synth")
    managed = _bundle(tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3")
    _seal_bundle(managed, plugin)
    existing = tmp_path / "checkout/plugins/Example Synth.vst3"
    existing.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="not a symlink"):
        link_plugin(
            plugin,
            artifact_lock=_example_lock(),
            plugins_dir=tmp_path / "managed",
            links_dir=existing.parent,
        )


@pytest.mark.requires_vst
@pytest.mark.slow
def test_link_plugin_sealed_real_bundle_loads_in_pedalboard(tmp_path: Path) -> None:
    """A sealed adopted alias remains consumable by the production VST host.

    :param tmp_path: Scratch root for managed state and the checkout alias.
    """
    plugin = ManagedPlugin(
        package="surge-synthesizer/surge",
        version="1.3.4",
        renderer_version="1.3.4",
        bundle="Surge XT.vst3",
    )
    manifest = PluginManifest.load(PROJECT_ROOT / "studiorack.json")
    artifact_lock = ArtifactLock.load(PROJECT_ROOT / "studiorack.lock.json", manifest)
    _adopt_bundle(
        plugin,
        plugins_dir=tmp_path / "managed",
        bundle=Path(PLUGIN_PATH),
        artifact_lock=artifact_lock,
    )
    alias = link_plugin(
        plugin,
        artifact_lock=artifact_lock,
        plugins_dir=tmp_path / "managed",
        links_dir=tmp_path / "checkout/plugins",
    )

    loaded = load_plugin(str(alias))
    assert loaded.version == "1.3.4"
    assert loaded.parameters  # type: ignore[attr-defined]
