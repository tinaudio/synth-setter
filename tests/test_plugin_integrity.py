"""Contract tests for the focused managed-plugin integrity boundary."""

import errno
import json
import os
import stat
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

import synth_setter.plugin_integrity as plugin_integrity
import synth_setter.plugin_runtime as plugin_runtime
from synth_setter import plugin_manager
from synth_setter.plugin_integrity import (
    ArtifactLock,
    BundleEntry,
    BundleSeal,
    LockedArtifact,
    LockedPackage,
    ManagedBundleRecord,
    ManagedBundleStorage,
    PluginIntegrityError,
    advisory_file_lease,
    bundle_entries,
    bundle_is_sealed,
    locked_package_digest,
    seal_plugin_bundle,
    write_atomic_record,
)
from synth_setter.plugin_manager import PluginManifest, resolve_plugin_bundle
from synth_setter.plugin_runtime import (
    ManagedAliasRecord,
    managed_plugin_digest,
    plugin_bundle_version,
    validate_plugin_bundle_for_runtime,
)
from tests.plugin_manager_test_support import (
    _artifact_lock,
    _binary_path,
    _bundle,
    _cross_platform_lock,
    _example_lock,
    _example_plugin,
    _manifest,
    _pe_test_binary_magic,
    _seal_bundle,
    _test_binary_magic,
)
from tests.plugin_manager_test_support import (
    _platform_binary_environment as _platform_binary_environment,
)


def test_plugin_integrity_public_api_imports_from_focused_module() -> None:
    """Lock and seal APIs are owned by the focused integrity module."""
    assert ArtifactLock.__module__ == "synth_setter.plugin_integrity"
    assert BundleEntry.__module__ == "synth_setter.plugin_integrity"
    assert BundleSeal.__module__ == "synth_setter.plugin_integrity"
    assert LockedArtifact.__module__ == "synth_setter.plugin_integrity"
    assert LockedPackage.__module__ == "synth_setter.plugin_integrity"
    assert ManagedAliasRecord.__module__ == "synth_setter.plugin_runtime"
    assert ManagedBundleRecord.__module__ == "synth_setter.plugin_integrity"
    assert PluginIntegrityError.__module__ == "synth_setter.plugin_integrity"
    assert advisory_file_lease.__module__ == "synth_setter.plugin_integrity"
    assert bundle_entries.__module__ == "synth_setter.plugin_integrity"
    assert bundle_is_sealed.__module__ == "synth_setter.plugin_integrity"
    assert managed_plugin_digest.__module__ == "synth_setter.plugin_runtime"
    assert plugin_bundle_version.__module__ == "synth_setter.plugin_runtime"
    assert seal_plugin_bundle.__module__ == "synth_setter.plugin_integrity"
    assert validate_plugin_bundle_for_runtime.__module__ == "synth_setter.plugin_runtime"


def test_write_atomic_record_privileged_writer_remains_world_readable(tmp_path: Path) -> None:
    """Runtime metadata written by another user remains readable.

    :param tmp_path: Scratch root for one public integrity record.
    """
    record = tmp_path / "record.json"

    write_atomic_record(record, "{}")

    assert stat.S_IMODE(record.stat().st_mode) == 0o644


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory lease semantics")
def test_advisory_file_lease_permission_publication_window_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A consumer retries until the installer's lock directory is traversable.

    :param tmp_path: Scratch root containing one unreadable marker.
    :param monkeypatch: Simulates two permission failures during publication.
    """
    lock_path = tmp_path / "managed/package.lock"
    lock_path.parent.mkdir()
    lock_path.touch(mode=0o444)
    real_open = plugin_integrity._posix_directory_descriptor
    attempts = 0

    def _publishing_open(directory: Path) -> int:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(errno.EACCES, "permissions not published", directory)
        return real_open(directory)

    monkeypatch.setattr(plugin_integrity, "_posix_directory_descriptor", _publishing_open)

    with plugin_integrity.advisory_file_lease(lock_path):
        pass

    assert attempts == 3


def _restricted_package_lock_tree(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    """Build one package-lock tree with privileged-install modes.

    :param tmp_path: Scratch root for the managed hierarchy.
    :returns: Managed root, lock marker, and root-first directory list.
    """
    plugins_dir = tmp_path / "managed"
    lock_path = plugin_integrity.package_install_lock_path("example/synth", "1.2.3", plugins_dir)
    lock_path.parent.mkdir(parents=True)
    directories = [
        plugins_dir,
        plugins_dir / ".synth-setter-install-locks",
        plugins_dir / ".synth-setter-install-locks/example",
        lock_path.parent,
    ]
    for directory in directories:
        directory.chmod(0o700)
    lock_path.touch(mode=0o600)
    return plugins_dir, lock_path, directories


def _assert_runtime_readable_lock_tree(lock_path: Path, directories: list[Path]) -> None:
    """Assert runtime access across one published package-lock hierarchy.

    :param lock_path: Package lock marker expected to be readable.
    :param directories: Hierarchy expected to be traversable.
    """
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o755 for path in directories)
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o644


@pytest.mark.skipif(os.name == "nt", reason="POSIX marker permissions")
def test_advisory_file_lease_unreadable_marker_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A consumer never bypasses an unreadable marker's writer lock.

    :param tmp_path: Scratch directory containing an unreadable marker.
    :param monkeypatch: Makes the denied marker open independent of runner identity.
    """
    lock_path = tmp_path / "managed/package.lock"
    lock_path.parent.mkdir()
    lock_path.touch()
    real_open = os.open

    def _deny_marker(
        path: str | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == lock_path.name:
            raise PermissionError(errno.EACCES, "denied", path)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", _deny_marker)

    with pytest.raises(PermissionError):
        with plugin_integrity.advisory_file_lease(lock_path):
            pass


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory lease semantics")
def test_posix_lease_directory_permanent_permission_failure_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A consumer fails after the permission-publication deadline.

    :param tmp_path: Scratch inaccessible lock directory.
    :param monkeypatch: Makes permission failure and deadline deterministic.
    """
    timestamps = iter([0.0, 11.0])

    def _deny(_path: Path) -> int:
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(plugin_integrity.time, "monotonic", lambda: next(timestamps))
    monkeypatch.setattr(plugin_integrity, "_posix_directory_descriptor", _deny)

    with pytest.raises(PermissionError):
        plugin_integrity._posix_lease_directory_descriptor(tmp_path / "locks")


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock compatibility")
def test_advisory_file_lease_waits_for_marker_only_writer(tmp_path: Path) -> None:
    """A consumer waits for an installer that locks only the marker inode.

    :param tmp_path: Scratch package marker shared across lock implementations.
    """
    import fcntl

    lock_path = tmp_path / "managed/package.lock"
    lock_path.parent.mkdir()
    lock_path.touch()
    acquired = threading.Event()

    def _lease() -> None:
        with plugin_integrity.advisory_file_lease(lock_path):
            acquired.set()

    with lock_path.open("rb") as marker:
        fcntl.flock(marker.fileno(), fcntl.LOCK_EX)
        consumer = threading.Thread(target=_lease)
        consumer.start()
        assert not acquired.wait(0.2)
        fcntl.flock(marker.fileno(), fcntl.LOCK_UN)
    consumer.join(10)

    assert acquired.is_set()


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode publication semantics")
def test_package_install_lock_privileged_writer_keeps_runtime_path_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package locks remain accessible to consumers after a restrictive install.

    :param tmp_path: Scratch root with privileged-install permission modes.
    :param monkeypatch: Observes when the managed root becomes traversable.
    """
    plugins_dir, lock_path, lock_directories = _restricted_package_lock_tree(tmp_path)
    root_published = False
    real_fchmod = os.fchmod

    def _observe_fchmod(descriptor: int, mode: int) -> None:
        nonlocal root_published
        real_fchmod(descriptor, mode)
        opened = os.fstat(descriptor)
        root = plugins_dir.stat()
        if (opened.st_dev, opened.st_ino) == (root.st_dev, root.st_ino):
            root_published = True
            assert stat.S_IMODE(lock_path.stat().st_mode) == 0o644
            assert all(stat.S_IMODE(path.stat().st_mode) == 0o755 for path in lock_directories[1:])

    if os.name != "nt":
        monkeypatch.setattr(os, "fchmod", _observe_fchmod)

    with plugin_integrity.package_install_lock("example/synth", "1.2.3", plugins_dir):
        pass

    assert os.name == "nt" or root_published
    _assert_runtime_readable_lock_tree(lock_path, lock_directories)


@pytest.mark.skipif(os.name != "nt", reason="native Windows handle semantics")
def test_windows_open_regular_lock_reparse_point_rejected(tmp_path: Path) -> None:
    """The retained Windows marker handle never follows a reparse point.

    :param tmp_path: Scratch marker and external target.
    """
    external = tmp_path / "external.lock"
    external.write_text("unchanged")
    marker = tmp_path / "marker.lock"
    marker.symlink_to(external)

    with pytest.raises(FileExistsError, match="reparse point"):
        plugin_integrity._windows_open_regular_lock(marker)

    assert external.read_text() == "unchanged"


@pytest.mark.skipif(os.name != "nt", reason="native Windows locking semantics")
def test_package_install_lock_windows_fresh_marker_is_runtime_leasable(tmp_path: Path) -> None:
    """A fresh Windows package marker supports a subsequent runtime lease.

    :param tmp_path: Scratch managed storage root.
    """
    plugins_dir = tmp_path / "managed"
    with plugin_integrity.package_install_lock("example/synth", "1.2.3", plugins_dir):
        pass
    lock_path = plugin_integrity.package_install_lock_path("example/synth", "1.2.3", plugins_dir)

    with plugin_integrity.advisory_file_lease(lock_path):
        pass


def test_windows_lock_directories_real_hierarchy_returns_root_first(tmp_path: Path) -> None:
    """Windows fallback creates each package-lock directory without symlinks.

    :param tmp_path: Scratch root for a new hierarchy.
    """
    plugins_dir = tmp_path / "managed"
    lock_parent = plugins_dir / ".synth-setter-install-locks/example/synth"

    directories = plugin_integrity._windows_lock_directories(plugins_dir, lock_parent)

    assert directories == [
        plugins_dir,
        plugins_dir / ".synth-setter-install-locks",
        plugins_dir / ".synth-setter-install-locks/example",
        lock_parent,
    ]
    assert all(directory.is_dir() for directory in directories)


def test_windows_lock_directories_symlink_rejected_before_target_mutation(tmp_path: Path) -> None:
    """Windows fallback rejects each symlink before descending through it.

    :param tmp_path: Scratch root containing one external directory target.
    """
    plugins_dir = tmp_path / "managed"
    plugins_dir.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (plugins_dir / ".synth-setter-install-locks").symlink_to(external, target_is_directory=True)

    with pytest.raises(FileExistsError, match="not a directory"):
        plugin_integrity._windows_lock_directories(
            plugins_dir,
            plugins_dir / ".synth-setter-install-locks/example/synth",
        )

    assert list(external.iterdir()) == []


def test_package_install_lock_symlinked_hierarchy_rejected_without_target_mutation(
    tmp_path: Path,
) -> None:
    """Installer lock setup never follows a managed hierarchy symlink.

    :param tmp_path: Scratch root containing a symlink to an external directory.
    """
    plugins_dir = tmp_path / "managed"
    plugins_dir.mkdir()
    external = tmp_path / "external"
    external.mkdir(mode=0o700)
    (plugins_dir / ".synth-setter-install-locks").symlink_to(external, target_is_directory=True)

    with pytest.raises(FileExistsError, match="not a directory"):
        with plugin_integrity.package_install_lock("example/synth", "1.2.3", plugins_dir):
            pass

    assert stat.S_IMODE(external.stat().st_mode) == 0o700
    assert list(external.iterdir()) == []


def test_package_install_lock_windows_symlinked_marker_rejected_without_target_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows rejects a marker symlink before opening or chmodding its target.

    :param tmp_path: Scratch package-lock hierarchy and external target.
    :param monkeypatch: Selects the Windows package-lock branch.
    """
    plugins_dir = tmp_path / "managed"
    lock_path = plugin_integrity.package_install_lock_path("example/synth", "1.2.3", plugins_dir)
    lock_path.parent.mkdir(parents=True)
    external = tmp_path / "external.lock"
    external.write_text("unchanged")
    external.chmod(0o600)
    lock_path.symlink_to(external)
    monkeypatch.setattr(os, "name", "nt")

    with pytest.raises(FileExistsError, match="not a regular file"):
        with plugin_integrity.package_install_lock("example/synth", "1.2.3", plugins_dir):
            pass

    assert external.read_text() == "unchanged"
    assert stat.S_IMODE(external.stat().st_mode) == 0o600


def test_package_install_lock_racing_symlink_rejected_without_target_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hierarchy component replaced after creation is never followed.

    :param tmp_path: Scratch managed root and external symlink target.
    :param monkeypatch: Replaces the first lock directory before descriptor open.
    """
    plugins_dir = tmp_path / "managed"
    plugins_dir.mkdir(mode=0o700)
    external = tmp_path / "external"
    external.mkdir(mode=0o700)
    real_mkdir = os.mkdir

    def _swap_after_mkdir(path: str, mode: int = 0o777, *, dir_fd: int | None = None) -> None:
        real_mkdir(path, mode, dir_fd=dir_fd)
        if path == ".synth-setter-install-locks":
            created = plugins_dir / path
            created.rename(plugins_dir / f"{path}.replaced")
            created.symlink_to(external, target_is_directory=True)

    monkeypatch.setattr(os, "mkdir", _swap_after_mkdir)

    with pytest.raises(FileExistsError, match="not a directory"):
        with plugin_integrity.package_install_lock("example/synth", "1.2.3", plugins_dir):
            pass

    assert stat.S_IMODE(external.stat().st_mode) == 0o700
    assert list(external.iterdir()) == []


def test_managed_bundle_storage_validates_and_discards_integrity_records(
    tmp_path: Path,
) -> None:
    """The public storage facade owns managed bundle record access.

    :param tmp_path: Scratch root for one sealed managed bundle.
    """
    plugin = _example_plugin()
    bundle = _bundle(tmp_path / "Example Synth.vst3")
    _seal_bundle(bundle, plugin)
    storage = ManagedBundleStorage(bundle)

    ownership = storage.read_ownership()
    resolved, seal = storage.validate()

    assert ownership is not None
    assert ownership.package == plugin.package
    assert resolved == bundle.resolve(strict=True)
    assert seal.package_reference == plugin.reference
    assert storage.has_integrity_record()

    storage.discard()

    assert storage.read_ownership() is None
    assert not storage.has_integrity_record()


def test_managed_alias_paths_are_owned_by_plugin_runtime(tmp_path: Path) -> None:
    """One runtime API owns alias ownership and transaction filenames.

    :param tmp_path: Scratch root for a representative stable alias.
    """
    alias = tmp_path / "plugins/Example Synth.vst3"

    ownership, transaction = plugin_runtime.managed_alias_paths(alias)

    assert ownership == alias.parent / ".Example Synth.vst3.synth-setter-managed.json"
    assert transaction == alias.parent / ".Example Synth.vst3.synth-setter-publication.json"
    assert not hasattr(plugin_integrity, "_alias_record_path")
    assert not hasattr(plugin_integrity, "_alias_transaction_path")


def test_plugin_manager_compatibility_reexports_integrity_public_api() -> None:
    """Existing public plugin_manager imports retain object identity."""
    assert plugin_manager.ArtifactLock is ArtifactLock
    assert plugin_manager.ManagedBundleRecord is ManagedBundleRecord
    assert plugin_manager.PluginIntegrityError is PluginIntegrityError
    assert plugin_manager.managed_plugin_digest is managed_plugin_digest
    assert plugin_manager.plugin_bundle_version is plugin_bundle_version
    assert plugin_manager.seal_plugin_bundle is seal_plugin_bundle
    assert plugin_manager.validate_plugin_bundle_for_runtime is validate_plugin_bundle_for_runtime


def test_locked_package_digest_is_canonical_across_artifact_and_selector_order() -> None:
    """Equivalent lock JSON ordering produces one package identity."""
    first = LockedPackage.model_validate(
        {
            "artifacts": [
                {
                    "architectures": ["x64", "arm64"],
                    "sha256": "a" * 64,
                    "systems": ["linux", "mac"],
                    "type": "archive",
                    "url": "https://example.test/a.zip",
                },
                {
                    "architectures": ["arm64"],
                    "sha256": "b" * 64,
                    "systems": ["mac"],
                    "type": "installer",
                    "url": "https://example.test/b.pkg",
                },
            ]
        }
    )
    second_payload = first.model_dump(mode="json")
    second_payload["artifacts"].reverse()
    second_payload["artifacts"][1]["architectures"].reverse()
    second_payload["artifacts"][1]["systems"].reverse()
    second = LockedPackage.model_validate(second_payload)

    assert locked_package_digest("example/synth@1.2.3", first) == locked_package_digest(
        "example/synth@1.2.3", second
    )


def test_managed_plugin_digest_cross_platform_contents_share_package_identity(
    tmp_path: Path,
) -> None:
    """Host-specific bytes under one exact lock produce one dataset identity.

    :param tmp_path: Scratch root for platform-specific bundle contents.
    """
    artifact_lock = _cross_platform_lock()
    plugin = _example_plugin()
    mac_bundle = _bundle(tmp_path / "mac/Example Synth.vst3", payload=b"mac-arm64")
    linux_bundle = _bundle(tmp_path / "linux/Example Synth.vst3", payload=b"linux-x64")
    _seal_bundle(mac_bundle, plugin, artifact_lock)
    _seal_bundle(linux_bundle, plugin, artifact_lock)

    assert managed_plugin_digest(mac_bundle) == managed_plugin_digest(linux_bundle)


def test_managed_plugin_digest_lock_rotation_changes_package_identity(tmp_path: Path) -> None:
    """Any repository lock rotation changes the dataset identity.

    :param tmp_path: Scratch root for bundles sealed under two lock revisions.
    """
    plugin = _example_plugin()
    first_bundle = _bundle(tmp_path / "first/Example Synth.vst3")
    second_bundle = _bundle(tmp_path / "second/Example Synth.vst3")
    _seal_bundle(first_bundle, plugin, _cross_platform_lock(linux_sha256="a" * 64))
    _seal_bundle(second_bundle, plugin, _cross_platform_lock(linux_sha256="c" * 64))

    assert managed_plugin_digest(first_bundle) != managed_plugin_digest(second_bundle)


def test_managed_plugin_digest_local_content_mutation_fails(tmp_path: Path) -> None:
    """Package provenance never substitutes for local byte validation.

    :param tmp_path: Scratch root for the mutated managed bundle.
    """
    plugin_bundle = _bundle(tmp_path / "Example Synth.vst3", payload=b"original")
    _seal_bundle(plugin_bundle, _example_plugin(), _cross_platform_lock())
    _binary_path(plugin_bundle).write_bytes(_test_binary_magic() + b"mutated")

    with pytest.raises(PluginIntegrityError, match="managed bundle integrity"):
        managed_plugin_digest(plugin_bundle)


def test_adopt_plugin_bundle_records_distinct_explicit_source_identity(tmp_path: Path) -> None:
    """Explicit adoption combines its package lock with sealed source content.

    :param tmp_path: Scratch root for explicit and artifact-backed bundles.
    """
    artifact_lock = _cross_platform_lock()
    plugin = _example_plugin()
    source = _bundle(tmp_path / "source/Example Synth.vst3")

    managed = plugin_manager.adopt_plugin_bundle(
        plugin,
        plugins_dir=tmp_path / "managed",
        bundle=source,
        locked_package=artifact_lock.package_for(plugin),
    )
    registry_bundle = _bundle(tmp_path / "linux/Example Synth.vst3")
    _seal_bundle(registry_bundle, plugin, artifact_lock)

    seal = BundleSeal.model_validate_json(
        (managed.parent / ".synth-setter-complete.json").read_text()
    )
    assert seal.source_kind == "explicit"
    assert managed_plugin_digest(managed) != managed_plugin_digest(registry_bundle)


def test_adopt_plugin_bundle_distinct_source_trees_have_distinct_identities(
    tmp_path: Path,
) -> None:
    """Two explicit builds under one package lock cannot share provenance.

    :param tmp_path: Scratch root for two adopted source trees.
    """
    plugin = _example_plugin()
    artifact_lock = _cross_platform_lock()
    first = plugin_manager.adopt_plugin_bundle(
        plugin,
        plugins_dir=tmp_path / "first-managed",
        bundle=_bundle(tmp_path / "first/Example Synth.vst3", payload=b"first-build"),
        locked_package=artifact_lock.package_for(plugin),
    )
    second = plugin_manager.adopt_plugin_bundle(
        plugin,
        plugins_dir=tmp_path / "second-managed",
        bundle=_bundle(tmp_path / "second/Example Synth.vst3", payload=b"second-build"),
        locked_package=artifact_lock.package_for(plugin),
    )

    assert managed_plugin_digest(first) != managed_plugin_digest(second)


def test_adopt_plugin_bundle_lock_rotation_changes_explicit_identity(tmp_path: Path) -> None:
    """An explicit build retains source identity while package-lock rotation changes its pin.

    :param tmp_path: Scratch root for the adopted source and two managed roots.
    """
    plugin = _example_plugin()
    source = _bundle(tmp_path / "source/Example Synth.vst3", payload=b"same-build")
    first_lock = _cross_platform_lock(linux_sha256="a" * 64)
    second_lock = _cross_platform_lock(linux_sha256="c" * 64)
    first = plugin_manager.adopt_plugin_bundle(
        plugin,
        plugins_dir=tmp_path / "first-managed",
        bundle=source,
        locked_package=first_lock.package_for(plugin),
    )
    second = plugin_manager.adopt_plugin_bundle(
        plugin,
        plugins_dir=tmp_path / "second-managed",
        bundle=source,
        locked_package=second_lock.package_for(plugin),
    )

    assert managed_plugin_digest(first) != managed_plugin_digest(second)


def test_artifact_lock_load_exact_manifest_coverage_returns_lock(tmp_path: Path) -> None:
    """The artifact lock covers every manifest package at its exact version.

    :param tmp_path: Scratch root for repository trust-boundary files.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))

    lock = ArtifactLock.load(_artifact_lock(tmp_path / "studiorack.lock.json"), manifest)

    assert list(lock.root) == ["example/synth@1.2.3"]


def test_artifact_lock_load_unknown_artifact_field_rejected(tmp_path: Path) -> None:
    """Artifact-lock schema rejects registry fields outside trusted identity.

    :param tmp_path: Scratch root for repository trust-boundary files.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))
    lock_path = _artifact_lock(tmp_path / "studiorack.lock.json")
    payload = json.loads(lock_path.read_text())
    payload["example/synth@1.2.3"]["artifacts"][0]["size"] = 123
    lock_path.write_text(json.dumps(payload))

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ArtifactLock.load(lock_path, manifest)


def test_artifact_lock_load_http_artifact_url_rejected(tmp_path: Path) -> None:
    """Repository locks cannot downgrade artifact transport from HTTPS.

    :param tmp_path: Scratch root for repository trust-boundary files.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))
    lock_path = _artifact_lock(tmp_path / "studiorack.lock.json")
    payload = json.loads(lock_path.read_text())
    payload["example/synth@1.2.3"]["artifacts"][0]["url"] = "http://example.test/synth.zip"
    lock_path.write_text(json.dumps(payload))

    with pytest.raises(ValidationError, match="artifact URL must use HTTPS"):
        ArtifactLock.load(lock_path, manifest)


def test_artifact_lock_load_missing_manifest_reference_rejected(tmp_path: Path) -> None:
    """A lock cannot omit a manifest package version.

    :param tmp_path: Scratch root for repository trust-boundary files.
    """
    manifest = PluginManifest.load(_manifest(tmp_path / "studiorack.json"))
    lock_path = tmp_path / "studiorack.lock.json"
    lock_path.write_text("{}")

    with pytest.raises(ValueError, match="exactly cover"):
        ArtifactLock.load(lock_path, manifest)


def test_resolve_plugin_bundle_markerless_bundle_rejected(tmp_path: Path) -> None:
    """A realistic bundle without a completion record is not installed.

    :param tmp_path: Scratch root for markerless managed state.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    _bundle(tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3")

    with pytest.raises(FileNotFoundError, match="integrity"):
        resolve_plugin_bundle(plugin, tmp_path / "managed", artifact_lock=_example_lock())


@pytest.mark.parametrize(
    ("host_platform", "machine", "relative_binary", "signature"),
    [
        ("linux", "x86_64", "Contents/x86_64-linux/Example Synth.so", b"\x7fELF"),
        ("darwin", "arm64", "Contents/MacOS/Example Synth", b"\xcf\xfa\xed\xfe"),
        (
            "win32",
            "AMD64",
            "Contents/x86_64-win/Example Synth.vst3",
            _pe_test_binary_magic(),
        ),
    ],
)
def test_seal_plugin_bundle_platform_binary_signature_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host_platform: str,
    machine: str,
    relative_binary: str,
    signature: bytes,
) -> None:
    """Platform VST3 locations require their native executable signature.

    :param tmp_path: Scratch root for one platform-shaped bundle.
    :param monkeypatch: Selects the platform and architecture under validation.
    :param host_platform: Platform selector used by the validator.
    :param machine: Host architecture reported to the validator.
    :param relative_binary: Platform VST3 executable location.
    :param signature: Minimal native executable header.
    """
    monkeypatch.setattr(plugin_integrity.sys, "platform", host_platform)
    monkeypatch.setattr(plugin_integrity.platform, "machine", lambda: machine)
    bundle = tmp_path / "Example Synth.vst3"
    binary = bundle / relative_binary
    binary.parent.mkdir(parents=True)
    binary.write_bytes(signature + b"plugin")
    (bundle / "Contents/moduleinfo.json").write_text('{"Version":"1.2.3"}')

    marker = _seal_bundle(bundle)

    assert marker.is_file()


def test_seal_plugin_bundle_empty_bundle_rejected(tmp_path: Path) -> None:
    """An empty VST3 directory cannot receive a completion record.

    :param tmp_path: Scratch root for empty managed state.
    """
    bundle = tmp_path / "Example Synth.vst3"
    bundle.mkdir()

    with pytest.raises(ValueError, match="platform VST3 binary"):
        _seal_bundle(bundle)


def test_seal_plugin_bundle_escaping_symlink_rejected(tmp_path: Path) -> None:
    """A bundle cannot seal a symlink whose target escapes the bundle.

    :param tmp_path: Scratch root for bundle and external content.
    """
    bundle = _bundle(tmp_path / "Example Synth.vst3")
    external = tmp_path / "external.txt"
    external.write_text("outside")
    (bundle / "Contents/external.txt").symlink_to(external)

    with pytest.raises(ValueError, match="escapes"):
        _seal_bundle(bundle)


def test_seal_plugin_bundle_records_exact_package_lock_provenance(tmp_path: Path) -> None:
    """Registry seals bind content to the canonical exact-package lock digest.

    :param tmp_path: Scratch root for managed bundle state.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    bundle = _bundle(tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3")

    marker = _seal_bundle(bundle, plugin)

    seal = json.loads(marker.read_text())
    ownership = json.loads((bundle.parent / ".synth-setter-managed.json").read_text())
    assert ownership == {
        "bundle": "Example Synth.vst3",
        "package": "example/synth",
        "schema": 1,
        "source_kind": "artifact-lock",
        "version": "1.2.3",
    }
    assert seal["package_reference"] == "example/synth@1.2.3"
    assert seal["source_kind"] == "artifact-lock"
    assert seal["locked_package_sha256"] == (
        "ce68cc987663810c48dcd1de66953f8f278a569dc4aef0747a68b18044502c46"
    )


def test_resolve_plugin_bundle_modified_after_seal_rejected(tmp_path: Path) -> None:
    """Content drift invalidates an otherwise complete managed bundle.

    :param tmp_path: Scratch root for sealed managed state.
    """
    plugin = PluginManifest.load(_manifest(tmp_path / "studiorack.json")).resolve("example/synth")
    bundle = _bundle(tmp_path / "managed/VST3/example/synth/1.2.3/Example Synth.vst3")
    _seal_bundle(bundle, plugin)
    _binary_path(bundle).write_bytes(b"modified")

    with pytest.raises(FileNotFoundError, match="integrity"):
        resolve_plugin_bundle(plugin, tmp_path / "managed", artifact_lock=_example_lock())
