"""Publish managed aliases and lease validated VST3 bundles at runtime.

Typical usage holds validation through plugin construction::

    from pathlib import Path
    from pedalboard import VST3Plugin
    from synth_setter.plugin_runtime import validated_bundle_lease

    with validated_bundle_lease(Path("plugins/Cardinal.vst3")) as bundle:
        plugin = VST3Plugin(str(bundle))
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import plistlib
import shutil
import stat
import sys
import tempfile
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Literal

import pydantic

import synth_setter.plugin_integrity as integrity
from synth_setter.plugin_integrity import BundleSeal, ManagedBundleRecord, PluginIntegrityError

logger = logging.getLogger(__name__)

__all__ = [
    "ManagedAliasRecord",
    "discard_managed_bundle_records",
    "managed_alias_paths",
    "managed_alias_target",
    "managed_plugin_digest",
    "plugin_bundle_version",
    "record_managed_alias",
    "replace_managed_alias",
    "validate_plugin_bundle_for_runtime",
    "validated_bundle_lease",
]


class ManagedAliasRecord(pydantic.BaseModel):
    """Stable alias ownership pointing back to its managed bundle.

    .. attribute :: model_config

        Pydantic validation settings.

    .. attribute :: managed_bundle

        Absolute versioned bundle path governed by the alias.

    .. attribute :: schema_version

        Ownership-record schema version.
    """

    model_config = pydantic.ConfigDict(
        strict=True, extra="forbid", frozen=True, populate_by_name=True
    )

    managed_bundle: str
    schema_version: Literal[1] = pydantic.Field(default=1, alias="schema")

    @pydantic.field_validator("managed_bundle")
    @classmethod
    def _require_absolute_path(cls, value: str) -> str:
        """Reject ownership records whose meaning depends on process cwd.

        :param value: Candidate managed bundle path.
        :returns: Validated absolute path.
        :raises ValueError: The path is relative.
        """
        if not Path(value).is_absolute():
            raise ValueError("managed alias record path must be absolute")
        return value


class _AliasPublicationTransaction(pydantic.BaseModel):
    """Bridge alias publication across its two atomic replacements.

    .. attribute :: model_config

        Pydantic validation settings.

    .. attribute :: alias

        Absolute stable alias path.

    .. attribute :: next_ownership

        Ownership published before the alias swap.

    .. attribute :: prior_ownership

        Ownership valid before publication, if any.

    .. attribute :: prior_target

        Absolute lexical alias target valid before publication, if any.

    .. attribute :: schema_version

        Publication transaction schema version.
    """

    model_config = pydantic.ConfigDict(
        strict=True, extra="forbid", frozen=True, populate_by_name=True
    )

    alias: str
    next_ownership: ManagedAliasRecord
    prior_ownership: ManagedAliasRecord | None
    prior_target: str | None
    schema_version: Literal[1] = pydantic.Field(default=1, alias="schema")

    @pydantic.field_validator("alias")
    @classmethod
    def _require_absolute_alias(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("alias publication path must be absolute")
        return value

    @pydantic.field_validator("prior_target")
    @classmethod
    def _require_absolute_prior_target(cls, value: str | None) -> str | None:
        if value is not None and not Path(value).is_absolute():
            raise ValueError("alias publication prior target must be absolute")
        return value

    @pydantic.model_validator(mode="after")
    def _require_target_for_prior_ownership(self) -> _AliasPublicationTransaction:
        if self.prior_ownership is not None and self.prior_target is None:
            raise ValueError("alias publication prior ownership requires its target")
        return self


def managed_alias_paths(alias: Path) -> tuple[Path, Path]:
    """Return ownership and publication-transaction paths for one stable alias.

    :param alias: Stable consumer-facing or adopted alias.
    :returns: Ownership-record path followed by publication-transaction path.
    """
    ownership = alias.parent / f".{alias.name}.synth-setter-managed.json"
    transaction = alias.parent / f".{alias.name}.synth-setter-publication.json"
    return ownership, transaction


def record_managed_alias(alias: Path, managed_bundle: Path) -> None:
    """Atomically bind an alias ownership record to its managed target.

    :param alias: Consumer-facing or adopted symlink path.
    :param managed_bundle: Managed bundle governed by the alias.
    """
    record = ManagedAliasRecord(managed_bundle=str(managed_bundle.absolute()))
    serialized = record.model_dump_json(indent=2, by_alias=True)
    ownership_path, _ = managed_alias_paths(alias)
    integrity.write_atomic_record(ownership_path, serialized)


def discard_managed_bundle_records(bundle: Path) -> None:
    """Remove manager records for an unpublished bundle path.

    :param bundle: Bundle whose failed publication is being rolled back.
    """
    ownership_path, _ = managed_alias_paths(bundle)
    ownership_path.unlink(missing_ok=True)
    integrity.ManagedBundleStorage(bundle).discard()


def _optional_regular_text(path: Path, description: str) -> str | None:
    """Read an optional regular file without following its final path component.

    :param path: Candidate metadata path.
    :param description: Record description used in rejection messages.
    :returns: UTF-8 contents, or ``None`` when the path is absent.
    :raises FileExistsError: The path is a symlink or another non-regular file.
    :raises OSError: Opening or inspecting the path fails for another reason.
    """
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise FileExistsError(f"{description} is not a regular file: {path}") from exc
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise FileExistsError(f"{description} is not a regular file: {path}")
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            descriptor = -1
            return stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _existing_alias_record(alias: Path) -> ManagedAliasRecord | None:
    """Load an alias ownership record when present.

    :param alias: Stable alias whose sidecar is read.
    :returns: Validated ownership, or ``None`` when no sidecar exists.
    """
    ownership_path, _ = managed_alias_paths(alias)
    serialized = _optional_regular_text(ownership_path, "managed alias ownership")
    if serialized is None:
        return None
    return ManagedAliasRecord.model_validate_json(serialized)


def _restore_alias_record(alias: Path, record: ManagedAliasRecord | None) -> None:
    """Restore or remove the ownership sidecar for an alias.

    :param alias: Stable alias whose sidecar is restored.
    :param record: Prior ownership, or ``None`` when no sidecar should remain.
    """
    path, _ = managed_alias_paths(alias)
    if record is None:
        path.unlink(missing_ok=True)
        return
    integrity.write_atomic_record(path, record.model_dump_json(indent=2, by_alias=True))


def managed_alias_target(path: Path) -> Path | None:
    """Return a symlink's absolute or cwd-independent lexical target.

    :param path: Candidate symlink.
    :returns: Lexical target path, or ``None`` for an absent non-symlink.
    """
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return None
    if not stat.S_ISLNK(mode):
        return None
    target = Path(os.readlink(path))
    return target if target.is_absolute() else path.parent / target


def _absolute_alias_target(alias: Path) -> Path | None:
    target = managed_alias_target(alias)
    return None if target is None else Path(os.path.abspath(target))


def _target_matches_record(target: Path | None, record: ManagedAliasRecord) -> bool:
    if target is None:
        return False
    recorded = Path(record.managed_bundle)
    if Path(os.path.abspath(target)) == recorded.absolute():
        return True
    try:
        return target.resolve(strict=True) == recorded.resolve(strict=True)
    except FileNotFoundError:
        return False


def _read_alias_transaction(alias: Path) -> _AliasPublicationTransaction | None:
    _, transaction_path = managed_alias_paths(alias)
    serialized = _optional_regular_text(
        transaction_path,
        "managed alias publication transaction",
    )
    if serialized is None:
        return None
    transaction = _AliasPublicationTransaction.model_validate_json(serialized)
    if Path(transaction.alias) != alias.absolute():
        raise ValueError("managed alias publication transaction belongs to another alias")
    return transaction


def _transaction_target_matches(target: Path | None, expected: str | None) -> bool:
    return target is None if expected is None else target == Path(expected)


def _recover_alias_publication(alias: Path) -> None:
    transaction = _read_alias_transaction(alias)
    if transaction is None:
        return
    target = _absolute_alias_target(alias)
    if _target_matches_record(target, transaction.next_ownership):
        recovered = transaction.next_ownership
    elif _transaction_target_matches(target, transaction.prior_target):
        recovered = transaction.prior_ownership
    else:
        raise ValueError("managed alias target does not match its publication transaction")
    _restore_alias_record(alias, recovered)
    _, transaction_path = managed_alias_paths(alias)
    transaction_path.unlink()


def _alias_publication_transaction(
    alias: Path,
    managed_bundle: Path,
) -> _AliasPublicationTransaction:
    prior_ownership = _existing_alias_record(alias)
    prior_target = _absolute_alias_target(alias)
    if prior_ownership is not None and not _target_matches_record(prior_target, prior_ownership):
        raise ValueError("managed alias target does not match its ownership record")
    return _AliasPublicationTransaction(
        alias=str(alias.absolute()),
        next_ownership=ManagedAliasRecord(managed_bundle=str(managed_bundle.absolute())),
        prior_ownership=prior_ownership,
        prior_target=None if prior_target is None else str(prior_target),
    )


def replace_managed_alias(alias: Path, managed_bundle: Path) -> None:
    """Publish ownership then alias with crash recovery.

    :param alias: Stable consumer-facing symlink path.
    :param managed_bundle: Validated managed bundle target.
    :raises OSError: Transaction, ownership, or symlink publication fails.
    """
    _recover_alias_publication(alias)
    transaction = _alias_publication_transaction(alias, managed_bundle)
    _, transaction_path = managed_alias_paths(alias)
    integrity.write_atomic_record(
        transaction_path,
        transaction.model_dump_json(indent=2, by_alias=True),
    )
    temporary_directory = Path(tempfile.mkdtemp(dir=alias.parent))
    temporary = temporary_directory / ".candidate"
    try:
        temporary.symlink_to(managed_bundle.absolute(), target_is_directory=True)
        try:
            _restore_alias_record(alias, transaction.next_ownership)
            os.replace(temporary, alias)
        except OSError:
            _recover_alias_publication(alias)
            raise
        transaction_path.unlink()
    finally:
        if temporary.is_symlink():
            temporary.unlink()
        try:
            temporary_directory.rmdir()
        except OSError:
            pass


def _managed_storage_identity(managed: Path) -> tuple[Path, str, str] | None:
    """Parse a canonical versioned managed-bundle path.

    :param managed: Candidate managed bundle path.
    :returns: Storage root, package, and version, or ``None`` for another layout.
    """
    absolute = managed.absolute()
    if len(absolute.parents) < 5 or absolute.parents[3].name != "VST3":
        return None
    root = absolute.parents[4]
    package = f"{absolute.parents[2].name}/{absolute.parents[1].name}"
    version = absolute.parents[0].name
    expected = root / "VST3" / package / version / absolute.name
    return (root, package, version) if absolute == expected else None


def _managed_from_publication(
    target: Path | None,
    ownership: ManagedAliasRecord | None,
    transaction: _AliasPublicationTransaction,
) -> Path | None:
    """Recover the managed target represented by a publication transaction.

    :param target: Current alias target, if present.
    :param ownership: Current ownership record, if present.
    :param transaction: Interrupted publication state.
    :returns: Recoverable managed target, or ``None`` when neither state matches.
    :raises ValueError: Ownership is unrelated to the publication transaction.
    """
    allowed_ownership = (transaction.next_ownership, transaction.prior_ownership)
    if ownership not in allowed_ownership:
        raise ValueError("managed alias ownership does not match its publication transaction")
    if _target_matches_record(target, transaction.next_ownership):
        return Path(transaction.next_ownership.managed_bundle)
    prior = transaction.prior_ownership
    if (
        prior is not None
        and _transaction_target_matches(target, transaction.prior_target)
        and _target_matches_record(target, prior)
    ):
        return Path(prior.managed_bundle)
    return None


def _stable_alias_snapshot(
    alias: Path,
) -> tuple[ManagedAliasRecord | None, Path | None, _AliasPublicationTransaction | None] | None:
    """Sample alias ownership and target across one stable transaction read.

    :param alias: Consumer-facing alias under concurrent publication.
    :returns: Ownership, target, and transaction, or ``None`` when publication changed.
    :raises ValueError: A stable transaction names another alias.
    """
    _, transaction_path = managed_alias_paths(alias)
    transaction_before = _optional_regular_text(
        transaction_path,
        "managed alias publication transaction",
    )
    ownership = _existing_alias_record(alias)
    target = _absolute_alias_target(alias)
    transaction_after = _optional_regular_text(
        transaction_path,
        "managed alias publication transaction",
    )
    if transaction_before != transaction_after:
        return None
    transaction = (
        None
        if transaction_before is None
        else _AliasPublicationTransaction.model_validate_json(transaction_before)
    )
    if transaction is not None and Path(transaction.alias) != alias.absolute():
        raise ValueError("managed alias publication transaction belongs to another alias")
    return ownership, target, transaction


def _adjudicate_alias_snapshot(
    alias: Path,
    ownership: ManagedAliasRecord | None,
    target: Path | None,
    transaction: _AliasPublicationTransaction | None,
) -> tuple[bool, Path | None]:
    """Resolve a sampled alias state to a final managed identity.

    :param alias: Consumer-facing alias represented by the sample.
    :param ownership: Stable ownership record, if present.
    :param target: Stable lexical alias target, if present.
    :param transaction: Stable in-progress publication, if present.
    :returns: Final-state flag and managed bundle, where ``None`` may mean unmanaged.
    """
    if ownership is not None:
        managed = Path(ownership.managed_bundle)
        if managed == alias.absolute():
            return True, alias
        if target is None and alias.resolve(strict=True) == managed.resolve(strict=True):
            return True, managed
        if _target_matches_record(target, ownership):
            return True, managed
    if transaction is not None:
        published = _managed_from_publication(target, ownership, transaction)
        if published is not None:
            return True, published
    if ownership is None and transaction is None:
        return True, None
    return False, None


def _runtime_alias_managed_bundle(alias: Path) -> Path | None:
    mismatch: ValueError | None = None
    for _ in range(4):
        snapshot = _stable_alias_snapshot(alias)
        if snapshot is None:
            continue
        is_stable, managed = _adjudicate_alias_snapshot(alias, *snapshot)
        if is_stable:
            return managed
        mismatch = ValueError("managed alias target does not match its ownership record")
    if mismatch is not None:
        raise mismatch
    raise ValueError("managed alias publication changed too frequently to validate")


def _runtime_managed_bundle(bundle: Path) -> Path | None:
    """Follow aliases to the first managed bundle identity.

    :param bundle: Candidate managed bundle or alias.
    :returns: Managed bundle, or ``None`` when the alias chain is unmanaged.
    :raises ValueError: The alias chain contains a cycle or invalid ownership.
    """
    current = bundle
    visited: set[Path] = set()
    while current not in visited:
        visited.add(current)
        managed = _runtime_alias_managed_bundle(current)
        if managed is not None:
            return managed
        if integrity.ManagedBundleStorage(current).has_integrity_record():
            return current
        target = managed_alias_target(current)
        if target is None:
            return None
        if integrity.ManagedBundleStorage(target).has_integrity_record():
            return target
        current = Path(os.path.abspath(target))
    raise ValueError("managed alias chain contains a cycle")


def _snapshot_digest(seal: BundleSeal) -> str:
    """Return the content-and-provenance key for a runtime snapshot.

    :param seal: Validated managed content record.
    :returns: Lowercase SHA256 key.
    """
    serialized = seal.model_dump_json(by_alias=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _remove_runtime_snapshot(path: Path) -> None:
    """Remove one rejected snapshot without following a top-level symlink.

    :param path: Snapshot path to remove when present.
    """
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISDIR(mode):
        shutil.rmtree(path)
    else:
        path.unlink()


def _ensure_runtime_snapshot_directory(path: Path) -> None:
    """Create a runtime-readable snapshot directory and reject symlink substitution.

    :param path: Managed snapshot directory.
    :raises ValueError: The resulting path is not a real, runtime-readable directory.
    """
    path.mkdir(mode=0o755, parents=True, exist_ok=True)
    mode = path.lstat().st_mode
    if not stat.S_ISDIR(mode):
        raise ValueError(f"runtime snapshot path is not a directory: {path}")
    if (
        stat.S_IMODE(mode) & integrity.RUNTIME_DIRECTORY_ACCESS_MASK
        != integrity.RUNTIME_DIRECTORY_ACCESS_MASK
    ):
        try:
            path.chmod(mode | integrity.RUNTIME_DIRECTORY_ACCESS_MASK)
        except PermissionError:
            logger.warning(
                "Runtime snapshot directory permission publication was denied",
                extra={"snapshot_path": str(path)},
                exc_info=True,
            )
    if (
        stat.S_IMODE(path.lstat().st_mode) & integrity.RUNTIME_DIRECTORY_ACCESS_MASK
        != integrity.RUNTIME_DIRECTORY_ACCESS_MASK
    ):
        raise ValueError(f"runtime snapshot path is not runtime-readable: {path}")


def _publish_runtime_tree_permissions(root: Path) -> None:
    """Publish read and traversal permissions without following symlinks.

    :param root: Existing snapshot tree owned by the current publisher.
    :raises ValueError: A traversed path is not a real directory or regular file.
    """
    for current, directories, files in os.walk(root, followlinks=False):
        _ensure_runtime_snapshot_directory(Path(current))
        directories[:] = [
            name
            for name in directories
            if not stat.S_ISLNK((Path(current) / name).lstat().st_mode)
        ]
        for name in files:
            child = Path(current) / name
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode):
                continue
            if not stat.S_ISREG(mode):
                raise ValueError(f"runtime snapshot path is not a regular file: {child}")
            child.chmod(mode | integrity.RUNTIME_FILE_READ_MASK)


def publish_runtime_snapshot_permissions(version_dir: Path) -> None:
    """Publish runtime-readable permissions for existing snapshot directories.

    :param version_dir: Managed package-version directory containing snapshots.
    :raises ValueError: Snapshot state contains a symlink or non-directory.
    """
    snapshots = version_dir / ".synth-setter-runtime-snapshots"
    try:
        mode = snapshots.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(mode):
        raise ValueError(f"runtime snapshot path is not a directory: {snapshots}")
    _publish_runtime_tree_permissions(snapshots)


def _publish_posix_runtime_parent_permissions(plugins_dir: Path, version_dir: Path) -> None:
    """Publish traversal permissions through a descriptor-relative managed hierarchy.

    :param plugins_dir: Managed storage root.
    :param version_dir: Descendant package-version directory.
    """
    relative = version_dir.relative_to(plugins_dir)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors: list[int] = []
    try:
        descriptor = os.open(plugins_dir, flags)
        descriptors.append(descriptor)
        os.fchmod(
            descriptor,
            os.fstat(descriptor).st_mode | integrity.RUNTIME_DIRECTORY_ACCESS_MASK,
        )
        for component in relative.parts:
            descriptor = os.open(component, flags, dir_fd=descriptor)
            descriptors.append(descriptor)
            os.fchmod(
                descriptor,
                os.fstat(descriptor).st_mode | integrity.RUNTIME_DIRECTORY_ACCESS_MASK,
            )
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _publish_windows_runtime_parent_permissions(plugins_dir: Path, version_dir: Path) -> None:
    """Reject links and publish traversal through a Windows managed hierarchy.

    :param plugins_dir: Managed storage root.
    :param version_dir: Descendant package-version directory.
    :raises OSError: A hierarchy component is not a real directory.
    """
    relative = version_dir.relative_to(plugins_dir)
    current = plugins_dir
    for component in (None, *relative.parts):
        if component is not None:
            current /= component
        mode = current.lstat().st_mode
        if os.path.isjunction(current) or stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise OSError(f"runtime hierarchy path is not a real directory: {current}")
        current.chmod(mode | integrity.RUNTIME_DIRECTORY_ACCESS_MASK)


def _publish_runtime_parent_permissions(plugins_dir: Path, version_dir: Path) -> None:
    """Publish traversal from managed storage through one package version.

    :param plugins_dir: Managed storage root.
    :param version_dir: Descendant package-version directory.
    """
    if os.name == "nt":
        _publish_windows_runtime_parent_permissions(plugins_dir, version_dir)
    else:
        _publish_posix_runtime_parent_permissions(plugins_dir, version_dir)


def prepare_managed_bundle_for_runtime(managed: Path, plugins_dir: Path) -> None:
    """Publish a validated adopted snapshot while installer authority is held.

    :param managed: Managed bundle path produced by the installer transaction.
    :param plugins_dir: Managed storage root whose package hierarchy is published.
    :raises ValueError: Managed bundle ownership disappears during preparation.
    """
    try:
        managed.lstat()
    except FileNotFoundError:
        return
    snapshots = managed.parent / ".synth-setter-runtime-snapshots"
    try:
        snapshot_mode = snapshots.lstat().st_mode
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISDIR(snapshot_mode):
            _remove_runtime_snapshot(snapshots)
    identity = _validated_runtime_bundle_identity(managed)
    if identity is None:
        raise ValueError(f"managed bundle ownership disappeared while preparing {managed}")
    _publish_runtime_parent_permissions(plugins_dir, managed.parent)
    if not stat.S_ISLNK(managed.lstat().st_mode):
        _publish_runtime_tree_permissions(managed)
    publish_runtime_snapshot_permissions(managed.parent)


def _runtime_snapshot_matches(destination: Path, seal: BundleSeal) -> bool:
    """Return whether a published snapshot matches its managed seal.

    :param destination: Candidate persistent runtime snapshot.
    :param seal: Expected managed content identity.
    :returns: Whether the destination is a real directory with matching entries.
    """
    try:
        if not stat.S_ISDIR(destination.lstat().st_mode):
            return False
        return integrity.bundle_entries(destination) == seal.entries
    except (FileNotFoundError, ValueError):
        return False


def _locked_snapshot_matches(
    destination: Path,
    seal: BundleSeal,
    parent_descriptor: int | None,
) -> bool:
    """Match a snapshot to its seal and retained publication directory.

    :param destination: Candidate persistent runtime snapshot.
    :param seal: Expected managed content identity.
    :param parent_descriptor: Retained POSIX publication directory descriptor.
    :returns: Whether the lexical candidate is the child validated under the lock.
    """
    if not _runtime_snapshot_matches(destination, seal):
        return False
    if parent_descriptor is None:
        return True
    try:
        retained = os.stat(destination.name, dir_fd=parent_descriptor, follow_symlinks=False)
        lexical = destination.lstat()
    except FileNotFoundError:
        return False
    return (retained.st_dev, retained.st_ino) == (lexical.st_dev, lexical.st_ino)


def _snapshot_destination_exists(destination: Path, parent_descriptor: int | None) -> bool:
    """Return whether the locked publication directory contains the destination.

    :param destination: Candidate snapshot destination.
    :param parent_descriptor: Retained POSIX publication directory descriptor.
    :returns: Whether any object occupies the destination name.
    """
    try:
        if parent_descriptor is None:
            destination.lstat()
        else:
            os.stat(destination.name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _publish_runtime_snapshot(
    candidate: Path,
    destination: Path,
    parent_descriptor: int | None,
) -> None:
    """Atomically publish a candidate under its retained destination directory.

    :param candidate: Verified snapshot in private temporary storage.
    :param destination: Final snapshot path.
    :param parent_descriptor: Retained POSIX publication directory descriptor.
    :raises ValueError: The publication directory is replaced before return.
    """
    if parent_descriptor is None:
        os.replace(candidate, destination)
        return
    candidate_parent = os.open(
        candidate.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        os.rename(
            candidate.name,
            destination.name,
            src_dir_fd=candidate_parent,
            dst_dir_fd=parent_descriptor,
        )
    finally:
        os.close(candidate_parent)
    retained = os.fstat(parent_descriptor)
    lexical = destination.parent.lstat()
    if (retained.st_dev, retained.st_ino) != (lexical.st_dev, lexical.st_ino):
        raise ValueError(
            f"runtime snapshot publication directory was replaced: {destination.parent}"
        )


def _verified_runtime_snapshot(managed: Path, source: Path, seal: BundleSeal) -> Path:
    """Return manager-owned bytes matching an adopted source seal.

    :param managed: Managed bundle whose top-level symlink marks adoption.
    :param source: Resolved installer- or caller-owned content.
    :param seal: Validated expected content and provenance.
    :returns: Persistent managed snapshot, or the source for direct managed bundles.
    :raises ValueError: Snapshot publication escapes storage or copies changed content.
    """
    try:
        is_adopted = stat.S_ISLNK(managed.lstat().st_mode)
    except FileNotFoundError:
        is_adopted = False
    if not is_adopted:
        return source

    snapshots = managed.parent / ".synth-setter-runtime-snapshots"
    digest = _snapshot_digest(seal)
    destination = snapshots / digest / managed.name
    _ensure_runtime_snapshot_directory(snapshots)
    with integrity.advisory_child_directory_lock(snapshots, digest) as parent_descriptor:
        if _locked_snapshot_matches(destination, seal, parent_descriptor):
            return destination
        if _snapshot_destination_exists(destination, parent_descriptor):
            raise ValueError(f"runtime snapshot destination is invalid: {destination}")
        temporary_parent = (
            Path("/dev/fd") / str(parent_descriptor)
            if parent_descriptor is not None and sys.platform.startswith("linux")
            else destination.parent
        )
        with tempfile.TemporaryDirectory(dir=temporary_parent) as temporary:
            candidate = Path(temporary) / managed.name
            shutil.copytree(source, candidate, symlinks=True)
            _publish_runtime_tree_permissions(candidate)
            if integrity.bundle_entries(candidate) != seal.entries:
                raise ValueError("managed source changed while creating its runtime snapshot")
            _publish_runtime_snapshot(candidate, destination, parent_descriptor)
    return destination


def _validated_runtime_bundle_identity(bundle: Path) -> tuple[Path, Path, BundleSeal] | None:
    """Resolve managed identity and immutable runtime content.

    :param bundle: Candidate managed bundle or alias.
    :returns: Managed path, consumable path, and seal, or ``None`` when unmanaged.
    """
    managed = _runtime_managed_bundle(bundle)
    if managed is None:
        return None
    resolved_bundle, recorded = integrity.ManagedBundleStorage(managed).validate()
    snapshot = _verified_runtime_snapshot(managed, resolved_bundle, recorded)
    return managed, snapshot, recorded


def _runtime_lock_path(
    managed: Path,
    ownership: ManagedBundleRecord | None,
) -> Path:
    """Select the durable installer or fallback runtime lock path.

    :param managed: Managed bundle identity.
    :param ownership: Bundle ownership for paths outside versioned storage.
    :returns: Lock path shared by runtime consumers and installers.
    :raises ValueError: A non-versioned managed bundle lacks ownership.
    """
    storage = _managed_storage_identity(managed)
    if storage is not None:
        plugins_dir, package, version = storage
        return integrity.package_install_lock_path(package, version, plugins_dir)
    if ownership is None:
        raise ValueError(f"managed bundle ownership is missing at {managed}")
    return managed.parent / f".{managed.name}.synth-setter-runtime.lock"


def _runtime_identity_and_lock(bundle: Path) -> tuple[Path, Path] | None:
    """Resolve the current managed identity and its lock path.

    :param bundle: Candidate managed bundle or alias.
    :returns: Managed identity and lock path, or ``None`` when unmanaged.
    """
    managed = _runtime_managed_bundle(bundle)
    if managed is None:
        return None
    ownership = integrity.ManagedBundleStorage(managed).read_ownership()
    return managed, _runtime_lock_path(managed, ownership)


def _locked_runtime_identity(
    bundle: Path,
    lock_path: Path,
) -> tuple[tuple[Path, Path], tuple[Path, Path, BundleSeal] | None]:
    """Re-resolve and validate managed identity after acquiring its candidate lock.

    :param bundle: Managed VST3 bundle or stable alias.
    :param lock_path: Candidate package lock acquired by the caller.
    :returns: Current lock candidate and validated identity, or ``None`` when the lock changed.
    :raises ValueError: Managed ownership disappears while resolving or validating.
    """
    locked_candidate = _runtime_identity_and_lock(bundle)
    if locked_candidate is None:
        raise ValueError("managed bundle ownership disappeared while leasing")
    if locked_candidate[1] != lock_path:
        return locked_candidate, None
    identity = _validated_runtime_bundle_identity(bundle)
    if identity is None:
        raise ValueError("managed bundle ownership disappeared while validating")
    return locked_candidate, identity


def _validated_runtime_identity_lease(
    bundle: Path,
) -> AbstractContextManager[tuple[Path, BundleSeal] | None]:
    @contextmanager
    def _leased() -> Iterator[tuple[Path, BundleSeal] | None]:
        try:
            candidate = _runtime_identity_and_lock(bundle)
        except (FileNotFoundError, ValueError) as exc:
            raise PluginIntegrityError(f"{bundle} failed managed bundle integrity") from exc
        if candidate is None:
            yield None
            return

        while True:
            _, lock_path = candidate
            with integrity.advisory_file_lease(lock_path):
                try:
                    locked_candidate, identity = _locked_runtime_identity(bundle, lock_path)
                except (OSError, ValueError) as exc:
                    raise PluginIntegrityError(
                        f"{bundle} failed managed bundle integrity"
                    ) from exc
                if identity is None:
                    candidate = locked_candidate
                    continue
                _, resolved, seal = identity
                yield resolved, seal
                return

    return _leased()


def validated_bundle_lease(bundle: Path) -> AbstractContextManager[Path]:
    """Lease validated content against managed or external-source replacement.

    Adopted external content is copied into a manager-owned snapshot and checked
    against its seal before this context yields. Entering raises
    :class:`PluginIntegrityError` when managed ownership, provenance, or content
    is invalid.

    :param bundle: VST3 bundle or stable manager-owned alias.
    :returns: Context manager yielding validated content or the original unmanaged path.
    """

    @contextmanager
    def _validated() -> Iterator[Path]:
        with _validated_runtime_identity_lease(bundle) as identity:
            yield bundle if identity is None else identity[0]

    return _validated()


def validate_plugin_bundle_for_runtime(bundle: Path) -> Path:
    """Validate a managed bundle and return its resolved content path.

    Callers that consume the returned path must use :func:`validated_bundle_lease`
    so installation remains excluded through consumption.

    Managed integrity failures propagate as :class:`PluginIntegrityError`.

    :param bundle: VST3 bundle or stable alias.
    :returns: Resolved managed content, or the original unmanaged path.
    """
    with validated_bundle_lease(bundle) as validated:
        return validated


def managed_plugin_digest(bundle: Path) -> str | None:
    """Validate local bytes and return their canonical managed provenance.

    Managed integrity failures propagate as :class:`PluginIntegrityError`.

    :param bundle: Managed VST3 bundle or stable alias, or an unmanaged path.
    :returns: Canonical managed identity, otherwise ``None``.
    """
    with _validated_runtime_identity_lease(bundle) as identity:
        if identity is None:
            return None
        return integrity.bundle_identity_digest(identity[1])


def plugin_bundle_version(bundle: Path, plugin_name: str | None = None) -> str:
    """Read static or factory version metadata while holding the bundle lease.

    :param bundle: Existing VST3 bundle.
    :param plugin_name: Factory class selected when the bundle exposes multiple plugins.
    :returns: Version reported by static metadata or the VST3 factory.
    :raises RuntimeError: The VST3 factory reports no version.
    """
    with validated_bundle_lease(bundle) as validated_bundle:
        moduleinfo = validated_bundle / "Contents/moduleinfo.json"
        if moduleinfo.is_file():
            return str(json.loads(moduleinfo.read_text(encoding="utf-8"))["Version"])
        info_plist = validated_bundle / "Contents/Info.plist"
        if info_plist.is_file():
            return str(plistlib.loads(info_plist.read_bytes())["CFBundleShortVersionString"])

        from pedalboard import VST3Plugin

        plugin = (
            VST3Plugin(str(validated_bundle))
            if plugin_name is None
            else VST3Plugin(str(validated_bundle), plugin_name=plugin_name)
        )
        version = plugin.version
        if not version:
            raise RuntimeError(f"Could not extract version from {validated_bundle}")
        return version
