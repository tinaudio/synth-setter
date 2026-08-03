"""Validate repository locks and manager-owned VST3 bundle content.

Typical usage loads the repository lock before selecting a managed package::

    from pathlib import Path
    from synth_setter.plugin_integrity import ArtifactLock
    from synth_setter.plugin_manager import PluginManifest

    manifest = PluginManifest.load(Path("studiorack.json"))
    lock = ArtifactLock.load(Path("studiorack.lock.json"), manifest)
    locked_package = lock.package_for(manifest.resolve("open-audio/cardinal"))
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import platform
import stat
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal, Protocol, cast

import pydantic

if TYPE_CHECKING:
    from synth_setter.plugin_manager import ManagedPlugin, PluginManifest


class _Msvcrt(Protocol):
    LK_NBLCK: int
    LK_UNLCK: int

    def locking(self, file_descriptor: int, mode: int, byte_count: int, /) -> None: ...


__all__ = [
    "ArtifactLock",
    "BundleEntry",
    "BundleSeal",
    "LockedArtifact",
    "LockedPackage",
    "ManagedBundleRecord",
    "ManagedBundleStorage",
    "PluginIntegrityError",
    "advisory_file_lock",
    "bundle_entries",
    "bundle_identity_digest",
    "bundle_is_sealed",
    "locked_bundle_is_sealed",
    "locked_package_digest",
    "package_install_lock",
    "package_install_lock_path",
    "seal_plugin_bundle",
]


class PluginIntegrityError(FileNotFoundError):
    """A manager-owned plugin no longer matches its recorded integrity state."""


class LockedArtifact(pydantic.BaseModel):
    """Repository-pinned identity for one registry-selected artifact.

    .. attribute :: model_config

        Pydantic validation settings.

    .. attribute :: architectures

        Registry architectures selecting the artifact.

    .. attribute :: sha256

        Lowercase SHA256 digest published by the registry.

    .. attribute :: systems

        Registry systems selecting the artifact.

    .. attribute :: type

        Studiorack installation strategy.

    .. attribute :: url

        HTTPS artifact source.
    """

    model_config = pydantic.ConfigDict(strict=True, extra="forbid", frozen=True)

    architectures: list[Literal["arm64", "x64"]] = pydantic.Field(min_length=1)
    sha256: str = pydantic.Field(pattern=r"^[0-9a-f]{64}$")
    systems: list[Literal["linux", "mac"]] = pydantic.Field(min_length=1)
    type: Literal["archive", "installer"]
    url: str

    @pydantic.field_validator("url")
    @classmethod
    def _require_https(cls, value: str) -> str:
        """Reject artifact locations outside HTTPS.

        :param value: Candidate artifact URL.
        :returns: Validated HTTPS URL.
        :raises ValueError: The URL does not use HTTPS.
        """
        if not value.startswith("https://"):
            raise ValueError("artifact URL must use HTTPS")
        return value


class LockedPackage(pydantic.BaseModel):
    """Selected supported artifacts for one exact package version.

    .. attribute :: model_config

        Pydantic validation settings.

    .. attribute :: artifacts

        Artifact identities selected on supported provisioning hosts.
    """

    model_config = pydantic.ConfigDict(strict=True, extra="forbid", frozen=True)

    artifacts: list[LockedArtifact] = pydantic.Field(min_length=1)


class ArtifactLock(pydantic.RootModel[dict[str, LockedPackage]]):
    """Exact package references mapped to repository-trusted artifacts.

    .. attribute :: model_config

        Pydantic validation settings.
    """

    model_config = pydantic.ConfigDict(strict=True, frozen=True)

    def package_for(self, plugin: ManagedPlugin) -> LockedPackage:
        """Return locked artifact identity for one exact managed package.

        :param plugin: Exact package selected from the corresponding manifest.
        :returns: Locked package identity.
        :raises KeyError: The exact package reference is absent.
        """
        try:
            return self.root[plugin.reference]
        except KeyError:
            raise KeyError(plugin.reference) from None

    @classmethod
    def load(cls, path: Path, manifest: PluginManifest) -> ArtifactLock:
        """Parse a lock whose exact references cover the manifest.

        :param path: Artifact-lock path.
        :param manifest: Validated project manifest requiring exact coverage.
        :returns: Validated artifact lock.
        :raises ValueError: Lock references differ from the manifest pins.
        """
        lock = cls.model_validate(json.loads(path.read_text(encoding="utf-8")), strict=True)
        manifest_references = {
            f"{package}@{version}" for package, version in manifest.plugins.items()
        }
        if lock.root.keys() != manifest_references:
            raise ValueError("artifact lock must exactly cover manifest package versions")
        return lock


class BundleEntry(pydantic.BaseModel):
    """Content identity for one regular file or symlink.

    .. attribute :: model_config

        Pydantic validation settings.

    .. attribute :: path

        POSIX path relative to the VST3 bundle.

    .. attribute :: sha256

        Regular-file byte digest.

    .. attribute :: size

        Regular-file byte count.

    .. attribute :: target

        Symlink target text.

    .. attribute :: type

        Filesystem entry type.
    """

    model_config = pydantic.ConfigDict(strict=True, extra="forbid", frozen=True)

    path: str
    sha256: str | None = pydantic.Field(default=None, pattern=r"^[0-9a-f]{64}$")
    size: int | None = pydantic.Field(default=None, ge=0)
    target: str | None = None
    type: Literal["file", "symlink"]

    @pydantic.model_validator(mode="after")
    def _validate_type_fields(self) -> BundleEntry:
        """Require bytes for files and target text for symlinks.

        :returns: Validated entry.
        :raises ValueError: Fields do not match the entry type.
        """
        has_digest = self.sha256 is not None
        has_size = self.size is not None
        if self.type == "file" and has_digest and has_size and self.target is None:
            return self
        if self.type == "symlink" and not has_digest and not has_size and self.target is not None:
            return self
        raise ValueError("bundle entry fields do not match its type")


class BundleSeal(pydantic.BaseModel):
    """Atomic content and provenance record for one managed VST3 bundle.

    .. attribute :: model_config

        Pydantic validation settings.

    .. attribute :: bundle

        Expected VST3 bundle basename.

    .. attribute :: entries

        Sorted content identities.

    .. attribute :: locked_package_sha256

        Canonical registry lock identity for artifact-backed installs.

    .. attribute :: package_reference

        Exact package version represented by the bundle.

    .. attribute :: schema_version

        Completion-record schema version.

    .. attribute :: source_kind

        Registry artifact or explicit source adoption provenance.
    """

    model_config = pydantic.ConfigDict(
        strict=True, extra="forbid", frozen=True, populate_by_name=True
    )

    bundle: str
    entries: list[BundleEntry] = pydantic.Field(min_length=1)
    locked_package_sha256: str | None = pydantic.Field(default=None, pattern=r"^[0-9a-f]{64}$")
    package_reference: str
    schema_version: Literal[3] = pydantic.Field(default=3, alias="schema")
    source_kind: Literal["artifact-lock", "explicit"]

    @pydantic.model_validator(mode="after")
    def _validate_provenance(self) -> BundleSeal:
        """Require exact repository provenance for every managed source.

        :returns: Validated seal.
        :raises ValueError: The seal omits its package-lock identity.
        """
        if self.locked_package_sha256 is None:
            raise ValueError("bundle seal requires locked package provenance")
        return self


class ManagedBundleRecord(pydantic.BaseModel):
    """Durable ownership identity independent of a bundle's content seal.

    .. attribute :: model_config

        Pydantic validation settings.

    .. attribute :: bundle

        Expected VST3 bundle basename.

    .. attribute :: package

        Studiorack package slug.

    .. attribute :: schema_version

        Ownership-record schema version.

    .. attribute :: source_kind

        Registry artifact or explicit source adoption provenance.

    .. attribute :: version

        Exact Studiorack package version.
    """

    model_config = pydantic.ConfigDict(
        strict=True, extra="forbid", frozen=True, populate_by_name=True
    )

    bundle: str
    package: str
    schema_version: Literal[1] = pydantic.Field(default=1, alias="schema")
    source_kind: Literal["artifact-lock", "explicit"]
    version: str


def bundle_identity_digest(seal: BundleSeal) -> str:
    """Return the canonical runtime identity for one validated bundle seal.

    Registry artifacts share their platform-independent package-lock identity. Explicit sources
    additionally bind the sealed source tree and source kind.

    :param seal: Validated seal whose content has already been rechecked.
    :returns: Stable lowercase SHA256 identity.
    :raises ValueError: The seal lacks package-lock provenance.
    """
    locked_package_sha256 = seal.locked_package_sha256
    if locked_package_sha256 is None:
        raise ValueError("bundle identity requires locked package provenance")
    if seal.source_kind == "artifact-lock":
        return locked_package_sha256
    entries = [entry.model_dump(mode="json", exclude_none=True) for entry in seal.entries]
    canonical = json.dumps(
        {
            "entries": sorted(entries, key=lambda entry: entry["path"]),
            "locked_package_sha256": locked_package_sha256,
            "source_kind": seal.source_kind,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def locked_package_digest(package_reference: str, locked_package: LockedPackage) -> str:
    """Return one exact package pin's canonical repository-lock identity.

    :param package_reference: Exact ``package@version`` key owning the artifacts.
    :param locked_package: Repository-trusted package artifacts.
    :returns: Stable lowercase SHA256 digest.
    """
    artifacts = []
    for artifact in locked_package.artifacts:
        normalized = artifact.model_dump(mode="json")
        normalized["architectures"] = sorted(normalized["architectures"])
        normalized["systems"] = sorted(normalized["systems"])
        artifacts.append(normalized)
    artifacts.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    canonical = json.dumps(
        {"package_reference": package_reference, "artifacts": artifacts},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _bundle_root(bundle: Path) -> Path:
    root = bundle.resolve(strict=True)
    if not stat.S_ISDIR(root.stat().st_mode):
        raise ValueError(f"bundle is not a directory: {bundle}")
    return root


def _regular_file_entry(child: Path, relative: str) -> BundleEntry:
    size = child.stat(follow_symlinks=False).st_size
    with child.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return BundleEntry(path=relative, sha256=digest, size=size, type="file")


def _symlink_entry(child: Path, root: Path, relative: str) -> BundleEntry:
    target = os.readlink(child)
    resolved_target = (child.parent / target).resolve(strict=True)
    if not resolved_target.is_relative_to(root):
        raise ValueError(f"bundle symlink escapes its root: {relative}")
    return BundleEntry(path=relative, target=target, type="symlink")


def bundle_entries(bundle: Path) -> list[BundleEntry]:
    """Hash a bundle without following directory symlinks.

    :param bundle: Real bundle directory or managed symlink.
    :returns: Sorted file and symlink identities.
    :raises ValueError: The bundle is empty, escapes by symlink, or contains special files.
    """
    root = _bundle_root(bundle)
    entries: list[BundleEntry] = []
    directories = [root]
    while directories:
        directory = directories.pop()
        for child in sorted(directory.iterdir(), key=lambda path: path.name):
            mode = child.lstat().st_mode
            relative = child.relative_to(root).as_posix()
            if stat.S_ISDIR(mode):
                directories.append(child)
            elif stat.S_ISREG(mode):
                entries.append(_regular_file_entry(child, relative))
            elif stat.S_ISLNK(mode):
                entries.append(_symlink_entry(child, root, relative))
            else:
                raise ValueError(f"bundle contains a special file: {relative}")
    if not any(entry.type == "file" and entry.size for entry in entries):
        raise ValueError("bundle must contain at least one nonempty regular file")
    return sorted(entries, key=lambda entry: entry.path)


def _directory_files(directory: Path, suffix: str | None = None) -> list[Path]:
    try:
        children = sorted(directory.iterdir(), key=lambda path: path.name)
    except FileNotFoundError:
        return []
    return [
        child
        for child in children
        if stat.S_ISREG(child.lstat().st_mode) and (suffix is None or child.suffix == suffix)
    ]


def _platform_architecture() -> str:
    """Map supported host names to VST3 directory architecture names.

    :returns: VST3 architecture directory prefix.
    :raises ValueError: The host architecture is unsupported.
    """
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        return "aarch64" if sys.platform.startswith("linux") else "arm64"
    if machine in {"amd64", "x86_64"}:
        return "x86_64"
    raise ValueError(f"unsupported VST3 host architecture: {machine}")


def _platform_binary_paths(bundle: Path) -> list[Path]:
    contents = _bundle_root(bundle) / "Contents"
    if sys.platform.startswith("linux"):
        return _directory_files(contents / f"{_platform_architecture()}-linux", ".so")
    if sys.platform == "darwin":
        return _directory_files(contents / "MacOS")
    return _directory_files(contents / f"{_platform_architecture()}-win", ".vst3")


def _has_pe_signature(stream: BinaryIO) -> bool:
    if stream.read(2) != b"MZ":
        return False
    stream.seek(0x3C)
    offset_bytes = stream.read(4)
    if len(offset_bytes) != 4:
        return False
    stream.seek(int.from_bytes(offset_bytes, "little"))
    return stream.read(4) == b"PE\0\0"


def _binary_has_platform_signature(binary: Path) -> bool:
    with binary.open("rb") as stream:
        if sys.platform.startswith("linux"):
            return stream.read(4) == b"\x7fELF"
        if sys.platform == "darwin":
            return stream.read(4) in {
                b"\xbe\xba\xfe\xca",
                b"\xbf\xba\xfe\xca",
                b"\xca\xfe\xba\xbe",
                b"\xca\xfe\xba\xbf",
                b"\xce\xfa\xed\xfe",
                b"\xcf\xfa\xed\xfe",
                b"\xfe\xed\xfa\xce",
                b"\xfe\xed\xfa\xcf",
            }
        return _has_pe_signature(stream)


def _validate_platform_binary(bundle: Path) -> Path:
    binaries = _platform_binary_paths(bundle)
    valid = [binary for binary in binaries if _binary_has_platform_signature(binary)]
    if len(valid) != 1:
        raise ValueError(f"{bundle} must contain exactly one valid platform VST3 binary")
    return valid[0]


def _seal_path(bundle: Path) -> Path:
    """Place a version-local completion record beside its managed bundle.

    :param bundle: Managed bundle path.
    :returns: Completion-record path.
    """
    return bundle.parent / ".synth-setter-complete.json"


def _managed_record_path(bundle: Path) -> Path:
    """Place durable bundle ownership beside its independently removable seal.

    :param bundle: Managed bundle path.
    :returns: Bundle ownership-record path.
    """
    return bundle.parent / ".synth-setter-managed.json"


def write_atomic_record(path: Path, serialized: str) -> None:
    """Replace a strict record only after its bytes are durable and runtime-readable.

    Records are mode 0644 because privileged installers hand them to unprivileged consumers.

    :param path: Final record path whose parent already exists.
    :param serialized: Complete UTF-8 record text.
    """
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, encoding="utf-8", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _managed_bundle_record(
    plugin: ManagedPlugin,
    *,
    source_kind: Literal["artifact-lock", "explicit"],
) -> ManagedBundleRecord:
    return ManagedBundleRecord(
        bundle=plugin.bundle,
        package=plugin.package,
        source_kind=source_kind,
        version=plugin.version,
    )


def _read_managed_bundle_record(bundle: Path) -> ManagedBundleRecord | None:
    path = _managed_record_path(bundle)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(mode):
        raise ValueError(f"managed bundle ownership is not a regular file: {path}")
    return ManagedBundleRecord.model_validate_json(path.read_text(encoding="utf-8"))


def _bundle_seal(
    plugin: ManagedPlugin,
    entries: list[BundleEntry],
    *,
    locked_package: LockedPackage,
    source_kind: Literal["artifact-lock", "explicit"],
) -> BundleSeal:
    return BundleSeal(
        bundle=plugin.bundle,
        entries=entries,
        locked_package_sha256=locked_package_digest(plugin.reference, locked_package),
        package_reference=plugin.reference,
        source_kind=source_kind,
    )


def seal_plugin_bundle(
    bundle: Path,
    plugin: ManagedPlugin,
    *,
    locked_package: LockedPackage,
    record_for: Path | None = None,
    source_kind: Literal["artifact-lock", "explicit"] = "artifact-lock",
) -> Path:
    """Atomically seal structurally valid bytes and exact package provenance.

    :param bundle: Bundle content to inspect.
    :param plugin: Exact package represented by the bundle.
    :param locked_package: Repository identity for the exact package pin.
    :param record_for: Managed bundle path owning the record; defaults to ``bundle``.
    :param source_kind: Registry artifact install or explicit source adoption.
    :returns: Completion-record path.
    """
    _validate_platform_binary(bundle)
    managed_bundle = bundle if record_for is None else record_for
    seal = _bundle_seal(
        plugin,
        bundle_entries(bundle),
        locked_package=locked_package,
        source_kind=source_kind,
    )
    ownership = _managed_bundle_record(plugin, source_kind=source_kind)
    marker = _seal_path(managed_bundle)
    marker.parent.mkdir(parents=True, exist_ok=True)
    write_atomic_record(
        _managed_record_path(managed_bundle),
        ownership.model_dump_json(indent=2, by_alias=True),
    )
    serialized = seal.model_dump_json(indent=2, exclude_none=True, by_alias=True)
    write_atomic_record(marker, serialized)
    return marker


def _seal_matches_content(bundle: Path, recorded: BundleSeal) -> bool:
    _validate_platform_binary(bundle)
    current = BundleSeal(
        bundle=recorded.bundle,
        entries=bundle_entries(bundle),
        locked_package_sha256=recorded.locked_package_sha256,
        package_reference=recorded.package_reference,
        source_kind=recorded.source_kind,
    )
    return recorded == current


@dataclass(frozen=True)
class ManagedBundleStorage:
    """Access integrity-owned records for one managed bundle.

    .. attribute :: bundle

        Managed VST3 bundle path whose sibling records are accessed.
    """

    bundle: Path

    @property
    def _ownership_path(self) -> Path:
        return _managed_record_path(self.bundle)

    @property
    def _seal_path(self) -> Path:
        return _seal_path(self.bundle)

    def discard(self) -> None:
        """Remove ownership and seal records for an unpublished bundle."""
        self._ownership_path.unlink(missing_ok=True)
        self._seal_path.unlink(missing_ok=True)

    def has_integrity_record(self) -> bool:
        """Return whether ownership or seal state marks this bundle as managed.

        :returns: Whether either integrity record exists as a regular file.
        :raises ValueError: A record path exists but is not a regular file.
        """
        if self.read_ownership() is not None:
            return True
        try:
            mode = self._seal_path.lstat().st_mode
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(mode):
            raise ValueError(f"managed bundle seal is not a regular file: {self._seal_path}")
        return True

    def read_ownership(self) -> ManagedBundleRecord | None:
        """Read durable bundle ownership when present.

        :returns: Validated ownership, or ``None`` when absent.
        """
        return _read_managed_bundle_record(self.bundle)

    def validate(self) -> tuple[Path, BundleSeal]:
        """Resolve content and validate ownership against its seal.

        :returns: Resolved bundle content and validated seal.
        :raises ValueError: Ownership, seal, provenance, or content is invalid.
        """
        ownership = self.read_ownership()
        if ownership is None:
            raise ValueError(f"managed bundle ownership is missing at {self._ownership_path}")
        resolved_bundle = self.bundle.resolve(strict=True)
        recorded = BundleSeal.model_validate_json(self._seal_path.read_text(encoding="utf-8"))
        if ownership.bundle != self.bundle.name:
            raise ValueError(
                f"ownership bundle {ownership.bundle!r} does not match managed path "
                f"{self.bundle.name!r}"
            )
        if recorded.package_reference != f"{ownership.package}@{ownership.version}":
            raise ValueError("managed bundle ownership does not match its seal package")
        if recorded.source_kind != ownership.source_kind:
            raise ValueError("managed bundle ownership does not match its seal source kind")
        if recorded.bundle != self.bundle.name:
            raise ValueError(
                f"seal bundle {recorded.bundle!r} does not match managed path {self.bundle.name!r}"
            )
        if not _seal_matches_content(resolved_bundle, recorded):
            raise ValueError(f"managed bundle content does not match seal at {self._seal_path}")
        return resolved_bundle, recorded


def _matching_bundle_seal(
    bundle: Path,
    plugin: ManagedPlugin,
    locked_package: LockedPackage,
) -> BundleSeal | None:
    try:
        ownership = _read_managed_bundle_record(bundle)
        recorded = BundleSeal.model_validate_json(_seal_path(bundle).read_text(encoding="utf-8"))
        if ownership != _managed_bundle_record(plugin, source_kind=recorded.source_kind):
            return None
        if recorded.package_reference != plugin.reference or recorded.bundle != plugin.bundle:
            return None
        expected_digest = locked_package_digest(plugin.reference, locked_package)
        if recorded.locked_package_sha256 != expected_digest:
            return None
        return recorded if _seal_matches_content(bundle, recorded) else None
    except (FileNotFoundError, ValueError):
        return None


def bundle_is_sealed(
    bundle: Path,
    plugin: ManagedPlugin,
    locked_package: LockedPackage,
) -> bool:
    """Check content, ownership, and provenance against an exact package identity.

    :param bundle: Candidate managed bundle path.
    :param plugin: Exact package and bundle identity.
    :param locked_package: Current exact package-lock identity.
    :returns: Whether every managed integrity record and content entry matches.
    """
    return _matching_bundle_seal(bundle, plugin, locked_package) is not None


def locked_bundle_is_sealed(
    bundle: Path,
    plugin: ManagedPlugin,
    locked_package: LockedPackage,
) -> bool:
    """Require a valid seal with exact registry artifact-lock provenance.

    :param bundle: Candidate managed bundle path.
    :param plugin: Exact package and bundle identity.
    :param locked_package: Required registry artifact identity.
    :returns: Whether content and artifact-lock provenance both match.
    """
    seal = _matching_bundle_seal(bundle, plugin, locked_package)
    return seal is not None and seal.source_kind == "artifact-lock"


def _windows_lock(stream: BinaryIO) -> None:
    """Acquire one byte through Windows' standard advisory lock API.

    :param stream: Open binary lock-file stream.
    :raises OSError: Locking fails for a reason other than contention.
    """
    import msvcrt

    windows_runtime = cast("_Msvcrt", msvcrt)
    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b"\0")
        stream.flush()
        os.fsync(stream.fileno())
    while True:
        try:
            stream.seek(0)
            windows_runtime.locking(stream.fileno(), windows_runtime.LK_NBLCK, 1)
            return
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise
            time.sleep(0.05)


def _windows_unlock(stream: BinaryIO) -> None:
    """Release the byte held by ``_windows_lock``.

    :param stream: Open binary lock-file stream.
    """
    import msvcrt

    windows_runtime = cast("_Msvcrt", msvcrt)
    stream.seek(0)
    windows_runtime.locking(stream.fileno(), windows_runtime.LK_UNLCK, 1)


def advisory_file_lock(path: Path) -> AbstractContextManager[None]:
    """Build a cross-process exclusive lock for one durable path.

    :param path: Lock file outside removable package-version state.
    :returns: Context manager holding the lock around its body.
    """

    @contextmanager
    def _locked() -> Iterator[None]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as stream:
            if os.name == "nt":
                _windows_lock(stream)
                try:
                    yield
                finally:
                    _windows_unlock(stream)
                return

            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    return _locked()


def package_install_lock_path(package: str, version: str, plugins_dir: Path) -> Path:
    """Return the durable lock path for one exact managed package.

    :param package: Studiorack ``organization/package`` slug.
    :param version: Exact package version.
    :param plugins_dir: Resolved managed storage root.
    :returns: Lock path outside removable package-version state.
    """
    organization, package_name = package.split("/")
    return (
        plugins_dir
        / ".synth-setter-install-locks"
        / organization
        / package_name
        / f"{version}.lock"
    )


def package_install_lock(
    package: str,
    version: str,
    plugins_dir: Path,
) -> AbstractContextManager[None]:
    """Serialize consumers and installers for one exact package version.

    :param package: Studiorack ``organization/package`` slug.
    :param version: Exact package version.
    :param plugins_dir: Resolved managed storage root.
    :returns: Context manager holding the package lock around its body.
    """
    return advisory_file_lock(package_install_lock_path(package, version, plugins_dir))
