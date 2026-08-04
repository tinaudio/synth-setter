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
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
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
    "RUNTIME_DIRECTORY_ACCESS_MASK",
    "RUNTIME_FILE_READ_MASK",
    "advisory_child_directory_lock",
    "advisory_directory_lock",
    "advisory_file_lease",
    "advisory_file_lock",
    "bundle_entries",
    "bundle_identity_digest",
    "bundle_is_sealed",
    "locked_bundle_is_sealed",
    "locked_package_digest",
    "open_posix_nofollow_directory",
    "package_install_lock",
    "package_install_lock_path",
    "seal_plugin_bundle",
    "windows_retained_nofollow_directories",
]

RUNTIME_FILE_READ_MASK = 0o044
RUNTIME_DIRECTORY_ACCESS_MASK = 0o055
_LOCK_RETRY_INTERVAL_SECONDS = 0.05
_LEASE_DIRECTORY_TIMEOUT_SECONDS = 10.0
_POSIX_SEARCH_ONLY_OPEN_FLAG = getattr(
    os,
    "O_PATH",
    getattr(os, "O_SEARCH", getattr(os, "O_EVTONLY", os.O_RDONLY)),
)
_WINDOWS_GENERIC_READ_WRITE = 0xC0000000
_WINDOWS_SHARE_READ_WRITE = 0x00000003
_WINDOWS_OPEN_ALWAYS = 4
_WINDOWS_NORMAL_OPEN_REPARSE_POINT = 0x00200080
_WINDOWS_FILE_ATTRIBUTE_TAG_INFO = 9
_WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_WINDOWS_OPEN_EXISTING = 3
_WINDOWS_DIRECTORY_OPEN_REPARSE_POINT = 0x02200000


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
    """Resolve a bundle and require a real directory.

    :param bundle: Path resolved without accepting a non-directory final target.
    :returns: Strictly resolved bundle root.
    :raises ValueError: The resolved path is not a directory.
    """
    root = bundle.resolve(strict=True)
    if not stat.S_ISDIR(root.stat().st_mode):
        raise ValueError(f"bundle is not a directory: {bundle}")
    return root


def _regular_file_entry(child: Path, relative: str) -> BundleEntry:
    """Hash one regular bundle file into its sealed identity.

    :param child: Regular file inside the resolved bundle root.
    :param relative: POSIX path relative to the bundle root.
    :returns: Content entry with size and SHA256 digest.
    """
    size = child.stat(follow_symlinks=False).st_size
    with child.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return BundleEntry(path=relative, sha256=digest, size=size, type="file")


def _symlink_entry(child: Path, root: Path, relative: str) -> BundleEntry:
    """Record one bundle symlink only when its target stays inside the root.

    :param child: Symlink inside the resolved bundle root.
    :param root: Resolved bundle root constraining the target.
    :param relative: POSIX path relative to the bundle root.
    :returns: Symlink entry carrying its lexical target.
    :raises ValueError: The resolved target escapes the bundle root.
    """
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
            time.sleep(_LOCK_RETRY_INTERVAL_SECONDS)


def _windows_create_handle(
    path: Path,
    *,
    access: int,
    disposition: int,
    flags: int,
    kind: str,
) -> int:
    """Call the shared CreateFileW binding with explicit security flags.

    :param path: Native path opened without implicit traversal policy.
    :param access: Desired Windows access mask.
    :param disposition: Windows creation disposition.
    :param flags: Windows file or directory flags.
    :param kind: Object kind used in diagnostics.
    :returns: Native Windows handle value.
    :raises OSError: Native handle creation fails.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path),
        access,
        _WINDOWS_SHARE_READ_WRITE,
        None,
        disposition,
        flags,
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        error_code = getattr(ctypes, "get_last_error")()
        raise OSError(error_code, f"CreateFileW failed for {kind}: {path}")
    return cast("int", handle)


def _windows_create_lock_handle(path: Path) -> int:
    """Create or open one marker without reparse-point traversal.

    :param path: Durable Windows lock marker.
    :returns: Native Windows handle value.
    """
    return _windows_create_handle(
        path,
        access=_WINDOWS_GENERIC_READ_WRITE,
        disposition=_WINDOWS_OPEN_ALWAYS,
        flags=_WINDOWS_NORMAL_OPEN_REPARSE_POINT,
        kind="lock marker",
    )


def _windows_open_directory_handle(path: Path) -> int:
    """Retain a directory handle without following its final reparse point.

    :param path: Existing Windows publication directory.
    :returns: Validated native directory handle.
    :raises FileExistsError: The opened directory is a reparse point.
    :raises OSError: Retained-handle inspection fails.
    """
    retained = _windows_create_handle(
        path,
        access=0,
        disposition=_WINDOWS_OPEN_EXISTING,
        flags=_WINDOWS_DIRECTORY_OPEN_REPARSE_POINT,
        kind="publication directory",
    )
    try:
        if _windows_lock_handle_is_reparse_point(retained, path):
            raise FileExistsError(f"publication path is a reparse point: {path}")
    except (FileExistsError, OSError):
        _windows_close_handle(retained)
        raise
    return retained


def _windows_lock_handle_is_reparse_point(handle: int, path: Path) -> bool:
    """Inspect the opened marker handle rather than its mutable path.

    :param handle: Retained native marker handle.
    :param path: Marker path used in diagnostics.
    :returns: Whether the opened object is a reparse point.
    :raises OSError: Native handle inspection fails.
    """
    import ctypes
    from ctypes import wintypes

    class _FileAttributeTagInfo(ctypes.Structure):
        _fields_ = [("file_attributes", wintypes.DWORD), ("reparse_tag", wintypes.DWORD)]

    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    get_file_information = kernel32.GetFileInformationByHandleEx
    get_file_information.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    get_file_information.restype = wintypes.BOOL
    info = _FileAttributeTagInfo()
    if not get_file_information(
        handle,
        _WINDOWS_FILE_ATTRIBUTE_TAG_INFO,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        error_code = getattr(ctypes, "get_last_error")()
        raise OSError(error_code, f"lock handle inspection failed: {path}")
    return bool(info.file_attributes & _WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT)


def _windows_close_handle(handle: int) -> None:
    """Close one native Windows handle not transferred to a Python stream.

    :param handle: Native Windows handle value.
    """
    import ctypes
    from ctypes import wintypes

    close_handle = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True).CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    close_handle(handle)


def _windows_open_regular_lock(path: Path) -> BinaryIO:
    """Open a Windows lock marker without following a reparse point.

    :param path: Existing or newly created durable lock marker.
    :returns: Binary stream owning the validated native handle.
    :raises FileExistsError: The opened object is a reparse point.
    :raises OSError: Native handle creation, inspection, or conversion fails.
    """
    import msvcrt

    handle = _windows_create_lock_handle(path)
    try:
        if _windows_lock_handle_is_reparse_point(handle, path):
            raise FileExistsError(f"lock is a reparse point: {path}")
        open_osfhandle = getattr(msvcrt, "open_osfhandle")
        descriptor = open_osfhandle(handle, os.O_RDWR | getattr(os, "O_BINARY", 0))
    except (FileExistsError, OSError):
        _windows_close_handle(handle)
        raise
    return os.fdopen(descriptor, "a+b")


def _windows_unlock(stream: BinaryIO) -> None:
    """Release the byte held by ``_windows_lock``.

    :param stream: Open binary lock-file stream.
    """
    import msvcrt

    windows_runtime = cast("_Msvcrt", msvcrt)
    stream.seek(0)
    windows_runtime.locking(stream.fileno(), windows_runtime.LK_UNLCK, 1)


def _posix_advisory_descriptor_lock(
    descriptor: int,
    operation: int,
) -> AbstractContextManager[None]:
    """Apply a POSIX advisory lock to an open file or directory descriptor.

    :param descriptor: Descriptor retained for the full lock lifetime.
    :param operation: ``fcntl.LOCK_SH`` or ``fcntl.LOCK_EX``.
    :returns: Context manager holding the requested lock around its body.
    """

    @contextmanager
    def _locked() -> Iterator[None]:
        import fcntl

        fcntl.flock(descriptor, operation)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)

    return _locked()


def _posix_existing_child_directory(
    parent_descriptor: int,
    component: str,
    access_flag: int,
) -> int:
    """Open one existing no-follow child directory.

    :param parent_descriptor: Retained parent directory descriptor.
    :param component: Single child directory name.
    :param access_flag: Search-only or read-only final access.
    :returns: Retained child directory descriptor.
    """
    flags = access_flag | os.O_DIRECTORY | os.O_NOFOLLOW
    return os.open(component, flags, dir_fd=parent_descriptor)


def _posix_create_or_open_child_directory(
    parent_descriptor: int,
    component: str,
    access_flag: int,
) -> int:
    """Create or reopen one no-follow child directory.

    :param parent_descriptor: Retained parent directory descriptor.
    :param component: Single child directory name.
    :param access_flag: Search-only or read-only final access.
    :returns: Retained child directory descriptor.
    """
    try:
        os.mkdir(component, dir_fd=parent_descriptor)
    except FileExistsError:
        pass
    return _posix_existing_child_directory(parent_descriptor, component, access_flag)


def _posix_directory_hierarchy_descriptor(
    directory: Path,
    open_child: Callable[[int, str, int], int],
) -> int:
    """Traverse one absolute hierarchy under a shared no-follow policy.

    :param directory: Directory hierarchy to open.
    :param open_child: Existing-only or create-and-open child operation.
    :returns: Read-only descriptor for the final directory.
    :raises FileExistsError: Any hierarchy component is a symlink or non-directory.
    :raises OSError: Opening or creating a hierarchy component fails.
    """
    absolute = directory.absolute()
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY)
    components = absolute.parts[1:]
    try:
        for index, component in enumerate(components):
            access_flag = (
                os.O_RDONLY if index == len(components) - 1 else _POSIX_SEARCH_ONLY_OPEN_FLAG
            )
            child = open_child(descriptor, component, access_flag)
            os.close(descriptor)
            descriptor = child
    except OSError as exc:
        os.close(descriptor)
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise FileExistsError(f"lock path is not a real directory: {directory}") from exc
        raise
    return descriptor


def open_posix_nofollow_directory(directory: Path) -> int:
    """Open an existing absolute directory hierarchy without following symlinks.

    :param directory: Existing directory hierarchy.
    :returns: Read-only descriptor for the final directory.
    """
    return _posix_directory_hierarchy_descriptor(directory, _posix_existing_child_directory)


def _posix_create_directory_descriptor(directory: Path) -> int:
    """Create an absolute hierarchy relative to retained no-follow parents.

    :param directory: Directory hierarchy to create or open.
    :returns: Read-only descriptor for the final directory.
    """
    return _posix_directory_hierarchy_descriptor(
        directory,
        _posix_create_or_open_child_directory,
    )


def advisory_directory_lock(path: Path) -> AbstractContextManager[int | None]:
    """Lock one retained real directory for descriptor-relative publication.

    :param path: Existing publication directory.
    :returns: Context manager yielding a POSIX descriptor or ``None`` on Windows.
    """

    @contextmanager
    def _locked() -> Iterator[int | None]:
        if os.name == "nt":
            lock_digest = hashlib.sha256(str(path.absolute()).encode()).hexdigest()
            marker = (
                type(path)(tempfile.gettempdir())
                / ".synth-setter-publication-locks"
                / f"{lock_digest}.lock"
            )
            directory_handle = _windows_open_directory_handle(path)
            try:
                with advisory_file_lock(marker):
                    mode = path.lstat().st_mode
                    if os.path.isjunction(path) or stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                        raise FileExistsError(f"publication path is not a real directory: {path}")
                    yield None
            finally:
                _windows_close_handle(directory_handle)
            return

        import fcntl

        descriptor = open_posix_nofollow_directory(path)
        try:
            with _posix_advisory_descriptor_lock(descriptor, fcntl.LOCK_EX):
                yield descriptor
        finally:
            os.close(descriptor)

    return _locked()


def advisory_child_directory_lock(
    parent: Path,
    name: str,
) -> AbstractContextManager[int | None]:
    """Create and lock one child relative to a retained real parent.

    :param parent: Existing trusted publication root.
    :param name: Single child directory name.
    :returns: Context manager yielding a POSIX child descriptor or ``None`` on Windows.
    """

    @contextmanager
    def _locked() -> Iterator[int | None]:
        if name in {"", ".", ".."} or Path(name).name != name:
            raise ValueError(f"publication child must be one path component: {name}")
        if os.name == "nt":
            with advisory_directory_lock(parent):
                child = parent / name
                child.mkdir(exist_ok=True)
                with advisory_directory_lock(child):
                    yield None
            return

        import fcntl

        parent_descriptor = open_posix_nofollow_directory(parent)
        try:
            with _posix_advisory_descriptor_lock(parent_descriptor, fcntl.LOCK_EX):
                try:
                    os.mkdir(name, dir_fd=parent_descriptor)
                except FileExistsError:
                    pass
                child_descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent_descriptor,
                )
                try:
                    with _posix_advisory_descriptor_lock(child_descriptor, fcntl.LOCK_EX):
                        yield child_descriptor
                finally:
                    os.close(child_descriptor)
        finally:
            os.close(parent_descriptor)

    return _locked()


def advisory_file_lock(path: Path) -> AbstractContextManager[None]:
    """Build a cross-process exclusive writer lock for one durable path.

    :param path: Lock file outside removable package-version state.
    :returns: Context manager holding the lock around its body.
    """

    @contextmanager
    def _locked() -> Iterator[None]:
        path.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            with _windows_open_regular_lock(path) as stream:
                _windows_lock(stream)
                try:
                    yield
                finally:
                    _windows_unlock(stream)
            return

        import fcntl

        canonical_parent = path.parent.resolve(strict=True)
        directory_descriptor = open_posix_nofollow_directory(canonical_parent)
        try:
            with _posix_advisory_descriptor_lock(directory_descriptor, fcntl.LOCK_EX):
                flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW
                marker_descriptor = os.open(path.name, flags, 0o666, dir_fd=directory_descriptor)
                try:
                    if not stat.S_ISREG(os.fstat(marker_descriptor).st_mode):
                        raise FileExistsError(f"lock is not a regular file: {path}")
                    with _posix_advisory_descriptor_lock(marker_descriptor, fcntl.LOCK_EX):
                        _require_posix_retained_directory(canonical_parent, directory_descriptor)
                        yield
                        _require_posix_retained_directory(canonical_parent, directory_descriptor)
                finally:
                    os.close(marker_descriptor)
        finally:
            os.close(directory_descriptor)

    return _locked()


def _posix_lease_directory_descriptor(directory: Path) -> int:
    """Wait briefly for installer permission publication and open its directory.

    :param directory: Existing synchronization directory published by the installer.
    :returns: Read-only directory descriptor.
    :raises FileNotFoundError: The directory is removed for ten seconds.
    :raises PermissionError: Permissions remain unpublished after ten seconds.
    """
    deadline = time.monotonic() + _LEASE_DIRECTORY_TIMEOUT_SECONDS
    while True:
        try:
            return _posix_create_directory_descriptor(directory)
        except (FileNotFoundError, PermissionError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(_LOCK_RETRY_INTERVAL_SECONDS)


def _posix_consumer_marker_descriptor(path: Path, directory_descriptor: int) -> int:
    """Open the durable marker read-only, creating it only when absent.

    :param path: Marker path represented relative to the directory descriptor.
    :param directory_descriptor: Open synchronization directory.
    :returns: Descriptor suitable for a shared flock.
    :raises FileExistsError: The marker is not a regular file.
    :raises OSError: Publishing a newly created marker's permissions fails.
    """
    flags = os.O_RDONLY | os.O_NOFOLLOW
    try:
        marker = os.open(path.name, flags, dir_fd=directory_descriptor)
    except FileNotFoundError:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        try:
            marker = os.open(path.name, flags, 0o666, dir_fd=directory_descriptor)
            try:
                os.fchmod(marker, os.fstat(marker).st_mode | RUNTIME_FILE_READ_MASK)
            except OSError:
                os.close(marker)
                raise
        except FileExistsError:
            marker = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_descriptor)
    if stat.S_ISREG(os.fstat(marker).st_mode):
        return marker
    os.close(marker)
    raise FileExistsError(f"lock is not a regular file: {path}")


def _require_posix_retained_directory(path: Path, descriptor: int) -> None:
    """Require a lexical lock directory to still name its retained inode.

    :param path: Lexical synchronization directory.
    :param descriptor: Retained directory descriptor.
    :raises ValueError: The directory is missing or was replaced.
    """
    retained = os.fstat(descriptor)
    try:
        lexical = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"lock directory was replaced: {path}") from exc
    if (retained.st_dev, retained.st_ino) != (lexical.st_dev, lexical.st_ino):
        raise ValueError(f"lock directory was replaced: {path}")


def _posix_advisory_lease(path: Path) -> AbstractContextManager[None]:
    """Acquire shared directory and marker locks for a POSIX consumer.

    :param path: Durable marker shared with writers from all supported releases.
    :returns: Context manager holding both shared locks.
    """

    @contextmanager
    def _leased() -> Iterator[None]:
        import fcntl

        with ExitStack() as resources:
            directory = _posix_lease_directory_descriptor(path.parent)
            resources.callback(os.close, directory)
            resources.enter_context(_posix_advisory_descriptor_lock(directory, fcntl.LOCK_SH))
            marker = _posix_consumer_marker_descriptor(path, directory)
            resources.callback(os.close, marker)
            resources.enter_context(_posix_advisory_descriptor_lock(marker, fcntl.LOCK_SH))
            _require_posix_retained_directory(path.parent, directory)
            yield
            _require_posix_retained_directory(path.parent, directory)

    return _leased()


def advisory_file_lease(path: Path) -> AbstractContextManager[None]:
    """Build a shared POSIX consumer lease without writing existing markers.

    POSIX consumers lock both the marker directory and marker inode. Windows retains the exclusive
    writable file lock.

    :param path: Durable lock file shared with exclusive writers.
    :returns: Context manager holding the consumer lease around its body.
    """

    @contextmanager
    def _leased() -> Iterator[None]:
        if os.name == "nt":
            with advisory_file_lock(path):
                yield
            return
        with _posix_advisory_lease(path):
            yield

    return _leased()


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


def windows_retained_nofollow_directories(
    plugins_dir: Path,
    lock_parent: Path,
) -> AbstractContextManager[list[Path]]:
    """Build a Windows hierarchy while handles prevent component replacement.

    :param plugins_dir: Managed storage root.
    :param lock_parent: Parent directory of the package lock marker.
    :returns: Context manager yielding validated root-first directories.
    """

    @contextmanager
    def _retained() -> Iterator[list[Path]]:
        plugins_dir.mkdir(parents=True, exist_ok=True)
        directories: list[Path] = []
        handles: list[int] = []
        current = plugins_dir
        try:
            for component in (None, *lock_parent.relative_to(plugins_dir).parts):
                if component is not None:
                    current /= component
                    current.mkdir(exist_ok=True)
                handle = _windows_open_directory_handle(current)
                handles.append(handle)
                directories.append(current)
            yield directories
        finally:
            for handle in reversed(handles):
                _windows_close_handle(handle)

    return _retained()


def _posix_lock_hierarchy(
    plugins_dir: Path,
    lock_parent: Path,
) -> AbstractContextManager[list[int]]:
    """Open a package-lock hierarchy through retained no-follow descriptors.

    :param plugins_dir: Managed storage root.
    :param lock_parent: Parent directory of the package lock marker.
    :returns: Context manager yielding descriptors from managed root to lock parent.
    """

    @contextmanager
    def _opened() -> Iterator[list[int]]:
        plugins_dir.mkdir(parents=True, exist_ok=True)
        descriptors = [open_posix_nofollow_directory(plugins_dir)]
        try:
            for component in lock_parent.relative_to(plugins_dir).parts:
                try:
                    os.mkdir(component, dir_fd=descriptors[-1])
                except FileExistsError:
                    pass
                flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                try:
                    descriptor = os.open(component, flags, dir_fd=descriptors[-1])
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise FileExistsError(
                            f"install lock hierarchy component is not a directory: {component}"
                        ) from exc
                    raise
                descriptors.append(descriptor)
            yield descriptors
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    return _opened()


def _publish_posix_lock_hierarchy(descriptors: list[int]) -> None:
    """Publish package-lock traversal from the deepest directory outward.

    :param descriptors: Retained hierarchy descriptors ordered root-first.
    """
    for descriptor in reversed(descriptors):
        os.fchmod(
            descriptor,
            os.fstat(descriptor).st_mode | RUNTIME_DIRECTORY_ACCESS_MASK,
        )


def _require_posix_lock_hierarchy(
    plugins_dir: Path,
    lock_parent: Path,
    descriptors: list[int],
) -> None:
    """Require every retained install-lock directory to keep its lexical name.

    :param plugins_dir: Managed storage root.
    :param lock_parent: Deepest synchronization directory.
    :param descriptors: Retained hierarchy descriptors ordered root-first.
    """
    paths = [plugins_dir]
    for component in lock_parent.relative_to(plugins_dir).parts:
        paths.append(paths[-1] / component)
    for path, descriptor in zip(paths, descriptors, strict=True):
        _require_posix_retained_directory(path, descriptor)


def _windows_package_install_lock(
    path: Path,
    plugins_dir: Path,
) -> AbstractContextManager[None]:
    """Acquire and publish one Windows package lock.

    :param path: Durable package marker.
    :param plugins_dir: Managed storage root.
    :returns: Context manager holding the exclusive lock.
    """

    @contextmanager
    def _locked() -> Iterator[None]:
        with windows_retained_nofollow_directories(plugins_dir, path.parent):
            try:
                marker_mode = path.lstat().st_mode
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISREG(marker_mode):
                    raise FileExistsError(f"install lock is not a regular file: {path}")
            with advisory_file_lock(path):
                yield

    return _locked()


def _posix_package_install_lock(
    path: Path,
    plugins_dir: Path,
) -> AbstractContextManager[None]:
    """Acquire and publish one descriptor-relative POSIX package lock.

    :param path: Durable package marker.
    :param plugins_dir: Managed storage root.
    :returns: Context manager holding directory and marker locks.
    """

    @contextmanager
    def _locked() -> Iterator[None]:
        import fcntl

        with _posix_lock_hierarchy(plugins_dir, path.parent) as descriptors:
            with _posix_advisory_descriptor_lock(descriptors[-1], fcntl.LOCK_EX):
                flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW
                marker = os.open(path.name, flags, 0o666, dir_fd=descriptors[-1])
                try:
                    marker_mode = os.fstat(marker).st_mode
                    if not stat.S_ISREG(marker_mode):
                        raise FileExistsError(f"install lock is not a regular file: {path}")
                    os.fchmod(marker, marker_mode | RUNTIME_FILE_READ_MASK)
                    _publish_posix_lock_hierarchy(descriptors)
                    with _posix_advisory_descriptor_lock(marker, fcntl.LOCK_EX):
                        _require_posix_lock_hierarchy(plugins_dir, path.parent, descriptors)
                        yield
                        _require_posix_lock_hierarchy(plugins_dir, path.parent, descriptors)
                finally:
                    os.close(marker)

    return _locked()


def package_install_lock(
    package: str,
    version: str,
    plugins_dir: Path,
) -> AbstractContextManager[None]:
    """Acquire the installer lock and publish runtime-readable permissions.

    Runtime consumers call :func:`advisory_file_lease` directly and never enter
    this permission-normalizing installer wrapper.

    :param package: Studiorack ``organization/package`` slug.
    :param version: Exact package version.
    :param plugins_dir: Resolved managed storage root.
    :returns: Context manager holding the package lock around its body.
    """
    path = package_install_lock_path(package, version, plugins_dir)
    if os.name == "nt":
        return _windows_package_install_lock(path, plugins_dir)
    return _posix_package_install_lock(path, plugins_dir)
