"""Verify pinned Studiorack artifacts and expose sealed VST3 bundle aliases.

Typical module usage validates the repository lock before installing its manifest::

    manifest = PluginManifest.load(Path("studiorack.json"))
    ArtifactLock.load(Path("studiorack.lock.json"), manifest)
    install_plugins(
        manifest.selected(()),
        artifact_lock=Path("studiorack.lock.json"),
        plugins_dir=default_plugins_dir(),
        studiorack_executable=Path("node_modules/.bin/studiorack"),
    )
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pydantic
import tenacity

import synth_setter.plugin_integrity as integrity
import synth_setter.plugin_runtime as runtime
from synth_setter.plugin_integrity import (
    ArtifactLock,
    BundleEntry,
    BundleSeal,
    LockedArtifact,
    LockedPackage,
    ManagedBundleRecord,
    PluginIntegrityError,
    seal_plugin_bundle,
)
from synth_setter.plugin_runtime import (
    ManagedAliasRecord,
    managed_plugin_digest,
    plugin_bundle_version,
    validate_plugin_bundle_for_runtime,
)

__all__ = [
    "ArtifactLock",
    "BundleEntry",
    "BundleSeal",
    "LockedArtifact",
    "LockedPackage",
    "ManagedAliasRecord",
    "ManagedBundleRecord",
    "ManagedPlugin",
    "PluginIntegrityError",
    "PluginManifest",
    "adopt_plugin_bundle",
    "default_plugins_dir",
    "default_system_vst3_dirs",
    "install_plugins",
    "link_plugin",
    "managed_plugin_digest",
    "plugin_bundle_version",
    "resolve_plugin_bundle",
    "seal_plugin_bundle",
    "validate_plugin_bundle_for_runtime",
]

_SEMVER_NUMBER = r"(?:0|[1-9][0-9]*)"
_SEMVER_PRERELEASE_IDENTIFIER = rf"(?:{_SEMVER_NUMBER}|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
_EXACT_SEMVER = re.compile(
    rf"^{_SEMVER_NUMBER}\.{_SEMVER_NUMBER}\.{_SEMVER_NUMBER}"
    rf"(?:-{_SEMVER_PRERELEASE_IDENTIFIER}(?:\.{_SEMVER_PRERELEASE_IDENTIFIER})*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_PACKAGE_SLUG = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]*/[0-9A-Za-z][0-9A-Za-z._-]*$")
_INSTALL_LOCK_DIR = ".synth-setter-install-locks"
_NATIVE_TRANSACTION_DIR = ".synth-setter-native-install"
_STUDIORACK_ATTEMPTS = 3
_STUDIORACK_TIMEOUT_SECONDS = 900
# These messages are stable contracts from patched core/ManagerLocal 3.0.6 or HTTP status text.
_PERMANENT_STUDIORACK_ERROR_PATTERNS = (
    re.compile(r"\bartifact lock (?:mismatch for|missing) [0-9A-Za-z._-]+/[0-9A-Za-z._-]+@[^\s]+"),
    re.compile(r"\bStudiorack artifact lock path is required\b"),
    re.compile(r"\bENOENT: no such file or directory, open\b"),
    re.compile(r"\bSyntaxError: (?:.*\bis not valid JSON\b|(?:Expected|Unexpected).*\bJSON\b)"),
    re.compile(r"\bInvalid package (?:slug|version): [^\r\n]+"),
    re.compile(
        r"\bPackage [0-9A-Za-z._-]+/[0-9A-Za-z._-]+"
        r"(?: version [^\s]+)? not found in registry\b"
    ),
    re.compile(
        r"\bNo compatible files found(?: to install)? for [0-9A-Za-z._-]+/[0-9A-Za-z._-]+\b"
    ),
    re.compile(r"\b(?:401 Unauthorized|403 Forbidden)\b"),
    re.compile(r"\bHTTP(?:Error)?(?: status(?: code)?)?[: ]+404(?: Not Found)?\b"),
)


@dataclass(frozen=True)
class ManagedPlugin:
    """One exact Studiorack package and its expected VST3 bundle.

    .. attribute :: package

        ``organization/package`` registry slug.

    .. attribute :: version

        Exact package version.

    .. attribute :: renderer_version

        Exact version reported by the VST3 renderer.

    .. attribute :: bundle

        VST3 bundle basename exposed to synth-setter.
    """

    package: str
    version: str
    renderer_version: str
    bundle: str

    @property
    def reference(self) -> str:
        """Return the package reference accepted by ``studiorack plugins install``."""
        return f"{self.package}@{self.version}"


class NativeInstallTransaction(pydantic.BaseModel):
    """Persisted pre-install identity for resumable native adoption.

    .. attribute :: model_config

        Pydantic validation settings.

    .. attribute :: bundle

        Expected native VST3 bundle basename.

    .. attribute :: candidates

        Absolute candidate paths mapped to their original content identities.

    .. attribute :: locked_package_sha256

        Canonical artifact identity governing this install attempt.

    .. attribute :: package_reference

        Exact package version being installed.

    .. attribute :: schema_version

        Transaction-record schema version.
    """

    model_config = pydantic.ConfigDict(
        strict=True, extra="forbid", frozen=True, populate_by_name=True
    )

    bundle: str
    candidates: dict[str, list[BundleEntry] | None]
    locked_package_sha256: str = pydantic.Field(pattern=r"^[0-9a-f]{64}$")
    package_reference: str
    schema_version: Literal[1] = pydantic.Field(default=1, alias="schema")


@dataclass(frozen=True)
class _InstallContext:
    plugin: ManagedPlugin
    plugins_dir: Path
    executable: str
    env: dict[str, str]
    roots: tuple[Path, ...]
    locked_package: LockedPackage

    @property
    def managed_bundle(self) -> Path:
        """Return the exact managed output path for this install.

        :returns: Versioned VST3 bundle path.
        """
        return _managed_version_dir(self.plugin, self.plugins_dir) / self.plugin.bundle

    @property
    def studiorack_argv(self) -> list[str]:
        """Return the exact package installation argument vector.

        :returns: Studiorack install command arguments.
        """
        return [self.executable, "plugins", "install", self.plugin.reference]


class PluginManifest(pydantic.BaseModel):
    """Validated Studiorack project manifest with synth-setter bundle metadata.

    .. attribute :: model_config

        Pydantic validation settings.

    .. attribute :: name

        Project name shown in Studiorack-compatible metadata.

    .. attribute :: type

        Open Audio Stack package type; synth-setter manifests are projects.

    .. attribute :: plugins

        Studiorack package slug to exact version.

    .. attribute :: vst3_bundles

        Package slug to the VST3 bundle synth-setter loads.

    .. attribute :: vst3_versions

        Package slug to the exact version reported by the installed VST3, when declared.
    """

    model_config = pydantic.ConfigDict(
        strict=True, extra="forbid", populate_by_name=True, frozen=True
    )

    name: str
    type: Literal["project"]
    plugins: dict[str, str]
    vst3_bundles: dict[str, str] = pydantic.Field(alias="vst3Bundles")
    vst3_versions: dict[str, str] | None = pydantic.Field(default=None, alias="vst3Versions")

    @pydantic.model_validator(mode="after")
    def _validate_packages(self) -> PluginManifest:
        """Require exact versions and one bundle mapping per package.

        :returns: The validated manifest.
        :raises ValueError: A slug/version is invalid or metadata keys differ from plugin keys.
        """
        package_keys = self.plugins.keys()
        if package_keys != self.vst3_bundles.keys():
            raise ValueError("plugins and vst3Bundles must contain the same package keys")
        vst3_versions = self.vst3_versions
        if vst3_versions is not None and package_keys != vst3_versions.keys():
            raise ValueError("plugins and vst3Versions must contain the same package keys")
        for package, package_version in self.plugins.items():
            if _PACKAGE_SLUG.fullmatch(package) is None:
                raise ValueError(f"invalid Studiorack package slug: {package!r}")
            if _EXACT_SEMVER.fullmatch(package_version) is None:
                raise ValueError(f"{package} version must be an exact semantic version")
            renderer_version = package_version if vst3_versions is None else vst3_versions[package]
            if _EXACT_SEMVER.fullmatch(renderer_version) is None:
                raise ValueError(f"{package} VST3 version must be an exact semantic version")
            bundle = self.vst3_bundles[package]
            if Path(bundle).name != bundle or not bundle.endswith(".vst3"):
                raise ValueError(f"{package} bundle must be a .vst3 basename")
        return self

    @classmethod
    def load(cls, path: Path) -> PluginManifest:
        """Parse and validate a UTF-8 JSON manifest.

        :param path: Manifest path.
        :returns: Validated project manifest.
        """
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def resolve(self, package: str) -> ManagedPlugin:
        """Resolve a package slug to its exact version and bundle.

        :param package: ``organization/package`` slug.
        :returns: Managed plugin metadata.
        :raises KeyError: The package is absent from the manifest.
        """
        try:
            package_version = self.plugins[package]
            return ManagedPlugin(
                package=package,
                version=package_version,
                renderer_version=(
                    package_version if self.vst3_versions is None else self.vst3_versions[package]
                ),
                bundle=self.vst3_bundles[package],
            )
        except KeyError:
            raise KeyError(package) from None

    def selected(self, packages: Sequence[str]) -> tuple[ManagedPlugin, ...]:
        """Return selected plugins, or every plugin when no selection is given.

        :param packages: Requested package slugs.
        :returns: Plugins in manifest order.
        """
        selected = packages or tuple(self.plugins)
        return tuple(self.resolve(package) for package in selected)


def default_plugins_dir() -> Path:
    """Return the dedicated Studiorack storage root for synth-setter.

    :returns: Environment override or the platform user-data path.
    """
    override = os.environ.get("STUDIORACK_PLUGINS_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/synth-setter/studiorack"
    return Path.home() / ".local/share/synth-setter/studiorack"


def default_system_vst3_dirs() -> tuple[Path, ...]:
    """Return platform VST3 directories used by native installer packages.

    :returns: VST3 roots searched for installer-controlled bundles.
    """
    if sys.platform == "darwin":
        return (
            Path.home() / "Library/Audio/Plug-Ins/VST3",
            Path("/Library/Audio/Plug-Ins/VST3"),
        )
    if sys.platform.startswith("linux"):
        multiarch = sorted(Path("/usr/lib").glob("*-linux-gnu/vst3"))
        return (Path("/usr/lib/vst3"), Path("/usr/local/lib/vst3"), *multiarch)
    local_app_data = os.environ.get("LOCALAPPDATA")
    windows_dirs = [Path("C:/Program Files/Common Files/VST3")]
    if local_app_data:
        windows_dirs.insert(0, Path(local_app_data) / "Programs/Common/VST3")
    return tuple(windows_dirs)


def resolve_plugin_bundle(
    plugin: ManagedPlugin,
    plugins_dir: Path,
    *,
    artifact_lock: ArtifactLock,
) -> Path:
    """Resolve a package only when content and current provenance are valid.

    :param plugin: Exact package metadata.
    :param plugins_dir: Studiorack ``pluginsDir`` value.
    :param artifact_lock: Current repository artifact identities.
    :returns: Existing bundle path in Studiorack's exact-version tree.
    :raises FileNotFoundError: The expected bundle is absent or invalid.
    """
    managed = plugins_dir.expanduser().resolve() / "VST3" / plugin.package / plugin.version
    managed_bundle = managed / plugin.bundle
    try:
        is_directory = stat.S_ISDIR(managed_bundle.stat().st_mode)
    except FileNotFoundError:
        is_directory = False
    if is_directory and integrity.bundle_is_sealed(
        managed_bundle,
        plugin,
        artifact_lock.package_for(plugin),
    ):
        return managed_bundle
    raise FileNotFoundError(
        f"{plugin.bundle} failed managed bundle integrity for {plugin.reference}; run "
        f"`studiorack plugins install {plugin.reference}` and retry"
    )


def link_plugin(
    plugin: ManagedPlugin,
    *,
    artifact_lock: ArtifactLock,
    plugins_dir: Path,
    links_dir: Path,
) -> Path:
    """Link a managed bundle into synth-setter's stable namespace.

    :param plugin: Exact package metadata.
    :param artifact_lock: Current repository artifact identities.
    :param plugins_dir: Studiorack storage root.
    :param links_dir: Checkout directory containing stable aliases.
    :returns: Stable alias path.
    :raises FileExistsError: A different real file or directory occupies the alias.
    """
    resolved_dir = plugins_dir.expanduser().resolve()
    with _package_install_lock(plugin, resolved_dir):
        bundle = resolve_plugin_bundle(plugin, resolved_dir, artifact_lock=artifact_lock)
        links_dir.mkdir(parents=True, exist_ok=True)
        alias = links_dir / plugin.bundle
        with _alias_publication_lock(alias):
            try:
                is_same_bundle = alias.resolve(strict=True) == bundle.resolve(strict=True)
            except FileNotFoundError:
                is_same_bundle = False
            if is_same_bundle and not alias.is_symlink():
                runtime.record_managed_alias(alias, bundle)
            else:
                if alias.exists() and not alias.is_symlink():
                    raise FileExistsError(f"refusing to replace {alias}: it is not a symlink")
                runtime.replace_managed_alias(alias, bundle)
    validate_plugin_bundle_for_runtime(alias)
    return alias


def _alias_publication_lock(alias: Path) -> AbstractContextManager[None]:
    """Serialize publishers that share one stable consumer alias.

    :param alias: Stable checkout alias.
    :returns: Context manager holding the alias-specific publication lock.
    """
    return integrity.advisory_file_lock(alias.parent / f".{alias.name}.synth-setter.lock")


def _package_install_lock(
    plugin: ManagedPlugin, plugins_dir: Path
) -> AbstractContextManager[None]:
    """Build the transaction lock for one exact package version.

    :param plugin: Exact package identity selecting the lock.
    :param plugins_dir: Resolved managed storage root.
    :returns: Context manager serializing installation and publication.
    """
    organization, package = plugin.package.split("/")
    lock_parent = plugins_dir / _INSTALL_LOCK_DIR / organization / package
    _ensure_managed_version_dir(lock_parent, plugins_dir)
    return integrity.package_install_lock(plugin.package, plugin.version, plugins_dir)


def _managed_version_dir(plugin: ManagedPlugin, plugins_dir: Path) -> Path:
    """Map exact package identity into Studiorack's VST3 version tree.

    :param plugin: Exact package metadata.
    :param plugins_dir: Studiorack storage root.
    :returns: Exact managed version directory.
    """
    organization, package = plugin.package.split("/")
    return plugins_dir / "VST3" / organization / package / plugin.version


def _validate_managed_parent_chain(version_dir: Path, plugins_dir: Path) -> None:
    """Reject existing non-directory or symlinked managed parents.

    :param version_dir: Exact package-version directory to validate.
    :param plugins_dir: Resolved Studiorack storage root.
    :raises FileExistsError: A parent component is not a directory.
    :raises ValueError: A parent component is a symlink.
    """
    current = plugins_dir
    for component in version_dir.relative_to(plugins_dir).parts:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"refusing managed parent symlink {current}")
        if not stat.S_ISDIR(mode):
            raise FileExistsError(f"managed parent is not a directory: {current}")


def _ensure_managed_version_dir(version_dir: Path, plugins_dir: Path) -> None:
    _validate_managed_parent_chain(version_dir, plugins_dir)
    plugins_dir.mkdir(parents=True, exist_ok=True)
    current = plugins_dir
    for component in version_dir.relative_to(plugins_dir).parts:
        current /= component
        current.mkdir(exist_ok=True)
        _validate_managed_parent_chain(current, plugins_dir)


def _adoption_paths(
    plugin: ManagedPlugin,
    plugins_dir: Path,
    bundle: Path,
) -> tuple[Path, Path, Path]:
    try:
        source = bundle.expanduser().resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"fallback bundle is not installed: {bundle.expanduser()}"
        ) from None
    if not stat.S_ISDIR(source.stat().st_mode):
        raise ValueError(f"bundle is not a directory: {bundle}")
    resolved_dir = plugins_dir.expanduser().resolve()
    version_dir = _managed_version_dir(plugin, resolved_dir)
    _validate_managed_parent_chain(version_dir, resolved_dir)
    return source, version_dir, version_dir / plugin.bundle


def _managed_adoption_matches(managed: Path, source: Path) -> bool:
    target = runtime.managed_alias_target(managed)
    return target is not None and target.resolve(strict=True) == source


def _reseal_adopted_bundle(
    plugin: ManagedPlugin,
    source: Path,
    managed: Path,
    *,
    locked_package: LockedPackage,
    source_kind: Literal["artifact-lock", "explicit"],
) -> Path:
    runtime.record_managed_alias(managed, managed)
    seal_plugin_bundle(
        source,
        plugin,
        locked_package=locked_package,
        record_for=managed,
        source_kind=source_kind,
    )
    return managed


def _create_adopted_bundle_alias(
    plugin: ManagedPlugin,
    source: Path,
    managed: Path,
    *,
    locked_package: LockedPackage,
    source_kind: Literal["artifact-lock", "explicit"],
) -> Path:
    seal_plugin_bundle(
        source,
        plugin,
        locked_package=locked_package,
        record_for=managed,
        source_kind=source_kind,
    )
    try:
        managed.symlink_to(source, target_is_directory=True)
        runtime.record_managed_alias(managed, managed)
    except OSError:
        managed.unlink(missing_ok=True)
        runtime.discard_managed_bundle_records(managed)
        raise
    return managed


def adopt_plugin_bundle(
    plugin: ManagedPlugin,
    *,
    plugins_dir: Path,
    bundle: Path,
    locked_package: LockedPackage,
) -> Path:
    """Record an explicitly adopted source bundle under its exact package lock.

    :param plugin: Exact package metadata corresponding to the fallback build.
    :param plugins_dir: Studiorack storage root.
    :param bundle: Existing installer- or source-controlled VST3 bundle.
    :param locked_package: Repository identity for the explicitly adopted pin.
    :returns: Versioned managed alias.
    """
    resolved_dir = plugins_dir.expanduser().resolve()
    with _package_install_lock(plugin, resolved_dir):
        return _adopt_plugin_bundle(
            plugin,
            plugins_dir=resolved_dir,
            bundle=bundle,
            locked_package=locked_package,
            source_kind="explicit",
        )


def _require_renderer_version(plugin: ManagedPlugin, bundle: Path) -> None:
    actual_version = plugin_bundle_version(bundle)
    if actual_version != plugin.renderer_version:
        raise ValueError(
            f"expected {plugin.renderer_version} for {plugin.bundle}, found {actual_version}"
        )


def _adopt_plugin_bundle(
    plugin: ManagedPlugin,
    *,
    plugins_dir: Path,
    bundle: Path,
    locked_package: LockedPackage,
    source_kind: Literal["artifact-lock", "explicit"],
) -> Path:
    source, version_dir, managed = _adoption_paths(plugin, plugins_dir, bundle)
    is_same_source = _managed_adoption_matches(managed, source)
    if is_same_source and integrity.bundle_is_sealed(managed, plugin, locked_package):
        return managed
    if (managed.exists() or managed.is_symlink()) and not is_same_source:
        raise FileExistsError(f"refusing to replace managed bundle {managed}")

    _require_renderer_version(plugin, source)
    resolved_dir = plugins_dir.expanduser().resolve()
    _ensure_managed_version_dir(version_dir, resolved_dir)
    if is_same_source:
        return _reseal_adopted_bundle(
            plugin,
            source,
            managed,
            locked_package=locked_package,
            source_kind=source_kind,
        )
    return _create_adopted_bundle_alias(
        plugin,
        source,
        managed,
        locked_package=locked_package,
        source_kind=source_kind,
    )


def _snapshot_bundle(bundle: Path) -> list[BundleEntry] | None:
    try:
        return integrity.bundle_entries(bundle)
    except (FileNotFoundError, ValueError):
        return None


def _remove_path(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISDIR(mode):
        shutil.rmtree(path)
    else:
        path.unlink()


def _remove_managed_version(plugin: ManagedPlugin, plugins_dir: Path) -> None:
    """Remove one exact VST3 package version without following symlinks.

    :param plugin: Exact package identity to clear.
    :param plugins_dir: Resolved Studiorack storage root.
    """
    version_dir = _managed_version_dir(plugin, plugins_dir)
    _validate_managed_parent_chain(version_dir, plugins_dir)
    _remove_path(version_dir)


def _native_candidates(plugin: ManagedPlugin, roots: Iterable[Path]) -> tuple[Path, ...]:
    """Build native candidates without resolving missing installer output.

    :param plugin: Package naming the expected bundle.
    :param roots: Platform VST3 search roots.
    :returns: Absolute lexical candidate paths.
    """
    return tuple((root.expanduser() / plugin.bundle).absolute() for root in roots)


def _new_native_transaction(
    plugin: ManagedPlugin,
    candidates: tuple[Path, ...],
    locked_package: LockedPackage,
) -> NativeInstallTransaction:
    return NativeInstallTransaction(
        bundle=plugin.bundle,
        candidates={str(candidate): _snapshot_bundle(candidate) for candidate in candidates},
        locked_package_sha256=integrity.locked_package_digest(
            plugin.reference,
            locked_package,
        ),
        package_reference=plugin.reference,
    )


def _native_transaction_record(plugin: ManagedPlugin, plugins_dir: Path) -> Path:
    organization, package = plugin.package.split("/")
    parent = plugins_dir / _NATIVE_TRANSACTION_DIR / organization / package
    _ensure_managed_version_dir(parent, plugins_dir)
    return parent / f"{plugin.version}.json"


def _prepare_native_transaction(plugin: ManagedPlugin, plugins_dir: Path) -> Path:
    """Reset managed output while retaining the transaction path for diagnostics.

    :param plugin: Exact package being installed.
    :param plugins_dir: Resolved Studiorack storage root.
    :returns: Transaction-record path overwritten before each installer attempt.
    """
    version_dir = _managed_version_dir(plugin, plugins_dir)
    _validate_managed_parent_chain(version_dir, plugins_dir)
    record = _native_transaction_record(plugin, plugins_dir)
    _remove_managed_version(plugin, plugins_dir)
    return record


def _persist_native_transaction(
    context: _InstallContext,
    candidates: tuple[Path, ...],
    record: Path,
) -> NativeInstallTransaction:
    transaction = _new_native_transaction(
        context.plugin,
        candidates,
        context.locked_package,
    )
    serialized = transaction.model_dump_json(indent=2, exclude_none=False, by_alias=True)
    integrity.write_atomic_record(record, serialized)
    return transaction


def _changed_native_candidates(transaction: NativeInstallTransaction) -> list[Path]:
    changed: list[Path] = []
    for candidate_text, original in transaction.candidates.items():
        candidate = Path(candidate_text)
        current = _snapshot_bundle(candidate)
        if current is not None and current != original:
            changed.append(candidate)
    return changed


def _adopt_native_transaction(
    context: _InstallContext,
    transaction: NativeInstallTransaction,
    record: Path,
) -> bool:
    changed = _changed_native_candidates(transaction)
    if not changed:
        return False
    if len(changed) != 1:
        raise FileNotFoundError(
            f"installer did not create or change exactly one {context.plugin.bundle} candidate"
        )
    _adopt_plugin_bundle(
        context.plugin,
        plugins_dir=context.plugins_dir,
        bundle=changed[0],
        locked_package=context.locked_package,
        source_kind="artifact-lock",
    )
    record.unlink()
    return True


def _invoke_studiorack(argv: Sequence[str], env: dict[str, str]) -> None:
    result = subprocess.run(  # noqa: S603 — fixed executable and validated argv
        argv,
        check=False,
        env=env,
        stderr=subprocess.PIPE,
        text=True,
        timeout=_STUDIORACK_TIMEOUT_SECONDS,
    )
    if result.returncode == 0:
        return
    stderr = result.stderr or ""
    # Studiorack 3.0.6 shares one exit code; retry policy matches only its stable stderr contracts.
    raise subprocess.CalledProcessError(result.returncode, argv, stderr=stderr)


def _is_retryable_studiorack_exit(exc: BaseException) -> bool:
    """Classify a Studiorack process failure using stable permanent messages.

    :param exc: Failure raised by the subprocess boundary.
    :returns: Whether the failure lacks a permanent classification and may be retried.
    """
    if isinstance(exc, subprocess.TimeoutExpired):
        return True
    if not isinstance(exc, subprocess.CalledProcessError):
        return False
    stderr = exc.stderr or ""
    return not any(pattern.search(stderr) for pattern in _PERMANENT_STUDIORACK_ERROR_PATTERNS)


def _studiorack_retrying() -> tenacity.Retrying:
    """Centralize bounded retries for transient Studiorack exits.

    :returns: Retry iterator configured for transient process failures.
    """
    return tenacity.Retrying(
        reraise=True,
        retry=tenacity.retry_if_exception(_is_retryable_studiorack_exit),
        sleep=time.sleep,
        stop=tenacity.stop_after_attempt(_STUDIORACK_ATTEMPTS),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=4),
    )


def _studiorack_timeout_message() -> str:
    """Render the exhausted timeout policy with recovery guidance.

    :returns: Actionable timeout message for the CLI boundary.
    """
    return (
        f"Studiorack command timed out after {_STUDIORACK_TIMEOUT_SECONDS} seconds on each of "
        f"{_STUDIORACK_ATTEMPTS} attempts; check registry/network availability and retry"
    )


def _run_studiorack(argv: Sequence[str], *, env: dict[str, str]) -> None:
    """Run one Studiorack operation, retrying only transient exits.

    :param argv: Validated command argument vector.
    :param env: Child environment carrying the repository artifact lock.
    :raises RuntimeError: All retry attempts exceed the Studiorack timeout.
    """
    try:
        for attempt in _studiorack_retrying():
            with attempt:
                _invoke_studiorack(argv, env)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(_studiorack_timeout_message()) from exc


def _invoke_native_install_attempt(context: _InstallContext) -> None:
    """Run one installer attempt and discard manager-owned output on failure.

    :param context: Shared package, path, and subprocess state.
    :raises subprocess.CalledProcessError: The Studiorack process exits nonzero.
    :raises subprocess.TimeoutExpired: The Studiorack process exceeds its attempt timeout.
    """
    try:
        _invoke_studiorack(context.studiorack_argv, context.env)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        _remove_managed_version(context.plugin, context.plugins_dir)
        raise


def _run_native_install(
    context: _InstallContext,
    *,
    candidates: tuple[Path, ...],
    record: Path,
) -> NativeInstallTransaction:
    """Retry native installation against one invocation-local original snapshot.

    A retry belongs to the same validated installer operation, so output changed by
    an earlier transiently failing attempt remains causally attributable. A process
    restart writes a fresh snapshot and therefore rejects pre-invocation bytes: no
    installer receipt exists that could authenticate them safely across invocations.

    :param context: Shared package, lock, path, and subprocess state.
    :param candidates: Native output paths snapshotted before the first attempt.
    :param record: Transaction path cleared only after successful adoption.
    :returns: The snapshot preceding every attempt in this invocation.
    :raises RuntimeError: The retry iterator terminates without yielding an outcome.
    """
    transaction = _persist_native_transaction(context, candidates, record)
    try:
        for attempt in _studiorack_retrying():
            with attempt:
                _invoke_native_install_attempt(context)
            outcome = attempt.retry_state.outcome
            if outcome is not None and outcome.failed:
                continue
            return transaction
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(_studiorack_timeout_message()) from exc
    raise RuntimeError("Studiorack retry loop ended without an outcome")


def _install_current_bundle(context: _InstallContext) -> bool:
    managed = context.managed_bundle
    if not integrity.locked_bundle_is_sealed(
        managed,
        context.plugin,
        context.locked_package,
    ):
        return False
    _run_studiorack(context.studiorack_argv, env=context.env)
    if not integrity.locked_bundle_is_sealed(
        managed,
        context.plugin,
        context.locked_package,
    ):
        raise ValueError(f"{context.plugin.bundle} changed during installation")
    return True


def _managed_directory_exists(managed: Path) -> bool:
    """Recognize real managed output but not a native-adoption symlink.

    :param managed: Expected managed bundle path.
    :returns: Whether the path is a real directory.
    """
    try:
        return stat.S_ISDIR(managed.lstat().st_mode)
    except FileNotFoundError:
        return False


def _finish_install_output(
    context: _InstallContext,
    transaction: NativeInstallTransaction,
    record: Path,
) -> bool:
    if _managed_directory_exists(context.managed_bundle):
        _require_renderer_version(context.plugin, context.managed_bundle)
        seal_plugin_bundle(
            context.managed_bundle,
            context.plugin,
            locked_package=context.locked_package,
        )
        record.unlink()
        return True
    return _adopt_native_transaction(context, transaction, record)


def _install_new_bundle(context: _InstallContext) -> None:
    candidates = _native_candidates(context.plugin, context.roots)
    record = _prepare_native_transaction(context.plugin, context.plugins_dir)
    transaction = _run_native_install(context, candidates=candidates, record=record)
    if _finish_install_output(context, transaction, record):
        return
    raise FileNotFoundError(
        f"installer did not create or change exactly one {context.plugin.bundle} candidate"
    )


def _install_plugin(context: _InstallContext) -> None:
    """Install and seal one exact package under its current artifact identity.

    :param context: Shared package, lock, path, and subprocess state.
    """
    with _package_install_lock(context.plugin, context.plugins_dir):
        version_dir = _managed_version_dir(context.plugin, context.plugins_dir)
        _validate_managed_parent_chain(version_dir, context.plugins_dir)
        if _install_current_bundle(context):
            return
        _install_new_bundle(context)


def install_plugins(
    plugins: Iterable[ManagedPlugin],
    *,
    artifact_lock: Path,
    plugins_dir: Path,
    studiorack_executable: Path,
    system_dirs: Iterable[Path] | None = None,
) -> None:
    """Install exact packages through the Studiorack CLI.

    :param plugins: Exact package metadata.
    :param artifact_lock: Repository-controlled artifact lock for the selected manifest.
    :param plugins_dir: Absolute Studiorack storage root.
    :param studiorack_executable: Pinned CLI executable.
    :param system_dirs: Native-installer VST3 roots; platform defaults when omitted.
    :raises FileNotFoundError: The Studiorack executable is unavailable.
    """
    executable = shutil.which(str(studiorack_executable))
    if executable is None:
        raise FileNotFoundError(
            f"Studiorack CLI not found at {studiorack_executable}; run `npm ci` first"
        )
    lock_path = artifact_lock.expanduser().resolve()
    lock = ArtifactLock.model_validate_json(lock_path.read_text(encoding="utf-8"))
    resolved_dir = plugins_dir.expanduser().resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "SYNTH_SETTER_STUDIORACK_ARTIFACT_LOCK": str(lock_path),
    }
    # Persist the lock for elevated children while the environment pins this invocation.
    _run_studiorack([executable, "config", "set", "pluginsDir", str(resolved_dir)], env=env)
    _run_studiorack([executable, "config", "set", "artifactLockPath", str(lock_path)], env=env)
    roots = default_system_vst3_dirs() if system_dirs is None else tuple(system_dirs)
    for plugin in plugins:
        _install_plugin(
            _InstallContext(
                plugin=plugin,
                plugins_dir=resolved_dir,
                executable=executable,
                env=env,
                roots=roots,
                locked_package=lock.package_for(plugin),
            )
        )
