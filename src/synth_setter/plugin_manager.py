"""Resolve pinned Studiorack packages to stable synth-setter plugin paths.

Typical usage loads ``studiorack.json``, installs selected pins, then links their
bundles into a checkout's ``plugins/`` directory.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential

_EXACT_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
_PACKAGE_SLUG = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]*/[0-9A-Za-z][0-9A-Za-z._-]*$")


@dataclass(frozen=True)
class ManagedPlugin:
    """One exact Studiorack package and its expected VST3 bundle.

    .. attribute :: package

        ``organization/package`` registry slug.

    .. attribute :: version

        Exact package version.

    .. attribute :: bundle

        VST3 bundle basename exposed to synth-setter.
    """

    package: str
    version: str
    bundle: str

    @property
    def reference(self) -> str:
        """Return the package reference accepted by ``studiorack plugins install``."""
        return f"{self.package}@{self.version}"


class PluginManifest(BaseModel):
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

    model_config = ConfigDict(strict=True, extra="forbid", populate_by_name=True, frozen=True)

    name: str
    type: Literal["project"]
    plugins: dict[str, str]
    vst3_bundles: dict[str, str] = Field(alias="vst3Bundles")
    vst3_versions: dict[str, str] | None = Field(default=None, alias="vst3Versions")

    @model_validator(mode="after")
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
            if (
                vst3_versions is not None
                and _EXACT_SEMVER.fullmatch(vst3_versions[package]) is None
            ):
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
            return ManagedPlugin(
                package=package,
                version=self.plugins[package],
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


def resolve_plugin_bundle(plugin: ManagedPlugin, plugins_dir: Path) -> Path:
    """Resolve a Studiorack package to the VST3 bundle installed for this host.

    Archive packages live under Studiorack's versioned VST3 tree. Native
    installers may place bundles in platform VST3 directories instead.

    :param plugin: Exact package metadata.
    :param plugins_dir: Studiorack ``pluginsDir`` value.
    :returns: Existing bundle path in Studiorack's exact-version tree.
    :raises FileNotFoundError: The expected bundle is not installed.
    """
    managed = plugins_dir.expanduser().resolve() / "VST3" / plugin.package / plugin.version
    managed_bundle = managed / plugin.bundle
    if managed_bundle.is_dir():
        return managed_bundle

    raise FileNotFoundError(
        f"{plugin.bundle} is not installed for {plugin.reference}; run "
        f"`studiorack plugins install {plugin.reference}` and retry"
    )


def link_plugin(
    plugin: ManagedPlugin,
    *,
    plugins_dir: Path,
    links_dir: Path,
) -> Path:
    """Link a managed bundle into synth-setter's stable ``plugins/`` namespace.

    :param plugin: Exact package metadata.
    :param plugins_dir: Studiorack storage root.
    :param links_dir: Checkout directory containing stable aliases.
    :returns: Stable alias path.
    :raises FileExistsError: A different real file or directory occupies the alias.
    """
    bundle = resolve_plugin_bundle(plugin, plugins_dir)
    links_dir.mkdir(parents=True, exist_ok=True)
    alias = links_dir / plugin.bundle
    if alias.is_symlink():
        if alias.resolve() == bundle.resolve():
            return alias
        alias.unlink()
    elif alias.exists():
        if alias.resolve() == bundle.resolve():
            return alias
        raise FileExistsError(f"refusing to replace {alias}: it is not a symlink")
    alias.symlink_to(bundle.resolve(), target_is_directory=True)
    return alias


def adopt_plugin_bundle(
    plugin: ManagedPlugin,
    *,
    plugins_dir: Path,
    bundle: Path,
) -> Path:
    """Record an exact source/native fallback in Studiorack's versioned tree.

    :param plugin: Exact package metadata corresponding to the fallback build.
    :param plugins_dir: Studiorack storage root.
    :param bundle: Existing installer- or source-controlled VST3 bundle.
    :returns: Versioned managed alias.
    :raises FileNotFoundError: The fallback bundle does not exist.
    :raises FileExistsError: A different managed bundle already occupies the path.
    """
    source = bundle.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"fallback bundle is not installed: {source}")
    managed = (
        plugins_dir.expanduser().resolve()
        / "VST3"
        / plugin.package
        / plugin.version
        / plugin.bundle
    )
    managed.parent.mkdir(parents=True, exist_ok=True)
    if managed.is_symlink() and managed.resolve() == source:
        return managed
    if managed.exists() or managed.is_symlink():
        raise FileExistsError(f"refusing to replace managed bundle {managed}")
    managed.symlink_to(source, target_is_directory=True)
    return managed


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
)
def _run_studiorack(argv: Sequence[str]) -> None:
    """Run one retryable Studiorack operation.

    :param argv: Validated command argument vector.
    """
    subprocess.run(argv, check=True)  # noqa: S603 — fixed executable and validated argv


def install_plugins(
    plugins: Iterable[ManagedPlugin],
    *,
    plugins_dir: Path,
    studiorack_executable: Path,
    system_dirs: Iterable[Path] | None = None,
) -> None:
    """Install exact packages through the Studiorack CLI.

    :param plugins: Exact package metadata.
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
    resolved_dir = plugins_dir.expanduser().resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    _run_studiorack([executable, "config", "set", "pluginsDir", str(resolved_dir)])
    roots = default_system_vst3_dirs() if system_dirs is None else tuple(system_dirs)
    for plugin in plugins:
        _run_studiorack([executable, "plugins", "install", plugin.reference])
        managed = resolved_dir / "VST3" / plugin.package / plugin.version / plugin.bundle
        if managed.is_dir():
            continue
        native_candidates = (root.expanduser() / plugin.bundle for root in roots)
        native = next((candidate for candidate in native_candidates if candidate.is_dir()), None)
        if native is not None:
            adopt_plugin_bundle(plugin, plugins_dir=resolved_dir, bundle=native)
