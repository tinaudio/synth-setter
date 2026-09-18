"""Compare the Studiorack locks against the live registry and report drift.

The locks pin artifacts from a registry that is regenerated upstream. Without this report, drift
surfaces only as a red image build on every branch (#3623).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REGISTRY_INDEX_URL = (
    "https://open-audio-stack.github.io/open-audio-stack-registry/plugins/index.json"
)


@dataclass(frozen=True)
class Artifact:
    """One downloadable artifact, identified by URL across lock and registry.

    .. attribute :: url

        Download URL; the artifact's identity.

    .. attribute :: platforms

        ``(system, arch)`` pairs the artifact installs on.

    .. attribute :: type

        Registry artifact kind, e.g. ``installer`` or ``archive``.

    .. attribute :: sha256

        Pinned or published digest, or ``None`` when absent.
    """

    url: str
    platforms: frozenset[tuple[str, str]]
    type: str
    sha256: str | None

    @property
    def platform(self) -> str:
        """Render the artifact's platforms for the report.

        :returns: Sorted ``system/arch`` pairs joined by commas, or ``"any"``.
        """
        pairs = sorted(f"{system}/{arch}" for system, arch in self.platforms)
        return ", ".join(pairs) if pairs else "any"


@dataclass(frozen=True)
class PackageDrift:
    """Per-package differences between a pinned lock entry and the live registry.

    ``locked`` and ``live`` carry the complete lists, not just the differing entries,
    so a re-pin can be written from the report without a second live fetch.

    .. attribute :: package

        ``slug@version`` the lock pins.

    .. attribute :: absent

        Locked artifacts the registry no longer offers.

    .. attribute :: added

        Live artifacts on a locked platform that the lock does not pin.

    .. attribute :: changed

        ``(locked, live)`` pairs sharing a URL but not a digest.

    .. attribute :: locked

        Every artifact the lock pins.

    .. attribute :: live

        Every artifact the registry offers for this version.
    """

    package: str
    absent: tuple[Artifact, ...]
    added: tuple[Artifact, ...]
    changed: tuple[tuple[Artifact, Artifact], ...]
    locked: tuple[Artifact, ...]
    live: tuple[Artifact, ...]


def _platforms(systems: Sequence[Any], architectures: Sequence[Any]) -> frozenset[tuple[str, str]]:
    """Return an artifact's ``(system, arch)`` pairs, tolerating both document shapes.

    The lock stores systems as plain strings; the registry nests them as objects.

    :param systems: System entries, each a string or a mapping carrying ``type``.
    :param architectures: Architecture names.
    :returns: Every system/architecture pair the artifact covers.
    """
    names = [system["type"] if isinstance(system, dict) else str(system) for system in systems]
    return frozenset((name, str(arch)) for name in names for arch in architectures)


def _artifacts(entries: Sequence[Any]) -> dict[str, Artifact]:
    """Index artifact entries by URL.

    :param entries: Artifact mappings from either a lock or a registry version.
    :returns: Artifacts keyed by URL.
    :raises ValueError: If an entry carries no URL to identify it by.
    """
    indexed: dict[str, Artifact] = {}
    for entry in entries:
        url = entry.get("url")
        if not url:
            raise ValueError(f"artifact entry has no url: {entry!r}")
        indexed[url] = Artifact(
            url=url,
            platforms=_platforms(entry.get("systems", []), entry.get("architectures", [])),
            type=str(entry.get("type", "unknown")),
            sha256=entry.get("sha256"),
        )
    return indexed


def _live_artifacts(registry: dict[str, Any], slug: str, version: str) -> dict[str, Artifact]:
    """Return the artifacts the registry currently offers for one pinned version.

    A withdrawn package or version yields no artifacts, so every pin reads as absent rather than as
    a clean comparison.

    :param registry: Registry index keyed by package slug.
    :param slug: Package slug without its version.
    :param version: Pinned version.
    :returns: Live artifacts keyed by URL.
    """
    versions = registry.get(slug, {}).get("versions", {})
    return _artifacts(versions.get(version, {}).get("files", []))


def compare_lock_to_registry(lock: dict[str, Any], registry: dict[str, Any]) -> list[PackageDrift]:
    """Compare every pinned package in one lock against the live registry.

    Artifacts are matched by URL: a URL only the lock has is ``absent``, one only the
    registry has is ``added``, and a shared URL whose digest moved is ``changed``.

    :param lock: Lock document keyed by ``slug@version``.
    :param registry: Registry index keyed by package slug.
    :returns: One entry per package that drifted, in lock order.
    :raises ValueError: If a lock key carries no ``@version`` suffix.
    """
    drifts: list[PackageDrift] = []
    for package, entry in lock.items():
        slug, separator, version = package.rpartition("@")
        if not separator:
            raise ValueError(f"lock key is not slug@version: {package!r}")
        locked = _artifacts(entry.get("artifacts", []))
        live = _live_artifacts(registry, slug, version)
        # Upstream ships platforms we never install (Windows, 32-bit); only the
        # platforms this lock pins can affect a build, so only those are compared.
        pinned = (
            frozenset().union(*(a.platforms for a in locked.values())) if locked else frozenset()
        )
        live = {url: a for url, a in live.items() if a.platforms & pinned}

        absent = tuple(locked[url] for url in locked if url not in live)
        added = tuple(live[url] for url in live if url not in locked)
        changed = tuple(
            (locked[url], live[url])
            for url in locked
            if url in live and locked[url].sha256 != live[url].sha256
        )
        if absent or added or changed:
            drifts.append(
                PackageDrift(
                    package=package,
                    absent=absent,
                    added=added,
                    changed=changed,
                    locked=tuple(locked.values()),
                    live=tuple(live.values()),
                )
            )
    return drifts


def _artifact_lines(label: str, artifacts: Sequence[Artifact]) -> list[str]:
    """Render one labelled artifact list.

    :param label: Drift category shown against each artifact.
    :param artifacts: Artifacts to render.
    :returns: One line per artifact.
    """
    return [
        f"    {label}: [{artifact.platform}] {artifact.type} {artifact.url}"
        for artifact in artifacts
    ]


def format_drift_report(drifts: Sequence[PackageDrift]) -> str:
    """Render drift so a re-pin needs no second live fetch by hand.

    :param drifts: Drift entries to render.
    :returns: A human-readable report, or a single clean line when nothing drifted.
    """
    if not drifts:
        return "no drift: every pinned artifact is still offered with its pinned digest"

    lines: list[str] = []
    for drift in drifts:
        lines.append(f"{drift.package}")
        lines.extend(_artifact_lines("absent ", drift.absent))
        lines.extend(_artifact_lines("added  ", drift.added))
        for locked_artifact, live_artifact in drift.changed:
            lines.append(
                f"    changed: [{locked_artifact.platform}] {locked_artifact.url}\n"
                f"             locked sha256 {locked_artifact.sha256}"
                f" -> live {live_artifact.sha256}"
            )
        lines.extend(_artifact_lines("  locked", drift.locked))
        lines.extend(_artifact_lines("  live  ", drift.live))
    return "\n".join(lines)


def _load(source: str) -> dict[str, Any]:
    """Read a JSON document from a path or an HTTP(S) URL.

    :param source: Local path or URL.
    :returns: Parsed JSON document.
    """
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source) as response:  # noqa: S310 - scheme checked above
            return json.load(response)
    return json.loads(Path(source).read_text())


def main(argv: Sequence[str] | None = None) -> int:
    """Report lock drift for every pinned package.

    :param argv: Command-line arguments, defaulting to ``sys.argv``.
    :returns: ``1`` when a pin is absent or its digest moved, otherwise ``0``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        action="append",
        required=True,
        help="Path to a Studiorack lock file; repeatable.",
    )
    parser.add_argument(
        "--registry",
        default=REGISTRY_INDEX_URL,
        help="Registry index URL or local path.",
    )
    args = parser.parse_args(argv)

    registry = _load(args.registry)
    blocking = False
    for lock_source in args.lock:
        drifts = compare_lock_to_registry(_load(lock_source), registry)
        sys.stdout.write(f"== {lock_source}\n{format_drift_report(drifts)}\n")
        blocking = blocking or any(drift.absent or drift.changed for drift in drifts)
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
