"""Drift detection between the Studiorack locks and the live registry."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from synth_setter.pipeline.ci.studiorack_lock_drift import (
    compare_lock_to_registry,
    format_drift_report,
    main,
)

_LINUX_ARCHIVE = {
    "architectures": ["x64"],
    "sha256": "a" * 64,
    "systems": ["linux"],
    "type": "archive",
    "url": "https://example.invalid/surge-1.3.4-linux.tar.gz",
}
_LINUX_INSTALLER = {
    "architectures": ["x64"],
    "sha256": "e" * 64,
    "systems": ["linux"],
    "type": "installer",
    "url": "https://example.invalid/surge-1.3.4-linux.deb",
}
_MAC_INSTALLER = {
    "architectures": ["x64"],
    "sha256": "b" * 64,
    "systems": ["mac"],
    "type": "installer",
    "url": "https://example.invalid/surge-1.3.4-mac.pkg",
}


def _lock(*artifacts: dict) -> dict:
    """Build a one-package lock document.

    :param *artifacts: Lock-shaped artifact entries for the pinned package.
    :returns: Lock document keyed by the package reference.
    """
    return {"surge-synthesizer/surge@1.3.4": {"artifacts": list(artifacts)}}


def _registry(*artifacts: dict) -> dict:
    """Build a one-package registry index offering the given artifacts.

    The registry nests systems as objects, unlike the lock's plain strings.

    :param *artifacts: Lock-shaped artifact entries to expose as registry files.
    :returns: Registry index document keyed by package slug.
    """
    files = [
        {
            **artifact,
            "systems": [{"type": system} for system in artifact["systems"]],
        }
        for artifact in artifacts
    ]
    return {
        "surge-synthesizer/surge": {
            "slug": "surge-synthesizer/surge",
            "versions": {"1.3.4": {"files": files}},
        }
    }


def test_compare_matching_lock_and_registry_reports_no_drift() -> None:
    """A registry offering exactly the pinned artifacts is not drift."""
    drifts = compare_lock_to_registry(_lock(_LINUX_ARCHIVE), _registry(_LINUX_ARCHIVE))

    assert drifts == []


def test_compare_registry_dropping_a_pinned_artifact_reports_it_absent() -> None:
    """An artifact the registry no longer offers will fail the build."""
    drifts = compare_lock_to_registry(
        _lock(_LINUX_ARCHIVE, _MAC_INSTALLER), _registry(_LINUX_ARCHIVE)
    )

    assert [artifact.url for artifact in drifts[0].absent] == [_MAC_INSTALLER["url"]]
    assert drifts[0].added == ()


def test_compare_registry_adding_an_artifact_reports_it_added_not_absent() -> None:
    """An upstream addition is informational, the #3623 case after #3671."""
    drifts = compare_lock_to_registry(
        _lock(_LINUX_ARCHIVE), _registry(_LINUX_ARCHIVE, _LINUX_INSTALLER)
    )

    assert [artifact.url for artifact in drifts[0].added] == [_LINUX_INSTALLER["url"]]
    assert drifts[0].absent == ()


def test_compare_ignores_additions_on_platforms_the_lock_does_not_pin() -> None:
    """Upstream ships Windows builds we never install; that is not drift worth a report."""
    windows = {
        "architectures": ["x64"],
        "sha256": "d" * 64,
        "systems": ["win"],
        "type": "archive",
        "url": "https://example.invalid/surge-1.3.4-win.zip",
    }

    drifts = compare_lock_to_registry(_lock(_LINUX_ARCHIVE), _registry(_LINUX_ARCHIVE, windows))

    assert drifts == []


def test_compare_same_url_with_a_new_digest_reports_a_changed_artifact() -> None:
    """A re-pointed digest at a pinned URL is a supply-chain event, not routine drift."""
    repointed = {**_LINUX_ARCHIVE, "sha256": "c" * 64}

    drifts = compare_lock_to_registry(_lock(_LINUX_ARCHIVE), _registry(repointed))

    locked, live = drifts[0].changed[0]
    assert (locked.sha256, live.sha256) == ("a" * 64, "c" * 64)
    assert drifts[0].absent == ()
    assert drifts[0].added == ()


def test_compare_package_missing_from_registry_reports_every_artifact_absent() -> None:
    """A withdrawn package must not read as a clean comparison."""
    drifts = compare_lock_to_registry(_lock(_LINUX_ARCHIVE), {})

    assert [artifact.url for artifact in drifts[0].absent] == [_LINUX_ARCHIVE["url"]]


def test_compare_version_missing_from_registry_reports_every_artifact_absent() -> None:
    """A registry offering the package at other versions still cannot serve the pin."""
    registry = _registry(_LINUX_ARCHIVE)
    registry["surge-synthesizer/surge"]["versions"] = {"9.9.9": {"files": []}}

    drifts = compare_lock_to_registry(_lock(_LINUX_ARCHIVE), registry)

    assert [artifact.url for artifact in drifts[0].absent] == [_LINUX_ARCHIVE["url"]]


def test_report_names_the_package_platform_and_both_artifact_lists() -> None:
    """A re-pin must not require a second live fetch by hand."""
    drifts = compare_lock_to_registry(
        _lock(_LINUX_ARCHIVE, _MAC_INSTALLER), _registry(_LINUX_ARCHIVE)
    )

    report = format_drift_report(drifts)

    assert "surge-synthesizer/surge@1.3.4" in report
    assert "mac/x64" in report
    assert _MAC_INSTALLER["url"] in report
    assert _LINUX_ARCHIVE["url"] in report


def test_main_exits_nonzero_and_names_the_package_when_a_pin_is_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Drive the real entrypoint over on-disk documents.

    :param tmp_path: Directory holding the lock and registry documents.
    :param capsys: Captures the report the entrypoint writes.
    """
    lock_path = tmp_path / "studiorack.lock.json"
    lock_path.write_text(json.dumps(_lock(_LINUX_ARCHIVE, _MAC_INSTALLER)))
    registry_path = tmp_path / "index.json"
    registry_path.write_text(json.dumps(_registry(_LINUX_ARCHIVE)))

    exit_code = main(["--lock", str(lock_path), "--registry", str(registry_path)])

    assert exit_code == 1
    assert "surge-synthesizer/surge@1.3.4" in capsys.readouterr().out


def test_main_exits_zero_when_the_registry_only_added_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An addition alone must not red the scheduled job.

    :param tmp_path: Directory holding the lock and registry documents.
    :param capsys: Captures the report the entrypoint writes.
    """
    lock_path = tmp_path / "studiorack.lock.json"
    lock_path.write_text(json.dumps(_lock(_LINUX_ARCHIVE)))
    registry_path = tmp_path / "index.json"
    registry_path.write_text(json.dumps(_registry(_LINUX_ARCHIVE, _LINUX_INSTALLER)))

    exit_code = main(["--lock", str(lock_path), "--registry", str(registry_path)])

    assert exit_code == 0
    assert _LINUX_INSTALLER["url"] in capsys.readouterr().out


def test_checker_imports_no_third_party_packages() -> None:
    """Keep the scheduled drift job free of a dependency install.

    The workflow runs this module on the runner's bare Python, so a third-party import here would
    turn a seconds-long job into a full environment build.
    """
    import ast

    from synth_setter.pipeline.ci import studiorack_lock_drift

    source = Path(studiorack_lock_drift.__file__).read_text()
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])

    assert roots <= set(sys.stdlib_module_names)
