"""Compatibility-patch tests for Linux VST3 bundle detection in Studiorack core."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATCH_SCRIPT = PROJECT_ROOT / "scripts/studiorack/patch-core.mjs"
MANAGER_SOURCE_FIXTURE = (
    PROJECT_ROOT / "tests/fixtures/studiorack/ManagerLocal.patch-preconditions.txt"
)

pytestmark = pytest.mark.infra

_node = shutil.which("node")
if _node is None:
    pytest.skip("node not on PATH", allow_module_level=True)
NODE = os.path.realpath(_node)


def _helper_source(path: Path) -> None:
    """Write the upstream helper fragment patched by the postinstall script.

    :param path: Destination for the JavaScript fixture.
    """
    path.write_text(
        "else if (isTarFile) {\n"
        "    return await tar.extract({ file: filePath, cwd: dirPath });\n"
        "}\n"
        "const pkgs = dirRead(path.join(mountPoint, '**', '*.pkg'));\n"
        "if (dirIs(f)) {\n"
        "    // Check if this is a macOS application bundle or plugin bundle\n"
        "    if (fileExists(path.join(f, 'Contents', 'Info.plist'))) {\n"
        "        bundleDirs.add(f);\n"
        "    }\n"
        "}\n"
    )


def _manager_source(path: Path) -> None:
    """Copy the canonical upstream ManagerLocal patch-precondition fixture.

    :param path: Destination for the JavaScript fixture.
    """
    shutil.copyfile(MANAGER_SOURCE_FIXTURE, path)


def _run_patch(helper: Path, manager: Path) -> int:
    """Run the patch script against core fixtures and return its exit status.

    :param helper: JavaScript filesystem helper to patch.
    :param manager: JavaScript package manager to patch.
    :returns: Node process exit status.
    """
    return os.spawnv(  # noqa: S606 — absolute Node path and fixed argv
        os.P_WAIT, NODE, [NODE, str(PATCH_SCRIPT), str(helper), str(manager)]
    )


def test_patch_core_adds_linux_vst3_bundle_detection(tmp_path: Path) -> None:
    """The patch recognizes extension-only Linux VST3 bundles.

    :param tmp_path: Scratch root for the upstream helper fixture.
    """
    helper = tmp_path / "file.js"
    manager = tmp_path / "ManagerLocal.js"
    _helper_source(helper)
    _manager_source(manager)

    assert _run_patch(helper, manager) == 0

    patched = helper.read_text()
    assert "path.extname(f).toLowerCase() === '.vst3'" in patched
    assert "Contents', 'Info.plist'" in patched
    assert "readdirSync(mountPoint)" in patched
    assert "dirCreate(dirPath)" in patched
    manager_text = manager.read_text()
    assert "files.every(file => file.type === FileType.Archive)" in manager_text
    assert "if (files.some(file => file.type === FileType.Archive))" in manager_text
    assert "files = files.filter(file => file.type === FileType.Archive)" in manager_text


def test_patch_core_repeated_run_is_idempotent(tmp_path: Path) -> None:
    """Repeated npm postinstall runs leave the first patch unchanged.

    :param tmp_path: Scratch root for the upstream helper fixture.
    """
    helper = tmp_path / "file.js"
    manager = tmp_path / "ManagerLocal.js"
    _helper_source(helper)
    _manager_source(manager)

    assert _run_patch(helper, manager) == 0
    once = (helper.read_text(), manager.read_text())
    assert _run_patch(helper, manager) == 0

    assert (helper.read_text(), manager.read_text()) == once


def test_patch_core_unknown_partial_manager_fails_closed(tmp_path: Path) -> None:
    """A partial upstream change cannot pass without every patch marker.

    :param tmp_path: Scratch root for the incompatible manager fixture.
    """
    helper = tmp_path / "file.js"
    manager = tmp_path / "ManagerLocal.js"
    _helper_source(helper)
    _manager_source(manager)
    manager.write_text(
        manager.read_text().replace(
            "        if (!isAdmin() && !isTests()) {\n",
            "        if (files.some(file => file.type === FileType.Archive)) "
            "files = files.filter(file => file.type === FileType.Archive);\n"
            "        if (!isAdmin() && !isTests() && "
            "!files.every(file => file.type === FileType.Archive)) {\n",
        )
    )

    assert _run_patch(helper, manager) != 0


def test_patch_core_unknown_source_fails_closed(tmp_path: Path) -> None:
    """An upstream source change blocks installation instead of mispatching.

    :param tmp_path: Scratch root for the incompatible helper fixture.
    """
    helper = tmp_path / "file.js"
    manager = tmp_path / "ManagerLocal.js"
    helper.write_text("export function filesMove() {}\n")
    _manager_source(manager)

    assert _run_patch(helper, manager) != 0
