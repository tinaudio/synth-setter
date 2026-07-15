"""Make targets resolve Python tools from the checkout-local virtualenv."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[2]


def _write_tool(path: Path, origin: str) -> None:
    """Write a command stub that records which environment supplied it.

    :param path: Executable path to create.
    :param origin: Value written to ``TOOL_MARKER`` when invoked.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "{origin}" > "$TOOL_MARKER"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.mark.parametrize(
    ("target", "tool"),
    [("format", "pre-commit"), ("test-fast", "pytest")],
)
def test_make_target_with_foreign_environment_uses_checkout_venv(
    tmp_path: Path, target: str, tool: str
) -> None:
    """An inherited environment cannot redirect developer Make targets.

    :param tmp_path: Pytest fixture providing a throwaway checkout.
    :param target: Make target under test.
    :param tool: Python environment executable used by the target.
    """
    checkout = tmp_path / "checkout $HOME with spaces"
    checkout.mkdir()
    shutil.copy(PROJECT_ROOT / "Makefile", checkout / "Makefile")
    marker = tmp_path / "tool-origin.txt"
    global_bin = tmp_path / "global-bin"
    _write_tool(global_bin / tool, "global")
    _write_tool(checkout / ".venv" / "bin" / tool, "worktree")
    make = shutil.which("make")
    assert make is not None

    env = {
        **os.environ,
        "PATH": f"{global_bin}:{os.environ['PATH']}",
        "TOOL_MARKER": str(marker),
        "VIRTUAL_ENV": "/foreign/checkout/.venv",
    }
    subprocess.run(  # noqa: S603 — resolved make binary and allowlisted target
        [make, target], cwd=checkout, env=env, check=True
    )

    assert marker.read_text(encoding="utf-8") == "worktree\n"
