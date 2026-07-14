"""Worker Python repair refuses destructive paths outside the image venv."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def _source_worker_python_helper(
    project_root: Path, virtual_env: str
) -> subprocess.CompletedProcess[str]:
    """Source the worker helper with a controlled virtual environment.

    :param project_root: Repository root fixture.
    :param virtual_env: Value exposed to the helper as ``VIRTUAL_ENV``.
    :returns: Completed shell process.
    """
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = virtual_env

    return subprocess.run(
        ["/bin/bash", "-c", "source scripts/ensure_worker_python.sh"],
        cwd=project_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_ensure_worker_python_unexpected_venv_preserves_its_contents(
    project_root: Path, tmp_path: Path
) -> None:
    """An unexpected venv directory is rejected without deleting its contents.

    :param project_root: Repository root fixture.
    :param tmp_path: Pytest-provided directory used as the unexpected venv.
    """
    unsafe_venv = tmp_path / "unexpected-venv"
    unsafe_venv.mkdir()
    sentinel = unsafe_venv / "keep-me"
    sentinel.write_text("present")

    result = _source_worker_python_helper(project_root, str(unsafe_venv))

    assert result.returncode != 0
    assert "/venv/main" in result.stderr
    assert sentinel.read_text() == "present"


@pytest.mark.parametrize("unsafe_venv", ["/", ".", "/venv/main/.."])
def test_ensure_worker_python_dangerous_venv_is_rejected(
    project_root: Path, unsafe_venv: str
) -> None:
    """Dangerous path spellings never reach worker-venv repair.

    :param project_root: Repository root fixture.
    :param unsafe_venv: Dangerous environment path supplied by the caller.
    """
    result = _source_worker_python_helper(project_root, unsafe_venv)

    assert result.returncode != 0
    assert "/venv/main" in result.stderr
