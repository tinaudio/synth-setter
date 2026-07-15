"""Worker Python repair refuses destructive paths outside the image venv."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def _source_worker_python_helper(
    project_root: Path, virtual_env: str, worker_venv: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Source the worker helper with a controlled virtual environment.

    :param project_root: Repository root fixture.
    :param virtual_env: Value exposed to the helper as ``VIRTUAL_ENV``.
    :param worker_venv: Explicit repair target, used to exercise the real helper safely.
    :returns: Completed shell process.
    """
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = virtual_env

    command = "source scripts/ensure_worker_python.sh"
    if worker_venv is not None:
        command += ' "$1"'
    command += '; printf "recreated=%s\\n" "$SYNTH_SETTER_WORKER_PYTHON_RECREATED"'

    return subprocess.run(  # noqa: S603 -- fixed shell and repository-owned helper
        ["/bin/bash", "-c", command, "worker-python-test", str(worker_venv or "")],
        cwd=project_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_ensure_worker_python_stale_temp_venv_recreates_with_canonical_python(
    project_root: Path, tmp_path: Path
) -> None:
    """A stale worker venv is replaced by a usable Python 3.12.13 environment.

    :param project_root: Repository root fixture.
    :param tmp_path: Safe repair target outside the image-owned worker venv.
    """
    worker_venv = tmp_path / "worker-venv"
    worker_python = worker_venv / "bin/python"
    worker_python.parent.mkdir(parents=True)
    worker_python.write_text("#!/bin/bash\nexit 1\n")
    worker_python.chmod(0o755)

    result = _source_worker_python_helper(project_root, str(worker_venv), worker_venv)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "recreated=1" in result.stdout
    version = subprocess.run(  # noqa: S603 -- controlled executable in tmp_path
        [worker_python, "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert version == "3.12.13"


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


@pytest.mark.parametrize("unsafe_venv", ["/", ".", "/var/lib/worker/.."])
def test_ensure_worker_python_explicit_dangerous_target_is_rejected(
    project_root: Path, unsafe_venv: str
) -> None:
    """An explicit destructive target is rejected before repair.

    :param project_root: Repository root fixture.
    :param unsafe_venv: Dangerous path supplied as the helper's target.
    """
    result = _source_worker_python_helper(project_root, unsafe_venv, Path(unsafe_venv))

    assert result.returncode != 0
    assert "absolute and normalized" in result.stderr


def test_sync_worker_checkout_unpinned_recreated_venv_reinstalls_runtime(
    project_root: Path, tmp_path: Path
) -> None:
    """The no-ref worker path restores dependencies removed with a stale venv.

    :param project_root: Repository root fixture.
    :param tmp_path: Isolated checkout and fake runtime tools.
    """
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "sync_worker_checkout.sh").write_text(
        (project_root / "scripts/sync_worker_checkout.sh").read_text()
    )
    (scripts_dir / "ensure_worker_python.sh").write_text(
        "export SYNTH_SETTER_WORKER_PYTHON_RECREATED=1\n"
    )
    trace = tmp_path / "uv.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text(f'#!/bin/bash\nprintf "%s\\n" "$*" > "{trace}"\n')
    uv.chmod(0o755)
    env = os.environ.copy()
    env.pop("WORKER_GIT_REF", None)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(  # noqa: S603 -- fixed shell and repository-owned script
        ["/bin/bash", "scripts/sync_worker_checkout.sh"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert trace.read_text().strip() == "pip install --group runtime -e ."
