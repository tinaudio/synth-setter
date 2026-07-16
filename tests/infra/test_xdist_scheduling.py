"""Behavioral coverage for xdist scheduling constraints."""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import NoReturn

import pytest


def _run_pytest(args: list[str]) -> NoReturn:
    """Run a nested pytest session in an isolated interpreter.

    :param args: Arguments passed to pytest.
    :raises SystemExit: Always, with pytest's session status.
    """
    raise SystemExit(pytest.main(args))


def test_shared_xdist_group_runs_on_one_worker(tmp_path: Path, project_root: Path) -> None:
    """Tests in one resource group execute serially on one xdist worker.

    :param tmp_path: Holds the generated probe and worker observations.
    :param project_root: Repository root containing the pytest configuration.
    """
    worker_log = tmp_path / "workers.txt"
    probe = tmp_path / "test_group_probe.py"
    tests = "\n\n".join(
        f"def test_group_member_{index}():\n"
        f"    with open({str(worker_log)!r}, 'a') as stream:\n"
        "        stream.write(os.environ['PYTEST_XDIST_WORKER'] + '\\n')"
        for index in range(8)
    )
    probe.write_text(
        "import os\n\n"
        "import pytest\n\n"
        "pytestmark = pytest.mark.xdist_group(name='shared-resource')\n\n"
        f"{tests}\n"
    )

    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_run_pytest,
        args=(
            [
                "-c",
                str(project_root / "pyproject.toml"),
                "-n",
                "2",
                str(probe),
                "-q",
            ],
        ),
    )
    process.start()
    process.join(timeout=60)
    if process.is_alive():
        process.kill()
        process.join()

    assert process.exitcode == 0
    assert len(set(worker_log.read_text().splitlines())) == 1
