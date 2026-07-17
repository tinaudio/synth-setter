"""Behavioral coverage for xdist scheduling constraints."""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import NoReturn

import pytest

_GROUP_PROBE_COUNT = 8
_NESTED_PYTEST_TIMEOUT_SECONDS = 60


@pytest.mark.parametrize(
    ("timed_out", "exitcode", "expected"),
    [
        (True, -9, "Nested pytest timed out after 60 seconds (exit code -9)"),
        (False, 1, "Nested pytest failed (exit code 1)"),
    ],
)
def test_nested_pytest_diagnostic_status_expected(
    timed_out: bool, exitcode: int, expected: str
) -> None:
    """Report whether a nested pytest process timed out or failed.

    :param timed_out: Whether the parent killed the nested process at its timeout.
    :param exitcode: Nested pytest process exit status.
    :param expected: Expected diagnostic for the observed process result.
    """
    assert _nested_pytest_failure_message(timed_out, exitcode) == expected


def _nested_pytest_failure_message(timed_out: bool, exitcode: int | None) -> str:
    """Describe an unsuccessful nested pytest session.

    :param timed_out: Whether the parent killed the nested process at its timeout.
    :param exitcode: Nested pytest process exit status.
    :returns: A diagnostic that distinguishes timeout from test failure.
    """
    status = f"timed out after {_NESTED_PYTEST_TIMEOUT_SECONDS} seconds" if timed_out else "failed"
    return f"Nested pytest {status} (exit code {exitcode})"


def _run_pytest(args: list[str]) -> NoReturn:
    """Run a nested pytest session in an isolated interpreter.

    :param args: Selects the nested session's config, workers, and probe.
    :raises SystemExit: Always, with pytest's session status.
    """
    raise SystemExit(pytest.main(args))


def _write_group_probe(probe: Path, worker_log: Path) -> None:
    """Write grouped tests that record their assigned xdist worker.

    :param probe: Destination for the generated pytest module.
    :param worker_log: Shared output path embedded in each generated test.
    """
    tests = "\n\n".join(
        f"def test_group_member_{index}() -> None:\n"
        f"    with open({str(worker_log)!r}, 'a') as stream:\n"
        "        stream.write(os.environ['PYTEST_XDIST_WORKER'] + '\\n')"
        for index in range(_GROUP_PROBE_COUNT)
    )
    probe.write_text(
        "import os\n\n"
        "import pytest\n\n"
        "pytestmark = pytest.mark.xdist_group(name='shared-resource')\n\n"
        f"{tests}\n"
    )


def test_shared_xdist_group_runs_on_one_worker(tmp_path: Path, project_root: Path) -> None:
    """Launch nested pytest and verify grouped probes share one xdist worker.

    :param tmp_path: Holds the generated probe and worker observations.
    :param project_root: Repository root containing the pytest configuration.
    """
    worker_log = tmp_path / "workers.txt"
    probe = tmp_path / "test_group_probe.py"
    _write_group_probe(probe, worker_log)

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
    process.join(timeout=_NESTED_PYTEST_TIMEOUT_SECONDS)
    timed_out = process.is_alive()
    if timed_out:
        process.kill()
        process.join()

    assert process.exitcode == 0, _nested_pytest_failure_message(timed_out, process.exitcode)
    worker_ids = worker_log.read_text().splitlines()
    assert len(worker_ids) == _GROUP_PROBE_COUNT
    assert len(set(worker_ids)) == 1
