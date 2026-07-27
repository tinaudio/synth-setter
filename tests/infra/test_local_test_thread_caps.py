"""Local parallel test lanes cap BLAS/OpenMP threads at one per xdist worker.

Without the caps every worker's torch builds an intra-op pool sized to all
cores, so N workers spawn N x cores compute threads and saturate the host
(observed 200-290% CPU per worker while `make test-fast` ran). One thread per
worker lets xdist's process-level parallelism own the cores; the env-var form
also propagates into spawned DataLoader children, which a runtime
``torch.set_num_threads`` call would not.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Local lanes that run pytest-xdist in parallel and must carry the caps.
_PARALLEL_TARGETS = ["test-fast", "test-full-cpu"]

_THREAD_CAPS = ["OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1"]


def _dry_run(target: str) -> str:
    """Return the expanded recipe ``make -n`` would execute for ``target``.

    :param target: Makefile target to expand.
    :returns: The dry-run command text with Makefile variables substituted.
    """
    make = shutil.which("make")
    assert make is not None, "make binary not found despite skipif guard"
    return subprocess.run(  # noqa: S603 — resolved make binary over an allowlisted target
        [make, "-n", target],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.mark.infra
@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
@pytest.mark.parametrize("target", _PARALLEL_TARGETS)
def test_parallel_test_target_caps_worker_threads(target: str) -> None:
    """Every ``-n auto`` pytest invocation in the target carries the thread caps.

    :param target: Local parallel Makefile test lane under test.
    """
    expanded = _dry_run(target)
    parallel_lines = [line for line in expanded.splitlines() if "-n auto" in line]
    assert parallel_lines, f"{target}: no parallel pytest invocation found in dry run"
    for line in parallel_lines:
        for cap in _THREAD_CAPS:
            assert cap in line, f"{target}: parallel invocation missing {cap!r}: {line}"
