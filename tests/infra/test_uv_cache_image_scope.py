"""Every uv cache suffix in ``cpu-slow.yml`` resolves to a value the workflow sets.

surgepy builds from source, so a cached wheel links the builder image's glibc and the 22.04 and
24.04 lanes must not share a cache key (#3507). The expression that separates them has to name a
variable the workflow itself publishes: GitHub's ``env`` context holds only workflow-, job- and
step-set variables and *"does not contain variables inherited by the runner process"*, so an
image id read straight off the runner (``${{ env.ImageOS }}``) expands to the empty string and
silently restores the shared key it was added to split.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from workflow_fixtures import load_workflow

_SETUP_UV = "astral-sh/setup-uv"
_ENV_REFERENCE = re.compile(r"\$\{\{\s*env\.([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_GITHUB_ENV_EXPORT = re.compile(r"^\s*(?:echo\s+)?\"?([A-Za-z_][A-Za-z0-9_]*)=", re.MULTILINE)


def _cached_uv_steps(job: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    """Return each cache-enabled ``setup-uv`` step of `job` with its index.

    :param job: Parsed job mapping.
    :returns: ``(step index, step)`` pairs, in workflow order.
    """
    return [
        (index, step)
        for index, step in enumerate(job.get("steps", []))
        if _SETUP_UV in str(step.get("uses", "")) and step.get("with", {}).get("enable-cache")
    ]


def _names_published_before(job: dict[str, Any], index: int) -> set[str]:
    """Return variable names readable from the ``env`` context at step `index`.

    :param job: Parsed job mapping.
    :param index: Index of the step whose expressions are being resolved.
    :returns: Names set by workflow/job ``env:`` blocks or written to ``$GITHUB_ENV`` earlier.
    """
    names = set(job.get("env", {}))
    for step in job.get("steps", [])[:index]:
        names.update(step.get("env", {}))
        script = str(step.get("run", ""))
        if "GITHUB_ENV" in script:
            names.update(_GITHUB_ENV_EXPORT.findall(script))
    return names


@pytest.fixture
def cpu_slow_jobs(project_root: Path) -> dict[str, Any]:
    """Return the jobs of ``cpu-slow.yml``.

    :param project_root: Repo root supplied by the infra conftest.
    :returns: Mapping of job id to parsed job.
    """
    return load_workflow(project_root, "cpu-slow.yml")["jobs"]


def test_cached_uv_steps_carry_a_cache_suffix(cpu_slow_jobs: dict[str, Any]) -> None:
    """Each cache-enabled ``setup-uv`` step scopes its key with a suffix.

    :param cpu_slow_jobs: Parsed jobs of ``cpu-slow.yml``.
    """
    missing = [
        job_id
        for job_id, job in cpu_slow_jobs.items()
        for _, step in _cached_uv_steps(job)
        if not step["with"].get("cache-suffix")
    ]

    assert not missing, f"cache-enabled setup-uv steps without a cache-suffix: {missing}"


def test_cache_suffix_names_a_variable_the_workflow_publishes(
    cpu_slow_jobs: dict[str, Any],
) -> None:
    """No suffix reads a runner-process variable the ``env`` context cannot see.

    :param cpu_slow_jobs: Parsed jobs of ``cpu-slow.yml``.
    """
    unresolved = [
        (job_id, name)
        for job_id, job in cpu_slow_jobs.items()
        for index, step in _cached_uv_steps(job)
        for name in _ENV_REFERENCE.findall(str(step["with"].get("cache-suffix", "")))
        if name not in _names_published_before(job, index)
    ]

    assert not unresolved, f"cache-suffix reads variables the workflow never sets: {unresolved}"
