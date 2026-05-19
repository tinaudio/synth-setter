"""Static assertions on the test-dataset-generation workflow's matrix shape.

The PR exercises both `hdf5` and `wds` shard formats through the same generate +
validate plumbing. These tests parse `test-dataset-generation.yml` and assert
that the `output_format` axis is wired into the `generate-launcher` strategy
matrix, that cluster names are namespaced by `matrix.output_format` so the
matrix cells don't collide on per-cell R2 prefixes, and that the validate
job consumes the launcher's canonical `spec_uri` output.

Kept as a stand-alone YAML parse (no `act`) so the assertion runs on every CI
worker without needing the `act` binary on PATH.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOW_RELATIVE_PATH = Path(".github") / "workflows" / "test-dataset-generation.yml"
LAUNCHER_JOBS = ("generate-launcher",)


@pytest.fixture(scope="module")
def workflow(project_root: Path) -> dict:
    """Load `test-dataset-generation.yml` once per module.

    :param project_root: Repo root provided by the `tests/infra/conftest.py` fixture.
    :returns: Parsed YAML as a plain dict.
    :rtype: dict
    """
    return yaml.safe_load((project_root / WORKFLOW_RELATIVE_PATH).read_text())


@pytest.mark.parametrize("job_name", LAUNCHER_JOBS)
def test_generate_job_has_output_format_matrix_axis(workflow: dict, job_name: str) -> None:
    """Assert both generate jobs declare an `output_format` strategy.matrix axis.

    :param workflow: Parsed workflow YAML from the module-scoped fixture.
    :param job_name: The generate job (parametrized; currently `generate-launcher`).
    """
    job = workflow["jobs"][job_name]
    matrix = job["strategy"]["matrix"]
    assert "output_format" in matrix, (
        f"{job_name}.strategy.matrix is missing the `output_format` axis — "
        f"found keys: {sorted(matrix.keys())}"
    )


@pytest.mark.parametrize("job_name", LAUNCHER_JOBS)
def test_generate_job_cluster_name_includes_output_format(workflow: dict, job_name: str) -> None:
    """Assert cluster name interpolates `matrix.output_format` so the cells diverge.

    :param workflow: Parsed workflow YAML from the module-scoped fixture.
    :param job_name: The generate job (parametrized; currently `generate-launcher`).
    """
    job = workflow["jobs"][job_name]
    cluster_template = _cluster_name_template(job, job_name)
    assert "matrix.output_format" in cluster_template, (
        f"{job_name} cluster_name template does not interpolate matrix.output_format — "
        f"got: {cluster_template!r}"
    )


def test_validate_spec_uri_consumes_generate_launcher_output(workflow: dict) -> None:
    """Assert the validate job's spec_uri reads from ``needs.generate-launcher.outputs.spec_uri``.

    :param workflow: Parsed workflow YAML from the module-scoped fixture.
    """
    validate = workflow["jobs"]["validate"]
    spec_uri = validate["with"]["spec_uri"]
    assert "needs.generate-launcher.outputs.spec_uri" in spec_uri, (
        f"validate.with.spec_uri does not consume the launcher's canonical spec_uri output — "
        f"got: {spec_uri!r}"
    )


def test_setup_emits_output_formats_with_both_rows(workflow: dict) -> None:
    """Assert `setup` emits an `output_formats` output containing hdf5 + wds.

    :param workflow: Parsed workflow YAML from the module-scoped fixture.
    """
    matrix_step = _find_step(workflow["jobs"]["setup"]["steps"], step_id="matrix")
    run_script = matrix_step["run"]
    assert "hdf5" in run_script and "wds" in run_script, (
        "setup.matrix step must emit both 'hdf5' and 'wds' in its output_formats list"
    )
    assert "output_formats" in workflow["jobs"]["setup"]["outputs"], (
        "setup.outputs is missing the `output_formats` key consumed by downstream jobs"
    )


def test_setup_matrix_step_branches_on_event_name(workflow: dict) -> None:
    """Assert `setup.matrix` sets `output_formats` in all three event-name branches.

    The bash logic in `setup.matrix` has three branches that each must assign
    `output_formats`:

    1. ``providers == '[]'`` (fork PR / unsupported event) → ``output_formats='[]'``.
    2. ``pull_request`` (same-repo) → both rows: ``output_formats='["hdf5","wds"]'``.
    3. ``workflow_dispatch`` → collapse to the single format the dispatched
       experiment resolves to, via Hydra compose of ``DISPATCH_DATASET_CONFIG``.

    A future edit that drops one branch (e.g. accidentally removing the
    workflow_dispatch fallback or the providers-empty short-circuit) would
    leave a code path where ``output_formats`` is never set and the downstream
    ``fromJSON(...)`` matrix expansion fails opaquely. This test pins the three
    branches at the source-script level so the regression fails fast in CI.

    :param workflow: Parsed workflow YAML from the module-scoped fixture.
    """
    matrix_step = _find_step(workflow["jobs"]["setup"]["steps"], step_id="matrix")
    run_script = matrix_step["run"]

    assert "output_formats='[]'" in run_script, (
        "setup.matrix is missing the providers-empty branch that emits an empty "
        "output_formats list — without it, fork-PR runs would leave the output unset."
    )
    assert 'output_formats=\'["hdf5","wds"]\'' in run_script, (
        "setup.matrix is missing the pull_request branch that emits both hdf5 and wds "
        "rows — without it, PR-time CI would never exercise the new wds matrix cell."
    )
    assert "DISPATCH_DATASET_CONFIG" in run_script and "compose" in run_script, (
        "setup.matrix is missing the workflow_dispatch fallback that composes the "
        "dispatched experiment with Hydra to resolve a single output_format — "
        "without it, dispatch runs would fall through to an unset output_formats."
    )
    assert 'output_formats="[\\"${DISPATCH_FORMAT}\\"]"' in run_script, (
        "setup.matrix's workflow_dispatch branch must assign output_formats from the "
        "Hydra-composed DISPATCH_FORMAT — without that assignment, dispatch runs "
        "would leave output_formats unset and break fromJSON downstream."
    )


def _cluster_name_template(job: dict, job_name: str) -> str:
    """Return the cluster_name interpolation string for a generate job.

    :param job: Parsed job dict.
    :param job_name: Job name. The launcher job reads `with.cluster_name`
        (reusable workflow input).
    :returns: Raw `${{ ... }}` template string, unresolved.
    :rtype: str
    """
    return job["with"]["cluster_name"]


def _find_step(steps: list[dict], *, step_id: str) -> dict:
    """Return the first step in `steps` whose `id` matches `step_id`.

    :param steps: The job's `steps` list.
    :param step_id: Value to match against each step's `id` field.
    :returns: The matched step dict.
    :rtype: dict
    :raises AssertionError: If no step in `steps` has the given `id`. Surfaces as a
        normal pytest failure rather than a soft None return.
    """
    for step in steps:
        if step.get("id") == step_id:
            return step
    raise AssertionError(f"No step with id={step_id!r} found in {steps!r}")
