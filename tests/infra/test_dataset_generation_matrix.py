"""Static assertions on the test-dataset-generation workflow's matrix shape.

The PR exercises both `hdf5` and `wds` shard formats through the same generate +
validate plumbing. These tests parse `test-dataset-generation.yml` and assert
that the `output_format` axis is wired into both the `generate-local` and
`generate-launcher` strategy matrices, and that cluster names + spec URIs are
namespaced by `matrix.output_format` so the matrix cells don't collide on the
launcher's R2 spec key.

Kept as a stand-alone YAML parse (no `act`) so the assertion runs on every CI
worker without needing the `act` binary on PATH.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOW_RELATIVE_PATH = Path(".github") / "workflows" / "test-dataset-generation.yml"
LAUNCHER_JOBS = ("generate-local", "generate-launcher")


@pytest.fixture(scope="module")
def workflow(project_root: Path) -> dict:
    """Parsed test-dataset-generation.yml as a plain dict."""
    return yaml.safe_load((project_root / WORKFLOW_RELATIVE_PATH).read_text())


@pytest.mark.parametrize("job_name", LAUNCHER_JOBS)
def test_generate_job_has_output_format_matrix_axis(workflow: dict, job_name: str) -> None:
    """Both generate jobs declare an `output_format` strategy.matrix axis."""
    job = workflow["jobs"][job_name]
    matrix = job["strategy"]["matrix"]
    assert "output_format" in matrix, (
        f"{job_name}.strategy.matrix is missing the `output_format` axis — "
        f"found keys: {sorted(matrix.keys())}"
    )


@pytest.mark.parametrize("job_name", LAUNCHER_JOBS)
def test_generate_job_cluster_name_includes_output_format(workflow: dict, job_name: str) -> None:
    """Cluster name interpolates `matrix.output_format` so wds and hdf5 cells diverge."""
    job = workflow["jobs"][job_name]
    cluster_template = _cluster_name_template(job, job_name)
    assert "matrix.output_format" in cluster_template, (
        f"{job_name} cluster_name template does not interpolate matrix.output_format — "
        f"got: {cluster_template!r}"
    )


def test_validate_spec_uri_includes_output_format(workflow: dict) -> None:
    """The validate job's spec_uri template namespaces by matrix.output_format."""
    validate = workflow["jobs"]["validate"]
    spec_uri = validate["with"]["spec_uri"]
    assert "matrix.output_format" in spec_uri, (
        f"validate.with.spec_uri does not interpolate matrix.output_format — got: {spec_uri!r}"
    )


def test_setup_emits_output_formats_with_both_rows(workflow: dict) -> None:
    """`setup` job's matrix step emits an output_formats list containing hdf5 + wds."""
    matrix_step = _find_step(workflow["jobs"]["setup"]["steps"], step_id="matrix")
    run_script = matrix_step["run"]
    assert "hdf5" in run_script and "wds" in run_script, (
        "setup.matrix step must emit both 'hdf5' and 'wds' in its output_formats list"
    )
    assert "output_formats" in workflow["jobs"]["setup"]["outputs"], (
        "setup.outputs is missing the `output_formats` key consumed by downstream jobs"
    )


def _cluster_name_template(job: dict, job_name: str) -> str:
    """Return the cluster_name interpolation string for a generate job."""
    if job_name == "generate-local":
        return job["env"]["CLUSTER_NAME"]
    return job["with"]["cluster_name"]


def _find_step(steps: list[dict], *, step_id: str) -> dict:
    """Return the first step in `steps` whose `id` matches `step_id`."""
    for step in steps:
        if step.get("id") == step_id:
            return step
    raise AssertionError(f"No step with id={step_id!r} found in {steps!r}")
