"""Static assertions on the test-dataset-generation workflow's matrix shape.

The workflow exercises both `hdf5` and `wds` shard formats through the same
generate + validate plumbing. These tests parse `test-dataset-generation.yml`
and assert that the `output_format` axis is wired into the `generate-launcher`
strategy matrix, that cluster names are namespaced by `matrix.output_format` so
the matrix cells don't collide on per-cell R2 prefixes, that the per-cell
`run_id` formula reaches the launcher, and that the validate job reads its
URI from the `setup.outputs.spec_uris` map keyed by `<provider>-<output_format>`
(see #1154 for the matrix-output collapse the map routing dodges).

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


def test_validate_spec_uri_indexes_setup_map_by_matrix_coords(workflow: dict) -> None:
    """Validate cells must index ``setup.outputs.spec_uris`` by matrix coords.

    Routing through the non-matrix ``setup`` output dodges the
    ``needs.<matrix-job>.outputs.<x>`` scalar collapse documented at #1154;
    per-cell pairing then holds because both the URI map and the per-cell
    ``run_id`` formula encode the same ``<provider>-<output_format>`` coord.

    :param workflow: Parsed workflow YAML from the module-scoped fixture.
    """
    validate = workflow["jobs"]["validate"]
    spec_uri = validate["with"]["spec_uri"]
    assert "fromJSON(needs.setup.outputs.spec_uris)" in spec_uri, (
        "validate.with.spec_uri must read from the setup job's spec_uris map "
        f"(non-matrix output) per #1154 — got: {spec_uri!r}"
    )
    assert "matrix.provider" in spec_uri and "matrix.output_format" in spec_uri, (
        "validate.with.spec_uri must index the map by both matrix.provider and "
        f"matrix.output_format so per-cell pairing holds — got: {spec_uri!r}"
    )


def test_generate_launcher_pins_run_id_per_cell(workflow: dict) -> None:
    """Assert ``generate-launcher.with.run_id`` interpolates both matrix axes.

    The per-cell ``run_id`` pin (threaded into the launcher as ``+run_id=<value>``)
    must encode the same ``<provider>-<output_format>`` coordinate the validate
    cell uses to index the spec_uris map, so generate and validate cells pair
    on the same R2 prefix.

    :param workflow: Parsed workflow YAML from the module-scoped fixture.
    """
    generate_launcher = workflow["jobs"]["generate-launcher"]
    run_id = generate_launcher["with"]["run_id"]
    assert "matrix.provider" in run_id and "matrix.output_format" in run_id, (
        "generate-launcher.with.run_id must interpolate matrix.provider and "
        f"matrix.output_format so each cell writes under its own R2 prefix — got: {run_id!r}"
    )


def test_setup_emits_spec_uris_map_output(workflow: dict) -> None:
    """Assert `setup` emits a `spec_uris` output computed by `synth-setter-spec-uri`.

    :param workflow: Parsed workflow YAML from the module-scoped fixture.
    """
    setup_job = workflow["jobs"]["setup"]
    assert "spec_uris" in setup_job["outputs"], (
        "setup.outputs is missing the `spec_uris` key the validate job indexes."
    )
    map_step = _find_step(setup_job["steps"], step_id="spec_uris")
    run_script = map_step["run"]
    assert "synth-setter-spec-uri" in run_script, (
        "setup.spec_uris step must invoke synth-setter-spec-uri to derive each cell's URI."
    )
    assert "--from-experiment" in run_script and "--run-id-override" in run_script, (
        "setup.spec_uris step must pass both --from-experiment and --run-id-override "
        "to synth-setter-spec-uri so the URI derivation matches the launcher-side compose."
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

    1. ``providers == '[]'`` (unsupported event or unknown ``SCHEDULE_CRON``)
       → ``output_formats='[]'``.
    2. ``schedule`` (hourly or weekly cron) → both rows:
       ``output_formats='["hdf5","wds"]'``.
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
        "output_formats list — without it, unsupported-event runs would leave the output unset."
    )
    assert 'output_formats=\'["hdf5","wds"]\'' in run_script, (
        "setup.matrix is missing the schedule branch that emits both hdf5 and wds "
        "rows — without it, scheduled CI would never exercise the wds matrix cell."
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
