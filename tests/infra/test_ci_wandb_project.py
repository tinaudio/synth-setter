"""CI workflows route authenticated W&B runs to the test project."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from workflow_fixtures import load_workflow

_CITEST_PROJECT = "synth-setter-citest"
_PRODUCTION_PROJECT = "synth-setter"
_PRODUCTION_WANDB_WORKFLOWS = frozenset({"eval.yml", "train.yml"})
_EXPECTED_CI_WANDB_PROJECTS = {
    "finalize-dataset.yaml": "${{ inputs.wandb_project }}",
    "generate-dataset-shards.yaml": "${{ inputs.wandb_project }}",
    "test-gpu.yml": _CITEST_PROJECT,
    "test-mps.yml": _CITEST_PROJECT,
}
_EXPECTED_REUSABLE_WANDB_PROJECTS = {
    ("generate-dataset-shards.yaml", "./.github/workflows/finalize-dataset.yaml"): (
        "${{ inputs.wandb_project }}"
    ),
    ("test-dataset-finalization.yml", "./.github/workflows/generate-dataset-shards.yaml"): (
        _CITEST_PROJECT
    ),
    ("test-dataset-generation.yml", "./.github/workflows/generate-dataset-shards.yaml"): (
        _CITEST_PROJECT
    ),
}


def _on(workflow: Mapping[str, object]) -> Mapping[str, object]:
    """Return workflow triggers parsed by PyYAML's YAML 1.1 loader.

    :param workflow: Parsed GitHub Actions workflow.
    :returns: Workflow trigger mapping.
    """
    # PyYAML 1.1 coerces the unquoted GitHub Actions `on` key to True.
    yaml_mapping = cast(Mapping[object, object], workflow)
    return cast(Mapping[str, object], yaml_mapping[True])


def _step(workflow: Mapping[str, object], job_name: str, step_name: str) -> Mapping[str, object]:
    """Return one named workflow step.

    :param workflow: Parsed GitHub Actions workflow.
    :param job_name: Owning job name.
    :param step_name: Exact workflow step name.
    :returns: Matching step definition.
    """
    jobs = cast(Mapping[str, object], workflow["jobs"])
    job = cast(Mapping[str, object], jobs[job_name])
    steps = cast(list[Mapping[str, object]], job["steps"])
    return next(step for step in steps if step.get("name") == step_name)


def _job_wandb_project_offenders(
    job_scope: str,
    job: Mapping[str, object],
    workflow_env: Mapping[str, object],
) -> list[str]:
    """Find authenticated scopes in one CI job without a W&B project.

    :param job_scope: Workflow filename and job key used in diagnostics.
    :param job: Parsed job definition.
    :param workflow_env: Environment inherited from the workflow.
    :returns: Job or step scopes that can fall back to the production project.
    """
    workflow_name = job_scope.partition(":")[0]
    expected_project = _EXPECTED_CI_WANDB_PROJECTS.get(workflow_name)
    job_env = cast(Mapping[str, object], job.get("env", {}))
    inherited_api_key = "WANDB_API_KEY" in workflow_env or "WANDB_API_KEY" in job_env
    inherited_project = job_env.get("WANDB_PROJECT", workflow_env.get("WANDB_PROJECT"))
    steps = cast(list[Mapping[str, object]], job.get("steps", []))
    if (
        inherited_api_key
        and not steps
        and (expected_project is None or inherited_project != expected_project)
    ):
        return [job_scope]

    offenders: list[str] = []
    for step in steps:
        step_env = cast(Mapping[str, object], step.get("env", {}))
        effective_project = step_env.get("WANDB_PROJECT", inherited_project)
        step_is_authenticated = inherited_api_key or "WANDB_API_KEY" in step_env
        if step_is_authenticated and (
            expected_project is None or effective_project != expected_project
        ):
            offenders.append(f"{job_scope}:{step.get('name', '<unnamed>')}")
    return offenders


def _reusable_wandb_project_offenders(job_scope: str, job: Mapping[str, object]) -> list[str]:
    """Validate project routing for authenticated reusable workflow calls.

    :param job_scope: Caller workflow filename and job key used in diagnostics.
    :param job: Parsed reusable-workflow job definition.
    :returns: Caller scope when inherited secrets could reach the wrong project.
    """
    if job.get("secrets") != "inherit":
        return []
    workflow_name = job_scope.partition(":")[0]
    reusable = job.get("uses")
    if not isinstance(reusable, str):
        return []
    expected_project = _EXPECTED_REUSABLE_WANDB_PROJECTS.get((workflow_name, reusable))
    if expected_project is None:
        return []
    inputs = cast(Mapping[str, object], job.get("with", {}))
    if inputs.get("wandb_project") == expected_project:
        return []
    return [job_scope]


def _ci_steps_missing_wandb_project(project_root: Path) -> list[str]:
    """Find authenticated CI scopes without an explicit W&B project.

    :param project_root: Repository root containing GitHub Actions workflows.
    :returns: Workflow, job, or step scopes that can fall back to production.
    """
    offenders: list[str] = []
    workflows_dir = project_root / ".github" / "workflows"
    for path in sorted(workflows_dir.glob("*.y*ml")):
        if path.name in _PRODUCTION_WANDB_WORKFLOWS:
            continue
        workflow = load_workflow(project_root, path.name)
        workflow_env = cast(Mapping[str, object], workflow.get("env", {}))
        jobs = cast(Mapping[str, object], workflow["jobs"])
        for job_name, job_value in jobs.items():
            job = cast(Mapping[str, object], job_value)
            job_scope = f"{path.name}:{job_name}"
            offenders.extend(_reusable_wandb_project_offenders(job_scope, job))
            offenders.extend(_job_wandb_project_offenders(job_scope, job, workflow_env))
    return offenders


@pytest.mark.infra
def test_dataset_ci_callers_select_citest_project(project_root: Path) -> None:
    """Dataset smoke workflows explicitly select the isolated W&B project.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    generation = load_workflow(project_root, "test-dataset-generation.yml")
    finalization = load_workflow(project_root, "test-dataset-finalization.yml")

    assert generation["jobs"]["generate-launcher"]["with"]["wandb_project"] == _CITEST_PROJECT
    assert finalization["jobs"]["smoke-pipeline"]["with"]["wandb_project"] == _CITEST_PROJECT


@pytest.mark.infra
def test_reusable_dataset_workflows_preserve_production_default(project_root: Path) -> None:
    """Manual reusable-workflow dispatches retain the production W&B default.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    for filename in ("generate-dataset-shards.yaml", "finalize-dataset.yaml"):
        workflow = load_workflow(project_root, filename)
        triggers = _on(workflow)
        workflow_call = cast(Mapping[str, object], triggers["workflow_call"])
        workflow_dispatch = cast(Mapping[str, object], triggers["workflow_dispatch"])
        call_inputs = cast(Mapping[str, object], workflow_call["inputs"])
        dispatch_inputs = cast(Mapping[str, object], workflow_dispatch["inputs"])
        call_project = cast(Mapping[str, object], call_inputs["wandb_project"])
        dispatch_project = cast(Mapping[str, object], dispatch_inputs["wandb_project"])
        assert call_project["default"] == _PRODUCTION_PROJECT
        assert dispatch_project["default"] == _PRODUCTION_PROJECT


@pytest.mark.infra
def test_generation_project_reaches_worker_and_finalization(project_root: Path) -> None:
    """Generation and chained finalization use the caller's same W&B project.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    workflow = load_workflow(project_root, "generate-dataset-shards.yaml")
    generate = workflow["jobs"]["generate"]
    run_step = _step(
        workflow, "generate", "Run synth-setter-generate-dataset (runpod row; in container)"
    )

    assert generate["env"]["WANDB_PROJECT"] == "${{ inputs.wandb_project }}"
    assert "-e WANDB_PROJECT" in cast(str, run_step["run"])
    assert workflow["jobs"]["finalize"]["with"]["wandb_project"] == "${{ inputs.wandb_project }}"


@pytest.mark.infra
def test_finalize_project_reaches_cli(project_root: Path) -> None:
    """The reusable finalizer exposes its selected W&B project to the CLI.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    workflow = load_workflow(project_root, "finalize-dataset.yaml")
    run_step = _step(workflow, "finalize", "Run synth-setter-finalize-dataset")
    step_env = cast(Mapping[str, object], run_step["env"])

    assert step_env["WANDB_PROJECT"] == "${{ inputs.wandb_project }}"


@pytest.mark.infra
def test_authenticated_ci_steps_select_wandb_project(project_root: Path) -> None:
    """Every authenticated CI step outside production workflows selects a project.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    assert _ci_steps_missing_wandb_project(project_root) == []


def test_job_level_wandb_key_without_project_is_reported() -> None:
    """A job-level API key cannot bypass the CI project guard."""
    job = {"env": {"WANDB_API_KEY": "secret"}, "steps": [{"name": "test"}]}

    offenders = _job_wandb_project_offenders("test.yml:run_tests", job, {})

    assert offenders == ["test.yml:run_tests:test"]


def test_job_level_key_with_citest_on_every_step_is_allowed() -> None:
    """Per-step project selections can safely scope inherited authentication."""
    job = {
        "env": {"WANDB_API_KEY": "secret"},
        "steps": [{"name": "test", "env": {"WANDB_PROJECT": _CITEST_PROJECT}}],
    }

    offenders = _job_wandb_project_offenders("test-gpu.yml:run_tests", job, {})

    assert offenders == []


def test_inherited_key_with_production_step_project_is_reported() -> None:
    """A step cannot override an inherited authenticated project to production."""
    job = {
        "env": {"WANDB_API_KEY": "secret", "WANDB_PROJECT": _CITEST_PROJECT},
        "steps": [{"name": "test", "env": {"WANDB_PROJECT": _PRODUCTION_PROJECT}}],
    }

    offenders = _job_wandb_project_offenders("test-gpu.yml:run_tests", job, {})

    assert offenders == ["test-gpu.yml:run_tests:test"]


def test_inherited_reusable_call_without_project_is_reported() -> None:
    """Inherited secrets require explicit project routing at reusable calls."""
    job = {
        "secrets": "inherit",
        "uses": "./.github/workflows/generate-dataset-shards.yaml",
        "with": {},
    }

    offenders = _reusable_wandb_project_offenders(
        "test-dataset-generation.yml:generate-launcher", job
    )

    assert offenders == ["test-dataset-generation.yml:generate-launcher"]


def test_empty_wandb_project_is_reported() -> None:
    """An empty project cannot satisfy the authenticated CI guard."""
    job = {"steps": [{"name": "test", "env": {"WANDB_API_KEY": "secret", "WANDB_PROJECT": ""}}]}

    offenders = _job_wandb_project_offenders("test-gpu.yml:run_tests", job, {})

    assert offenders == ["test-gpu.yml:run_tests:test"]
