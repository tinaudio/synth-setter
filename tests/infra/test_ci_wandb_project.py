"""CI workflows route authenticated W&B runs to the test project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from workflow_fixtures import load_workflow

_CITEST_PROJECT = "synth-setter-citest"
_PRODUCTION_PROJECT = "synth-setter"


def _on(workflow: dict[Any, Any]) -> dict[str, Any]:
    """Return workflow triggers parsed by PyYAML's YAML 1.1 loader.

    :param workflow: Parsed GitHub Actions workflow.
    :returns: Workflow trigger mapping.
    """
    return workflow[True]


def _step(workflow: dict[str, Any], job_name: str, step_name: str) -> dict[str, Any]:
    """Return one named workflow step.

    :param workflow: Parsed GitHub Actions workflow.
    :param job_name: Owning job name.
    :param step_name: Exact workflow step name.
    :returns: Matching step definition.
    """
    return next(
        step for step in workflow["jobs"][job_name]["steps"] if step.get("name") == step_name
    )


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
        assert (
            triggers["workflow_call"]["inputs"]["wandb_project"]["default"] == _PRODUCTION_PROJECT
        )
        assert (
            triggers["workflow_dispatch"]["inputs"]["wandb_project"]["default"]
            == _PRODUCTION_PROJECT
        )


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
    assert "-e WANDB_PROJECT" in run_step["run"]
    assert workflow["jobs"]["finalize"]["with"]["wandb_project"] == "${{ inputs.wandb_project }}"


@pytest.mark.infra
def test_finalize_project_reaches_cli(project_root: Path) -> None:
    """The reusable finalizer exposes its selected W&B project to the CLI.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    workflow = load_workflow(project_root, "finalize-dataset.yaml")
    run_step = _step(workflow, "finalize", "Run synth-setter-finalize-dataset")

    assert run_step["env"]["WANDB_PROJECT"] == "${{ inputs.wandb_project }}"


@pytest.mark.infra
@pytest.mark.parametrize(
    ("filename", "job_name", "step_name"),
    [
        ("test-gpu.yml", "run_tests", "Dispatch GPU tests via SkyPilot"),
        ("test-mps.yml", "run_tests", "Run MPS tests"),
    ],
)
def test_authenticated_compute_ci_selects_citest_project(
    project_root: Path, filename: str, job_name: str, step_name: str
) -> None:
    """Authenticated GPU and MPS test runs cannot fall back to production.

    :param project_root: Repo root supplied by the infra test fixtures.
    :param filename: Workflow file under test.
    :param job_name: Job that executes authenticated tests.
    :param step_name: Step receiving W&B credentials.
    """
    workflow = load_workflow(project_root, filename)

    assert _step(workflow, job_name, step_name)["env"]["WANDB_PROJECT"] == _CITEST_PROJECT
