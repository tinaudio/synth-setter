"""Contracts for the GitHub Actions OCI removal."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import cast

import pytest
from workflow_fixtures import load_workflow

DELETED_WORKFLOWS = (
    "oci-image-bake.yaml",
    "push-to-ocir.yml",
    "test-oci-image-bake.yml",
)
OCI_CLOUD_PATTERN = re.compile(r"\b(?:oci|ocir)\b|oracle cloud", re.IGNORECASE)
ALLOWED_OCI_REFERENCES = (
    ("docker-build-validation.yml", "[worker.oci]"),
    ("docker-build-validation.yml", "[worker.oci]"),
)


@pytest.mark.infra
@pytest.mark.parametrize("workflow_filename", DELETED_WORKFLOWS)
def test_oci_only_workflow_is_absent(project_root: Path, workflow_filename: str) -> None:
    """OCI-only workflows are removed from the Actions surface.

    :param project_root: Repository root containing the workflow directory.
    :param workflow_filename: OCI-only workflow expected to be absent.
    """
    workflow_path = project_root / ".github" / "workflows" / workflow_filename
    assert not workflow_path.exists()


@pytest.mark.infra
def test_workflows_retain_only_buildkit_oci_references(project_root: Path) -> None:
    """Only BuildKit's Open Container Initiative worker references remain.

    :param project_root: Repository root containing the workflow directory.
    """
    references: list[tuple[str, str]] = []
    workflows_dir = project_root / ".github" / "workflows"
    for workflow_path in sorted(workflows_dir.glob("*.y*ml")):
        for line in workflow_path.read_text().splitlines():
            if OCI_CLOUD_PATTERN.search(line):
                references.append((workflow_path.name, line.strip()))

    assert tuple(references) == ALLOWED_OCI_REFERENCES


@pytest.mark.infra
def test_check_auth_matrix_targets_cloud_providers_only(project_root: Path) -> None:
    """Credential checks cover every supported authenticated provider.

    :param project_root: Repository root containing the workflow directory.
    """
    workflow = load_workflow(project_root, "check-auth.yml")
    matrix = workflow["jobs"]["check-auth"]["strategy"]["matrix"]
    assert matrix["provider"] == ["runpod", "vast"]


def _provider_validation_run(
    project_root: Path, provider: str
) -> subprocess.CompletedProcess[str]:
    """Execute the reusable workflow's provider-validation shell block.

    :param project_root: Repository root containing the workflow directory.
    :param provider: Provider input exposed to the shell block.
    :returns: Captured result from the workflow's provider-validation block.
    """
    workflow = load_workflow(project_root, "generate-dataset-shards.yaml")
    steps = workflow["jobs"]["generate"]["steps"]
    validation = next(step for step in steps if step.get("name") == "Validate provider")
    return subprocess.run(  # noqa: S603 — static Bash script from the checked-in workflow
        ["bash", "-c", validation["run"]],  # noqa: S607 — bash resolved from controlled PATH
        capture_output=True,
        env={"PATH": os.environ["PATH"], "PROVIDER": provider},
        text=True,
    )


@pytest.mark.infra
@pytest.mark.parametrize("provider", ["runpod", "skypilot-local"])
def test_reusable_generation_supported_provider_succeeds(
    project_root: Path, provider: str
) -> None:
    """Reusable generation accepts each supported provider.

    :param project_root: Repository root containing the workflow directory.
    :param provider: Supported provider input.
    """
    result = _provider_validation_run(project_root, provider)
    assert result.returncode == 0, result.stderr


@pytest.mark.infra
@pytest.mark.parametrize("provider", ["aws", "oci"])
def test_reusable_generation_unsupported_provider_fails(project_root: Path, provider: str) -> None:
    """Reusable generation rejects removed and unknown providers.

    :param project_root: Repository root containing the workflow directory.
    :param provider: Unsupported provider input.
    """
    result = _provider_validation_run(project_root, provider)
    assert result.returncode != 0
    assert result.stdout == ""
    assert f"unsupported provider: {provider}" in result.stderr


@pytest.mark.infra
def test_dataset_dispatch_choices_target_runpod_and_local_only(project_root: Path) -> None:
    """Manual generation exposes no removed cloud provider.

    :param project_root: Repository root containing the workflow directory.
    """
    workflow = cast(
        dict[object, object], load_workflow(project_root, "test-dataset-generation.yml")
    )
    triggers = workflow.get("on") or workflow[True]
    assert isinstance(triggers, dict)
    workflow_dispatch = triggers["workflow_dispatch"]
    assert isinstance(workflow_dispatch, dict)
    inputs = workflow_dispatch["inputs"]
    assert isinstance(inputs, dict)
    provider = inputs["provider"]
    assert isinstance(provider, dict)
    assert provider["options"] == ["all", "skypilot-local", "runpod"]
