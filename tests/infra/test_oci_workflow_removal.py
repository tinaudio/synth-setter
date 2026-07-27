"""Contracts for the GitHub Actions OCI removal."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pytest
from workflow_fixtures import load_workflow

DELETED_WORKFLOWS = (
    "oci-image-bake.yaml",
    "push-to-ocir.yml",
    "test-oci-image-bake.yml",
)
OCI_CLOUD_PATTERN = re.compile(r"\b(?:oci|ocir)\b|oracle cloud", re.IGNORECASE)
ALLOWED_OCI_REFERENCES = [
    ("docker-build-validation.yml", "[worker.oci]"),
    ("docker-build-validation.yml", "[worker.oci]"),
]


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
    workflows_dir = project_root / ".github" / "workflows"
    references = [
        (workflow_path.name, line.strip())
        for workflow_path in sorted(workflows_dir.glob("*.y*ml"))
        for line in workflow_path.read_text().splitlines()
        if OCI_CLOUD_PATTERN.search(line)
    ]
    assert references == ALLOWED_OCI_REFERENCES


@pytest.mark.infra
def test_check_auth_matrix_targets_runpod_and_local_only(project_root: Path) -> None:
    """Credential checks retain only the supported compute paths.

    :param project_root: Repository root containing the workflow directory.
    """
    workflow = load_workflow(project_root, "check-auth.yml")
    matrix = workflow["jobs"]["check-auth"]["strategy"]["matrix"]
    assert matrix["provider"] == ["runpod", "local"]


@pytest.mark.infra
def test_dataset_dispatch_choices_target_runpod_and_local_only(project_root: Path) -> None:
    """Manual generation exposes no removed cloud provider.

    :param project_root: Repository root containing the workflow directory.
    """
    workflow = cast(dict[object, Any], load_workflow(project_root, "test-dataset-generation.yml"))
    triggers_value = workflow.get("on")
    if triggers_value is None:
        triggers_value = workflow[True]
    triggers = cast(dict[str, Any], triggers_value)
    provider = triggers["workflow_dispatch"]["inputs"]["provider"]
    assert provider["options"] == ["all", "skypilot-local", "runpod"]
