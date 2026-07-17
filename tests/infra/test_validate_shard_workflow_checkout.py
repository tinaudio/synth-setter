"""Shard-validation workflows must execute the checked-out project code.

The `validate_shard` smoke steps run inside `dev-snapshot`, but the contract
under test is the PR checkout mounted into that container. These invariants pin
that both workflow entrypoints sync the mounted checkout into `/venv/main`
before importing `synth_setter.pipeline.ci.validate_shard`, so the validator
never silently falls back to the image-baked package.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from workflow_fixtures import load_workflow

VALIDATE_SHARD_STEPS = (
    ("generate-dataset-shards.yaml", "generate", "Validate all shards from R2"),
    ("validate-dataset-shards.yaml", "validate-shard", "Validate every shard via R2"),
)


def _step_run(project_root: Path, workflow_filename: str, job_name: str, step_name: str) -> str:
    """Return the `run:` block for a named workflow step.

    :param project_root: Repo root containing `.github/workflows/`.
    :param workflow_filename: Workflow filename under `.github/workflows/`.
    :param job_name: Job containing the target step.
    :param step_name: Human-readable `name:` of the target step.
    :returns: The step's `run:` block.
    """
    workflow = load_workflow(project_root, workflow_filename)
    jobs = cast(dict[str, object], workflow["jobs"])
    job = cast(dict[str, object], jobs[job_name])
    steps = cast(list[dict[str, object]], job["steps"])
    for step in steps:
        if step.get("name") == step_name:
            run = step.get("run")
            assert isinstance(run, str), (
                f"{workflow_filename}:{job_name}:{step_name} missing string `run:` block"
            )
            return run
    pytest.fail(f"{workflow_filename}:{job_name} missing step {step_name!r}")


@pytest.mark.infra
@pytest.mark.parametrize(
    ("workflow_filename", "job_name", "step_name"),
    VALIDATE_SHARD_STEPS,
)
def test_validate_shard_steps_sync_checkout_before_running_validator(
    project_root: Path, workflow_filename: str, job_name: str, step_name: str
) -> None:
    """Each workflow syncs the mounted checkout into `/venv/main` before validation.

    :param project_root: Checkout containing the workflow contract.
    :param workflow_filename: Workflow filename under test.
    :param job_name: Job containing the target step.
    :param step_name: Step whose container command validates shards.
    """
    run = _step_run(project_root, workflow_filename, job_name, step_name)

    assert '-v "${{ github.workspace }}:/home/build/synth-setter" \\' in run
    assert '-v "${{ github.workspace }}:/home/build/synth-setter:ro" \\' not in run
    assert "-e UV_PROJECT_ENVIRONMENT=/venv/main" in run
    assert "cd /home/build/synth-setter" in run
    assert "uv sync --frozen --no-default-groups --group data" in run
    assert 'uv run python3 -m synth_setter.pipeline.ci.validate_shard "$SPEC_URI"' in run
    assert "uv pip install --group data" not in run
