"""Contract tests for the manually dispatched evaluation workflow."""

from pathlib import Path
from typing import Any, cast

from workflow_fixtures import load_workflow

_WORKFLOW = "eval.yml"


def _dispatch_step(project_root: Path) -> dict[str, object]:
    """Load the evaluation workflow's dispatch step.

    :param project_root: Repository root supplied by infra fixtures.
    :returns: Dispatch step mapping.
    """
    workflow = load_workflow(project_root, _WORKFLOW)
    return next(
        step
        for step in workflow["jobs"]["evaluate"]["steps"]
        if step.get("name") == "Dispatch via SkyPilot"
    )


def test_eval_workflow_exposes_science_and_compute_inputs(project_root: Path) -> None:
    """Evaluation no longer requires a task-specific launch YAML.

    :param project_root: Repository root supplied by infra fixtures.
    """
    workflow = load_workflow(project_root, _WORKFLOW)
    workflow_doc = cast(dict[object, Any], workflow)
    inputs = workflow_doc[True]["workflow_dispatch"]["inputs"]

    assert "launch_config" not in inputs
    assert inputs["experiment"]["default"] == "surge/ffn_simple"
    assert inputs["compute"]["default"] == "runpod/smoke"
    assert inputs["checkpoint_ref"]["required"] is True


def test_eval_workflow_validates_inputs_before_command_construction(project_root: Path) -> None:
    """Workflow inputs cannot inject worker shell syntax.

    :param project_root: Repository root supplied by infra fixtures.
    """
    workflow = load_workflow(project_root, _WORKFLOW)
    validate = next(
        step
        for step in workflow["jobs"]["evaluate"]["steps"]
        if step.get("name") == "Validate launcher inputs"
    )

    assert "unsupported shell characters" in str(validate["run"])
    assert "^r2://" in str(validate["run"])


def test_eval_workflow_dispatches_hydra_launcher_with_generic_command(
    project_root: Path,
) -> None:
    """Evaluation command and infrastructure are Hydra launcher overrides.

    :param project_root: Repository root supplied by infra fixtures.
    """
    run = str(_dispatch_step(project_root)["run"])

    assert '"skypilot_launch/compute=$COMPUTE_OPTION"' in run
    assert "src/synth_setter/scripts/run-linux-vst-headless.sh" in run
    assert "synth-setter-eval experiment=$EXPERIMENT" in run
    assert "ckpt_path=" in run
    assert "hydra.run.dir=/home/build/synth-setter/eval-run" in run
    assert "src/synth_setter/configs/launch" not in run
