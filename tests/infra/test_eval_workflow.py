"""Contract tests for the manually dispatched evaluation workflow."""

from pathlib import Path
from typing import cast

import yaml

_WORKFLOW = "eval.yml"


def _load_workflow(project_root: Path) -> dict[object, object]:
    """Parse the evaluation workflow.

    :param project_root: Repository root supplied by infra fixtures.
    :returns: Workflow document mapping.
    """
    path = project_root / ".github" / "workflows" / _WORKFLOW
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _workflow_steps(project_root: Path) -> list[dict[str, object]]:
    """Load the evaluation workflow's ordered steps.

    :param project_root: Repository root supplied by infra fixtures.
    :returns: Evaluation step mappings.
    """
    workflow = _load_workflow(project_root)
    jobs = cast(dict[str, dict[str, object]], workflow["jobs"])
    return cast(list[dict[str, object]], jobs["evaluate"]["steps"])


def _dispatch_step(project_root: Path) -> dict[str, object]:
    """Load the evaluation workflow's dispatch step.

    :param project_root: Repository root supplied by infra fixtures.
    :returns: Dispatch step mapping.
    """
    return next(
        step
        for step in _workflow_steps(project_root)
        if step.get("name") == "Dispatch via SkyPilot"
    )


def test_eval_workflow_exposes_science_and_compute_inputs(project_root: Path) -> None:
    """Evaluation no longer requires a task-specific launch YAML.

    :param project_root: Repository root supplied by infra fixtures.
    """
    workflow = _load_workflow(project_root)
    trigger = cast(dict[str, object], workflow[True])
    dispatch = cast(dict[str, object], trigger["workflow_dispatch"])
    inputs = cast(dict[str, dict[str, object]], dispatch["inputs"])

    assert "launch_config" not in inputs
    assert inputs["experiment"]["default"] == "surge/ffn_simple"
    assert inputs["compute"]["default"] == "runpod/smoke"
    assert inputs["checkpoint_ref"]["required"] is True
    assert "default" not in inputs["checkpoint_ref"]


def test_eval_workflow_validates_inputs_before_command_construction(project_root: Path) -> None:
    """Workflow inputs cannot inject worker shell syntax.

    :param project_root: Repository root supplied by infra fixtures.
    """
    validate = next(
        step
        for step in _workflow_steps(project_root)
        if step.get("name") == "Validate launcher inputs"
    )

    assert "unsupported shell characters" in str(validate["run"])
    assert "immutable W&B artifact version" in str(validate["run"])
    assert "^r2://" in str(validate["run"])


def test_eval_workflow_dispatches_hydra_launcher_with_generic_command(
    project_root: Path,
) -> None:
    """Evaluation command and infrastructure are Hydra launcher overrides.

    :param project_root: Repository root supplied by infra fixtures.
    """
    run = str(_dispatch_step(project_root)["run"])

    assert '"skypilot_launch/compute=$COMPUTE_OPTION"' in run
    assert "skypilot_launch.worker_image_tag=dev-snapshot" in run
    assert "src/synth_setter/scripts/run-linux-vst-headless.sh" in run
    assert "synth-setter-eval experiment=$EXPERIMENT" in run
    assert "ckpt_path='\\\\\\${wandb:$CHECKPOINT_REF}'" in run
    assert "hydra.run.dir=/home/build/synth-setter/eval-run" in run
    assert "src/synth_setter/configs/launch" not in run
