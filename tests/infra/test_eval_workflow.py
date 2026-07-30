"""Contract tests for the manually dispatched evaluation workflow."""

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
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
    """Evaluation exposes generic launcher inputs instead of a launch recipe.

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
    assert inputs["dataset_root_uri"]["required"] is True
    assert "default" not in inputs["dataset_root_uri"]


@pytest.mark.parametrize(
    ("input_name", "input_value"),
    [
        ("EXPERIMENT", "surge/ffn_simple;echo-owned"),
        ("COMPUTE_OPTION", "runpod/smoke;echo-owned"),
        ("CHECKPOINT_REF", "model:v0;echo-owned"),
        ("DATASET_ROOT_URI", "r2://bucket/data;echo-owned"),
        ("DATASET_ROOT_URI", "r2:///dataset"),
    ],
)
def test_eval_workflow_rejects_shell_syntax_in_inputs(
    project_root: Path, input_name: str, input_value: str
) -> None:
    """Execute workflow validation against each command-injection boundary.

    :param project_root: Repository root supplied by infra fixtures.
    :param input_name: Environment variable receiving the malicious value.
    :param input_value: Input containing unsupported shell syntax.
    """
    validate = next(
        step
        for step in _workflow_steps(project_root)
        if step.get("name") == "Validate launcher inputs"
    )
    env = {
        **os.environ,
        "CHECKPOINT_REF": "tinaudio/synth-setter/model:v0",
        "COMPUTE_OPTION": "runpod/smoke",
        "DATASET_ROOT_URI": "r2://experiments/data/test/",
        "EXPERIMENT": "surge/ffn_simple",
    }
    env[input_name] = input_value

    result = subprocess.run(  # noqa: S603 - executes the checked-in workflow script
        ["bash", "-c", str(validate["run"])],  # noqa: S607
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode != 0


def test_eval_workflow_accepts_valid_launcher_inputs(project_root: Path) -> None:
    """Execute workflow validation with representative production inputs.

    :param project_root: Repository root supplied by infra fixtures.
    """
    validate = next(
        step
        for step in _workflow_steps(project_root)
        if step.get("name") == "Validate launcher inputs"
    )
    env = {
        **os.environ,
        "CHECKPOINT_REF": "tinaudio/synth-setter/model-ffn_simple:v12",
        "COMPUTE_OPTION": "runpod/smoke",
        "DATASET_ROOT_URI": "r2://experiments/data/test/",
        "EXPERIMENT": "surge/ffn_simple",
    }

    result = subprocess.run(  # noqa: S603 - executes the checked-in workflow script
        ["bash", "-c", str(validate["run"])],  # noqa: S607
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr


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
    assert '"synth-setter-eval "' in run
    assert '"experiment=$EXPERIMENT "' in run
    assert r"ckpt_path=\047\\\${wandb:$CHECKPOINT_REF}\047" in run
    assert "hydra.run.dir=/home/build/synth-setter/eval-run" in run
    assert "src/synth_setter/configs/launch" not in run


def test_eval_workflow_command_composes_with_literal_wandb_resolver(
    project_root: Path,
) -> None:
    """Execute the workflow's inner shell through launcher Hydra composition.

    :param project_root: Repository root supplied by infra fixtures.
    """
    run = str(_dispatch_step(project_root)["run"])
    inner = run.split("bash -c '\n", 1)[1].rsplit("\n  ' 2>&1", 1)[0]
    inner = inner.replace("cd /home/build/synth-setter", f"cd {shlex.quote(str(project_root))}")
    launcher = f"{shlex.quote(sys.executable)} -m synth_setter.pipeline.skypilot_launch"
    inner = inner.replace(
        "python -m synth_setter.pipeline.skypilot_launch \\",
        f"{launcher} --cfg job \\",
        1,
    )
    env = {
        **os.environ,
        "CHECKPOINT_REF": "tinaudio/synth-setter/model-ffn_simple:v12",
        "COMPUTE_OPTION": "runpod/smoke",
        "DATASET_ROOT_URI": "r2://experiments/data/test/",
        "EXPERIMENT": "surge/ffn_simple",
        "TAIL": "false",
    }

    result = subprocess.run(  # noqa: S603 - executes the checked-in workflow shell
        ["bash", "-c", inner],  # noqa: S607
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "ckpt_path='\\${wandb:tinaudio/synth-setter/model-ffn_simple:v12}'" in result.stdout
