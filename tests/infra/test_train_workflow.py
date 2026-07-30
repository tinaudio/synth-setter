"""Contract tests for the manually dispatched training workflow."""

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).parents[2]
_WORKFLOW_PATH = _REPO_ROOT / ".github/workflows/train.yml"


def _load_workflow() -> dict:
    """Parse the training workflow YAML.

    :returns: The workflow document as a mapping.
    """
    return yaml.safe_load(_WORKFLOW_PATH.read_text(encoding="utf-8"))


def _steps_by_name(workflow: dict) -> dict[str, dict]:
    """Index the train job's steps by their display name.

    :param workflow: Parsed workflow document.
    :returns: Mapping of step name to step definition.
    """
    return {step["name"]: step for step in workflow["jobs"]["train"]["steps"]}


def test_train_workflow_requires_experiment_and_accepts_compute_override() -> None:
    """Training dispatch selects science and infrastructure independently."""
    workflow = _load_workflow()

    inputs = workflow[True]["workflow_dispatch"]["inputs"]
    assert inputs["experiment"]["required"] is True
    assert inputs["compute"]["required"] is False
    assert inputs["compute"]["default"] == ""
    assert "launch_config" not in inputs
    assert "dataset_root_uri" not in inputs


def test_train_workflow_maps_expensive_experiment_to_training_compute() -> None:
    """The 440k experiment retains its larger-disk compute default."""
    workflow = _load_workflow()

    script = _steps_by_name(workflow)["Resolve compute option"]["run"]
    assert "surge/flow_simple_440k)" in script
    assert "COMPUTE_OPTION=runpod/training" in script
    assert "COMPUTE_OPTION=runpod/smoke" in script
    assert "unsupported shell characters" in script


def test_train_workflow_dispatches_hydra_launcher_with_generic_command() -> None:
    """The workflow passes compute and train command through Hydra overrides."""
    workflow = _load_workflow()

    dispatch = _steps_by_name(workflow)["Dispatch via SkyPilot"]["run"]
    assert '"skypilot_launch/compute=$COMPUTE_OPTION"' in dispatch
    assert "skypilot_launch.worker_image_tag=dev-snapshot" in dispatch
    assert '"exec synth-setter-train "' in dispatch
    assert '"experiment=$EXPERIMENT "' in dispatch
    assert "training.upload_checkpoints_during_training=true" in dispatch
    assert "hydra.run.dir=/home/build/synth-setter/train-run" in dispatch
    assert "src/synth_setter/configs/launch" not in dispatch
