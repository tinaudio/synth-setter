"""Contract tests for the manually dispatched training workflow."""

from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).parents[2]
_LAUNCH_CONFIG_DIR = _REPO_ROOT / "src/synth_setter/configs/launch"
_WORKFLOW_PATH = _REPO_ROOT / ".github/workflows/train.yml"


def test_train_workflow_experiment_input_reaches_launcher_extra_env() -> None:
    """The dispatch experiment input is forwarded to the SkyPilot worker."""
    workflow = yaml.load(_WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    experiment = workflow["on"]["workflow_dispatch"]["inputs"]["experiment"]
    steps_by_name = {step["name"]: step for step in workflow["jobs"]["train"]["steps"]}
    dispatch_step = steps_by_name["Dispatch via SkyPilot"]

    assert experiment["default"] == ""
    assert dispatch_step["env"]["EXPERIMENT"] == "${{ inputs.experiment }}"
    assert '--extra-env EXPERIMENT "$EXPERIMENT"' in dispatch_step["run"]


@pytest.mark.parametrize(
    ("name", "default_experiment"),
    [
        ("train-runpod-flow-simple-440k.yaml", "surge/flow_simple"),
        ("train-runpod-smoke.yaml", "surge/ffn_simple"),
        ("train-runpod.yaml", "surge/ffn_simple"),
    ],
)
def test_train_runpod_config_uses_experiment_input_with_existing_default(
    name: str,
    default_experiment: str,
) -> None:
    """Each worker command honors an override without changing its default.

    :param name: Shipped training launch-config filename.
    :param default_experiment: Experiment used when the workflow input is empty.
    """
    launch_config_path = _LAUNCH_CONFIG_DIR / name
    launch_config = yaml.safe_load(launch_config_path.read_text(encoding="utf-8"))

    expected_override = f'"experiment=${{EXPERIMENT:-{default_experiment}}}"'
    assert expected_override in launch_config["cmd"]
