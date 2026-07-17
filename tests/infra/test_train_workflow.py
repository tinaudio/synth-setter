"""Contract tests for the manually dispatched training workflow."""

from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).parents[2]
_LAUNCH_CONFIG_DIR = _REPO_ROOT / "src/synth_setter/configs/launch"
_WORKFLOW_PATH = _REPO_ROOT / ".github/workflows/train.yml"


def test_train_workflow_optional_inputs_reach_launcher_extra_env() -> None:
    """Optional dispatch overrides are forwarded to the SkyPilot worker."""
    workflow = yaml.safe_load(_WORKFLOW_PATH.read_text(encoding="utf-8"))

    # PyYAML's YAML 1.1 resolver interprets the unquoted GitHub key `on` as True.
    inputs = workflow[True]["workflow_dispatch"]["inputs"]
    steps_by_name = {step["name"]: step for step in workflow["jobs"]["train"]["steps"]}
    dispatch_step = steps_by_name["Dispatch via SkyPilot"]

    assert inputs["dataset_root_uri"]["default"] == ""
    assert inputs["experiment"]["default"] == ""
    assert dispatch_step["env"]["DATASET_ROOT_URI"] == "${{ inputs.dataset_root_uri }}"
    assert dispatch_step["env"]["EXPERIMENT"] == "${{ inputs.experiment }}"
    assert '--extra-env DATASET_ROOT_URI "$DATASET_ROOT_URI"' in dispatch_step["run"]
    assert '--extra-env EXPERIMENT "$EXPERIMENT"' in dispatch_step["run"]


@pytest.mark.parametrize(
    ("name", "default_experiment", "default_dataset_root_uri"),
    [
        (
            "train-runpod-flow-simple-440k.yaml",
            "surge/flow_simple",
            "r2://experiments/data/surge-simple-lance-440k-20k-20k/"
            "surge-simple-lance-440k-20k-20k-20260706T005448315Z/",
        ),
        (
            "train-runpod-smoke.yaml",
            "surge/ffn_simple",
            "r2://experiments/data/surge-simple-lance-1k-2k-2k/"
            "surge-simple-lance-1k-2k-2k-20260716T163226347Z/",
        ),
        (
            "train-runpod.yaml",
            "surge/ffn_simple",
            "r2://experiments/data/surge-simple-lance-1k-2k-2k/"
            "surge-simple-lance-1k-2k-2k-20260716T163226347Z/",
        ),
    ],
)
def test_train_runpod_config_uses_optional_inputs_with_existing_defaults(
    name: str,
    default_experiment: str,
    default_dataset_root_uri: str,
) -> None:
    """Each worker command honors overrides without changing its defaults.

    :param name: Shipped training launch-config filename.
    :param default_experiment: Experiment used when its workflow input is empty.
    :param default_dataset_root_uri: Dataset used when its workflow input is empty.
    """
    launch_config_path = _LAUNCH_CONFIG_DIR / name
    launch_config = yaml.safe_load(launch_config_path.read_text(encoding="utf-8"))

    expected_dataset = (
        f'"datamodule.download_dataset_root_uri=${{DATASET_ROOT_URI:-{default_dataset_root_uri}}}"'
    )
    expected_experiment = f'"experiment=${{EXPERIMENT:-{default_experiment}}}"'
    assert expected_dataset in launch_config["cmd"]
    assert expected_experiment in launch_config["cmd"]
