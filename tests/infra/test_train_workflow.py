"""Contract tests for the manually dispatched training workflow."""

from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).parents[2]
_LAUNCH_CONFIG_DIR = _REPO_ROOT / "src/synth_setter/configs/launch"
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


def test_train_workflow_requires_experiment_and_forwards_it_to_launcher() -> None:
    """One-selector dispatch (#2196): `experiment` is the only required input."""
    workflow = _load_workflow()

    # PyYAML's YAML 1.1 resolver interprets the unquoted GitHub key `on` as True.
    inputs = workflow[True]["workflow_dispatch"]["inputs"]
    assert inputs["experiment"]["required"] is True
    assert "dataset_root_uri" not in inputs
    assert inputs["launch_config"]["required"] is False
    assert inputs["launch_config"]["default"] == ""

    assert workflow["jobs"]["train"]["env"]["EXPERIMENT"] == "${{ inputs.experiment }}"
    dispatch_step = _steps_by_name(workflow)["Dispatch via SkyPilot"]
    assert '--extra-env EXPERIMENT "$EXPERIMENT"' in dispatch_step["run"]
    assert "DATASET_ROOT_URI" not in dispatch_step["run"]


def test_train_workflow_maps_experiments_to_existing_launch_configs() -> None:
    """The hardcoded experiment → launch-config mapping only names shipped configs."""
    workflow = _load_workflow()

    resolve_step = _steps_by_name(workflow)["Resolve launch config"]
    script = resolve_step["run"]
    assert "surge/flow_simple_440k)" in script
    assert "surge/ffn_simple_smoke)" in script

    mapped = [
        token
        for line in script.splitlines()
        for token in line.split()
        if token.startswith("LAUNCH_CONFIG=src/")
    ]
    assert mapped, "mapping must assign repo-relative launch-config paths"
    for assignment in mapped:
        path = _REPO_ROOT / assignment.removeprefix("LAUNCH_CONFIG=")
        assert path.is_file(), f"mapped launch config does not exist: {path}"


@pytest.mark.parametrize(
    ("name", "default_experiment"),
    [
        ("train-runpod-flow-simple-440k.yaml", "surge/flow_simple_440k"),
        ("train-runpod-smoke.yaml", "surge/ffn_simple_smoke"),
        ("train-runpod.yaml", "surge/ffn_simple_smoke"),
    ],
)
def test_train_runpod_config_defaults_to_self_contained_experiment(
    name: str,
    default_experiment: str,
) -> None:
    """Each worker command honors the EXPERIMENT override and pins no dataset itself.

    :param name: Shipped training launch-config filename.
    :param default_experiment: Experiment used when the forwarded EXPERIMENT is empty.
    """
    launch_config_path = _LAUNCH_CONFIG_DIR / name
    launch_config = yaml.safe_load(launch_config_path.read_text(encoding="utf-8"))

    expected_experiment = f'"experiment=${{EXPERIMENT:-{default_experiment}}}"'
    assert expected_experiment in launch_config["cmd"]
    assert "DATASET_ROOT_URI" not in launch_config["cmd"]
