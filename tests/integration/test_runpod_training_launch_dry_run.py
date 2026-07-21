"""End-to-end dry runs for shipped RunPod training launches."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import sky
import yaml
from sky.volumes import Volume

from synth_setter.pipeline.schemas.skypilot_launch import SkypilotLaunchConfig
from synth_setter.pipeline.skypilot_launch import (
    _inject_network_volume,
    _load_compute_template_with_cmd,
    load_launch_config,
)

_REPO_ROOT = Path(__file__).parents[2]
_LAUNCH_DIR = _REPO_ROOT / "src/synth_setter/configs/launch"


def _compose_task_doc(launch_config: SkypilotLaunchConfig) -> dict[str, object]:
    """Load the launch config's template and inject cmd + network volume, as dispatch does.

    :param launch_config: Loaded ``SkypilotLaunchConfig`` with template and cmd set.
    :return: Task YAML dict ready for ``sky.Task.from_yaml_config``.
    """
    assert launch_config.compute_template is not None
    assert launch_config.cmd is not None
    template_path = _REPO_ROOT / launch_config.compute_template
    task_doc = _load_compute_template_with_cmd(template_path, launch_config.cmd)
    return _inject_network_volume(task_doc, launch_config.network_volume, source=template_path)


@pytest.mark.parametrize(
    "launch_config_name",
    [
        "train-runpod-smoke.yaml",
        "train-runpod-flow-simple-440k.yaml",
        "train-runpod-flow-simple-440k-volume.yaml",
        "train-runpod-flow-simple-440k-volume-jp-b200.yaml",
    ],
    ids=[
        "smoke",
        "flow-simple-440k",
        "flow-simple-440k-volume",
        "flow-simple-440k-volume-jp-b200",
    ],
)
def test_runpod_training_launch_dry_run_composes_worker_task_and_hydra_config(
    launch_config_name: str,
) -> None:
    """Prepare the real SkyPilot task and compose its worker command without submission.

    :param launch_config_name: Shipped RunPod training launch config to exercise.
    """
    launch_config = load_launch_config(_LAUNCH_DIR / launch_config_name)
    assert launch_config.compute_template is not None
    assert launch_config.cmd is not None

    task = sky.Task.from_yaml_config(_compose_task_doc(launch_config))
    task.validate()
    assert isinstance(task.run, str)
    _, entrypoint, train_args = task.run.partition("exec synth-setter-train")
    assert entrypoint

    train_entrypoint = Path(sys.executable).with_name("synth-setter-train")
    result = subprocess.run(  # noqa: S603 - real packaged CLI with config-owned arguments
        [
            "/bin/bash",
            "-c",
            f"exec {train_entrypoint} {train_args} --cfg job --resolve",
        ],
        cwd=_REPO_ROOT,
        env={
            **os.environ,
            "DATASET_ROOT_URI": "",
            "EXPERIMENT": "",
            "HYDRA_FULL_ERROR": "1",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert task.to_yaml_config()["run"] == task.run
    assert "synth_setter.data.lance_datamodule.LanceVSTDataModule" in result.stdout


def test_runpod_network_volume_training_hydrates_local_disk_from_mount() -> None:
    """The network-volume launch copies its mounted dataset to pod-local storage."""
    launch_config = load_launch_config(_LAUNCH_DIR / "train-runpod-flow-simple-440k-volume.yaml")
    assert launch_config.compute_template is not None
    assert launch_config.cmd is not None

    task = sky.Task.from_yaml_config(_compose_task_doc(launch_config))

    assert task.to_yaml_config()["volumes"] == {"/workspace/network-volume": "ss-datasets-us-ca-2"}
    assert isinstance(task.run, str)
    assert "download_dataset_root_uri=file:///workspace/network-volume/" in task.run
    assert "test -f /workspace/network-volume/" in task.run


def test_runpod_network_volume_staging_task_uses_versioned_dataset_path() -> None:
    """The staging launch seeds the mounted volume through the checked script."""
    launch_config = load_launch_config(_LAUNCH_DIR / "stage-runpod-surge-simple-440k-volume.yaml")
    assert launch_config.compute_template is not None
    assert launch_config.cmd is not None

    task = sky.Task.from_yaml_config(_compose_task_doc(launch_config))

    assert task.to_yaml_config()["volumes"] == {"/workspace/network-volume": "ss-datasets-us-ca-2"}
    assert isinstance(task.run, str)
    assert "scripts/stage_runpod_network_volume.sh" in task.run
    assert "surge-simple-lance-440k-20k-20k-20260706T005448315Z" in task.run


@pytest.mark.parametrize(
    ("volume_file", "zone"),
    [
        ("ss-datasets-us-ca-2.yaml", "US-CA-2"),
        ("ss-datasets-ap-jp-1.yaml", "AP-JP-1"),
    ],
    ids=["us-ca-2", "ap-jp-1"],
)
def test_runpod_network_volume_definition_is_valid(volume_file: str, zone: str) -> None:
    """Each committed volume definition targets RunPod's network-volume tier in its zone.

    :param volume_file: Volume definition filename under ``configs/volumes/``.
    :param zone: RunPod data center the definition must pin.
    """
    volume_path = _REPO_ROOT / "src/synth_setter/configs/volumes" / volume_file
    config = yaml.safe_load(volume_path.read_text())
    volume = Volume.from_yaml_config(config)

    volume.validate()
    assert volume.type == "runpod-network-volume"
    assert volume.zone == zone
    assert volume.size == "750"
