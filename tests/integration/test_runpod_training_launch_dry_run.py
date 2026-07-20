"""End-to-end dry runs for shipped RunPod training launches."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import sky

from synth_setter.pipeline.skypilot_launch import (
    _load_compute_template_with_cmd,
    load_launch_config,
)

_REPO_ROOT = Path(__file__).parents[2]
_LAUNCH_DIR = _REPO_ROOT / "src/synth_setter/configs/launch"


@pytest.mark.parametrize(
    "launch_config_name",
    ["train-runpod-smoke.yaml", "train-runpod-flow-simple-440k.yaml"],
    ids=["smoke", "flow-simple-440k"],
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

    task_doc = _load_compute_template_with_cmd(
        _REPO_ROOT / launch_config.compute_template,
        launch_config.cmd,
    )
    task = sky.Task.from_yaml_config(task_doc)
    task.validate()
    assert isinstance(task.run, str)
    _, entrypoint, train_args = task.run.partition("exec synth-setter-train")
    assert entrypoint

    result = subprocess.run(  # noqa: S603 - real packaged CLI with config-owned arguments
        [
            "/bin/bash",
            "-c",
            f"exec synth-setter-train {train_args} --cfg job --resolve",
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
