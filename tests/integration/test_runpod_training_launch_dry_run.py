"""End-to-end dry runs for shipped RunPod training launches."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import sky
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra

from synth_setter.pipeline.compute_task import build_task_doc
from synth_setter.pipeline.schemas.skypilot_launch import SkypilotLaunchConfig
from synth_setter.pipeline.skypilot_launch import (
    _override_image_id,
    _sky_cfg_from_hydra,
    load_launch_config,
)

_REPO_ROOT = Path(__file__).parents[2]
_LAUNCH_DIR = _REPO_ROOT / "src/synth_setter/configs/launch"


def _compose_generic_task(command: str, compute_option: str = "runpod/smoke") -> sky.Task:
    """Compose the Hydra launcher and build its real SkyPilot task.

    :param command: Generic worker shell command.
    :param compute_option: Compute option selected through Hydra.
    :return: Constructed SkyPilot task without submission.
    """
    GlobalHydra.instance().clear()
    try:
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            cfg = compose(
                config_name="skypilot_launch/default",
                overrides=[
                    f"skypilot_launch/compute={compute_option}",
                    f"skypilot_launch.cmd={command}",
                ],
            )
    finally:
        GlobalHydra.instance().clear()
    return _compose_task(_sky_cfg_from_hydra(cfg))


def _compose_task(launch_config: SkypilotLaunchConfig) -> sky.Task:
    """Build the real ``sky.Task`` from the launch config's compute option, as dispatch does.

    :param launch_config: Loaded ``SkypilotLaunchConfig`` with compute and cmd set.
    :return: Constructed ``sky.Task`` (no submission).
    """
    assert launch_config.compute is not None
    assert launch_config.cmd is not None
    task = sky.Task.from_yaml_config(build_task_doc(launch_config.compute, cmd=launch_config.cmd))
    _override_image_id(task, f"tinaudio/synth-setter:{launch_config.worker_image_tag}")
    return task


def _run_worker_config(
    task: sky.Task, executable_name: str, *, wrapper: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Feed a generic task's command into its real packaged worker CLI.

    :param task: Composed SkyPilot task containing the wrapped command.
    :param executable_name: Packaged worker executable passed to ``exec`` or ``wrapper``.
    :param wrapper: Repository-relative executable wrapping the worker CLI, if any.
    :return: Completed worker config-composition subprocess.
    """
    assert isinstance(task.run, str)
    command_prefix = (
        f"exec {wrapper} {executable_name} " if wrapper else f"exec {executable_name} "
    )
    _, marker, raw_args = task.run.partition(command_prefix)
    assert marker
    args = shlex.split(raw_args.removesuffix(")"))
    entrypoint = (
        _REPO_ROOT / wrapper if wrapper else Path(sys.executable).with_name(executable_name)
    )
    command = [entrypoint, executable_name, *args] if wrapper else [entrypoint, *args]
    return subprocess.run(  # noqa: S603 - real packaged worker CLI
        [*command, "--cfg", "job"],
        cwd=_REPO_ROOT,
        env={**os.environ, "DATASET_ROOT_URI": "", "HYDRA_FULL_ERROR": "1"},
        check=False,
        capture_output=True,
        text=True,
    )


def test_training_hclass_hydra_command_composes_through_worker_entrypoint() -> None:
    """The short high-tier launch command is consumable by the real training CLI."""
    task = _compose_generic_task(
        '"exec synth-setter-train experiment=torchsynth/flow_audio_same"',
        compute_option="runpod/training-hclass",
    )

    result = _run_worker_config(task, "synth-setter-train")

    assert result.returncode == 0, result.stderr
    assert "run_name: flow_audio_same" in result.stdout
    assert not task.volumes


@pytest.mark.skipif(sys.platform != "linux", reason="worker headless wrapper requires Xvfb")
def test_generic_hydra_eval_command_composes_through_headless_worker_entrypoint() -> None:
    """The workflow-shaped eval command preserves its resolver through the headless CLI."""
    wrapper = "src/synth_setter/scripts/run-linux-vst-headless.sh"
    task = _compose_generic_task(
        f'"exec {wrapper} synth-setter-eval experiment=surge/ffn_simple '
        "ckpt_path='\\${wandb:tinaudio/synth-setter/model-ffn_simple:v0}'"
        ' datamodule.download_dataset_root_uri=r2://experiments/data/test/"'
    )

    result = _run_worker_config(task, "synth-setter-eval", wrapper=wrapper)

    assert result.returncode == 0, result.stderr
    assert "ckpt_path: ${wandb:tinaudio/synth-setter/model-ffn_simple:v0}" in result.stdout
    assert "download_dataset_root_uri: r2://experiments/data/test/" in result.stdout


@pytest.mark.parametrize(
    ("launch_config_name", "high_memory_materialization", "memory_floor"),
    [
        ("train-runpod-smoke.yaml", False, None),
        ("train-runpod-flow-simple-440k.yaml", True, "128+"),
    ],
    ids=["smoke", "flow-simple-440k"],
)
def test_runpod_training_launch_dry_run_composes_worker_task_and_hydra_config(
    launch_config_name: str,
    high_memory_materialization: bool,
    memory_floor: str | None,
) -> None:
    """Prepare the real SkyPilot task and compose its worker command without submission.

    :param launch_config_name: Shipped RunPod training launch config to exercise.
    :param high_memory_materialization: Expected worker-side hydration setting.
    :param memory_floor: Expected SkyPilot host-memory request.
    """
    launch_config = load_launch_config(_LAUNCH_DIR / launch_config_name)
    assert launch_config.compute is not None
    assert launch_config.cmd is not None

    task = _compose_task(launch_config)
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
    assert {resource.memory for resource in task.resources} == {memory_floor}
    assert "synth_setter.data.lance_datamodule.LanceVSTDataModule" in result.stdout
    expected_setting = str(high_memory_materialization).lower()
    assert f"high_memory_materialization: {expected_setting}" in result.stdout
