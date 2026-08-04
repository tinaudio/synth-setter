"""Contracts for the Simple 440k checkpoint evaluation matrix."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pytest
import sky

from synth_setter.pipeline.compute_task import build_task_doc
from synth_setter.pipeline.skypilot_launch import load_launch_config

_REPO_ROOT = Path(__file__).parents[2]
_LAUNCH_CONFIG = (
    _REPO_ROOT / "src/synth_setter/configs/launch/eval-runpod-flow-simple-440k-all-modes.yaml"
)
_WORKER_SCRIPT = _REPO_ROOT / "jobs/eval/run-flow-simple-440k-all-modes.sh"


class _ArmCase(NamedTuple):
    arm: str
    train_config_id: str
    artifact_name: str
    conditioning: str | None


_ARM_CASES = (
    _ArmCase("mel", "flow_simple_440k_100k", "model-flow_simple_440k_100k:v0", None),
    _ArmCase(
        "clap",
        "flow_simple_440k_clap_100k",
        "model-flow_simple_440k_clap_100k:v0",
        "clap",
    ),
    _ArmCase(
        "m2l",
        "flow_simple_440k_m2l_100k",
        "model-flow_simple_440k_m2l_100k:v0",
        "m2l",
    ),
    _ArmCase(
        "same_s",
        "flow_simple_440k_same_s_100k",
        "model-flow_simple_440k_same_s_100k:v0",
        "same_s",
    ),
)
_MODES = ("test", "validate", "val", "predict")


def test_eval_flow_simple_440k_launch_config_selects_one_mid_tier_job() -> None:
    """The checked-in config uses attached execution on the medium RunPod pool."""
    cfg = load_launch_config(_LAUNCH_CONFIG)

    assert cfg.compute is not None
    assert cfg.compute.name == "runpod-training"
    assert cfg.tier.value == "mid"
    assert cfg.num_workers == 1
    assert cfg.tail is True
    assert cfg.job_name == "eval-flow-simple-440k-all-modes"
    assert cfg.cmd is not None
    assert shlex.split(cfg.cmd) == [
        "cd",
        "/home/build/synth-setter",
        "&&",
        "bash",
        "scripts/sync_worker_checkout.sh",
        "&&",
        "exec",
        "src/synth_setter/scripts/run-linux-vst-headless.sh",
        "jobs/eval/run-flow-simple-440k-all-modes.sh",
        "--execute",
    ]
    task = sky.Task.from_yaml_config(build_task_doc(cfg.compute, cmd=cfg.cmd))
    task.validate()
    assert task.run == cfg.cmd


def test_eval_flow_simple_440k_worker_dry_run_covers_complete_matrix() -> None:
    """The unfiltered worker emits every checkpoint/mode cell."""
    result = subprocess.run(  # noqa: S603 - checked-in script, dry-run by default
        [str(_WORKER_SCRIPT)],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    lines = [line.split("\t", maxsplit=1) for line in result.stdout.splitlines()]
    assert [label for label, _command in lines] == [
        "DRY RUN mel/test",
        "DRY RUN mel/validate",
        "DRY RUN mel/val",
        "DRY RUN mel/predict",
        "DRY RUN clap/test",
        "DRY RUN clap/validate",
        "DRY RUN clap/val",
        "DRY RUN clap/predict",
        "DRY RUN m2l/test",
        "DRY RUN m2l/validate",
        "DRY RUN m2l/val",
        "DRY RUN m2l/predict",
        "DRY RUN same_s/test",
        "DRY RUN same_s/validate",
        "DRY RUN same_s/val",
        "DRY RUN same_s/predict",
    ]
    commands = {label: set(shlex.split(command)) for label, command in lines}
    predict_overrides = {
        "evaluation.compute_metrics=true",
        "evaluation.render_vst=true",
        "evaluation.rerender_target=true",
    }
    assert predict_overrides <= commands["DRY RUN mel/predict"]
    assert predict_overrides | {"conditioning=clap"} <= commands["DRY RUN clap/predict"]
    assert predict_overrides | {"conditioning=m2l"} <= commands["DRY RUN m2l/predict"]
    assert predict_overrides | {"conditioning=same_s"} <= commands["DRY RUN same_s/predict"]


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ("--arm=bogus", "unsupported arm: bogus"),
        ("--mode=bogus", "unsupported mode: bogus"),
        ("--arm=", "unsupported arm:"),
        ("--mode=", "unsupported mode:"),
        ("--arm=mel clap", "unsupported arm: mel clap"),
        ("--mode=test validate", "unsupported mode: test validate"),
    ],
)
def test_eval_flow_simple_440k_worker_unknown_filter_fails(argument: str, message: str) -> None:
    """An unknown arm or mode filter fails instead of producing an empty run.

    :param argument: Unsupported worker filter argument.
    :param message: Expected error prefix.
    """
    result = subprocess.run(  # noqa: S603 - checked-in script, dry-run by default
        [str(_WORKER_SCRIPT), argument],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert message in result.stderr


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("case", _ARM_CASES, ids=lambda case: case.arm)
def test_eval_flow_simple_440k_worker_command_composes_checkpoint_mode(
    case: _ArmCase,
    mode: str,
) -> None:
    """Each checkpoint/mode cell composes through the packaged eval entrypoint.

    :param case: Checkpoint arm selected from the worker matrix.
    :param mode: Evaluation mode selected from the worker matrix.
    """
    dry_run = subprocess.run(  # noqa: S603 - checked-in script with fixed test arguments
        [str(_WORKER_SCRIPT), f"--arm={case.arm}", f"--mode={mode}"],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert dry_run.returncode == 0, dry_run.stderr
    label, command_text = dry_run.stdout.rstrip().split("\t", maxsplit=1)
    assert label == f"DRY RUN {case.arm}/{mode}"
    worker_args = shlex.split(command_text)
    assert worker_args[0] == "synth-setter-eval"

    eval_entrypoint = Path(sys.executable).with_name("synth-setter-eval")
    composed = subprocess.run(  # noqa: S603 - real packaged CLI consumes checked-in args
        [str(eval_entrypoint), *worker_args[1:], "--cfg", "job"],
        cwd=_REPO_ROOT,
        env={**os.environ, "HYDRA_FULL_ERROR": "1"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert composed.returncode == 0, composed.stderr
    assert f"mode: {mode}" in composed.stdout
    assert f"consumed_train_config_id: {case.train_config_id}" in composed.stdout
    expected_ref = "ckpt_path: ${wandb:khaledtinubu-n-a/synth-setter/" + case.artifact_name + "}"
    assert expected_ref in composed.stdout
    if case.conditioning is None:
        assert "conditioning: mel" in composed.stdout
    else:
        assert f"column: {case.conditioning}" in composed.stdout
    if mode == "predict":
        assert "render_vst: true" in composed.stdout
        assert "compute_metrics: true" in composed.stdout
    else:
        assert "render_vst: false" in composed.stdout
        assert "compute_metrics: false" in composed.stdout
