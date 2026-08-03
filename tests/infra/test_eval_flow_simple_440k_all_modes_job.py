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


class _EvalCase(NamedTuple):
    arm: str
    mode: str
    train_config_id: str
    artifact_name: str
    conditioning: str | None


# Cover every arm and every mode through the packaged CLI; the matrix test below
# separately pins their complete Cartesian product without 16 heavyweight subprocesses.
_EVAL_CASES = (
    _EvalCase("mel", "test", "flow_simple_440k_100k", "model-flow_simple_440k_100k:v0", None),
    _EvalCase("mel", "validate", "flow_simple_440k_100k", "model-flow_simple_440k_100k:v0", None),
    _EvalCase("mel", "val", "flow_simple_440k_100k", "model-flow_simple_440k_100k:v0", None),
    _EvalCase("mel", "predict", "flow_simple_440k_100k", "model-flow_simple_440k_100k:v0", None),
    _EvalCase(
        "clap",
        "test",
        "flow_simple_440k_clap_100k",
        "model-flow_simple_440k_clap_100k:v0",
        "clap",
    ),
    _EvalCase(
        "m2l",
        "test",
        "flow_simple_440k_m2l_100k",
        "model-flow_simple_440k_m2l_100k:v0",
        "m2l",
    ),
    _EvalCase(
        "same_s",
        "test",
        "flow_simple_440k_same_s_100k",
        "model-flow_simple_440k_same_s_100k:v0",
        "same_s",
    ),
)


def test_eval_flow_simple_440k_launch_config_selects_one_mid_tier_job() -> None:
    """The checked-in config runs one attached job on the medium RunPod pool."""
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
    """The unfiltered worker emits all sixteen checkpoint/mode cells."""
    result = subprocess.run(  # noqa: S603 - checked-in script, dry-run by default
        [str(_WORKER_SCRIPT)],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert [line.split("\t", maxsplit=1)[0] for line in result.stdout.splitlines()] == [
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


@pytest.mark.parametrize(("flag", "value"), [("--arm", "bogus"), ("--mode", "bogus")])
def test_eval_flow_simple_440k_worker_unknown_filter_fails(flag: str, value: str) -> None:
    """An unknown retry filter fails instead of producing a false-green empty run.

    :param flag: Worker filter option under test.
    :param value: Unsupported filter value.
    """
    result = subprocess.run(  # noqa: S603 - checked-in script, dry-run by default
        [str(_WORKER_SCRIPT), flag, value],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert f"unsupported {flag.removeprefix('--')}: {value}" in result.stderr


@pytest.mark.parametrize("flag", ["--arm", "--mode"])
def test_eval_flow_simple_440k_worker_missing_filter_value_fails(flag: str) -> None:
    """A filter without an operand exits with a usage error.

    :param flag: Worker filter option missing its required operand.
    """
    result = subprocess.run(  # noqa: S603 - checked-in script, dry-run by default
        [str(_WORKER_SCRIPT), flag],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert f"{flag} requires a value" in result.stderr


@pytest.mark.parametrize("case", _EVAL_CASES, ids=lambda case: f"{case.arm}-{case.mode}")
def test_eval_flow_simple_440k_worker_command_composes_checkpoint_mode(
    case: _EvalCase,
) -> None:
    """Each arm and mode composes through the packaged eval entrypoint.

    :param case: Checkpoint/mode case selected from the worker matrix.
    """
    dry_run = subprocess.run(  # noqa: S603 - checked-in script with fixed test arguments
        [str(_WORKER_SCRIPT), "--arm", case.arm, "--mode", case.mode],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert dry_run.returncode == 0, dry_run.stderr
    label, command_text = dry_run.stdout.rstrip().split("\t", maxsplit=1)
    assert label == f"DRY RUN {case.arm}/{case.mode}"
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
    assert f"mode: {case.mode}" in composed.stdout
    assert f"consumed_train_config_id: {case.train_config_id}" in composed.stdout
    expected_ref = "ckpt_path: ${wandb:khaledtinubu-n-a/synth-setter/" + case.artifact_name + "}"
    assert expected_ref in composed.stdout
    if case.conditioning is None:
        assert "conditioning: mel" in composed.stdout
    else:
        assert f"column: {case.conditioning}" in composed.stdout
    if case.mode == "predict":
        assert "render_vst: true" in composed.stdout
        assert "compute_metrics: true" in composed.stdout
    else:
        assert "render_vst: false" in composed.stdout
        assert "compute_metrics: false" in composed.stdout
