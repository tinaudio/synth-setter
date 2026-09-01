"""Entrypoint contracts for the flow-sketch CFG ablation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]
_LAUNCHER = _REPO_ROOT / "jobs/eval/launch_flow_sketch_cfg_ablation.sh"


def _dry_run(args: tuple[str, ...] = ()) -> subprocess.CompletedProcess[str]:
    """Run the launcher without its paid-compute opt-in.

    :param args: Launcher arguments appended after the script path.
    :returns: Captured dry-run process result.
    """
    return subprocess.run(  # noqa: S603 — repository-owned script
        [str(_LAUNCHER), *args],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_cfg_ablation_dry_run_emits_full_factorial_without_launching() -> None:
    """Protect the paid-compute boundary while rendering the complete plan."""
    result = _dry_run()

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("DRY RUN:") == 9
    assert result.stdout.count("mode=validate") == 9
    assert result.stdout.count("mode=predict") == 9
    assert "content_cfg=0 sketch_cfg=0" in result.stdout
    assert "content_cfg=0 sketch_cfg=1" in result.stdout
    assert "content_cfg=0 sketch_cfg=2" in result.stdout
    assert "content_cfg=1 sketch_cfg=0" in result.stdout
    assert "content_cfg=1 sketch_cfg=1" in result.stdout
    assert "content_cfg=1 sketch_cfg=2" in result.stdout
    assert "content_cfg=2 sketch_cfg=0" in result.stdout
    assert "content_cfg=2 sketch_cfg=1" in result.stdout
    assert "content_cfg=2 sketch_cfg=2" in result.stdout
    assert "Nothing submitted" in result.stdout


def test_cfg_ablation_dry_run_routes_checkpoint_wandb_and_r2_outputs() -> None:
    """Qualitative outputs retain arm identity through downstream systems."""
    result = _dry_run(("--ablation-id", "test-grid"))

    assert result.returncode == 0, result.stderr
    arm_line = next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("DRY RUN: content_cfg=0 sketch_cfg=2")
    )
    assert "r2://intermediate-data/checkpoints/flow_sketch_prelim/model.ckpt" in arm_line
    assert "run_name=cfg-c0-s2-validation" in arm_line
    assert "model.validation_cfg_strength=0" in arm_line
    assert "model.validation_sketch_cfg_strength=2" in arm_line
    assert "trainer.limit_val_batches=20" in arm_line
    assert "run_name=cfg-c0-s2-audio" in arm_line
    assert "model.test_cfg_strength=0" in arm_line
    assert "model.test_sketch_cfg_strength=2" in arm_line
    assert "datamodule.batch_size=32" in arm_line
    assert "trainer.limit_predict_batches=1" in arm_line
    assert "/test-grid/c0-s2/validation" in arm_line
    assert "/test-grid/c0-s2/audio" in arm_line


def test_cfg_ablation_rejects_missing_ablation_id_before_planning() -> None:
    """Report operator errors before launch planning."""
    result = _dry_run(("--ablation-id",))

    assert result.returncode == 2
    assert "--ablation-id requires a value" in result.stderr
    assert "DRY RUN:" not in result.stdout


def test_cfg_ablation_execute_clears_remote_skypilot_credentials(tmp_path: Path) -> None:
    """Local dispatch does not inherit incompatible remote-client authentication.

    :param tmp_path: Temporary directory for the non-launching command sentinel.
    """
    uv_sentinel = tmp_path / "uv"
    uv_sentinel.write_text(
        '#!/bin/bash\n[[ -z "${SKYPILOT_SERVICE_ACCOUNT_TOKEN+x}" ]]\n',
        encoding="utf-8",
    )
    uv_sentinel.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["SKYPILOT_SERVICE_ACCOUNT_TOKEN"] = str(tmp_path)

    result = subprocess.run(  # noqa: S603 — repository-owned script
        [str(_LAUNCHER), "--execute", "--ablation-id", "credential-test"],
        cwd=_REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "DRY RUN:" not in result.stdout


def test_cfg_ablation_experiment_loads_through_eval_entrypoint() -> None:
    """Hydra discovers the experiment through the installed evaluation CLI."""
    result = subprocess.run(  # noqa: S603 — repository-owned module
        [
            sys.executable,
            "-m",
            "synth_setter.cli.eval",
            "--cfg",
            "job",
            "experiment=surge/flow_sketch_cfg_ablation",
        ],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "consumed_train_config_id: flow_sketch_prelim" in result.stdout
    assert "predict_file: ${datamodule.dataset_root}/val.lance" in result.stdout
    assert "test_sample_steps: 50" in result.stdout
