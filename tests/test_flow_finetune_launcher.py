"""Behavior tests for the simulator-feedback arm launcher."""

import os
from pathlib import Path

import sh


def test_launcher_forwards_checkpoint_source_as_single_worker_arguments(tmp_path: Path) -> None:
    """The materialization and training commands retain an unsafe-looking source verbatim.

    :param tmp_path: Pytest-provided directory for fake worker executables and captures.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    worker_capture = tmp_path / "worker-command.txt"
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        """#!/bin/bash
for argument in "$@"; do
  case "${argument}" in
    skypilot_launch.cmd=*)
      printf '%s' "${argument#skypilot_launch.cmd=}" > "${WORKER_CAPTURE}"
      ;;
  esac
done
"""
    )
    fake_uv.chmod(0o755)

    rclone_capture = tmp_path / "rclone-arguments.txt"
    fake_rclone = bin_dir / "rclone"
    fake_rclone.write_text('#!/bin/bash\nprintf \'%s\\n\' "$@" > "${RCLONE_CAPTURE}"\n')
    fake_rclone.chmod(0o755)
    train_capture = tmp_path / "train-arguments.txt"
    fake_train = bin_dir / "synth-setter-train"
    fake_train.write_text('#!/bin/bash\nprintf \'%s\\n\' "$@" > "${TRAIN_CAPTURE}"\n')
    fake_train.chmod(0o755)

    injected = tmp_path / "injected"
    source = f"r2:training-checkpoints/flow/base checkpoint; touch {injected}; #"
    launcher = (
        Path(__file__).resolve().parent.parent
        / "jobs"
        / "train"
        / "torchsynth"
        / "launch_flow_finetune_arms.sh"
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "WORKER_CAPTURE": str(worker_capture),
    }

    sh.Command("/bin/bash")(
        str(launcher),
        "--base-checkpoint",
        source,
        "--arms",
        "flow_finetune",
        "--seeds",
        "1",
        "--execute",
        _env=env,
    )
    captured_worker = worker_capture.read_text()
    assert captured_worker.startswith('"') and captured_worker.endswith('"')
    sh.Command("/bin/bash")(
        "-c",
        captured_worker[1:-1],
        _env={
            **env,
            "RCLONE_CAPTURE": str(rclone_capture),
            "TRAIN_CAPTURE": str(train_capture),
        },
    )

    assert rclone_capture.read_text().splitlines() == [
        "copyto",
        "--checksum",
        source,
        "/home/build/base.ckpt",
    ]
    assert "model.base_checkpoint=/home/build/base.ckpt" in train_capture.read_text().splitlines()
    assert f"model.base_checkpoint_source={source}" in train_capture.read_text().splitlines()
    assert not injected.exists()
