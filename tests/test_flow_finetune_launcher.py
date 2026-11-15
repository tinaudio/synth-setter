"""Behavior tests for the simulator-feedback arm launcher."""

import os
from pathlib import Path

import sh


def _write_executable(path: Path, content: str) -> None:
    """Write one executable used by the local worker-command harness.

    :param path: Fake executable destination.
    :param content: Complete shell program.
    """
    path.write_text(content)
    path.chmod(0o755)


def test_launcher_preserves_checkpoint_source_without_executing_shell_input(
    tmp_path: Path,
) -> None:
    """The worker receives the source verbatim without evaluating its contents.

    :param tmp_path: Pytest-provided directory for fake worker executables and captures.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    worker_capture = tmp_path / "worker-command.txt"
    _write_executable(
        bin_dir / "uv",
        """#!/bin/bash
for argument in "$@"; do
  case "${argument}" in
    skypilot_launch.cmd=*)
      printf '%s' "${argument#skypilot_launch.cmd=}" > "${WORKER_CAPTURE}"
      ;;
  esac
done
""",
    )

    rclone_capture = tmp_path / "rclone-arguments.txt"
    _write_executable(
        bin_dir / "rclone",
        '#!/bin/bash\nprintf \'%s\\n\' "$@" > "${RCLONE_CAPTURE}"\n',
    )
    source_capture = tmp_path / "source.txt"
    _write_executable(
        bin_dir / "synth-setter-train",
        '#!/bin/bash\nprintf \'%s\' "${SYNTH_SETTER_BASE_CHECKPOINT_SOURCE}" > "${SOURCE_CAPTURE}"\n',
    )

    launcher = (
        Path(__file__).resolve().parent.parent
        / "jobs"
        / "train"
        / "torchsynth"
        / "launch_flow_finetune_arms.sh"
    )
    injected = tmp_path / "injected"
    source = f"r2:checkpoints/base checkpoint; touch {injected}; #"
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
    sh.Command("/bin/bash")(
        "-c",
        captured_worker[1:-1],
        _env={
            **env,
            "RCLONE_CAPTURE": str(rclone_capture),
            "SOURCE_CAPTURE": str(source_capture),
        },
    )

    assert rclone_capture.read_text().splitlines() == [
        "copyto",
        "--checksum",
        source,
        "/home/build/base.ckpt",
    ]
    assert source_capture.read_text() == source
    assert not injected.exists()
