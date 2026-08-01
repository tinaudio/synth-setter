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


def _run_captured_worker(tmp_path: Path, source: str) -> tuple[list[str], list[str], str]:
    """Launch locally, then execute the exact worker command against recording binaries.

    :param tmp_path: Root for fake executables and captures.
    :param source: Checkpoint source passed through both command layers.
    :returns: Rclone arguments, train arguments, and the worker's source environment value.
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
    train_capture = tmp_path / "train-arguments.txt"
    source_capture = tmp_path / "source.txt"
    _write_executable(
        bin_dir / "synth-setter-train",
        "#!/bin/bash\n"
        'printf \'%s\\n\' "$@" > "${TRAIN_CAPTURE}"\n'
        'printf \'%s\' "${SYNTH_SETTER_BASE_CHECKPOINT_SOURCE}" > "${SOURCE_CAPTURE}"\n',
    )

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
            "SOURCE_CAPTURE": str(source_capture),
            "TRAIN_CAPTURE": str(train_capture),
        },
    )
    return (
        rclone_capture.read_text().splitlines(),
        train_capture.read_text().splitlines(),
        source_capture.read_text(),
    )


def test_launcher_forwards_checkpoint_source_as_single_worker_arguments(tmp_path: Path) -> None:
    """The materialization and training commands retain an unsafe-looking source verbatim.

    :param tmp_path: Pytest-provided directory for fake worker executables and captures.
    """
    injected = tmp_path / "injected"
    source = f"r2:training-checkpoints/flow/base checkpoint; touch {injected}; #"

    rclone_args, train_args, source_value = _run_captured_worker(tmp_path, source)

    assert rclone_args == ["copyto", "--checksum", source, "/home/build/base.ckpt"]
    assert "model.base_checkpoint=/home/build/base.ckpt" in train_args
    assert source_value == source
    assert not injected.exists()
