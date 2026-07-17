"""Process-boundary regression tests for interrupted training."""

import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_EXCEPTION_MARKER_ENV = "SYNTH_SETTER_TEST_EXCEPTION_MARKER"
_LOGGER_MARKER_ENV = "SYNTH_SETTER_TEST_LOGGER_MARKER"
_READY_FIFO_ENV = "SYNTH_SETTER_TEST_READY_FIFO"


def _wait_for_fit_start(process: subprocess.Popen[str], ready_fd: int) -> str:
    """Wait for the callback FIFO while retaining subprocess output for failures.

    :param process: Training subprocess whose stdout is captured.
    :param ready_fd: Non-blocking callback FIFO descriptor.
    :returns: Output emitted before fit started.
    """
    assert process.stdout is not None
    output: list[str] = []
    deadline = time.monotonic() + 60
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pytest.fail("training did not reach on_train_start within 60 seconds")
        readable, _, _ = select.select([ready_fd, process.stdout], [], [], remaining)
        if ready_fd in readable and os.read(ready_fd, 1) == b"1":
            return "".join(output)
        if process.stdout in readable:
            line = process.stdout.readline()
            if line:
                output.append(line)
            elif process.poll() is not None:
                pytest.fail(f"training exited before on_train_start:\n{''.join(output)}")


@pytest.mark.skipif(os.name != "posix", reason="SIGTERM and FIFO synchronization require POSIX")
def test_train_sigterm_after_fit_starts_exits_nonzero_after_cleanup(tmp_path: Path) -> None:
    """SIGTERM produces status 143 after interruption hooks and logger cleanup.

    :param tmp_path: Isolated Hydra output and synchronization directory.
    """
    ready_fifo = tmp_path / "fit-ready.fifo"
    exception_marker = tmp_path / "exception.txt"
    logger_marker = tmp_path / "logger.txt"
    os.mkfifo(ready_fifo)
    ready_fd = os.open(ready_fifo, os.O_RDONLY | os.O_NONBLOCK)
    train_executable = Path(sys.executable).with_name("synth-setter-train")
    repo_root = Path(__file__).parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(repo_root), os.environ.get("PYTHONPATH", "")]),
        _EXCEPTION_MARKER_ENV: str(exception_marker),
        _LOGGER_MARKER_ENV: str(logger_marker),
        _READY_FIFO_ENV: str(ready_fifo),
    }
    command = [
        str(train_executable),
        "datamodule=ksin",
        "model=ffn",
        "trainer=cpu",
        "logger=csv",
        "logger.csv._target_=tests.helpers.sigterm_callback.SignalFinalizeLogger",
        "test=false",
        "extras.print_config=false",
        "model.compile=false",
        "datamodule.num_workers=0",
        "datamodule.batch_size=1",
        "datamodule.train_val_test_sizes=[1000,2,2]",
        "+trainer.max_epochs=1000",
        "+trainer.limit_val_batches=0",
        "+trainer.num_sanity_val_steps=0",
        "+trainer.enable_progress_bar=false",
        "callbacks.model_checkpoint=null",
        "callbacks.lr_monitor=null",
        "callbacks.model_summary=null",
        "callbacks.rich_progress_bar=null",
        "callbacks.plot_pos_enc=null",
        "callbacks.plot_proj=null",
        "callbacks.plot_proj_ii=null",
        "+callbacks.sigterm_lifecycle._target_=tests.helpers.sigterm_callback.SignalLifecycleCallback",
        f"hydra.run.dir={tmp_path}",
    ]
    process = subprocess.Popen(  # noqa: S603 — fixed test CLI plus local Hydra overrides
        command,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        output = _wait_for_fit_start(process, ready_fd)
        process.send_signal(signal.SIGTERM)
        remaining_output, _ = process.communicate(timeout=60)
        output += remaining_output
    finally:
        os.close(ready_fd)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)

    assert "Received SIGTERM" in output
    assert exception_marker.read_text() == "SIGTERMException"
    assert logger_marker.read_text() == "failed"
    assert process.returncode == 128 + signal.SIGTERM
