"""Tests for Hydra sweep configurations."""

from pathlib import Path

import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "src/synth_setter/cli/train.py"
# `model=`/`data=` are mandatory after #687; `+run_name=` is required by the
# `hydra.run.dir` interpolation in configs/hydra/default.yaml. `ksin` pairs
# cleanly with `ffn` (no `dataset_root`, minimal config). `logger=[]` disables
# wandb/tensorboard for these throwaway subprocess runs. `~callbacks.lr_monitor`
# works around #517 — LearningRateMonitor hard-requires a logger and crashes at
# on_train_start when logger is empty.
overrides = [
    "model=ffn",
    "data=ksin",
    "+run_name=sweep",
    "logger=[]",
    "~callbacks.lr_monitor",
]


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_hydra_sweep(tmp_path: Path) -> None:
    """Test default hydra sweep.

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "model.optimizer.lr=0.005,0.01",
        "++trainer.fast_dev_run=true",
    ] + overrides

    run_sh_command(command)


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_hydra_sweep_ddp_sim(tmp_path: Path) -> None:
    """Test default hydra sweep with ddp sim.

    :param tmp_path: The temporary logging path.
    """
    # ksin's default `train_val_test_sizes` is 409M train samples; DDP-sim forks
    # the dataloader and the large shared-memory transfer trips
    # `RuntimeError: unable to resize file ... Invalid argument` in
    # torch.multiprocessing. Shrink + disable worker MP for the sweep smoke test.
    command = [
        startfile,
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "trainer=ddp_sim",
        "+trainer.max_epochs=3",
        "+trainer.limit_train_batches=1",
        "+trainer.limit_val_batches=1",
        "+trainer.limit_test_batches=1",
        "data.train_val_test_sizes=[100,100,100]",
        "data.num_workers=0",
        "model.optimizer.lr=0.005,0.01,0.02",
    ] + overrides
    run_sh_command(command)
