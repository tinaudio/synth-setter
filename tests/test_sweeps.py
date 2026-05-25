"""Tests for Hydra sweep configurations."""

from pathlib import Path

import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "src/synth_setter/cli/train.py"
# Group selections must precede value overrides — later `model=`/`data=`
# would re-compose the group and silently drop earlier `model.*`/`data.*`
# tweaks (collapsing sweeps to a single run).
_SWEEP_OVERRIDES: tuple[str, ...] = (
    "model=ffn",  # satisfies `model: ???` in configs/train.yaml
    "data=ksin",  # satisfies `data: ???` in configs/train.yaml
    "+run_name=sweep",  # consumed by hydra.run.dir interpolation
    "logger=[]",  # silence wandb/tensorboard for throwaway runs
    "~callbacks.lr_monitor",  # #517: LearningRateMonitor crashes with empty logger
)
# Shrink ksin split + disable worker MP — ksin's default sizes blow out
# torch.multiprocessing shmem when DDP forks the dataloader.
_DDP_SIM_SPLIT: tuple[int, int, int] = (100, 100, 100)
_DDP_SIM_NUM_WORKERS = 0


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
        *_SWEEP_OVERRIDES,
        "hydra.sweep.dir=" + str(tmp_path),
        "model.optimizer.lr=0.005,0.01",
        "++trainer.fast_dev_run=true",
    ]

    run_sh_command(command)


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_hydra_sweep_ddp_sim(tmp_path: Path) -> None:
    """Test default hydra sweep with ddp sim.

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        *_SWEEP_OVERRIDES,
        "hydra.sweep.dir=" + str(tmp_path),
        "trainer=ddp_sim",
        "+trainer.max_epochs=3",
        "+trainer.limit_train_batches=1",
        "+trainer.limit_val_batches=1",
        "+trainer.limit_test_batches=1",
        f"data.train_val_test_sizes={list(_DDP_SIM_SPLIT)}",
        f"data.num_workers={_DDP_SIM_NUM_WORKERS}",
        "model.optimizer.lr=0.005,0.01,0.02",
    ]
    run_sh_command(command)
