from pathlib import Path

import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "src/train.py"
overrides = ["logger=[]"]


# TODO(#514): migrate from mnist configs to ksin. test_experiments globs every
# experiment config including example.yaml, which overrides model=mnist — but
# configs/model/mnist.yaml does not exist in this repo.
@pytest.mark.skip(reason="Blocked on #514 — example.yaml references missing model=mnist")
@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_experiments(tmp_path: Path) -> None:
    """Test running all available experiment configs with `fast_dev_run=True.`

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "experiment=glob(*)",
        "hydra.sweep.dir=" + str(tmp_path),
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


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
    command = [
        startfile,
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "trainer=ddp_sim",
        "trainer.max_epochs=3",
        "+trainer.limit_train_batches=0.01",
        "+trainer.limit_val_batches=0.1",
        "+trainer.limit_test_batches=0.1",
        "model.optimizer.lr=0.005,0.01,0.02",
    ] + overrides
    run_sh_command(command)


# TODO(#514): migrate from mnist_optuna to ksin_optuna. hparams_search=mnist_optuna
# sweeps model.net.lin1_size/lin2_size/lin3_size — fields only defined on
# SimpleDenseNet, which is not referenced by any active model config.
@pytest.mark.skip(reason="Blocked on #514 — mnist_optuna sweeps fields not on active models")
@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_optuna_sweep(tmp_path: Path) -> None:
    """Test Optuna hyperparam sweeping.

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "hparams_search=mnist_optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=10",
        "hydra.sweeper.sampler.n_startup_trials=5",
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


# TODO(#514): migrate from mnist_optuna to ksin_optuna — same reason as test_optuna_sweep.
@pytest.mark.skip(reason="Blocked on #514 — mnist_optuna sweeps fields not on active models")
@pytest.mark.gpu
@RunIf(min_gpus=1, wandb=True)
@pytest.mark.slow
def test_optuna_sweep_ddp_sim_wandb(tmp_path: Path) -> None:
    """Test Optuna sweep with wandb logging and ddp sim.

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "hparams_search=mnist_optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=5",
        "trainer=ddp_sim",
        "trainer.max_epochs=3",
        "+trainer.limit_train_batches=0.01",
        "+trainer.limit_val_batches=0.1",
        "+trainer.limit_test_batches=0.1",
        "logger=wandb",
    ]
    run_sh_command(command)
