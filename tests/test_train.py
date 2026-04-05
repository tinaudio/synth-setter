import os
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.train import train
from tests.helpers.run_if import RunIf

# TODO(#39): replace hardcoded accelerator overrides with --accelerator pytest flag
# TODO(#40): add @pytest.mark.ram gate for memory-intensive CPU tests test_train_fast_dev_run


def test_train_fast_dev_run_tiny_model_tiny_data(cfg_train: DictConfig) -> None:
    # plumb:req-8fdf646c
    # plumb:req-b524cdce
    # plumb:req-c3aed2b5
    # plumb:req-ac34a5ca
    """Run for 1 train, val and test step with small batch size, no compile.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        # Prevent CPU unittest OOM by shrinking model,
        # batch, training example, dataset size.
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.data.batch_size = 32
        cfg_train.model.net.channels = 4
        cfg_train.model.net.encoder_blocks = 1
        cfg_train.model.net.trunk_blocks = 1
        cfg_train.model.net.hidden_dim = 32
        cfg_train.data.signal_length = 64
        cfg_train.data.train_val_test_sizes = [4, 4, 4]
    train(cfg_train)


@pytest.mark.slow
def test_train_fast_dev_run(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step with torch.compile enabled.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.model.compile = True
    train(cfg_train)


@pytest.mark.gpu
@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_train: DictConfig) -> None:
    # plumb:req-a69d39b3
    """Run for 1 train, val and test step on GPU.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.data.batch_size = 32
    train(cfg_train)


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_fast_dev_run_gpu_compile(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step on GPU with torch.compile enabled.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.model.compile = True
    train(cfg_train)


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(cfg_train: DictConfig) -> None:
    """Train 1 epoch on GPU with mixed-precision.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.precision = 16
    train(cfg_train)


# TODO: fix val_check_interval incompatibility with check_val_every_n_epoch=None (#47)
@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_train: DictConfig) -> None:
    """Train 1 epoch with validation loop twice per epoch.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.val_check_interval = 0.5
    train(cfg_train)


@pytest.mark.slow
def test_train_ddp_sim(cfg_train: DictConfig) -> None:
    """Simulate DDP (Distributed Data Parallel) on 2 CPU processes.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 2
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 2
        cfg_train.trainer.strategy = "ddp_spawn"
        # Integer limits avoid the `fraction * batches < 1` error when the
        # fixture's fractions (0.01, 0.1) shrink too far under DDP sharding.
        cfg_train.trainer.limit_train_batches = 1
        cfg_train.trainer.limit_val_batches = 1
        cfg_train.trainer.limit_test_batches = 1
        # Shrink model, batch, and dataset to keep DDP-on-CPU fast.
        cfg_train.data.batch_size = 2
        cfg_train.data.signal_length = 64
        cfg_train.data.train_val_test_sizes = [4, 4, 4]
        cfg_train.model.net.channels = 4
        cfg_train.model.net.encoder_blocks = 1
        cfg_train.model.net.trunk_blocks = 1
        cfg_train.model.net.hidden_dim = 32
    train(cfg_train)


@pytest.mark.slow
def test_train_resume(tmp_path: Path, cfg_train: DictConfig) -> None:
    # plumb:req-931b548d
    # plumb:req-f778efcb
    # plumb:req-072e56c1
    # plumb:req-4962a50a
    # plumb:req-5af60254
    # plumb:req-51621478
    """Run 1 epoch, finish, and resume for another epoch.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1

    HydraConfig().set_config(cfg_train)
    metric_dict_1, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_train.trainer.max_epochs = 2

    metric_dict_2, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "epoch_001.ckpt" in files
    assert "epoch_002.ckpt" not in files

    assert metric_dict_1["train/acc"] < metric_dict_2["train/acc"]
    assert metric_dict_1["val/acc"] < metric_dict_2["val/acc"]
