"""Exercise normalization checkpointing through real two-rank training."""

from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from synth_setter.cli.train import train


@pytest.mark.slow
@pytest.mark.parametrize("cfg_pyfdn_train", ["pyfdn/flow_ast_online"], indirect=True)
def test_online_ast_normalization_two_cpu_ranks_restores_identical_features(
    cfg_pyfdn_train: DictConfig,
) -> None:
    """Distributed fitting preserves training statistics in a portable checkpoint.

    :param cfg_pyfdn_train: One-step online-AST configuration over real Lance rows.
    """
    stats_path = Path(cfg_pyfdn_train.datamodule.dataset_root) / "stats.npz"
    np.savez(stats_path, mean=np.float32(-40.0), std=np.float32(20.0))
    with open_dict(cfg_pyfdn_train):
        cfg_pyfdn_train.estimate_normalization_stats = True
        cfg_pyfdn_train.seed = 1234
        cfg_pyfdn_train.trainer.devices = 2
        cfg_pyfdn_train.trainer.strategy = "ddp_spawn"
    HydraConfig().set_config(cfg_pyfdn_train)

    train(cfg_pyfdn_train)

    checkpoint_path = Path(cfg_pyfdn_train.paths.output_dir) / "checkpoints" / "last.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stats_path.unlink()
    restored = hydra.utils.instantiate(cfg_pyfdn_train.model)
    restored.load_state_dict(checkpoint["state_dict"])
    frontend = restored.encoder.frontend
    waveform = torch.randn(1, frontend.in_dim, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(frontend(waveform), (frontend.forward_raw(waveform) + 40.0) / 20.0)
