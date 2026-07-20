"""Regression coverage for evaluation with partial W&B experiment overlays."""

import math
from pathlib import Path

from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from synth_setter.cli.eval import evaluate
from synth_setter.workspace import operator_workspace


def test_evaluate_csv_logger_with_partial_wandb_overlay_returns_metrics(
    tmp_path: Path,
    fake_surge_smoke_datasets: Path,
) -> None:
    """A train experiment composed with CSV logging evaluates without W&B.

    :param tmp_path: Isolated evaluation output and CSV log directory.
    :param fake_surge_smoke_datasets: Tiny real Lance splits consumed by evaluation.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=surge/fake_oracle",
                "logger=csv",
                "mode=test",
            ],
        )
    with open_dict(cfg):
        cfg.paths.root_dir = str(operator_workspace())
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.datamodule.dataset_root = str(fake_surge_smoke_datasets)
        cfg.datamodule.batch_size = 1
        cfg.datamodule.num_workers = 0
        cfg.datamodule.use_saved_mean_and_variance = True
        cfg.ckpt_path = None
        cfg.trainer.limit_test_batches = 1

    assert OmegaConf.select(cfg, "logger.csv._target_") is not None
    assert OmegaConf.select(cfg, "logger.wandb._target_") is None

    HydraConfig().set_config(cfg)
    try:
        metric_dict, _ = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert math.isfinite(metric_dict["test/param_mse"].item())
    assert (tmp_path / "csv" / "version_0" / "metrics.csv").is_file()
