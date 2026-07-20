"""End-to-end coverage for stage-aware Lance evaluation on partial local data."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from synth_setter.cli.eval import evaluate


def test_evaluate_test_mode_partial_lance_root_returns_metric(
    cfg_train_lance: DictConfig,
) -> None:
    """Real ``evaluate`` consumes ``test.lance`` when train and val are absent.

    :param cfg_train_lance: Tiny production-composed Lance configuration.
    """
    dataset_root = Path(cfg_train_lance.datamodule.dataset_root)
    shutil.rmtree(dataset_root / "train.lance")
    shutil.rmtree(dataset_root / "val.lance")
    with open_dict(cfg_train_lance):
        cfg_train_lance.mode = "test"
        cfg_train_lance.ckpt_path = None

    HydraConfig().set_config(cfg_train_lance)
    metric_dict, object_dict = evaluate(cfg_train_lance)

    assert math.isfinite(metric_dict["test/param_mse"].item())
    assert Path(object_dict["datamodule"].dataset_root) == dataset_root
