import os
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.eval import evaluate
from src.train import train
from tests.helpers.run_if import RunIf


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_eval(tmp_path: Path, cfg_train: DictConfig, cfg_eval: DictConfig) -> None:
    """Tests training and evaluation by training for 1 epoch with `train.py` then evaluating with
    `eval.py`.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    :param cfg_eval: A DictConfig containing a valid evaluation configuration.
    """
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.test = True
    with open_dict(cfg_eval):
        cfg_eval.trainer.accelerator = "gpu"
        # `configs/eval.yaml` defaults to `data: surge_mini`, which hardcodes a researcher-local
        # path that does not exist on the CI GPU runner. Reuse the training data config so the
        # test exercises a self-consistent train->eval roundtrip.
        cfg_eval.data = cfg_train.data
        # `configs/eval.yaml` also defaults `model: surge_flow` and `callbacks: eval_surge`, which
        # do not match the checkpoint produced by `cfg_train` (ksin feedforward). Align both so the
        # evaluator loads the same LightningModule the trainer just saved.
        cfg_eval.model = cfg_train.model
        cfg_eval.callbacks = cfg_train.callbacks

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")

    HydraConfig().set_config(cfg_eval)
    test_metric_dict, _ = evaluate(cfg_eval)

    # `ksin_ff_module.test_step` logs `test/loss` (MSE), not `test/acc`. Use loss for the sanity
    # bound and parity check — the train-time test phase and the standalone eval should produce
    # identical `test/loss` on the same checkpoint and data.
    assert test_metric_dict["test/loss"] < float("inf")
    assert (
        abs(train_metric_dict["test/loss"].item() - test_metric_dict["test/loss"].item()) < 0.001
    )
