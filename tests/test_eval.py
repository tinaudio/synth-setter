"""Tests for the ``synth-setter-eval`` CLI entrypoint.

Every test composes a Hydra ``cfg`` and drives the in-process ``evaluate(cfg)``
entrypoint. Helper-level unit tests live in the sibling ``test_eval_*`` modules:
postprocessing argv in ``test_eval_postprocessing``, metric IO in
``test_eval_metrics``, and R2 upload / CLI e2e in ``test_eval_upload``.
``tests/_meta/test_entrypoint_test_modules.py`` enforces that no private
``synth_setter.cli`` helper is imported here.
"""

import math
import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from synth_setter.cli.eval import evaluate
from synth_setter.cli.train import train
from synth_setter.data.vst import param_specs
from synth_setter.workspace import operator_workspace
from tests.conftest import NUM_FIXTURE_SAMPLES
from tests.helpers.run_if import RunIf


@pytest.mark.requires_vst
@pytest.mark.slow
def test_evaluate_runs_oracle_with_null_ckpt_path(
    tmp_path: Path,
    surge_xt_smoke_datasets: Path,
) -> None:
    """Fake oracle returns ``batch["params"]`` verbatim, so ``test/param_mse`` is exactly zero.

    The load-bearing invariant is that ``ckpt_path=null`` survives Hydra
    composition into ``evaluate()`` and the oracle's exact-zero MSE reaches
    the metric dict.

    :param tmp_path: Pinned as Hydra ``paths.output_dir`` / ``paths.log_dir``.
    :param surge_xt_smoke_datasets: Holds ``{train,val,test}.h5`` + ``stats.npz``.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=surge/test-mps-fake-oracle",
                "trainer=cpu",
                # The experiment defaults to mode=predict; this invariant is test-mode.
                "mode=test",
                f"model.net.d_out={len(param_specs['surge_4'])}",
                "callbacks.log_per_param_mse.param_spec=surge_4",
            ],
        )

    with open_dict(cfg):
        cfg.paths.root_dir = str(operator_workspace())
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.datamodule.dataset_root = str(surge_xt_smoke_datasets)
        cfg.datamodule.predict_file = str(surge_xt_smoke_datasets / "test.h5")
        cfg.datamodule.batch_size = 1
        cfg.datamodule.num_workers = 0
        cfg.ckpt_path = None

    HydraConfig().set_config(cfg)
    try:
        metric_dict, _ = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    param_mse = metric_dict["test/param_mse"]
    assert isinstance(param_mse, torch.Tensor)
    assert param_mse.numel() == 1
    assert param_mse.dtype.is_floating_point
    assert torch.isfinite(param_mse), f"oracle test/param_mse must be finite; got {param_mse!r}"
    assert param_mse.item() == 0.0


@pytest.mark.requires_vst
@pytest.mark.slow
def test_evaluate_predict_shuffle_pred_audio_reaches_metrics(
    tmp_path: Path,
    cfg_surge_xt: DictConfig,
    cfg_surge_xt_eval: DictConfig,
) -> None:
    """``evaluate()`` predict mode with the shuffle on still lands finite audio metrics.

    Drives the real train->eval roundtrip end-to-end with ``shuffle_pred_audio``
    enabled, exercising the ``evaluate()`` -> ``_run_predict_postprocessing`` ->
    shuffle wiring (rank-zero gate, cfg default, render->shuffle->metrics order).
    Asserts the metrics still aggregate and every pred.wav survives the permutation
    (#489); the shuffle deliberately mismatches pred to target, so metric magnitudes
    are not asserted — only that they are present and finite.

    :param tmp_path: Shared output dir for the train run and the eval run.
    :param cfg_surge_xt: Surge XT smoke-test training config.
    :param cfg_surge_xt_eval: Matching predict-mode eval config (render + metrics on).
    """
    HydraConfig().set_config(cfg_surge_xt)
    train(cfg_surge_xt)
    assert Path(cfg_surge_xt_eval.ckpt_path).exists()

    with open_dict(cfg_surge_xt_eval):
        cfg_surge_xt_eval.evaluation.shuffle_pred_audio = True
        cfg_surge_xt_eval.evaluation.shuffle_seed = 42

    HydraConfig().set_config(cfg_surge_xt_eval)
    metric_dict, _ = evaluate(cfg_surge_xt_eval)

    audio_metric_keys = [key for key in metric_dict if key.startswith("audio/")]
    assert audio_metric_keys, f"expected audio/* metrics; got {sorted(metric_dict)}"
    for key in audio_metric_keys:
        assert math.isfinite(float(metric_dict[key])), f"{key} not finite: {metric_dict[key]!r}"

    sample_dirs = sorted(d for d in (tmp_path / "audio").iterdir() if d.is_dir())
    assert len(sample_dirs) == NUM_FIXTURE_SAMPLES
    for sample_dir in sample_dirs:
        assert (sample_dir / "pred.wav").is_file()


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_eval(tmp_path: Path, cfg_train: DictConfig, cfg_eval: DictConfig) -> None:
    """Train for 1 epoch with ``train.py`` then evaluate the resulting checkpoint with ``eval.py``.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    :param cfg_eval: A DictConfig containing a valid evaluation configuration.
    """
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.test = True
    with open_dict(cfg_eval):
        cfg_eval.trainer.accelerator = "gpu"

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")

    HydraConfig().set_config(cfg_eval)
    test_metric_dict, _ = evaluate(cfg_eval)

    assert math.isfinite(test_metric_dict["test/loss"].item())
    assert (
        abs(train_metric_dict["test/loss"].item() - test_metric_dict["test/loss"].item()) < 0.001
    )


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_validate(tmp_path: Path, cfg_train: DictConfig, cfg_eval: DictConfig) -> None:
    """Train one epoch then validate the checkpoint via ``eval.py`` ``mode=validate``.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    :param cfg_eval: A DictConfig containing a valid evaluation configuration.
    """
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.test = False
    with open_dict(cfg_eval):
        cfg_eval.trainer.accelerator = "gpu"

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_eval.mode = "validate"

    HydraConfig().set_config(cfg_eval)
    val_metric_dict, _ = evaluate(cfg_eval)

    assert math.isfinite(val_metric_dict["val/loss"].item())
    assert abs(train_metric_dict["val/loss"].item() - val_metric_dict["val/loss"].item()) < 0.001
