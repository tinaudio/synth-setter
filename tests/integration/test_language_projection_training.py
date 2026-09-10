"""Real synth data traverses language-conditioned training and checkpoint consumers."""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from lightning import seed_everything
from omegaconf import DictConfig, open_dict

from synth_setter.cli.eval import evaluate
from synth_setter.cli.train import train
from synth_setter.models.components.language_projection import LanguageParameterProjection
from synth_setter.pipeline.data.param_language import prepare_param_language
from tests.conftest import PLUGIN_PATH, _surge_smoke_render_config
from tests.helpers.grouped_projection_training import build_grouped_projection_config

pytestmark = [pytest.mark.slow, pytest.mark.requires_vst, pytest.mark.gpu]


def test_language_flow_real_audio_checkpoint_rerenders_without_metadata(
    tmp_path: Path, surge_xt_smoke_datasets: Path
) -> None:
    """Real audio trains flow and its source-free checkpoint produces finite rendered metrics.

    :param tmp_path: Isolated training and evaluation output root.
    :param surge_xt_smoke_datasets: Production-rendered Surge Lance splits.
    """
    artifact = prepare_param_language(surge_xt_smoke_datasets, "surge_4", "surge_4", dimension=128)
    cfg = build_grouped_projection_config(
        tmp_path, config_name="train.yaml", projection_name="language"
    )
    with open_dict(cfg):
        cfg.datamodule.dataset_root = str(surge_xt_smoke_datasets)
        cfg.trainer.accelerator = "gpu"
        cfg.trainer.max_steps = 2
        cfg.trainer.min_steps = 2
    HydraConfig().set_config(cfg)
    _, objects = train(cfg)
    assert objects["trainer"].global_step == 2
    projection = objects["model"].vector_field.projection
    assert isinstance(projection, LanguageParameterProjection)
    assert projection.text_adapter.weight.requires_grad
    artifact.unlink()
    eval_cfg = build_grouped_projection_config(
        tmp_path, config_name="eval.yaml", projection_name="language"
    )
    with open_dict(eval_cfg):
        eval_cfg.datamodule.dataset_root = str(surge_xt_smoke_datasets)
        eval_cfg.datamodule.predict_file = str(surge_xt_smoke_datasets / "test.lance")
        eval_cfg.trainer.accelerator = "gpu"
        eval_cfg.ckpt_path = objects["trainer"].checkpoint_callback.last_model_path
        eval_cfg.evaluation.render_vst = True
        eval_cfg.evaluation.compute_metrics = True
        render_values = _surge_smoke_render_config("surge_4", PLUGIN_PATH)
        eval_cfg.synth = render_values.pop("synth")
        eval_cfg.render = render_values
    HydraConfig().set_config(eval_cfg)
    evaluate(eval_cfg)
    metrics = pd.read_csv(tmp_path / "metrics" / "metrics.csv")
    assert np.isfinite(metrics.select_dtypes(include="number").to_numpy()).all()
    assert list((tmp_path / "audio").glob("sample_*/pred.wav"))


@pytest.mark.parametrize("cfg_slap_train_lance", ["surge/slap_language"], indirect=True)
def test_language_slap_real_audio_checkpoint_tests_without_metadata(
    cfg_slap_train_lance: DictConfig, surge_xt_smoke_datasets: Path
) -> None:
    """SLAP fits real audio and tests a restored checkpoint after the embedding source is removed.

    :param cfg_slap_train_lance: Tiny shipped language-SLAP configuration.
    :param surge_xt_smoke_datasets: Production-rendered Surge Lance splits.
    """
    artifact = prepare_param_language(surge_xt_smoke_datasets, "surge_4", "surge_4", dimension=128)
    cfg = cfg_slap_train_lance
    with open_dict(cfg):
        cfg.datamodule.dataset_root = str(surge_xt_smoke_datasets)
        cfg.trainer.accelerator = "gpu"
        cfg.trainer.precision = "32-true"
        cfg.trainer.max_steps = 2
        cfg.trainer.min_steps = 2
        cfg.trainer.max_epochs = 2
        cfg.test = False
    HydraConfig().set_config(cfg)
    seed_everything(cfg.seed, workers=True)
    initial = hydra.utils.instantiate(cfg.model)
    initial_adapter = (
        initial.param_encoder.encoder.patch_embed.projection.text_adapter.weight.detach().clone()
    )
    del initial
    _, objects = train(cfg)
    assert objects["trainer"].global_step == 2
    projection = objects["model"].param_encoder.encoder.patch_embed.projection
    assert not torch.equal(projection.text_adapter.weight.detach().cpu(), initial_adapter)
    assert isinstance(projection, LanguageParameterProjection)
    assert all(not parameter.requires_grad for parameter in projection.decoders.parameters())
    artifact.unlink()
    restored = hydra.utils.instantiate(cfg.model)
    results = objects["trainer"].test(
        restored,
        datamodule=objects["datamodule"],
        ckpt_path=objects["trainer"].checkpoint_callback.last_model_path,
    )
    assert results
    assert all(np.isfinite(value) for value in results[0].values())
