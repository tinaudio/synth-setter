"""Hydra contract tests for shipped SLAP training experiments."""

import hydra
import torch
from hydra import compose, initialize_config_module
from omegaconf import DictConfig

from synth_setter.models.slap_module import SLAPModule


def _compose_slap_experiment() -> DictConfig:
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        return compose(
            config_name="train.yaml",
            overrides=["experiment=surge/slap_ast_audio_mlp_param"],
        )


def test_slap_ast_audio_mlp_param_experiment_instantiates_complete_model() -> None:
    """Instantiate the shipped AST-audio/MLP-parameter training model."""
    cfg = _compose_slap_experiment()

    model = hydra.utils.instantiate(cfg.model)

    assert isinstance(model, SLAPModule)
    assert cfg.datamodule.conditioning == "mel"
    assert cfg.datamodule.ot is False
    assert cfg.model.audio_input_key == "mel"


def test_slap_ast_audio_mlp_param_model_accepts_paired_surge_batch() -> None:
    """Compute a finite objective from production-shaped mel and parameter inputs."""
    cfg = _compose_slap_experiment()
    cfg.model.audio_encoder.encoder._args_[0].n_layers = 1
    model = hydra.utils.instantiate(cfg.model)
    batch = {
        "audio": None,
        "mel": torch.randn(2, 2, 128, 401),
        "params": torch.rand(2, 7),
    }

    loss = model.training_step(batch, batch_idx=0)

    assert torch.isfinite(loss)
