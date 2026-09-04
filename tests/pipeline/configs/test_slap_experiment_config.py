"""Hydra contract tests for shipped SLAP training experiments."""

import hydra
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import DictConfig

from synth_setter.models.slap_module import SLAPModule
from tests.helpers.run_if import RunIf


def _compose_slap_experiment() -> DictConfig:
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        return compose(
            config_name="train.yaml",
            overrides=["experiment=surge/slap_ast_audio_mlp_param"],
        )


def test_slap_ast_audio_mlp_param_experiment_instantiates_complete_model() -> None:
    """Hydra must resolve mel conditioning and a concrete SLAPModule together."""
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
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.audio_encoder.parameters()
    )
    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.text_encoder.parameters()
    )


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_slap_ast_audio_mlp_param_model_overfits_one_batch() -> None:
    """The shipped backbones must reduce their fixed-pair objective substantially."""
    torch.manual_seed(7)
    cfg = _compose_slap_experiment()
    cfg.model.audio_encoder.encoder._args_[0].n_layers = 1
    model = hydra.utils.instantiate(cfg.model).cuda()
    batch = {
        "audio": None,
        "mel": torch.randn(4, 2, 128, 401, device="cuda"),
        "params": torch.rand(4, 7, device="cuda"),
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial_loss = model._losses(batch)["total_loss"].detach()

    for _ in range(30):
        optimizer.zero_grad()
        loss = model._losses(batch)["total_loss"]
        loss.backward()
        optimizer.step()

    assert loss < 0.75 * initial_loss
