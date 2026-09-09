"""Hydra contract tests for shipped SLAP training experiments."""

import hydra
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import DictConfig

from synth_setter.models.slap_module import SLAPModule
from tests.helpers.run_if import RunIf

_SLAP_EXPERIMENTS = ("surge/slap_ast_audio_vst_ff_param",)
_AST_TARGET = "synth_setter.models.components.transformer.AudioSpectrogramTransformer"


def _compose_slap_experiment(experiment: str) -> DictConfig:
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        return compose(
            config_name="train.yaml",
            overrides=[f"experiment={experiment}"],
        )


def _shrink_ast_layers(cfg: DictConfig) -> None:
    cfg.model.param_encoder.encoder.n_layers = 1
    for arm in (cfg.model.audio_encoder,):
        if "_args_" not in arm.encoder:
            continue
        for layer in arm.encoder._args_:
            if layer.get("_target_") == _AST_TARGET:
                layer.n_layers = 1


@pytest.mark.parametrize("experiment", _SLAP_EXPERIMENTS)
def test_slap_experiment_instantiates_complete_model(experiment: str) -> None:
    """Hydra must resolve mel conditioning and a concrete SLAPModule together.

    :param experiment: Shipped SLAP experiment name.
    """
    cfg = _compose_slap_experiment(experiment)

    model = hydra.utils.instantiate(cfg.model)

    assert isinstance(model, SLAPModule)
    assert cfg.datamodule.conditioning == "mel"
    assert cfg.datamodule.ot is False
    assert cfg.model.audio_input_key == "mel"


@pytest.mark.parametrize("experiment", _SLAP_EXPERIMENTS)
def test_slap_model_accepts_paired_surge_batch(experiment: str) -> None:
    """Compute a finite objective and route gradients into every trainable weight.

    :param experiment: Shipped SLAP experiment name.
    """
    cfg = _compose_slap_experiment(experiment)
    _shrink_ast_layers(cfg)
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
        for parameter in model.param_encoder.parameters()
        if parameter.requires_grad
    )


def test_slap_param_arm_frozen_weights_are_only_unused_inverse_projection() -> None:
    """Only the projection's unused token-to-parameter half may skip training."""
    cfg = _compose_slap_experiment("surge/slap_ast_audio_vst_ff_param")
    _shrink_ast_layers(cfg)
    model = hydra.utils.instantiate(cfg.model)

    frozen = [
        name
        for name, parameter in model.param_encoder.named_parameters()
        if not parameter.requires_grad
    ]

    assert len(frozen) == 1
    assert frozen[0].endswith("projection._out_projection")


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
@pytest.mark.parametrize("experiment", _SLAP_EXPERIMENTS)
def test_slap_model_overfits_one_batch(experiment: str) -> None:
    """Check fixed-batch loss reduction with frozen target arms.

    Manual optimizer steps bypass EMA updates; the reduction threshold is a smoke check, not
    evidence of full memorization, non-collapse, or a nonzero cosine-loss floor.

    :param experiment: Shipped SLAP experiment name.
    """
    torch.manual_seed(7)
    cfg = _compose_slap_experiment(experiment)
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

    assert loss < 0.5 * initial_loss
