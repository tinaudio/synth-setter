"""Behavioral tests for self-describing model checkpoints."""

from pathlib import Path

import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.demos.boring_classes import BoringModel
from omegaconf import DictConfig, OmegaConf, open_dict

from synth_setter.cli.train import train
from synth_setter.models.checkpoint_bundle import load_model_checkpoint
from synth_setter.models.slap_module import SLAPModule


@pytest.mark.slow
def test_load_model_checkpoint_missing_bundle_fails_with_actionable_error(tmp_path: Path) -> None:
    """Reject a legacy checkpoint without attempting architecture inference.

    :param tmp_path: Temporary checkpoint destination.
    """
    checkpoint_path = tmp_path / "legacy.ckpt"
    torch.save({"state_dict": {}}, checkpoint_path)

    with pytest.raises(ValueError, match="Legacy checkpoints are smoke-only"):
        load_model_checkpoint(checkpoint_path)


def test_load_model_checkpoint_missing_state_tensor_fails_strictly(tmp_path: Path) -> None:
    """Reject a bundled checkpoint whose state is incomplete.

    :param tmp_path: Temporary checkpoint destination.
    """
    checkpoint_path = tmp_path / "incomplete.ckpt"
    model = BoringModel()
    torch.save(
        {
            "state_dict": {"layer.weight": model.layer.weight.detach().clone()},
            "synth_setter_model_bundle": {
                "schema_version": 1,
                "model": {"_target_": "lightning.pytorch.demos.boring_classes.BoringModel"},
            },
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="Missing key"):
        load_model_checkpoint(checkpoint_path)


def test_load_model_checkpoint_unknown_schema_version_fails(tmp_path: Path) -> None:
    """Reject a bundle schema newer than the loader understands.

    :param tmp_path: Temporary checkpoint destination.
    """
    checkpoint_path = tmp_path / "future.ckpt"
    torch.save(
        {
            "state_dict": {},
            "synth_setter_model_bundle": {"schema_version": 2, "model": {}},
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="invalid model bundle"):
        load_model_checkpoint(checkpoint_path)


@pytest.mark.slow
def test_load_model_checkpoint_real_slap_training_restores_online_and_ema_weights(
    cfg_slap_train_lance: DictConfig,
) -> None:
    """Load exact online and EMA tensors from a real one-step SLAP checkpoint.

    :param cfg_slap_train_lance: Tiny real Lance SLAP training configuration.
    """
    with open_dict(cfg_slap_train_lance):
        cfg_slap_train_lance.test = False
    HydraConfig().set_config(cfg_slap_train_lance)

    _, objects = train(cfg_slap_train_lance)
    checkpoint_path = Path(cfg_slap_train_lance.paths.output_dir) / "checkpoints" / "last.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    loaded = load_model_checkpoint(checkpoint_path)

    assert isinstance(loaded, SLAPModule)
    assert checkpoint["synth_setter_model_bundle"] == {
        "schema_version": 1,
        "model": OmegaConf.to_container(cfg_slap_train_lance.model, resolve=True),
    }
    loaded_state = loaded.state_dict()
    assert not torch.equal(
        checkpoint["state_dict"]["audio_encoder.projector.0.weight"],
        checkpoint["state_dict"]["audio_ema.projector.0.weight"],
    )
    assert torch.equal(
        loaded_state["audio_encoder.projector.0.weight"],
        checkpoint["state_dict"]["audio_encoder.projector.0.weight"],
    )
    assert torch.equal(
        loaded_state["audio_ema.projector.0.weight"],
        checkpoint["state_dict"]["audio_ema.projector.0.weight"],
    )
