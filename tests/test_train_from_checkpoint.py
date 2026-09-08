"""Tests for explicit checkpoint-based training semantics."""

import os
import subprocess
import sys
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from lightning import Trainer
from omegaconf import DictConfig, OmegaConf, open_dict

from synth_setter.cli.train import train
from synth_setter.cli.train_from_checkpoint import train_from_checkpoint


def _continuation_cfg(source: DictConfig, output_dir: Path, max_epochs: int) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(source, resolve=False))
    with open_dict(cfg):
        cfg.paths.output_dir = str(output_dir)
        cfg.paths.log_dir = str(output_dir)
        cfg.trainer.max_epochs = max_epochs
        cfg.trainer.limit_train_batches = 0
        cfg.trainer.limit_val_batches = 0
        cfg.test = False
    return cast(DictConfig, cfg)


def test_train_from_checkpoint_incompatible_model_config_fails(tmp_path: Path) -> None:
    """Reject a runtime model override that conflicts with bundled construction.

    :param tmp_path: Temporary checkpoint destination.
    """
    checkpoint = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": {},
            "synth_setter_model_bundle": {
                "schema_version": 1,
                "model": {"_target_": "package.BundledModel"},
            },
        },
        checkpoint,
    )
    cfg = OmegaConf.create(
        {
            "model": {"_target_": "package.OverrideModel"},
            "ckpt_path": None,
            "training": {"resume": None, "weights_only_checkpoint": None},
        }
    )

    with pytest.raises(ValueError, match="cfg.model is incompatible"):
        train_from_checkpoint(cfg, checkpoint, mode="weights-only")


def test_train_from_checkpoint_full_resume_without_fit_fails(tmp_path: Path) -> None:
    """Reject full resume when no fit loop would restore trainer state.

    :param tmp_path: Temporary checkpoint destination.
    """
    checkpoint = tmp_path / "model.ckpt"
    model_config = {"_target_": "package.Model"}
    torch.save(
        {
            "state_dict": {},
            "synth_setter_model_bundle": {
                "schema_version": 1,
                "model": model_config,
            },
        },
        checkpoint,
    )
    cfg = OmegaConf.create(
        {
            "model": model_config,
            "train": False,
            "test": True,
            "ckpt_path": None,
            "training": {"resume": None, "weights_only_checkpoint": None},
        }
    )

    with pytest.raises(ValueError, match="full-resume mode requires train=true"):
        train_from_checkpoint(cfg, checkpoint, mode="full-resume")


@pytest.mark.slow
def test_train_from_checkpoint_modes_resume_or_start_new_optimizer_state(
    cfg_slap_train_lance: DictConfig,
    tmp_path: Path,
) -> None:
    """Distinguish full Lightning resume from a weights-only new run.

    :param cfg_slap_train_lance: Tiny real Lance SLAP training configuration.
    :param tmp_path: Isolated outputs for all three training runs.
    """
    with open_dict(cfg_slap_train_lance):
        cfg_slap_train_lance.test = False
    HydraConfig().set_config(cfg_slap_train_lance)
    _, initial_objects = train(cfg_slap_train_lance)
    checkpoint = Path(cfg_slap_train_lance.paths.output_dir) / "checkpoints" / "last.ckpt"
    assert initial_objects["trainer"].global_step == 1
    source_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_checkpoint["state_dict"]["audio_encoder.projector.0.weight"].fill_(0.125)
    source_checkpoint["state_dict"]["audio_ema.projector.0.weight"].fill_(0.75)
    torch.save(source_checkpoint, checkpoint)

    full_cfg = _continuation_cfg(cfg_slap_train_lance, tmp_path / "full", max_epochs=2)
    _, full_objects = train_from_checkpoint(full_cfg, checkpoint, mode="full-resume")

    weights_cfg = _continuation_cfg(cfg_slap_train_lance, tmp_path / "weights", max_epochs=1)
    with patch(
        "synth_setter.models.checkpoint_bundle.torch.load",
        wraps=torch.load,
    ) as checkpoint_load:
        _, weights_objects = train_from_checkpoint(weights_cfg, checkpoint, mode="weights-only")

    assert checkpoint_load.call_count == 1

    full_trainer = cast(Trainer, full_objects["trainer"])
    weights_trainer = cast(Trainer, weights_objects["trainer"])
    full_model = cast(torch.nn.Module, full_objects["model"])
    weights_model = cast(torch.nn.Module, weights_objects["model"])
    full_state = full_model.state_dict()
    weights_state = weights_model.state_dict()
    full_optimizer_state = next(iter(full_trainer.optimizers[0].state.values()))

    assert full_trainer.global_step == 1
    assert weights_trainer.global_step == 0
    assert int(full_optimizer_state["step"].item()) == 1
    assert not weights_trainer.optimizers[0].state
    for state_key in (
        "audio_encoder.projector.0.weight",
        "audio_ema.projector.0.weight",
    ):
        assert torch.equal(full_state[state_key], source_checkpoint["state_dict"][state_key])
        assert torch.equal(weights_state[state_key], source_checkpoint["state_dict"][state_key])


@pytest.mark.slow
def test_train_from_checkpoint_cli_real_slap_checkpoint_starts_weights_only_run(
    cfg_slap_train_lance: DictConfig,
    tmp_path: Path,
) -> None:
    """Drive the installed CLI from a real checkpoint into a new saved run.

    :param cfg_slap_train_lance: Tiny real Lance SLAP training configuration.
    :param tmp_path: Isolated CLI output root.
    """
    with open_dict(cfg_slap_train_lance):
        cfg_slap_train_lance.test = False
        cfg_slap_train_lance.model.optimizer.lr = 0.0
        cfg_slap_train_lance.model.ma_callback.every_n_steps = 999
    HydraConfig().set_config(cfg_slap_train_lance)
    train(cfg_slap_train_lance)
    checkpoint = Path(cfg_slap_train_lance.paths.output_dir) / "checkpoints" / "last.ckpt"
    source_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_checkpoint["state_dict"]["audio_encoder.projector.0.weight"].fill_(0.25)
    source_checkpoint["state_dict"]["audio_ema.projector.0.weight"].fill_(0.625)
    torch.save(source_checkpoint, checkpoint)
    output_dir = tmp_path / "cli-run"
    command = [
        str(Path(sys.executable).with_name("synth-setter-train-from-checkpoint")),
        f"checkpoint_path={checkpoint}",
        "checkpoint_mode=weights-only",
        "experiment=surge/slap_ast_audio_mlp_param",
        "trainer=cpu",
        "logger=[]",
        f"paths.root_dir={cfg_slap_train_lance.paths.root_dir}",
        f"paths.output_dir={output_dir}",
        f"paths.log_dir={output_dir}",
        f"hydra.run.dir={output_dir}",
        "test=false",
        f"datamodule.dataset_root={cfg_slap_train_lance.datamodule.dataset_root}",
        "datamodule.download_dataset_root_uri=null",
        "datamodule.batch_size=2",
        "datamodule.num_workers=0",
        "++datamodule.pin_memory=false",
        "model.audio_encoder.encoder._args_.0.n_layers=1",
        "model.optimizer.lr=0.0",
        "model.ma_callback.every_n_steps=999",
        "model.compile=false",
        "callbacks.model_checkpoint.every_n_epochs=1",
        "callbacks.model_checkpoint.every_n_train_steps=null",
        "++trainer.max_epochs=1",
        "++trainer.min_steps=null",
        "++trainer.max_steps=-1",
        "++trainer.limit_train_batches=1",
        "++trainer.limit_val_batches=1",
        "++trainer.num_sanity_val_steps=0",
        "trainer.val_check_interval=1",
        "++trainer.enable_model_summary=false",
        "training.val_audio_probe=false",
    ]

    result = subprocess.run(  # noqa: S603 — installed CLI with test-owned arguments
        command,
        cwd=tmp_path,
        env={**os.environ, "HYDRA_FULL_ERROR": "1"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    cli_checkpoint = torch.load(
        output_dir / "checkpoints" / "last.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert cli_checkpoint["global_step"] == 1
    for state_key in (
        "audio_encoder.projector.0.weight",
        "audio_ema.projector.0.weight",
    ):
        assert torch.equal(
            cli_checkpoint["state_dict"][state_key],
            source_checkpoint["state_dict"][state_key],
        )
