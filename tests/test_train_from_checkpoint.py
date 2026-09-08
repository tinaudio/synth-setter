"""Tests for explicit checkpoint-based training semantics."""

import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from synth_setter.cli.train import train
from synth_setter.cli.train_from_checkpoint import train_from_checkpoint


def _continuation_cfg(source: DictConfig, output_dir: Path, max_epochs: int) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(source, resolve=False))
    with open_dict(cfg):
        cfg.paths.output_dir = str(output_dir)
        cfg.paths.log_dir = str(output_dir)
        cfg.trainer.max_epochs = max_epochs
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

    full_cfg = _continuation_cfg(cfg_slap_train_lance, tmp_path / "full", max_epochs=2)
    _, full_objects = train_from_checkpoint(full_cfg, checkpoint, mode="full-resume")

    weights_cfg = _continuation_cfg(cfg_slap_train_lance, tmp_path / "weights", max_epochs=1)
    _, weights_objects = train_from_checkpoint(weights_cfg, checkpoint, mode="weights-only")

    assert full_objects["trainer"].global_step == 2
    assert weights_objects["trainer"].global_step == 1


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
    HydraConfig().set_config(cfg_slap_train_lance)
    train(cfg_slap_train_lance)
    checkpoint = Path(cfg_slap_train_lance.paths.output_dir) / "checkpoints" / "last.ckpt"
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
