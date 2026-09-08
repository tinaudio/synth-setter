"""Train from a trusted self-describing checkpoint with explicit state semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from synth_setter.cli.train import train
from synth_setter.models.checkpoint_bundle import (
    assert_model_config_compatible,
    load_model_config,
)
from synth_setter.utils import extras, get_metric_value

CheckpointMode = Literal["full-resume", "weights-only"]
_DISABLED_RESUME_VALUES = (None, False, "off")


def train_from_checkpoint(
    cfg: DictConfig,
    checkpoint_path: Path,
    *,
    mode: CheckpointMode,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run training with explicit full-resume or weights-only/new-run semantics.

    The checkpoint pickle and bundled Hydra targets are executable input. Use
    this API only with trusted artifacts. The bundled model config is the model
    construction authority; the current config continues to supply data,
    trainer, callbacks, and logging settings.

    :param cfg: Current training/runtime configuration.
    :param checkpoint_path: Trusted self-describing local checkpoint.
    :param mode: ``full-resume`` restores all Lightning state; ``weights-only``
        starts a new run with fresh optimizer, scheduler, epoch, and step state.
    :returns: Metrics and instantiated objects returned by :func:`train`.
    :raises ValueError: If mode, model config, or checkpoint selections conflict.
    """
    if mode not in ("full-resume", "weights-only"):
        raise ValueError(f"checkpoint mode must be full-resume or weights-only; got {mode!r}")
    if cfg.get("ckpt_path") is not None:
        raise ValueError("checkpoint_path argument and cfg.ckpt_path are mutually exclusive")
    if OmegaConf.select(cfg, "training.weights_only_checkpoint") is not None:
        raise ValueError(
            "checkpoint_path argument and training.weights_only_checkpoint are mutually exclusive"
        )
    if OmegaConf.select(cfg, "training.resume") not in _DISABLED_RESUME_VALUES:
        raise ValueError("checkpoint_path argument and training.resume are mutually exclusive")

    bundled_model = load_model_config(checkpoint_path)
    assert_model_config_compatible(cfg.model, bundled_model)
    with open_dict(cfg):
        cfg.model = bundled_model
        if mode == "full-resume":
            cfg.ckpt_path = str(checkpoint_path)
        else:
            cfg.training.weights_only_checkpoint = str(checkpoint_path)
    return train(cfg)


@hydra.main(
    version_base="1.3",
    config_path="pkg://synth_setter.configs",
    config_name="train_from_checkpoint.yaml",
)
def main(cfg: DictConfig) -> float | None:
    """Run checkpoint-based training from Hydra configuration.

    :param cfg: Configuration carrying ``checkpoint_path`` and ``checkpoint_mode``.
    :returns: Optional optimized metric value.
    """
    extras(cfg)
    metrics, _ = train_from_checkpoint(
        cfg,
        Path(cfg.checkpoint_path),
        mode=cfg.checkpoint_mode,
    )
    return get_metric_value(metrics, cfg.get("optimized_metric"))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
