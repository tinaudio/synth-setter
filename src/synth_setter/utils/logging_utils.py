"""Hyperparameter logging helpers, run-id conventions, and wandb-config provenance writer."""

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import PurePosixPath
from typing import Any

from hydra.core.hydra_config import HydraConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from synth_setter.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def resolve_run_config_id(cfg: DictConfig) -> str:
    """Resolve the run config_id from the chosen Hydra experiment, else ``task_name``.

    The experiment choice (e.g. ``surge/flow_simple``) is the train/eval analog of
    the dataset config stem; its basename becomes the config_id. Falls back to
    ``cfg.task_name`` when no experiment is selected or there is no Hydra context.

    :param cfg: Hydra-composed cfg carrying ``task_name``.
    :returns: The experiment basename, or ``cfg.task_name`` as a fallback.
    """
    try:
        experiment = HydraConfig.get().runtime.choices.get("experiment")
    except ValueError:
        experiment = None
    if experiment in (None, "null"):
        return cfg.task_name
    return PurePosixPath(experiment).name


def pin_wandb_run_id(cfg: DictConfig, run_id: str, job_type: str) -> None:
    """Pin the W&B run id and ``job_type`` onto ``cfg`` before logger instantiation.

    No-op when the cfg has no ``logger.wandb`` group (e.g. ``logger=tensorboard``
    or ``logger=null``), so ``OmegaConf.update`` never raises on the missing key.

    :param cfg: Hydra-composed cfg; ``logger.wandb.{id,job_type}`` are updated in place.
    :param run_id: The W&B run id to pin (see :func:`synth_setter.run_id.make_wandb_run_id`).
    :param job_type: W&B ``job_type`` (``training`` / ``evaluation`` / ``data-generation``).
    """
    if OmegaConf.select(cfg, "logger.wandb") is None:
        return
    OmegaConf.update(cfg, "logger.wandb.id", run_id)
    OmegaConf.update(cfg, "logger.wandb.job_type", job_type)


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    """Control which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def resolve_git_sha() -> str:
    """Return the current ``HEAD`` commit SHA, or ``"unknown"`` outside a git tree.

    Shared by :func:`log_wandb_provenance` (writes ``github_sha`` to
    ``wandb.config``) and the train CLI's model-artifact metadata so both record
    the same provenance value, per storage-provenance-spec.md §6.

    :returns: The 40-char ``HEAD`` SHA, or ``"unknown"`` when git is unavailable.
    """
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],  # noqa: S603, S607
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@rank_zero_only
def log_wandb_provenance() -> None:
    """Log provenance metadata to wandb.config.

    Records github_sha, image_tag, and command per storage-provenance-spec.md. Must be called after
    WandbLogger instantiation (which calls wandb.init). No-op if wandb is not installed or no run
    is active.
    """
    if not find_spec("wandb"):
        return
    import wandb

    if not wandb.run:
        return

    wandb.config.update(
        {
            "github_sha": resolve_git_sha(),
            "image_tag": os.environ.get("IMAGE_TAG", "unknown"),
            "command": " ".join(sys.argv),
        },
        allow_val_change=True,
    )
