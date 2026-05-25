"""Hydra entrypoint for evaluating a trained model on a datamodule's test split."""

from typing import Any

import hydra
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from synth_setter.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    log_wandb_provenance,
    register_resolvers,
    task_wrapper,
)
from synth_setter.workspace import operator_workspace

# Resolve workspace at import so ``${oc.env:PROJECT_ROOT}`` in
# ``configs/paths/default.yaml`` interpolates under any install layout.
operator_workspace()

register_resolvers()

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate the given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
        "callbacks": callbacks,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)
        log_wandb_provenance()

    mode = cfg.get("mode", "test")

    if mode == "test":
        log.info("Starting testing!")
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.ckpt_path,
            weights_only=False,
        )
    # Accept both spellings for backwards compatibility with older configs.
    elif mode == "val" or mode == "validate":
        log.info("Starting validating!")
        trainer.validate(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.ckpt_path,
            weights_only=False,
        )
    elif mode == "predict":
        predictions = trainer.predict(
            model=model,
            dataloaders=datamodule,
            ckpt_path=cfg.ckpt_path,
            return_predictions=False,
            weights_only=False,
        )

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="pkg://synth_setter.configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Run the evaluation entrypoint.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
