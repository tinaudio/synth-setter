"""Hydra entrypoint for training and (optionally) test-set evaluation of a Lightning model."""

from typing import Any

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from synth_setter.run_id import make_wandb_run_id
from synth_setter.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    log_wandb_provenance,
    pin_wandb_run_id,
    register_resolvers,
    resolve_run_config_id,
    task_wrapper,
    use_input_artifacts,
    watch_gradients,
)
from synth_setter.workspace import operator_workspace

# Resolve workspace at import so ``${oc.env:PROJECT_ROOT}`` in
# ``configs/paths/default.yaml`` interpolates under any install layout.
operator_workspace()

register_resolvers()

log = RankedLogger(__name__, rank_zero_only=True)


def _consumed_artifact_refs(cfg: DictConfig) -> list[tuple[str, str]]:
    """Build the consumed-artifact lineage edges for a training run (spec §5).

    Training consumes the dataset it trains on. The edge is opt-in:
    ``consumed_dataset_config_id`` is null by default and yields no edges, so a
    run without the field set records no lineage and never calls
    ``use_artifact``.

    :param cfg: Hydra-composed cfg; reads ``consumed_dataset_config_id`` and
        ``consumed_artifact_alias`` (default ``latest``).
    :returns: ``[("data-{id}", alias)]`` when the dataset id is set, else ``[]``.
    """
    dataset_id = cfg.get("consumed_dataset_config_id")
    if not dataset_id:
        return []
    alias = cfg.get("consumed_artifact_alias") or "latest"
    return [(f"data-{dataset_id}", alias)]


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train the model and optionally evaluate on a testset using best-checkpoint weights.

    Wrapped in the optional ``@task_wrapper`` decorator, which controls behaviour on
    failure — useful for multiruns, saving info about crashes, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    pin_wandb_run_id(cfg, make_wandb_run_id(resolve_run_config_id(cfg)), "training")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)
        log_wandb_provenance()

    if cfg.get("watch_gradients"):
        log.info("Watching gradients!")
        watch_gradients(model, logger)

    # Record the dataset lineage edge before any consuming work so the run links to
    # its input artifact in the W&B DAG (storage-provenance-spec §5). A test-only run
    # (train: False, test: True) consumes the dataset too, so gate on either.
    if cfg.get("train") or cfg.get("test"):
        use_input_artifacts(logger, _consumed_artifact_refs(cfg))

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
            weights_only=False,
        )

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            weights_only=False,
        )
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="pkg://synth_setter.configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """Run the training entrypoint.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
