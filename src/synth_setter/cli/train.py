"""Hydra entrypoint for training and (optionally) test-set evaluation of a Lightning model."""

from pathlib import Path
from typing import Any

import hydra
import lightning as L
import torch
import wandb
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

from synth_setter.pipeline import r2_io
from synth_setter.run_id import make_wandb_run_id
from synth_setter.schemas.validate import validate_composed_config
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
    resolve_git_sha,
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


def _derive_checkpoint_uri(cfg: DictConfig) -> str:
    """Return the ``r2://`` URI the best checkpoint uploads to.

    Honors ``training.upload_checkpoints_uri`` verbatim when set; otherwise
    derives ``r2://{r2.bucket}/checkpoints/{config_id}/model.ckpt``, where
    ``config_id`` is :func:`~synth_setter.utils.resolve_run_config_id`. The fixed
    ``model.ckpt`` basename lets the ``${wandb:...}`` resolver select the
    checkpoint unambiguously.

    :param cfg: Hydra-composed train cfg; reads ``r2.bucket`` and the optional
        ``training.upload_checkpoints_uri`` override.
    :returns: The canonical ``r2://`` checkpoint URI for this run.
    """
    override = OmegaConf.select(cfg, "training.upload_checkpoints_uri")
    if override:
        return str(override)
    return f"r2://{cfg.r2.bucket}/checkpoints/{resolve_run_config_id(cfg)}/model.ckpt"


def _upload_best_checkpoint(cfg: DictConfig, best_model_path: str) -> str | None:
    """Upload the best checkpoint to its derived ``r2://`` URI; return that URI or ``None``.

    Best-effort and degrades to ``None`` (a lineage-only model artifact) when no
    checkpoint was written (``best_model_path`` empty — e.g. ``fast_dev_run``),
    when R2 is unavailable (local CPU / CI — missing creds or failed auth), or
    when the upload itself fails — so a completed run is never aborted by
    checkpoint persistence. :func:`r2_io.ensure_r2_env_loaded` populates the
    structural ``RCLONE_CONFIG_R2_*`` defaults (so a runtime wiring only the
    secret keys still resolves the ``r2:`` remote) and auth-pings before the
    upload; ``upload_to_uri`` renames the source to the URI's ``model.ckpt`` basename.

    :param cfg: Train cfg forwarded to :func:`_derive_checkpoint_uri`.
    :param best_model_path: ``trainer.checkpoint_callback.best_model_path``;
        empty when no checkpoint exists.
    :returns: The ``r2://`` URI the checkpoint landed at, or ``None`` when no
        upload happened.
    """
    if not best_model_path:
        log.warning("No best checkpoint to upload; logging lineage-only model artifact.")
        return None
    try:
        r2_io.ensure_r2_env_loaded()
    except Exception as exc:  # noqa: BLE001 — R2 unavailable must not abort a completed run
        log.info(f"R2 unavailable; logging lineage-only model artifact (no upload): {exc}")
        return None
    uri = _derive_checkpoint_uri(cfg)
    try:
        r2_io.upload_to_uri(Path(best_model_path), uri)
    except Exception as exc:  # noqa: BLE001 — upload failure must not abort a completed run
        log.warning(f"Checkpoint upload to {uri} failed; logging lineage-only artifact: {exc}")
        return None
    return uri


def build_model_artifact(cfg: DictConfig, ckpt_uri: str | None = None) -> wandb.Artifact:
    """Build the canonical ``model`` W&B artifact for a training run.

    Names the artifact ``model-{config_id}`` (type ``model``) per
    storage-provenance-spec.md §4, where ``config_id`` is
    :func:`~synth_setter.utils.resolve_run_config_id`, and records ``git_sha``
    in ``artifact.metadata`` per §6. When ``ckpt_uri`` is given (the upload
    succeeded), that ``r2://`` location is referenced as an ``s3://`` URI
    (``checksum=False`` — R2's custom endpoint is not reachable by W&B's
    reference handler, so the URI records lineage, not a content hash). With no
    ``ckpt_uri`` the artifact is lineage-only, so a reference never points at a
    checkpoint that was not uploaded.

    :param cfg: Hydra-composed train cfg; ``task_name``/experiment determine the name.
    :param ckpt_uri: The ``r2://`` URI the checkpoint was uploaded to, or ``None``
        for a lineage-only artifact.
    :returns: An unlogged ``wandb.Artifact`` ready for ``log_artifact``.
    """
    artifact = wandb.Artifact(
        name=f"model-{resolve_run_config_id(cfg)}",
        type="model",
        metadata={"git_sha": resolve_git_sha()},
    )
    if ckpt_uri:
        artifact.add_reference(r2_io.to_s3_uri(ckpt_uri), checksum=False)
    return artifact


def _has_wandb_logger(loggers: list[Logger]) -> bool:
    """Return whether any logger is a ``WandbLogger``.

    Gates the train-end checkpoint upload + artifact logging: with no
    ``WandbLogger`` there is no run to reference the checkpoint, so the upload is
    skipped rather than pushing bytes to R2 no artifact points at.

    :param loggers: Instantiated Lightning loggers.
    :returns: ``True`` if at least one entry is a ``WandbLogger``.
    """
    return any(isinstance(lg, WandbLogger) for lg in loggers)


def _log_model_artifact(loggers: list[Logger], cfg: DictConfig, ckpt_uri: str | None) -> None:
    """Log the canonical ``model`` artifact to each ``WandbLogger`` in ``loggers``.

    A wandb failure warns and is swallowed so artifact logging never aborts a
    completed training run. Non-``WandbLogger`` entries (and an empty list) are a
    no-op — the path every wandb-free caller takes.

    :param loggers: Lightning loggers; only ``WandbLogger`` entries log.
    :param cfg: Train cfg forwarded to :func:`build_model_artifact`.
    :param ckpt_uri: The uploaded checkpoint URI to reference, or ``None`` for
        a lineage-only artifact.
    """
    for lg in loggers:
        if not isinstance(lg, WandbLogger):
            continue
        try:
            lg.experiment.log_artifact(build_model_artifact(cfg, ckpt_uri))
        except Exception as exc:  # noqa: BLE001 — wandb artifact failure must not abort training
            log.warning(f"_log_model_artifact failed on {type(lg).__name__}: {exc}")


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

    # After train/test so the best checkpoint exists; gated on a WandbLogger (no run ⇒
    # nothing references the upload) and global-zero (so DDP ranks don't race duplicate
    # artifact versions). Degrades to lineage-only when R2 is unreachable or no ckpt exists.
    if trainer.is_global_zero and _has_wandb_logger(logger):
        best_model_path = getattr(trainer.checkpoint_callback, "best_model_path", "") or ""
        ckpt_uri = _upload_best_checkpoint(cfg, best_model_path)
        _log_model_artifact(logger, cfg, ckpt_uri)

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="pkg://synth_setter.configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """Run the training entrypoint.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # Fail fast on a malformed composition (blank task_name, negative seed/lr,
    # non-positive trainer cadence, bad float32 precision) before any work.
    validate_composed_config(cfg, include_train_config=True)

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
