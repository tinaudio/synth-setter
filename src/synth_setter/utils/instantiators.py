"""Hydra helpers that turn callback and logger config groups into instantiated objects."""

from importlib.util import find_spec

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from synth_setter.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiate callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :returns: A list of instantiated callbacks.
    :rtype: list[Callback]
    :raises TypeError: If ``callbacks_cfg`` is not a :class:`DictConfig`.
    """
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    """Instantiate loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :returns: A list of instantiated loggers.
    :rtype: list[Logger]
    :raises TypeError: If ``logger_cfg`` is not a :class:`DictConfig`.
    """
    logger: list[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def close_loggers(loggers: list[Logger], status: str) -> None:
    """Finalize each logger and flush any live wandb run.

    ``WandbLogger.finalize`` records status but does not close the run;
    ``wandb.finish()`` is what flushes the offline ``.wandb`` binary. The run
    is closed only when a ``WandbLogger`` is in ``loggers`` — i.e. this process
    opened it — so a stale ``wandb.run`` started elsewhere is left untouched.

    :param loggers: Lightning loggers; ``finalize`` is invoked on each.
    :param status: ``"success"`` or ``"failed"``; forwarded verbatim to each
        logger's ``finalize`` contract.
    """
    for lg in loggers:
        try:
            lg.finalize(status)
        except Exception as exc:  # noqa: BLE001 — finalize errors must not mask the original raise
            log.warning(f"logger finalize failed on {type(lg).__name__}: {exc}")
    if not find_spec("wandb"):
        return
    from lightning.pytorch.loggers.wandb import WandbLogger

    import wandb

    if any(isinstance(lg, WandbLogger) for lg in loggers) and wandb.run is not None:
        try:
            wandb.finish()
        except Exception as exc:  # noqa: BLE001 — finish errors must not mask the original raise
            log.warning(f"wandb.finish() failed: {exc}")
