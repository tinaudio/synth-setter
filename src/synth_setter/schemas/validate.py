"""Validate a Hydra-composed config against the training-config schemas.

Called at the CLI config-load boundary (``cli/train.py`` / ``cli/eval.py``)
right after ``@hydra.main`` composes ``cfg`` — so a bad override (blank
``task_name``, negative ``seed``/``lr``, non-positive trainer cadence, an
out-of-enum ``float32_matmul_precision``) fails loudly up front instead of
crashing cryptically deep inside ``hydra.utils.instantiate`` or torch.
"""

from __future__ import annotations

from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from synth_setter.schemas.callbacks_config import CallbacksConfig
from synth_setter.schemas.extras_config import ExtrasConfig
from synth_setter.schemas.logger_config import LoggerConfig
from synth_setter.schemas.model_config import ModelConfig
from synth_setter.schemas.paths_config import PathsConfig
from synth_setter.schemas.train_config import TrainConfig
from synth_setter.schemas.trainer_config import TrainerConfig

__all__ = ["validate_composed_config"]


def validate_composed_config(
    cfg: DictConfig | dict[str, Any], *, include_train_config: bool
) -> None:
    """Validate the composed ``cfg`` against the Pydantic training-config schemas.

    Converts a ``DictConfig`` with ``resolve=False`` so interpolations stay
    opaque strings (the schemas accept them) and ``model_validate``s each
    subtree the entrypoint composes, propagating ``pydantic.ValidationError``
    when any subtree is malformed. ``callbacks`` and ``logger`` compose to
    ``None`` for the ``none``/``null`` variants, so they are validated only
    when truthy.

    :param cfg: Hydra-composed config — a ``DictConfig`` (the CLI boundary) or
        an already-converted plain dict.
    :param include_train_config: Validate the top-level :class:`TrainConfig`
        (train-only fields). ``False`` for ``eval.yaml``, which composes the
        shared subtrees but none of the training-only top-level fields.
    """
    container: dict[str, Any] = (
        cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=False))
        if isinstance(cfg, DictConfig)
        else cfg
    )

    if include_train_config:
        TrainConfig.model_validate(container)

    PathsConfig.model_validate(container["paths"])
    TrainerConfig.model_validate(container["trainer"])
    ModelConfig.model_validate(container["model"])
    ExtrasConfig.model_validate(container["extras"])

    callbacks = container.get("callbacks")
    if callbacks:
        CallbacksConfig.model_validate(callbacks)

    logger = container.get("logger")
    if logger:
        LoggerConfig.model_validate(logger)
