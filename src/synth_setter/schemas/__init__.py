"""Pydantic schemas for the training-time Hydra configs.

Source of truth for the mkdocs config-reference pages (rendered by
``mkdocstrings`` + ``griffe-pydantic``) and for
:func:`~synth_setter.schemas.validate.validate_composed_config`, which the
train/eval CLI entrypoints call at the config-load boundary to fail fast on a
malformed composition.

Distributed-pipeline schemas live in ``synth_setter.pipeline.schemas`` —
that's a different trust boundary (closed-shape ``extra="forbid"``) and is
intentionally separate.

The ``hydra`` composition group is not modeled here; Hydra owns that schema.
The ``datamodule`` group has no schema either — only ``_target_`` matters and
Hydra already enforces it.
"""

from synth_setter.schemas._types import NonBlankStr, StrictAllowExtraModel
from synth_setter.schemas.callbacks_config import CallbackInstance, CallbacksConfig
from synth_setter.schemas.extras_config import ExtrasConfig
from synth_setter.schemas.logger_config import LoggerConfig, LoggerInstance
from synth_setter.schemas.model_config import ModelConfig, OptimizerConfig, SchedulerConfig
from synth_setter.schemas.paths_config import PathsConfig
from synth_setter.schemas.train_config import TrainConfig
from synth_setter.schemas.trainer_config import TrainerConfig
from synth_setter.schemas.validate import validate_composed_config

__all__ = [
    "CallbackInstance",
    "CallbacksConfig",
    "ExtrasConfig",
    "LoggerConfig",
    "LoggerInstance",
    "ModelConfig",
    "NonBlankStr",
    "OptimizerConfig",
    "PathsConfig",
    "SchedulerConfig",
    "StrictAllowExtraModel",
    "TrainConfig",
    "TrainerConfig",
    "validate_composed_config",
]
