"""Pydantic schemas that document and validate the project's Hydra configs.

These models are the source of truth for the published mkdocs config-reference
pages (``mkdocstrings`` + ``griffe-pydantic`` renders their field tables) and
a sanity check that the training-time YAMLs stay in sync with what the
entrypoints expect.

Training-time schemas live in this package; distributed-pipeline schemas
(``DatasetSpec``, ``ImageConfig``, and friends) live in
``synth_setter.pipeline.schemas`` so the two trust boundaries can evolve
their strictness independently.

The schemas use ``strict=True`` to refuse type-coercion at the trust
boundary, and ``extra="allow"`` so subtrees that Hydra owns the shape of
pass through unchanged.

Each composition group under ``configs/`` is covered by one schema module
in this package, plus ``train_config`` for the top-level ``configs/train.yaml``.
The ``hydra`` composition group (``configs/hydra/default.yaml``) only sets
partial overrides on Hydra's own internal config and is intentionally not
modeled here — Hydra owns that schema.
"""

# _types is a private module; only the names re-exported here are public.
from synth_setter.schemas._types import NonBlankStr, StrictAllowExtraModel
from synth_setter.schemas.callbacks_config import CallbackInstance, CallbacksConfig
from synth_setter.schemas.data_config import DataConfig
from synth_setter.schemas.extras_config import ExtrasConfig
from synth_setter.schemas.logger_config import LoggerConfig, LoggerInstance
from synth_setter.schemas.model_config import ModelConfig, OptimizerConfig, SchedulerConfig
from synth_setter.schemas.paths_config import PathsConfig
from synth_setter.schemas.train_config import TrainConfig
from synth_setter.schemas.trainer_config import TrainerConfig

__all__ = [
    "CallbackInstance",
    "CallbacksConfig",
    "DataConfig",
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
]
