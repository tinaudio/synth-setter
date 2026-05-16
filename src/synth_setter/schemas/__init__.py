"""Pydantic schemas that document and validate the project's Hydra configs.

These models are the source of truth for the published mkdocs config-reference
pages (``mkdocstrings`` + ``griffe-pydantic`` renders their field tables) and
a sanity check that the training-time YAMLs stay in sync with what the
entrypoints expect.

Parallel to ``synth_setter.pipeline.schemas`` (which owns ``DatasetSpec`` and
friends for the distributed data pipeline), this package owns the schemas
that describe training-time configuration consumed by ``cli/train.py`` —
kept separate so the two trust boundaries can evolve their strictness
independently.

The schemas use ``strict=True`` to refuse type-coercion at the trust
boundary, and ``extra="allow"`` so subtrees that Hydra owns the shape of
pass through unchanged.
"""

from synth_setter.schemas._types import NonBlankStr
from synth_setter.schemas.model_config import ModelConfig, OptimizerConfig, SchedulerConfig
from synth_setter.schemas.train_config import TrainConfig

__all__ = [
    "ModelConfig",
    "NonBlankStr",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainConfig",
]
