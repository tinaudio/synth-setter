"""Pydantic schema for the top-level ``configs/train.yaml``.

Documentation-first: validated by ``tests/schemas/test_train_config.py``
against the live composed DictConfig; ``cli/train.py`` continues to consume
the raw ``DictConfig`` rather than calling ``model_validate``.
"""

from __future__ import annotations

from pydantic import (
    Field,
    NonNegativeInt,
    StrictBool,
    StrictStr,
)

from synth_setter.schemas._types import NonBlankStr, StrictAllowExtraModel

__all__ = ["TrainConfig"]


def _default_tags() -> list[str]:
    """Return the placeholder ``["dev"]`` tag list.

    :return: Single-element list ``["dev"]``.
    """
    return ["dev"]


class TrainConfig(StrictAllowExtraModel):  # noqa: DOC601,DOC603
    """Top-level training config composed from ``configs/train.yaml``.

    Defaults below mirror ``configs/train.yaml``. Hydra-managed subtrees
    (``data``, ``model``, ``trainer``, ...) pass through via ``extra="allow"``
    — ``TrainConfig()`` does not reconstruct them on its own.
    """

    task_name: NonBlankStr = Field(
        default="train",
        description=(
            "Logical name for this training run. Used as the output-directory "
            "stem and to derive the wandb run id."
        ),
    )
    tags: list[StrictStr] = Field(
        default_factory=_default_tags,
        description=(
            "Free-form tags propagated to the logger (wandb / TensorBoard). "
            "Overwrite from the command line with ``tags='[first, second]'``."
        ),
    )
    train: StrictBool = Field(
        default=True,
        description="Run the fit loop. Set to ``False`` to skip training entirely.",
    )
    test: StrictBool = Field(
        default=True,
        description=(
            "Run the test loop on the best checkpoint after fit. Lightning picks "
            "the best checkpoint via the ``model_checkpoint`` callback."
        ),
    )
    ckpt_path: NonBlankStr | None = Field(
        default=None,
        description=(
            "Path to a Lightning checkpoint. If set, ``trainer.fit`` resumes from "
            "this checkpoint and ``trainer.test`` loads it as the test weights."
        ),
    )
    seed: NonNegativeInt | None = Field(
        default=None,
        description=(
            "Seed forwarded to ``lightning.seed_everything`` for PyTorch, NumPy, "
            "and Python's ``random``. ``None`` means non-deterministic."
        ),
    )
    optimized_metric: NonBlankStr | None = Field(
        default=None,
        description=(
            "Name of the callback metric returned to Hydra for hyperparameter "
            "sweeps. ``None`` means ``main()`` returns ``None`` and sweepers "
            "fall back to their default objective."
        ),
    )
    watch_gradients: StrictBool | None = Field(
        default=None,
        description=(
            "If truthy, attaches a gradient watcher to the logger. ``None`` and "
            "``False`` behave the same — the watcher is not attached."
        ),
    )
