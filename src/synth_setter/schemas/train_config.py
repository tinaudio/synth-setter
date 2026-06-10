"""Pydantic schema for the top-level ``configs/train.yaml``.

``cli/train.py`` validates the composed config against this model at the
config-load boundary (via
:func:`~synth_setter.schemas.validate.validate_composed_config`) before
training starts, then consumes the raw ``DictConfig`` as before.
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


class TrainConfig(StrictAllowExtraModel):
    """Top-level training config composed from ``configs/train.yaml``.

    Defaults below mirror ``configs/train.yaml``. Hydra-managed subtrees
    (``datamodule``, ``model``, ``trainer``, ...) pass through via ``extra="allow"``
    — ``TrainConfig()`` does not reconstruct them on its own.

    .. attribute :: task_name

        Logical name for this training run.

    .. attribute :: tags

        Free-form tags propagated to the logger (wandb / TensorBoard).

    .. attribute :: train

        Run the fit loop.

    .. attribute :: test

        Run the test loop on the best checkpoint after fit.

    .. attribute :: ckpt_path

        Path to a Lightning checkpoint to resume from.

    .. attribute :: seed

        Seed forwarded to ``lightning.seed_everything``.

    .. attribute :: optimized_metric

        Name of the callback metric returned to Hydra for sweeps.

    .. attribute :: watch_gradients

        If truthy, attaches a gradient watcher to the logger.

    .. attribute :: consumed_dataset_config_id

        Dataset config_id whose ``data-`` artifact this run consumes for W&B lineage.

    .. attribute :: consumed_artifact_alias

        Alias appended to each consumed-artifact ref (default ``latest``).
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
    consumed_dataset_config_id: NonBlankStr | None = Field(
        default=None,
        description=(
            "Dataset config_id whose ``data-{id}`` W&B artifact this run consumes "
            "for lineage (storage-provenance-spec §5). ``None`` records no edge."
        ),
    )
    consumed_artifact_alias: NonBlankStr = Field(
        default="latest",
        description=(
            "Alias appended to each consumed-artifact ref, e.g. ``data-{id}:{alias}``. "
            "Defaults to ``latest``."
        ),
    )
