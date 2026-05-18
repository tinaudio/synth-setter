"""Pydantic schema for the top-level training configuration.

Documents the scalar fields the training entrypoint reads from
``configs/train.yaml`` and treats the Hydra-composed subtrees as opaque
dicts — those subtrees have their own composition groups under ``configs/``
and are validated by ``hydra.utils.instantiate`` at call time.

Intentionally documentation-first: tests assert the live composed DictConfig
validates against ``TrainConfig``, but the entrypoint continues to consume
the raw ``DictConfig``. Currently validated only by the test suite
(``tests/schemas/test_train_config.py``); the training entrypoint
(``cli/train.py``) does not call ``TrainConfig.model_validate`` at runtime.
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
    """Return the default ``tags`` list for a fresh ``TrainConfig``.

    :returns: A one-element list containing the placeholder ``"dev"`` tag.
    :rtype: list[str]
    """
    return ["dev"]


class TrainConfig(StrictAllowExtraModel):  # noqa: DOC601,DOC603
    """Top-level training configuration as composed from ``configs/train.yaml``.

    Only the typed scalar surface is represented here — the defaults below
    match the corresponding values in ``configs/train.yaml``. Hydra-managed
    subtrees (``data``, ``model``, ``trainer``, ``callbacks``, ``logger``,
    ``paths``, ``extras``, ``hydra``, ...) are NOT reconstructed by
    ``TrainConfig()``; they're accepted via ``extra="allow"`` when validating
    an externally-composed ``DictConfig`` and must come from real Hydra
    composition. Per-field descriptions live on the ``Field`` definitions
    below and render into the auto-generated docs via ``griffe-pydantic``.
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
