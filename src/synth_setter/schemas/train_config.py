"""Pydantic schema for the top-level ``configs/train.yaml``.

Documentation-first: validated by ``tests/schemas/test_train_config.py``
against the live composed DictConfig; ``cli/train.py`` continues to consume
the raw ``DictConfig`` rather than calling ``model_validate``.
"""

from __future__ import annotations

from typing import Any, Self, cast

from omegaconf import DictConfig, OmegaConf
from pydantic import (
    Field,
    NonNegativeInt,
    StrictBool,
    StrictStr,
    model_validator,
)

from synth_setter.schemas._types import NonBlankStr, StrictAllowExtraModel
from synth_setter.schemas.datamodule_config import DataModuleConfig
from synth_setter.schemas.model_config import ModelConfig

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

    .. attribute :: model

        Typed model subtree when present in a composed Hydra config.

    .. attribute :: datamodule

        Typed datamodule subtree when present in a composed Hydra config.
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
    model: ModelConfig | None = None
    datamodule: DataModuleConfig | None = None

    @classmethod
    def from_hydra_cfg(cls, cfg: DictConfig) -> Self:
        """Validate typed fields after resolving their Hydra interpolations.

        Runtime-only interpolations in untyped extra fields remain unresolved.

        :param cfg: Composed Hydra configuration.
        :returns: Validated training configuration.
        """
        values = cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=False))
        for section_name in ("datamodule", "model"):
            section = cfg.get(section_name)
            section_values = values.get(section_name)
            if not isinstance(section, DictConfig) or not isinstance(section_values, dict):
                continue
            if "conditioning" in section:
                conditioning = section.conditioning
                section_values["conditioning"] = (
                    OmegaConf.to_container(conditioning, resolve=True, throw_on_missing=True)
                    if OmegaConf.is_config(conditioning)
                    else conditioning
                )
        return cls.model_validate(values)

    @model_validator(mode="after")
    def model_and_datamodule_conditioning_match(self) -> Self:
        """Require explicitly configured conditioning consumers to agree.

        :returns: Validated composed training config.
        :raises ValueError: Model and datamodule conditioning differ.
        """
        if self.model is None or self.datamodule is None:
            return self
        model_conditioning = self.model.conditioning
        datamodule_conditioning = self.datamodule.conditioning
        if (
            model_conditioning is not None
            and datamodule_conditioning is not None
            and model_conditioning != datamodule_conditioning
        ):
            raise ValueError("model and datamodule conditioning must match")
        return self
