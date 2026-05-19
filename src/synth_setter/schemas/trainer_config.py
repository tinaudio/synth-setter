"""Pydantic schema for the YAMLs under ``configs/trainer/``.

Variants compose onto ``default.yaml``. The typed surface covers the keys
shipped variants set; other ``Trainer`` kwargs pass through via
``extra="allow"``.
"""

from __future__ import annotations

from pydantic import (
    Field,
    PositiveFloat,
    PositiveInt,
    StrictBool,
)

from synth_setter.schemas._types import NonBlankStr, StrictAllowExtraModel

__all__ = ["TrainerConfig"]


class TrainerConfig(StrictAllowExtraModel):
    """One of the YAMLs under ``configs/trainer/``; other kwargs ride ``extra="allow"``.

    .. attribute :: target_

        Fully-qualified ``lightning.pytorch.trainer.Trainer`` class path.

    .. attribute :: default_root_dir

        Where Lightning writes logs and checkpoints when no logger overrides it.

    .. attribute :: accelerator

        Hardware backend (``cpu``, ``gpu``, ``mps``, ``tpu``, ``auto``).

    .. attribute :: devices

        Number of devices on the chosen accelerator.

    .. attribute :: log_every_n_steps

        Logging cadence (steps) for training metrics.

    .. attribute :: val_check_interval

        Cadence of validation runs.

    .. attribute :: gradient_clip_val

        Maximum gradient L2 norm before clipping.

    .. attribute :: check_val_every_n_epoch

        Run validation every N epochs (``None`` → step-based).

    .. attribute :: deterministic

        Force deterministic ops where Lightning supports it.
    """

    target_: NonBlankStr = Field(
        alias="_target_",
        description=(
            "Fully-qualified ``lightning.pytorch.trainer.Trainer`` class path. "
            "Resolved by ``hydra.utils.instantiate(cfg.trainer, ...)`` in "
            "``cli/train.py``."
        ),
    )
    default_root_dir: NonBlankStr = Field(
        description=(
            "Where Lightning writes logs and checkpoints when no logger overrides "
            "it. Defaults to ``${paths.output_dir}``, which is Hydra's per-run "
            "output dir."
        ),
    )
    accelerator: NonBlankStr = Field(
        description=(
            "Hardware backend: ``cpu``, ``gpu``, ``mps``, ``tpu``, or ``auto``. "
            "Each shipped variant under ``configs/trainer/`` pins one explicitly."
        ),
    )
    devices: PositiveInt = Field(
        description=(
            "Number of devices on the chosen accelerator. ``1`` for single-GPU "
            "or CPU; ``4`` for the DDP variant; etc. ``Trainer`` also accepts "
            "strings / lists but the shipped configs use plain ints."
        ),
    )
    log_every_n_steps: PositiveInt = Field(
        description=(
            "Logging cadence (steps) for training metrics; passed straight to ``Trainer``."
        ),
    )
    val_check_interval: PositiveInt = Field(
        description="Cadence of validation runs; passed straight to ``Trainer``.",
    )
    gradient_clip_val: PositiveFloat = Field(
        description="Maximum gradient L2 norm before clipping; passed straight to ``Trainer``.",
    )
    check_val_every_n_epoch: PositiveInt | None = Field(
        default=None,
        description=(
            "Run validation every N epochs. ``None`` (the shipped default) means "
            "Lightning falls back to ``val_check_interval`` for step-based "
            "validation cadence."
        ),
    )
    deterministic: StrictBool = Field(
        default=False,
        description=(
            "Force deterministic ops where Lightning supports it. ``False`` "
            "across shipped variants (deterministic ops are slower and not all "
            "MPS ops support them); set ``True`` for reproducibility runs."
        ),
    )
