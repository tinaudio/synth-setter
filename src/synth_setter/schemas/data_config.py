"""Pydantic schema for per-datamodule Hydra configs under ``configs/data/``.

Every YAML under ``configs/data/`` selects one ``LightningDataModule`` via
``_target_`` and supplies the constructor kwargs that module needs. The shape
varies wildly across modules (``SurgeDataModule`` is path-driven,
``KSinDataModule`` is purely synthetic with seed tuples), so the typed
surface is intentionally narrow — only the keys ``cli/train.py`` itself
reads — and the rest is accepted via ``extra="allow"`` so a new
datamodule ships without a schema edit.
"""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from synth_setter.schemas._types import NonBlankStr

__all__ = ["DataConfig"]


class DataConfig(BaseModel):  # noqa: DOC601,DOC603
    """Per-datamodule Hydra config (one of the YAMLs under ``configs/data/``).

    Only ``_target_`` is typed at this layer — that's the single key
    ``cli/train.py`` reads off ``cfg.data`` before delegating to
    ``hydra.utils.instantiate(cfg.data)``. Datamodule constructor kwargs
    (``batch_size``, ``num_workers``, ``train_val_test_sizes``,
    ``dataset_root``, ...) vary across modules and pass through via
    ``extra="allow"``; their constraints live on the datamodule's
    ``__init__`` signature rather than being re-stated here.
    """

    model_config = ConfigDict(strict=True, extra="allow", populate_by_name=True)

    target_: NonBlankStr = Field(
        alias="_target_",
        description=(
            "Fully-qualified ``LightningDataModule`` class path. Resolved by "
            "``hydra.utils.instantiate(cfg.data)`` in ``cli/train.py``."
        ),
    )
