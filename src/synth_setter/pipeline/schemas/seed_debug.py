"""Typed row-level seed provenance stored in Lance debug documents."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ParameterSource = Literal["fixed", "mixed", "sampled"]


class SeedDebugDocument(BaseModel):
    """Describe the seed inputs and accepted attempt for one rendered row.

    .. attribute :: model_config

        Strict, frozen, extra-forbid Pydantic configuration.

    .. attribute :: seed

        Concrete sampler seed consumed by the row, or ``None`` when sampling was bypassed.

    .. attribute :: master_seed

        Dataset or split master seed.

    .. attribute :: sample_idx

        Stable logical row index within the seed stream.

    .. attribute :: attempt

        Accepted loudness-gate attempt for the row.

    .. attribute :: shard_id

        Logical shard number, or ``None`` for an ad hoc render.

    .. attribute :: parameter_source

        Whether parameters were sampled, fixed, or mixed.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    seed: int | None = Field(default=None, ge=0)
    master_seed: int
    sample_idx: int = Field(ge=0)
    attempt: int = Field(ge=0)
    shard_id: int | None = Field(default=None, ge=0)
    parameter_source: ParameterSource
