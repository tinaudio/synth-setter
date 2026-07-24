"""Typed row-level seed provenance stored in Lance debug documents."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ParameterSource = Literal["fixed", "mixed", "sampled"]


class SeedDebugDocument(BaseModel):
    """Describe the seed inputs and accepted attempt for one rendered row.

    .. attribute :: model_config

        Strict, frozen, extra-forbid Pydantic configuration.

    .. attribute :: seed

        Concrete seed used to render the row.

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

    .. attribute :: parameter_seed

        Concrete seed for a reused shard-cadence patch.

    .. attribute :: parameter_sample_idx

        Seed-stream row that supplied a reused patch.

    .. attribute :: parameter_attempt

        Accepted attempt that supplied a reused patch.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    seed: int = Field(ge=0)
    master_seed: int
    sample_idx: int = Field(ge=0)
    attempt: int = Field(ge=0)
    shard_id: int | None = Field(default=None, ge=0)
    parameter_source: ParameterSource
    parameter_seed: int | None = Field(default=None, ge=0)
    parameter_sample_idx: int | None = Field(default=None, ge=0)
    parameter_attempt: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _parameter_provenance_is_complete(self) -> Self:
        """Require reused-parameter seed fields to be present as one unit.

        :returns: The validated debug document.
        :raises ValueError: Any reused-parameter seed field is present without the others.
        """
        parameter_provenance = (
            self.parameter_seed,
            self.parameter_sample_idx,
            self.parameter_attempt,
        )
        if any(value is not None for value in parameter_provenance) and not all(
            value is not None for value in parameter_provenance
        ):
            raise ValueError(
                "parameter_seed, parameter_sample_idx, and parameter_attempt "
                "must be provided together"
            )
        return self
