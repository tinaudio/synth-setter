"""SkyPilot launcher config schema.

``SkypilotLaunchConfig`` is the Pydantic-validated payload consumed by
``synth_setter.pipeline.skypilot_launch.dispatch_via_skypilot``. Captures
*launcher* knobs only; worker-side secrets (R2 creds, WORKER_GIT_REF) are
resolved separately via ``resolve_worker_env``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class SkypilotLaunchConfig(BaseModel):  # noqa: DOC601,DOC603
    """Validated SkyPilot launch parameters consumed by dispatch_via_skypilot."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    compute_template: str | None = None
    cmd: str | None = None
    env_file: str | None = None
    job_name: str | None = None
    num_workers: int = 1
    worker_image_tag: str = "dev-snapshot"
    tail: bool = False
    api_server: str | None = None
    local: bool = False

    @field_validator("num_workers")
    @classmethod
    def num_workers_must_be_positive(cls, v: int) -> int:  # noqa: DOC101,DOC103,DOC201,DOC203,DOC501,DOC503
        """Reject zero or negative worker counts."""
        if v < 1:
            raise ValueError(f"num_workers must be >= 1, got {v}")
        return v

    @field_validator("api_server")
    @classmethod
    def api_server_must_be_non_blank(cls, v: str | None) -> str | None:  # noqa: DOC101,DOC103,DOC201,DOC203,DOC501,DOC503
        """Reject blank/whitespace-only api_server values; strip surrounding whitespace."""
        if v is None:
            return v
        if not v.strip():
            raise ValueError("api_server must be a non-empty URL when set")
        return v.strip()
