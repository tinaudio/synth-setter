"""SkyPilot launcher config schema.

``SkypilotLaunchConfig`` is the Pydantic-validated payload consumed by
``synth_setter.pipeline.skypilot_launch.dispatch_via_skypilot``. Captures
*launcher* knobs only; worker-side secrets (R2 creds, WORKER_GIT_REF) are
resolved separately via ``resolve_worker_env``.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

_ENV_IDENT_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


class SkypilotLaunchConfig(BaseModel):
    """Validated SkyPilot launch parameters consumed by dispatch_via_skypilot.

    .. attribute :: model_config

        Pydantic model config sentinel — see ``ConfigDict(...)`` below for active settings.

    .. attribute :: compute_template

        Path to the SkyPilot compute-template YAML.

    .. attribute :: cmd

        Override command passed to the worker entrypoint.

    .. attribute :: env_file

        Path to an ``.env`` file forwarded to workers.

    .. attribute :: job_name

        SkyPilot job name (shown in ``sky status``).

    .. attribute :: num_workers

        Number of worker replicas to launch.

    .. attribute :: worker_image_tag

        Docker image tag pulled by each worker.

    .. attribute :: tail

        Whether to tail logs after launch.

    .. attribute :: api_server

        SkyPilot API server URL override.

    .. attribute :: local

        Run the job on the local SkyPilot context instead of remote.

    .. attribute :: extra_envs

        Caller-supplied env vars merged into every rank's worker env after
        ``resolve_worker_env``. Keys must match the POSIX env-var identifier
        grammar; rank/world keys injected later still win.
    """

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
    extra_envs: dict[str, str] = Field(default_factory=dict)

    @field_validator("num_workers")
    @classmethod
    def num_workers_must_be_positive(cls, v: int) -> int:
        """Reject zero or negative worker counts.

        :param v: Candidate ``num_workers`` value pre-validation.
        :return: ``v`` unchanged when ``>= 1``.
        :raises ValueError: ``v`` is less than 1.
        """
        if v < 1:
            raise ValueError(f"num_workers must be >= 1, got {v}")
        return v

    @field_validator("api_server")
    @classmethod
    def api_server_must_be_non_blank(cls, v: str | None) -> str | None:
        """Reject blank/whitespace-only api_server values; strip surrounding whitespace.

        :param v: Candidate ``api_server`` value pre-validation (``None`` permitted).
        :return: ``None`` when input is ``None``; else ``v`` with whitespace stripped.
        :raises ValueError: ``v`` is a non-``None`` string that is blank/whitespace-only.
        """
        if v is None:
            return v
        if not v.strip():
            raise ValueError("api_server must be a non-empty URL when set")
        return v.strip()

    @field_validator("extra_envs")
    @classmethod
    def extra_envs_keys_must_be_env_identifiers(cls, v: dict[str, str]) -> dict[str, str]:
        """Reject keys that aren't POSIX env-var identifiers.

        :param v: Candidate ``extra_envs`` mapping pre-validation.
        :return: ``v`` unchanged when every key matches ``[A-Z_][A-Z0-9_]*``.
        :raises ValueError: one or more keys violate the env-identifier grammar.
        """
        bad = [k for k in v if not _ENV_IDENT_RE.match(k)]
        if bad:
            raise ValueError(
                f"extra_envs keys must match POSIX env-var identifiers; got invalid: {bad}"
            )
        return v
