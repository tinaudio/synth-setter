"""SkyPilot launcher config schema.

``SkypilotLaunchConfig`` is the Pydantic-validated payload consumed by
``synth_setter.pipeline.skypilot_launch.dispatch_via_skypilot``. Captures
*launcher* knobs only; worker-side secrets (R2 creds, WORKER_GIT_REF) are
resolved separately via ``resolve_worker_env``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_SKYPILOT_API_SERVER_ENDPOINT: Final = "SKYPILOT_API_SERVER_ENDPOINT"
ENV_SKYPILOT_SERVICE_ACCOUNT_TOKEN: Final = "SKYPILOT_SERVICE_ACCOUNT_TOKEN"  # noqa: S105
SKYPILOT_CLIENT_AUTH_ENV_KEYS: Final[tuple[str, ...]] = (
    ENV_SKYPILOT_API_SERVER_ENDPOINT,
    ENV_SKYPILOT_SERVICE_ACCOUNT_TOKEN,
)

_ENV_IDENT_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip() or None


def _client_settings_kwargs_from_sources(env_file: Path | None) -> dict[str, str]:
    candidates: dict[str, str | None] = {}
    if env_file is not None and env_file.is_file():
        candidates.update(dotenv_values(env_file))

    kwargs: dict[str, str] = {}
    for env_key in SKYPILOT_CLIENT_AUTH_ENV_KEYS:
        field_name = env_key.removeprefix("SKYPILOT_").lower()
        for source in (candidates, os.environ):
            value = _clean(source.get(env_key))
            if value is not None:
                kwargs[field_name] = value
                break
    return kwargs


class SkypilotClientSettings(BaseSettings):
    """SkyPilot client authentication loaded from ``SKYPILOT_*`` env.

    .. attribute :: model_config

        Pydantic settings config sentinel.

    .. attribute :: api_server_endpoint

        Optional remote SkyPilot API server URL.

    .. attribute :: service_account_token

        Optional SkyPilot service-account token.
    """

    model_config = SettingsConfigDict(
        env_prefix="SKYPILOT_",
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )

    api_server_endpoint: str | None = None
    service_account_token: SecretStr | None = None

    @field_validator("api_server_endpoint")
    @classmethod
    def _endpoint_is_http_url(cls, value: str | None) -> str | None:
        """Require a remote endpoint to use HTTP(S) and name a host.

        :param value: Candidate SkyPilot API server endpoint.
        :returns: Stripped endpoint, if configured.
        :raises ValueError: The endpoint is not an HTTP(S) URL with a host.
        """
        if value is None:
            return None
        endpoint = value.strip()
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ValueError("must be an HTTP(S) URL with a host")
        return endpoint

    @field_validator("service_account_token")
    @classmethod
    def _token_has_skypilot_prefix(cls, value: SecretStr | None) -> SecretStr | None:
        """Require SkyPilot's service-account token format when configured.

        :param value: Candidate service-account token.
        :returns: Valid token, if configured.
        :raises ValueError: The token is blank or lacks the ``sky_`` prefix.
        """
        if value is None:
            return None
        token = value.get_secret_value().strip()
        if not token.startswith("sky_"):
            raise ValueError("must start with 'sky_'")
        return SecretStr(token)

    @model_validator(mode="after")
    def _token_requires_endpoint(self) -> SkypilotClientSettings:
        """Reject a service token with no remote server target.

        :returns: Validated client settings.
        :raises ValueError: A token is configured without an API endpoint.
        """
        if self.service_account_token is not None and self.api_server_endpoint is None:
            raise ValueError("service account token requires a SkyPilot API server endpoint")
        return self

    def as_env(self) -> dict[str, str]:
        """Project configured client authentication into SkyPilot env vars.

        :returns: Non-empty SkyPilot client environment entries.
        """
        env: dict[str, str] = {}
        if self.api_server_endpoint is not None:
            env[ENV_SKYPILOT_API_SERVER_ENDPOINT] = self.api_server_endpoint
        if self.service_account_token is not None:
            env[ENV_SKYPILOT_SERVICE_ACCOUNT_TOKEN] = self.service_account_token.get_secret_value()
        return env


def skypilot_client_settings_from_sources(
    env_file: Path | None = None, *, api_server_endpoint: str | None = None
) -> SkypilotClientSettings:
    """Load SkyPilot client auth from config, dotenv, then process environment.

    :param env_file: Optional dotenv file to inspect before process environment.
    :param api_server_endpoint: Optional launch-config endpoint override.
    :returns: Validated SkyPilot client settings.
    """
    kwargs = _client_settings_kwargs_from_sources(env_file)
    if api_server_endpoint is not None:
        kwargs["api_server_endpoint"] = api_server_endpoint
    return SkypilotClientSettings.model_validate(kwargs)


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
        ``resolve_worker_env``. Keys must match ``[A-Z_][A-Z0-9_]*``
        (uppercase-only env-var identifiers — POSIX-portable across the
        shells SkyPilot exports to) and may not collide with the launcher's
        resolved-env keys (use ``.env`` or process env for those); rank/world
        keys injected later still win.
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

    @field_validator("env_file")
    @classmethod
    def env_file_must_be_non_blank(cls, v: str | None) -> str | None:
        """Reject blank/whitespace-only env_file values; strip surrounding whitespace.

        :param v: Candidate ``env_file`` value pre-validation (``None`` permitted).
        :return: ``None`` when input is ``None``; else ``v`` with whitespace stripped.
        :raises ValueError: ``v`` is a non-``None`` string that is blank/whitespace-only.
        """
        if v is None:
            return v
        if not v.strip():
            raise ValueError("env_file must be a non-empty path when set")
        return v.strip()

    @field_validator("extra_envs")
    @classmethod
    def extra_envs_keys_must_be_env_identifiers(cls, v: dict[str, str]) -> dict[str, str]:
        """Reject keys that aren't uppercase env-var identifiers.

        The accepted grammar (``[A-Z_][A-Z0-9_]*``) is intentionally narrower
        than full POSIX — uppercase-only matches the convention every worker
        env this launcher exports has followed historically, and keeps caller-
        supplied vars visually distinct from shell locals on the worker side.

        :param v: Candidate ``extra_envs`` mapping pre-validation.
        :return: ``v`` unchanged when every key matches ``[A-Z_][A-Z0-9_]*``.
        :raises ValueError: one or more keys violate the env-identifier grammar.
        """
        bad = [k for k in v if not _ENV_IDENT_RE.match(k)]
        if bad:
            raise ValueError(
                "extra_envs keys must match the uppercase env-var grammar "
                f"[A-Z_][A-Z0-9_]*; got invalid: {bad}"
            )
        return v
