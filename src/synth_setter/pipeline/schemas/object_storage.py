"""Provider-neutral object-storage settings and backend projections."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "ENV_STORAGE_ACCESS_KEY_ID",
    "ENV_STORAGE_DEFAULT_BUCKET",
    "ENV_STORAGE_ENDPOINT_URL",
    "ENV_STORAGE_PROVIDER",
    "ENV_STORAGE_REGION",
    "ENV_STORAGE_RCLONE_REMOTE",
    "ENV_STORAGE_RCLONE_TYPE",
    "ENV_STORAGE_SECRET_ACCESS_KEY",
    "ObjectStoreProvider",
    "ObjectLocation",
    "ObjectStorage",
    "RCLONE_ENV_KEYS",
    "RCLONE_REQUIRED_ENV_KEYS",
    "RCLONE_STRUCTURAL_DEFAULTS",
    "STORAGE_REQUIRED_ENV_KEYS",
    "StorageConfig",
    "StorageSettings",
    "storage_settings_from_sources",
]

ENV_STORAGE_ACCESS_KEY_ID: Final = "SYNTH_SETTER_STORAGE_ACCESS_KEY_ID"
ENV_STORAGE_SECRET_ACCESS_KEY: Final = "SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY"  # noqa: S105
ENV_STORAGE_ENDPOINT_URL: Final = "SYNTH_SETTER_STORAGE_ENDPOINT_URL"
ENV_STORAGE_REGION: Final = "SYNTH_SETTER_STORAGE_REGION"
ENV_STORAGE_PROVIDER: Final = "SYNTH_SETTER_STORAGE_PROVIDER"
ENV_STORAGE_DEFAULT_BUCKET: Final = "SYNTH_SETTER_STORAGE_DEFAULT_BUCKET"
ENV_STORAGE_RCLONE_REMOTE: Final = "SYNTH_SETTER_STORAGE_RCLONE_REMOTE"
ENV_STORAGE_RCLONE_TYPE: Final = "SYNTH_SETTER_STORAGE_RCLONE_TYPE"

STORAGE_REQUIRED_ENV_KEYS: Final[tuple[str, ...]] = (
    ENV_STORAGE_ACCESS_KEY_ID,
    ENV_STORAGE_SECRET_ACCESS_KEY,
    ENV_STORAGE_ENDPOINT_URL,
)

_RCLONE_ENV_TYPE: Final = "RCLONE_CONFIG_R2_TYPE"
_RCLONE_ENV_PROVIDER: Final = "RCLONE_CONFIG_R2_PROVIDER"
_RCLONE_ENV_ACCESS_KEY_ID: Final = "RCLONE_CONFIG_R2_ACCESS_KEY_ID"
_RCLONE_ENV_SECRET_ACCESS_KEY: Final = "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY"  # noqa: S105
_RCLONE_ENV_ENDPOINT: Final = "RCLONE_CONFIG_R2_ENDPOINT"

RCLONE_STRUCTURAL_DEFAULTS: Final[Mapping[str, str]] = MappingProxyType(
    {_RCLONE_ENV_TYPE: "s3", _RCLONE_ENV_PROVIDER: "Cloudflare"}
)
RCLONE_REQUIRED_ENV_KEYS: Final[tuple[str, ...]] = (
    _RCLONE_ENV_ACCESS_KEY_ID,
    _RCLONE_ENV_SECRET_ACCESS_KEY,
    _RCLONE_ENV_ENDPOINT,
)
RCLONE_ENV_KEYS: Final[tuple[str, ...]] = (
    _RCLONE_ENV_TYPE,
    _RCLONE_ENV_PROVIDER,
    *RCLONE_REQUIRED_ENV_KEYS,
)

_DEFAULT_RCLONE_REMOTE: Final = "r2"
_RCLONE_REMOTE_RE: Final = re.compile(r"^[A-Za-z0-9_]+$")


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip() or None


def _settings_kwargs_from_sources(env_file: Path | None) -> dict[str, str]:
    candidates: dict[str, str | None] = {}
    if env_file is not None and env_file.is_file():
        candidates.update(dotenv_values(env_file))
    kwargs: dict[str, str] = {}
    env_to_field = {
        ENV_STORAGE_ACCESS_KEY_ID: "access_key_id",
        ENV_STORAGE_DEFAULT_BUCKET: "default_bucket",
        ENV_STORAGE_SECRET_ACCESS_KEY: "secret_access_key",
        ENV_STORAGE_ENDPOINT_URL: "endpoint_url",
        ENV_STORAGE_PROVIDER: "provider",
        ENV_STORAGE_RCLONE_REMOTE: "rclone_remote",
        ENV_STORAGE_RCLONE_TYPE: "rclone_type",
        ENV_STORAGE_REGION: "region",
    }
    for env_key, field_name in env_to_field.items():
        value = _clean(candidates.get(env_key))
        if value is None:
            value = _clean(os.environ.get(env_key))
        if value is not None:
            kwargs[field_name] = value
    return kwargs


class ObjectStoreProvider(StrEnum):
    """Supported S3-compatible provider profiles.

    .. attribute :: R2

        Cloudflare R2 profile.

    .. attribute :: S3

        AWS S3 profile.

    .. attribute :: CUSTOM

        Custom S3-compatible endpoint profile.
    """

    R2 = "r2"
    S3 = "s3"
    CUSTOM = "custom"


class StorageConfig(BaseModel):
    """Validated object-storage config without process-env coupling.

    .. attribute :: model_config

        Pydantic model config sentinel — strict, frozen, ``extra="forbid"``.

    .. attribute :: provider

        S3-compatible provider profile.

    .. attribute :: access_key_id

        Object-store access key id.

    .. attribute :: secret_access_key

        Object-store secret access key.

    .. attribute :: endpoint_url

        S3-compatible endpoint URL.

    .. attribute :: region

        S3 region string; R2 uses ``"auto"``.

    .. attribute :: default_bucket

        Optional bucket used by callers that accept a bucket-less shorthand.

    .. attribute :: rclone_remote

        rclone remote section name used by the current backend adapter.

    .. attribute :: rclone_type

        rclone backend type used by the current backend adapter.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    provider: ObjectStoreProvider = ObjectStoreProvider.R2
    access_key_id: SecretStr
    secret_access_key: SecretStr
    endpoint_url: str
    region: str = Field(default="auto")
    default_bucket: str | None = None
    rclone_remote: str = Field(default=_DEFAULT_RCLONE_REMOTE)
    rclone_type: str = Field(default=RCLONE_STRUCTURAL_DEFAULTS[_RCLONE_ENV_TYPE])

    @field_validator("access_key_id", "secret_access_key")
    @classmethod
    def _secret_is_nonblank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("must be non-blank")
        return value

    @field_validator("endpoint_url", "region", "rclone_remote", "rclone_type")
    @classmethod
    def _string_is_nonblank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be non-blank")
        return stripped

    @field_validator("default_bucket")
    @classmethod
    def _optional_string_is_nonblank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be non-blank when set")
        return stripped

    @field_validator("rclone_remote")
    @classmethod
    def _rclone_remote_is_env_safe(cls, value: str) -> str:
        if not _RCLONE_REMOTE_RE.match(value):
            raise ValueError("must contain only letters, numbers, or underscores")
        return value

    def lance_storage_options(self) -> dict[str, str]:
        """Return Lance/object-store storage options for S3-compatible storage.

        :returns: Mapping accepted by Lance/pyarrow object-store readers.
        """
        return {
            "access_key_id": self.access_key_id.get_secret_value(),
            "secret_access_key": self.secret_access_key.get_secret_value(),
            "endpoint": self.endpoint_url,
            "region": self.region,
        }

    def rclone_env(self) -> dict[str, str]:
        """Return the rclone env block for the current rclone-backed facade.

        :returns: Environment variables using rclone's per-remote naming convention.
        """
        prefix = f"RCLONE_CONFIG_{self.rclone_remote.upper()}_"
        return {
            f"{prefix}TYPE": self.rclone_type,
            f"{prefix}PROVIDER": RCLONE_STRUCTURAL_DEFAULTS[_RCLONE_ENV_PROVIDER],
            f"{prefix}ACCESS_KEY_ID": self.access_key_id.get_secret_value(),
            f"{prefix}SECRET_ACCESS_KEY": self.secret_access_key.get_secret_value(),
            f"{prefix}ENDPOINT": self.endpoint_url,
        }


class StorageSettings(BaseSettings):
    """Object-storage settings loaded from ``SYNTH_SETTER_STORAGE_*`` env.

    .. attribute :: model_config

        Pydantic settings config sentinel.

    .. attribute :: access_key_id

        Object-store access key id.

    .. attribute :: secret_access_key

        Object-store secret access key.

    .. attribute :: endpoint_url

        S3-compatible endpoint URL.

    .. attribute :: provider

        S3-compatible provider profile.

    .. attribute :: region

        S3 region string; R2 uses ``"auto"``.

    .. attribute :: default_bucket

        Optional bucket used by callers that accept a bucket-less shorthand.

    .. attribute :: rclone_remote

        rclone remote section name used by the current backend adapter.

    .. attribute :: rclone_type

        rclone backend type used by the current backend adapter.
    """

    model_config = SettingsConfigDict(
        env_prefix="SYNTH_SETTER_STORAGE_",
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
        str_strip_whitespace=True,
    )

    access_key_id: SecretStr
    secret_access_key: SecretStr
    endpoint_url: str
    provider: ObjectStoreProvider = ObjectStoreProvider.R2
    region: str = "auto"
    default_bucket: str | None = None
    rclone_remote: str = _DEFAULT_RCLONE_REMOTE
    rclone_type: str = RCLONE_STRUCTURAL_DEFAULTS[_RCLONE_ENV_TYPE]

    @field_validator("access_key_id", "secret_access_key")
    @classmethod
    def _secret_is_nonblank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("must be non-blank")
        return value

    @field_validator("endpoint_url", "region", "rclone_remote", "rclone_type")
    @classmethod
    def _string_is_nonblank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be non-blank")
        return stripped

    @field_validator("default_bucket")
    @classmethod
    def _optional_string_is_nonblank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be non-blank when set")
        return stripped

    @field_validator("rclone_remote")
    @classmethod
    def _rclone_remote_is_env_safe(cls, value: str) -> str:
        if not _RCLONE_REMOTE_RE.match(value):
            raise ValueError("must contain only letters, numbers, or underscores")
        return value

    def to_config(self) -> StorageConfig:
        """Return an env-free storage config value.

        :returns: Strict immutable storage configuration.
        """
        return StorageConfig(
            provider=self.provider,
            access_key_id=self.access_key_id,
            secret_access_key=self.secret_access_key,
            endpoint_url=self.endpoint_url,
            region=self.region,
            default_bucket=self.default_bucket,
            rclone_remote=self.rclone_remote,
            rclone_type=self.rclone_type,
        )


def storage_settings_from_sources(env_file: Path | None = None) -> StorageSettings:
    """Load settings with dotenv values taking precedence over process env.

    :param env_file: Optional dotenv path to read before falling back to ``os.environ``.
    :returns: Storage settings parsed from canonical storage environment keys.
    """
    if env_file is None:
        return StorageSettings()  # pyright: ignore[reportCallIssue]
    return StorageSettings(**_settings_kwargs_from_sources(env_file))  # pyright: ignore[reportCallIssue, reportArgumentType]


class ObjectLocation(BaseModel):
    """Strict bucket/key pointer for S3-compatible object storage.

    .. attribute :: model_config

        Pydantic model config sentinel — strict, frozen, ``extra="forbid"``.

    .. attribute :: bucket

        Object-store bucket name.

    .. attribute :: key

        Object key within the bucket.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    bucket: str
    key: str

    @classmethod
    def from_uri(cls, uri: str) -> ObjectLocation:
        """Normalize an ``s3://bucket/key`` URI into an object location.

        :param uri: Public object-storage URI.
        :returns: Parsed bucket/key location.
        :raises ValueError: ``uri`` is not an ``s3://bucket/key`` object URI.
        """
        parsed = urlsplit(uri)
        key = parsed.path.lstrip("/")
        if parsed.scheme != "s3" or not parsed.netloc or not key:
            raise ValueError(f"object storage locations must be s3://bucket/key, got {uri!r}")
        return cls(bucket=parsed.netloc, key=key)

    @property
    def uri(self) -> str:
        """Return the normalized public URI."""
        return f"s3://{self.bucket}/{self.key}"


def _join_key(location: ObjectLocation, *parts: str) -> ObjectLocation:
    suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
    if not suffix:
        return location
    return ObjectLocation(bucket=location.bucket, key=f"{location.key.rstrip('/')}/{suffix}")


def _to_rclone_path(location: ObjectLocation, *, remote: str) -> str:
    return f"{remote}:{location.bucket}/{location.key}"


class ObjectStorage(BaseModel):
    """Small rclone-backed facade for S3-compatible object storage.

    .. attribute :: model_config

        Pydantic model config sentinel — strict, frozen, ``extra="forbid"``.

    .. attribute :: config

        Env-free storage configuration projected into backend dialects.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    config: StorageConfig

    def upload_file(self, local_path: Path, destination: ObjectLocation) -> None:
        """Upload a file to an object location via rclone.

        :param local_path: Local file path to upload.
        :param destination: Object location to write.
        """
        env = {**os.environ, **self.config.rclone_env()}
        subprocess.check_call(  # noqa: S603
            [  # noqa: S607
                "rclone",
                "copyto",
                str(local_path),
                _to_rclone_path(destination, remote=self.config.rclone_remote),
                "--checksum",
            ],
            env=env,
        )

    def download_file(self, source: ObjectLocation, local_path: Path) -> None:
        """Download one object to a local file via rclone.

        :param source: Object location to read.
        :param local_path: Local file path to write.
        """
        env = {**os.environ, **self.config.rclone_env()}
        subprocess.check_call(  # noqa: S603
            [  # noqa: S607
                "rclone",
                "copyto",
                _to_rclone_path(source, remote=self.config.rclone_remote),
                str(local_path),
                "--checksum",
            ],
            env=env,
        )
