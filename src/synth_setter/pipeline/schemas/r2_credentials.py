"""Single source of truth for Cloudflare R2 access credentials and their dialects.

R2's one logical secret — access key, secret, endpoint — is consumed under
several env-var vocabularies (rclone's ``RCLONE_CONFIG_R2_*``, Lance's
object-store ``storage_options``). This module owns the canonical env-var names
and the structural constants once, parses them at the trust boundary into a
strict :class:`R2Credentials`, and projects each dialect from that single value
so call sites never re-derive a key list or an endpoint URL.

``from_env`` is the imperative shell (reads process env / a supplied mapping);
the model and its ``*_options`` / ``*_env`` projections are a pure functional
core.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, SecretStr

__all__ = [
    "ENV_ACCESS_KEY_ID",
    "ENV_ACCOUNT_ID",
    "ENV_ENDPOINT",
    "ENV_PROVIDER",
    "ENV_SECRET_ACCESS_KEY",
    "ENV_TYPE",
    "RCLONE_ENV_KEYS",
    "SECRET_ENV_KEYS",
    "STRUCTURAL_DEFAULTS",
    "R2Credentials",
    "endpoint_for_account",
]

# rclone reads these directly via its env-override convention; import, don't spell them.
ENV_ACCESS_KEY_ID: Final = "RCLONE_CONFIG_R2_ACCESS_KEY_ID"
ENV_SECRET_ACCESS_KEY: Final = "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY"  # noqa: S105 — env-var name
ENV_ENDPOINT: Final = "RCLONE_CONFIG_R2_ENDPOINT"
ENV_TYPE: Final = "RCLONE_CONFIG_R2_TYPE"
ENV_PROVIDER: Final = "RCLONE_CONFIG_R2_PROVIDER"

# Cloudflare account id; the SkyPilot R2 adaptor needs it and ``from_env`` derives
# the endpoint from it when the explicit endpoint var is unset.
ENV_ACCOUNT_ID: Final = "R2_ACCOUNT_ID"

# Order is load-bearing: it sets the order missing keys are reported in.
SECRET_ENV_KEYS: Final[tuple[str, ...]] = (
    ENV_ACCESS_KEY_ID,
    ENV_SECRET_ACCESS_KEY,
    ENV_ENDPOINT,
)

# rclone needs type+provider to assemble the ``r2:`` remote (else ``rclone lsd
# r2:`` reports "didn't find section in config file"); read-only to block aliased mutation.
STRUCTURAL_DEFAULTS: Final[Mapping[str, str]] = MappingProxyType(
    {ENV_TYPE: "s3", ENV_PROVIDER: "Cloudflare"}
)

# Composed here so the launcher's worker-env list and this key set cannot drift apart.
RCLONE_ENV_KEYS: Final[tuple[str, ...]] = (ENV_TYPE, ENV_PROVIDER, *SECRET_ENV_KEYS)

_R2_ENDPOINT_TEMPLATE: Final = "https://{account_id}.r2.cloudflarestorage.com"


def endpoint_for_account(account_id: str) -> str:
    """Build the per-account R2 S3-compatible endpoint URL.

    :param account_id: Cloudflare account id; becomes the endpoint subdomain.
    :returns: ``https://<account_id>.r2.cloudflarestorage.com``.
    """
    return _R2_ENDPOINT_TEMPLATE.format(account_id=account_id)


def _clean(value: str | None) -> str | None:
    """Return the stripped value, or ``None`` when blank/whitespace-only (treated as absent).

    :param value: Raw env value; ``None`` when the key is unset.
    :returns: The stripped value, or ``None`` if it was unset or blank.
    """
    if value is None:
        return None
    return value.strip() or None


class R2Credentials(BaseModel):
    """Resolved R2 access credentials — the canonical handle every R2 consumer reads.

    Construct directly from known values, or via :meth:`from_env` to resolve from
    a process-env mapping. :meth:`lance_storage_options` projects the Lance
    object-store dialect; :meth:`rclone_env` projects the resolved
    ``RCLONE_CONFIG_R2_*`` block callers write back into ``os.environ`` for the
    rclone subprocess. The two access secrets are :class:`~pydantic.SecretStr` so
    a stray ``repr`` / log of the model cannot leak them.

    .. attribute :: model_config

        Pydantic model config sentinel — strict, frozen, ``extra="forbid"``.

    .. attribute :: access_key_id

        R2 access key id (secret).

    .. attribute :: secret_access_key

        R2 secret access key (secret).

    .. attribute :: endpoint

        R2 S3-compatible endpoint URL (explicit, or derived from the account id).

    .. attribute :: rclone_type

        rclone remote ``type``; constant ``"s3"`` for Cloudflare R2.

    .. attribute :: rclone_provider

        rclone remote ``provider``; constant ``"Cloudflare"`` for R2.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    access_key_id: SecretStr = Field(description="R2 access key id (secret).")
    secret_access_key: SecretStr = Field(description="R2 secret access key (secret).")
    endpoint: str = Field(
        description="R2 S3-compatible endpoint URL (explicit or derived from the account id)."
    )
    rclone_type: str = Field(default="s3", description="rclone remote type; constant 's3' for R2.")
    rclone_provider: str = Field(
        default="Cloudflare", description="rclone remote provider; constant 'Cloudflare' for R2."
    )

    def lance_storage_options(self) -> dict[str, str]:
        """Project Lance's object-store ``storage_options`` for the R2 bucket.

        :returns: Mapping of ``access_key_id`` / ``secret_access_key`` / ``endpoint`` plus
            ``region="auto"`` — R2 ignores region but object-store requires it set for
            S3-compatible stores, and ``"auto"`` is R2's documented placeholder.
        """
        return {
            "access_key_id": self.access_key_id.get_secret_value(),
            "secret_access_key": self.secret_access_key.get_secret_value(),
            "endpoint": self.endpoint,
            "region": "auto",
        }

    def rclone_env(self) -> dict[str, str]:
        """Project the resolved ``RCLONE_CONFIG_R2_*`` env block rclone reads.

        Callers write this back into ``os.environ`` so the rclone subprocess sees
        the same stripped, blank-free, default-filled, account-id-derived values
        ``from_env`` validated — never a raw/blank/padded dotenv value.

        :returns: The five-key mapping (type, provider, and the three secrets).
        """
        return {
            ENV_TYPE: self.rclone_type,
            ENV_PROVIDER: self.rclone_provider,
            ENV_ACCESS_KEY_ID: self.access_key_id.get_secret_value(),
            ENV_SECRET_ACCESS_KEY: self.secret_access_key.get_secret_value(),
            ENV_ENDPOINT: self.endpoint,
        }

    @classmethod
    def from_env(cls, env: Mapping[str, str | None] | None = None) -> R2Credentials:
        """Resolve credentials from a process-env mapping.

        Applies the structural defaults (caller values win — an explicit
        ``RCLONE_CONFIG_R2_TYPE``/``_PROVIDER`` override is honored verbatim),
        derives the endpoint from :data:`ENV_ACCOUNT_ID` when the explicit
        endpoint var is absent, then validates the three secrets are present and
        non-blank. Blank/whitespace is treated as absent so a half-set ``KEY=``
        cannot build a partial credential.

        :param env: Mapping to read; defaults to ``os.environ``. Values may be
            ``None`` (``dotenv_values`` returns ``str | None``).
        :returns: The resolved, validated credentials.
        :raises RuntimeError: A required secret is unset or blank after resolution.
        """
        source: Mapping[str, str | None] = os.environ if env is None else env

        resolved: dict[str, str] = dict(STRUCTURAL_DEFAULTS)
        for key in RCLONE_ENV_KEYS:
            cleaned = _clean(source.get(key))
            if cleaned is not None:
                resolved[key] = cleaned

        if resolved.get(ENV_ENDPOINT) is None:
            account_id = _clean(source.get(ENV_ACCOUNT_ID))
            if account_id is not None:
                resolved[ENV_ENDPOINT] = endpoint_for_account(account_id)

        missing = [key for key in SECRET_ENV_KEYS if resolved.get(key) is None]
        if missing:
            raise RuntimeError(
                f"R2 credentials missing from process env: {', '.join(missing)}. "
                "Set RCLONE_CONFIG_R2_* (or R2_ACCOUNT_ID for endpoint derivation) first."
            )

        return cls(
            access_key_id=SecretStr(resolved[ENV_ACCESS_KEY_ID]),
            secret_access_key=SecretStr(resolved[ENV_SECRET_ACCESS_KEY]),
            endpoint=resolved[ENV_ENDPOINT],
            rclone_type=resolved[ENV_TYPE],
            rclone_provider=resolved[ENV_PROVIDER],
        )
