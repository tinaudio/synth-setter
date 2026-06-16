"""Tests for synth_setter.pipeline.schemas.r2_credentials — the R2 access source of truth.

The model is a pure value object: tests construct it directly and assert on the
projected dialect dicts. ``from_env`` is the imperative shell; its tests drive a
plain mapping (not ``os.environ``) so each case is isolated and order-free, with
one ``env=None`` case pinning the live ``os.environ`` branch.
"""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from synth_setter.pipeline.schemas import r2_credentials
from synth_setter.pipeline.schemas.r2_credentials import (
    RCLONE_ENV_KEYS,
    SECRET_ENV_KEYS,
    R2Credentials,
)

_ENDPOINT = "https://acct.r2.cloudflarestorage.com"
_VALID_FIELDS = {
    "access_key_id": "ak",
    "secret_access_key": "sk",  # noqa: S106 — test fixture value, not a real secret
    "endpoint": _ENDPOINT,
}
_VALID_ENV = {
    "RCLONE_CONFIG_R2_ACCESS_KEY_ID": "ak",
    "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": "sk",  # noqa: S105 — test fixture value
    "RCLONE_CONFIG_R2_ENDPOINT": _ENDPOINT,
}


def _creds(**overrides: str) -> R2Credentials:
    """Build an ``R2Credentials`` with valid defaults, overriding only what a test cares about.

    :param \\*\\*overrides: Field values to replace on the baseline credentials.
    :returns: A fully-populated ``R2Credentials`` instance.
    """
    # model_validate accepts plain str for SecretStr fields; R2Credentials(...) would type-error.
    return R2Credentials.model_validate({**_VALID_FIELDS, **overrides})


class TestLanceStorageOptions:
    """Tests for the Lance object-store projection."""

    def test_projects_documented_s3_keys_plus_auto_region(self) -> None:
        """The credential fields map to object-store keys with R2's ``region=auto`` placeholder."""
        assert _creds().lance_storage_options() == {
            "access_key_id": "ak",
            "secret_access_key": "sk",
            "endpoint": _ENDPOINT,
            "region": "auto",
        }

    def test_key_set_matches_documented_object_store_keys(self) -> None:
        """The Lance projection keys are pinned, so a stray addition/removal is caught."""
        assert set(_creds().lance_storage_options()) == {
            "access_key_id",
            "secret_access_key",
            "endpoint",
            "region",
        }


class TestRcloneEnv:
    """Tests for the resolved ``RCLONE_CONFIG_R2_*`` projection written back to ``os.environ``."""

    def test_projects_full_rclone_env_block(self) -> None:
        """All five rclone keys (two structural constants plus three secrets) are emitted."""
        assert _creds().rclone_env() == {
            "RCLONE_CONFIG_R2_TYPE": "s3",
            "RCLONE_CONFIG_R2_PROVIDER": "Cloudflare",
            "RCLONE_CONFIG_R2_ACCESS_KEY_ID": "ak",
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": "sk",
            "RCLONE_CONFIG_R2_ENDPOINT": _ENDPOINT,
        }

    def test_key_set_matches_rclone_env_keys_constant(self) -> None:
        """The projection's keys stay in lockstep with ``RCLONE_ENV_KEYS`` (drift guard)."""
        assert set(_creds().rclone_env()) == set(RCLONE_ENV_KEYS)


class TestModelStrictness:
    """Tests for the trust-boundary config: strict, frozen, no extras, redacted secrets."""

    def test_unknown_field_is_rejected(self) -> None:
        """``extra="forbid"`` rejects a stray key so typos surface at parse time."""
        with pytest.raises(ValidationError):
            R2Credentials.model_validate({**_VALID_FIELDS, "bucket": "oops"})

    def test_non_string_field_is_rejected(self) -> None:
        """``strict=True`` rejects a non-string where a ``str`` field is declared."""
        with pytest.raises(ValidationError):
            R2Credentials.model_validate({**_VALID_FIELDS, "endpoint": 1})

    def test_instance_is_frozen(self) -> None:
        """A constructed credential cannot be mutated in place."""
        creds = _creds()
        with pytest.raises(ValidationError):
            creds.access_key_id = "rotated"  # type: ignore[misc]

    def test_repr_and_str_redact_secret_values(self) -> None:
        """The access secrets are ``SecretStr`` so neither ``repr`` nor ``str`` leaks them."""
        creds = _creds(access_key_id="leakme", secret_access_key="leaktoo")  # noqa: S106
        for text in (repr(creds), str(creds)):
            assert "SecretStr(" in text  # the field rendered (redacted), not empty/raised
            assert "leakme" not in text
            assert "leaktoo" not in text


class TestFromEnv:
    """Tests for the env-resolution shell."""

    @pytest.fixture(autouse=True)
    def _clear_r2_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Drop stray ``RCLONE_CONFIG_R2_*`` / ``R2_ACCOUNT_ID`` so ``env=None`` reads a clean env.

        :param monkeypatch: Pytest fixture used to remove env vars.
        """
        import os

        for key in list(os.environ):
            if key.startswith("RCLONE_CONFIG_R2_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("R2_ACCOUNT_ID", raising=False)

    def test_reads_canonical_keys_and_defaults_structural_constants(self) -> None:
        """Three secrets in env yield a credential; type/provider default without being set."""
        creds = R2Credentials.from_env(_VALID_ENV)
        assert creds.rclone_type == "s3"
        assert creds.rclone_provider == "Cloudflare"
        assert creds.endpoint == _ENDPOINT
        assert creds.access_key_id.get_secret_value() == "ak"
        assert creds.secret_access_key.get_secret_value() == "sk"

    def test_reads_live_os_environ_when_env_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no mapping argument, resolution reads ``os.environ`` (the production call-site).

        :param monkeypatch: Pytest fixture used to set the R2 secret env vars.
        """
        for key, value in _VALID_ENV.items():
            monkeypatch.setenv(key, value)
        creds = R2Credentials.from_env()
        assert creds.endpoint == _ENDPOINT
        assert creds.access_key_id.get_secret_value() == "ak"

    def test_caller_override_of_structural_constant_wins(self) -> None:
        """An explicit ``RCLONE_CONFIG_R2_PROVIDER`` in env overrides the default."""
        env = {**_VALID_ENV, "RCLONE_CONFIG_R2_PROVIDER": "Other"}
        assert R2Credentials.from_env(env).rclone_provider == "Other"

    def test_strips_surrounding_whitespace_from_resolved_values(self) -> None:
        """A padded non-blank value is trimmed, not passed through verbatim."""
        env = {**_VALID_ENV, "RCLONE_CONFIG_R2_ACCESS_KEY_ID": "  ak  "}
        assert R2Credentials.from_env(env).lance_storage_options()["access_key_id"] == "ak"

    def test_derives_endpoint_from_account_id_when_endpoint_absent(self) -> None:
        """With no endpoint var but an account id, the endpoint is derived from the template."""
        env = {
            "RCLONE_CONFIG_R2_ACCESS_KEY_ID": "ak",
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": "sk",  # noqa: S105 — test fixture value
            "R2_ACCOUNT_ID": "abc123",
        }
        assert R2Credentials.from_env(env).endpoint == "https://abc123.r2.cloudflarestorage.com"

    def test_explicit_endpoint_wins_over_account_id_derivation(self) -> None:
        """A provided endpoint is used verbatim even when an account id could derive one."""
        env = {**_VALID_ENV, "R2_ACCOUNT_ID": "abc123"}
        assert R2Credentials.from_env(env).endpoint == _ENDPOINT

    def test_blank_account_id_does_not_derive_and_endpoint_stays_missing(self) -> None:
        """A blank ``R2_ACCOUNT_ID`` derives nothing, so the missing-endpoint check still raises."""
        env = {
            "RCLONE_CONFIG_R2_ACCESS_KEY_ID": "ak",
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": "sk",  # noqa: S105 — test fixture value
            "R2_ACCOUNT_ID": "   ",
        }
        with pytest.raises(RuntimeError, match="RCLONE_CONFIG_R2_ENDPOINT"):
            R2Credentials.from_env(env)

    def test_all_secrets_absent_reports_every_missing_canonical_name_in_order(self) -> None:
        """An empty mapping raises listing all three canonical secret names in declared order."""
        with pytest.raises(RuntimeError) as exc:
            R2Credentials.from_env({})
        message = str(exc.value)
        assert ", ".join(SECRET_ENV_KEYS) in message

    @pytest.mark.parametrize("absent_key", SECRET_ENV_KEYS)
    def test_missing_secret_raises_listing_the_absent_key(self, absent_key: str) -> None:
        """Each secret, when absent, is named in the raised RuntimeError.

        :param absent_key: The canonical secret env key dropped for this case.
        """
        env = {k: v for k, v in _VALID_ENV.items() if k != absent_key}
        with pytest.raises(RuntimeError, match=f"R2 credentials missing.*{re.escape(absent_key)}"):
            R2Credentials.from_env(env)

    @pytest.mark.parametrize("blank_key", SECRET_ENV_KEYS)
    def test_blank_secret_is_treated_as_missing(self, blank_key: str) -> None:
        """Each present-but-whitespace secret is rejected, not built into a partial credential.

        :param blank_key: The canonical secret env key blanked for this case.
        """
        env = {**_VALID_ENV, blank_key: "   "}
        with pytest.raises(RuntimeError, match=re.escape(blank_key)):
            R2Credentials.from_env(env)


class TestEndpointForAccount:
    """Tests for the account-id → endpoint derivation helper."""

    def test_builds_cloudflare_r2_endpoint(self) -> None:
        """The helper formats the documented per-account R2 S3 endpoint."""
        assert (
            r2_credentials.endpoint_for_account("abc123")
            == "https://abc123.r2.cloudflarestorage.com"
        )
