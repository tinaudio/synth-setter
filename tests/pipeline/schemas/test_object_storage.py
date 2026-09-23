"""Tests for provider-neutral object-storage settings and location contracts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from synth_setter.pipeline.schemas.object_storage import (
    RCLONE_ENV_KEYS,
    RCLONE_REQUIRED_ENV_KEYS,
    RCLONE_STRUCTURAL_DEFAULTS,
    ObjectLocation,
    ObjectStorage,
    ObjectStoreProvider,
    StorageConfig,
    StorageSettings,
    storage_settings_from_sources,
)

_ENDPOINT = "https://acct.r2.cloudflarestorage.com"
_VALID_FIELDS = {
    "access_key_id": "ak",
    "secret_access_key": "sk",  # noqa: S106 - test fixture value.
    "endpoint_url": _ENDPOINT,
    "default_bucket": "bucket",
}
_VALID_ENV = {
    "SYNTH_SETTER_STORAGE_ACCESS_KEY_ID": "ak",
    "SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY": "sk",  # noqa: S105 - env-var name.
    "SYNTH_SETTER_STORAGE_ENDPOINT_URL": _ENDPOINT,
    "SYNTH_SETTER_STORAGE_PROVIDER": "r2",
    "SYNTH_SETTER_STORAGE_DEFAULT_BUCKET": "bucket",
    "SYNTH_SETTER_STORAGE_RCLONE_TYPE": "s3",
}


@pytest.fixture(autouse=True)
def _clear_storage_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove storage-related env so tests do not read a developer shell.

    :param monkeypatch: Pytest fixture used to clear process env.
    """
    for key in list(os.environ):
        if key.startswith(("SYNTH_SETTER_STORAGE_", "RCLONE_CONFIG_R2_")):
            monkeypatch.delenv(key, raising=False)


def _config(**overrides: object) -> StorageConfig:
    return StorageConfig.model_validate({**_VALID_FIELDS, **overrides})


def _settings_from_env() -> StorageSettings:
    return StorageSettings()  # pyright: ignore[reportCallIssue]


class TestStorageSettings:
    """Settings accept canonical names and the legacy rclone aliases."""

    def test_reads_canonical_storage_env_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Canonical env names populate every public settings field.

        :param monkeypatch: Pytest fixture used to set process env.
        """
        for key, value in _VALID_ENV.items():
            monkeypatch.setenv(key, value)

        settings = _settings_from_env()

        assert settings.access_key_id.get_secret_value() == "ak"
        assert settings.secret_access_key.get_secret_value() == "sk"
        assert settings.endpoint_url == _ENDPOINT
        assert settings.provider is ObjectStoreProvider.R2
        assert settings.default_bucket == "bucket"
        assert settings.rclone_type == "s3"

    def test_reads_legacy_rclone_env_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Legacy rclone names populate settings when canonical names are absent.

        :param monkeypatch: Pytest fixture used to set process env.
        """
        monkeypatch.setenv("RCLONE_CONFIG_R2_ACCESS_KEY_ID", "ak")
        monkeypatch.setenv("RCLONE_CONFIG_R2_SECRET_ACCESS_KEY", "sk")
        monkeypatch.setenv("RCLONE_CONFIG_R2_ENDPOINT", _ENDPOINT)

        settings = storage_settings_from_sources()

        assert settings.access_key_id.get_secret_value() == "ak"
        assert settings.secret_access_key.get_secret_value() == "sk"
        assert settings.endpoint_url == _ENDPOINT
        assert settings.provider is ObjectStoreProvider.R2

    def test_canonical_process_env_overrides_legacy_dotenv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Canonical process credentials override legacy dotenv aliases.

        :param tmp_path: Pytest fixture providing a temp directory.
        :param monkeypatch: Pytest fixture used to set process env.
        """
        env_file = tmp_path / ".env"
        env_file.write_text(
            "RCLONE_CONFIG_R2_ACCESS_KEY_ID=legacy-access-key\n"
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=legacy-secret-key\n"
            "RCLONE_CONFIG_R2_ENDPOINT=https://legacy.example\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID", "canonical-access-key")
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY", "canonical-secret-key")
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_ENDPOINT_URL", "https://canonical.example")

        settings = storage_settings_from_sources(env_file)

        assert settings.access_key_id.get_secret_value() == "canonical-access-key"
        assert settings.secret_access_key.get_secret_value() == "canonical-secret-key"
        assert settings.endpoint_url == "https://canonical.example"

    def test_canonical_dotenv_overrides_legacy_dotenv(self, tmp_path: Path) -> None:
        """Canonical dotenv credentials override legacy aliases in the same file.

        :param tmp_path: Pytest fixture providing a temp directory.
        """
        env_file = tmp_path / ".env"
        env_file.write_text(
            "SYNTH_SETTER_STORAGE_ACCESS_KEY_ID=canonical-access-key\n"
            "SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY=canonical-secret-key\n"
            "SYNTH_SETTER_STORAGE_ENDPOINT_URL=https://canonical.example\n"
            "RCLONE_CONFIG_R2_ACCESS_KEY_ID=legacy-access-key\n"
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=legacy-secret-key\n"
            "RCLONE_CONFIG_R2_ENDPOINT=https://legacy.example\n",
            encoding="utf-8",
        )

        settings = storage_settings_from_sources(env_file)

        assert settings.access_key_id.get_secret_value() == "canonical-access-key"
        assert settings.secret_access_key.get_secret_value() == "canonical-secret-key"
        assert settings.endpoint_url == "https://canonical.example"

    def test_blank_canonical_dotenv_uses_legacy_dotenv(self, tmp_path: Path) -> None:
        """Blank canonical dotenv values fall back to legacy aliases.

        :param tmp_path: Pytest fixture providing a temp directory.
        """
        env_file = tmp_path / ".env"
        env_file.write_text(
            "SYNTH_SETTER_STORAGE_ACCESS_KEY_ID=\n"
            "SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY=\n"
            "SYNTH_SETTER_STORAGE_ENDPOINT_URL=\n"
            "RCLONE_CONFIG_R2_ACCESS_KEY_ID=legacy-access-key\n"
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=legacy-secret-key\n"
            "RCLONE_CONFIG_R2_ENDPOINT=https://legacy.example\n",
            encoding="utf-8",
        )

        settings = storage_settings_from_sources(env_file)

        assert settings.access_key_id.get_secret_value() == "legacy-access-key"
        assert settings.secret_access_key.get_secret_value() == "legacy-secret-key"
        assert settings.endpoint_url == "https://legacy.example"

    def test_legacy_process_env_overrides_legacy_dotenv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A legacy process-env alias takes precedence over the same dotenv key.

        :param tmp_path: Pytest fixture providing a temp directory.
        :param monkeypatch: Pytest fixture used to set process env.
        """
        env_file = tmp_path / ".env"
        env_file.write_text(
            "RCLONE_CONFIG_R2_ACCESS_KEY_ID=dotenv-access-key\n"
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=dotenv-secret-key\n"
            "RCLONE_CONFIG_R2_ENDPOINT=https://dotenv.example\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("RCLONE_CONFIG_R2_ACCESS_KEY_ID", "process-access-key")
        monkeypatch.setenv("RCLONE_CONFIG_R2_SECRET_ACCESS_KEY", "process-secret-key")
        monkeypatch.setenv("RCLONE_CONFIG_R2_ENDPOINT", "https://process.example")

        settings = storage_settings_from_sources(env_file)

        assert settings.access_key_id.get_secret_value() == "process-access-key"
        assert settings.secret_access_key.get_secret_value() == "process-secret-key"
        assert settings.endpoint_url == "https://process.example"

    def test_prefers_canonical_names_over_legacy_rclone_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Canonical names override legacy aliases when both are populated.

        :param monkeypatch: Pytest fixture used to set process env.
        """
        monkeypatch.setenv("RCLONE_CONFIG_R2_ACCESS_KEY_ID", "legacy-access-key")
        monkeypatch.setenv("RCLONE_CONFIG_R2_SECRET_ACCESS_KEY", "legacy-secret-key")
        monkeypatch.setenv("RCLONE_CONFIG_R2_ENDPOINT", "https://legacy.example")
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID", "canonical-access-key")
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY", "canonical-secret-key")
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_ENDPOINT_URL", "https://canonical.example")

        settings = storage_settings_from_sources()

        assert settings.access_key_id.get_secret_value() == "canonical-access-key"
        assert settings.secret_access_key.get_secret_value() == "canonical-secret-key"
        assert settings.endpoint_url == "https://canonical.example"

    def test_reads_legacy_rclone_names_from_env_file(self, tmp_path: Path) -> None:
        """Legacy rclone aliases in a dotenv file resolve storage settings.

        :param tmp_path: Pytest fixture providing a temp directory.
        """
        env_file = tmp_path / ".env"
        env_file.write_text(
            "RCLONE_CONFIG_R2_ACCESS_KEY_ID=ak\n"
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=sk\n"
            f"RCLONE_CONFIG_R2_ENDPOINT={_ENDPOINT}\n",
            encoding="utf-8",
        )

        settings = storage_settings_from_sources(env_file)

        assert settings.to_config().rclone_env() == {
            "RCLONE_CONFIG_R2_TYPE": "s3",
            "RCLONE_CONFIG_R2_PROVIDER": "Cloudflare",
            "RCLONE_CONFIG_R2_ACCESS_KEY_ID": "ak",
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": "sk",
            "RCLONE_CONFIG_R2_ENDPOINT": _ENDPOINT,
        }

    def test_process_env_overrides_env_file_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exported process-env value takes precedence over the dotenv file.

        :param tmp_path: Pytest fixture providing a temp directory.
        :param monkeypatch: Pytest fixture used to set process env.
        """
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID", "from-process")
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY", "sk")
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_ENDPOINT_URL", _ENDPOINT)
        env_file = tmp_path / ".env"
        env_file.write_text(
            "SYNTH_SETTER_STORAGE_ACCESS_KEY_ID=from-file\n",
            encoding="utf-8",
        )

        settings = storage_settings_from_sources(env_file)

        assert settings.access_key_id.get_secret_value() == "from-process"

    def test_blank_env_file_value_falls_back_to_process_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Blank dotenv values are absent and do not mask process env.

        :param tmp_path: Pytest fixture providing a temp directory.
        :param monkeypatch: Pytest fixture used to set process env.
        """
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID", "from-process")
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY", "sk")
        monkeypatch.setenv("SYNTH_SETTER_STORAGE_ENDPOINT_URL", _ENDPOINT)
        env_file = tmp_path / ".env"
        env_file.write_text("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID=\n", encoding="utf-8")

        settings = storage_settings_from_sources(env_file)

        assert settings.access_key_id.get_secret_value() == "from-process"

    def test_to_config_returns_env_free_value_object(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Settings project to the strict env-free config value.

        :param monkeypatch: Pytest fixture used to set process env.
        """
        for key, value in _VALID_ENV.items():
            monkeypatch.setenv(key, value)
        settings = _settings_from_env()

        assert settings.to_config() == _config()


class TestStorageConfig:
    """The storage config is strict, frozen, and projects backend dialects."""

    def test_lance_storage_options_use_object_store_keys(self) -> None:
        """The Lance projection uses object-store option names."""
        assert _config().lance_storage_options() == {
            "access_key_id": "ak",
            "secret_access_key": "sk",
            "endpoint": _ENDPOINT,
            "aws_endpoint": _ENDPOINT,
            "region": "auto",
        }

    def test_rclone_env_uses_pinned_r2_remote_keys(self) -> None:
        """The rclone projection always targets the pipeline's ``r2`` remote."""
        assert _config().rclone_env() == {
            "RCLONE_CONFIG_R2_TYPE": "s3",
            "RCLONE_CONFIG_R2_PROVIDER": "Cloudflare",
            "RCLONE_CONFIG_R2_ACCESS_KEY_ID": "ak",
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": "sk",
            "RCLONE_CONFIG_R2_ENDPOINT": _ENDPOINT,
        }

    def test_storage_env_round_trips_canonical_settings(self) -> None:
        """The canonical projection reproduces the settings' own env surface."""
        assert _config().storage_env() == {
            "SYNTH_SETTER_STORAGE_PROVIDER": "r2",
            "SYNTH_SETTER_STORAGE_ACCESS_KEY_ID": "ak",
            "SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY": "sk",
            "SYNTH_SETTER_STORAGE_ENDPOINT_URL": _ENDPOINT,
            "SYNTH_SETTER_STORAGE_REGION": "auto",
            "SYNTH_SETTER_STORAGE_RCLONE_TYPE": "s3",
            "SYNTH_SETTER_STORAGE_DEFAULT_BUCKET": "bucket",
        }

    def test_storage_env_omits_unset_default_bucket(self) -> None:
        """An unset optional bucket does not project a blank env entry."""
        config = StorageConfig.model_validate(
            {k: v for k, v in _VALID_FIELDS.items() if k != "default_bucket"}
        )
        assert "SYNTH_SETTER_STORAGE_DEFAULT_BUCKET" not in config.storage_env()

    def test_rclone_env_projects_current_rclone_type(self) -> None:
        """The rclone projection derives backend type from storage config."""
        config = _config(rclone_type="local")

        assert config.rclone_env()["RCLONE_CONFIG_R2_TYPE"] == "local"

    def test_projection_keys_match_constants(self) -> None:
        """Default rclone projection stays aligned with exported constants."""
        config = _config()
        assert set(config.rclone_env()) == set(RCLONE_ENV_KEYS)
        assert set(RCLONE_REQUIRED_ENV_KEYS).issubset(config.rclone_env())
        for key, value in RCLONE_STRUCTURAL_DEFAULTS.items():
            assert config.rclone_env()[key] == value

    def test_rejects_unknown_field(self) -> None:
        """Strict config rejects a misspelled or obsolete field."""
        with pytest.raises(ValidationError):
            StorageConfig.model_validate({**_VALID_FIELDS, "bucket": "oops"})

    def test_is_frozen(self) -> None:
        """Constructed config values cannot be mutated in place."""
        config = _config()
        with pytest.raises(ValidationError):
            config.endpoint_url = "https://other.example"  # type: ignore[misc]

    def test_repr_and_str_redact_secret_values(self) -> None:
        """Secret fields are redacted in repr and str output."""
        config = _config(access_key_id="leakme", secret_access_key="leaktoo")  # noqa: S106
        for text in (repr(config), str(config)):
            assert "leakme" not in text
            assert "leaktoo" not in text


class TestObjectLocation:
    """Object locations normalize public s3:// strings into bucket/key pairs."""

    def test_from_uri_accepts_s3_bucket_key(self) -> None:
        """An s3 URI parses into canonical bucket/key fields."""
        location = ObjectLocation.from_uri("s3://bucket/path/to/object.json")
        assert location.bucket == "bucket"
        assert location.key == "path/to/object.json"
        assert location.uri == "s3://bucket/path/to/object.json"

    @pytest.mark.parametrize(
        "uri",
        [
            "r2://bucket/path",
            "file:///tmp/object",
            "bucket/path",
            "s3://",
            "s3://bucket",
            "s3://bucket/",
        ],
    )
    def test_from_uri_rejects_non_object_locations(self, uri: str) -> None:
        """Only s3://bucket/key public object URIs are accepted.

        :param uri: Candidate URI to reject.
        """
        with pytest.raises(ValueError, match="s3://bucket/key"):
            ObjectLocation.from_uri(uri)

    def test_child_is_not_public_api(self) -> None:
        """ObjectLocation exposes no public child/join helper."""
        assert not hasattr(ObjectLocation(bucket="bucket", key="key"), "child")


class TestObjectStorage:
    """The facade translates object locations and injects rclone env."""

    def test_upload_file_invokes_rclone_with_projected_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The upload facade translates location and injects projected env.

        :param tmp_path: Pytest fixture providing a temp directory.
        :param monkeypatch: Pytest fixture used to patch subprocess calls.
        """
        source = tmp_path / "object.bin"
        source.write_bytes(b"payload")
        captured: dict[str, object] = {}

        def _capture(args: list[str], **kwargs: object) -> None:
            captured["args"] = args
            captured["env"] = kwargs["env"]

        monkeypatch.setattr(subprocess, "check_call", _capture)

        ObjectStorage(config=_config()).upload_file(
            source, ObjectLocation.from_uri("s3://bucket/path/object.bin")
        )

        assert captured["args"] == [
            "rclone",
            "copyto",
            str(source),
            "r2:bucket/path/object.bin",
            "--checksum",
        ]
        env = captured["env"]
        assert isinstance(env, dict)
        assert env["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] == "ak"

    def test_download_file_invokes_rclone_with_projected_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The download facade translates location and injects projected env.

        :param tmp_path: Pytest fixture providing a temp directory.
        :param monkeypatch: Pytest fixture used to patch subprocess calls.
        """
        destination = tmp_path / "object.bin"
        captured: dict[str, object] = {}

        def _capture(args: list[str], **kwargs: object) -> None:
            captured["args"] = args
            captured["env"] = kwargs["env"]

        monkeypatch.setattr(subprocess, "check_call", _capture)

        ObjectStorage(config=_config()).download_file(
            ObjectLocation.from_uri("s3://bucket/path/object.bin"), destination
        )

        assert captured["args"] == [
            "rclone",
            "copyto",
            "r2:bucket/path/object.bin",
            str(destination),
            "--checksum",
        ]
        env = captured["env"]
        assert isinstance(env, dict)
        assert env["RCLONE_CONFIG_R2_ENDPOINT"] == _ENDPOINT
