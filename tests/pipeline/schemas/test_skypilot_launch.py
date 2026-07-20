"""Tests for src/synth_setter/pipeline/schemas/skypilot_launch.py — launcher config schema."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from synth_setter.pipeline.schemas.skypilot_launch import (
    ENV_SKYPILOT_API_SERVER_ENDPOINT,
    ENV_SKYPILOT_SERVICE_ACCOUNT_TOKEN,
    SkypilotClientSettings,
    SkypilotLaunchConfig,
    skypilot_client_settings_from_sources,
)


class TestDefaults:
    """All fields default to safe local-only values when no input is given."""

    def test_default_compute_template_is_none(self) -> None:
        """Compute template defaults to None — the "don't dispatch" sentinel."""
        assert SkypilotLaunchConfig().compute_template is None

    def test_default_cmd_is_none(self) -> None:
        """Cmd defaults to None — populated by the Hydra entrypoint at dispatch time."""
        assert SkypilotLaunchConfig().cmd is None

    def test_default_num_workers_is_one(self) -> None:
        """Single worker is the default; >1 fans out parallel ranks."""
        assert SkypilotLaunchConfig().num_workers == 1

    def test_default_worker_image_tag_is_dev_snapshot(self) -> None:
        """Worker image tag defaults to the dev-snapshot rolling tag."""
        assert SkypilotLaunchConfig().worker_image_tag == "dev-snapshot"

    def test_default_tail_is_false(self) -> None:
        """Detach by default; ``tail`` is opt-in."""
        assert SkypilotLaunchConfig().tail is False

    def test_default_local_is_false(self) -> None:
        """No dispatch-mode preference by default; honor inherited env."""
        assert SkypilotLaunchConfig().local is False


class TestValidation:
    """Pydantic field validators reject invalid combinations early."""

    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_non_positive_num_workers_rejected(self, bad: int) -> None:
        """Any worker count below 1 surfaces as ValidationError naming the offending value.

        :param bad: Parametrized non-positive worker count.
        """
        with pytest.raises(ValidationError, match="num_workers must be >= 1"):
            SkypilotLaunchConfig(num_workers=bad)

    @pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
    def test_blank_api_server_rejected(self, blank: str) -> None:
        """Empty or whitespace-only api_server is rejected to surface typos loudly.

        :param blank: Parametrized blank/whitespace-only api_server value.
        """
        with pytest.raises(ValidationError, match="api_server must be a non-empty URL"):
            SkypilotLaunchConfig(api_server=blank)

    def test_api_server_is_stripped(self) -> None:
        """Surrounding whitespace is trimmed so the eventual env-export round-trips cleanly."""
        cfg = SkypilotLaunchConfig(api_server="  https://api.example.com  ")
        assert cfg.api_server == "https://api.example.com"

    def test_extra_fields_rejected_naming_the_offender(self) -> None:
        """``extra='forbid'`` catches misspelled Hydra overrides loudly and names the bad field."""
        with pytest.raises(ValidationError, match="compute_templat"):
            SkypilotLaunchConfig(compute_templat="typo.yaml")  # type: ignore[call-arg]

    def test_frozen_after_construction(self) -> None:
        """Trust-boundary models are frozen so dispatch can't mutate the config mid-launch."""
        cfg = SkypilotLaunchConfig()
        with pytest.raises(ValidationError):
            cfg.compute_template = "anything.yaml"  # type: ignore[misc]


class TestSkypilotClientSettings:
    """SkyPilot client auth resolves from dotenv without shell exports."""

    def test_env_file_loads_endpoint_and_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dotenv file overrides stale process auth as one validated pair.

        :param tmp_path: Isolates the dotenv source from developer files.
        :param monkeypatch: Supplies stale process auth and restores it after the test.
        """
        monkeypatch.setenv(ENV_SKYPILOT_API_SERVER_ENDPOINT, "https://stale.example.com")
        monkeypatch.delenv(ENV_SKYPILOT_SERVICE_ACCOUNT_TOKEN, raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text(
            f"{ENV_SKYPILOT_API_SERVER_ENDPOINT}=https://sky.example.com\n"
            f"{ENV_SKYPILOT_SERVICE_ACCOUNT_TOKEN}=sky_test-token\n"
        )

        settings = skypilot_client_settings_from_sources(env_file)

        assert settings.api_server_endpoint == "https://sky.example.com"
        assert settings.service_account_token is not None
        assert settings.service_account_token.get_secret_value() == "sky_test-token"

    def test_config_endpoint_combines_with_env_file_token(self, tmp_path: Path) -> None:
        """An explicit endpoint can pair with a dotenv service token.

        :param tmp_path: Pytest temporary directory.
        """
        env_file = tmp_path / ".env"
        env_file.write_text(f"{ENV_SKYPILOT_SERVICE_ACCOUNT_TOKEN}=sky_test-token\n")

        settings = skypilot_client_settings_from_sources(
            env_file,
            api_server_endpoint="https://config.example.com",
        )

        assert settings.api_server_endpoint == "https://config.example.com"
        assert settings.service_account_token is not None

    def test_invalid_config_endpoint_is_rejected(self, tmp_path: Path) -> None:
        """An explicit launcher override still passes through endpoint validation.

        :param tmp_path: Provides an otherwise empty dotenv source.
        """
        env_file = tmp_path / ".env"
        env_file.touch()

        with pytest.raises(ValidationError, match=r"HTTP\(S\) URL"):
            skypilot_client_settings_from_sources(
                env_file,
                api_server_endpoint="not-a-url",
            )

    def test_blank_env_file_value_falls_back_to_process_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Blank dotenv auth does not mask a usable process value.

        :param tmp_path: Pytest temporary directory.
        :param monkeypatch: Pytest environment fixture.
        """
        monkeypatch.setenv(ENV_SKYPILOT_API_SERVER_ENDPOINT, "https://process.example.com")
        env_file = tmp_path / ".env"
        env_file.write_text(f"{ENV_SKYPILOT_API_SERVER_ENDPOINT}=\n")

        settings = skypilot_client_settings_from_sources(env_file)

        assert settings.api_server_endpoint == "https://process.example.com"

    def test_service_account_token_without_endpoint_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A token without its target server fails before dispatch.

        :param monkeypatch: Pytest environment isolation fixture.
        """
        monkeypatch.delenv(ENV_SKYPILOT_API_SERVER_ENDPOINT, raising=False)
        with pytest.raises(ValidationError, match="requires.*API server endpoint"):
            SkypilotClientSettings(service_account_token=SecretStr("sky_orphan-token"))

    def test_invalid_service_account_token_rejected(self) -> None:
        """A malformed token fails before any authenticated request."""
        with pytest.raises(ValidationError, match="must start with 'sky_'"):
            SkypilotClientSettings(
                api_server_endpoint="https://sky.example.com",
                service_account_token=SecretStr("not-a-token"),
            )

    @pytest.mark.parametrize(
        "endpoint",
        ["not-a-url", "ftp://sky.example.com", "https://"],
        ids=["missing-scheme", "unsupported-scheme", "missing-host"],
    )
    def test_invalid_api_server_endpoint_rejected(self, endpoint: str) -> None:
        """Malformed endpoint values fail at the settings boundary.

        :param endpoint: Invalid endpoint candidate.
        """
        with pytest.raises(ValidationError, match="HTTP.*URL"):
            SkypilotClientSettings(api_server_endpoint=endpoint)


class TestModelCopy:
    """``model_copy(update=...)`` is the path the Hydra entrypoint uses to inject ``cmd``."""

    def test_model_copy_with_cmd_yields_new_instance(self) -> None:
        """Frozen + model_copy(update=…) is the only way to set cmd post-construction."""
        original = SkypilotLaunchConfig(compute_template="x.yaml")
        with_cmd = original.model_copy(update={"cmd": "echo hi"})
        assert with_cmd.cmd == "echo hi"
        assert with_cmd.compute_template == "x.yaml"
        # Original is untouched (frozen invariant).
        assert original.cmd is None


class TestExtraEnvs:
    """``extra_envs`` carries caller-supplied per-rank worker env additions."""

    def test_extra_envs_defaults_to_empty_dict(self) -> None:
        """No callers means no merge — the default must not surprise rank env composition."""
        assert SkypilotLaunchConfig().extra_envs == {}

    @pytest.mark.parametrize(
        "bad_key",
        ["lower", "Mixed", "1LEADING_DIGIT", "HAS-DASH", "HAS SPACE", ""],
        ids=["lowercase", "mixed-case", "leading-digit", "dash", "space", "empty"],
    )
    def test_extra_envs_rejects_invalid_identifier_keys(self, bad_key: str) -> None:
        """Keys must match the uppercase env-var grammar so exports round-trip cleanly.

        :param bad_key: Parametrized invalid key the validator must reject.
        """
        with pytest.raises(ValidationError, match="extra_envs keys"):
            SkypilotLaunchConfig(extra_envs={bad_key: "x"})

    @pytest.mark.parametrize(
        "good_key",
        ["FOO", "FOO_BAR", "_LEAD", "A0", "_"],
        ids=["plain", "with-underscore", "leading-underscore", "with-digit", "bare-underscore"],
    )
    def test_extra_envs_accepts_valid_identifier_keys(self, good_key: str) -> None:
        """Anchor the positive grammar so a future regex tightening can't silently drop legal keys.

        :param good_key: Parametrized valid key the validator must accept.
        """
        assert SkypilotLaunchConfig(extra_envs={good_key: "x"}).extra_envs == {good_key: "x"}
