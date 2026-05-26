"""Tests for src/synth_setter/pipeline/schemas/skypilot_launch.py — launcher config schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synth_setter.pipeline.schemas.skypilot_launch import SkypilotLaunchConfig


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
