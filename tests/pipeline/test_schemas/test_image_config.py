"""Tests for pipeline/schemas/image_config.py — image creation config schema and loader.

Tests are organized around the PUBLIC typed API:
- load_image_config(): loads YAML config, merges runtime inputs, validates via Pydantic
- ImageConfig: Pydantic model with github_sha, issue_number, image_config_id
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline.schemas.image_config import load_image_config

VALID_SHA = "a" * 40
VALID_ISSUE = 266

_COMPLETE_YAML = """\
dockerfile: docker/ubuntu22_04/Dockerfile
image: tinaudio/perm
base_image: "ubuntu@sha256:3ba65aa20f86a0fad9df2b2c259c613df006b2e6d0bfcc8a146afb8c525a9751"
base_image_tag: ubuntu22_04
build_mode: prebuilt
target_platform: linux/amd64
torch_backend: "cu128"
r2_bucket: test-bucket
"""


def _write_config(tmp_path: Path, overrides: str = "") -> Path:
    """Write a complete YAML config and return its path.

    If *overrides* is provided it is appended after the base config, allowing individual tests to
    add or shadow fields.
    """
    config_path = tmp_path / "dev-snapshot.yaml"
    config_path.write_text(_COMPLETE_YAML + overrides)
    return config_path


# ---------------------------------------------------------------------------
# load_image_config — valid inputs
# ---------------------------------------------------------------------------


class TestLoadImageConfigValid:
    """load_image_config returns correct ImageConfig for valid inputs."""

    def test_all_fields_populated(self, tmp_path: Path) -> None:
        # plumb:req-74aa845b
        # plumb:req-c8e1d2e0
        # plumb:req-788d4dac
        """Valid YAML + runtime inputs produce ImageConfig with all fields set."""
        config_path = _write_config(tmp_path)

        result = load_image_config(
            config_path,
            github_sha=VALID_SHA,
            issue_number=VALID_ISSUE,
        )

        assert result.github_sha == VALID_SHA
        assert result.issue_number == VALID_ISSUE
        assert result.image_config_id == "dev-snapshot"
        assert result.r2_bucket == "test-bucket"


# ---------------------------------------------------------------------------
# load_image_config — github_sha validation
# ---------------------------------------------------------------------------


class TestGithubShaValidation:
    """github_sha must be exactly 40 lowercase hex characters."""

    def test_short_sha_rejected(self, tmp_path: Path) -> None:
        # plumb:req-13e3f802
        """SHA shorter than 40 chars is rejected."""
        config_path = _write_config(tmp_path)

        with pytest.raises(ValidationError, match="github_sha"):
            load_image_config(config_path, github_sha="abc123", issue_number=VALID_ISSUE)

    def test_uppercase_sha_rejected(self, tmp_path: Path) -> None:
        """Uppercase hex chars are rejected — must be lowercase."""
        config_path = _write_config(tmp_path)

        with pytest.raises(ValidationError, match="github_sha"):
            load_image_config(config_path, github_sha="A" * 40, issue_number=VALID_ISSUE)

    def test_non_hex_sha_rejected(self, tmp_path: Path) -> None:
        """Non-hex characters are rejected."""
        config_path = _write_config(tmp_path)

        with pytest.raises(ValidationError, match="github_sha"):
            load_image_config(config_path, github_sha="g" * 40, issue_number=VALID_ISSUE)

    def test_empty_sha_rejected(self, tmp_path: Path) -> None:
        """Empty string is rejected."""
        config_path = _write_config(tmp_path)

        with pytest.raises(ValidationError, match="github_sha"):
            load_image_config(config_path, github_sha="", issue_number=VALID_ISSUE)


# ---------------------------------------------------------------------------
# load_image_config — issue_number validation
# ---------------------------------------------------------------------------


class TestIssueNumberValidation:
    """issue_number must be a positive integer."""

    def test_zero_rejected(self, tmp_path: Path) -> None:
        """Zero is not a valid issue number."""
        config_path = _write_config(tmp_path)

        with pytest.raises(ValidationError, match="issue_number"):
            load_image_config(config_path, github_sha=VALID_SHA, issue_number=0)

    def test_negative_rejected(self, tmp_path: Path) -> None:
        """Negative numbers are not valid issue numbers."""
        config_path = _write_config(tmp_path)

        with pytest.raises(ValidationError, match="issue_number"):
            load_image_config(config_path, github_sha=VALID_SHA, issue_number=-1)


# ---------------------------------------------------------------------------
# load_image_config — r2 field validation
# ---------------------------------------------------------------------------


class TestR2BucketValidation:
    """r2_bucket must not be empty or whitespace-only."""

    def test_empty_r2_bucket_rejected(self, tmp_path: Path) -> None:
        """Empty r2_bucket is rejected."""
        config_path = _write_config(tmp_path, overrides='r2_bucket: ""\n')

        with pytest.raises(ValidationError, match="must not be blank"):
            load_image_config(config_path, github_sha=VALID_SHA, issue_number=VALID_ISSUE)


# ---------------------------------------------------------------------------
# load_image_config — image_config_id derivation
# ---------------------------------------------------------------------------


class TestImageConfigIdDerivation:
    """image_config_id is derived from the config filename stem."""

    def test_dev_snapshot_yaml_gives_dev_snapshot_id(self, tmp_path: Path) -> None:
        # plumb:req-bf6287eb
        """dev-snapshot.yaml produces image_config_id 'dev-snapshot'."""
        config_path = _write_config(tmp_path)

        result = load_image_config(config_path, github_sha=VALID_SHA, issue_number=VALID_ISSUE)

        assert result.image_config_id == "dev-snapshot"

    def test_custom_name_gives_matching_id(self, tmp_path: Path) -> None:
        # plumb:req-fc5acded
        """Arbitrary filename stem becomes image_config_id."""
        config_path = tmp_path / "my-custom-image.yaml"
        config_path.write_text(_COMPLETE_YAML)

        result = load_image_config(config_path, github_sha=VALID_SHA, issue_number=VALID_ISSUE)

        assert result.image_config_id == "my-custom-image"


# ---------------------------------------------------------------------------
# load_image_config — error cases
# ---------------------------------------------------------------------------


class TestLoadImageConfigErrors:
    """load_image_config raises appropriate errors for bad inputs."""

    def test_nonexistent_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """Missing config file raises FileNotFoundError."""
        config_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            load_image_config(config_path, github_sha=VALID_SHA, issue_number=VALID_ISSUE)

    def test_is_file_rejects_directory(self, tmp_path: Path) -> None:
        """Directory path raises FileNotFoundError, not IsADirectoryError."""
        dir_path = tmp_path / "not-a-file"
        dir_path.mkdir()

        with pytest.raises(FileNotFoundError):
            load_image_config(dir_path, github_sha=VALID_SHA, issue_number=VALID_ISSUE)

    def test_yaml_non_mapping_raises(self, tmp_path: Path) -> None:
        # plumb:req-3cfbf391
        """YAML with a list instead of a mapping raises ValueError."""
        config_path = tmp_path / "bad.yaml"
        config_path.write_text("[1, 2, 3]\n")

        with pytest.raises(ValueError, match="must be a mapping"):
            load_image_config(config_path, github_sha=VALID_SHA, issue_number=VALID_ISSUE)

    def test_yaml_unknown_key_rejected(self, tmp_path: Path) -> None:
        """Unknown YAML key is rejected by Pydantic strict mode."""
        config_path = tmp_path / "unknown.yaml"
        config_path.write_text(_COMPLETE_YAML + "bogus_key: value\n")

        with pytest.raises(ValidationError):
            load_image_config(config_path, github_sha=VALID_SHA, issue_number=VALID_ISSUE)

    def test_yaml_empty_file_raises_validation_error(self, tmp_path: Path) -> None:
        """Empty YAML (comment-only) raises ValidationError since no fields have defaults."""
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("# just a comment\n")

        with pytest.raises(ValidationError):
            load_image_config(config_path, github_sha=VALID_SHA, issue_number=VALID_ISSUE)

    def test_missing_field_rejected(self, tmp_path: Path) -> None:
        """YAML missing a required field raises ValidationError."""
        config_path = tmp_path / "incomplete.yaml"
        # Write config missing r2_bucket
        config_path.write_text(
            "dockerfile: docker/ubuntu22_04/Dockerfile\n"
            "image: tinaudio/perm\n"
            'base_image: "ubuntu@sha256:3ba65aa20f86a0fad9df2b2c259c613df006b2e6d0bfcc8a146afb8c525a9751"\n'
            "base_image_tag: ubuntu22_04\n"
            "build_mode: prebuilt\n"
            "target_platform: linux/amd64\n"
            'torch_backend: "cu128"\n'
        )

        with pytest.raises(ValidationError):
            load_image_config(config_path, github_sha=VALID_SHA, issue_number=VALID_ISSUE)


# ---------------------------------------------------------------------------
# load_image_config — static fields and YAML merge
# ---------------------------------------------------------------------------


class TestStaticFieldsAndYamlMerge:
    """Static fields are loaded from YAML and merged with runtime inputs."""

    def test_yaml_last_key_wins_for_build_mode(self, tmp_path: Path) -> None:
        # plumb:req-c63cd79a
        """YAML value for build_mode is loaded correctly."""
        config_path = _write_config(tmp_path, overrides="build_mode: source\n")

        result = load_image_config(config_path, github_sha=VALID_SHA, issue_number=VALID_ISSUE)

        assert result.build_mode == "source"

    def test_build_mode_rejects_invalid_literal(self, tmp_path: Path) -> None:
        """build_mode only accepts 'source' or 'prebuilt'."""
        config_path = _write_config(tmp_path, overrides="build_mode: invalid\n")

        with pytest.raises(ValidationError, match="build_mode"):
            load_image_config(config_path, github_sha=VALID_SHA, issue_number=VALID_ISSUE)

    def test_target_platform_rejects_invalid_literal(self, tmp_path: Path) -> None:
        """target_platform only accepts 'linux/amd64' or 'linux/arm64'."""
        config_path = _write_config(tmp_path, overrides="target_platform: windows/amd64\n")

        with pytest.raises(ValidationError, match="target_platform"):
            load_image_config(config_path, github_sha=VALID_SHA, issue_number=VALID_ISSUE)

    def test_static_field_values_match_dev_snapshot_yaml(self) -> None:
        # plumb:req-758c3c4b
        # plumb:req-7be88525
        """Real dev-snapshot.yaml fields match expected defaults (catches drift)."""
        config_path = Path("configs/image/dev-snapshot.yaml")

        result = load_image_config(config_path, github_sha=VALID_SHA, issue_number=VALID_ISSUE)

        assert result.dockerfile == "docker/ubuntu22_04/Dockerfile"
        assert result.image == "tinaudio/perm"
        assert result.base_image == (
            "ubuntu@sha256:3ba65aa20f86a0fad9df2b2c259c613df006b2e6d0bfcc8a146afb8c525a9751"
        )
        assert result.base_image_tag == "ubuntu22_04"
        assert result.build_mode == "prebuilt"
        assert result.target_platform == "linux/amd64"
        assert result.torch_backend == "cu128"
        assert result.r2_bucket == "intermediate-data"
        assert result.image_config_id == "dev-snapshot"
