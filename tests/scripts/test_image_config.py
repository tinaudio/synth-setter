"""Tests for scripts/image_config.py — image creation config schema and loader.

Tests are organized around the PUBLIC typed API:
- load_image_config(): loads YAML config, merges runtime inputs, validates via Pydantic
- ImageConfig: Pydantic model with github_sha, issue_number, image_config_id
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.image_config import load_image_config

VALID_SHA = "a" * 40
VALID_ISSUE = 266


# ---------------------------------------------------------------------------
# load_image_config — valid inputs
# ---------------------------------------------------------------------------


class TestLoadImageConfigValid:
    """load_image_config returns correct ImageConfig for valid inputs."""

    def test_all_fields_populated(self, tmp_path: Path) -> None:
        """Valid YAML + runtime inputs produce ImageConfig with all fields set."""
        config_path = tmp_path / "dev-snapshot.yaml"
        config_path.write_text("# minimal config\n")

        result = load_image_config(
            config_path,
            github_sha=VALID_SHA,
            issue_number=VALID_ISSUE,
        )

        assert result.github_sha == VALID_SHA
        assert result.issue_number == VALID_ISSUE
        assert result.image_config_id == "dev-snapshot"


# ---------------------------------------------------------------------------
# load_image_config — github_sha validation
# ---------------------------------------------------------------------------


class TestGithubShaValidation:
    """github_sha must be exactly 40 lowercase hex characters."""

    def test_short_sha_rejected(self, tmp_path: Path) -> None:
        """SHA shorter than 40 chars is rejected."""
        config_path = tmp_path / "dev-snapshot.yaml"
        config_path.write_text("")

        with pytest.raises(ValidationError, match="github_sha"):
            load_image_config(config_path, github_sha="abc123", issue_number=VALID_ISSUE)

    def test_uppercase_sha_rejected(self, tmp_path: Path) -> None:
        """Uppercase hex chars are rejected — must be lowercase."""
        config_path = tmp_path / "dev-snapshot.yaml"
        config_path.write_text("")

        with pytest.raises(ValidationError, match="github_sha"):
            load_image_config(config_path, github_sha="A" * 40, issue_number=VALID_ISSUE)

    def test_non_hex_sha_rejected(self, tmp_path: Path) -> None:
        """Non-hex characters are rejected."""
        config_path = tmp_path / "dev-snapshot.yaml"
        config_path.write_text("")

        with pytest.raises(ValidationError, match="github_sha"):
            load_image_config(config_path, github_sha="g" * 40, issue_number=VALID_ISSUE)

    def test_empty_sha_rejected(self, tmp_path: Path) -> None:
        """Empty string is rejected."""
        config_path = tmp_path / "dev-snapshot.yaml"
        config_path.write_text("")

        with pytest.raises(ValidationError, match="github_sha"):
            load_image_config(config_path, github_sha="", issue_number=VALID_ISSUE)


# ---------------------------------------------------------------------------
# load_image_config — issue_number validation
# ---------------------------------------------------------------------------


class TestIssueNumberValidation:
    """issue_number must be a positive integer."""

    def test_zero_rejected(self, tmp_path: Path) -> None:
        """Zero is not a valid issue number."""
        config_path = tmp_path / "dev-snapshot.yaml"
        config_path.write_text("")

        with pytest.raises(ValidationError, match="issue_number"):
            load_image_config(config_path, github_sha=VALID_SHA, issue_number=0)

    def test_negative_rejected(self, tmp_path: Path) -> None:
        """Negative numbers are not valid issue numbers."""
        config_path = tmp_path / "dev-snapshot.yaml"
        config_path.write_text("")

        with pytest.raises(ValidationError, match="issue_number"):
            load_image_config(config_path, github_sha=VALID_SHA, issue_number=-1)


# ---------------------------------------------------------------------------
# load_image_config — image_config_id derivation
# ---------------------------------------------------------------------------


class TestImageConfigIdDerivation:
    """image_config_id is derived from the config filename stem."""

    def test_dev_snapshot_yaml_gives_dev_snapshot_id(self, tmp_path: Path) -> None:
        """dev-snapshot.yaml produces image_config_id 'dev-snapshot'."""
        config_path = tmp_path / "dev-snapshot.yaml"
        config_path.write_text("")

        result = load_image_config(config_path, github_sha=VALID_SHA, issue_number=VALID_ISSUE)

        assert result.image_config_id == "dev-snapshot"

    def test_custom_name_gives_matching_id(self, tmp_path: Path) -> None:
        """Arbitrary filename stem becomes image_config_id."""
        config_path = tmp_path / "my-custom-image.yaml"
        config_path.write_text("")

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
