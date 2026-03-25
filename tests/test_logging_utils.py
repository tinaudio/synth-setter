"""Tests for log_wandb_provenance() in src/utils/logging_utils.py.

Uses fakes (not mocks) for wandb, real subprocess where possible, and state assertions throughout.
See python-testing.md §Fakes.
"""

import os
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.utils.logging_utils import log_wandb_provenance

# ---------------------------------------------------------------------------
# Fake wandb module — captures config updates as inspectable state
# ---------------------------------------------------------------------------


class FakeWandbConfig:
    """Fake wandb.config that stores updates for state testing."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def update(self, d: dict, **kwargs: object) -> None:
        self.data.update(d)


def make_fake_wandb(*, has_run: bool = True) -> SimpleNamespace:
    """Factory for a fake wandb module with inspectable config state."""
    return SimpleNamespace(
        run=object() if has_run else None,
        config=FakeWandbConfig(),
        __spec__=object(),
    )


# ---------------------------------------------------------------------------
# Happy-path behavior tests (real subprocess, fake wandb only)
# ---------------------------------------------------------------------------


class TestLogWandbProvenanceHappyPath:
    """Provenance fields are logged correctly when all dependencies are available."""

    def test_logs_git_sha_as_valid_hex(self) -> None:
        """github_sha is the real 40-char hex SHA from the current git repo."""
        fake = make_fake_wandb()

        with patch.dict("sys.modules", {"wandb": fake}):
            log_wandb_provenance()

        sha = fake.config.data["github_sha"]
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)

    def test_logs_image_tag_from_env(self) -> None:
        """image_tag matches the IMAGE_TAG environment variable."""
        fake = make_fake_wandb()

        with (
            patch.dict("sys.modules", {"wandb": fake}),
            patch.dict(os.environ, {"IMAGE_TAG": "v1.2.3"}),
        ):
            log_wandb_provenance()

        assert fake.config.data["image_tag"] == "v1.2.3"

    def test_logs_command_from_argv(self) -> None:
        """Command is a non-empty string derived from sys.argv."""
        fake = make_fake_wandb()

        with patch.dict("sys.modules", {"wandb": fake}):
            log_wandb_provenance()

        assert isinstance(fake.config.data["command"], str)
        assert len(fake.config.data["command"]) > 0


# ---------------------------------------------------------------------------
# Fallback behavior tests
# ---------------------------------------------------------------------------


class TestLogWandbProvenanceFallbacks:
    """Graceful fallbacks when git or IMAGE_TAG are unavailable."""

    @pytest.mark.parametrize(
        "error",
        [
            FileNotFoundError("git not found"),
            subprocess.CalledProcessError(128, "git"),
        ],
        ids=["git_not_installed", "not_a_git_repo"],
    )
    def test_git_sha_unknown_on_subprocess_error(self, error: Exception) -> None:
        """github_sha falls back to 'unknown' when git rev-parse fails."""
        fake = make_fake_wandb()

        with (
            patch.dict("sys.modules", {"wandb": fake}),
            patch(
                "src.utils.logging_utils.subprocess.check_output",
                side_effect=error,
            ),
        ):
            log_wandb_provenance()

        assert fake.config.data["github_sha"] == "unknown"

    def test_image_tag_unknown_when_env_unset(self) -> None:
        """image_tag falls back to 'unknown' when IMAGE_TAG is absent."""
        fake = make_fake_wandb()
        env_without_image_tag = {k: v for k, v in os.environ.items() if k != "IMAGE_TAG"}

        with (
            patch.dict("sys.modules", {"wandb": fake}),
            patch.dict(os.environ, env_without_image_tag, clear=True),
        ):
            log_wandb_provenance()

        assert fake.config.data["image_tag"] == "unknown"


# ---------------------------------------------------------------------------
# Guard clause tests
# ---------------------------------------------------------------------------


class TestLogWandbProvenanceGuards:
    """Safe noop when wandb is unavailable or no run is active."""

    def test_noop_when_wandb_not_installed(self) -> None:
        """No crash when wandb is not installed."""
        with patch.dict("sys.modules", {"wandb": None}):
            log_wandb_provenance()

    def test_noop_when_no_active_run(self) -> None:
        """No config update when wandb.run is None."""
        fake = make_fake_wandb(has_run=False)

        with patch.dict("sys.modules", {"wandb": fake}):
            log_wandb_provenance()

        assert fake.config.data == {}
