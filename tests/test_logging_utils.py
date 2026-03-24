"""Tests for log_wandb_provenance() in src/utils/logging_utils.py."""

import os
import subprocess
from unittest.mock import MagicMock, patch


class TestLogWandbProvenance:
    """Tests for log_wandb_provenance — wandb config helper."""

    def test_log_wandb_provenance_logs_all_fields(self):
        """All three provenance fields are written to wandb.config."""
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()  # truthy = active run

        mock_sp = MagicMock()
        mock_sp.check_output.return_value = b"abc123def\n"
        mock_sp.CalledProcessError = subprocess.CalledProcessError
        mock_sp.DEVNULL = subprocess.DEVNULL

        with (
            patch("src.utils.logging_utils.find_spec", return_value=True),
            patch.dict("sys.modules", {"wandb": mock_wandb}),
            patch("src.utils.logging_utils.subprocess", mock_sp),
            patch.dict(os.environ, {"IMAGE_TAG": "v1.2.3"}, clear=False),
        ):
            from src.utils.logging_utils import log_wandb_provenance

            log_wandb_provenance()

        mock_wandb.config.update.assert_called_once()
        call_args = mock_wandb.config.update.call_args[0][0]
        assert call_args["github_sha"] == "abc123def"
        assert call_args["image_tag"] == "v1.2.3"
        assert isinstance(call_args["command"], str)

    def test_log_wandb_provenance_no_wandb_is_noop(self):
        """When wandb is not installed, function returns without error."""
        with patch("src.utils.logging_utils.find_spec", return_value=None):
            from src.utils.logging_utils import log_wandb_provenance

            # Should not raise
            log_wandb_provenance()

    def test_log_wandb_provenance_no_active_run_is_noop(self):
        """When wandb.run is None, config.update is not called."""
        mock_wandb = MagicMock()
        mock_wandb.run = None  # no active run

        with (
            patch("src.utils.logging_utils.find_spec", return_value=True),
            patch.dict("sys.modules", {"wandb": mock_wandb}),
        ):
            from src.utils.logging_utils import log_wandb_provenance

            log_wandb_provenance()

        mock_wandb.config.update.assert_not_called()

    def test_log_wandb_provenance_git_failure_uses_unknown(self):
        """When git rev-parse fails, github_sha falls back to 'unknown'."""
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        mock_sp = MagicMock()
        mock_sp.check_output.side_effect = subprocess.CalledProcessError(1, "git")
        mock_sp.CalledProcessError = subprocess.CalledProcessError
        mock_sp.FileNotFoundError = FileNotFoundError
        mock_sp.DEVNULL = subprocess.DEVNULL

        with (
            patch("src.utils.logging_utils.find_spec", return_value=True),
            patch.dict("sys.modules", {"wandb": mock_wandb}),
            patch("src.utils.logging_utils.subprocess", mock_sp),
            patch.dict(os.environ, {"IMAGE_TAG": "v1.0.0"}, clear=False),
        ):
            from src.utils.logging_utils import log_wandb_provenance

            log_wandb_provenance()

        call_args = mock_wandb.config.update.call_args[0][0]
        assert call_args["github_sha"] == "unknown"

    def test_log_wandb_provenance_missing_image_tag_uses_unknown(self):
        """When IMAGE_TAG env var is not set, image_tag falls back to 'unknown'."""
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        mock_sp = MagicMock()
        mock_sp.check_output.return_value = b"deadbeef\n"
        mock_sp.CalledProcessError = subprocess.CalledProcessError
        mock_sp.DEVNULL = subprocess.DEVNULL

        # Ensure IMAGE_TAG is not in the environment
        env_copy = {k: v for k, v in os.environ.items() if k != "IMAGE_TAG"}

        with (
            patch("src.utils.logging_utils.find_spec", return_value=True),
            patch.dict("sys.modules", {"wandb": mock_wandb}),
            patch("src.utils.logging_utils.subprocess", mock_sp),
            patch.dict(os.environ, env_copy, clear=True),
        ):
            from src.utils.logging_utils import log_wandb_provenance

            log_wandb_provenance()

        call_args = mock_wandb.config.update.call_args[0][0]
        assert call_args["image_tag"] == "unknown"
