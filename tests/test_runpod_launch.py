"""Unit tests for scripts/runpod_launch.py.

Tests the orchestration logic for launching RunPod pods for parallel shard
generation. All RunPod API calls are mocked — no real pods are created.

To run:
    pytest tests/test_runpod_launch.py -v
"""

import re
from unittest.mock import MagicMock, patch

import pytest
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts.runpod_launch import _make_pod_env, _make_run_id

# ---------------------------------------------------------------------------
# Tests — _make_run_id (pure function, no mocks)
# ---------------------------------------------------------------------------


class TestMakeRunId:
    """Tests for run ID generation."""

    def test_format_matches_timestamp_plus_hex(self):
        """Run ID follows YYYYMMDD-HHMMSS-<6hex> pattern."""
        run_id = _make_run_id()
        assert re.match(r"^\d{8}-\d{6}-[0-9a-f]{6}$", run_id)

    def test_unique_across_calls(self):
        """Two consecutive calls produce different run IDs."""
        assert _make_run_id() != _make_run_id()


# ---------------------------------------------------------------------------
# Tests — _make_pod_env (pure function, no mocks)
# ---------------------------------------------------------------------------


class TestMakePodEnv:
    """Tests for pod environment variable construction."""

    def test_required_keys_present(self):
        """Pod env contains all required keys for MODE=generate-shards."""
        env = _make_pod_env(
            run_id="20260310-143022-a3f2b1",
            shards_per_pod=10,
            shard_size=5000,
            param_spec="surge_simple",
            max_workers=None,
        )
        assert env["MODE"] == "generate-shards"
        assert env["NUM_SHARDS"] == "10"
        assert env["SHARD_SIZE"] == "5000"
        assert env["PARAM_SPEC"] == "surge_simple"
        assert env["R2_PREFIX"] == "runs/20260310-143022-a3f2b1"
        assert env["PARALLEL"] == "1"
        assert env["IDLE_AFTER"] == "0"

    def test_max_workers_included_when_set(self):
        """MAX_WORKERS is added to env when max_workers is not None."""
        env = _make_pod_env(
            run_id="run1",
            shards_per_pod=5,
            shard_size=1000,
            param_spec="surge_xt",
            max_workers=4,
        )
        assert env["MAX_WORKERS"] == "4"

    def test_max_workers_omitted_when_none(self):
        """MAX_WORKERS is not present in env when max_workers is None."""
        env = _make_pod_env(
            run_id="run1",
            shards_per_pod=5,
            shard_size=1000,
            param_spec="surge_simple",
            max_workers=None,
        )
        assert "MAX_WORKERS" not in env

    def test_all_values_are_strings(self):
        """All env values are strings (required by RunPod API)."""
        env = _make_pod_env(
            run_id="run1",
            shards_per_pod=10,
            shard_size=10000,
            param_spec="surge_simple",
            max_workers=8,
        )
        for key, value in env.items():
            assert isinstance(value, str), f"{key} value is {type(value)}, expected str"


# ---------------------------------------------------------------------------
# Tests — CLI (mock runpod at the architectural boundary)
# ---------------------------------------------------------------------------


class TestRunpodLaunchCLI:
    """Tests for the Click CLI entry point."""

    def test_missing_api_key_fails(self, monkeypatch):
        """CLI exits with error when RUNPOD_API_KEY is not set."""
        from click.testing import CliRunner

        from scripts.runpod_launch import main

        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        runner = CliRunner()
        with patch("scripts.runpod_launch.runpod", MagicMock()):
            result = runner.invoke(
                main,
                [
                    "--num-pods",
                    "1",
                    "--shards-per-pod",
                    "1",
                    "--shard-size",
                    "10",
                    "--image",
                    "test:latest",
                ],
            )
        assert result.exit_code != 0
        assert "RUNPOD_API_KEY" in result.output

    def test_dry_run_does_not_create_pods(self, monkeypatch):
        """--dry-run prints plan but does not call runpod.create_pod."""
        from click.testing import CliRunner

        from scripts.runpod_launch import main

        monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")

        mock_create = MagicMock()
        with patch("scripts.runpod_launch.runpod") as mock_runpod:
            mock_runpod.create_pod = mock_create
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--num-pods",
                    "5",
                    "--shards-per-pod",
                    "10",
                    "--shard-size",
                    "1000",
                    "--image",
                    "tinaudio/perm:test",
                    "--dry-run",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "DRY RUN" in result.output
        mock_create.assert_not_called()

    def test_dry_run_shows_total_shards(self, monkeypatch):
        """--dry-run output includes total shard and sample counts."""
        from click.testing import CliRunner

        from scripts.runpod_launch import main

        monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")

        with patch("scripts.runpod_launch.runpod"):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--num-pods",
                    "5",
                    "--shards-per-pod",
                    "10",
                    "--shard-size",
                    "1000",
                    "--image",
                    "tinaudio/perm:test",
                    "--dry-run",
                ],
            )

        assert "50" in result.output  # 5 * 10 = 50 total shards
        assert "50,000" in result.output  # 50 * 1000 = 50,000 total samples

    def test_missing_required_args_fails(self, monkeypatch):
        """CLI fails when required args are missing."""
        from click.testing import CliRunner

        from scripts.runpod_launch import main

        monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
        runner = CliRunner()
        result = runner.invoke(main, ["--num-pods", "1"])
        assert result.exit_code != 0
