"""Unit tests for scripts/runpod_stop.py.

Tests the emergency stop utility for terminating RunPod pods. All RunPod
API calls are mocked — no real pods are affected.

To run:
    pytest tests/test_runpod_stop.py -v
"""

from unittest.mock import MagicMock, call, patch

import pytest
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts.runpod_stop import _stop_pods

# ---------------------------------------------------------------------------
# Tests — _stop_pods (pure logic, RunPod API mocked at boundary)
# ---------------------------------------------------------------------------


class TestStopPods:
    """Tests for pod filtering and termination."""

    def test_terminates_matching_pods(self):
        """Pods whose name starts with prefix are terminated."""
        pods = [
            {"id": "pod-1", "name": "shardgen-run123-000", "desiredStatus": "RUNNING"},
            {"id": "pod-2", "name": "shardgen-run123-001", "desiredStatus": "RUNNING"},
            {"id": "pod-3", "name": "other-pod", "desiredStatus": "RUNNING"},
        ]
        with patch("scripts.runpod_stop.runpod") as mock_runpod:
            mock_runpod.get_pods.return_value = pods
            mock_runpod.terminate_pod = MagicMock()

            count = _stop_pods("shardgen-run123")

        assert count == 2
        mock_runpod.terminate_pod.assert_any_call("pod-1")
        mock_runpod.terminate_pod.assert_any_call("pod-2")
        assert mock_runpod.terminate_pod.call_count == 2

    def test_no_matching_pods_returns_zero(self):
        """Returns 0 when no pods match the prefix."""
        pods = [
            {"id": "pod-1", "name": "other-pod", "desiredStatus": "RUNNING"},
        ]
        with patch("scripts.runpod_stop.runpod") as mock_runpod:
            mock_runpod.get_pods.return_value = pods

            count = _stop_pods("shardgen-run999")

        assert count == 0

    def test_all_prefix_matches_all_shardgen_pods(self):
        """Prefix 'shardgen-' matches all shardgen pods regardless of run ID."""
        pods = [
            {"id": "pod-1", "name": "shardgen-runA-000", "desiredStatus": "RUNNING"},
            {"id": "pod-2", "name": "shardgen-runB-000", "desiredStatus": "RUNNING"},
            {"id": "pod-3", "name": "training-pod", "desiredStatus": "RUNNING"},
        ]
        with patch("scripts.runpod_stop.runpod") as mock_runpod:
            mock_runpod.get_pods.return_value = pods
            mock_runpod.terminate_pod = MagicMock()

            count = _stop_pods("shardgen-")

        assert count == 2


# ---------------------------------------------------------------------------
# Tests — CLI
# ---------------------------------------------------------------------------


class TestRunpodStopCLI:
    """Tests for the Click CLI entry point."""

    def test_missing_api_key_fails(self, monkeypatch):
        """CLI exits with error when RUNPOD_API_KEY is not set."""
        from click.testing import CliRunner

        from scripts.runpod_stop import main

        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        runner = CliRunner()
        with patch("scripts.runpod_stop.runpod", MagicMock()):
            result = runner.invoke(main, ["--run-id", "test123"])
        assert result.exit_code != 0
        assert "RUNPOD_API_KEY" in result.output

    def test_requires_run_id_or_all(self, monkeypatch):
        """CLI fails when neither --run-id nor --all is provided."""
        from click.testing import CliRunner

        from scripts.runpod_stop import main

        monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
        runner = CliRunner()
        with patch("scripts.runpod_stop.runpod", MagicMock()):
            result = runner.invoke(main, [])
        assert result.exit_code != 0

    def test_run_id_filters_by_prefix(self, monkeypatch):
        """--run-id filters pods by shardgen-<run_id[:12]> prefix."""
        from click.testing import CliRunner

        from scripts.runpod_stop import main

        monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
        pods = [
            {"id": "pod-1", "name": "shardgen-20260310-143-000", "desiredStatus": "RUNNING"},
        ]
        with patch("scripts.runpod_stop.runpod") as mock_runpod:
            mock_runpod.get_pods.return_value = pods
            mock_runpod.terminate_pod = MagicMock()

            runner = CliRunner()
            result = runner.invoke(main, ["--run-id", "20260310-143022-a3f2b1"])

        assert result.exit_code == 0, result.output
        mock_runpod.terminate_pod.assert_called_once_with("pod-1")
