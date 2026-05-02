"""Tests for pipeline.entrypoints.verify_skypilot_env.

The verify script is a deployment guard: it confirms the worker is actually
running under SkyPilot (which sets SKYPILOT_NODE_RANK / SKYPILOT_NUM_NODES)
before generate_dataset is invoked. Failure must be loud — non-zero exit and
a clear error to stderr — so a misconfigured launch fails fast instead of
silently rendering every shard from every node.

Env-reading and bounds-checking logic lives in
``pipeline.partitioning.read_rank_world_from_env`` and is covered exhaustively
in ``tests/pipeline/test_partitioning.py``. These tests focus on the script's
own contract: exit code, stderr message, success-path INFO log.
"""

from __future__ import annotations

import pytest

from pipeline.entrypoints.verify_skypilot_env import main


@pytest.fixture(autouse=True)
def _clear_skypilot_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip SkyPilot rank/world env vars from the test process for isolation."""
    monkeypatch.delenv("SKYPILOT_NODE_RANK", raising=False)
    monkeypatch.delenv("SKYPILOT_NUM_NODES", raising=False)


class TestMainExitCode:
    """Main() is the script entrypoint — must exit non-zero on failure so the SkyPilot YAML's `set
    -e` short-circuits before generate_dataset runs."""

    def test_exits_zero_on_valid_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Valid env → main() returns None (exit 0)."""
        monkeypatch.setenv("SKYPILOT_NODE_RANK", "0")
        monkeypatch.setenv("SKYPILOT_NUM_NODES", "1")
        assert main() is None

    def test_exits_non_zero_when_env_missing(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Missing env → SystemExit with non-zero code; stderr names both vars."""
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code != 0
        stderr = capsys.readouterr().err
        assert "SKYPILOT_NODE_RANK" in stderr
        assert "SKYPILOT_NUM_NODES" in stderr

    def test_exits_non_zero_on_invalid_rank_world(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Out-of-range rank → SystemExit non-zero; stderr names the offending rank."""
        monkeypatch.setenv("SKYPILOT_NODE_RANK", "5")
        monkeypatch.setenv("SKYPILOT_NUM_NODES", "2")
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code != 0
        assert "rank=5" in capsys.readouterr().err

    def test_exits_non_zero_on_non_integer_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Non-integer env value → SystemExit non-zero; stderr names the offending var."""
        monkeypatch.setenv("SKYPILOT_NODE_RANK", "abc")
        monkeypatch.setenv("SKYPILOT_NUM_NODES", "1")
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code != 0
        assert "SKYPILOT_NODE_RANK" in capsys.readouterr().err
