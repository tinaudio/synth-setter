"""Tests for pipeline.entrypoints.verify_skypilot_env.

The verify script is a deployment guard: it confirms the worker is actually
running under SkyPilot (which sets SKYPILOT_NODE_RANK / SKYPILOT_NUM_NODES)
before generate_dataset is invoked. Failure must be loud — non-zero exit and
a clear error to stderr — so a misconfigured launch fails fast instead of
silently rendering every shard from every node.
"""

from __future__ import annotations

import pytest

from pipeline.entrypoints.verify_skypilot_env import main, verify_env


@pytest.fixture(autouse=True)
def _clear_skypilot_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip SkyPilot rank/world env vars from the test process for isolation."""
    monkeypatch.delenv("SKYPILOT_NODE_RANK", raising=False)
    monkeypatch.delenv("SKYPILOT_NUM_NODES", raising=False)


class TestVerifyEnvSuccess:
    """Valid env → verify_env returns None (no exception)."""

    def test_returns_none_when_rank_and_world_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Single-worker valid env (rank=0, world=1) verifies successfully."""
        monkeypatch.setenv("SKYPILOT_NODE_RANK", "0")
        monkeypatch.setenv("SKYPILOT_NUM_NODES", "1")
        assert verify_env() is None

    def test_returns_none_when_multi_node_rank_in_range(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multi-node rank within bounds (rank=3, world=4) verifies successfully."""
        monkeypatch.setenv("SKYPILOT_NODE_RANK", "3")
        monkeypatch.setenv("SKYPILOT_NUM_NODES", "4")
        assert verify_env() is None


class TestVerifyEnvMissing:
    """Missing env vars → ValueError naming each missing var."""

    def test_both_missing_raises_with_both_names_in_message(self) -> None:
        """Both env vars missing → error message names both."""
        with pytest.raises(ValueError) as excinfo:
            verify_env()
        message = str(excinfo.value)
        assert "SKYPILOT_NODE_RANK" in message
        assert "SKYPILOT_NUM_NODES" in message

    def test_rank_missing_raises_with_rank_name_in_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only SKYPILOT_NODE_RANK missing → error names that var."""
        monkeypatch.setenv("SKYPILOT_NUM_NODES", "1")
        with pytest.raises(ValueError, match="SKYPILOT_NODE_RANK"):
            verify_env()

    def test_world_missing_raises_with_world_name_in_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only SKYPILOT_NUM_NODES missing → error names that var."""
        monkeypatch.setenv("SKYPILOT_NODE_RANK", "0")
        with pytest.raises(ValueError, match="SKYPILOT_NUM_NODES"):
            verify_env()


class TestVerifyEnvInvalidValues:
    """Set-but-invalid values fail loudly (no silent defaulting)."""

    def test_non_integer_rank_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-integer rank ('abc') → ValueError naming SKYPILOT_NODE_RANK."""
        monkeypatch.setenv("SKYPILOT_NODE_RANK", "abc")
        monkeypatch.setenv("SKYPILOT_NUM_NODES", "1")
        with pytest.raises(ValueError, match="SKYPILOT_NODE_RANK"):
            verify_env()

    def test_non_integer_world_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-integer world ('xyz') → ValueError naming SKYPILOT_NUM_NODES."""
        monkeypatch.setenv("SKYPILOT_NODE_RANK", "0")
        monkeypatch.setenv("SKYPILOT_NUM_NODES", "xyz")
        with pytest.raises(ValueError, match="SKYPILOT_NUM_NODES"):
            verify_env()

    def test_world_zero_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """World=0 violates the world>=1 invariant and raises ValueError."""
        monkeypatch.setenv("SKYPILOT_NODE_RANK", "0")
        monkeypatch.setenv("SKYPILOT_NUM_NODES", "0")
        with pytest.raises(ValueError, match="world=0"):
            verify_env()

    def test_rank_equal_to_world_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rank == world is out-of-range and raises ValueError."""
        monkeypatch.setenv("SKYPILOT_NODE_RANK", "2")
        monkeypatch.setenv("SKYPILOT_NUM_NODES", "2")
        with pytest.raises(ValueError, match="rank=2"):
            verify_env()

    def test_negative_rank_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Negative rank raises ValueError naming the rank value."""
        monkeypatch.setenv("SKYPILOT_NODE_RANK", "-1")
        monkeypatch.setenv("SKYPILOT_NUM_NODES", "2")
        with pytest.raises(ValueError, match="rank=-1"):
            verify_env()


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
