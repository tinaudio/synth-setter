"""Contract tests for project-local Pi subagent policy."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_explore_agent_disabled_to_prevent_unbounded_foreground_runs() -> None:
    """Keep Pi searches in the parent process where tool calls are interruptible."""
    policy_path = REPO_ROOT / ".pi" / "agents" / "Explore.md"

    policy = policy_path.read_text()
    _, frontmatter, guidance = policy.split("---", maxsplit=2)

    assert yaml.safe_load(frontmatter)["enabled"] is False
    assert "direct read, grep, find, ls, or bash tools" in guidance
