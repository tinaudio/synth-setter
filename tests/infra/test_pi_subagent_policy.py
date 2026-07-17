"""Contract tests for project-local Pi subagent policy."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_pi_subagent_registry_explore_override_excludes_explore_type(
    project_root: Path, tmp_path: Path
) -> None:
    """Keep Explore unavailable when Pi loads this project's agent override.

    :param project_root: Repository root containing the policy override.
    :param tmp_path: Empty Pi config directory for isolated extension loading.
    """
    node = shutil.which("node")
    package_root = (
        Path(os.environ.get("PI_CODING_AGENT_DIR", Path.home() / ".pi" / "agent"))
        / "npm"
        / "node_modules"
        / "@tintinweb"
        / "pi-subagents"
    )
    if node is None or not package_root.is_dir():
        pytest.skip("requires the installed @tintinweb/pi-subagents package and Node.js")

    script = """
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const packageRoot = process.env.PI_SUBAGENTS_PACKAGE_ROOT;
const { loadCustomAgents } = await import(
  pathToFileURL(join(packageRoot, "dist", "custom-agents.js")).href,
);
const { getAgentConfig, getAvailableTypes, registerAgents } = await import(
  pathToFileURL(join(packageRoot, "dist", "agent-types.js")).href,
);

registerAgents(loadCustomAgents(process.env.PI_POLICY_PROJECT_ROOT));
const explore = getAgentConfig("Explore");
process.stdout.write(JSON.stringify({
  availableTypes: getAvailableTypes(),
  systemPrompt: explore?.systemPrompt,
}));
"""
    result = subprocess.run(  # noqa: S603 — resolved Node executable, fixed argv, no shell
        [node, "--input-type=module", "--eval", script],
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "PI_CODING_AGENT_DIR": str(tmp_path / "pi-agent"),
            "PI_POLICY_PROJECT_ROOT": str(project_root),
            "PI_SUBAGENTS_PACKAGE_ROOT": str(package_root),
        },
        text=True,
    )

    assert result.returncode == 0, result.stderr
    policy = json.loads(result.stdout)
    assert "Explore" not in policy["availableTypes"]
    assert {"general-purpose", "Plan"} <= set(policy["availableTypes"])
    assert policy["systemPrompt"]
