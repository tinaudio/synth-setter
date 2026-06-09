"""Invariant: CLAUDE.md and GEMINI.md are generated from AGENTS.md.

Runs the generator script in check mode to verify no drifts exist between the
canonical AGENTS.md and the generated CLAUDE.md and GEMINI.md instruction files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.infra
def test_agent_instructions_are_in_sync(project_root: Path) -> None:
    """Run the instruction generator in check mode to assert no drift.

    :param project_root: Path to the repository root.
    """
    script_path = project_root / "scripts" / "ci" / "generate_instructions.py"

    # Run the generation script with --check flag
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(script_path), "--check"],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        check=False,
    )

    assert result.returncode == 0, (
        f"Agent instruction files are out of sync or drift detected!\n"
        f"Run `python3 scripts/ci/generate_instructions.py` to regenerate them.\n"
        f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
    )
