"""Shared workflow-YAML loading for the static `tests/infra` workflow assertions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

WORKFLOWS_DIR = Path(".github") / "workflows"


def load_workflow(project_root: Path, workflow_filename: str) -> dict[str, Any]:
    """Parse ``.github/workflows/<workflow_filename>`` into a plain dict.

    :param project_root: Repo root; the workflow lives under ``<project_root>/.github/workflows/``.
    :param workflow_filename: Workflow file name (e.g. ``cpu-slow.yml``), not a path.
    :returns: The parsed YAML document.
    """
    workflow_path = project_root / WORKFLOWS_DIR / workflow_filename
    return yaml.safe_load(workflow_path.read_text())
