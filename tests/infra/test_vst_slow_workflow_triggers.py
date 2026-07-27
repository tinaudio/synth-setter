"""`test-vst-slow.yml` gates push and pull-request events on the same paths.

GitHub Actions rejects YAML anchors, so the workflow repeats its path list once per event. This
pins the two copies together: without it, adding a path to only one trigger silently leaves the
other event ungated (#1354).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from workflow_fixtures import load_workflow

WORKFLOW_FILENAME = "test-vst-slow.yml"


def _load_triggers(project_root: Path) -> dict[str, dict[str, list[str]]]:
    """Return the workflow's event-trigger block.

    :param project_root: Repo root holding ``.github/workflows/``.
    :returns: Trigger mapping keyed by event name.
    """
    workflow = cast(dict[object, object], load_workflow(project_root, WORKFLOW_FILENAME))
    # PyYAML resolves a bare ``on`` key to the boolean ``True`` (YAML 1.1).
    on_key: object = "on" if "on" in workflow else True
    return cast(dict[str, dict[str, list[str]]], workflow[on_key])


@pytest.mark.infra
def test_vst_slow_push_and_pull_request_gate_identical_paths(project_root: Path) -> None:
    """Both events select the same VST-affecting files.

    :param project_root: Repo root holding ``.github/workflows/``.
    """
    triggers = _load_triggers(project_root)

    assert triggers["push"]["paths"] == triggers["pull_request"]["paths"]


@pytest.mark.infra
def test_vst_slow_triggers_declare_no_yaml_anchors(project_root: Path) -> None:
    """Neither trigger deduplicates through an anchor GitHub Actions cannot parse.

    :param project_root: Repo root holding ``.github/workflows/``.
    """
    workflow_text = (project_root / ".github" / "workflows" / WORKFLOW_FILENAME).read_text()

    assert "paths: &" not in workflow_text
    assert "paths: *" not in workflow_text
