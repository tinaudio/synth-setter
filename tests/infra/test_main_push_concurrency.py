"""Contract: a push to `main` must not cancel another `main` commit's run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from workflow_fixtures import WORKFLOWS_DIR, load_workflow

# GitHub renders a bare `on:` key as the boolean `True`, so trigger lookups check both.
_ON_KEYS = (True, "on")
# Expressions that give a run its own concurrency group; anything else groups by branch.
_COMMIT_SCOPED_CONTEXTS = ("github.sha", "github.run_id")


def _workflow_names() -> list[str]:
    """Name every workflow the repo defines.

    Resolved from this file rather than the working directory so parametrization works
    whatever directory pytest is invoked from.

    :returns: Workflow file names under ``.github/workflows``, sorted.
    """
    workflows_dir = Path(__file__).resolve().parents[2] / WORKFLOWS_DIR
    return sorted(
        path.name for path in workflows_dir.iterdir() if path.suffix in {".yml", ".yaml"}
    )


def _on_block(workflow: dict[Any, Any]) -> dict[str, Any]:
    """Return a workflow's trigger mapping.

    :param workflow: Parsed workflow document.
    :returns: The ``on:`` mapping, or an empty mapping when it is not one.
    """
    for key in _ON_KEYS:
        block = workflow.get(key)
        if isinstance(block, dict):
            return block
    return {}


def _pushes_to_main(workflow: dict[Any, Any]) -> bool:
    """Report whether the workflow runs on pushes to ``main``.

    :param workflow: Parsed workflow document.
    :returns: ``True`` when a ``push`` trigger lists ``main``.
    """
    push = _on_block(workflow).get("push")
    return isinstance(push, dict) and "main" in (push.get("branches") or [])


def _cancels_on_push(concurrency: dict[str, Any]) -> bool:
    """Report whether ``cancel-in-progress`` can be true for a push event.

    A literal ``true`` always cancels. An expression is treated as push-safe only when it
    branches on ``github.event_name``, which is the only context that separates the events.

    :param concurrency: The workflow's ``concurrency`` mapping.
    :returns: ``True`` when a push can cancel an in-flight run.
    """
    setting = concurrency.get("cancel-in-progress", False)
    if isinstance(setting, bool):
        return setting
    return "github.event_name" not in str(setting)


@pytest.mark.parametrize("workflow_name", _workflow_names())
def test_main_push_runs_do_not_cancel_each_other(workflow_name: str, project_root: Path) -> None:
    """Every main-push workflow that cancels gives each commit its own group (#3661).

    :param workflow_name: Workflow file under ``.github/workflows``.
    :param project_root: Repo root supplied by the infra fixtures.
    """
    workflow = load_workflow(project_root, workflow_name)
    concurrency = workflow.get("concurrency")
    if not _pushes_to_main(workflow) or not isinstance(concurrency, dict):
        pytest.skip(f"{workflow_name} does not cancel main-push runs")
    if not _cancels_on_push(concurrency):
        pytest.skip(f"{workflow_name} does not cancel on push")

    group = str(concurrency.get("group", ""))

    assert any(context in group for context in _COMMIT_SCOPED_CONTEXTS), (
        f"{workflow_name} groups every main push under `{group}` and cancels in progress, "
        "so a merge train discards the verdict of every commit but the last"
    )
