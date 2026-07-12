"""Pin the invariants that make ``nightly-vst-sweep.yml`` a durable VST net.

The sibling VST workflows (``test-vst-slow.yml``, ``docker-build-validation.yml``)
run hand-maintained file allowlists, so a newly added ``requires_vst`` test that
nobody registers runs nowhere — the fall-through that let
``test_dawdreamer_dataset_e2e.py`` ship broken (#1825). This workflow's whole
value is that its selection is marker-driven and its shard matrix is discovered
at run time, so these tests fail if a future edit reintroduces a static
allowlist or drops the marker filter.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.infra.workflow_fixtures import WORKFLOWS_DIR, load_workflow

WORKFLOW = "nightly-vst-sweep.yml"

# The canonical marker: every ``requires_vst`` test on CPU, gpu/mps excluded.
CANONICAL_MARKER = "requires_vst and not gpu and not mps"


@pytest.fixture(scope="module")
def workflow_text(project_root: Path) -> str:
    """Raw text of the nightly VST sweep workflow.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :returns: the workflow file contents.
    """
    return (project_root / WORKFLOWS_DIR / WORKFLOW).read_text()


@pytest.mark.infra
def test_marker_is_the_single_canonical_requires_vst_filter(workflow_text: str) -> None:
    """``VST_MARKER`` carries the canonical filter and drives selection.

    :param workflow_text: raw workflow YAML.
    """
    assert f'VST_MARKER: "{CANONICAL_MARKER}"' in workflow_text, (
        "the sweep must select tests via the canonical requires_vst marker"
    )
    assert workflow_text.count('-m "$VST_MARKER"') >= 2, (
        "both the discover collect and the per-shard run must select by "
        "$VST_MARKER so they can't drift"
    )


@pytest.mark.infra
def test_selection_is_marker_driven_not_a_static_allowlist(project_root: Path) -> None:
    """The shard matrix is discovered at run time, never a hard-coded file list.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    """
    workflow = load_workflow(project_root, WORKFLOW)
    sweep = workflow["jobs"]["vst_sweep"]
    assert sweep["needs"] == "discover", "the run job must consume the discover matrix"
    assert sweep["strategy"]["matrix"] == "${{ fromJSON(needs.discover.outputs.matrix) }}", (
        "the matrix must be the discover job's collected output, not a literal "
        "file list — a static allowlist is the fall-through this net prevents (#1825)"
    )
    assert sweep["strategy"]["fail-fast"] is False, (
        "fail-fast must be false so one failing shard can't mask another"
    )


@pytest.mark.infra
def test_discover_collects_by_marker_and_refuses_empty_matrix(project_root: Path) -> None:
    """Discover collects ``requires_vst`` by marker and errors on an empty set.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    """
    workflow = load_workflow(project_root, WORKFLOW)
    discover = workflow["jobs"]["discover"]
    steps = " ".join(step.get("run", "") for step in discover["steps"])
    assert "--collect-only" in steps and '-m "$VST_MARKER"' in steps, (
        "discover must enumerate the requires_vst set via `pytest --collect-only -m`"
    )
    assert "refusing to" in steps, (
        "discover must fail on a 0-test collection rather than emit a silently-green empty matrix"
    )


@pytest.mark.infra
def test_runs_in_dev_snapshot_image_under_headless_wrapper(workflow_text: str) -> None:
    """The suite runs inside the dev-snapshot image behind the Xvfb wrapper.

    :param workflow_text: raw workflow YAML.
    """
    assert "tinaudio/synth-setter:${{ inputs.image_tag || 'dev-snapshot' }}" in workflow_text
    assert "src/synth_setter/scripts/run-linux-vst-headless.sh" in workflow_text, (
        "pedalboard's VST-host import needs a live X display; pytest must run "
        "under the headless wrapper"
    )


@pytest.mark.infra
def test_is_nightly_schedule_plus_dispatch_only(workflow_text: str) -> None:
    """The workflow triggers on a cron schedule and manual dispatch only.

    A nightly net must not gate PRs, so ``push`` / ``pull_request`` triggers are
    forbidden. Asserted on raw text because PyYAML parses the bare ``on:`` key
    as the boolean ``True``.

    :param workflow_text: raw workflow YAML.
    """
    assert "\n  schedule:\n" in workflow_text and "cron:" in workflow_text, (
        "a nightly workflow needs a cron schedule"
    )
    assert "\n  workflow_dispatch:\n" in workflow_text, "manual dispatch must stay available"
    for pr_trigger in ("\n  push:\n", "\n  pull_request:\n"):
        assert pr_trigger not in workflow_text, (
            "a nightly net must not gate PRs: schedule + workflow_dispatch only"
        )
