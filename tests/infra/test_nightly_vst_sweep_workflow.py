"""Pin the invariants that keep ``nightly-vst-sweep.yml`` a marker-driven VST net.

The sibling VST workflows run hand-maintained file allowlists, so a newly added
``requires_vst`` test that nobody registers runs nowhere — the fall-through that
shipped ``test_dawdreamer_dataset_e2e.py`` broken (#1825). These tests fail if a
future edit drops the marker filter or replaces the discovered matrix with a
static list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

_WORKFLOW = "nightly-vst-sweep.yml"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW_PATH = _PROJECT_ROOT / ".github" / "workflows" / _WORKFLOW
_WORKFLOW_TEXT = _WORKFLOW_PATH.read_text(encoding="utf-8")
_WORKFLOW_YAML: dict[str, Any] = yaml.safe_load(_WORKFLOW_TEXT)

# The canonical marker: every ``requires_vst`` test on CPU, gpu/mps excluded.
_CANONICAL_MARKER = "requires_vst and not gpu and not mps"


def _step_run(job: str, step_name: str) -> str:
    """Return the ``run`` body of the named step in the named job.

    :param job: job key under ``jobs:``.
    :param step_name: the step's ``name:`` value.
    :returns: the step's ``run:`` script (empty string if it has none).
    :raises AssertionError: if no step in ``job`` has that name.
    """
    for step in _WORKFLOW_YAML["jobs"][job]["steps"]:
        if step.get("name") == step_name:
            return step.get("run", "")
    raise AssertionError(f"{_WORKFLOW}: job {job!r} has no step named {step_name!r}")


@pytest.mark.infra
def test_marker_env_carries_canonical_requires_vst_filter() -> None:
    """The workflow-level ``VST_MARKER`` env holds the canonical filter."""
    assert f'VST_MARKER: "{_CANONICAL_MARKER}"' in _WORKFLOW_TEXT


@pytest.mark.infra
def test_discover_and_run_both_select_via_the_shared_marker() -> None:
    """Collection and the per-shard run both filter by ``$VST_MARKER``, not a copy."""
    collect = _step_run("discover", "Collect requires_vst node IDs")
    run = _step_run("vst_sweep", "Run VST sweep shard in Docker")
    assert "--collect-only" in collect and '-m "$VST_MARKER"' in collect
    assert '-m "$VST_MARKER"' in run


@pytest.mark.infra
def test_matrix_is_discovered_at_runtime_not_a_static_allowlist() -> None:
    """The run matrix is the discover job's output, never a literal file list."""
    sweep = _WORKFLOW_YAML["jobs"]["vst_sweep"]
    assert sweep["needs"] == "discover"
    assert sweep["strategy"]["matrix"] == "${{ fromJSON(needs.discover.outputs.matrix) }}", (
        "a static allowlist is the #1825 fall-through this net prevents"
    )
    assert sweep["strategy"]["fail-fast"] is False


@pytest.mark.infra
def test_build_shard_matrix_derives_files_from_collected_output() -> None:
    """The matrix step builds ``files`` from the collected node IDs, not a constant."""
    build = _step_run("discover", "Build shard matrix")
    assert 'os.environ["COLLECTED"]' in build


@pytest.mark.infra
def test_build_shard_matrix_refuses_an_empty_selection() -> None:
    """A 0-test collection errors instead of emitting a silently-green matrix."""
    build = _step_run("discover", "Build shard matrix")
    assert "refusing to" in build


@pytest.mark.infra
def test_runs_in_dev_snapshot_image_under_headless_wrapper() -> None:
    """The suite runs inside the dev-snapshot image behind the Xvfb wrapper."""
    assert "tinaudio/synth-setter:${{ inputs.image_tag || 'dev-snapshot' }}" in _WORKFLOW_TEXT
    assert "src/synth_setter/scripts/run-linux-vst-headless.sh" in _WORKFLOW_TEXT


@pytest.mark.infra
def test_triggers_on_schedule_and_dispatch_only() -> None:
    """A nightly net triggers on cron + manual dispatch, never on push/PR.

    Asserted on raw text because PyYAML parses the bare ``on:`` key as ``True``.
    """
    assert "\n  schedule:\n" in _WORKFLOW_TEXT and "cron:" in _WORKFLOW_TEXT
    assert "\n  workflow_dispatch:\n" in _WORKFLOW_TEXT
    for pr_trigger in ("\n  push:\n", "\n  pull_request:\n"):
        assert pr_trigger not in _WORKFLOW_TEXT
