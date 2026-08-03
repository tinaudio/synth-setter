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


def _load_workflow(project_root: Path) -> dict[object, object]:
    """Return the parsed VST workflow.

    :param project_root: Repo root holding ``.github/workflows/``.
    :returns: Parsed workflow mapping.
    """
    return cast(dict[object, object], load_workflow(project_root, WORKFLOW_FILENAME))


def _load_triggers(project_root: Path) -> dict[str, dict[str, list[str]]]:
    """Return the workflow's event-trigger block.

    :param project_root: Repo root holding ``.github/workflows/``.
    :returns: Trigger mapping keyed by event name.
    """
    workflow = _load_workflow(project_root)
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
@pytest.mark.parametrize("event_name", ["push", "pull_request"])
def test_vst_slow_ssondo_changes_trigger_real_vst_e2e(project_root: Path, event_name: str) -> None:
    """S-SONDO implementation changes select the real-VST workflow.

    :param project_root: Repo root holding ``.github/workflows/``.
    :param event_name: GitHub event whose path filter is checked.
    """
    triggers = _load_triggers(project_root)

    assert "src/synth_setter/pipeline/data/ssondo.py" in triggers[event_name]["paths"]


@pytest.mark.infra
@pytest.mark.parametrize("event_name", ["push", "pull_request"])
def test_vst_slow_meanaudio_changes_trigger_real_eval_e2e(
    project_root: Path, event_name: str
) -> None:
    """MeanAudio implementation changes select its real train/eval workflow leg.

    :param project_root: Repo root holding ``.github/workflows/``.
    :param event_name: GitHub event whose path filter is checked.
    """
    triggers = _load_triggers(project_root)
    workflow_text = (project_root / ".github" / "workflows" / WORKFLOW_FILENAME).read_text()

    assert "src/synth_setter/pipeline/data/meanaudio.py" in triggers[event_name]["paths"]
    assert (
        "tests/test_eval.py::"
        "test_train_eval_meanaudio_conditioning_real_lance_returns_bounded_metric"
    ) in workflow_text


@pytest.mark.infra
def test_vst_slow_publishes_random_patch_diagnostics(project_root: Path) -> None:
    """Pin the JSON handoff and benchmark action required for publication.

    :param project_root: Repo root holding ``.github/workflows/``.
    """
    workflow = cast(dict[str, object], load_workflow(project_root, WORKFLOW_FILENAME))
    jobs = cast(dict[str, dict[str, object]], workflow["jobs"])
    run_steps = cast(list[dict[str, object]], jobs["run_vst_slow_tests"]["steps"])
    publish_steps = cast(list[dict[str, object]], jobs["publish_benchmarks"]["steps"])
    filename = "surge-host-parity-random-patches.json"

    surface = next(
        step
        for step in run_steps
        if step.get("name") == "Surface per-bucket bench JSON files on the runner"
    )
    upload = next(
        step
        for step in run_steps
        if step.get("name") == "Upload benchmark JSON for the publish job"
    )
    publish = next(
        step
        for step in publish_steps
        if step.get("name") == "Publish random-patch Surge host diagnostics"
    )

    assert filename in cast(str, surface["run"])
    expected_upload_path = "${{ github.workspace }}/" + filename
    assert expected_upload_path in cast(dict[str, str], upload["with"])["path"].splitlines()
    assert publish["if"] == "hashFiles('surge-host-parity-random-patches.json') != ''"
    assert str(publish["uses"]).startswith("benchmark-action/github-action-benchmark@")
    publish_inputs = cast(dict[str, object], publish["with"])
    assert publish_inputs["name"] == "Surge host diagnostics (random patches)"
    assert publish_inputs["tool"] == "customSmallerIsBetter"
    assert publish_inputs["output-file-path"] == filename


@pytest.mark.infra
def test_vst_slow_surge_r2_upload_folder_starts_with_utc_datetime(
    project_root: Path,
) -> None:
    """Surge R2 upload folders sort chronologically by their leading UTC datetime.

    :param project_root: Repo root holding ``.github/workflows/``.
    """
    workflow = _load_workflow(project_root)
    jobs = cast(dict[str, dict[str, object]], workflow["jobs"])
    steps = cast(list[dict[str, object]], jobs["upload_surge_comparison"]["steps"])
    upload_step = next(
        step for step in steps if step["name"] == "Upload comparison directory with checksums"
    )
    script = cast(str, upload_step["run"])

    assert "upload_datetime=\"$(date -u '+%Y-%m-%dT%H-%M-%SZ')\"" in script
    assert (
        'r2_destination="r2:experiments/surge-host-parity/'
        '${upload_datetime}-${GITHUB_SHA}-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"'
    ) in script


@pytest.mark.infra
def test_vst_slow_triggers_declare_no_yaml_anchors(project_root: Path) -> None:
    """Neither trigger deduplicates through an anchor GitHub Actions cannot parse.

    :param project_root: Repo root holding ``.github/workflows/``.
    """
    workflow_text = (project_root / ".github" / "workflows" / WORKFLOW_FILENAME).read_text()

    assert "paths: &" not in workflow_text
    assert "paths: *" not in workflow_text
