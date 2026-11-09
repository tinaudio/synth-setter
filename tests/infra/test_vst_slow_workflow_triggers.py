"""`test-vst-slow.yml` gates push and pull-request events on the same paths.

GitHub Actions rejects YAML anchors, so the workflow repeats its path list once per event. This
pins the two copies together: without it, adding a path to only one trigger silently leaves the
other event ungated (#1354).
"""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path
from typing import cast

import pytest
import sh
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
def test_vst_slow_dispatch_selects_registered_synth(project_root: Path) -> None:
    """Manual runs can select one registered VST matrix cell.

    :param project_root: Repo root holding ``.github/workflows/``.
    """
    workflow = _load_workflow(project_root)
    on_key: object = "on" if "on" in workflow else True
    triggers = cast(dict[str, dict[str, object]], workflow[on_key])
    dispatch = triggers["workflow_dispatch"]
    inputs = cast(dict[str, dict[str, object]], dispatch["inputs"])

    assert inputs["synth"]["type"] == "choice"
    assert inputs["synth"]["options"] == ["all", "obxf", "surge_xt", "ultramaster_kr106"]


@pytest.mark.infra
@pytest.mark.parametrize(
    "step_name",
    [
        "Checkout",
        "Hydrate gated embedding checkpoints",
        "Prepare Docker storage capacity",
        "Pull image",
        "Run VST slow tests in Docker",
    ],
)
def test_vst_slow_unselected_cells_skip_work(
    project_root: Path,
    step_name: str,
) -> None:
    """A manual synth selection prevents other matrix cells from doing work.

    :param project_root: Repo root holding ``.github/workflows/``.
    :param step_name: Work step that must honor the selection output.
    """
    workflow = _load_workflow(project_root)
    jobs = cast(dict[str, dict[str, object]], workflow["jobs"])
    steps = cast(list[dict[str, object]], jobs["run_vst_slow_tests"]["steps"])
    step = next(candidate for candidate in steps if candidate.get("name") == step_name)

    assert "steps.select_synth.outputs.selected == 'true'" in cast(str, step["if"])


@pytest.mark.infra
def test_vst_slow_kr106_cell_runs_native_pipeline_and_parity_suites(project_root: Path) -> None:
    """The KR106 matrix cell installs and tests the pinned native renderer.

    :param project_root: Repo root holding ``.github/workflows/``.
    """
    workflow = _load_workflow(project_root)
    jobs = cast(dict[str, dict[str, object]], workflow["jobs"])
    strategy = cast(dict[str, object], jobs["run_vst_slow_tests"]["strategy"])
    matrix = cast(dict[str, list[dict[str, str]]], strategy["matrix"])
    kr106 = next(row for row in matrix["include"] if row["synth"] == "ultramaster_kr106")
    targets = kr106["pytest_targets"].split()

    assert kr106["plugin_path"] == "/usr/lib/vst3/Ultramaster KR-106.vst3"
    assert "packages/kr106-native/tests" in targets
    assert "tests/data/vst/test_kr106_native_renderer.py" in targets
    assert "tests/data/vst/test_kr106_native_vst_parity_e2e.py" in targets
    assert "tests/integration/test_kr106_native_pipeline_e2e.py" in targets


@pytest.mark.infra
def test_native_preflight_remains_valid_python_after_shell_quoting(project_root: Path) -> None:
    """The Docker shell must preserve the embedded Python provenance check.

    :param project_root: Repo root holding the workflow.
    """
    workflow = _load_workflow(project_root)
    jobs = cast(dict[str, dict[str, object]], workflow["jobs"])
    steps = cast(list[dict[str, object]], jobs["run_vst_slow_tests"]["steps"])
    run = next(
        str(step["run"]) for step in steps if "import_kr106_native" in str(step.get("run", ""))
    )
    inner_script = shlex.split(run.split("bash -c ", 1)[1])[0]
    preflight = next(
        line.strip() for line in inner_script.splitlines() if line.strip().startswith("python -c ")
    )
    source = shlex.split(preflight)[2]
    compile(source, "native-preflight", "exec")
    assert "import_kr106_native()" in source


@pytest.mark.infra
@pytest.mark.parametrize("event_name", ["push", "pull_request"])
def test_vst_slow_native_package_and_pipeline_changes_trigger_kr106_ci(
    project_root: Path, event_name: str
) -> None:
    """Native package and real-pipeline coverage changes select the Docker workflow.

    :param project_root: Repo root holding ``.github/workflows/``.
    :param event_name: GitHub event whose path filter is checked.
    """
    triggers = _load_triggers(project_root)

    assert "packages/kr106-native/**" in triggers[event_name]["paths"]
    assert "presets/*-native.json" in triggers[event_name]["paths"]
    assert "src/synth_setter/synth_spec.py" in triggers[event_name]["paths"]
    assert "scripts/generate_kr106_native_maps.py" in triggers[event_name]["paths"]
    assert (
        "src/synth_setter/configs/synth/ultramaster_kr106*_native.yaml"
        in triggers[event_name]["paths"]
    )
    assert "tests/integration/test_kr106_native_pipeline_e2e.py" in triggers[event_name]["paths"]
    assert (
        "tests/pipeline/schemas/test_kr106_native_render_config.py"
        in triggers[event_name]["paths"]
    )


@pytest.mark.infra
def test_vst_slow_runs_parallel_queue_real_vst_e2e(project_root: Path) -> None:
    """The Surge cell proves two queue-owned VST renders overlap.

    :param project_root: Repo root holding ``.github/workflows/``.
    """
    workflow = _load_workflow(project_root)
    jobs = cast(dict[str, dict[str, object]], workflow["jobs"])
    strategy = cast(dict[str, object], jobs["run_vst_slow_tests"]["strategy"])
    matrix = cast(dict[str, list[dict[str, str]]], strategy["matrix"])
    surge = next(row for row in matrix["include"] if row["synth"] == "surge_xt")

    assert (
        "tests/test_generate_dataset.py::"
        "test_from_hydra_claims_mode_parallel_real_vst_writes_consumable_shards"
        in surge["pytest_targets"].split()
    )


@pytest.mark.infra
@pytest.mark.parametrize("event_name", ["push", "pull_request"])
def test_vst_slow_parallel_queue_changes_trigger_real_vst_e2e(
    project_root: Path, event_name: str
) -> None:
    """Queue implementation and E2E changes select the real-VST workflow.

    :param project_root: Repo root holding ``.github/workflows/``.
    :param event_name: GitHub event whose path filter is checked.
    """
    triggers = _load_triggers(project_root)

    assert "src/synth_setter/cli/generate_dataset.py" in triggers[event_name]["paths"]
    assert "tests/test_generate_dataset.py" in triggers[event_name]["paths"]


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


def _install_fixed_utc_date(tmp_path: Path) -> Path:
    """Install a deterministic ``date`` executable for the upload step.

    :param tmp_path: Temporary directory that will contain the executable.
    :returns: Directory to prepend to ``PATH``.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_date = bin_dir / "date"
    fake_date.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        '[[ "${1}" == "-u" ]]\n'
        '[[ "${2}" == "+%Y-%m-%dT%H-%M-%SZ" ]]\n'
        "printf '%s\\n' '2026-08-03T12-34-56Z'\n"
    )
    fake_date.chmod(0o755)
    return bin_dir


@pytest.mark.infra
def test_vst_slow_surge_r2_upload_folder_starts_with_utc_datetime(
    project_root: Path,
    tmp_path: Path,
) -> None:
    """Surge R2 upload folders sort chronologically by their leading UTC datetime.

    :param project_root: Repo root holding ``.github/workflows/``.
    :param tmp_path: Temporary local filesystem backing the fake R2 remote.
    """
    if shutil.which("rclone") is None:
        pytest.skip("rclone binary not available on PATH")

    workflow = _load_workflow(project_root)
    jobs = cast(dict[str, dict[str, object]], workflow["jobs"])
    steps = cast(list[dict[str, object]], jobs["upload_surge_comparison"]["steps"])
    upload_step = next(
        step for step in steps if step["name"] == "Upload comparison directory with checksums"
    )
    script = cast(str, upload_step["run"])

    comparison_dir = tmp_path / "surge-host-parity"
    comparison_dir.mkdir()
    (comparison_dir / "comparison.wav").write_bytes(b"comparison")

    bin_dir = _install_fixed_utc_date(tmp_path)
    env = os.environ | {
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_RUN_ID": "123456",
        "GITHUB_SHA": "0123456789abcdef",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "RCLONE_CONFIG_R2_TYPE": "local",
    }
    bash_path = shutil.which("bash")
    assert bash_path is not None, "bash is required to execute workflow run blocks"
    sh.Command(bash_path)("-c", script, _cwd=tmp_path, _env=env)

    uploaded_audio = (
        tmp_path
        / "experiments"
        / "surge-host-parity"
        / "2026-08-03T12-34-56Z-0123456789abcdef-123456-2"
        / "comparison.wav"
    )
    assert uploaded_audio.read_bytes() == b"comparison"


@pytest.mark.infra
def test_vst_slow_triggers_declare_no_yaml_anchors(project_root: Path) -> None:
    """Neither trigger deduplicates through an anchor GitHub Actions cannot parse.

    :param project_root: Repo root holding ``.github/workflows/``.
    """
    workflow_text = (project_root / ".github" / "workflows" / WORKFLOW_FILENAME).read_text()

    assert "paths: &" not in workflow_text
    assert "paths: *" not in workflow_text
