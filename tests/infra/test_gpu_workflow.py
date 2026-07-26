"""Pin the bounded production-path contract of ``test-gpu.yml``."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from workflow_fixtures import load_workflow

_WORKFLOW = "test-gpu.yml"


def _gpu_job(project_root: Path) -> dict[str, object]:
    """Load the GPU workflow job.

    :param project_root: Repo root supplied by the infra test fixtures.
    :returns: The ``run_tests`` job mapping.
    """
    workflow = load_workflow(project_root, _WORKFLOW)
    jobs = cast(dict[str, dict[str, object]], workflow["jobs"])
    return jobs["run_tests"]


def _named_step(project_root: Path, name: str) -> dict[str, object]:
    """Load one named GPU workflow step.

    :param project_root: Repo root supplied by the infra test fixtures.
    :param name: Exact workflow step name.
    :returns: The matching step mapping.
    """
    steps = cast(list[dict[str, object]], _gpu_job(project_root)["steps"])
    return next(step for step in steps if step.get("name") == name)


@pytest.mark.infra
def test_gpu_workflow_uses_bounded_github_orchestrator(project_root: Path) -> None:
    """The GitHub job orchestrates bounded remote GPU work from a standard runner.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    gpu_job = _gpu_job(project_root)
    launch_step = _named_step(project_root, "Launch GPU tests on RunPod")

    assert gpu_job["runs-on"] == "ubuntu-latest"
    assert gpu_job["timeout-minutes"] == 120
    assert launch_step["timeout-minutes"] == 105


@pytest.mark.infra
def test_gpu_workflow_remote_command_exercises_real_production_path(project_root: Path) -> None:
    """The RunPod task proves CUDA, VST, pytest, and remote coverage production paths.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    launch = cast(str, _named_step(project_root, "Launch GPU tests on RunPod")["run"])

    assert 'load_compute_option("runpod/smoke")' in launch
    assert "GpuTier.LOW" in launch
    assert "task.workdir = os.getcwd()" in launch
    assert "f\"tinaudio/synth-setter:{os.environ['IMAGE_TAG']}\"" in launch
    assert "nvidia-smi --query-gpu=name,memory.free --format=csv,noheader" in launch
    assert "torch.cuda.is_available()" in launch
    assert "load_plugin" in launch and "/usr/lib/vst3/Surge XT.vst3" in launch
    assert "bash src/synth_setter/scripts/run-linux-vst-headless.sh python" in launch
    assert "bash src/synth_setter/scripts/run-linux-vst-headless.sh pytest -vv -s -m gpu" in launch
    assert "--cov=src --cov-branch --cov-report=xml" in launch
    assert "rclone copyto coverage.xml" in launch and "--checksum" in launch
    assert "WANDB_API_KEY" not in launch


@pytest.mark.infra
def test_gpu_workflow_preflights_balance_and_round_trips_coverage(project_root: Path) -> None:
    """Provisioning follows balance preflight and real coverage returns for Codecov.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    preflight = cast(
        str,
        _named_step(project_root, "Configure RunPod credentials and check balance")["run"],
    )
    download = _named_step(project_root, "Download remote coverage")
    upload = _named_step(project_root, "Upload coverage to Codecov")

    assert "write_provider_creds.sh --provider runpod" in preflight
    assert "_check_runpod_balance" in preflight
    assert download["if"] == "always()"
    download_command = cast(str, download["run"])
    assert "rclone copyto" in download_command
    assert "&& [[ -s coverage.xml ]]" in download_command
    assert "--checksum" in download_command
    assert "GPU worker succeeded without a retrievable coverage.xml" in download_command
    assert upload["uses"] == "./.github/actions/upload-coverage"


@pytest.mark.infra
def test_gpu_workflow_always_tears_down_remote_cluster(project_root: Path) -> None:
    """Every outcome tears down the paid worker and verifies no cluster remains.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    steps = cast(list[dict[str, object]], _gpu_job(project_root)["steps"])
    names = [step.get("name") for step in steps]
    launch = _named_step(project_root, "Launch GPU tests on RunPod")
    teardown = _named_step(project_root, "Tear down RunPod worker")
    teardown_command = cast(str, teardown["run"])

    assert names.index("Download remote coverage") < names.index("Tear down RunPod worker")
    assert launch["timeout-minutes"] == 105
    assert "idle_minutes_to_autostop=5" in cast(str, launch["run"])
    assert "down=True" in cast(str, launch["run"])
    assert teardown["if"] == "always()"
    assert "sky.down(cluster)" in teardown_command
    assert "sky.status" in teardown_command
    assert "raise SystemExit" in teardown_command


@pytest.mark.infra
def test_gpu_workflow_triggers_on_schedule_and_dispatch_only(project_root: Path) -> None:
    """GPU validation remains post-merge/manual until the follow-up PR lane lands.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    workflow_text = (project_root / ".github" / "workflows" / _WORKFLOW).read_text()

    assert "\n  schedule:\n" in workflow_text
    assert "\n  workflow_dispatch:\n" in workflow_text
    assert "\n  pull_request:\n" not in workflow_text
