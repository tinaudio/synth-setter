"""Contract tests for the SkyPilot-dispatched GPU test workflow."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import cast

import pytest
from workflow_fixtures import load_workflow

from synth_setter.pipeline.compute_task import apply_tier_filter, build_task_doc
from synth_setter.pipeline.schemas.gpu_tier import GpuTier
from synth_setter.pipeline.skypilot_launch import load_launch_config

_REPO_ROOT = Path(__file__).parents[2]
_LAUNCH_CONFIG = _REPO_ROOT / "src/synth_setter/configs/launch/gpu-tests-runpod.yaml"
_WORKER_SCRIPT = _REPO_ROOT / "scripts/ci/gpu_tests.sh"
_WORKFLOW = "test-gpu.yml"


def _workflow_steps(project_root: Path) -> list[dict[str, object]]:
    """Load the GPU workflow's ordered steps.

    :param project_root: Repo root supplied by the infra test fixtures.
    :returns: Step mappings from the GPU job.
    """
    jobs = cast(dict[str, dict[str, object]], load_workflow(project_root, _WORKFLOW)["jobs"])
    return cast(list[dict[str, object]], jobs["run_tests"]["steps"])


def _named_step(project_root: Path, name: str) -> dict[str, object]:
    """Load one named step from the GPU workflow's only job.

    :param project_root: Repo root supplied by the infra test fixtures.
    :param name: Exact workflow step name.
    :returns: The matching step mapping.
    """
    return next(step for step in _workflow_steps(project_root) if step.get("name") == name)


def _launch_step(project_root: Path) -> dict[str, object]:
    """Load the workflow step that dispatches the GPU job.

    :param project_root: Repo root supplied by the infra test fixtures.
    :returns: The dispatch step mapping.
    """
    return _named_step(project_root, "Dispatch GPU tests via SkyPilot")


@pytest.mark.infra
def test_gpu_launch_config_pins_low_tier_smoke_pool_on_dev_snapshot() -> None:
    """The checked-in config carries every launch knob the workflow used to inline."""
    config = load_launch_config(_LAUNCH_CONFIG)

    assert config.compute is not None
    assert config.compute.name == "runpod-gpu-tests"
    assert config.tier is GpuTier.LOW
    assert config.worker_image_tag == "dev-snapshot"
    assert config.tail is True


@pytest.mark.infra
def test_gpu_compute_option_yaml_mounts_the_pinned_rclone() -> None:
    """RunPod rejects programmatic file_mounts (#749), so the mount must be YAML-declared."""
    config = load_launch_config(_LAUNCH_CONFIG)
    assert config.compute is not None

    task_doc = build_task_doc(config.compute, cmd=config.cmd)

    pod_destination = "/tmp/synth-setter-tools/rclone"  # noqa: S108 — remote pod path

    assert task_doc["file_mounts"] == {pod_destination: "/usr/local/bin/rclone"}


@pytest.mark.infra
def test_gpu_worker_script_prefers_the_mounted_rclone_over_the_image_one() -> None:
    """The image's apt rclone fails first R2 writes, so the mounted binary must win $PATH."""
    script = _WORKER_SCRIPT.read_text(encoding="utf-8")

    assert 'export PATH="/tmp/synth-setter-tools:${PATH}"' in script
    assert script.index("chmod u+x /tmp/synth-setter-tools/rclone") < script.index("export PATH=")
    assert script.index("export PATH=") < script.index("rclone copyto coverage.xml")


@pytest.mark.infra
def test_gpu_launch_config_resolves_to_cheap_consumer_gpus() -> None:
    """Tier filtering keeps the paid pool on consumer cards, excluding A40-class SKUs."""
    config = load_launch_config(_LAUNCH_CONFIG)
    assert config.compute is not None

    task_doc = build_task_doc(apply_tier_filter(config.compute, config.tier), cmd=config.cmd)

    resources = cast(dict[str, object], task_doc["resources"])
    assert resources["accelerators"] == {"RTX3070": 1, "RTX3080": 1, "RTX3090": 1, "RTX4090": 1}


@pytest.mark.infra
def test_gpu_launch_config_syncs_worker_checkout_before_running_tests() -> None:
    """The worker runs the dispatched commit's script, not the image-baked checkout."""
    config = load_launch_config(_LAUNCH_CONFIG)
    assert config.compute is not None

    run_block = cast(str, build_task_doc(config.compute, cmd=config.cmd)["run"])

    assert run_block.index("sync_worker_checkout.sh") < run_block.index("scripts/ci/gpu_tests.sh")


@pytest.mark.infra
def test_gpu_worker_script_is_valid_bash() -> None:
    """A syntax error in the worker script would only surface after paid provisioning."""
    syntax_check = subprocess.run(  # noqa: S603 — argv is a checked-in repo path
        ["bash", "-n", str(_WORKER_SCRIPT)],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )

    assert syntax_check.returncode == 0, syntax_check.stderr


@pytest.mark.infra
def test_gpu_worker_script_proves_cuda_and_vst_before_running_gpu_tests() -> None:
    """The remote body still exercises the CUDA, Surge XT, and headless-pytest paths."""
    script = _WORKER_SCRIPT.read_text(encoding="utf-8")

    assert "nvidia-smi" in script
    assert "torch.cuda.is_available()" in script
    assert "load_plugin" in script
    assert "run-linux-vst-headless.sh pytest -vv -s -m gpu" in script
    assert "--cov=src --cov-branch --cov-report=xml" in script


@pytest.mark.infra
def test_gpu_worker_script_uploads_coverage_even_when_pytest_fails() -> None:
    """Coverage leaves the worker through an EXIT trap, so partial results survive."""
    script = _WORKER_SCRIPT.read_text(encoding="utf-8")

    assert "trap upload_coverage EXIT" in script
    assert "rclone copyto coverage.xml" in script
    assert "--checksum" in script


@pytest.mark.infra
def test_gpu_workflow_dispatches_through_the_shared_launcher(project_root: Path) -> None:
    """Orchestration is one launcher call, not a re-implementation of dispatch.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    launch_command = cast(str, _launch_step(project_root)["run"])

    assert "python -m synth_setter.pipeline.skypilot_launch" in launch_command
    assert "sky.launch" not in launch_command
    assert "task.workdir" not in launch_command


@pytest.mark.infra
def test_gpu_workflow_dispatches_a_launch_config_that_exists(project_root: Path) -> None:
    """The dispatched config path is the one this suite pins the contract of.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    workflow = load_workflow(project_root, _WORKFLOW)
    dispatched = cast(dict[str, str], workflow["env"])["LAUNCH_CONFIG"]

    assert (project_root / dispatched).is_file()
    assert (project_root / dispatched) == _LAUNCH_CONFIG


@pytest.mark.infra
def test_gpu_workflow_fails_when_a_passing_run_returns_no_coverage(project_root: Path) -> None:
    """A green GPU run with no coverage means silent Codecov rot, so it fails the job.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    retrieve = _named_step(project_root, "Retrieve remote coverage")

    assert retrieve["if"] == "always()"
    assert "GPU worker succeeded without a retrievable coverage.xml" in cast(str, retrieve["run"])


@pytest.mark.infra
def test_gpu_workflow_pins_worker_checkout_to_the_dispatched_commit(project_root: Path) -> None:
    """WORKER_GIT_REF is how the worker gets this branch's code.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    step_env = cast(dict[str, str], _launch_step(project_root)["env"])

    assert step_env["WORKER_GIT_REF"] == "${{ github.sha }}"


@pytest.mark.infra
def test_gpu_workflow_supplies_local_launcher_and_worker_credentials(project_root: Path) -> None:
    """Local dispatch bootstraps R2 while the managed worker receives W&B auth.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    launch_step = _launch_step(project_root)
    step_env = cast(dict[str, str], launch_step["env"])

    assert step_env["R2_ACCOUNT_ID"] == "${{ secrets.R2_ACCOUNT_ID }}"
    assert step_env["WANDB_API_KEY"] == "${{ secrets.WANDB_API_KEY }}"
    assert '--extra-env WANDB_API_KEY "$WANDB_API_KEY"' in cast(str, launch_step["run"])


@pytest.mark.infra
def test_gpu_workflow_does_not_persist_launcher_logs(project_root: Path) -> None:
    """Launcher output can contain credentials, so it remains only in masked job logs.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    steps = _workflow_steps(project_root)

    assert all(step.get("name") != "Upload run metadata" for step in steps)
    assert "tee" not in cast(str, _launch_step(project_root)["run"])


@pytest.mark.infra
def test_gpu_workflow_pins_external_actions_to_commit_shas(project_root: Path) -> None:
    """External actions use immutable revisions while local actions stay relative.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    external_uses = [
        cast(str, step["uses"])
        for step in _workflow_steps(project_root)
        if "uses" in step and not cast(str, step["uses"]).startswith("./")
    ]

    assert external_uses == [
        "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803",
        "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1",
        "astral-sh/setup-uv@d0cc045d04ccac9d8b7881df0226f9e82c39688e",
    ]


@pytest.mark.infra
def test_gpu_workflow_triggers_on_schedule_and_dispatch_only(project_root: Path) -> None:
    """GPU validation stays post-merge/manual because every run spends RunPod credit.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    workflow_text = (project_root / ".github" / "workflows" / _WORKFLOW).read_text()

    assert "\n  schedule:\n" in workflow_text
    assert "\n  workflow_dispatch:\n" in workflow_text
    assert "\n  pull_request:\n" not in workflow_text
