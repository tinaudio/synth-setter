"""Contract tests for the SkyPilot-dispatched GPU test workflow."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import cast

import pytest
from workflow_fixtures import load_workflow

from synth_setter.pipeline.compute_task import (
    apply_tier_filter,
    build_task_doc,
    load_compute_option,
)
from synth_setter.pipeline.schemas.gpu_tier import GpuTier
from synth_setter.pipeline.skypilot_launch import resolve_worker_env

_REPO_ROOT = Path(__file__).parents[2]
_CRED_SCRIPT = _REPO_ROOT / "scripts/skypilot/write_provider_creds.sh"
_WORKER_SCRIPT = _REPO_ROOT / "scripts/ci/gpu_tests.sh"
_WORKFLOW = "test-gpu.yml"


def _write_executable(path: Path, body: str) -> Path:
    """Write an executable command fixture.

    :param path: Fixture command path.
    :param body: Complete shell script body.
    :returns: ``path`` after writing and setting executable mode.
    """
    path.write_text(body)
    path.chmod(0o755)
    return path


def _install_gpu_command_fixtures(tmp_path: Path, rclone: str) -> tuple[Path, Path]:
    """Install successful GPU probes plus a pytest-failing VST runner.

    :param tmp_path: Isolated test directory.
    :param rclone: Real rclone binary wrapped by the command fixture.
    :returns: Fake rclone mount and VST runner paths.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_rclone = _write_executable(fake_bin / "rclone", f'#!/bin/bash\nexec "{rclone}" "$@"\n')
    _write_executable(fake_bin / "nvidia-smi", "#!/bin/bash\nexit 0\n")
    _write_executable(fake_bin / "python", "#!/bin/bash\nexit 0\n")
    vst_runner = _write_executable(
        fake_bin / "run-linux-vst-headless.sh",
        '#!/bin/bash\nif [[ "$*" == *pytest* ]]; then exit 42; fi\nexit 0\n',
    )
    return fake_rclone, vst_runner


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
def test_gpu_workflow_pins_low_tier_pool_and_worker_image(project_root: Path) -> None:
    """The Hydra launcher invocation pins the GPU test infrastructure.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    launch_command = cast(str, _launch_step(project_root)["run"])

    assert "skypilot_launch/compute=runpod/gpu-tests" in launch_command
    assert "skypilot_launch.tier=low" in launch_command
    assert "skypilot_launch.worker_image_tag=dev-snapshot" in launch_command
    assert "skypilot_launch.tail=true" in launch_command


@pytest.mark.infra
def test_gpu_compute_option_yaml_mounts_the_pinned_rclone() -> None:
    """RunPod rejects programmatic file_mounts (#749), so the mount must be YAML-declared."""
    compute = load_compute_option("runpod/gpu-tests")

    task_doc = build_task_doc(compute, cmd="bash scripts/ci/gpu_tests.sh")

    pod_destination = "/tmp/synth-setter-tools/rclone"  # noqa: S108 — remote pod path

    assert task_doc["file_mounts"] == {pod_destination: "/usr/local/bin/rclone"}


@pytest.mark.infra
def test_gpu_worker_script_prefers_the_mounted_rclone_over_the_image_one() -> None:
    """The image's apt rclone fails first R2 writes, so the mounted binary must win $PATH."""
    script = _WORKER_SCRIPT.read_text(encoding="utf-8")

    assert 'export PATH="${rclone_mount_path%/*}:${PATH}"' in script
    assert script.index('chmod u+x "${rclone_mount_path}"') < script.index("export PATH=")
    assert script.index("export PATH=") < script.index("trap upload_coverage EXIT")


@pytest.mark.infra
def test_gpu_compute_option_resolves_to_cheap_consumer_gpus() -> None:
    """Tier filtering keeps the paid pool on consumer cards, excluding A40-class SKUs."""
    compute = load_compute_option("runpod/gpu-tests")

    task_doc = build_task_doc(
        apply_tier_filter(compute, GpuTier.LOW), cmd="bash scripts/ci/gpu_tests.sh"
    )

    resources = cast(dict[str, object], task_doc["resources"])
    assert resources["accelerators"] == {"RTX3070": 1, "RTX3080": 1, "RTX3090": 1, "RTX4090": 1}


@pytest.mark.infra
def test_gpu_workflow_runs_generic_worker_command(project_root: Path) -> None:
    """The workflow supplies the GPU script through ``skypilot_launch.cmd``.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    launch_command = cast(str, _launch_step(project_root)["run"])

    assert "skypilot_launch.cmd=exec bash scripts/ci/gpu_tests.sh" in launch_command


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
    assert "pytest -vv -s -m gpu" in script
    assert "--cov=src --cov-branch --cov-report=xml" in script


@pytest.mark.infra
def test_gpu_worker_script_failed_pytest_uploads_coverage_and_preserves_exit(
    tmp_path: Path,
) -> None:
    """The executable worker publishes partial coverage without masking pytest failure.

    :param tmp_path: Isolated worker directory and command fixtures.
    """
    rclone = shutil.which("rclone")
    if rclone is None:
        pytest.skip("requires rclone for the upload/retrieval round trip")
    fake_rclone, vst_runner = _install_gpu_command_fixtures(tmp_path, rclone)
    fake_bin = fake_rclone.parent
    (tmp_path / "coverage.xml").write_text("partial coverage")
    r2_root = tmp_path / "r2"
    coverage_key = "ci/gpu/test/coverage.xml"
    env = {
        "COVERAGE_KEY": coverage_key,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "R2_BUCKET": str(r2_root),
        "RCLONE_CONFIG_R2_TYPE": "local",
        "RCLONE_MOUNT_PATH": str(fake_rclone),
        "VST_RUNNER": str(vst_runner),
    }

    result = subprocess.run(  # noqa: S603 — executes the checked-in worker script
        [str(_WORKER_SCRIPT)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 42
    uploaded = r2_root / coverage_key
    assert uploaded.read_text() == "partial coverage"

    retrieved = tmp_path / "retrieved.xml"
    retrieve = subprocess.run(  # noqa: S603 — invokes the resolved rclone binary
        [rclone, "moveto", f"r2:{uploaded}", str(retrieved), "--checksum"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert retrieve.returncode == 0, retrieve.stderr
    assert retrieved.read_text() == "partial coverage"
    assert not uploaded.exists()


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
def test_gpu_workflow_r2_account_reaches_real_credential_bootstrap(
    project_root: Path, tmp_path: Path
) -> None:
    """The launch-step R2 account satisfies the real local bootstrap contract.

    :param project_root: Repo root supplied by the infra test fixtures.
    :param tmp_path: Isolated home for generated credential files.
    """
    step_env = cast(dict[str, str], _launch_step(project_root)["env"])
    assert step_env["R2_ACCOUNT_ID"] == "${{ secrets.R2_ACCOUNT_ID }}"

    result = subprocess.run(  # noqa: S603 — executes the checked-in bootstrap script
        [str(_CRED_SCRIPT), "--provider", "runpod", "--force"],
        env={
            "HOME": str(tmp_path),
            "PATH": os.environ["PATH"],
            "R2_ACCOUNT_ID": "account-id",
            "RCLONE_CONFIG_R2_ACCESS_KEY_ID": "access-key",
            "RCLONE_CONFIG_R2_ENDPOINT": "https://example.invalid",
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": "secret-key",
            "RUNPOD_API_KEY": "runpod-key",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / ".cloudflare/accountid").read_text() == "account-id\n"


@pytest.mark.infra
def test_gpu_workflow_wandb_secret_resolves_into_worker_env(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launch-step W&B secret reaches the launcher's managed-worker env.

    :param project_root: Repo root supplied by the infra test fixtures.
    :param tmp_path: Isolated empty launch env file.
    :param monkeypatch: Process-environment fixture.
    """
    launch_step = _launch_step(project_root)
    step_env = cast(dict[str, str], launch_step["env"])
    assert step_env["WANDB_API_KEY"] == "${{ secrets.WANDB_API_KEY }}"
    assert "--extra-env WANDB_API_KEY" not in cast(str, launch_step["run"])

    monkeypatch.setenv("WANDB_API_KEY", "wandb-key")
    env_file = tmp_path / ".env"
    env_file.write_text("")

    assert resolve_worker_env(env_file)["WANDB_API_KEY"] == "wandb-key"


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
