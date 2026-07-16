"""Contracts for relocating GitHub-hosted runner Docker storage to `/mnt`."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import cast

import pytest
from workflow_fixtures import load_composite_action, load_workflow

ACTION_NAME = "prepare-docker-storage"
ACTION_USES = "./.github/actions/prepare-docker-storage"
PREPARE_STEP_NAME = "Prepare Docker storage capacity"
MERGE_SCRIPT = Path(".github/actions/prepare-docker-storage/merge-daemon-config.sh")


def _load_steps(project_root: Path) -> list[dict[str, object]]:
    """Load the composite action's ordered steps.

    :param project_root: Checkout containing the action contract.
    :returns: Steps in declaration order.
    """
    action = load_composite_action(project_root, ACTION_NAME)
    runs = cast(dict[str, object], action["runs"])
    assert runs["using"] == "composite"
    return cast(list[dict[str, object]], runs["steps"])


def _step_names(project_root: Path, workflow_name: str, job_name: str) -> list[str]:
    """Load step names from the selected workflow job.

    :param project_root: Checkout containing the workflow contract.
    :param workflow_name: Workflow whose ordering is under test.
    :param job_name: Job whose ordering is under test.
    :returns: Step names in declaration order.
    """
    workflow = load_workflow(project_root, workflow_name)
    jobs = cast(dict[str, object], workflow["jobs"])
    job = cast(dict[str, object], jobs[job_name])
    steps = cast(list[dict[str, object]], job["steps"])
    return [cast(str, step.get("name", "")) for step in steps]


def _prepare_script(project_root: Path) -> str:
    """Load the relocation action's shell script.

    :param project_root: Checkout containing the action contract.
    :returns: Script executed by the composite action.
    """
    steps = _load_steps(project_root)
    assert len(steps) == 1
    step = steps[0]
    assert step["name"] == PREPARE_STEP_NAME
    assert step["shell"] == "bash"
    return cast(str, step["run"])


@pytest.mark.infra
def test_prepare_docker_storage_uses_separate_writable_mnt_when_available(
    project_root: Path,
) -> None:
    """The action selects `/mnt` only when it is separate and writable.

    :param project_root: Checkout containing the action contract.
    """
    script = _prepare_script(project_root)
    assert "mountpoint -q /mnt" in script
    assert "findmnt" in script
    assert "--target /" in script
    assert "--target /mnt" in script
    assert "mktemp /mnt/" in script
    assert 'docker_root="/mnt/docker"' in script


@pytest.mark.infra
def test_prepare_docker_storage_falls_back_to_clean_root_store(project_root: Path) -> None:
    """Runners without a separate `/mnt` still reset disposable Docker state.

    :param project_root: Checkout containing the action contract.
    """
    script = _prepare_script(project_root)
    assert 'docker_root="/var/lib/docker"' in script
    assert "::warning::/mnt is not separate storage" in script
    assert "exit 1" not in script.split("sudo systemctl stop", maxsplit=1)[0]


@pytest.mark.infra
def test_prepare_docker_storage_stops_docker_before_clearing_state(project_root: Path) -> None:
    """Docker's service and socket stop before disposable stores are removed.

    :param project_root: Checkout containing the action contract.
    """
    script = _prepare_script(project_root)
    stop = "sudo systemctl stop docker.service docker.socket"
    clear = 'sudo rm -rf /var/lib/docker "${docker_root}"'
    assert stop in script
    assert clear in script
    assert script.index(stop) < script.index(clear)


@pytest.mark.infra
def test_prepare_docker_storage_atomically_merges_daemon_config(project_root: Path) -> None:
    """The action delegates daemon configuration to the atomic merge script.

    :param project_root: Checkout containing the action contract.
    """
    script = _prepare_script(project_root)
    assert (
        'sudo "${GITHUB_ACTION_PATH}/merge-daemon-config.sh" '
        '"${daemon_json}" "${docker_root}"' in script
    )


@pytest.mark.infra
def test_merge_daemon_config_preserves_existing_settings(
    project_root: Path, tmp_path: Path
) -> None:
    """Relocation preserves unrelated daemon settings.

    :param project_root: Checkout containing the executable merge script.
    :param tmp_path: Isolated daemon configuration directory.
    """
    config = tmp_path / "daemon.json"
    config.write_text('{"features":{"containerd-snapshotter":true},"log-level":"warn"}\n')

    subprocess.run(  # noqa: S603 — repository-owned executable and isolated path
        [project_root / MERGE_SCRIPT, config, "/mnt/docker"], check=True
    )

    assert json.loads(config.read_text()) == {
        "data-root": "/mnt/docker",
        "features": {"containerd-snapshotter": True},
        "log-level": "warn",
    }
    assert list(tmp_path.glob("daemon.json.*")) == []


@pytest.mark.infra
def test_merge_daemon_config_missing_file_creates_data_root(
    project_root: Path, tmp_path: Path
) -> None:
    """Relocation creates a valid daemon config when none exists.

    :param project_root: Checkout containing the executable merge script.
    :param tmp_path: Isolated daemon configuration directory.
    """
    config = tmp_path / "daemon.json"

    subprocess.run(  # noqa: S603 — repository-owned executable and isolated path
        [project_root / MERGE_SCRIPT, config, "/var/lib/docker"], check=True
    )

    assert json.loads(config.read_text()) == {"data-root": "/var/lib/docker"}


@pytest.mark.infra
def test_prepare_docker_storage_restarts_and_reports_capacity(project_root: Path) -> None:
    """The action starts Docker and emits daemon and filesystem diagnostics.

    :param project_root: Checkout containing the action contract.
    """
    script = _prepare_script(project_root)
    restart = "sudo systemctl start docker.socket docker.service"
    assert restart in script
    assert script.index("merge-daemon-config.sh") < script.index(restart)
    assert script.index(restart) < script.index("docker info")
    assert "df -h / /mnt" in script


@pytest.mark.infra
def test_generate_dataset_prepares_storage_after_reclamation_before_pull(
    project_root: Path,
) -> None:
    """Dataset generation relocates Docker after cleanup and before image pull.

    :param project_root: Checkout containing the workflow contract.
    """
    names = _step_names(project_root, "generate-dataset-shards.yaml", "generate")
    expected = [
        "Checkout",
        "Free disk space for kind load (skypilot-local row)",
        "Purge unused toolchains for kind headroom (skypilot-local row)",
        PREPARE_STEP_NAME,
        "Pull image (background)",
    ]
    assert [name for name in names if name in expected] == expected


@pytest.mark.infra
@pytest.mark.parametrize(
    ("workflow_name", "job_name"),
    [
        ("test-vst-slow.yml", "run_vst_slow_tests"),
        ("test-generate-dataset-shards.yml", "generate-dataset-shards"),
    ],
)
def test_docker_test_workflows_prepare_storage_after_checkout_before_pull(
    project_root: Path, workflow_name: str, job_name: str
) -> None:
    """Docker test jobs relocate storage immediately before pulling the image.

    :param project_root: Checkout containing the workflow contract.
    :param workflow_name: Docker test workflow under test.
    :param job_name: Docker test job under test.
    """
    names = _step_names(project_root, workflow_name, job_name)
    expected = ["Checkout", PREPARE_STEP_NAME, "Pull image"]
    assert [name for name in names if name in expected] == expected


@pytest.mark.infra
@pytest.mark.parametrize(
    ("workflow_name", "job_name"),
    [
        ("generate-dataset-shards.yaml", "generate"),
        ("test-vst-slow.yml", "run_vst_slow_tests"),
        ("test-generate-dataset-shards.yml", "generate-dataset-shards"),
    ],
)
def test_target_workflows_invoke_prepare_docker_storage_once(
    project_root: Path, workflow_name: str, job_name: str
) -> None:
    """Each affected Docker job delegates relocation to the shared action once.

    :param project_root: Checkout containing the workflow contract.
    :param workflow_name: Target workflow under test.
    :param job_name: Target Docker job under test.
    """
    workflow = load_workflow(project_root, workflow_name)
    jobs = cast(dict[str, object], workflow["jobs"])
    job = cast(dict[str, object], jobs[job_name])
    steps = cast(list[dict[str, object]], job["steps"])
    uses = [step.get("uses") for step in steps]
    assert uses.count(ACTION_USES) == 1
