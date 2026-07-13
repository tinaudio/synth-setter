"""Static contracts for coding agents bundled in the devcontainer image."""

from pathlib import Path

import pytest


@pytest.mark.infra
def test_devcontainer_tools_installs_hermes_and_pi(project_root: Path) -> None:
    """Verify the devcontainer image installs the Hermes and Pi CLIs.

    :param project_root: Root path of the repository under test.
    """
    dockerfile = (project_root / "docker" / "ubuntu22_04" / "Dockerfile").read_text()

    assert "hermes-agent.nousresearch.com/install.sh" in dockerfile
    assert "--skip-browser" in dockerfile
    assert "@earendil-works/pi-coding-agent@${PI_VERSION}" in dockerfile
    assert "hermes --version" in dockerfile
    assert "pi --version" in dockerfile
