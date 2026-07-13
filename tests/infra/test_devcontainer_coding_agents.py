"""Static contracts for coding agents bundled in the devcontainer image."""

from pathlib import Path

import pytest


@pytest.mark.infra
def test_devcontainer_tools_installs_hermes_and_pi(project_root: Path) -> None:
    """Verify the devcontainer image installs the Hermes and Pi CLIs.

    :param project_root: Root path of the repository under test.
    """
    dockerfile = (project_root / "docker" / "ubuntu22_04" / "Dockerfile").read_text()

    assert "node:22-bullseye-slim" in dockerfile
    assert "ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python" in dockerfile
    assert "ARG HERMES_GIT_REF=v2026.7.7.2" in dockerfile
    assert "ARG HERMES_GIT_SHA=9de9c25f620ff7f1ce0fd5457d596052d5159596" in dockerfile
    assert "ARG HERMES_INSTALLER_SHA256=" in dockerfile
    assert (
        "raw.githubusercontent.com/NousResearch/hermes-agent/${HERMES_GIT_REF}/scripts/install.sh"
        in dockerfile
    )
    assert (
        'env -u VIRTUAL_ENV bash /tmp/hermes-install.sh --branch "${HERMES_GIT_REF}" '
        '--commit "${HERMES_GIT_SHA}" --skip-browser'
    ) in dockerfile
    assert "--skip-browser" in dockerfile
    assert "@earendil-works/pi-coding-agent@${PI_VERSION}" in dockerfile
    assert "hermes --version" in dockerfile
    assert "pi --version" in dockerfile
