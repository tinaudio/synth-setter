"""Static contracts for coding agents bundled in the devcontainer image."""

import re
from pathlib import Path

import pytest

_INFISICAL_RELEASE_URL_PREFIX = (
    "https://github.com/Infisical/infisical/releases/download/infisical-cli/v"
)


def _capture(pattern: str, text: str) -> str:
    """Return the single capture group of ``pattern``, asserting it matches once.

    :param pattern: Regular expression carrying exactly one capture group.
    :param text: File contents searched for the pin.
    :returns: The captured pin value.
    """
    matches = re.findall(pattern, text)
    assert len(matches) == 1, f"expected one match for {pattern!r}, got {matches}"
    return matches[0]


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
        "env -u VIRTUAL_ENV -u UV_PYTHON_INSTALL_DIR bash /tmp/hermes-install.sh "
        '--branch "${HERMES_GIT_REF}" --commit "${HERMES_GIT_SHA}" --skip-browser'
    ) in dockerfile
    assert "--skip-browser" in dockerfile
    assert "@earendil-works/pi-coding-agent@${PI_VERSION}" in dockerfile
    assert "hermes --version" in dockerfile
    assert "pi --version" in dockerfile


@pytest.mark.infra
def test_devcontainer_tools_declares_pinned_infisical_cli(project_root: Path) -> None:
    """Verify the Dockerfile pins the Infisical CLI and verifies its checksum.

    The built-image smoke test verifies this declaration installs and runs the CLI.

    :param project_root: Root path of the repository under test.
    """
    dockerfile = (project_root / "docker" / "ubuntu22_04" / "Dockerfile").read_text()

    assert re.search(r"ARG INFISICAL_VERSION=\d+\.\d+\.\d+", dockerfile)
    assert re.search(r"ARG INFISICAL_SHA256_AMD64=[0-9a-f]{64}", dockerfile)
    assert re.search(r"ARG INFISICAL_SHA256_ARM64=[0-9a-f]{64}", dockerfile)
    assert "infisical_${INFISICAL_VERSION}_linux_${TARGETARCH}.deb" in dockerfile
    assert 'echo "${infisical_sha}  /tmp/${package}" | sha256sum -c -' in dockerfile
    assert "infisical --version" in dockerfile


@pytest.mark.infra
def test_devcontainer_tools_fetches_infisical_from_its_release_tag(project_root: Path) -> None:
    """Verify the Dockerfile fetches Infisical from the upstream release, not Cloudsmith.

    Cloudsmith stopped serving the ``deb/debian/pool`` layout the pin was written
    against, so every version under that host 404s (#3682).

    :param project_root: Root path of the repository under test.
    """
    dockerfile = (project_root / "docker" / "ubuntu22_04" / "Dockerfile").read_text()

    assert _INFISICAL_RELEASE_URL_PREFIX + "${INFISICAL_VERSION}/" in dockerfile
    assert "dl.cloudsmith.io" not in dockerfile


@pytest.mark.infra
def test_pod_bootstrap_fetches_infisical_from_its_release_tag(project_root: Path) -> None:
    """Verify the vastai bootstrap fetches Infisical from the same live host.

    :param project_root: Root path of the repository under test.
    """
    script = (project_root / "scripts" / "runpod" / "bootstrap-vastai-pytorch-pod.sh").read_text()

    assert _INFISICAL_RELEASE_URL_PREFIX + "${INFISICAL_VERSION}/" in script
    assert "dl.cloudsmith.io" not in script


@pytest.mark.infra
def test_infisical_pins_agree_across_both_install_sites(project_root: Path) -> None:
    """Verify the Dockerfile and the pod bootstrap install the same Infisical build.

    The two sites pin the version and checksum independently, so a bump applied to one alone leaves
    the other fetching a build whose checksum can no longer match.

    :param project_root: Root path of the repository under test.
    """
    dockerfile = (project_root / "docker" / "ubuntu22_04" / "Dockerfile").read_text()
    script = (project_root / "scripts" / "runpod" / "bootstrap-vastai-pytorch-pod.sh").read_text()

    dockerfile_pins = (
        _capture(r"ARG INFISICAL_VERSION=(\S+)", dockerfile),
        _capture(r"ARG INFISICAL_SHA256_AMD64=(\S+)", dockerfile),
    )
    script_pins = (
        _capture(r"readonly INFISICAL_VERSION=(\S+)", script),
        _capture(r"readonly INFISICAL_SHA256=(\S+)", script),
    )

    assert dockerfile_pins == script_pins


@pytest.mark.infra
def test_infisical_smoke_expectation_matches_the_pinned_version(project_root: Path) -> None:
    """Verify the in-image smoke test expects the version the Dockerfile installs.

    That smoke test runs where the repository is absent, so it carries the version as a literal and
    can only be kept honest from outside the image.

    :param project_root: Root path of the repository under test.
    """
    dockerfile = (project_root / "docker" / "ubuntu22_04" / "Dockerfile").read_text()
    smoke = (project_root / "tests" / "docker" / "test_devcontainer_tools.py").read_text()

    pinned = _capture(r"ARG INFISICAL_VERSION=(\S+)", dockerfile)

    assert f'"infisical version {pinned}"' in smoke


@pytest.mark.infra
def test_devcontainer_tools_installs_latest_zellij_release(project_root: Path) -> None:
    """Verify image builds resolve Zellij through GitHub's latest-release URL.

    :param project_root: Root path of the repository under test.
    """
    dockerfile = (project_root / "docker" / "ubuntu22_04" / "Dockerfile").read_text()

    assert 'base_url="https://github.com/zellij-org/zellij/releases/latest/download"' in dockerfile
    assert '"${base_url}/zellij-${zellij_arch}-unknown-linux-musl.tar.gz"' in dockerfile
    assert "ARG ZELLIJ_VERSION=" not in dockerfile


@pytest.mark.infra
def test_devcontainer_tools_verifies_latest_zellij_release(project_root: Path) -> None:
    """Verify image builds check Zellij against its published release checksum.

    :param project_root: Root path of the repository under test.
    """
    dockerfile = (project_root / "docker" / "ubuntu22_04" / "Dockerfile").read_text()

    assert '"${base_url}/zellij-${zellij_arch}-unknown-linux-musl.sha256sum"' in dockerfile
    assert 'echo "${zellij_sha}  /usr/local/bin/zellij" | sha256sum -c -' in dockerfile


@pytest.mark.infra
def test_hermes_installer_does_not_inherit_the_root_owned_uv_python_dir(
    project_root: Path,
) -> None:
    """The hermes install runs as a non-root user and must not write to /opt/uv.

    ``UV_PYTHON_INSTALL_DIR=/opt/uv/python`` is set image-wide and root-owned,
    while the stage has already dropped to ``USER $USERNAME``. Inheriting it
    fails the build with ``Permission denied`` fetching a managed interpreter.

    :param project_root: Root path of the repository under test.
    """
    dockerfile = (project_root / "docker" / "ubuntu22_04" / "Dockerfile").read_text()
    install_line = next(
        line for line in dockerfile.splitlines() if "hermes-install.sh --branch" in line
    )

    assert "-u UV_PYTHON_INSTALL_DIR" in install_line
    assert "-u VIRTUAL_ENV" in install_line
