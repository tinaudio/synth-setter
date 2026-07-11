"""Repository Python runtime pins stay aligned."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

PYTHON_VERSION = "3.12"
REQUIRES_PYTHON = ">=3.12,<3.14"


def test_project_requires_python_uses_python_312_floor(project_root: Path) -> None:
    """Package metadata rejects interpreters older than Python 3.12.

    :param project_root: Repository root fixture.
    """
    with (project_root / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)

    assert pyproject["project"]["requires-python"] == REQUIRES_PYTHON


def test_pyright_targets_python_312(project_root: Path) -> None:
    """Static analysis uses the same Python version as the runtime floor.

    :param project_root: Repository root fixture.
    """
    config = json.loads((project_root / "pyrightconfig.json").read_text())

    assert config["pythonVersion"] == PYTHON_VERSION


def test_docker_runtime_uses_python_312(project_root: Path) -> None:
    """The Docker runtime venv is created and checked with Python 3.12.

    :param project_root: Repository root fixture.
    """
    dockerfile = (project_root / "docker/ubuntu22_04/Dockerfile").read_text()

    assert "uv venv --python 3.12" in dockerfile
    assert "sys.version_info[:2] == (3, 12)" in dockerfile


def test_worker_checkout_recreates_stale_python_venv(project_root: Path) -> None:
    """PR workers repair stale dev-snapshot images after checkout.

    :param project_root: Repository root fixture.
    """
    script = (project_root / "scripts/sync_worker_checkout.sh").read_text()

    assert 'venv_python="$venv_dir/bin/python"' in script
    assert "sys.version_info[:2] != (3, 12)" in script
    assert 'venv_dir="${VIRTUAL_ENV:-/venv/main}"' in script
    assert 'rm -rf "$venv_dir"' in script
    assert 'uv venv --python 3.12 "$venv_dir"' in script


def test_tart_default_python_version_is_python_312(project_root: Path) -> None:
    """The macOS Tart image defaults to Python 3.12.

    :param project_root: Repository root fixture.
    """
    template = (project_root / "tart/macos.pkr.hcl").read_text()

    assert re.search(
        r'variable "python_version" \{.*?default\s+= "3\.12"',
        template,
        flags=re.DOTALL,
    )


@pytest.mark.parametrize(
    "workflow_path",
    [
        Path(".github/actions/setup-precommit/action.yml"),
        *sorted(Path(".github/workflows").glob("*.y*ml")),
    ],
)
def test_github_actions_do_not_pin_python_311(project_root: Path, workflow_path: Path) -> None:
    """CI no longer sets up Python 3.11.

    :param project_root: Repository root fixture.
    :param workflow_path: Workflow or composite-action YAML path to scan.
    """
    assert 'python-version: "3.11"' not in (project_root / workflow_path).read_text()
