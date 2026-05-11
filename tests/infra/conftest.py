"""Shared fixtures for the `tests/infra` suite.

Self-contained — does NOT import from `tests/conftest.py`, which pulls in torch/Hydra/h5py/VST
fixtures unrelated to infrastructure checks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT_ANCHOR = ".project-root"


def _find_project_root(start: Path) -> Path:
    """Walk up from `start` until a directory containing `.project-root` is found."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / PROJECT_ROOT_ANCHOR).is_file():
            return candidate
    raise RuntimeError(f"Could not find {PROJECT_ROOT_ANCHOR} walking up from {start}")


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the repository root (the dir containing `.project-root`)."""
    return _find_project_root(Path(__file__).parent)


@pytest.fixture(scope="session")
def devcontainer_dir(project_root: Path) -> Path:
    """Absolute path to the `.devcontainer/` directory."""
    return project_root / ".devcontainer"


@pytest.fixture(scope="session")
def devcontainer_json_paths(devcontainer_dir: Path) -> list[Path]:
    """Paths to the three `devcontainer.json` files (cpu, gpu, root_gpu)."""
    paths = [
        devcontainer_dir / flavor / "devcontainer.json" for flavor in ("cpu", "gpu", "root_gpu")
    ]
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"Expected devcontainer config not found: {path}")
    return paths


@pytest.fixture(scope="session")
def post_create_script(devcontainer_dir: Path) -> Path:
    """Absolute path to `.devcontainer/post-create.sh`."""
    return devcontainer_dir / "post-create.sh"


@pytest.fixture(scope="session")
def initialize_script(devcontainer_dir: Path) -> Path:
    """Absolute path to `.devcontainer/initialize.sh`."""
    return devcontainer_dir / "initialize.sh"


@pytest.fixture(scope="session")
def dockerfile_path(devcontainer_dir: Path) -> Path:
    """Absolute path to `.devcontainer/Dockerfile`."""
    return devcontainer_dir / "Dockerfile"
