"""Pin `[project.dependencies]` to the lite import closure (issue #1139).

Lite CI jobs install the project with a bare `pip install -e .` and rely on the
base staying light — only the union closure of the three lite entrypoints
(`validate_spec.main`, `r2_io.ensure_r2_env_loaded`, `load_image_config.main`).
The heavy runtime lives in PEP 735 `[dependency-groups]`, which a plain
`pip install -e .` does not pull in. These tests fail fast if the base silently
grows a heavy dep back in.

Rationale: docs/reference/dependency-management.md.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# Names a `Requirement` line may carry before its version/extra/marker suffix.
_NAME_DELIMS = "<>=!~[; "

# The lite union closure: validate_spec (pydantic + python-dotenv via r2_io) and
# load_image_config (pydantic + pyyaml). r2_io alone needs only python-dotenv.
LITE_CLOSURE = {"pydantic", "python-dotenv", "pyyaml"}

# Heavy deps that must never reappear in the base; the lite env must import the
# three entrypoints without any of these.
HEAVY_DEPS = {
    "torch",
    "torchvision",
    "torchaudio",
    "lightning",
    "torchmetrics",
    "hydra-core",
    "hydra-colorlog",
    "hydra-optuna-sweeper",
    "omegaconf",
    "skypilot",
    "runpod",
    "oci",
    "kubernetes",
    "librosa",
    "pedalboard",
    "h5py",
    "hdf5plugin",
    "dask",
    "webdataset",
    "pandas",
    "numpy",
    "scipy",
    "pesto-pitch",
    "kymatio",
    "pot",
    "wandb",
    "tensorboard",
}


def _requirement_name(spec: str) -> str:
    """Return the PEP 503-normalized distribution name from a PEP 508 requirement.

    :param spec: A requirement string, e.g. ``"pydantic>=2"`` or ``"dask[distributed]"``.
    :returns: Name lowercased with runs of ``-``/``_``/``.`` collapsed to a single
        ``-``, so equivalent spellings compare equal against ``LITE_CLOSURE`` /
        ``HEAVY_DEPS``.
    """
    name = spec.strip()
    for delim in _NAME_DELIMS:
        name = name.split(delim, 1)[0]
    return re.sub(r"[-_.]+", "-", name.strip().lower())


@pytest.fixture(scope="session")
def project_dependency_names(project_root: Path) -> set[str]:
    """Return normalized distribution names declared in `[project.dependencies]`.

    :param project_root: Repo root holding ``pyproject.toml`` (from conftest).
    :returns: The base install's declared dependency names.
    """
    with (project_root / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)
    return {_requirement_name(dep) for dep in pyproject["project"]["dependencies"]}


def test_project_dependencies_equal_lite_closure(project_dependency_names: set[str]) -> None:
    """`[project.dependencies]` is exactly the lite union closure — nothing more.

    :param project_dependency_names: Declared base dependency names (fixture).
    """
    assert project_dependency_names == LITE_CLOSURE


def test_project_dependencies_exclude_heavy_runtime(project_dependency_names: set[str]) -> None:
    """No heavy runtime dep leaks into the base lite install.

    :param project_dependency_names: Declared base dependency names (fixture).
    """
    leaked = project_dependency_names & HEAVY_DEPS
    assert not leaked, f"heavy deps leaked into [project.dependencies]: {sorted(leaked)}"
