"""Repository Python runtime pins stay aligned."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement

PYTHON_MINOR = "3.12"
PYTHON_VERSION = "3.12.13"
REQUIRES_PYTHON = ">=3.12,<3.13"


def _python_version_values(value: object) -> list[str]:
    """Return scalar values assigned to any ``python-version`` key.

    :param value: Parsed workflow YAML value.
    :returns: Normalized scalar version values, including matrix members.
    """
    if isinstance(value, dict):
        versions: list[str] = []
        for key, child in value.items():
            if key == "python-version":
                versions.extend(_scalar_values(child))
            versions.extend(_python_version_values(child))
        return versions
    if isinstance(value, list):
        return [version for child in value for version in _python_version_values(child)]
    return []


def _scalar_values(value: object) -> list[str]:
    """Flatten a YAML scalar or list into normalized strings.

    :param value: Value assigned to a ``python-version`` key.
    :returns: String form of each scalar member.
    """
    if isinstance(value, list):
        return [version for child in value for version in _scalar_values(child)]
    return [str(value)]


def _workflow_paths(project_root: Path) -> list[Path]:
    """Return every workflow and Python-provisioning composite action.

    :param project_root: Repository root fixture.
    :returns: Absolute YAML paths whose Python pins must stay aligned.
    """
    return [
        project_root / ".github/actions/setup-precommit/action.yml",
        *sorted((project_root / ".github/workflows").glob("*.y*ml")),
    ]


def test_project_requires_python_uses_python_312_floor(project_root: Path) -> None:
    """Package metadata rejects interpreters older than Python 3.12.

    :param project_root: Repository root fixture.
    """
    with (project_root / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)

    assert pyproject["project"]["requires-python"] == REQUIRES_PYTHON


@pytest.mark.parametrize(
    ("sys_platform", "platform_machine", "expected"),
    [
        ("linux", "x86_64", True),
        ("linux", "aarch64", False),
        ("darwin", "x86_64", True),
        ("darwin", "arm64", True),
        ("win32", "AMD64", True),
        ("win32", "ARM64", False),
    ],
)
def test_dawdreamer_dependency_excludes_unsupported_worker_targets(
    project_root: Path, sys_platform: str, platform_machine: str, expected: bool
) -> None:
    """DawDreamer installs only where its CPython 3.12 wheel exists.

    :param project_root: Repository root fixture.
    :param sys_platform: PEP 508 operating-system value.
    :param platform_machine: PEP 508 machine-architecture value.
    :param expected: Whether DawDreamer should install on the target.
    """
    with (project_root / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)

    dependency = next(
        dependency
        for dependency in pyproject["dependency-groups"]["audio"]
        if dependency.startswith("dawdreamer")
    )
    requirement = Requirement(dependency)

    assert requirement.marker is not None
    assert (
        requirement.marker.evaluate(
            {"sys_platform": sys_platform, "platform_machine": platform_machine}
        )
        is expected
    )


def test_pyright_targets_python_312(project_root: Path) -> None:
    """Static analysis uses the same Python version as the runtime floor.

    :param project_root: Repository root fixture.
    """
    config = json.loads((project_root / "pyrightconfig.json").read_text())

    assert config["pythonVersion"] == PYTHON_MINOR


def test_python_source_tools_target_python_312(project_root: Path) -> None:
    """Ruff and Python pre-commit hooks parse the supported syntax floor.

    :param project_root: Repository root fixture.
    """
    with (project_root / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)
    precommit = yaml.safe_load((project_root / ".pre-commit-config.yaml").read_text())

    assert pyproject["tool"]["ruff"]["target-version"] == "py312"
    assert precommit["default_language_version"]["python"] == "python3.12"
    interrogate = next(
        repository for repository in precommit["repos"] if "interrogate" in repository["repo"]
    )
    assert interrogate["hooks"][0]["language_version"] == "python3.12"


def test_local_environment_provisioning_pins_python_31213(project_root: Path) -> None:
    """Developer environment entry points use the canonical patch release.

    :param project_root: Repository root fixture.
    """
    assert (project_root / ".python-version").read_text().strip() == PYTHON_VERSION
    assert f"python={PYTHON_VERSION}" in (project_root / "environment.yaml").read_text()
    assert f"venv --python {PYTHON_VERSION}" in (project_root / "Makefile").read_text()


def test_docker_runtime_uses_python_312(project_root: Path) -> None:
    """The Docker runtime venv is created and checked with Python 3.12.

    :param project_root: Repository root fixture.
    """
    dockerfile = (project_root / "docker/ubuntu22_04/Dockerfile").read_text()

    assert f"uv venv --python {PYTHON_VERSION}" in dockerfile
    assert "sys.version_info[:2] == (3, 12)" in dockerfile


def test_worker_checkout_uses_guarded_runtime_repair(project_root: Path) -> None:
    """PR workers repair stale dev-snapshot images after checkout.

    :param project_root: Repository root fixture.
    """
    script = (project_root / "scripts/sync_worker_checkout.sh").read_text()
    helper = (project_root / "scripts/ensure_worker_python.sh").read_text()

    assert "source scripts/ensure_worker_python.sh" in script
    assert "rm -rf" not in script
    assert 'local worker_venv="/venv/main"' in helper
    assert f"uv venv --python {PYTHON_VERSION}" in helper


def test_tart_default_python_version_is_python_312(project_root: Path) -> None:
    """The macOS Tart image defaults to Python 3.12.

    :param project_root: Repository root fixture.
    """
    template = (project_root / "tart/macos.pkr.hcl").read_text()

    assert re.search(
        rf'variable "python_version" \{{.*?default\s+= "{re.escape(PYTHON_VERSION)}"',
        template,
        flags=re.DOTALL,
    )


@pytest.mark.parametrize(
    ("yaml_text", "expected"),
    [
        ('python-version: "3.11"', ["3.11"]),
        ("python-version: '3.11.9'", ["3.11.9"]),
        ("python-version: 3.11", ["3.11"]),
        ("python-version: 3.11.x", ["3.11.x"]),
        ('python-version: ["3.11", "3.12.13"]', ["3.11", "3.12.13"]),
    ],
)
def test_python_version_values_reads_yaml_forms(yaml_text: str, expected: list[str]) -> None:
    """Python-version discovery handles scalars and matrix lists.

    :param yaml_text: Synthetic workflow fragment.
    :param expected: Normalized versions expected from the fragment.
    """
    assert _python_version_values(yaml.safe_load(yaml_text)) == expected


def test_github_actions_pin_canonical_python_from_any_cwd(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CI Python pins stay exact when pytest starts outside the repository.

    :param project_root: Repository root fixture.
    :param tmp_path: Pytest-provided directory outside the repository.
    :param monkeypatch: Pytest fixture used to change the process directory.
    """
    monkeypatch.chdir(tmp_path)

    version_values = {
        path.relative_to(project_root): _python_version_values(yaml.safe_load(path.read_text()))
        for path in _workflow_paths(project_root)
    }
    literal_versions = {
        path: [value for value in values if not value.startswith("${{")]
        for path, values in version_values.items()
        if values
    }

    assert len(version_values) > 20
    assert literal_versions
    assert all(
        values and set(values) == {PYTHON_VERSION} for values in literal_versions.values()
    ), literal_versions


def test_launcher_cache_key_pins_canonical_python(project_root: Path) -> None:
    """The launcher environment cache changes with the operational Python pin.

    :param project_root: Repository root fixture.
    """
    workflow = (project_root / ".github/workflows/generate-dataset-shards.yaml").read_text()

    cache_token = PYTHON_VERSION.replace(".", "")
    assert f"launcher-uv-${{{{ runner.os }}}}-py{cache_token}-" in workflow
