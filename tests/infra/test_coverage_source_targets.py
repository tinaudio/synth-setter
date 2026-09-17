"""Coverage sources in CI name directories on disk, never importable dotted modules."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from workflow_fixtures import WORKFLOWS_DIR

# `--cov=<value>`; `--cov-report=` and `--cov-branch` deliberately do not match.
_COV_SOURCE = re.compile(r"--cov=(\S+)")


def _run_scripts(node: object) -> list[str]:
    """Collect every `run:` script in a parsed workflow document.

    :param node: A node of the parsed YAML document.
    :returns: The `run:` scripts reachable from `node`.
    """
    if isinstance(node, dict):
        own = [value for key, value in node.items() if key == "run" and isinstance(value, str)]
        return own + [script for value in node.values() for script in _run_scripts(value)]
    if isinstance(node, list):
        return [script for value in node for script in _run_scripts(value)]
    return []


def _coverage_sources(project_root: Path) -> list[tuple[str, str]]:
    """Collect every `--cov=` argument a workflow passes to pytest.

    :param project_root: Repo root supplied by the infra test fixtures.
    :returns:`(workflow filename, coverage source)` pairs.
    """
    sources = []
    workflow_dir = project_root / WORKFLOWS_DIR
    for workflow_path in sorted([*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")]):
        document = yaml.safe_load(workflow_path.read_text())
        for script in _run_scripts(document):
            sources += [(workflow_path.name, value) for value in _COV_SOURCE.findall(script)]
    return sources


@pytest.mark.infra
def test_workflow_coverage_sources_name_directories_on_disk(project_root: Path) -> None:
    """No CI job asks coverage to resolve its source by import.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    offenders = [
        f"{workflow}: --cov={source}"
        for workflow, source in _coverage_sources(project_root)
        if not (project_root / source).is_dir()
    ]

    assert not offenders, (
        "A coverage source that is not a directory is either a dotted module — which coverage "
        "resolves by importing the parent package before tests/conftest.py runs — or a single "
        f"file, which collects nothing (#3276): {offenders}"
    )


@pytest.mark.infra
def test_dotted_coverage_source_imports_the_package_root_and_a_directory_does_not(
    tmp_path: Path,
) -> None:
    """Coverage imports `pkg` to locate `pkg.leaf`, which runs the package root's side effects.

    :param tmp_path: Scratch directory holding the probe package.
    """
    marker = tmp_path / "package-root-imported"
    package = tmp_path / "probe_pkg"
    package.mkdir()
    (package / "__init__.py").write_text(
        f"from pathlib import Path\n\nPath({str(marker)!r}).touch()\n"
    )
    (package / "leaf.py").write_text("VALUE = 1\n")
    (tmp_path / "workload.py").write_text("VALUE = 2\n")

    def coverage_run(source: str) -> None:
        marker.unlink(missing_ok=True)
        subprocess.run(  # noqa: S603 — sys.executable with a test-owned argv
            [sys.executable, "-m", "coverage", "run", f"--source={source}", "workload.py"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    coverage_run("probe_pkg.leaf")
    assert marker.exists()

    coverage_run("probe_pkg")
    assert not marker.exists()
