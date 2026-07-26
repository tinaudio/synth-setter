"""Regression contracts for the mutation-testing workflow."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW_PATH = _REPO_ROOT / ".github/workflows/mutmut.yaml"
_MUTANT_SHARDS = [
    {"name": "evaluation", "pattern": "synth_setter.evaluation.*"},
    {"name": "tools", "pattern": "synth_setter.tools.*"},
    {"name": "pipeline-data", "pattern": "synth_setter.pipeline.data.*"},
    {"name": "ci-scripts", "pattern": "scripts.ci.*"},
]


def _workflow() -> dict[Any, Any]:
    """Load the mutation workflow.

    :returns: Parsed workflow mapping.
    """
    return yaml.safe_load(_WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_mutmut_workflow_shards_full_sandbox_with_bounded_jobs() -> None:
    """Each configured mutation root runs in a parallel job bounded to 40 minutes."""
    with (_REPO_ROOT / "pyproject.toml").open("rb") as file:
        mutation_paths = tomllib.load(file)["tool"]["mutmut"]["paths_to_mutate"]
    job = _workflow()["jobs"]["mutmut_run"]

    assert mutation_paths == [
        "src/synth_setter/evaluation/",
        "src/synth_setter/tools/",
        "src/synth_setter/pipeline/data/",
        "scripts/ci/",
    ]
    assert job["strategy"] == {"fail-fast": False, "matrix": {"include": _MUTANT_SHARDS}}
    assert job["timeout-minutes"] == 40
    run_step = next(step for step in job["steps"] if step["name"] == "Run mutmut")
    assert run_step["run"] == 'uv run mutmut run --max-children 4 "${{ matrix.pattern }}"'


def test_mutmut_configuration_disables_pytest_capture_for_in_process_runner() -> None:
    """The in-process pytest runner avoids W&B's import-time stream wrapper conflict."""
    with (_REPO_ROOT / "pyproject.toml").open("rb") as file:
        config = tomllib.load(file)["tool"]["mutmut"]

    assert "--capture=no" in config["pytest_add_cli_args"]


def test_mutmut_workflow_always_reports_each_shard_without_pr_trigger() -> None:
    """Every shard prints and uploads results while pull-request gating stays deferred."""
    workflow = _workflow()
    triggers = workflow[True]
    assert "pull_request" not in triggers

    steps = {step["name"]: step for step in workflow["jobs"]["mutmut_run"]["steps"]}
    assert steps["Print results"]["if"] == "always()"
    upload = steps["Upload mutants/ artifact"]
    assert upload["if"] == "always()"
    assert upload["with"]["name"] == "mutmut-${{ matrix.name }}-mutants"
    assert upload["with"]["path"] == "mutants/"


def test_mutmut_shard_command_with_configured_pytest_args_completes(tmp_path: Path) -> None:
    """A real mutmut sandbox survives W&B import and accepts the CI shard pattern.

    :param tmp_path: Synthetic repository used by the real mutmut process.
    """
    with (_REPO_ROOT / "pyproject.toml").open("rb") as file:
        pytest_args = tomllib.load(file)["tool"]["mutmut"]["pytest_add_cli_args"]
    ci_pattern = next(
        shard["pattern"] for shard in _MUTANT_SHARDS if shard["name"] == "ci-scripts"
    )

    source_dir = tmp_path / "scripts/ci"
    source_dir.mkdir(parents=True)
    (source_dir / "smoke.py").write_text("def increment(value):\n    return value + 1\n")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_smoke.py").write_text(
        "import wandb\n"
        "from scripts.ci.smoke import increment\n\n"
        "def test_increment_returns_successor():\n"
        "    assert increment(1) == 2\n"
    )
    (tmp_path / "pyproject.toml").write_text(
        "[tool.mutmut]\n"
        'paths_to_mutate = ["scripts/ci/"]\n'
        'tests_dir = ["tests/"]\n'
        f"pytest_add_cli_args = {json.dumps(pytest_args)}\n"
    )

    assert ci_pattern == "scripts.ci.*"
    result = subprocess.run(  # noqa: S603 — sys.executable and argv are test-owned
        [
            sys.executable,
            "-c",
            "from mutmut.__main__ import cli; cli()",
            "run",
            "--max-children",
            "1",
            "scripts.ci.*",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "mutants/mutmut-stats.json").is_file()
