"""Regression contracts for the mutation-testing workflow."""

from __future__ import annotations

import ast
import fnmatch
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml
from mutmut.file_mutation import mutate_file_contents

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW_PATH = _REPO_ROOT / ".github/workflows/mutmut.yaml"
_PREDICT_MODULE = "synth_setter.evaluation.predict_vst_audio"
_PREDICT_SHARDS = {
    "evaluation-predict-misc": [
        f"{_PREDICT_MODULE}.x_[!_r]*",
        f"{_PREDICT_MODULE}.x_render_prediction_audio__mutmut_1",
        f"{_PREDICT_MODULE}.x_render_prediction_audio__mutmut_1?*",
        f"{_PREDICT_MODULE}.x_render_prediction_audio__mutmut_[5-9]",
        f"{_PREDICT_MODULE}.x_render_prediction_audio__mutmut_[5-9]?*",
        f"{_PREDICT_MODULE}.x__make_render_fn*",
        f"{_PREDICT_MODULE}.x__render_prediction_artifacts__mutmut_1",
        f"{_PREDICT_MODULE}.x__render_prediction_artifacts__mutmut_1?",
    ],
    "evaluation-predict-render-audio-leading-2": [
        f"{_PREDICT_MODULE}.x_render_prediction_audio__mutmut_2",
        f"{_PREDICT_MODULE}.x_render_prediction_audio__mutmut_2?*",
    ],
    "evaluation-predict-render-audio-leading-3-4": [
        f"{_PREDICT_MODULE}.x_render_prediction_audio__mutmut_[3-4]",
        f"{_PREDICT_MODULE}.x_render_prediction_audio__mutmut_[3-4]?*",
    ],
    "evaluation-predict-render-artifacts-leading-2-9-and-100-149": [
        f"{_PREDICT_MODULE}.x__render_prediction_artifacts__mutmut_[2-9]",
        f"{_PREDICT_MODULE}.x__render_prediction_artifacts__mutmut_[2-9]?*",
        f"{_PREDICT_MODULE}.x__render_prediction_artifacts__mutmut_1[0-4]?",
    ],
    "evaluation-predict-render-artifacts-150-199": [
        f"{_PREDICT_MODULE}.x__render_prediction_artifacts__mutmut_1[5-9]?",
    ],
}


def _workflow() -> dict[Any, Any]:
    """Load the mutation workflow.

    :returns: Parsed workflow mapping.
    """
    return yaml.safe_load(_WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_mutmut_workflow_shards_full_sandbox_with_bounded_jobs() -> None:
    """Each mutation selector maps to one job bounded to 40 minutes."""
    with (_REPO_ROOT / "pyproject.toml").open("rb") as file:
        mutation_paths = tomllib.load(file)["tool"]["mutmut"]["paths_to_mutate"]
    job = _workflow()["jobs"]["mutmut_run"]

    assert mutation_paths == [
        "src/synth_setter/evaluation/",
        "src/synth_setter/tools/",
        "src/synth_setter/pipeline/data/",
        "scripts/ci/",
    ]
    assert job["strategy"]["fail-fast"] is False
    shards = job["strategy"]["matrix"]["include"]
    assert len({shard["name"] for shard in shards}) == len(shards)
    patterns = []
    for shard in shards:
        patterns.extend(shard["patterns"])
    predict_shards = {
        shard["name"]: shard["patterns"]
        for shard in shards
        if shard["name"].startswith("evaluation-predict-")
    }
    assert predict_shards == _PREDICT_SHARDS

    for mutation_path in mutation_paths:
        root = _REPO_ROOT / mutation_path
        for source_path in root.rglob("*.py"):
            if source_path.name == "__init__.py":
                continue
            relative_path = source_path.relative_to(_REPO_ROOT).with_suffix("")
            parts = relative_path.parts
            module = ".".join(parts[1:] if parts[0] == "src" else parts)
            source = source_path.read_text(encoding="utf-8")
            if module == _PREDICT_MODULE:
                _, mutant_names = mutate_file_contents(str(source_path), source)
                selectors = [f"{module}.{name}" for name in mutant_names]
            else:
                tree = ast.parse(source)
                function_names = [
                    node.name
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                ]
                selectors = [f"{module}.x_{name}" for name in function_names] or [f"{module}.x"]
            for selector in selectors:
                assert sum(fnmatch.fnmatchcase(selector, pattern) for pattern in patterns) == 1
    assert job["timeout-minutes"] == 40
    run_step = next(step for step in job["steps"] if step["name"] == "Run mutmut")
    assert run_step["env"] == {
        "HYDRA_FULL_ERROR": "1",
        "MUTMUT_PATTERNS": "${{ toJson(matrix.patterns) }}",
    }
    assert run_step["run"] == (
        "set -euo pipefail\n"
        'patterns_json=$(jq -er \'if type == "array" and length > 0 and '
        'all(.[]; type == "string" and length > 0) then .[] else error("invalid mutation '
        'selector array") end\' <<<"$MUTMUT_PATTERNS")\n'
        "patterns=()\n"
        "while IFS= read -r pattern; do\n"
        '  patterns+=("${pattern}")\n'
        'done <<<"${patterns_json}"\n'
        'uv run mutmut run --max-children 4 "${patterns[@]}"\n'
    )


def _run_workflow_shell(
    tmp_path: Path, patterns_json: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run the checked-in mutation shell with a recording ``uv`` executable.

    :param tmp_path: Directory holding the recording executable.
    :param patterns_json: Selector payload supplied by the matrix expression.
    :returns: Bash result and the path receiving ``uv`` arguments.
    """
    uv_args_path = tmp_path / "uv-args"
    uv_path = tmp_path / "uv"
    uv_path.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" > "$UV_ARGS_PATH"\n')
    uv_path.chmod(0o755)
    env = os.environ | {
        "MUTMUT_PATTERNS": patterns_json,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "UV_ARGS_PATH": str(uv_args_path),
    }
    run_step = next(
        step for step in _workflow()["jobs"]["mutmut_run"]["steps"] if step["name"] == "Run mutmut"
    )
    bash_without_mapfile = "enable -n mapfile 2>/dev/null || true\n" + run_step["run"]
    result = subprocess.run(  # noqa: S603 — bash executes the checked-in workflow script
        ["/bin/bash", "-c", bash_without_mapfile],
        capture_output=True,
        env=env,
        text=True,
        timeout=10,
        check=False,
    )
    return result, uv_args_path


def test_mutmut_workflow_run_step_expands_every_selector(tmp_path: Path) -> None:
    """The workflow shell passes every matrix selector as a distinct argument.

    :param tmp_path: Directory holding a recording ``uv`` executable.
    """
    patterns = [
        "selector with spaces",
        r"selector\with\backslashes",
        "selector;$(printf not-executed) [!_]*?",
    ]

    result, uv_args_path = _run_workflow_shell(tmp_path, json.dumps(patterns))

    assert result.returncode == 0, result.stdout + result.stderr
    assert uv_args_path.read_text().splitlines() == [
        "run",
        "mutmut",
        "run",
        "--max-children",
        "4",
        *patterns,
    ]


def test_mutmut_workflow_run_step_with_invalid_selector_json_fails_closed(tmp_path: Path) -> None:
    """Malformed matrix JSON cannot fall through to an unfiltered mutation run.

    :param tmp_path: Directory holding a recording ``uv`` executable.
    """
    result, uv_args_path = _run_workflow_shell(tmp_path, "not-json")

    assert result.returncode != 0
    assert not uv_args_path.exists()


def test_mutmut_multi_selector_shard_runs_only_matching_mutants(tmp_path: Path) -> None:
    """A real mutmut run accepts every selector in a multi-pattern shard.

    :param tmp_path: Synthetic repository used by the real mutmut process.
    """
    synthetic_module = "sandbox.predict_vst_audio"
    source_dir = tmp_path / "src/sandbox"
    source_dir.mkdir(parents=True)
    (source_dir / "__init__.py").touch()
    (source_dir / "predict_vst_audio.py").write_text(
        "def _make_render_fn(value):\n"
        "    return value + 1\n\n"
        "def _render_prediction_artifacts(value):\n"
        "    first = value + 1\n"
        "    second = first * 2\n"
        "    third = second - 3\n"
        "    if third > 0:\n"
        "        return third / 4\n"
        "    return third + 5\n"
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_predict.py").write_text(
        "from sandbox.predict_vst_audio import (\n"
        "    _make_render_fn,\n"
        "    _render_prediction_artifacts,\n"
        ")\n\n"
        "def test_prediction_helpers():\n"
        "    assert _make_render_fn(1) == 2\n"
        "    assert _render_prediction_artifacts(3) == 1.25\n"
    )
    (tmp_path / "pyproject.toml").write_text(
        "[tool.mutmut]\n"
        'paths_to_mutate = ["src/sandbox/"]\n'
        'tests_dir = ["tests/"]\n'
        'pytest_add_cli_args = ["-q", "--capture=no"]\n'
        "[tool.pytest.ini_options]\n"
        'pythonpath = ["src"]\n'
    )
    patterns = [
        pattern.replace(_PREDICT_MODULE, synthetic_module)
        for pattern in _PREDICT_SHARDS["evaluation-predict-misc"]
    ]

    result = subprocess.run(  # noqa: S603 — sys.executable and argv are test-owned
        [
            sys.executable,
            "-c",
            "from mutmut.__main__ import cli; cli()",
            "run",
            "--max-children",
            "1",
            *patterns,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    metadata_path = tmp_path / "mutants/src/sandbox/predict_vst_audio.py.meta"
    statuses = json.loads(metadata_path.read_text())["exit_code_by_key"]
    for mutant_name, status in statuses.items():
        selected = any(fnmatch.fnmatchcase(mutant_name, pattern) for pattern in patterns)
        assert (status is not None) == selected
    assert any(status is None for status in statuses.values())


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
    ci_patterns = next(
        shard["patterns"]
        for shard in _workflow()["jobs"]["mutmut_run"]["strategy"]["matrix"]["include"]
        if shard["name"] == "ci-scripts"
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

    assert ci_patterns == ["scripts.ci.*"]
    result = subprocess.run(  # noqa: S603 — sys.executable and argv are test-owned
        [
            sys.executable,
            "-c",
            "from mutmut.__main__ import cli; cli()",
            "run",
            "--max-children",
            "1",
            *ci_patterns,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "mutants/mutmut-stats.json").is_file()
