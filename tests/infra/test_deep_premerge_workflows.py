"""Contracts for path-scoped deep pull-request validation workflows."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import cast

import pytest
import sh
from workflow_fixtures import load_workflow

YamlMapping = dict[object, object]
WorkflowSet = dict[str, YamlMapping]

DISPATCH_DATASET_CONFIG = "generate_dataset/smoke-shard-lance"

CPU_SLOW_PR_PATHS: frozenset[str] = frozenset(
    {
        ".github/actions/install-rclone/**",
        ".github/workflows/cpu-slow.yml",
        "Makefile",
        "pyproject.toml",
        "scripts/ci/**",
        "src/synth_setter/**",
        "tests/**",
        "uv.lock",
    }
)
CODE_QUALITY_PR_PATHS: frozenset[str] = frozenset(
    {
        ".editorconfig",
        ".github/actions/setup-precommit/**",
        ".github/workflows/code-quality-main.yaml",
        ".gitlint",
        ".pre-commit-config.yaml",
        ".pydoclint-baseline.txt",
        "agent/hooks/**",
        "checkmake.ini",
        "pyproject.toml",
        "pyrightconfig.json",
        "scripts/check_no_new_funcs_in_pydoclint_excluded.py",
        "scripts/ci/count_doc_noqa.sh",
        "scripts/ci/pr_body_lint.py",
        "uv.lock",
    }
)
DATASET_GENERATION_PR_PATHS: frozenset[str] = frozenset(
    {
        ".github/workflows/generate-dataset-shards.yaml",
        ".github/workflows/test-dataset-generation.yml",
        ".github/workflows/validate-dataset-shards.yaml",
        "src/synth_setter/cli/generate_dataset.py",
        "src/synth_setter/configs/dataset.yaml",
        "src/synth_setter/configs/experiment/generate_dataset/**",
        "src/synth_setter/pipeline/skypilot_launch.py",
        "tests/infra/test_dataset_generation_matrix.py",
    }
)
SPEC_MATERIALIZATION_PR_PATHS: frozenset[str] = frozenset(
    {
        ".github/workflows/spec-materialization.yml",
        ".github/workflows/test-spec-materialization.yml",
        "src/synth_setter/configs/experiment/generate_dataset/ci-materialize-test.yaml",
        "src/synth_setter/pipeline/ci/materialize_spec.py",
        "src/synth_setter/pipeline/ci/validate_spec.py",
        "tests/pipeline/ci_config/test_materialize_spec.py",
    }
)


def _mapping(value: object, context: str) -> YamlMapping:
    """Narrow a parsed YAML value to a mapping.

    :param value: Parsed YAML value.
    :param context: Location named when the value has the wrong shape.
    :returns: Value narrowed to a YAML mapping.
    """
    assert isinstance(value, dict), f"{context} must be a mapping"
    return cast(YamlMapping, value)


def _string(mapping: YamlMapping, key: str) -> str:
    """Return a required string field from a YAML mapping.

    :param mapping: Parsed YAML mapping.
    :param key: Required field name.
    :returns: String field value.
    """
    value = mapping[key]
    assert isinstance(value, str), f"{key} must be a string"
    return value


def _triggers(workflow: YamlMapping) -> YamlMapping:
    """Return a workflow's event trigger mapping.

    :param workflow: Parsed workflow document.
    :returns: Mapping under the YAML ``on`` key.
    """
    value = workflow.get("on")
    if value is None:
        value = workflow.get(True)
    return _mapping(value, "workflow triggers")


def _pull_request_paths(workflow: YamlMapping) -> set[str]:
    """Return the configured pull-request path filters.

    :param workflow: Parsed workflow document.
    :returns: Pull-request path filter values.
    """
    pull_request = _mapping(_triggers(workflow)["pull_request"], "pull_request trigger")
    paths = pull_request["paths"]
    assert isinstance(paths, list) and all(isinstance(path, str) for path in paths)
    return set(paths)


def _job(workflow: YamlMapping, name: str) -> YamlMapping:
    """Return a named job from a parsed workflow.

    :param workflow: Parsed workflow document.
    :param name: Job identifier.
    :returns: Parsed job mapping.
    """
    jobs = _mapping(workflow["jobs"], "workflow jobs")
    return _mapping(jobs[name], f"job {name}")


def _steps(workflow: YamlMapping, job_name: str) -> list[YamlMapping]:
    """Return a job's ordered step mappings.

    :param workflow: Parsed workflow document.
    :param job_name: Job identifier.
    :returns: Parsed step mappings.
    """
    value = _job(workflow, job_name)["steps"]
    assert isinstance(value, list), f"job {job_name} steps must be a list"
    return [_mapping(step, f"job {job_name} step") for step in value]


def _named_step(workflow: YamlMapping, job_name: str, step_name: str) -> YamlMapping:
    """Return a step selected by its display name.

    :param workflow: Parsed workflow document.
    :param job_name: Job identifier.
    :param step_name: Step display name.
    :returns: Matching step mapping.
    """
    return next(step for step in _steps(workflow, job_name) if step.get("name") == step_name)


def _run_dataset_matrix(
    workflow: YamlMapping,
    tmp_path: Path,
    *,
    event_name: str,
    schedule: str = "",
    provider_input: str = "runpod",
) -> dict[str, str]:
    """Execute the dataset workflow's real matrix resolver script.

    :param workflow: Parsed dataset-generation workflow.
    :param tmp_path: Directory receiving the synthetic ``GITHUB_OUTPUT`` file.
    :param event_name: GitHub event name supplied to the resolver.
    :param schedule: Schedule cron value, empty for non-schedule events.
    :param provider_input: Manual-dispatch provider selection.
    :returns: Resolver outputs written through ``GITHUB_OUTPUT``.
    """
    matrix_step = next(step for step in _steps(workflow, "setup") if step.get("id") == "matrix")
    output_path = tmp_path / "github-output"
    env = os.environ | {
        "DISPATCH_DATASET_CONFIG": DISPATCH_DATASET_CONFIG,
        "EVENT_NAME": event_name,
        "GITHUB_OUTPUT": str(output_path),
        "PROVIDER_INPUT": provider_input,
        "SCHEDULE_CRON": schedule,
    }
    bash_path = shutil.which("bash")
    assert bash_path is not None, "bash is required to execute workflow run blocks"
    sh.Command(bash_path)("-c", _string(matrix_step, "run"), _env=env)
    return dict(line.split("=", maxsplit=1) for line in output_path.read_text().splitlines())


@pytest.fixture(scope="module")
def workflows(project_root: Path) -> WorkflowSet:
    """Load workflows changed by the deep pre-merge validation feature.

    :param project_root: Repository root containing the workflow documents.
    :returns: Parsed workflows keyed by filename.
    """
    filenames = (
        "code-quality-main.yaml",
        "cpu-slow.yml",
        "test-dataset-generation.yml",
        "test-spec-materialization.yml",
    )
    return {name: cast(YamlMapping, load_workflow(project_root, name)) for name in filenames}


def test_cpu_slow_pr_paths_cover_owned_surfaces(workflows: WorkflowSet) -> None:
    """Slow CPU PR validation follows source, tests, and runner configuration.

    :param workflows: Four parsed workflow documents keyed by filename.
    """
    assert CPU_SLOW_PR_PATHS <= _pull_request_paths(workflows["cpu-slow.yml"])


def test_cpu_slow_pr_lane_installs_rclone_without_r2_setup(workflows: WorkflowSet) -> None:
    """Pull requests install rclone without R2 credentials or configuration.

    :param workflows: Four parsed workflow documents keyed by filename.
    """
    workflow = workflows["cpu-slow.yml"]
    setup_r2 = next(
        step
        for step in _steps(workflow, "run_slow_tests")
        if step.get("uses") == "./.github/actions/setup-r2"
    )
    assert "github.event_name != 'pull_request'" in _string(setup_r2, "if")

    install_rclone = next(
        step
        for step in _steps(workflow, "run_slow_tests")
        if step.get("uses") == "./.github/actions/install-rclone"
    )
    assert "github.event_name == 'pull_request'" in _string(install_rclone, "if")
    assert "secrets." not in str(install_rclone)

    pr_step = _named_step(workflow, "run_slow_tests", "Run slow PR tests")
    assert "github.event_name == 'pull_request'" in _string(pr_step, "if")
    assert _string(pr_step, "run") == "make test-ci-slow-pr"
    assert "RCLONE_CONFIG_R2" not in str(pr_step)


def test_cpu_slow_non_pr_lane_preserves_live_r2_target(workflows: WorkflowSet) -> None:
    """Push and dispatch runs retain the live-R2 slow target.

    :param workflows: Four parsed workflow documents keyed by filename.
    """
    test_step = _named_step(
        workflows["cpu-slow.yml"],
        "run_slow_tests",
        "Run slow (non-GPU, non-MPS, non-VST) tests",
    )
    assert "github.event_name != 'pull_request'" in _string(test_step, "if")
    assert _string(test_step, "run") == "make test-ci-slow"


def test_cpu_slow_non_pr_events_and_failure_filing_remain_unchanged(
    workflows: WorkflowSet,
) -> None:
    """Dispatch, main pushes, and post-merge failure filing remain enabled.

    :param workflows: Four parsed workflow documents keyed by filename.
    """
    workflow = workflows["cpu-slow.yml"]
    triggers = _triggers(workflow)
    assert triggers["workflow_dispatch"] is None
    assert triggers["push"] == {
        "branches": ["main"],
        "paths-ignore": ["docs/**", "**/*.md"],
    }
    filing_step = _named_step(workflow, "run_slow_tests", "Auto-file failure ticket")
    assert _string(filing_step, "if") == (
        "failure() && github.event_name == 'push' && github.ref == 'refs/heads/main'"
    )


def test_code_quality_main_pr_paths_cover_global_hook_infrastructure(
    workflows: WorkflowSet,
) -> None:
    """Global lint configuration changes trigger all-files validation.

    :param workflows: Four parsed workflow documents keyed by filename.
    """
    workflow = workflows["code-quality-main.yaml"]
    assert CODE_QUALITY_PR_PATHS <= _pull_request_paths(workflow)
    step = _named_step(workflow, "code-quality", "Run pre-commits")
    assert _string(step, "run") == "uv run pre-commit run --all-files"


def test_code_quality_main_skips_branch_guard_only_on_push(workflows: WorkflowSet) -> None:
    """Main pushes skip the branch guard while pull requests execute it.

    :param workflows: Four parsed workflow documents keyed by filename.
    """
    step = _named_step(workflows["code-quality-main.yaml"], "code-quality", "Run pre-commits")
    env = _mapping(step["env"], "pre-commit environment")
    skip = _string(env, "SKIP")
    assert "github.event_name == 'push'" in skip
    assert "no-commit-to-branch" in skip
    assert "|| ''" in skip


def test_dataset_generation_pr_paths_cover_local_lance_pipeline(
    workflows: WorkflowSet,
) -> None:
    """Relevant local generation changes trigger the production-path smoke test.

    :param workflows: Four parsed workflow documents keyed by filename.
    """
    assert DATASET_GENERATION_PR_PATHS <= _pull_request_paths(
        workflows["test-dataset-generation.yml"]
    )


def test_dataset_generation_pr_matrix_is_local_lance_static_only(
    workflows: WorkflowSet, tmp_path: Path
) -> None:
    """Pull requests cannot select paid providers or queue scenarios.

    :param workflows: Four parsed workflow documents keyed by filename.
    :param tmp_path: Directory receiving the synthetic ``GITHUB_OUTPUT`` file.
    """
    workflow = workflows["test-dataset-generation.yml"]
    guard = _string(_job(workflow, "setup"), "if")
    assert "github.event_name != 'pull_request'" in guard
    assert "github.event.pull_request.head.repo.full_name == github.repository" in guard
    assert "github.actor != 'dependabot[bot]'" in guard
    assert _run_dataset_matrix(workflow, tmp_path, event_name="pull_request") == {
        "output_formats": '["lance"]',
        "providers": '["skypilot-local"]',
        "scenarios": '["static"]',
    }
    assert "needs.setup.result == 'success'" in _string(_job(workflow, "validate"), "if")


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        (
            "0 * * * *",
            {
                "output_formats": '["lance"]',
                "providers": '["skypilot-local"]',
                "scenarios": '["static"]',
            },
        ),
        (
            "0 7 * * 0",
            {
                "output_formats": '["lance"]',
                "providers": '["skypilot-local","runpod"]',
                "scenarios": '["static","queue"]',
            },
        ),
    ],
)
def test_dataset_generation_schedule_matrices_target_supported_providers(
    workflows: WorkflowSet,
    tmp_path: Path,
    schedule: str,
    expected: dict[str, str],
) -> None:
    """Hourly and weekly matrices target local and RunPod providers.

    :param workflows: Four parsed workflow documents keyed by filename.
    :param tmp_path: Directory receiving the synthetic ``GITHUB_OUTPUT`` file.
    :param schedule: Cron value under test.
    :param expected: Exact resolved matrix outputs.
    """
    assert (
        _run_dataset_matrix(
            workflows["test-dataset-generation.yml"],
            tmp_path,
            event_name="schedule",
            schedule=schedule,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("provider_input", "expected_providers"),
    [
        ("all", '["skypilot-local","runpod"]'),
        ("skypilot-local", '["skypilot-local"]'),
        ("runpod", '["runpod"]'),
    ],
)
def test_dataset_generation_dispatch_matrix_resolves_supported_providers(
    workflows: WorkflowSet,
    tmp_path: Path,
    provider_input: str,
    expected_providers: str,
) -> None:
    """Manual dispatch resolves each supported provider selection.

    :param workflows: Four parsed workflow documents keyed by filename.
    :param tmp_path: Directory receiving the synthetic ``GITHUB_OUTPUT`` file.
    :param provider_input: Manual-dispatch provider selection.
    :param expected_providers: JSON provider array emitted by the resolver.
    """
    assert _run_dataset_matrix(
        workflows["test-dataset-generation.yml"],
        tmp_path,
        event_name="workflow_dispatch",
        provider_input=provider_input,
    ) == {
        "output_formats": '["lance"]',
        "providers": expected_providers,
        "scenarios": '["static","queue"]',
    }


@pytest.mark.parametrize("provider_input", ["aws", "oci"])
def test_dataset_generation_dispatch_matrix_rejects_unsupported_providers(
    workflows: WorkflowSet, tmp_path: Path, provider_input: str
) -> None:
    """Manual dispatch rejects removed and unknown provider selections.

    :param workflows: Four parsed workflow documents keyed by filename.
    :param tmp_path: Directory receiving the synthetic ``GITHUB_OUTPUT`` file.
    :param provider_input: Unsupported manual-dispatch provider selection.
    """
    with pytest.raises(sh.ErrorReturnCode):
        _run_dataset_matrix(
            workflows["test-dataset-generation.yml"],
            tmp_path,
            event_name="workflow_dispatch",
            provider_input=provider_input,
        )


def test_spec_materialization_pr_paths_cover_materializer_and_validator(
    workflows: WorkflowSet,
) -> None:
    """Spec composition and validation changes trigger the PR-safe lane.

    :param workflows: Four parsed workflow documents keyed by filename.
    """
    assert SPEC_MATERIALIZATION_PR_PATHS <= _pull_request_paths(
        workflows["test-spec-materialization.yml"]
    )


def test_spec_materialization_pr_job_uses_local_docker_without_secrets(
    workflows: WorkflowSet,
) -> None:
    """PR materialization composes in Docker and validates without live R2.

    :param workflows: Four parsed workflow documents keyed by filename.
    """
    workflow = workflows["test-spec-materialization.yml"]
    manual_job = _job(workflow, "materialize")
    assert "github.event_name == 'workflow_dispatch'" in _string(manual_job, "if")
    assert manual_job["secrets"] == "inherit"

    pr_job = _job(workflow, "materialize-pr")
    assert "github.event_name == 'pull_request'" in _string(pr_job, "if")
    serialized = str(pr_job)
    assert "secrets." not in serialized
    assert "setup-r2" not in serialized
    assert "synth_setter.pipeline.ci.materialize_spec" in serialized
    assert "synth_setter.pipeline.ci.validate_spec" in serialized
    assert "--test-values" in serialized
