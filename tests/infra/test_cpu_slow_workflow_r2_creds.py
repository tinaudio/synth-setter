"""Pin trusted-PR and non-PR R2 setup contracts in ``cpu-slow.yml``."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import pytest
from workflow_fixtures import load_composite_action, load_workflow

SETUP_R2_USES = "./.github/actions/setup-r2"
SECRET_INPUT_TO_KEY: dict[str, str] = {
    "access-key-id": "RCLONE_CONFIG_R2_ACCESS_KEY_ID",
    "secret-access-key": "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY",
    "endpoint": "RCLONE_CONFIG_R2_ENDPOINT",
}

SETUP_R2_STEP_NAME = "Set up R2"
PYTEST_STEP_NAME = "Run slow (non-GPU, non-MPS, non-VST) tests"
PR_PYTEST_STEP_NAME = "Run slow PR tests"
PR_R2_E2E_STEP_NAME = "Run growing Lance R2 E2E"

INSTALL_RCLONE_ACTION_PATH = ".github/actions/install-rclone/**"
WORKFLOW_SELF_PATH = ".github/workflows/cpu-slow.yml"
TESTS_PATH = "tests/**"


def _load_workflow(project_root: Path) -> dict[str, object]:
    """Return the parsed ``cpu-slow.yml`` workflow document.

    :param project_root: repo root; the workflow is read from
        ``<project_root>/.github/workflows/cpu-slow.yml``.
    :returns: the parsed YAML mapping.
    """
    return cast(dict[str, object], load_workflow(project_root, "cpu-slow.yml"))


def _load_run_slow_tests_job(project_root: Path) -> dict[str, object]:
    """Return the ``run_slow_tests`` job mapping from ``cpu-slow.yml``.

    :param project_root: repo root; the workflow is read from
        ``<project_root>/.github/workflows/cpu-slow.yml``.
    :returns: the job mapping, including ``if``, ``steps``, ``runs-on``, etc.
    """
    workflow = _load_workflow(project_root)
    jobs = cast(dict[str, object], workflow["jobs"])
    return cast(dict[str, object], jobs["run_slow_tests"])


def _load_workflow_steps(project_root: Path) -> list[dict[str, object]]:
    """Return the ordered ``steps`` list of the ``run_slow_tests`` job.

    :param project_root: repo root; the workflow is read from
        ``<project_root>/.github/workflows/cpu-slow.yml``.
    :returns: the job's ``steps`` list.
    """
    job = _load_run_slow_tests_job(project_root)
    return cast(list[dict[str, object]], job["steps"])


def _setup_r2_step(project_root: Path) -> dict[str, object]:
    """Return the step that invokes the ``setup-r2`` composite.

    :param project_root: repo root; the workflow is read from
        ``<project_root>/.github/workflows/cpu-slow.yml``.
    :returns: the matching step mapping; fails the test if absent.
    """
    for step in _load_workflow_steps(project_root):
        if step.get("uses") == SETUP_R2_USES:
            return step
    pytest.fail(f"cpu-slow.yml missing a step that `uses: {SETUP_R2_USES}`")


def _load_pull_request_paths(project_root: Path) -> list[str]:
    """Return the ``on.pull_request.paths`` filter list from ``cpu-slow.yml``.

    PyYAML parses the bare ``on`` key as the boolean ``True``, so callers
    cannot index the document with the string ``"on"``; we resolve whichever
    key the loader actually produced before reading the trigger block.

    :param project_root: repo root; the workflow is read from
        ``<project_root>/.github/workflows/cpu-slow.yml``.
    :returns: the configured ``paths`` list; fails the test if the
        ``pull_request`` trigger or its ``paths`` filter is missing.
    """
    workflow = _load_workflow(project_root)
    on_key: object = "on" if "on" in workflow else True
    triggers = cast(dict[str, object], workflow[on_key])  # type: ignore[index]
    pull_request = triggers.get("pull_request")
    if not isinstance(pull_request, dict):
        pytest.fail(
            "cpu-slow.yml missing `on.pull_request` trigger — PRs that touch the "
            "workflow must exercise it pre-merge (see #1206)"
        )
    paths = pull_request.get("paths")
    if not isinstance(paths, list):
        pytest.fail(
            "cpu-slow.yml `on.pull_request` missing `paths` filter — without it "
            "every PR would trigger the slow suite (see #1206)"
        )
    return cast(list[str], paths)


@pytest.mark.infra
def test_cpu_slow_pins_production_faust_toolchain(project_root: Path) -> None:
    """The native parity lane verifies the worker image's Faust version.

    :param project_root: Session fixture rooted at the repository checkout.
    """
    job = _load_run_slow_tests_job(project_root)
    version_step = next(
        step
        for step in _load_workflow_steps(project_root)
        if step.get("name") == "Verify production Faust version"
    )

    assert job["runs-on"] == "ubuntu-22.04"
    assert version_step["run"] == 'faust --version 2>&1 | grep -F "FAUST Version 2.37.3"'


@pytest.mark.infra
def test_install_rclone_action_remains_secret_free(project_root: Path) -> None:
    """The shared rclone installer never reads or configures storage credentials.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    """
    install_action = load_composite_action(project_root, "install-rclone")
    serialized_action = str(install_action)
    assert "RCLONE_CONFIG_R2" not in serialized_action
    assert "SYNTH_SETTER_STORAGE" not in serialized_action
    assert "secrets." not in serialized_action


@pytest.mark.infra
def test_cpu_slow_uses_setup_r2_with_rclone_install(project_root: Path) -> None:
    """The R2 step installs rclone so ``is_r2_reachable()`` finds the binary.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    """
    step = _setup_r2_step(project_root)
    install = cast(dict[str, object], step.get("with") or {}).get("install-rclone")
    assert str(install).lower() == "true", (
        "cpu-slow.yml runs on a native runner with no rclone preinstalled — it "
        "must pass `install-rclone: true` to setup-r2 or `integration_r2` tests "
        "skip via is_r2_reachable() (see #1185)"
    )


@pytest.mark.infra
@pytest.mark.parametrize(("input_name", "key"), sorted(SECRET_INPUT_TO_KEY.items()))
def test_cpu_slow_setup_r2_secret_inputs_reference_matching_secrets(
    project_root: Path, input_name: str, key: str
) -> None:
    """Each secret input sources from the same-named ``secrets.RCLONE_CONFIG_R2_*``.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :param input_name: the ``setup-r2`` input under test.
    :param key: the secret name it must reference; parametrized so a missing
        wiring fails this case independently with the key in the output.
    """
    with_block = cast(dict[str, object], _setup_r2_step(project_root).get("with") or {})
    actual = with_block.get(input_name)
    pattern = rf"\$\{{\{{\s*secrets\.{re.escape(key)}\s*\}}\}}"
    assert isinstance(actual, str) and re.fullmatch(pattern, actual), (
        f"cpu-slow.yml setup-r2 `with.{input_name}` = {actual!r}; expected a "
        f"`${{{{ secrets.{key} }}}}` expression"
    )


@pytest.mark.infra
def test_cpu_slow_sets_up_r2_before_non_pr_pytest(project_root: Path) -> None:
    """``Set up R2`` precedes the broad live-R2 pytest step.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    """
    names = [step.get("name") for step in _load_workflow_steps(project_root)]
    assert names.index(SETUP_R2_STEP_NAME) < names.index(PYTEST_STEP_NAME)


@pytest.mark.infra
def test_cpu_slow_sets_up_r2_before_pr_e2e(project_root: Path) -> None:
    """``Set up R2`` precedes the targeted pull-request E2E step.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    """
    names = [step.get("name") for step in _load_workflow_steps(project_root)]
    assert names.index(SETUP_R2_STEP_NAME) < names.index(PR_R2_E2E_STEP_NAME)


@pytest.mark.infra
def test_cpu_slow_r2_steps_restrict_pr_secrets_to_same_repo(project_root: Path) -> None:
    """R2 credentials and the targeted E2E are unavailable to fork pull requests.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    """
    trusted_pr_clause = "github.event.pull_request.head.repo.full_name == github.repository"
    setup_guard = cast(str, _setup_r2_step(project_root).get("if", ""))
    assert "github.event_name != 'pull_request'" in setup_guard
    assert trusted_pr_clause in setup_guard

    e2e_step = next(
        step
        for step in _load_workflow_steps(project_root)
        if step.get("name") == PR_R2_E2E_STEP_NAME
    )
    e2e_guard = cast(str, e2e_step.get("if", ""))
    assert "github.event_name == 'pull_request'" in e2e_guard
    assert trusted_pr_clause in e2e_guard
    assert e2e_step.get("run") == "make test-ci-slow-pr-r2-e2e"


@pytest.mark.infra
def test_cpu_slow_pr_e2e_runs_after_prior_failure_when_r2_setup_succeeds(
    project_root: Path,
) -> None:
    """The targeted E2E ignores prior failures but requires successful R2 setup.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    """
    setup_step = _setup_r2_step(project_root)
    assert setup_step.get("id") == "setup_r2"

    e2e_step = next(
        step
        for step in _load_workflow_steps(project_root)
        if step.get("name") == PR_R2_E2E_STEP_NAME
    )
    e2e_guard = cast(str, e2e_step.get("if", ""))
    assert "!cancelled()" in e2e_guard
    assert "steps.setup_r2.outcome == 'success'" in e2e_guard


@pytest.mark.infra
@pytest.mark.parametrize(
    "expected_path", [INSTALL_RCLONE_ACTION_PATH, WORKFLOW_SELF_PATH, TESTS_PATH]
)
def test_cpu_slow_pull_request_self_trigger_present(
    project_root: Path, expected_path: str
) -> None:
    """``on.pull_request.paths`` covers the workflow and its invariant test.

    A PR that edits ``cpu-slow.yml`` or any test must exercise the slow suite
    pre-merge instead of waiting for the post-merge push run — see #1206.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :param expected_path: a repo-relative path that must appear verbatim in
        the ``paths`` filter; parametrized so a missing entry names the
        offending path in the failure output.
    """
    paths = _load_pull_request_paths(project_root)
    assert expected_path in paths, (
        f"cpu-slow.yml `on.pull_request.paths` missing {expected_path!r}; "
        f"got {paths!r} — PRs touching that file must trigger the workflow "
        f"pre-merge (see #1206)"
    )


@pytest.mark.infra
def test_cpu_slow_job_gated_against_fork_prs(project_root: Path) -> None:
    """``run_slow_tests.if`` blocks fork-PR runs of the 90-min suite.

    The job-level guard explicitly skips fork pull requests. Same-repository
    PRs can run the targeted R2 E2E without exposing secrets to forks.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    """
    job = _load_run_slow_tests_job(project_root)
    guard = job.get("if")
    assert isinstance(guard, str), (
        "cpu-slow.yml `run_slow_tests` missing job-level `if:` guard — "
        "fork-PR runs of the 90-min slow suite are unrestricted (see #1206)"
    )
    fork_guard_pattern = re.compile(
        r"github\.event\.pull_request\.head\.repo\.full_name"
        r"\s*==\s*"
        r"github\.repository"
    )
    assert fork_guard_pattern.search(guard), (
        f"cpu-slow.yml `run_slow_tests.if` = {guard!r}; expected a "
        "`github.event.pull_request.head.repo.full_name == github.repository` "
        "clause so fork PRs skip the 90-min slow suite (see #1206)"
    )
