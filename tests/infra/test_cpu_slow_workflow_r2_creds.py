"""`cpu-slow.yml` wires the `setup-r2` action so `integration_r2` tests run.

`integration_r2`-marked tests also carry `slow`, so they collect into
`cpu-slow.yml`'s pytest invocation. Without the rclone binary and the
`RCLONE_CONFIG_R2_*` env, `r2_io.is_r2_reachable()` short-circuits and they
skip silently — degrading coverage with no signal. That env now comes from the
shared `setup-r2` composite (whose own contract is pinned in
`test_setup_r2_action.py`); these tests pin that `cpu-slow.yml` invokes it,
with rclone install, before the pytest step (see #1185, #1353).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import pytest
from workflow_fixtures import load_workflow

SETUP_R2_USES = "./.github/actions/setup-r2"
SECRET_INPUT_TO_KEY: dict[str, str] = {
    "access-key-id": "RCLONE_CONFIG_R2_ACCESS_KEY_ID",
    "secret-access-key": "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY",
    "endpoint": "RCLONE_CONFIG_R2_ENDPOINT",
}

SETUP_R2_STEP_NAME = "Set up R2"
PYTEST_STEP_NAME = "Run slow (non-GPU, non-MPS, non-VST) tests"

WORKFLOW_SELF_PATH = ".github/workflows/cpu-slow.yml"
INVARIANT_TEST_SELF_PATH = "tests/infra/test_cpu_slow_workflow_r2_creds.py"


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
def test_cpu_slow_sets_up_r2_before_pytest(project_root: Path) -> None:
    """``Set up R2`` runs before the pytest step so the env is in place.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    """
    names = [step.get("name") for step in _load_workflow_steps(project_root)]
    assert SETUP_R2_STEP_NAME in names, f"cpu-slow.yml missing step {SETUP_R2_STEP_NAME!r}"
    assert names.index(SETUP_R2_STEP_NAME) < names.index(PYTEST_STEP_NAME), (
        f"{SETUP_R2_STEP_NAME!r} must precede {PYTEST_STEP_NAME!r} in cpu-slow.yml"
    )


@pytest.mark.infra
@pytest.mark.parametrize("expected_path", [WORKFLOW_SELF_PATH, INVARIANT_TEST_SELF_PATH])
def test_cpu_slow_pull_request_self_trigger_present(
    project_root: Path, expected_path: str
) -> None:
    """``on.pull_request.paths`` covers the workflow and its invariant test.

    A PR that edits ``cpu-slow.yml`` (or this test, whose contract pins the
    R2 wiring the workflow promises) must exercise the slow suite pre-merge
    instead of waiting for the post-merge push run — see #1206.

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

    Fork PRs can't see ``secrets.RCLONE_CONFIG_R2_*``, so the
    ``integration_r2`` surface skips anyway — but the rest of the slow
    suite still burns a 4-core runner for up to 90 minutes. The job-level
    ``if`` guard short-circuits ``pull_request`` events whose head repo
    differs from the workflow repo; ``workflow_dispatch`` / ``push`` /
    ``schedule`` runs are unaffected.

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
