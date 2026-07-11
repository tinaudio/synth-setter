"""`setup-r2` exports canonical storage env and projected rclone env.

The action maps the repo's existing R2 secret names into canonical
``SYNTH_SETTER_STORAGE_*`` variables, then keeps exporting the rclone projection
that current workflow shells still need. These tests pin both contracts so the
action cannot drift from the storage settings model or the current backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from workflow_fixtures import load_composite_action, load_workflow

from synth_setter.pipeline.schemas.object_storage import (
    ENV_STORAGE_ACCESS_KEY_ID,
    ENV_STORAGE_ENDPOINT_URL,
    ENV_STORAGE_PROVIDER,
    ENV_STORAGE_SECRET_ACCESS_KEY,
    RCLONE_REQUIRED_ENV_KEYS,
    RCLONE_STRUCTURAL_DEFAULTS,
    ObjectStoreProvider,
)

ACTION_NAME = "setup-r2"
EXPORT_STEP_NAME = "Export R2 credentials"
INSTALL_STEP_NAME = "Install rclone"

# input name -> the `RCLONE_REQUIRED_ENV_KEYS` member it supplies.
SECRET_INPUTS: dict[str, str] = {
    "access-key-id": "RCLONE_CONFIG_R2_ACCESS_KEY_ID",
    "secret-access-key": "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY",
    "endpoint": "RCLONE_CONFIG_R2_ENDPOINT",
}

STORAGE_SECRET_EXPORTS: dict[str, str] = {
    "access-key-id": ENV_STORAGE_ACCESS_KEY_ID,
    "secret-access-key": ENV_STORAGE_SECRET_ACCESS_KEY,
    "endpoint": ENV_STORAGE_ENDPOINT_URL,
}

STORAGE_STRUCTURAL_EXPORTS: dict[str, str] = {
    ENV_STORAGE_PROVIDER: ObjectStoreProvider.R2.value,
}


def _load_action(project_root: Path) -> dict[str, object]:
    """Return the parsed ``setup-r2`` ``action.yml`` document.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :returns: the parsed YAML mapping.
    """
    return cast(dict[str, object], load_composite_action(project_root, ACTION_NAME))


def _load_steps(project_root: Path) -> list[dict[str, object]]:
    """Return the composite action's ``runs.steps`` list.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :returns: the ordered list of step mappings.
    """
    runs = cast(dict[str, object], _load_action(project_root)["runs"])
    return cast(list[dict[str, object]], runs["steps"])


def _find_step(project_root: Path, name: str) -> dict[str, object]:
    """Return the action step named ``name``, failing the test if absent.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :param name: the step ``name:`` to locate.
    :returns: the matching step mapping.
    """
    for step in _load_steps(project_root):
        if step.get("name") == name:
            return step
    pytest.fail(f"setup-r2 action missing step {name!r}")


@pytest.mark.infra
def test_setup_r2_is_a_composite_action(project_root: Path) -> None:
    """The action runs as a `composite` so callers `uses:` it like any step.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    """
    runs = cast(dict[str, object], _load_action(project_root)["runs"])
    assert runs.get("using") == "composite"


@pytest.mark.infra
def test_setup_r2_declares_required_secret_inputs(project_root: Path) -> None:
    """Each secret rclone needs is a required input, not hard-coded.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    """
    inputs = cast(dict[str, object], _load_action(project_root)["inputs"])
    for input_name in SECRET_INPUTS:
        spec = cast(dict[str, object], inputs.get(input_name) or {})
        assert spec.get("required") is True, (
            f"setup-r2 input {input_name!r} must be `required: true` — it carries a secret"
        )


@pytest.mark.infra
@pytest.mark.parametrize(("key", "expected"), sorted(RCLONE_STRUCTURAL_DEFAULTS.items()))
def test_setup_r2_exports_structural_literals(project_root: Path, key: str, expected: str) -> None:
    """The export step writes each structural key with its canonical literal.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :param key: one of the rclone structural defaults.
    :param expected: the literal rclone needs (``s3`` / ``Cloudflare``); drift
        from ``RCLONE_STRUCTURAL_DEFAULTS`` fails here loudly.
    """
    run = cast(str, _find_step(project_root, EXPORT_STEP_NAME)["run"])
    assert f"{key}={expected}" in run, (
        f"setup-r2 export step must write `{key}={expected}` to $GITHUB_ENV"
    )


@pytest.mark.infra
@pytest.mark.parametrize(("key", "expected"), sorted(STORAGE_STRUCTURAL_EXPORTS.items()))
def test_setup_r2_exports_canonical_storage_structural_literals(
    project_root: Path, key: str, expected: str
) -> None:
    """The export step writes each non-secret storage setting with its literal.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :param key: canonical ``SYNTH_SETTER_STORAGE_*`` env key.
    :param expected: the literal required by current CI wiring.
    """
    run = cast(str, _find_step(project_root, EXPORT_STEP_NAME)["run"])
    assert f"{key}={expected}" in run, (
        f"setup-r2 export step must write `{key}={expected}` to $GITHUB_ENV"
    )


@pytest.mark.infra
def test_setup_r2_appends_to_github_env(project_root: Path) -> None:
    """The export step persists the vars to `$GITHUB_ENV` for later job steps.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    """
    run = cast(str, _find_step(project_root, EXPORT_STEP_NAME)["run"])
    assert "GITHUB_ENV" in run and ">>" in run


@pytest.mark.infra
@pytest.mark.parametrize(("input_name", "env_key"), sorted(SECRET_INPUTS.items()))
def test_setup_r2_routes_secrets_through_step_env(
    project_root: Path, input_name: str, env_key: str
) -> None:
    """Secrets reach `$GITHUB_ENV` via the step env, never spliced into the script.

    Interpolating ``${{ inputs.<secret> }}`` straight into a ``run:`` body would
    expose the value to shell history / process listings; passing it through the
    step ``env`` keeps it a masked variable. This pins both that the secret is
    wired up and that the export key matches a ``RCLONE_REQUIRED_ENV_KEYS`` member.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :param input_name: composite input carrying the secret.
    :param env_key: the ``RCLONE_CONFIG_R2_*`` key it is exported as.
    """
    assert env_key in RCLONE_REQUIRED_ENV_KEYS
    step = _find_step(project_root, EXPORT_STEP_NAME)
    env = cast(dict[str, str], step.get("env") or {})
    referenced = [v for v in env.values() if f"inputs.{input_name}" in v]
    assert referenced, (
        f"setup-r2 export step env must reference `inputs.{input_name}` so the "
        f"secret is not inlined into the run script"
    )
    run = cast(str, step["run"])
    env_var = next(name for name, val in env.items() if f"inputs.{input_name}" in val)
    assert f"{env_key}=${{{env_var}}}" in run or f"{env_key}=${env_var}" in run, (
        f"setup-r2 export step must write `{env_key}` from the `${env_var}` step-env var"
    )


@pytest.mark.infra
@pytest.mark.parametrize(("input_name", "env_key"), sorted(STORAGE_SECRET_EXPORTS.items()))
def test_setup_r2_exports_canonical_storage_secrets(
    project_root: Path, input_name: str, env_key: str
) -> None:
    """Existing R2 secrets are mapped into canonical storage settings.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :param input_name: composite input carrying the secret.
    :param env_key: canonical ``SYNTH_SETTER_STORAGE_*`` key exported by the action.
    """
    step = _find_step(project_root, EXPORT_STEP_NAME)
    env = cast(dict[str, str], step.get("env") or {})
    run = cast(str, step["run"])
    env_var = next(name for name, val in env.items() if f"inputs.{input_name}" in val)
    assert f"{env_key}=${{{env_var}}}" in run or f"{env_key}=${env_var}" in run, (
        f"setup-r2 export step must write `{env_key}` from the `${env_var}` step-env var"
    )


@pytest.mark.infra
def test_setup_r2_install_rclone_is_opt_in(project_root: Path) -> None:
    """`install-rclone` defaults off and gates the apt-get install step.

    Docker-based callers ship rclone in the image and pass the default; only
    native runners opt in. The gate keeps the action a no-op install elsewhere.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    """
    inputs = cast(dict[str, object], _load_action(project_root)["inputs"])
    install_spec = cast(dict[str, object], inputs.get("install-rclone") or {})
    assert str(install_spec.get("default")).lower() == "false"

    step = _find_step(project_root, INSTALL_STEP_NAME)
    assert "install-rclone" in cast(str, step.get("if", ""))
    assert "rclone" in cast(str, step["run"])


SETUP_R2_USES = "./.github/actions/setup-r2"
INLINE_R2_ENV_KEY = "RCLONE_CONFIG_R2_ACCESS_KEY_ID:"
AUTH_PROBE_WORKFLOW = "r2-auth-probe.yaml"
PURE_RCLONE_DOCKER_COMMANDS = ("rclone purge",)


def _workflows_using_setup_r2(project_root: Path) -> list[Path]:
    """Return every workflow file whose YAML invokes the ``setup-r2`` action.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :returns: workflow paths under ``.github/workflows`` referencing the action.
    """
    workflows_dir = project_root / ".github" / "workflows"
    paths = [*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")]
    return [p for p in paths if SETUP_R2_USES in p.read_text()]


@pytest.mark.infra
def test_setup_r2_callers_do_not_reinline_r2_env(project_root: Path) -> None:
    """No workflow both delegates to ``setup-r2`` and re-inlines the R2 env.

    The marker side has `test_workflow_has_no_inline_pytest_marker` as its
    anti-drift backstop; this is the R2 analogue. A workflow that calls the
    action AND keeps an inline ``RCLONE_CONFIG_R2_ACCESS_KEY_ID:`` env key has a
    half-finished migration — the duplicate would silently outlive a future
    edit to the action. (The ``with: access-key-id:`` input form does not match
    this key, so passing the secret to the action is fine.)

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    """
    callers = _workflows_using_setup_r2(project_root)
    assert callers, "no workflow references setup-r2 — wiring regressed (see #1353)"
    offenders = [p.name for p in callers if INLINE_R2_ENV_KEY in p.read_text()]
    assert not offenders, (
        f"workflows delegate to setup-r2 but still inline {INLINE_R2_ENV_KEY!r}: "
        f"{sorted(offenders)} — drop the inline env and rely on the action (#1353)"
    )


@pytest.mark.infra
@pytest.mark.parametrize(
    "key",
    sorted([*STORAGE_SECRET_EXPORTS.values(), *STORAGE_STRUCTURAL_EXPORTS.keys()]),
)
def test_r2_auth_probe_sets_canonical_storage_env(project_root: Path, key: str) -> None:
    """The direct auth probe maps repo secrets into canonical storage env.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :param key: canonical ``SYNTH_SETTER_STORAGE_*`` env key expected by
        ``ensure_r2_env_loaded``.
    """
    workflow = cast(dict[str, object], load_workflow(project_root, AUTH_PROBE_WORKFLOW))
    jobs = cast(dict[str, object], workflow["jobs"])
    job = cast(dict[str, object], jobs["r2-auth-probe"])
    env = cast(dict[str, object], job["env"])
    assert key in env


def _docker_run_blocks(path: Path) -> list[tuple[int, str]]:
    """Return ``docker run`` command blocks with their 1-based start lines.

    :param path: workflow file to scan.
    :returns: ``(line_number, block_text)`` tuples for each Docker command.
    """
    lines = path.read_text().splitlines()
    blocks: list[tuple[int, str]] = []
    index = 0
    while index < len(lines):
        if "docker run" not in lines[index]:
            index += 1
            continue
        start = index
        block: list[str] = []
        while index < len(lines):
            block.append(lines[index])
            stripped = lines[index].rstrip()
            index += 1
            if len(block) > 1 and not stripped.endswith("\\"):
                break
        blocks.append((start + 1, "\n".join(block)))
    return blocks


@pytest.mark.infra
@pytest.mark.parametrize("key", sorted(STORAGE_SECRET_EXPORTS.values()))
def test_workflow_docker_rclone_blocks_forward_canonical_storage_env(
    project_root: Path, key: str
) -> None:
    """Docker Python paths that receive rclone env also receive storage env.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :param key: canonical secret env key expected inside the container.
    """
    workflows_dir = project_root / ".github" / "workflows"
    offenders: list[str] = []
    for workflow in [*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")]:
        for line_number, block in _docker_run_blocks(workflow):
            if "RCLONE_CONFIG_R2_" not in block:
                continue
            if any(command in block for command in PURE_RCLONE_DOCKER_COMMANDS):
                continue
            if key not in block:
                offenders.append(f"{workflow.name}:{line_number}")
    assert not offenders, (
        f"docker run blocks forward rclone env without canonical storage env {key}: "
        f"{sorted(offenders)}"
    )
