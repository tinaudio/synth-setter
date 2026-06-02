"""`setup-r2` composite exports every `RCLONE_CONFIG_R2_*` key rclone needs.

The action is the single source of truth for the R2 environment that the
data-pipeline workflows (`cpu-slow.yml`, `finalize-dataset.yaml`, …) previously
spelled inline — see #1353. `is_r2_reachable()` does not apply the structural
`setdefault` that `ensure_r2_env_loaded()` does, so the action must export both
the structural literals AND the secret keys or `rclone lsd r2:` fails and
`integration_r2` tests skip silently (see #1185). These tests pin that contract
against `r2_io` so the action can't drift from the values rclone expects.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from workflow_fixtures import load_composite_action

from synth_setter.pipeline.r2_io import _R2_STRUCTURAL_DEFAULTS, _SECRET_R2_ENV_KEYS

ACTION_NAME = "setup-r2"
EXPORT_STEP_NAME = "Export R2 credentials"
INSTALL_STEP_NAME = "Install rclone"

# input name -> the `_SECRET_R2_ENV_KEYS` member it supplies.
SECRET_INPUTS: dict[str, str] = {
    "access-key-id": "RCLONE_CONFIG_R2_ACCESS_KEY_ID",
    "secret-access-key": "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY",
    "endpoint": "RCLONE_CONFIG_R2_ENDPOINT",
}


def _load_action(project_root: Path) -> dict[str, object]:
    return cast(dict[str, object], load_composite_action(project_root, ACTION_NAME))


def _load_steps(project_root: Path) -> list[dict[str, object]]:
    runs = cast(dict[str, object], _load_action(project_root)["runs"])
    return cast(list[dict[str, object]], runs["steps"])


def _find_step(project_root: Path, name: str) -> dict[str, object]:
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
@pytest.mark.parametrize(("key", "expected"), sorted(_R2_STRUCTURAL_DEFAULTS.items()))
def test_setup_r2_exports_structural_literals(project_root: Path, key: str, expected: str) -> None:
    """The export step writes each structural key with its canonical literal.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :param key: one of ``_R2_STRUCTURAL_DEFAULTS`` (``RCLONE_CONFIG_R2_TYPE`` / ``_PROVIDER``).
    :param expected: the literal rclone needs (``s3`` / ``Cloudflare``); drift
        from ``r2_io._R2_STRUCTURAL_DEFAULTS`` fails here loudly.
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
    wired up and that the export key matches a ``_SECRET_R2_ENV_KEYS`` member.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :param input_name: composite input carrying the secret.
    :param env_key: the ``RCLONE_CONFIG_R2_*`` key it is exported as.
    """
    assert env_key in _SECRET_R2_ENV_KEYS
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
