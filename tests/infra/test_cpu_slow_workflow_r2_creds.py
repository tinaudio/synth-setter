"""`cpu-slow.yml` injects `RCLONE_CONFIG_R2_*` so `integration_r2` tests run.

`integration_r2`-marked tests also carry `slow`, so they collect into
`cpu-slow.yml`'s pytest invocation. Without the rclone-prefixed R2 env vars,
`r2_io.is_r2_reachable()` short-circuits and they skip silently — degrading
coverage with no signal. This test pins the env-injection so a future
restructure can't drop the keys unnoticed (see #1185).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import yaml

REQUIRED_R2_ENV_KEYS: frozenset[str] = frozenset(
    {
        "RCLONE_CONFIG_R2_ACCESS_KEY_ID",
        "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY",
        "RCLONE_CONFIG_R2_ENDPOINT",
    }
)

PYTEST_STEP_NAME = "Run slow (non-GPU, non-MPS, non-VST) tests"


def _load_pytest_step_env(project_root: Path) -> dict[str, str]:
    """Return the ``env`` mapping of the pytest step in ``cpu-slow.yml``.

    :param project_root: repo root; the workflow is read from
        ``<project_root>/.github/workflows/cpu-slow.yml``.
    :returns: the step's ``env`` block, or an empty dict if unset.
    """
    workflow_path = project_root / ".github" / "workflows" / "cpu-slow.yml"
    workflow = yaml.safe_load(workflow_path.read_text())
    for step in workflow["jobs"]["run_slow_tests"]["steps"]:
        if step.get("name") == PYTEST_STEP_NAME:
            return cast("dict[str, str]", step.get("env") or {})
    pytest.fail(f"Could not find step {PYTEST_STEP_NAME!r} in {workflow_path}")


@pytest.mark.infra
def test_cpu_slow_pytest_step_injects_r2_creds(project_root: Path) -> None:
    """The pytest step exposes all three ``RCLONE_CONFIG_R2_*`` keys via ``env:``.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    """
    env = _load_pytest_step_env(project_root)
    missing = REQUIRED_R2_ENV_KEYS - env.keys()
    assert not missing, (
        f"cpu-slow.yml pytest step missing R2 env keys: {sorted(missing)} — "
        f"`integration_r2` tests will skip via r2_io.is_r2_reachable() without them"
    )


@pytest.mark.infra
def test_cpu_slow_r2_env_values_reference_matching_secrets(project_root: Path) -> None:
    """Each R2 env key sources from the same-named ``secrets.RCLONE_CONFIG_R2_*``.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    """
    env = _load_pytest_step_env(project_root)
    for key in sorted(REQUIRED_R2_ENV_KEYS):
        expected = f"${{{{ secrets.{key} }}}}"
        assert env[key] == expected, (
            f"cpu-slow.yml env[{key!r}] = {env[key]!r}; expected {expected!r} to match the "
            f"secret-name convention used by generate-dataset-shards.yaml + "
            f"test-local-launcher-roundtrip.yml"
        )
