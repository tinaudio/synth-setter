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

from synth_setter.pipeline.r2_io import _R2_STRUCTURAL_DEFAULTS, _SECRET_R2_ENV_KEYS

# Union of the secret keys rclone needs to authenticate AND the structural
# defaults rclone needs to assemble the `r2:` remote. `is_r2_reachable()`
# does NOT apply the `setdefault` that `ensure_r2_env_loaded()` does, so
# both sets must be present in the runner env — see #1185.
REQUIRED_R2_ENV_KEYS: frozenset[str] = frozenset(_SECRET_R2_ENV_KEYS) | frozenset(
    _R2_STRUCTURAL_DEFAULTS
)

PYTEST_STEP_NAME = "Run slow (non-GPU, non-MPS, non-VST) tests"
RCLONE_INSTALL_STEP_NAME = "Install rclone"


def _load_workflow_steps(project_root: Path) -> list[dict[str, object]]:
    """Return the ordered ``steps`` list of the ``run_slow_tests`` job.

    :param project_root: repo root; the workflow is read from
        ``<project_root>/.github/workflows/cpu-slow.yml``.
    :returns: the job's ``steps`` list.
    """
    workflow_path = project_root / ".github" / "workflows" / "cpu-slow.yml"
    workflow = yaml.safe_load(workflow_path.read_text())
    return cast(list[dict[str, object]], workflow["jobs"]["run_slow_tests"]["steps"])


def _load_pytest_step_env(project_root: Path) -> dict[str, str]:
    """Return the ``env`` mapping of the pytest step in ``cpu-slow.yml``.

    :param project_root: repo root; the workflow is read from
        ``<project_root>/.github/workflows/cpu-slow.yml``.
    :returns: the step's ``env`` block, or an empty dict if unset.
    """
    workflow_path = project_root / ".github" / "workflows" / "cpu-slow.yml"
    for step in _load_workflow_steps(project_root):
        if step.get("name") == PYTEST_STEP_NAME:
            return cast(dict[str, str], step.get("env") or {})
    pytest.fail(f"Could not find step {PYTEST_STEP_NAME!r} in {workflow_path}")


@pytest.mark.infra
def test_cpu_slow_pytest_step_injects_r2_creds(project_root: Path) -> None:
    """The pytest step exposes every ``RCLONE_CONFIG_R2_*`` key ``is_r2_reachable()`` reads.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    """
    env = _load_pytest_step_env(project_root)
    missing = REQUIRED_R2_ENV_KEYS - env.keys()
    assert not missing, (
        f"cpu-slow.yml pytest step missing R2 env keys: {sorted(missing)} — "
        f"`integration_r2` tests will skip via r2_io.is_r2_reachable() without them"
    )


@pytest.mark.infra
@pytest.mark.parametrize("key", sorted(_SECRET_R2_ENV_KEYS))
def test_cpu_slow_r2_secret_env_values_reference_matching_secrets(
    project_root: Path, key: str
) -> None:
    """Each secret R2 env key sources from the same-named ``secrets.RCLONE_CONFIG_R2_*``.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :param key: one of ``_SECRET_R2_ENV_KEYS``; parametrized so a missing
        key fails this case independently with the key name in the output
        rather than collapsing into a ``KeyError`` from the others.
    """
    env = _load_pytest_step_env(project_root)
    actual = env.get(key)
    expected = f"${{{{ secrets.{key} }}}}"
    assert actual == expected, (
        f"cpu-slow.yml env[{key!r}] = {actual!r}; expected {expected!r} to match the "
        f"secret-name convention used by generate-dataset-shards.yaml + "
        f"test-local-launcher-roundtrip.yml"
    )


@pytest.mark.infra
@pytest.mark.parametrize(("key", "expected"), sorted(_R2_STRUCTURAL_DEFAULTS.items()))
def test_cpu_slow_r2_structural_env_values_use_canonical_literals(
    project_root: Path, key: str, expected: str
) -> None:
    """Each structural R2 env key uses the canonical literal from ``r2_io``.

    ``is_r2_reachable()`` does not apply ``_R2_STRUCTURAL_DEFAULTS`` via
    ``setdefault`` the way ``ensure_r2_env_loaded()`` does, so the workflow
    must export these literals itself or ``rclone lsd r2:`` will fail with
    ``didn't find section in config file`` and ``integration_r2`` tests
    skip silently — see #1185.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :param key: one of ``_R2_STRUCTURAL_DEFAULTS``.
    :param expected: the canonical literal value rclone needs (``s3`` /
        ``Cloudflare``); drift from ``r2_io._R2_STRUCTURAL_DEFAULTS`` fails
        this case loudly instead of letting the workflow silently regress.
    """
    env = _load_pytest_step_env(project_root)
    actual = env.get(key)
    assert actual == expected, (
        f"cpu-slow.yml env[{key!r}] = {actual!r}; expected literal {expected!r} "
        f"(canonical value from r2_io._R2_STRUCTURAL_DEFAULTS)"
    )


@pytest.mark.infra
def test_cpu_slow_installs_rclone_before_pytest(project_root: Path) -> None:
    """``Install rclone`` runs before the pytest step so ``is_r2_reachable()`` passes.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    """
    steps = _load_workflow_steps(project_root)
    names = [step.get("name") for step in steps]
    assert RCLONE_INSTALL_STEP_NAME in names, (
        f"cpu-slow.yml missing step {RCLONE_INSTALL_STEP_NAME!r} — "
        f"`integration_r2` tests skip via r2_io.is_r2_reachable() without the binary"
    )
    assert names.index(RCLONE_INSTALL_STEP_NAME) < names.index(PYTEST_STEP_NAME), (
        f"{RCLONE_INSTALL_STEP_NAME!r} must precede {PYTEST_STEP_NAME!r} in cpu-slow.yml"
    )
