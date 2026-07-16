"""Nightly CI registers arm64 emulation before running the full suite."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from workflow_fixtures import load_workflow

QEMU_ACTION = "docker/setup-qemu-action@v3"
FULL_SUITE_STEP_NAME = "Run full test suite"


@pytest.mark.infra
def test_nightly_sets_up_arm64_qemu_before_full_test_suite(project_root: Path) -> None:
    """The arm64 buildx test runs only after its binfmt handler is registered.

    :param project_root: Session fixture from ``tests/infra/conftest.py``.
    """
    workflow = load_workflow(project_root, "nightly.yml")
    jobs = cast(dict[str, object], workflow["jobs"])
    job = cast(dict[str, object], jobs["nightly-full-suite"])
    steps = cast(list[dict[str, object]], job["steps"])

    qemu_indices = [index for index, step in enumerate(steps) if step.get("uses") == QEMU_ACTION]
    assert qemu_indices, f"nightly.yml missing a step that uses {QEMU_ACTION}"

    qemu_step = steps[qemu_indices[0]]
    qemu_inputs = cast(dict[str, object], qemu_step.get("with") or {})
    assert qemu_inputs.get("platforms") == "arm64"

    step_names = [step.get("name") for step in steps]
    assert qemu_indices[0] < step_names.index(FULL_SUITE_STEP_NAME)
