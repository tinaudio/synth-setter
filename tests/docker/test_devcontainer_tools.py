"""Smoke tests for the built devcontainer-tools Docker image."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_RUN_DEVCONTAINER_SMOKE = os.environ.get("SYNTH_SETTER_RUN_DEVCONTAINER_SMOKE") == "1"


def _run_text(*args: str) -> str:
    """Run command argv and return stdout without surrounding whitespace.

    :param *args: Command executed without a shell unless ``bash`` is passed explicitly.
    :returns: Normalized stdout value used for image-contract assertions.
    """
    result = subprocess.run(  # noqa: S603 - fixed argv from test call sites
        args,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.mark.docker_smoke
@pytest.mark.skipif(
    not _RUN_DEVCONTAINER_SMOKE,
    reason="set SYNTH_SETTER_RUN_DEVCONTAINER_SMOKE=1 inside the built devcontainer image",
)
def test_codex_sandbox_prerequisites_available() -> None:
    """Validate the Codex CLI and sandbox prerequisite in the built image."""
    system_codex = Path("/usr/local/bin/codex")
    user_codex = Path("/home/dev/.npm-global/bin/codex")

    assert _run_text("bash", "-lc", "command -v bwrap") == "/usr/bin/bwrap"
    assert _run_text("bwrap", "--version").startswith("bubblewrap ")
    assert system_codex.exists()
    assert user_codex.resolve() == system_codex.resolve()
    assert _run_text(str(system_codex), "--version") == _run_text(
        str(user_codex),
        "--version",
    )
