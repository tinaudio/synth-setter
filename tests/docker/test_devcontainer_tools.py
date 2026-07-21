"""Smoke tests for the devcontainer-tools and devcontainer-tools-dev-user Docker images."""

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
def test_image_default_user_matches_expected() -> None:
    """Validate the image's default user against the SkyPilot/RunPod contract.

    ``devcontainer-tools`` must default to root so SkyPilot's RunPod backend can
    install sshd; ``devcontainer-tools-dev-user`` must default to non-root
    ``dev`` for local VS Code devcontainers. The runner declares which image it
    built via ``SYNTH_SETTER_DEVCONTAINER_EXPECT_USER`` (defaults to ``root``).
    """
    expected_user = os.environ.get("SYNTH_SETTER_DEVCONTAINER_EXPECT_USER", "root")
    assert _run_text("whoami") == expected_user


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


@pytest.mark.docker_smoke
@pytest.mark.skipif(
    not _RUN_DEVCONTAINER_SMOKE,
    reason="set SYNTH_SETTER_RUN_DEVCONTAINER_SMOKE=1 inside the built devcontainer image",
)
def test_doom_emacs_available() -> None:
    """Validate Emacs and the dev user's initialized Doom installation."""
    assert _run_text("emacs", "--batch", "--eval", '(princ "emacs-ready")') == "emacs-ready"
    assert _run_text("doom", "version", "--short") == "2.2.0"
    assert Path("/home/dev/.config/doom/init.el").is_file()


@pytest.mark.docker_smoke
@pytest.mark.skipif(
    not _RUN_DEVCONTAINER_SMOKE,
    reason="set SYNTH_SETTER_RUN_DEVCONTAINER_SMOKE=1 inside the built devcontainer image",
)
def test_infisical_cli_installed_with_pinned_version() -> None:
    """Validate the image runs the pinned Infisical CLI version."""
    assert _run_text("infisical", "--version") == "infisical version 0.38.0"
