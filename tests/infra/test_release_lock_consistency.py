"""Semantic-release keeps ``uv.lock`` in sync with the version bump (#1913).

A release commit rewrites ``pyproject.toml`` (via ``version_toml``) and
``CHANGELOG.md`` but previously left ``uv.lock`` stale: ``build_command`` was
empty and ``uv.lock`` was not a release asset, so the
``synth-setter`` version recorded in the lock lagged every release. These
tests pin the release-process contract that closes that drift.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml


def _semantic_release_config(project_root: Path) -> dict:
    """Load the ``[tool.semantic_release]`` table from ``pyproject.toml``.

    :param project_root: Repository root fixture.
    :returns: The parsed ``semantic_release`` config table.
    """
    with (project_root / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)
    return pyproject["tool"]["semantic_release"]


def test_semantic_release_commits_uv_lock_as_release_asset(project_root: Path) -> None:
    """``uv.lock`` is listed as a release asset so PSR commits it with the bump.

    :param project_root: Repository root fixture.
    """
    config = _semantic_release_config(project_root)

    assert "uv.lock" in config.get("assets", []), (
        "uv.lock must be a semantic_release asset so the release commit includes "
        "the refreshed lock alongside the pyproject.toml version bump (#1913)."
    )


def test_semantic_release_build_command_refreshes_uv_lock(project_root: Path) -> None:
    """``build_command`` re-resolves ``uv.lock`` so the lock tracks the new version.

    :param project_root: Repository root fixture.
    """
    config = _semantic_release_config(project_root)

    build_command = config.get("build_command", "")
    assert build_command, (
        "build_command must be set; an empty command skips the lock refresh that "
        "keeps uv.lock's synth-setter version in sync with pyproject.toml (#1913)."
    )
    assert "uv lock" in build_command, (
        f"build_command {build_command!r} must re-resolve uv.lock (#1913)."
    )


def test_release_workflow_provisions_uv_for_build_command(project_root: Path) -> None:
    """The release job installs uv so ``build_command = "uv lock"`` can run.

    python-semantic-release executes ``build_command`` in the OS shell with a
    subset of env vars; uv is not provisioned by the PSR action itself, so the
    release workflow must set it up before invoking PSR.

    :param project_root: Repository root fixture.
    """
    release_workflow = yaml.safe_load((project_root / ".github/workflows/release.yml").read_text())

    steps = release_workflow["jobs"]["release"]["steps"]
    step_uses = [step.get("uses", "") for step in steps if "uses" in step]
    assert any("setup-uv" in use for use in step_uses), (
        "release.yml must set up uv (e.g. astral-sh/setup-uv) before the "
        "python-semantic-release action so build_command's `uv lock` is on PATH (#1913)."
    )


def test_uv_lock_check_runs_on_every_main_push(project_root: Path) -> None:
    """The drift detector has no ``paths`` filter on push so [skip ci] drift is caught.

    Release commits carry ``[skip ci]`` and touch ``pyproject.toml``; a
    ``paths:`` filter scoped to specific files would skip when the drift is
    introduced elsewhere, so the push trigger must run unconditionally on main.
    Asserted on raw text because PyYAML parses the bare ``on:`` key as ``True``.

    :param project_root: Repository root fixture.
    """
    text = (project_root / ".github/workflows/uv-lock-check.yml").read_text()

    push_block = re.search(r"^  push:\s*\n(?:    .*\n)+", text, re.MULTILINE)
    assert push_block, "uv-lock-check.yml must define a `push:` trigger (#1913)."
    assert "paths:" not in push_block.group(0), (
        "uv-lock-check.yml push trigger must not carry a `paths` filter, otherwise "
        "release-commit drift on [skip ci] commits evades detection (#1913)."
    )
