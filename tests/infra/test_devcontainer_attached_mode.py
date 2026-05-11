"""Invariant 1: devcontainer starts in attached mode (PID 1 doesn't exit).

A developer must be able to `Attach to Container` and have a shell. That
requires the container's MODE env var to put `docker_entrypoint.py` into the
`idle` subcommand (which execs `sleep infinity`), plus the standard
devcontainer fields (`postCreateCommand`, `remoteUser`, `workspaceFolder`).
The `--env-file` runArg lets credentials reach the container.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


def _load(path: Path) -> dict:
    """Parse a devcontainer.json file into a plain dict."""
    return json.loads(path.read_text())


@pytest.mark.infra
def test_every_devcontainer_sets_mode_idle_for_attached_pid1(
    devcontainer_json_paths: list[Path],
) -> None:
    """containerEnv.MODE == 'idle' so docker_entrypoint.py execs sleep infinity."""
    for path in devcontainer_json_paths:
        config = _load(path)
        container_env = config.get("containerEnv", {})
        assert container_env.get("MODE") == "idle", (
            f"{path}: expected containerEnv.MODE == 'idle' to keep PID 1 alive "
            f"for attached-mode, got {container_env.get('MODE')!r}"
        )


@pytest.mark.infra
def test_every_devcontainer_has_post_create_command_attached_mode(
    devcontainer_json_paths: list[Path],
) -> None:
    """Each devcontainer.json has postCreateCommand set."""
    for path in devcontainer_json_paths:
        config = _load(path)
        assert config.get("postCreateCommand"), f"{path}: postCreateCommand is required"


@pytest.mark.infra
def test_every_devcontainer_has_remote_user_attached_mode(
    devcontainer_json_paths: list[Path],
) -> None:
    """RemoteUser must be set so VSCode attaches as a real user."""
    for path in devcontainer_json_paths:
        config = _load(path)
        assert config.get("remoteUser"), f"{path}: remoteUser is required"


@pytest.mark.infra
def test_every_devcontainer_has_workspace_folder_attached_mode(
    devcontainer_json_paths: list[Path],
) -> None:
    """WorkspaceFolder must be set so the attached shell starts in the project."""
    for path in devcontainer_json_paths:
        config = _load(path)
        assert config.get("workspaceFolder"), f"{path}: workspaceFolder is required"


@pytest.mark.infra
def test_every_devcontainer_run_args_includes_env_file_attached_mode(
    devcontainer_json_paths: list[Path],
) -> None:
    """RunArgs must include `--env-file .env` so credentials reach the container."""
    for path in devcontainer_json_paths:
        config = _load(path)
        run_args = config.get("runArgs", [])
        assert "--env-file" in run_args, (
            f"{path}: runArgs must include '--env-file' so secrets reach the container; "
            f"got {run_args!r}"
        )
        env_file_index = run_args.index("--env-file")
        assert env_file_index + 1 < len(run_args), (
            f"{path}: '--env-file' must be followed by a path"
        )


@pytest.mark.infra
def test_docker_entrypoint_idle_mode_blocks_attached_mode(project_root: Path) -> None:
    """docker_entrypoint.py's idle command must exec `sleep infinity` (not exit).

    Narrowed to the idle function's AST body — substring matches on the whole
    file would pass even if `sleep`/`infinity` only appeared in a docstring.
    """
    entrypoint = project_root / "scripts" / "docker_entrypoint.py"
    tree = ast.parse(entrypoint.read_text())
    idle_fn: ast.FunctionDef | None = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "idle"),
        None,
    )
    assert idle_fn is not None, "scripts/docker_entrypoint.py: missing `idle` function"
    body_strings = [node.value for node in ast.walk(idle_fn) if isinstance(node, ast.Constant)]
    assert "sleep" in body_strings and "infinity" in body_strings, (
        f"scripts/docker_entrypoint.py: idle() must exec `sleep infinity` so PID 1 stays alive "
        f"for attached-mode; constants found in idle() body: {body_strings!r}"
    )
