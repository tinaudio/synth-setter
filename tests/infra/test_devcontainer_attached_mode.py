"""Invariant 1: devcontainer starts in attached mode (PID 1 doesn't exit).

A developer must be able to `Attach to Container` and have a shell. That
requires the standard devcontainer fields (`postCreateCommand`, `remoteUser`,
`workspaceFolder`). The `--env-file` runArg lets credentials reach the
container.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


def _load(path: Path) -> dict:
    """Parse a devcontainer.json file into a plain dict."""
    return json.loads(path.read_text())


@pytest.mark.infra
def test_every_devcontainer_sets_mode_idle_for_attached_pid1(
    devcontainer_json_paths: list[Path],
) -> None:
    """containerEnv.MODE == 'idle' (legacy invariant from the click-CLI entrypoint era)."""
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
    """Each devcontainer.json runs the shared post-create script.

    A truthiness check passes for a placeholder; the script path is the contract.
    """
    for path in devcontainer_json_paths:
        command = _load(path).get("postCreateCommand")
        rendered = " ".join(command) if isinstance(command, list) else str(command or "")
        assert "post-create.sh" in rendered, (
            f"{path}: postCreateCommand must invoke post-create.sh, got {command!r}"
        )


@pytest.mark.infra
def test_every_devcontainer_has_remote_user_attached_mode(
    devcontainer_json_paths: list[Path],
) -> None:
    """RemoteUser must stay overridable via ``DEVCONTAINER_USER`` with a baked default.

    A hardcoded user would satisfy truthiness but break per-developer attach.
    """
    for path in devcontainer_json_paths:
        remote_user = _load(path).get("remoteUser")
        assert re.fullmatch(r"\$\{localEnv:DEVCONTAINER_USER:\w+\}", str(remote_user)), (
            f"{path}: remoteUser must be ${{localEnv:DEVCONTAINER_USER:<default>}}, "
            f"got {remote_user!r}"
        )


@pytest.mark.infra
def test_every_devcontainer_has_workspace_folder_attached_mode(
    devcontainer_json_paths: list[Path],
) -> None:
    """WorkspaceFolder must be the in-image checkout path, identically across flavors.

    A truthiness check would accept a stale path and land the attached shell outside the project.
    """
    for path in devcontainer_json_paths:
        workspace_folder = _load(path).get("workspaceFolder")
        assert workspace_folder == "/home/build/synth-setter", (
            f"{path}: unexpected workspaceFolder {workspace_folder!r}"
        )


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
