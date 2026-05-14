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
def test_every_devcontainer_tmux_profile_uses_independent_session(
    devcontainer_json_paths: list[Path],
) -> None:
    """The VS Code tmux terminal profile must NOT attach to a shared session.

    Regression guard for #1053: `new-session -A -s main` made every terminal
    attach to the same session named `main`, so opening a second terminal
    mirrored the first. Each terminal must launch an independent session.
    """
    for path in devcontainer_json_paths:
        config = _load(path)
        profiles = (
            config.get("customizations", {})
            .get("vscode", {})
            .get("settings", {})
            .get("terminal.integrated.profiles.linux", {})
        )
        tmux_args = profiles.get("tmux", {}).get("args", [])
        assert "-A" not in tmux_args, (
            f"{path}: tmux profile must not pass `-A` (attach-if-exists) — "
            f"causes every VS Code terminal to share one session. Got args: {tmux_args!r}"
        )
        assert "main" not in tmux_args, (
            f"{path}: tmux profile must not target a fixed session name like `main` — "
            f"causes every VS Code terminal to share one session. Got args: {tmux_args!r}"
        )


@pytest.mark.infra
def test_post_create_installs_tmux_conf_to_home(
    post_create_script: Path,
) -> None:
    """post-create.sh must install .devcontainer/tmux.conf to ~/.tmux.conf.

    Regression guard for #1053: tmux mouse mode (and other settings) is shipped by installing the
    conf into the current user's HOME from post-create.sh. The install must run BEFORE the root→dev
    exec so the root variant's /root/.tmux.conf is populated for terminals that open as root.
    """
    script = post_create_script.read_text()
    install_line = next(
        (
            i
            for i, line in enumerate(script.splitlines(), start=1)
            if line.startswith("install ") and "tmux.conf" in line and '"$HOME/.tmux.conf"' in line
        ),
        None,
    )
    assert install_line is not None, (
        f"{post_create_script}: must contain a line installing tmux.conf to "
        f'"$HOME/.tmux.conf" (e.g., `install -m 0644 "$_devc_dir/tmux.conf" '
        f'"$HOME/.tmux.conf"`).'
    )
    root_drop_line = next(
        (
            i
            for i, line in enumerate(script.splitlines(), start=1)
            if line.strip().startswith('if [ "$(id -u)" -eq 0 ]')
        ),
        None,
    )
    assert root_drop_line is not None, (
        f"{post_create_script}: expected a root→dev privilege-drop guard "
        f'(`if [ "$(id -u)" -eq 0 ]; then ...`).'
    )
    assert install_line < root_drop_line, (
        f"{post_create_script}: tmux.conf install (line {install_line}) must run "
        f"BEFORE the root→dev exec (line {root_drop_line}) so /root/.tmux.conf "
        f"is populated for the root variant."
    )


@pytest.mark.infra
def test_docker_entrypoint_idle_mode_blocks_attached_mode(project_root: Path) -> None:
    """docker_entrypoint.py's idle command must exec `sleep infinity` (not exit).

    Narrowed to the idle function's AST body — substring matches on the whole
    file would pass even if `sleep`/`infinity` only appeared in a docstring.
    """
    entrypoint = project_root / "src" / "synth_setter" / "tools" / "docker_entrypoint.py"
    tree = ast.parse(entrypoint.read_text())
    idle_fn: ast.FunctionDef | None = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "idle"),
        None,
    )
    assert idle_fn is not None, (
        "src/synth_setter/tools/docker_entrypoint.py: missing `idle` function"
    )
    body_strings = [node.value for node in ast.walk(idle_fn) if isinstance(node, ast.Constant)]
    assert "sleep" in body_strings and "infinity" in body_strings, (
        f"src/synth_setter/tools/docker_entrypoint.py: idle() must exec `sleep infinity` so PID 1 stays alive "
        f"for attached-mode; constants found in idle() body: {body_strings!r}"
    )
