"""Regression guards for the VS Code devcontainer zellij terminal config.

The default terminal profile is `zellij`. Two behaviors must hold across the
devcontainer setup:

1. `.devcontainer/zellij.kdl` silences the first-run startup popups
   (`show_startup_tips false`, `show_release_notes false`) so a fresh container
   doesn't open onto a tips/release-notes overlay.
2. The same config attaches every terminal to one shared session
   (`session_name` + `attach_to_session true`) so opening a second VS Code
   terminal lands in the same session rather than spawning a fresh one. This is
   the deliberate inverse of the tmux profile, which uses independent sessions
   (`test_devcontainer_tmux_session.py`).

`post-create.sh` must install the config to `~/.config/zellij/config.kdl` BEFORE
the root->dev `exec`, so the root-user terminal variant (root_gpu) is quieted
too — mirroring the tmux.conf install.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


@pytest.mark.infra
def test_zellij_config_disables_startup_popups(zellij_config: Path) -> None:
    """`zellij.kdl` must turn off the startup-tips and release-notes popups.

    :param zellij_config: Path to `.devcontainer/zellij.kdl`, provided by the
        `tests/infra/conftest.py` fixture.
    """
    text = zellij_config.read_text()
    for option in ("show_startup_tips false", "show_release_notes false"):
        assert option in text, (
            f"{zellij_config}: must set `{option}` to suppress the zellij startup popup."
        )


@pytest.mark.infra
def test_zellij_config_attaches_to_shared_session(zellij_config: Path) -> None:
    """`zellij.kdl` must reattach new terminals to one named session.

    `attach_to_session true` reattaches to the session named by `session_name`
    if it exists, so a second VS Code terminal mirrors the first instead of
    opening a fresh session.

    :param zellij_config: Path to `.devcontainer/zellij.kdl`, provided by the
        `tests/infra/conftest.py` fixture.
    """
    text = zellij_config.read_text()
    assert "attach_to_session true" in text, (
        f"{zellij_config}: must set `attach_to_session true` so new terminals "
        f"reattach to the shared session."
    )
    assert re.search(r'session_name\s+"[^"]+"', text), (
        f"{zellij_config}: must set `session_name` to a non-empty quoted name — "
        f"`attach_to_session` only reattaches to a session of that name."
    )


@pytest.mark.infra
def test_post_create_installs_zellij_config_to_home(post_create_script: Path) -> None:
    """`post-create.sh` must install `zellij.kdl` to `~/.config/zellij/config.kdl` early.

    The install must run BEFORE the root->dev `exec` so the root variant's
    `/root/.config/zellij/config.kdl` is populated for terminals that open as
    root (otherwise root_gpu keeps the popups and per-terminal sessions).

    :param post_create_script: Path to `.devcontainer/post-create.sh`, provided
        by the `tests/infra/conftest.py` fixture.
    """
    lines = post_create_script.read_text().splitlines()
    install_line = next(
        (
            i
            for i, line in enumerate(lines, start=1)
            if "zellij.kdl" in line and "config.kdl" in line
        ),
        None,
    )
    assert install_line is not None, (
        f"{post_create_script}: must contain a line installing zellij.kdl to "
        f"`~/.config/zellij/config.kdl`."
    )
    root_drop_line = next(
        (
            i
            for i, line in enumerate(lines, start=1)
            if line.strip().startswith('if [ "$(id -u)" -eq 0 ]')
        ),
        None,
    )
    assert root_drop_line is not None, (
        f"{post_create_script}: expected a root->dev privilege-drop guard "
        f'(`if [ "$(id -u)" -eq 0 ]; then ...`).'
    )
    assert install_line < root_drop_line, (
        f"{post_create_script}: zellij config install (line {install_line}) must "
        f"run BEFORE the root->dev exec (line {root_drop_line}) so "
        f"/root/.config/zellij/config.kdl is populated for the root variant."
    )
