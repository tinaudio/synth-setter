"""Regression guards for the VS Code devcontainer tmux terminal profile (#1053).

The default terminal profile is `tmux`. Two invariants must hold so the profile
behaves correctly in every `.devcontainer/<flavor>/devcontainer.json`:

1. The tmux args must NOT attach every terminal to a shared session
   (no `-A` flag, no fixed session name like `main`) — otherwise opening a
   second VS Code terminal mirrors the first.
2. `post-create.sh` must install `.devcontainer/tmux.conf` to the current
   user's `$HOME/.tmux.conf` BEFORE the root→dev `exec`, so the
   root-user terminal variant (root_gpu) also picks up mouse mode and the
   other tmux defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _load_devcontainer(path: Path) -> dict:
    """Parse a devcontainer.json file into a plain dict.

    :param path: Filesystem path to a `<flavor>/devcontainer.json`.
    :returns: Parsed JSON as a dict.
    :rtype: dict
    """
    return json.loads(path.read_text())


@pytest.mark.infra
def test_every_devcontainer_tmux_profile_uses_independent_session(
    devcontainer_json_paths: list[Path],
) -> None:
    """Each devcontainer's tmux profile must NOT attach to a shared session.

    Regression guard for #1053: `new-session -A -s main` made every VS Code
    terminal attach to the same session named `main`, so opening a second
    terminal mirrored the first. Each terminal must launch an independent
    session — i.e. the args must contain neither `-A` (attach-if-exists)
    nor a fixed session name like `main`.

    :param devcontainer_json_paths: Paths to every `<flavor>/devcontainer.json`,
        provided by the `tests/infra/conftest.py` fixture.
    """
    for path in devcontainer_json_paths:
        config = _load_devcontainer(path)
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
def test_post_create_installs_tmux_conf_to_home(post_create_script: Path) -> None:
    """`post-create.sh` must install `tmux.conf` to `$HOME/.tmux.conf` early.

    Regression guard for #1053: tmux mouse mode (and the rest of the conf)
    is shipped by copying `.devcontainer/tmux.conf` into the current user's
    HOME from `post-create.sh`. The install must run BEFORE the root→dev
    `exec` so the root variant's `/root/.tmux.conf` is populated for
    terminals that open as root (otherwise root_gpu loses mouse mode).

    :param post_create_script: Path to `.devcontainer/post-create.sh`, provided
        by the `tests/infra/conftest.py` fixture.
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


# Substrings that would reintroduce tmux session persistence (the TPM plugin
# stack) if they reappeared in the devcontainer. `plugins/tpm` (not bare `tpm`)
# anchors on the plugin path so incidental three-letter tokens can't false-fire.
_PERSISTENCE_MARKERS = ("resurrect", "continuum", "tmux-plugins", "@plugin", "plugins/tpm")


@pytest.mark.infra
def test_no_devcontainer_mounts_tmux_resurrect_volume(
    devcontainer_json_paths: list[Path],
) -> None:
    """No `mounts` entry may reference a tmux-resurrect volume.

    Matches on `resurrect` so a renamed source still trips the guard as long as
    it targets the `~/.local/share/tmux/resurrect` state dir.

    :param devcontainer_json_paths: Paths to every `<flavor>/devcontainer.json`,
        provided by the `tests/infra/conftest.py` fixture.
    """
    for path in devcontainer_json_paths:
        config = _load_devcontainer(path)
        offenders = [m for m in config.get("mounts", []) if "resurrect" in m]
        assert not offenders, (
            f"{path}: tmux session persistence was removed — no mount may reference a "
            f"tmux-resurrect volume. Offending mounts: {offenders!r}"
        )


@pytest.mark.infra
def test_tmux_conf_declares_no_session_persistence_plugins(tmux_conf: Path) -> None:
    """`tmux.conf` must not declare TPM / resurrect / continuum plugins.

    :param tmux_conf: Path to `.devcontainer/tmux.conf`, provided by the `tests/infra/conftest.py`
        fixture.
    """
    text = tmux_conf.read_text().lower()
    offenders = [marker for marker in _PERSISTENCE_MARKERS if marker in text]
    assert not offenders, (
        f"{tmux_conf}: tmux session persistence was removed — these plugin markers "
        f"must not appear: {offenders!r}"
    )


@pytest.mark.infra
def test_post_create_does_not_bootstrap_tmux_plugins(post_create_script: Path) -> None:
    """`post-create.sh` must not bootstrap TPM or run `install_plugins`.

    Installing `tmux.conf` itself is still required — see
    `test_post_create_installs_tmux_conf_to_home`.

    :param post_create_script: Path to `.devcontainer/post-create.sh`, provided
        by the `tests/infra/conftest.py` fixture.
    """
    text = post_create_script.read_text().lower()
    offenders = [marker for marker in (*_PERSISTENCE_MARKERS, "install_plugins") if marker in text]
    assert not offenders, (
        f"{post_create_script}: tmux session persistence was removed — these markers "
        f"must not appear: {offenders!r}"
    )
