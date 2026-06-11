"""Invariant: post-create.sh makes the bundled coding agents non-interactive.

The devcontainer is itself the sandbox, so Codex and Antigravity (`agy`) should
run full-auto — no per-command approval prompts. post-create.sh's
`configure_agent_autonomy` seeds a Codex `config.toml` and an `agy` wrapper to
enforce that on every fresh container. The static tests pin the contract tokens;
the behavioral test runs the function under an isolated `$HOME` (safe — it only
writes there) to prove the files materialize, the seed-only-when-absent guard
holds, and the wrapper append is idempotent.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

_FUNC_NAME = "configure_agent_autonomy"


def _extract_function(script_text: str) -> str:
    """Return the `configure_agent_autonomy` definition (def-line through its column-0 `}`).

    Sliced in Python — not `sed` — so the bounds match identically on GNU and
    BSD userlands. The terminator is the bare top-level call line (`{_FUNC_NAME}`
    alone), which no heredoc-body line matches, so the whole function is captured.

    :param script_text: Full source of `.devcontainer/post-create.sh`.
    :returns: The function definition, ready to source.
    """
    lines = script_text.splitlines()
    start = lines.index(f"{_FUNC_NAME}() {{")
    call = lines.index(_FUNC_NAME, start + 1)
    return "\n".join(lines[start:call])


def _run_configure_agent_autonomy(script: Path, home: Path, *, times: int) -> None:
    """Source `configure_agent_autonomy` from `script` and run it `times` times under HOME=`home`.

    Only the function definition is sourced — not the whole script, which execs
    and mutates the workspace — so just its `$HOME`-scoped writes happen. Calling
    it `times` times exercises idempotency.

    :param script: Path to `.devcontainer/post-create.sh`.
    :param home: Directory to use as `$HOME` for the run.
    :param times: Number of `configure_agent_autonomy` invocations.
    """
    definition = _extract_function(script.read_text())
    calls = "\n".join([_FUNC_NAME] * times)
    subprocess.run(  # noqa: S603 — fixed argv, no shell injection (paths are test-controlled)
        ["bash", "-c", f"{definition}\n{calls}"],  # noqa: S607 — bash on PATH
        env={"HOME": str(home), "PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.mark.infra
def test_post_create_seeds_codex_full_auto_config(post_create_script: Path) -> None:
    """The script seeds a Codex config with `never` approval and full access.

    :param post_create_script: Path to `.devcontainer/post-create.sh`.
    """
    text = post_create_script.read_text()
    assert ".codex/config.toml" in text
    assert re.search(r'approval_policy\s*=\s*"never"', text)
    assert re.search(r'sandbox_mode\s*=\s*"danger-full-access"', text)


@pytest.mark.infra
def test_post_create_wraps_agy_with_skip_permissions(post_create_script: Path) -> None:
    """The script wraps `agy` so it auto-approves, via `command agy` for opt-out.

    :param post_create_script: Path to `.devcontainer/post-create.sh`.
    """
    text = post_create_script.read_text()
    assert re.search(r"command\s+agy\s+--dangerously-skip-permissions", text)


@pytest.mark.infra
def test_configure_agent_autonomy_materializes_config_idempotently(
    post_create_script: Path, tmp_path: Path
) -> None:
    """Running the function seeds both files, preserves a pre-existing Codex config, and is idempotent.

    :param post_create_script: Path to `.devcontainer/post-create.sh`.
    :param tmp_path: Isolated `$HOME` so the run touches no real config.
    """
    sentinel = "# pre-existing — must not be clobbered\n"
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex/config.toml").write_text(sentinel)

    _run_configure_agent_autonomy(post_create_script, tmp_path, times=2)

    assert (tmp_path / ".codex/config.toml").read_text() == sentinel, (
        "an existing Codex config must win over the seeded default"
    )
    bashrc = (tmp_path / ".bashrc").read_text()
    assert bashrc.count("agy()") == 1, "the agy wrapper must be appended exactly once"
    assert "--dangerously-skip-permissions" in bashrc


@pytest.mark.infra
def test_configure_agent_autonomy_seeds_codex_config_when_absent(
    post_create_script: Path, tmp_path: Path
) -> None:
    """From an empty `$HOME`, the function seeds the full-auto Codex defaults and a rerun leaves them intact.

    :param post_create_script: Path to `.devcontainer/post-create.sh`.
    :param tmp_path: Isolated `$HOME` so the run touches no real config.
    """
    _run_configure_agent_autonomy(post_create_script, tmp_path, times=2)

    config = (tmp_path / ".codex/config.toml").read_text()
    assert 'approval_policy = "never"' in config
    assert 'sandbox_mode = "danger-full-access"' in config
    assert (tmp_path / ".bashrc").read_text().count("agy()") == 1
