"""Pin the Node heap ceiling baked into the ``pyright`` pre-commit hook.

pyright runs under Node, whose default old-space ceiling is derived from the
machine's *total* RAM, not its *free* RAM. On a loaded box the analysis exhausts
that ceiling and V8 aborts, surfacing as ``exit code 241`` with no diagnostics —
indistinguishable from a real type error (#2200). Baking ``NODE_OPTIONS`` into
the hook entry makes every invocation path (``pre-commit run``, ``make format``,
the git hooks, CI) deterministic, so this test guards that entry from drifting
back to a bare ``uv run pyright``.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest
import yaml

MIN_HEAP_MB = 6144


@pytest.fixture(scope="session")
def pyright_hook_entry(project_root: Path) -> str:
    """Read the ``entry`` command string of the local ``pyright`` pre-commit hook.

    :param project_root: Repo root holding ``.pre-commit-config.yaml`` (from conftest).
    :returns: The hook's ``entry`` value.
    """
    config = yaml.safe_load((project_root / ".pre-commit-config.yaml").read_text())
    entries = [
        hook["entry"]
        for repo in config["repos"]
        for hook in repo["hooks"]
        if hook["id"] == "pyright"
    ]
    assert len(entries) == 1, f"expected exactly one `pyright` hook, found {len(entries)}"
    return entries[0]


def test_pyright_hook_entry_caps_node_old_space_above_default(pyright_hook_entry: str) -> None:
    """The hook entry pins ``--max-old-space-size`` to at least ``MIN_HEAP_MB``.

    :param pyright_hook_entry: The configured ``entry`` command string.
    """
    assignments = dict(
        token.split("=", 1) for token in shlex.split(pyright_hook_entry) if "=" in token
    )
    node_options = assignments.get("NODE_OPTIONS", "")
    match = re.search(r"--max-old-space-size=(\d+)", node_options)

    assert match is not None, (
        "the `pyright` hook entry must set NODE_OPTIONS=--max-old-space-size=<MB>; without it "
        f"V8 aborts with a bare `exit code 241` under memory pressure (#2200). Entry: "
        f"{pyright_hook_entry!r}"
    )
    assert int(match.group(1)) >= MIN_HEAP_MB, (
        f"the pinned Node heap ({match.group(1)} MB) is below the {MIN_HEAP_MB} MB verified to "
        "carry a full-repo pyright pass (#2200)."
    )


def test_pyright_hook_entry_still_runs_pyright(pyright_hook_entry: str) -> None:
    """The env prefix wraps the real pyright invocation rather than replacing it.

    :param pyright_hook_entry: The configured ``entry`` command string.
    """
    tokens = shlex.split(pyright_hook_entry)
    assert tokens[-1] == "pyright", f"entry must still invoke pyright, got {pyright_hook_entry!r}"
