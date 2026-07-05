"""Shared fixtures for the hook tests under ``tests/claude_hooks/``."""

from __future__ import annotations

import pytest

# Gate-mode knobs agent/hooks/* read with block/warn defaults; agent-session
# overrides (e.g. REVIEW_COMMENT_GATE=warn) leak via os.environ and flip them.
GATE_MODE_ENV_VARS = (
    "REVIEW_COMMENT_GATE",
    "REVIEW_BLOCK_GATE",
    "PR_TITLE_GATE",
    "WORKTREE_GUARD_MODE",
    "PR_READINESS_GATE",
    "REVIEW_MAX_LAG",
)


def scrub_gate_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delete every gate-mode override from ``os.environ``.

    :param monkeypatch: Owns the deletions; restores them at its teardown.
    """
    for var in GATE_MODE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _scrub_gate_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run hooks with their documented default modes in every test here.

    Intentionally directory-wide: any test in this package may spawn a hook
    subprocess that inherits ``os.environ``. Tests exercising a non-default
    mode opt back in explicitly via their subprocess ``env`` overlay.

    :param monkeypatch: Pytest fixture used to delete the env vars per-test.
    """
    scrub_gate_mode_env(monkeypatch)
