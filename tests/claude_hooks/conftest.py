"""Shared fixtures for the hook tests under ``tests/claude_hooks/``."""

from __future__ import annotations

import pytest

# Gate knobs agent/hooks/* read with defaults (block/warn modes plus the
# REVIEW_MAX_LAG threshold); agent-session overrides leak via os.environ.
GATE_OVERRIDE_ENV_VARS = (
    "PR_READINESS_GATE",
    "PR_TITLE_GATE",
    "REVIEW_BLOCK_GATE",
    "REVIEW_COMMENT_GATE",
    "REVIEW_MAX_LAG",
    "WORKTREE_GUARD_MODE",
)


def scrub_gate_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delete every gate override from ``os.environ``.

    :param monkeypatch: Owns the deletions; restores them at its teardown.
    """
    for var in GATE_OVERRIDE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _scrub_gate_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run hooks with their documented default modes in every test here.

    Intentionally directory-wide: any test in this package may spawn a hook
    subprocess that inherits ``os.environ``. Tests exercising a non-default
    mode opt back in explicitly via their subprocess ``env`` overlay.

    :param monkeypatch: Pytest fixture used to delete the env vars per-test.
    """
    scrub_gate_overrides(monkeypatch)
