"""Shared fixtures for the ``tests/integration`` package.

Re-exports ``fake_r2_remote`` from ``tests/pipeline/conftest.py`` so integration
tests that exercise the launcher against a local-typed rclone remote (instead
of real R2) can pick it up without duplicating the fixture.
"""

from __future__ import annotations

from tests.pipeline.conftest import fake_r2_remote  # noqa: F401 — pytest fixture re-export
