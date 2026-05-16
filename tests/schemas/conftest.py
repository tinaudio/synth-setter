"""Shared fixtures for the ``tests/schemas/`` module.

The Hydra global-state dance (``GlobalHydra.instance().clear()`` before each
``initialize`` block) belongs in a fixture rather than copy-pasted into every
helper — that way an xdist-parallelised run can't race on the singleton, and
a mid-compose exception in one test can't leak initialised state into the
next.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from hydra.core.global_hydra import GlobalHydra


@pytest.fixture(autouse=True)
def clean_global_hydra() -> Iterator[None]:
    """Clear Hydra's global singleton before and after every test in this package."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
