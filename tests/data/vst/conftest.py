"""Shared fixtures for the introspection test modules."""

from __future__ import annotations

import random

import numpy as np
import pytest


@pytest.fixture
def seeded_rng(request: pytest.FixtureRequest) -> None:
    """Seed the global RNGs ``ParamSpec.sample`` draws from, restoring state on teardown.

    :param request: Fixture request; registers the state-restore finalizer.
    """
    np_state = np.random.get_state()
    py_state = random.getstate()

    def _restore() -> None:
        np.random.set_state(np_state)
        random.setstate(py_state)

    request.addfinalizer(_restore)
    np.random.seed(11)
    random.seed(11)
