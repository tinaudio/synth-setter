"""Shared fixtures for the introspection test modules."""

from __future__ import annotations

import random

import numpy as np
import pytest

from tests.data.vst._introspect_fakes import IntrospectFakeParameter, IntrospectFakePlugin


@pytest.fixture
def fake_plugin() -> IntrospectFakePlugin:
    """Build a two-parameter fake plugin standing in for the loaded VST3.

    :returns: Fake with one continuous and one categorical parameter.
    """
    return IntrospectFakePlugin(
        {
            # Dense sweep: a real continuous knob reports a value per host step,
            # far above the numeric categorical cap.
            "cutoff": IntrospectFakeParameter(float, [i / 100 for i in range(101)]),
            "filter_type": IntrospectFakeParameter(str, ["LP", "HP"], raw_values=[0.0, 1.0]),
        },
        preset_data=b"VST3\x01\x00fake-state",
        name="Fake Synth",
    )


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
