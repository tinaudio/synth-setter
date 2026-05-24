"""Shared fixtures for the VST test suite — fake-plugin injection lives here."""

from __future__ import annotations

import pytest

from synth_setter.data.vst import core
from tests.data.vst._fake_plugin import FakeVST3Plugin


@pytest.fixture
def fake_vst3_plugin() -> FakeVST3Plugin:
    """Return a fresh ``FakeVST3Plugin`` instance per test (no shared state).

    :returns: Stand-in plugin whose ``plugin_path`` field is set but never
        read from disk; downstream production code receives this via
        ``install_fake_plugin``.
    """
    return FakeVST3Plugin("plugins/fake.vst3")


@pytest.fixture
def install_fake_plugin(
    monkeypatch: pytest.MonkeyPatch, fake_vst3_plugin: FakeVST3Plugin
) -> FakeVST3Plugin:
    """Patch ``core.load_plugin`` and ``core.VST3Plugin`` to yield the fake.

    Both seams are covered: ``load_plugin`` is the normal pipeline entry
    point; ``VST3Plugin`` is constructed directly by
    ``extract_renderer_version``'s fallback path.

    :param monkeypatch: Pytest fixture used to swap the two ``core``
        callables for the test's duration; teardown restores both.
    :param fake_vst3_plugin: The instance the patched callables return.
    :returns: The same ``fake_vst3_plugin`` instance, so tests asserting
        on it can compare by identity.
    """
    monkeypatch.setattr(core, "load_plugin", lambda _path, **_kw: fake_vst3_plugin)
    monkeypatch.setattr(core, "VST3Plugin", lambda _path: fake_vst3_plugin)
    return fake_vst3_plugin
