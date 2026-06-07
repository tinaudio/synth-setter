"""Tests for the shared VST plugin-discovery helper (``tests._vst``).

``PLUGIN_PATH`` and ``VST_AVAILABLE`` are resolved at import, so each case reloads
the module under a patched environment and the fixture reloads once more at teardown
to restore the real-environment values for the rest of the session.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

_ENV_VAR = "SYNTH_SETTER_PLUGIN_PATH"
_DEFAULT_PATH = "plugins/Surge XT.vst3"


@pytest.fixture
def reload_vst(request: pytest.FixtureRequest) -> Callable[[], ModuleType]:
    """Reload ``tests._vst`` on demand, restoring real-env values at teardown.

    The finalizer restores ``SYNTH_SETTER_PLUGIN_PATH`` itself (not via
    ``monkeypatch``, whose teardown may run after this one) before the final
    reload, so the module is left resolved against the real environment
    regardless of fixture-finalization order.

    :param request: registers the teardown that restores env + reloads the module.
    :returns: a callable that reloads and returns the freshly imported module.
    """
    original = os.environ.get(_ENV_VAR)

    def _reload() -> ModuleType:
        return importlib.reload(importlib.import_module("tests._vst"))

    def _restore() -> None:
        if original is None:
            os.environ.pop(_ENV_VAR, None)
        else:
            os.environ[_ENV_VAR] = original
        _reload()

    request.addfinalizer(_restore)
    return _reload


@pytest.mark.infra
def test_plugin_path_defaults_to_bundle_when_env_unset(
    monkeypatch: pytest.MonkeyPatch, reload_vst: Callable[[], ModuleType]
) -> None:
    """PLUGIN_PATH falls back to the in-repo bundle when the override is unset.

    :param monkeypatch: removes ``SYNTH_SETTER_PLUGIN_PATH``.
    :param reload_vst: re-resolves the module constants under the patched env.
    """
    monkeypatch.delenv("SYNTH_SETTER_PLUGIN_PATH", raising=False)
    assert reload_vst().PLUGIN_PATH == _DEFAULT_PATH


@pytest.mark.infra
def test_plugin_path_uses_env_override_when_set(
    monkeypatch: pytest.MonkeyPatch, reload_vst: Callable[[], ModuleType]
) -> None:
    """PLUGIN_PATH honors a non-empty ``SYNTH_SETTER_PLUGIN_PATH`` override.

    :param monkeypatch: sets a custom override path.
    :param reload_vst: re-resolves the module constants under the patched env.
    """
    monkeypatch.setenv("SYNTH_SETTER_PLUGIN_PATH", "/custom/Surge XT.vst3")
    assert reload_vst().PLUGIN_PATH == "/custom/Surge XT.vst3"


@pytest.mark.infra
def test_plugin_path_falls_back_to_bundle_when_env_empty(
    monkeypatch: pytest.MonkeyPatch, reload_vst: Callable[[], ModuleType]
) -> None:
    """An empty override falls back to the bundle (the normalized ``or`` semantic).

    :param monkeypatch: sets an empty ``SYNTH_SETTER_PLUGIN_PATH``.
    :param reload_vst: re-resolves the module constants under the patched env.
    """
    monkeypatch.setenv("SYNTH_SETTER_PLUGIN_PATH", "")
    assert reload_vst().PLUGIN_PATH == _DEFAULT_PATH


@pytest.mark.infra
def test_vst_available_true_when_path_exists(
    monkeypatch: pytest.MonkeyPatch, reload_vst: Callable[[], ModuleType], tmp_path: Path
) -> None:
    """VST_AVAILABLE is True when the resolved path exists on disk.

    :param monkeypatch: points the override at an existing file.
    :param reload_vst: re-resolves the module constants under the patched env.
    :param tmp_path: provides a real path to stand in for the plugin bundle.
    """
    plugin = tmp_path / "Surge XT.vst3"
    plugin.touch()
    monkeypatch.setenv("SYNTH_SETTER_PLUGIN_PATH", str(plugin))
    assert reload_vst().VST_AVAILABLE is True


@pytest.mark.infra
def test_vst_available_false_when_path_absent(
    monkeypatch: pytest.MonkeyPatch, reload_vst: Callable[[], ModuleType], tmp_path: Path
) -> None:
    """VST_AVAILABLE is False when the resolved path does not exist.

    :param monkeypatch: points the override at a nonexistent path.
    :param reload_vst: re-resolves the module constants under the patched env.
    :param tmp_path: supplies a guaranteed-absent path.
    """
    monkeypatch.setenv("SYNTH_SETTER_PLUGIN_PATH", str(tmp_path / "missing.vst3"))
    assert reload_vst().VST_AVAILABLE is False
