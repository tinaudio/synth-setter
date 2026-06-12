"""Tests for the pedalboard-free param-spec registry helpers."""

from __future__ import annotations

import pytest

from synth_setter.data.vst.param_spec_registry import (
    default_plugin_path,
    param_specs,
    preset_paths,
)

_ENV_VAR = "SYNTH_SETTER_PLUGIN_PATH"
_BUNDLED_PATH = "plugins/Surge XT.vst3"


def test_default_plugin_path_falls_back_to_bundle_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the override unset, the helper returns the in-repo bundle path.

    :param monkeypatch: removes ``SYNTH_SETTER_PLUGIN_PATH``.
    """
    monkeypatch.delenv(_ENV_VAR, raising=False)
    assert default_plugin_path() == _BUNDLED_PATH


def test_default_plugin_path_uses_env_override_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-empty ``SYNTH_SETTER_PLUGIN_PATH`` overrides the bundle path.

    :param monkeypatch: sets a custom override path.
    """
    monkeypatch.setenv(_ENV_VAR, "/custom/Surge XT.vst3")
    assert default_plugin_path() == "/custom/Surge XT.vst3"


def test_default_plugin_path_falls_back_to_bundle_when_env_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty override falls back to the bundle (the ``or`` semantic).

    :param monkeypatch: sets an empty ``SYNTH_SETTER_PLUGIN_PATH``.
    """
    monkeypatch.setenv(_ENV_VAR, "")
    assert default_plugin_path() == _BUNDLED_PATH


def test_param_spec_widths_match_known_values() -> None:
    """Hardcoded width tripwires guard the shipped specs against silent drift."""
    assert len(param_specs["surge_xt"]) == 300
    assert len(param_specs["surge_simple"]) == 92
    assert len(param_specs["surge_4"]) == 7


def test_every_param_spec_has_a_preset_path() -> None:
    """``param_specs`` and ``preset_paths`` cover the same keys — no spec lacks a preset."""
    assert set(param_specs) == set(preset_paths)
