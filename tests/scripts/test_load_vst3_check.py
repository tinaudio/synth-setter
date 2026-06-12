"""Unit tests for the load_vst3_check CLI contract.

The real per-synth loads run in tests/docker/test_smoke.py inside the image;
these host-side tests pin the script's argv normalization and failure exits
with a stubbed ``VST3Plugin`` so they stay in the fast suite.
"""

import pytest

pytest.importorskip("pedalboard")

from synth_setter.scripts import load_vst3_check  # noqa: E402

# (bundle_path, plugin_name) of the most recent stub construction.
_last_init: "list[tuple[str, str | None]]" = []


class _StubPlugin:
    """Records constructor args and exposes a fixed parameters mapping."""

    def __init__(self, bundle_path: str, plugin_name: "str | None" = None) -> None:
        """Record the construction and expose one fake parameter.

        :param bundle_path: Path the caller asked to load.
        :param plugin_name: Plugin selector the caller passed, if any.
        """
        _last_init.append((bundle_path, plugin_name))
        self.parameters = {"osc1": object()}


class _ZeroParamPlugin(_StubPlugin):
    """Stub whose bundle loads but exposes no parameters."""

    def __init__(self, bundle_path: str, plugin_name: "str | None" = None) -> None:
        """Record the construction and expose no parameters.

        :param bundle_path: Path the caller asked to load.
        :param plugin_name: Plugin selector the caller passed, if any.
        """
        super().__init__(bundle_path, plugin_name)
        self.parameters = {}


def test_main_missing_bundle_arg_exits_with_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare invocation exits with a usage message, not an IndexError.

    :param monkeypatch: Pytest fixture for argv and VST3Plugin stubbing.
    """
    monkeypatch.setattr("sys.argv", ["load_vst3_check.py"])
    with pytest.raises(SystemExit, match="usage:"):
        load_vst3_check.main()


def test_main_zero_parameters_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bundle that loads but exposes no parameters fails the check.

    :param monkeypatch: Pytest fixture for argv and VST3Plugin stubbing.
    """
    monkeypatch.setattr(load_vst3_check, "VST3Plugin", _ZeroParamPlugin)
    monkeypatch.setattr("sys.argv", ["load_vst3_check.py", "/fake/Synth.vst3"])
    with pytest.raises(SystemExit, match="no parameters"):
        load_vst3_check.main()


def test_main_missing_plugin_name_loads_sole_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting the plugin-name argument loads the bundle's sole plugin.

    :param monkeypatch: Pytest fixture for argv and VST3Plugin stubbing.
    """
    monkeypatch.setattr(load_vst3_check, "VST3Plugin", _StubPlugin)
    monkeypatch.setattr("sys.argv", ["load_vst3_check.py", "/fake/Synth.vst3"])
    load_vst3_check.main()
    assert _last_init[-1] == ("/fake/Synth.vst3", None)


def test_main_empty_plugin_name_normalizes_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty plugin-name argument normalizes to None.

    :param monkeypatch: Pytest fixture for argv and VST3Plugin stubbing.
    """
    monkeypatch.setattr(load_vst3_check, "VST3Plugin", _StubPlugin)
    monkeypatch.setattr("sys.argv", ["load_vst3_check.py", "/fake/Synth.vst3", ""])
    load_vst3_check.main()
    assert _last_init[-1] == ("/fake/Synth.vst3", None)


def test_main_plugin_name_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-empty plugin-name argument reaches VST3Plugin unchanged.

    :param monkeypatch: Pytest fixture for argv and VST3Plugin stubbing.
    """
    monkeypatch.setattr(load_vst3_check, "VST3Plugin", _StubPlugin)
    monkeypatch.setattr("sys.argv", ["load_vst3_check.py", "/fake/Synth.vst3", "Six Sines"])
    load_vst3_check.main()
    assert _last_init[-1] == ("/fake/Synth.vst3", "Six Sines")
