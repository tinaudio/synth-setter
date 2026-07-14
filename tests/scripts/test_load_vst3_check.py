"""Unit tests for the load_vst3_check CLI contract.

The real per-synth loads run in tests/docker/test_smoke.py inside the image;
these host-side tests pin the script's argv normalization and failure exits
with a stubbed ``VST3Plugin`` so they stay in the fast suite.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("pedalboard")

from synth_setter.scripts import load_vst3_check  # noqa: E402

# (bundle_path, plugin_name) appended per stub construction; cleared per test.
_init_calls: list[tuple[str, str | None]] = []


class _StubPlugin:
    """Records constructor args and exposes a fixed parameters mapping."""

    def __init__(self, bundle_path: str, plugin_name: str | None = None) -> None:
        """Record the construction and expose one fake parameter.

        :param bundle_path: Path the caller asked to load.
        :param plugin_name: Plugin selector the caller passed, if any.
        """
        _init_calls.append((bundle_path, plugin_name))
        self.parameters = {"osc1": object()}


class _ZeroParamPlugin(_StubPlugin):
    """Stub whose bundle loads but exposes no parameters."""

    def __init__(self, bundle_path: str, plugin_name: str | None = None) -> None:
        """Record the construction and expose no parameters.

        :param bundle_path: Path the caller asked to load.
        :param plugin_name: Plugin selector the caller passed, if any.
        """
        super().__init__(bundle_path, plugin_name)
        self.parameters = {}


@pytest.fixture(autouse=True)
def _fresh_init_calls() -> None:
    """Clear the recorder so each test asserts only its own constructions."""
    _init_calls.clear()


def test_main_missing_bundle_arg_exits_with_usage() -> None:
    """Bare invocation exits non-zero with a usage message, not an IndexError."""
    with pytest.raises(SystemExit, match="usage:") as excinfo:
        load_vst3_check.main([])
    assert excinfo.value.code


def test_main_zero_parameters_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bundle that loads but exposes no parameters fails the check.

    :param monkeypatch: Pytest fixture for VST3Plugin stubbing.
    """
    monkeypatch.setattr(load_vst3_check, "VST3Plugin", _ZeroParamPlugin)
    with pytest.raises(SystemExit, match="no parameters") as excinfo:
        load_vst3_check.main(["/fake/Synth.vst3"])
    assert excinfo.value.code


def test_main_missing_plugin_name_loads_sole_plugin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Omitting the plugin-name argument loads the bundle's sole plugin.

    :param monkeypatch: Pytest fixture for VST3Plugin stubbing.
    :param capsys: Captures the success-path param_count report.
    """
    monkeypatch.setattr(load_vst3_check, "VST3Plugin", _StubPlugin)
    load_vst3_check.main(["/fake/Synth.vst3"])
    assert _init_calls == [("/fake/Synth.vst3", None)]
    assert "param_count=1" in capsys.readouterr().out


def test_main_empty_plugin_name_normalizes_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty plugin-name argument normalizes to None.

    :param monkeypatch: Pytest fixture for VST3Plugin stubbing.
    """
    monkeypatch.setattr(load_vst3_check, "VST3Plugin", _StubPlugin)
    load_vst3_check.main(["/fake/Synth.vst3", ""])
    assert _init_calls == [("/fake/Synth.vst3", None)]


def test_main_plugin_name_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-empty plugin-name argument reaches VST3Plugin unchanged.

    :param monkeypatch: Pytest fixture for VST3Plugin stubbing.
    """
    monkeypatch.setattr(load_vst3_check, "VST3Plugin", _StubPlugin)
    load_vst3_check.main(["/fake/Synth.vst3", "Six Sines"])
    assert _init_calls == [("/fake/Synth.vst3", "Six Sines")]


def test_module_execution_bare_invocation_exits_with_usage() -> None:
    """Bare ``python -m`` invocation (the smoke-test entry point) exits non-zero with usage."""
    result = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [sys.executable, "-m", "synth_setter.scripts.load_vst3_check"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode != 0
    assert "usage:" in result.stderr
