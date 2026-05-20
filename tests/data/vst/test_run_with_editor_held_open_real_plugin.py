"""Real-plugin contract test for ``run_with_editor_held_open`` (#1187, #1204).

Pins ``synth_setter.data.vst.core.run_with_editor_held_open`` against the real
Surge XT VST3. The MagicMock-based tests in ``test_core.py`` cannot observe
pedalboard's main-thread invariant for ``show_editor`` — only a real plugin
can. This test exists so a regression where the editor is invoked off the
process main thread surfaces from a test ID that names the broken contract,
not from inside the much-heavier shard-render integration test (#1199).

Runs through ``docker/ubuntu22_04/run-linux-vst-headless.sh`` (Xvfb +
xsettingsd + openbox + dbus) inside the existing ``test-vst-slow.yml``
workflow.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from synth_setter.data.vst import core

_PLUGIN_PATH = os.environ.get("SYNTH_SETTER_PLUGIN_PATH") or "plugins/Surge XT.vst3"

skip_no_vst = pytest.mark.skipif(
    not Path(_PLUGIN_PATH).exists(),
    reason=f"VST plugin not found at {_PLUGIN_PATH}",
)


@pytest.mark.slow
@pytest.mark.requires_vst
@skip_no_vst
def test_run_with_editor_held_open_completes_cleanly_on_real_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_with_editor_held_open(real_plugin, body)`` returns without raising.

    The helper holds the plugin's editor realised on the main thread while
    ``body`` runs on a worker. A regression where the editor codepath
    violates a pedalboard or JUCE invariant surfaces as either an
    editor-thread ``logger.exception("vst-editor-window crashed: ...")`` from
    ``_run_editor`` or — when ``body`` did not raise — a re-raise propagated
    from the worker. This test pins both with an empty body so the failure
    cannot be confused with a render/writer regression.

    :param monkeypatch: Stubs ``core.logger`` so the editor-thread crash log
        is observable (loguru does not propagate to ``caplog``).
    """
    plugin = core.load_plugin(_PLUGIN_PATH)
    fake_logger = MagicMock(wraps=core.logger)
    monkeypatch.setattr(core, "logger", fake_logger)

    result = core.run_with_editor_held_open(plugin, lambda: None)

    assert result is None
    crash_log_calls = [
        call
        for call in fake_logger.exception.call_args_list
        if "vst-editor-window crashed" in str(call.args[0])
    ]
    assert not crash_log_calls, f"editor-thread crash logged: {crash_log_calls}"
