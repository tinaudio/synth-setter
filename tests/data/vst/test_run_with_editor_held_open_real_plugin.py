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
def test_run_with_editor_held_open_completes_cleanly_on_real_plugin() -> None:
    """``run_with_editor_held_open(real_plugin, body)`` returns without raising.

    The helper runs ``body`` on a worker thread while ``show_editor`` blocks
    the caller on the main thread (#1187). Any pedalboard or JUCE invariant
    violation in that codepath surfaces as a ``RuntimeError`` from
    ``show_editor`` (e.g. the "Plugin UI windows can only be shown from the
    main thread" guard); a body exception is re-raised after the worker
    drains; a worker that outlives the post-``show_editor`` join window
    raises ``core.RenderWorkerLeaked``. The empty ``body`` keeps the
    assertion focused on the threading contract so a failure cannot be
    confused with a render/writer regression.
    """
    plugin = core.load_plugin(_PLUGIN_PATH)

    result = core.run_with_editor_held_open(plugin, lambda: None)

    assert result is None
