"""Preset-coverage audit: flush pattern vs. show_editor pattern produce the same params.

Justifies skipping ``show_editor`` on Darwin (#714) where it accumulates AppKit
commit-handler state and crashes the unbundled python process after a few
plugin reloads. If this test ever finds a divergence — for any preset, any
parameter, any pedalboard or Surge XT version — dropping ``show_editor`` would
silently fall back to Surge defaults for the diverging parameter, which is
exactly the failure mode this guard exists to prevent.

Pattern compared:
    A) ``VST3Plugin → load_preset → flush``  (the render_params path)
    B) ``VST3Plugin → show_editor → load_preset → flush``  (the pedalboard #394
       workaround order, kept on Linux)
"""

import os
import sys
from pathlib import Path

import pytest
from pedalboard import VST3Plugin

from synth_setter.data.vst.core import warmup_plugin

_PLUGIN_PATH = os.environ.get("SYNTH_SETTER_PLUGIN_PATH") or "plugins/Surge XT.vst3"
_PRESET_DIR = Path("presets")
_SAMPLE_RATE = 44100.0
_CHANNELS = 2
_FLUSH_DURATION_S = 32.0
_FLUSH_BLOCK_SIZE = 2048

# pedalboard.VST3Plugin.parameters is a dynamic C extension attribute that
# pyright cannot resolve statically. All .parameters accesses below use
# type: ignore[attr-defined] for this reason.

requires_vst = pytest.mark.requires_vst
skip_no_vst = pytest.mark.skipif(
    not Path(_PLUGIN_PATH).exists(),
    reason=f"VST plugin not found at {_PLUGIN_PATH}",
)
skip_darwin = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="show_editor SIGTRAPs on Darwin (#714) — the very crash this PR avoids",
)


def _flush(plugin: VST3Plugin) -> None:
    """Run a silent process()+reset() to commit pending preset state."""
    plugin.process([], _FLUSH_DURATION_S, _SAMPLE_RATE, _CHANNELS, _FLUSH_BLOCK_SIZE, True)
    plugin.reset()


def _read_all_params(plugin: VST3Plugin) -> dict[str, float]:
    """Snapshot every parameter's raw value into a name -> value mapping."""
    return {
        k: plugin.parameters[k].raw_value  # type: ignore[attr-defined]
        for k in plugin.parameters.keys()  # type: ignore[attr-defined]
    }


@pytest.mark.parametrize(
    "preset_path",
    sorted(p.as_posix() for p in _PRESET_DIR.glob("*.vstpreset")),
)
@pytest.mark.slow
@requires_vst
@skip_no_vst
@skip_darwin
def test_flush_pattern_matches_show_editor_pattern(preset_path: str) -> None:
    """Flush pattern in render_params is sufficient to commit Surge XT preset state."""
    p_no = VST3Plugin(_PLUGIN_PATH)
    p_no.load_preset(preset_path)
    _flush(p_no)
    no_editor_state = _read_all_params(p_no)

    p_we = VST3Plugin(_PLUGIN_PATH)
    # Production's editor warm-up (spotify/pedalboard#394): show_editor closes via
    # the threading.Event the editor exposes, not a test-local wall-clock sleep.
    warmup_plugin(p_we)
    p_we.load_preset(preset_path)
    _flush(p_we)
    with_editor_state = _read_all_params(p_we)

    diffs = {
        k: (with_editor_state.get(k), no_editor_state.get(k))
        for k in set(with_editor_state) | set(no_editor_state)
        if with_editor_state.get(k) != no_editor_state.get(k)
    }
    assert not diffs, (
        f"flush pattern diverged from show_editor pattern for "
        f"{len(diffs)} param(s): {dict(list(diffs.items())[:5])}..."
    )
