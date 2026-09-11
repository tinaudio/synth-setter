"""Preset-coverage audit: flush pattern vs. show_editor pattern produce the same params.

Justifies skipping ``show_editor`` on Darwin (#714) where it accumulates AppKit
commit-handler state and crashes the unbundled python process after a few
plugin reloads. If this test ever finds a divergence — for any preset, any
parameter, any pedalboard or plugin version — dropping ``show_editor`` would
silently fall back to plugin defaults for the diverging parameter, which is
exactly the failure mode this guard exists to prevent.

Pattern compared:
    A) ``VST3Plugin → load_preset → flush``  (the render_params path)
    B) ``VST3Plugin → show_editor → load_preset → flush``  (the pedalboard #394
       workaround order, kept on Linux)
"""

import sys
from pathlib import Path

import pytest
from pedalboard import VST3Plugin

from synth_setter.data.vst.core import warmup_plugin
from synth_setter.synth_spec import SYNTHS, SynthSpec

_PRESET_DIR = Path("presets")
_SAMPLE_RATE = 44100.0
_CHANNELS = 2
_FLUSH_DURATION_S = 32.0
_FLUSH_BLOCK_SIZE = 2048

# pedalboard.VST3Plugin.parameters is a dynamic C extension attribute that
# pyright cannot resolve statically. All .parameters accesses below use
# type: ignore[attr-defined] for this reason.

requires_vst = pytest.mark.requires_vst
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


def _preset_synths() -> list[SynthSpec]:
    """Resolve every packaged preset to its registered synth identity.

    :returns: Registered synths ordered by preset path.
    :raises ValueError: If a packaged preset has no registered owner.
    """
    synths_by_preset = {
        synth.plugin_state_path: synth for synth in SYNTHS.values() if synth.plugin_state_path
    }
    preset_paths = sorted(path.as_posix() for path in _PRESET_DIR.glob("*.vstpreset"))
    unregistered_paths = set(preset_paths).difference(synths_by_preset)
    if unregistered_paths:
        raise ValueError(f"Preset files have no registered synth owner: {sorted(unregistered_paths)}")
    return [synths_by_preset[path] for path in preset_paths]


@pytest.mark.parametrize(
    "synth",
    [pytest.param(synth, id=synth.plugin_state_path) for synth in _preset_synths()],
)
@pytest.mark.slow
@requires_vst
@skip_darwin
def test_flush_pattern_matches_show_editor_pattern(synth: SynthSpec) -> None:
    """Flush pattern in render_params commits each registered synth's preset state.

    :param synth: Registered owner of the plugin-state file under test.
    """
    if not Path(synth.plugin_path).exists():
        pytest.skip(f"Preset owner plugin not found at {synth.plugin_path!r} for {synth.name}")

    p_no = VST3Plugin(synth.plugin_path)
    p_no.load_preset(synth.plugin_state_path)
    _flush(p_no)
    no_editor_state = _read_all_params(p_no)

    p_we = VST3Plugin(synth.plugin_path)
    # Production's editor warm-up (spotify/pedalboard#394): show_editor closes via
    # the threading.Event the editor exposes, not a test-local wall-clock sleep.
    warmup_plugin(p_we)
    p_we.load_preset(synth.plugin_state_path)
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
