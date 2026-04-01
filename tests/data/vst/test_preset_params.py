"""Regression test: preset-dependent VST parameters require post-load flush.

Surge XT dynamically exposes parameters based on oscillator type. The
surge-base preset sets an oscillator type that exposes waveform mix
parameters (sawtooth, pulse, triangle). These only appear in the
plugin's parameter map after a process()+reset() cycle following
load_preset().

Regression for: https://github.com/tinaudio/synth-setter/issues/225
"""

from pathlib import Path

import pytest

PLUGIN_PATH = "/usr/lib/vst3/Surge XT.vst3"
PRESET_PATH = "presets/surge-base.vstpreset"

# pedalboard.VST3Plugin.parameters is a dynamic C extension attribute that
# pyright cannot resolve statically. All .parameters accesses below use
# type: ignore[attr-defined] for this reason.

requires_vst = pytest.mark.requires_vst
skip_no_vst = pytest.mark.skipif(
    not Path(PLUGIN_PATH).exists(),
    reason=f"VST plugin not found at {PLUGIN_PATH}",
)


@requires_vst
@skip_no_vst
def test_preset_dependent_params_accessible_after_flush():
    # plumb:req-b3db5915
    """Preset-dependent params (e.g. sawtooth) exist after load + flush + reset."""
    from pedalboard import VST3Plugin

    plugin = VST3Plugin(PLUGIN_PATH)
    plugin.load_preset(PRESET_PATH)
    plugin.process([], 32.0, 44100, 2, 2048, True)
    plugin.reset()

    assert "a_osc_1_sawtooth" in plugin.parameters  # type: ignore[attr-defined]
    assert "a_osc_1_pulse" in plugin.parameters  # type: ignore[attr-defined]
    assert "a_osc_1_triangle" in plugin.parameters  # type: ignore[attr-defined]


@requires_vst
@skip_no_vst
def test_preset_dependent_params_missing_without_flush():
    """Without flush+reset after load_preset, dynamic params are not exposed."""
    from pedalboard import VST3Plugin

    plugin = VST3Plugin(PLUGIN_PATH)
    plugin.load_preset(PRESET_PATH)

    assert "a_osc_1_sawtooth" not in plugin.parameters  # type: ignore[attr-defined]


@requires_vst
@skip_no_vst
def test_render_params_sets_preset_dependent_param():
    """render_params must successfully set preset-dependent params."""
    from pedalboard import VST3Plugin

    from src.data.vst.core import render_params

    plugin = VST3Plugin(PLUGIN_PATH)
    params = {"a_osc_1_sawtooth": 0.5}

    # Should not raise KeyError
    render_params(
        plugin,
        params=params,
        midi_note=60,
        velocity=100,
        note_start_and_end=(0.0, 0.5),
        signal_duration_seconds=1.0,
        sample_rate=44100.0,
        channels=2,
        preset_path=PRESET_PATH,
    )

    assert plugin.parameters["a_osc_1_sawtooth"].raw_value == pytest.approx(  # type: ignore[attr-defined]
        0.5, abs=0.01
    )
