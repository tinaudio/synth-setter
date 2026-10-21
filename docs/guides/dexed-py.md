# Native Dexed rendering

Select `synth=dexed_py render=dexed` in dataset-generation configurations to
use [dexed-py](https://github.com/DBraun/dexed-py) without a VST host, plugin
binary, display server, or preset file. The audio dependency group pins
`dexed-py==0.3.0`; the renderer rejects a different installed/configured version.
The upstream package is GPL-3.0-or-later.

## Parameters and audio

`dexed_py` has 145 synth coordinates in the order of upstream
`Preset.to_array()`, plus MIDI pitch and two timing coordinates: 148 encoded
values total. Array fields use dotted zero-based coordinates, for example
`op_env_rates.0.2` and `op_output_level.5`. Continuous native values use [0, 1];
discrete native values retain their integer ranges (algorithm 0–31, for example).
The project's parameter encoder maps both to [0, 1] for model/storage use.
This specification is **not** a Dexed VST automation vector.

Each render starts with a fresh `DexedSynth` and upstream default `Preset`, then
applies the supplied parameters. The backend requires
`plugin_reload_cadence: render` and `gui_toggle_cadence: never`. It accepts one
or two output channels; stereo duplicates the native mono waveform. Feedback
normalization stays at the upstream default, disabled.

Native gain is preserved, with no clipping or per-patch normalization. Some
patches (including the default patch) can exceed [-1, 1]; dataset generation
uses its existing clipping/loudness rejection and retry logic. The render config
uses float32 audio. SysEx banks, custom operator graphs, and individual-operator
outputs are not exposed by this backend.

## Direct render

```python
from synth_setter.data.vst.dexed_renderer import DexedRenderer

renderer = DexedRenderer(
    plugin_path="dexed",
    sample_rate=44100,
    channels=1,
    signal_duration_seconds=1.5,
)
audio = renderer.render(
    {"algorithm": 15, "feedback": 0.5, "op_output_level.0": 0.6},
    midi_note=60,
    velocity=100,
    note_start_and_end=(0.0, 1.0),
)
# audio.shape == (1, 66150)
```

## Verification

```bash
.venv/bin/python -m pytest tests/data/vst/test_dexed_*.py
```

Tests exercise the real native engine, parameter encoding, configuration and
factory selection, repeatability across renders, and dataset-row generation
including mel features and an MP3 preview. Existing Dexed VST installation
support remains unchanged.
