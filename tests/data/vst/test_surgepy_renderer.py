"""SurgePy renderer contracts against the real in-process Surge engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from synth_setter.data.vst.param_map import load_param_map
from synth_setter.data.vst.renderers import SurgePyRenderer
from synth_setter.data.vst.surgepy_runtime import surge_component_state


def test_surge_component_state_rejects_unstructured_marker_bytes(tmp_path: Path) -> None:
    """Marker-like bytes outside a valid container cannot establish provenance.

    :param tmp_path: Temporary malformed preset destination.
    """
    malformed = tmp_path / "decoy.vstpreset"
    malformed.write_bytes(b"sub3decoy JUCEPrivateData")

    with pytest.raises(ValueError, match="not a VST3 preset container"):
        surge_component_state(malformed)


@pytest.mark.slow
def test_surgepy_renderer_real_patch_produces_finite_non_silent_audio() -> None:
    """A real Surge patch and note produce the shared renderer output contract."""
    renderer = SurgePyRenderer(
        plugin_path="surgepy",
        sample_rate=48_000,
        channels=2,
        signal_duration_seconds=1.0,
        plugin_state_path="presets/surge-base.fxp",
        parameter_map=load_param_map(
            Path("src/synth_setter/data/vst/surge_xt_param_map.json")
        ),
    )

    audio = renderer.render(
        {"a_amp_eg_attack": 0.2, "a_filter_1_cutoff": 0.7},
        midi_note=60,
        velocity=100,
        note_start_and_end=(0.0, 0.5),
    )

    assert audio.shape == (2, 48_000)
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 1e-4


@pytest.mark.slow
def test_surgepy_renderer_mono_downmix_preserves_sample_count() -> None:
    """The native stereo output can satisfy the shared mono contract."""
    renderer = SurgePyRenderer(
        plugin_path="surgepy",
        sample_rate=44_100,
        channels=1,
        signal_duration_seconds=0.1,
        plugin_state_path="presets/surge-simple.fxp",
        parameter_map=load_param_map(
            Path("src/synth_setter/data/vst/surge_simple_param_map.json")
        ),
    )

    audio = renderer.render({}, 60, 100, (0.0, 0.05))

    assert audio.shape == (1, 4_410)
    assert np.isfinite(audio).all()


@pytest.mark.slow
def test_surgepy_renderer_matches_surge_discrete_normalization() -> None:
    """Integer and Boolean boundaries reproduce Surge host automation semantics."""
    renderer = SurgePyRenderer(
        plugin_path="surgepy",
        sample_rate=44_100,
        channels=2,
        signal_duration_seconds=0.1,
        plugin_state_path="presets/surge-base.fxp",
        parameter_map=load_param_map(
            Path("src/synth_setter/data/vst/surge_xt_param_map.json")
        ),
    )
    renderer._initialize_synth()

    octave = renderer._parameter_ids["a_osc_1_octave"]
    mute = renderer._parameter_ids["a_osc_1_mute"]

    assert renderer._native_parameter_value(octave, 0.0) == -3.0
    assert renderer._native_parameter_value(octave, 0.084) == -3.0
    assert renderer._native_parameter_value(octave, 1.0) == 3.0
    assert renderer._native_parameter_value(mute, 0.0) == 0.0
    assert renderer._native_parameter_value(mute, 0.5) == 0.0
    assert renderer._native_parameter_value(mute, 0.500_001) == 1.0
    assert renderer._native_parameter_value(mute, 1.0) == 1.0
    for invalid in (-0.001, 1.001, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and in"):
            renderer._native_parameter_value(octave, invalid)


@pytest.mark.slow
def test_surgepy_renderer_rejects_undersized_native_output_buffer() -> None:
    """Native block processing fails before writing beyond the output buffer."""
    renderer = SurgePyRenderer(
        plugin_path="surgepy",
        sample_rate=44_100,
        channels=2,
        signal_duration_seconds=0.1,
        plugin_state_path="presets/surge-simple.fxp",
        parameter_map=load_param_map(
            Path("src/synth_setter/data/vst/surge_simple_param_map.json")
        ),
    )
    renderer._initialize_synth()
    one_block = renderer.synth.createMultiBlock(1)

    with pytest.raises(ValueError, match="output buffer"):
        renderer._process_blocks(one_block, start_block=0, num_blocks=2)


@pytest.mark.slow
def test_surgepy_renderer_short_note_window_spans_a_processing_block() -> None:
    """Sub-block MIDI windows cannot collapse before native processing."""
    renderer = SurgePyRenderer(
        plugin_path="surgepy",
        sample_rate=44_100,
        channels=2,
        signal_duration_seconds=0.1,
        plugin_state_path="presets/surge-simple.fxp",
        parameter_map=load_param_map(
            Path("src/synth_setter/data/vst/surge_simple_param_map.json")
        ),
    )

    audio = renderer.render({}, 60, 100, (10 / 44_100, 11 / 44_100))

    assert np.max(np.abs(audio)) > 1e-4


@pytest.mark.slow
def test_surgepy_renderer_rechecks_patch_hash_before_each_render(tmp_path: Path) -> None:
    """A patch replaced after construction cannot bypass mapped provenance.

    :param tmp_path: Temporary mutable copy of the checked-in patch.
    """
    patch_path = tmp_path / "surge-base.fxp"
    patch_path.write_bytes(Path("presets/surge-base.fxp").read_bytes())
    renderer = SurgePyRenderer(
        plugin_path="surgepy",
        sample_rate=44_100,
        channels=2,
        signal_duration_seconds=0.1,
        plugin_state_path=str(patch_path),
        parameter_map=load_param_map(
            Path("src/synth_setter/data/vst/surge_xt_param_map.json")
        ),
    )
    patch_path.write_bytes(patch_path.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="patch SHA-256"):
        renderer.render({}, 60, 100, (0.0, 0.05))


def test_surgepy_renderer_rejects_non_surgepy_plugin_identity() -> None:
    """Direct construction fails closed on a VST path."""
    with pytest.raises(ValueError, match='plugin_path="surgepy"'):
        SurgePyRenderer(
            plugin_path="plugin.vst3",
            sample_rate=44_100,
            channels=2,
            signal_duration_seconds=1.0,
            plugin_state_path="presets/surge-base.fxp",
            parameter_map=load_param_map(
                Path("src/synth_setter/data/vst/surge_xt_param_map.json")
            ),
        )
