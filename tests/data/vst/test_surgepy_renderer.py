"""SurgePy renderer contracts against the real in-process Surge engine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from synth_setter.data.vst.generate_vst_dataset import (
    AudioAmplitudeError,
    _reject_clipped_audio,
)
from synth_setter.data.vst.param_map import load_param_map
from synth_setter.data.vst.renderers import SurgePyRenderer
from synth_setter.data.vst.surgepy_runtime import surge_component_state


def _simple_renderer() -> SurgePyRenderer:
    """Build the checked-in minimal SurgePy renderer.

    :returns: Renderer configured for a short stereo signal.
    """
    return SurgePyRenderer(
        plugin_path="surgepy",
        sample_rate=44_100,
        channels=2,
        signal_duration_seconds=0.1,
        plugin_state_path="presets/surge-simple.fxp",
        parameter_map=load_param_map(
            Path("src/synth_setter/data/vst/surge_simple_param_map.json")
        ),
    )


def test_surge_component_state_rejects_unstructured_marker_bytes(tmp_path: Path) -> None:
    """Marker-like bytes outside a valid container cannot establish provenance.

    :param tmp_path: Temporary malformed preset destination.
    """
    malformed = tmp_path / "decoy.vstpreset"
    malformed.write_bytes(b"sub3decoy JUCEPrivateData")

    with pytest.raises(ValueError, match="not a VST3 preset container"):
        surge_component_state(malformed)


@pytest.mark.slow
@pytest.mark.requires_surgepy
def test_surgepy_renderer_real_patch_produces_finite_non_silent_audio() -> None:
    """A real Surge patch and note produce the shared renderer output contract."""
    renderer = SurgePyRenderer(
        plugin_path="surgepy",
        sample_rate=48_000,
        channels=2,
        signal_duration_seconds=1.0,
        plugin_state_path="presets/surge-base.fxp",
        parameter_map=load_param_map(Path("src/synth_setter/data/vst/surge_xt_param_map.json")),
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
@pytest.mark.requires_surgepy
def test_surgepy_renderer_render_rejects_unknown_parameter_key() -> None:
    """The public render contract rejects parameters absent from the joint map."""
    with pytest.raises(KeyError, match=r"unknown SurgePy parameter key\(s\): not_mapped"):
        _simple_renderer().render({"not_mapped": 0.5}, 60, 100, (0.0, 0.05))


@pytest.mark.slow
@pytest.mark.requires_surgepy
@pytest.mark.parametrize(
    "note_interval",
    [
        pytest.param((0.05, 0.05), id="start-equals-end"),
        pytest.param((0.0, 0.100_001), id="end-after-signal"),
    ],
)
def test_surgepy_renderer_render_rejects_malformed_note_interval(
    note_interval: tuple[float, float],
) -> None:
    """The public render contract enforces an ordered, in-signal note interval.

    :param note_interval: Invalid note interval under test.
    """
    with pytest.raises(ValueError, match="note times must satisfy"):
        _simple_renderer().render({}, 60, 100, note_interval)


@pytest.mark.slow
@pytest.mark.requires_surgepy
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
@pytest.mark.requires_surgepy
def test_surgepy_renderer_defers_clipped_audio_to_the_generation_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Amplitude acceptance stays with generation, which retries the draw (#2001).

    :param monkeypatch: Replaces the native block-render seam with clipped output.
    """
    renderer = _simple_renderer()
    clipped = np.full((2, 4_410), 1.01, dtype=np.float32)
    monkeypatch.setattr(SurgePyRenderer, "_render_note_blocks", lambda *_, **__: clipped)

    rendered = renderer.render({}, 60, 100, (0.0, 0.05))

    assert np.array_equal(rendered, clipped)
    with pytest.raises(AudioAmplitudeError, match=r"within \[-1, 1\]"):
        _reject_clipped_audio(rendered)


@pytest.mark.slow
@pytest.mark.requires_surgepy
def test_surgepy_renderer_matches_surge_native_normalization() -> None:
    """Float, integer, and Boolean values reproduce Surge automation semantics."""
    renderer = SurgePyRenderer(
        plugin_path="surgepy",
        sample_rate=44_100,
        channels=2,
        signal_duration_seconds=0.1,
        plugin_state_path="presets/surge-base.fxp",
        parameter_map=load_param_map(Path("src/synth_setter/data/vst/surge_xt_param_map.json")),
    )
    # Continuous cutoff spans [-60, 70] dB-scaled native units; integer octave and
    # Boolean mute round through Surge's legacy automation grid.
    assert renderer.native_parameter_values(
        {
            "a_filter_1_cutoff": 0.0,
            "a_osc_1_octave": 0.0,
            "a_osc_1_mute": 0.0,
        }
    ) == {"a_filter_1_cutoff": -60.0, "a_osc_1_octave": -3.0, "a_osc_1_mute": 0.0}
    assert renderer.native_parameter_values(
        {
            "a_filter_1_cutoff": 0.5,
            "a_osc_1_octave": 0.084,
            "a_osc_1_mute": 0.5,
        }
    ) == {"a_filter_1_cutoff": 5.0, "a_osc_1_octave": -3.0, "a_osc_1_mute": 0.0}
    assert renderer.native_parameter_values(
        {
            "a_filter_1_cutoff": 1.0,
            "a_osc_1_octave": 1.0,
            "a_osc_1_mute": 0.500_001,
        }
    ) == {"a_filter_1_cutoff": 70.0, "a_osc_1_octave": 3.0, "a_osc_1_mute": 1.0}
    for invalid in (-0.001, 1.001, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and in"):
            renderer.native_parameter_values({"a_osc_1_octave": invalid})


@pytest.mark.slow
@pytest.mark.requires_surgepy
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
@pytest.mark.requires_surgepy
def test_surgepy_renderer_non_block_aligned_note_has_no_early_audio() -> None:
    """A requested note cannot sound before its sample-accurate start time."""
    renderer = _simple_renderer()
    note_start_sample = 336

    audio = renderer.render(
        {},
        60,
        100,
        (note_start_sample / renderer.sample_rate, 656 / renderer.sample_rate),
    )

    audible = np.flatnonzero(np.max(np.abs(audio), axis=0) > 1e-8)
    assert audible[0] == note_start_sample


def test_surgepy_renderer_non_block_aligned_note_preserves_first_native_sample() -> None:
    """Sub-block alignment delays, rather than discards, the native note attack."""

    block_size = 64
    state = {"is_playing": False, "note_age": 0}
    synth = Mock()
    synth.getBlockSize.return_value = block_size
    synth.createMultiBlock.side_effect = lambda capacity: np.zeros(
        (2, capacity * block_size), dtype=np.float32
    )
    synth.playNote.side_effect = lambda *_: state.__setitem__("is_playing", True)
    synth.releaseNote.side_effect = lambda *_: state.__setitem__("is_playing", False)
    synth.allNotesOff.side_effect = lambda: state.__setitem__("is_playing", False)

    def process_blocks(
        output: np.ndarray,
        startBlock: int = 0,
        nBlocks: int = -1,
    ) -> None:
        """Emit increasing note-age samples into active native blocks.

        :param output: Stereo destination buffer.
        :param startBlock: First destination block.
        :param nBlocks: Number of blocks to process.
        """
        for block in range(nBlocks):
            left = (startBlock + block) * block_size
            if state["is_playing"]:
                first_sample = state["note_age"] + 1
                values = np.arange(
                    first_sample,
                    first_sample + block_size,
                    dtype=np.float32,
                )
                output[:, left : left + block_size] = values
                state["note_age"] += block_size

    synth.processMultiBlock.side_effect = process_blocks
    renderer = object.__new__(SurgePyRenderer)
    renderer.sample_rate = 44_100
    renderer.synth = synth

    audio = renderer._render_note_blocks(
        midi_note=60,
        velocity=100,
        samples=1_024,
        start=336 / 44_100,
        end=656 / 44_100,
    )

    expected_attack = np.tile(np.arange(1, 65, dtype=np.float32), (2, 1))
    np.testing.assert_array_equal(audio[:, 336:400], expected_attack)


@pytest.mark.slow
@pytest.mark.requires_surgepy
def test_surgepy_renderer_final_partial_block_note_is_audible() -> None:
    """A valid note in the final partial block cannot collapse to silence."""
    renderer = _simple_renderer()
    note_start_sample = 4_401

    audio = renderer.render(
        {},
        60,
        100,
        (note_start_sample / renderer.sample_rate, 4_409 / renderer.sample_rate),
    )

    audible = np.flatnonzero(np.max(np.abs(audio), axis=0) > 1e-8)
    assert audible[0] == note_start_sample


@pytest.mark.slow
@pytest.mark.requires_surgepy
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
@pytest.mark.requires_surgepy
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
        parameter_map=load_param_map(Path("src/synth_setter/data/vst/surge_xt_param_map.json")),
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
