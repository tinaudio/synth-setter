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
from synth_setter.data.vst.renderers import (
    SurgePyRenderer,
    _align_native_attack,
    _sample_index_at_or_after,
)
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


def _mock_renderer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    block_size: int,
    output_samples: int,
    sample_rate: int = 44_100,
) -> tuple[SurgePyRenderer, Mock, dict[str, int | bool]]:
    """Build a public renderer around a stateful block-processing mock.

    :param monkeypatch: Installs the native synth at the external-engine boundary.
    :param block_size: Native samples per block.
    :param output_samples: Retained stereo output length.
    :param sample_rate: Output samples per second.
    :returns: Renderer, synth mock, and mutable note state.
    """
    state: dict[str, int | bool] = {"playing": False, "note_age": 0}
    synth = Mock()
    synth.getBlockSize.return_value = block_size
    synth.createMultiBlock.return_value = np.zeros(
        (2, output_samples),
        dtype=np.float32,
    )
    synth.playNote.side_effect = lambda *_: state.__setitem__("playing", True)
    synth.releaseNote.side_effect = lambda *_: state.__setitem__("playing", False)
    renderer = _simple_renderer()
    renderer.sample_rate = sample_rate
    renderer.signal_duration_seconds = (output_samples + 0.5) / sample_rate
    monkeypatch.setattr(
        renderer,
        "_initialize_synth",
        lambda: setattr(renderer, "synth", synth),
    )
    monkeypatch.setattr(renderer, "_apply_parameters", lambda _: None)
    return renderer, synth, state


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
@pytest.mark.requires_surgepy
def test_surgepy_renderer_non_block_aligned_note_starts_at_requested_sample() -> None:
    """A native note attack begins at its requested non-block-aligned sample."""
    renderer = _simple_renderer()
    requested_start_sample = 336

    audio = renderer.render(
        {},
        60,
        100,
        (requested_start_sample / renderer.sample_rate, 656 / renderer.sample_rate),
    )

    audible_samples = np.flatnonzero(np.max(np.abs(audio), axis=0) > 1e-8)
    assert audible_samples[0] == requested_start_sample


@pytest.mark.parametrize(
    ("event_time", "expected_sample"),
    [
        pytest.param(13 / 44_100, 13, id="quotient-sample-boundary"),
        pytest.param(
            np.nextafter(17 / 44_100, np.inf),
            18,
            id="rounded-down-just-after-boundary",
        ),
        pytest.param(32 / 44_100, 32, id="native-block-boundary"),
        pytest.param(
            np.nextafter(32 / 44_100, np.inf),
            33,
            id="just-after-native-block-boundary",
        ),
        pytest.param(
            np.nextafter(32 / 44_100, -np.inf),
            32,
            id="just-before-native-block-boundary",
        ),
    ],
)
def test_sample_index_at_or_after_float_boundary_quantizes_event_exactly(
    event_time: float,
    expected_sample: int,
) -> None:
    """Floating event boundaries do not gain early or spurious late samples.

    :param event_time: Requested event time in seconds.
    :param expected_sample: First sample at or after the event.
    """
    assert _sample_index_at_or_after(event_time, 44_100) == expected_sample


def test_align_native_attack_places_unmodified_source_after_silent_prefix() -> None:
    """Alignment shifts the native attack without altering retained samples."""
    audio = np.tile(np.arange(8, dtype=np.float32), (2, 1))

    aligned = _align_native_attack(
        audio,
        samples=6,
        start_sample=2,
        source_start=1,
    )

    assert aligned.shape == (2, 6)
    assert aligned.dtype == np.float32
    np.testing.assert_array_equal(aligned[:, :2], np.zeros((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(aligned[:, 2:], audio[:, 1:5])


def test_align_native_attack_undersized_source_raises() -> None:
    """Alignment rejects a native buffer that cannot fill the retained output."""
    audio = np.zeros((2, 31), dtype=np.float32)

    with pytest.raises(ValueError, match="attack buffer"):
        _align_native_attack(
            audio,
            samples=32,
            start_sample=0,
            source_start=16,
        )


@pytest.mark.requires_surgepy
@pytest.mark.parametrize(
    ("note_end", "expected_release_sample"),
    [
        pytest.param(48 / 44_100, 48, id="exact-sample-boundary"),
        pytest.param(
            np.nextafter(48 / 44_100, np.inf),
            80,
            id="just-after-sample-boundary",
        ),
    ],
)
def test_surgepy_renderer_note_end_preserves_quantized_duration(
    note_end: float,
    expected_release_sample: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shifted attack retains its native block-quantized note duration.

    :param note_end: Requested note end in seconds.
    :param expected_release_sample: First output sample rendered after note release.
    :param monkeypatch: Installs the controlled native block processor.
    """
    block_size = 32
    renderer, synth, state = _mock_renderer(
        monkeypatch,
        block_size=block_size,
        output_samples=96,
    )

    def process_blocks(output: np.ndarray, start_block: int, num_blocks: int) -> None:
        """Mark each processed block with the current note state.

        :param output: Stereo destination buffer.
        :param start_block: First destination block.
        :param num_blocks: Number of blocks to mark.
        """
        value = 1.0 if state["playing"] else -1.0
        start_sample = start_block * block_size
        output[:, start_sample : start_sample + num_blocks * block_size] = value

    synth.processMultiBlock.side_effect = process_blocks

    audio = renderer.render({}, 60, 100, (16 / 44_100, note_end))

    assert np.all(audio[:, 16:expected_release_sample] == 1.0)
    assert np.all(audio[:, expected_release_sample:] == -1.0)


@pytest.mark.requires_surgepy
def test_surgepy_renderer_event_after_retained_output_returns_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An event beyond a fractional retained sample count does not invoke Surge.

    :param monkeypatch: Installs the controlled native block processor.
    """
    renderer, synth, _ = _mock_renderer(
        monkeypatch,
        block_size=32,
        output_samples=4,
        sample_rate=10,
    )

    audio = renderer.render({}, 60, 100, (0.41, 0.44))

    np.testing.assert_array_equal(audio, np.zeros((2, 4), dtype=np.float32))
    assert synth.method_calls == []


@pytest.mark.requires_surgepy
def test_surgepy_renderer_non_block_aligned_note_preserves_native_attack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sub-block placement delays the attack without discarding native samples.

    :param monkeypatch: Installs the controlled native block processor.
    """
    block_size = 32
    renderer, synth, state = _mock_renderer(
        monkeypatch,
        block_size=block_size,
        output_samples=128,
    )

    def process_blocks(output: np.ndarray, start_block: int, num_blocks: int) -> None:
        """Emit increasing native attack samples while the note is active.

        :param output: Stereo destination buffer.
        :param start_block: First destination block.
        :param num_blocks: Number of blocks to process.
        """
        for block in range(num_blocks):
            if not state["playing"]:
                continue
            output_start = (start_block + block) * block_size
            attack_start = state["note_age"] + 1
            output[:, output_start : output_start + block_size] = np.arange(
                attack_start,
                attack_start + block_size,
                dtype=np.float32,
            )
            state["note_age"] += block_size

    synth.processMultiBlock.side_effect = process_blocks

    audio = renderer.render({}, 60, 100, (48 / 44_100, 80 / 44_100))

    expected_attack = np.tile(np.arange(1, 33, dtype=np.float32), (2, 1))
    np.testing.assert_array_equal(audio[:, 48:80], expected_attack)


@pytest.mark.slow
@pytest.mark.requires_surgepy
def test_surgepy_renderer_native_block_boundary_starts_at_requested_sample() -> None:
    """An exact native block boundary does not shift the note."""
    renderer = _simple_renderer()
    requested_start_sample = 320

    audio = renderer.render(
        {},
        60,
        100,
        (requested_start_sample / renderer.sample_rate, 640 / renderer.sample_rate),
    )

    audible_samples = np.flatnonzero(np.max(np.abs(audio), axis=0) > 1e-8)
    assert audible_samples[0] == requested_start_sample


@pytest.mark.slow
@pytest.mark.requires_surgepy
def test_surgepy_renderer_final_partial_block_starts_at_requested_sample() -> None:
    """A note in the retained tail of the final native block remains audible."""
    renderer = _simple_renderer()
    requested_start_sample = 4_401

    audio = renderer.render(
        {},
        60,
        100,
        (requested_start_sample / renderer.sample_rate, 4_409 / renderer.sample_rate),
    )

    audible_samples = np.flatnonzero(np.max(np.abs(audio), axis=0) > 1e-8)
    assert audible_samples[0] == requested_start_sample


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
        parameter_map=load_param_map(
            Path("src/synth_setter/data/vst/surge_xt_param_map.json")
        ),
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
