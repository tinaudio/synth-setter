"""Unit tests for the shape primitives in ``synth_setter.data.vst.shapes``.

The writer and the (planned) shard-validator inner-shape checks share these
helpers, so each test pins one shape against the legacy inline calculation
that lived in ``create_datasets_and_get_start_idx`` / ``make_spectrogram`` on
``main`` before this PR. If anyone changes a constant (n_mels, fps, ...),
multiple tests here should fail loudly rather than the writer and validator
silently drifting apart.
"""

import numpy as np

from synth_setter.data.vst.generate_vst_dataset import make_spectrogram
from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    DATASET_FIELD_NAMES,
    MEL_FRAMES_PER_SECOND,
    MEL_N_FFT_FRACTION_OF_SAMPLE_RATE,
    MEL_N_MELS,
    MEL_SPEC_FIELD,
    MEL_WINDOW,
    PARAM_ARRAY_FIELD,
    audio_dataset_shape,
    dataset_field_shapes,
    mel_dataset_shape,
    mel_hop_length,
    mel_n_fft,
    mel_n_frames,
    param_array_dataset_shape,
)
from synth_setter.pipeline.schemas.spec import RenderConfig


def test_dataset_field_names_match_writer_emissions() -> None:
    """Pins the public field-name tuple; adding a field forces writer + validator update."""
    assert DATASET_FIELD_NAMES == ("audio", "mel_spec", "param_array")


def test_dataset_field_constants_match_tuple_order() -> None:
    """``DATASET_FIELD_NAMES`` is built from the per-field constants the writer uses."""
    assert DATASET_FIELD_NAMES == (AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD)
    assert (AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD) == (
        "audio",
        "mel_spec",
        "param_array",
    )


def test_mel_front_end_constants_match_legacy_values() -> None:
    """Pin the four module-level mel-front-end constants against legacy literals."""
    assert MEL_FRAMES_PER_SECOND == 100
    assert MEL_N_MELS == 128
    assert MEL_N_FFT_FRACTION_OF_SAMPLE_RATE == 0.025
    assert MEL_WINDOW == "hamming"


def test_mel_hop_length_matches_legacy_inline_calc() -> None:
    """``mel_hop_length(16000)`` equals the legacy ``int(sr / 100.0)`` literal (160)."""
    assert mel_hop_length(16000) == 160
    assert mel_hop_length(44100) == 441


def test_mel_hop_length_raises_when_hop_would_be_zero() -> None:
    """Reject sample rates below MEL_FRAMES_PER_SECOND so n_frames never divides by zero."""
    import pytest

    with pytest.raises(ValueError, match="hop length 0"):
        mel_hop_length(50)


def test_mel_n_fft_matches_legacy_inline_calc() -> None:
    """``mel_n_fft(16000)`` equals the legacy ``int(0.025 * sr)`` literal (400)."""
    assert mel_n_fft(16000) == 400
    assert mel_n_fft(44100) == 1102


def test_mel_n_fft_raises_when_n_fft_would_be_zero() -> None:
    """Reject sample rates where ``int(0.025 * sr)`` rounds to 0 (not a valid librosa n_fft)."""
    import pytest

    with pytest.raises(ValueError, match="n_fft 0"):
        mel_n_fft(10)


def test_mel_n_frames_matches_legacy_inline_calc() -> None:
    """``mel_n_frames`` reproduces librosa's center=True frame count: ``1 + len // hop``."""
    # 16k * 4s = 64000 samples, hop = 160 -> 1 + 64000 // 160 = 401.
    assert mel_n_frames(16000, 4.0) == 401
    # 44.1k * 4s = 176400 samples, hop = 441 -> 1 + 176400 // 441 = 401.
    assert mel_n_frames(44100, 4.0) == 401


def test_audio_dataset_shape_matches_legacy_inline_calc() -> None:
    """Pins ``(num_samples, channels, int(sample_rate * signal_duration_seconds))``."""
    assert audio_dataset_shape(2, 2, 16000, 4.0) == (2, 2, 64000)
    assert audio_dataset_shape(5, 2, 44100, 4.0) == (5, 2, 176400)


def test_mel_dataset_shape_matches_legacy_inline_calc() -> None:
    """Pins ``(num_samples, channels, 128, n_frames)`` for 16k x 4s and 44.1k x 4s."""
    assert mel_dataset_shape(2, 2, 16000, 4.0) == (2, 2, 128, 401)
    assert mel_dataset_shape(5, 2, 44100, 4.0) == (5, 2, 128, 401)


def test_param_array_dataset_shape_matches_legacy_inline_calc() -> None:
    """Pins ``(num_samples, num_params)``."""
    assert param_array_dataset_shape(2, 175) == (2, 175)
    assert param_array_dataset_shape(0, 0) == (0, 0)


def test_dataset_field_shapes_maps_every_field_to_full_writer_shape() -> None:
    """``dataset_field_shapes`` returns the three per-field shapes the writers emit."""
    render = RenderConfig(
        plugin_path="/fake/Plugin.vst3",
        preset_path="presets/fake.vstpreset",
        param_spec_name="surge_simple",
        renderer_version="1.0.0-test",
        sample_rate=44100,
        channels=2,
        velocity=100,
        signal_duration_seconds=4.0,
        min_loudness=-55.0,
        samples_per_render_batch=4,
        samples_per_shard=4,
        gui_toggle_cadence="never",
    )

    assert dataset_field_shapes(render, num_params=7) == {
        AUDIO_FIELD: (4, 2, 176400),
        MEL_SPEC_FIELD: (4, 2, 128, 401),
        PARAM_ARRAY_FIELD: (4, 7),
    }


def test_make_spectrogram_output_shape_matches_mel_dataset_shape_helper() -> None:
    """make_spectrogram and mel_dataset_shape agree on the trailing (n_mels, n_frames).

    This is the load-bearing invariant the writer and validator both rely on:
    the actual librosa call inside ``make_spectrogram`` and the precomputed
    ``mel_dataset_shape`` must produce the same trailing dimensions. Tests a
    1-channel and a 2-channel render — librosa's behavior is per-channel so
    the helper pads the channel dimension symmetrically.
    """
    sample_rate = 16000
    duration = 4.0
    audio_length = int(sample_rate * duration)

    mono = np.zeros((audio_length,), dtype=np.float32)
    mono_spec = make_spectrogram(mono, sample_rate)
    assert mono_spec.shape == mel_dataset_shape(1, 1, sample_rate, duration)[2:]

    stereo = np.zeros((2, audio_length), dtype=np.float32)
    stereo_spec = make_spectrogram(stereo, sample_rate)
    assert stereo_spec.shape == (2, *mel_dataset_shape(1, 1, sample_rate, duration)[2:])
