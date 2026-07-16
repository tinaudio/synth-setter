"""Behavior contracts for the torchsynth ``AudioRenderer`` backend."""

from __future__ import annotations

import numpy as np
import pytest

from synth_setter.data.vst.generate_vst_dataset import generate_sample
from synth_setter.data.vst.renderers import TorchSynthRenderer
from synth_setter.data.vst.torchsynth_param_spec import TORCHSYNTH_ADSR_PARAM_SPEC

_SAMPLE_RATE = 22_050
_DURATION_SECONDS = 1.0
_SAMPLES = int(_SAMPLE_RATE * _DURATION_SECONDS)


def _make_renderer(channels: int = 1) -> TorchSynthRenderer:
    """Build a small, fast renderer for the tests.

    :param channels: Output channel count under test.
    :returns: Renderer bound to a one-second mono/stereo geometry.
    """
    return TorchSynthRenderer(
        plugin_path="torchsynth",
        sample_rate=_SAMPLE_RATE,
        channels=channels,
        signal_duration_seconds=_DURATION_SECONDS,
    )


_ADSR_PATCH = {
    "adsr_1.attack": 0.1,
    "adsr_1.decay": 0.3,
    "adsr_1.sustain": 0.8,
    "adsr_1.release": 0.2,
    "vco_2.shape": 0.5,
}


def test_render_returns_finite_normalized_audio_with_requested_shape() -> None:
    """A real render satisfies the shared backend output contract."""
    audio = _make_renderer().render(_ADSR_PATCH, 60, 100, (0.0, 0.5))

    assert audio.shape == (1, _SAMPLES)
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()
    assert np.abs(audio).max() <= 1.0


def test_render_same_inputs_is_deterministic() -> None:
    """Two renders of the same patch and note produce identical audio."""
    renderer = _make_renderer()
    first = renderer.render(_ADSR_PATCH, 60, 100, (0.0, 0.5))
    second = renderer.render(_ADSR_PATCH, 60, 100, (0.0, 0.5))

    assert np.array_equal(first, second)


def test_render_baseline_patch_is_audible() -> None:
    """An empty override dict renders the baseline patch, which must not be silent."""
    audio = _make_renderer().render({}, 60, 100, (0.0, 0.9))

    assert np.abs(audio).max() > 0.05


def test_render_responds_to_sampled_params() -> None:
    """Changing a sampled knob changes the audio (params actually reach the voice)."""
    renderer = _make_renderer()
    slow_attack = dict(_ADSR_PATCH, **{"adsr_1.attack": 0.9})
    fast_attack = dict(_ADSR_PATCH, **{"adsr_1.attack": 0.0})

    assert not np.array_equal(
        renderer.render(slow_attack, 60, 100, (0.0, 0.9)),
        renderer.render(fast_attack, 60, 100, (0.0, 0.9)),
    )


def test_render_responds_to_midi_pitch() -> None:
    """Different MIDI notes produce different audio."""
    renderer = _make_renderer()

    assert not np.array_equal(
        renderer.render(_ADSR_PATCH, 48, 100, (0.0, 0.5)),
        renderer.render(_ADSR_PATCH, 72, 100, (0.0, 0.5)),
    )


def test_render_note_start_shifts_audio_by_the_start_offset() -> None:
    """The note-on offset delays the rendered note and zero-fills the head."""
    renderer = _make_renderer()
    at_zero = renderer.render(_ADSR_PATCH, 60, 100, (0.0, 0.4))
    offset_samples = int(0.25 * _SAMPLE_RATE)
    delayed = renderer.render(_ADSR_PATCH, 60, 100, (0.25, 0.65))

    assert np.array_equal(delayed[:, :offset_samples], np.zeros((1, offset_samples)))
    assert np.array_equal(delayed[:, offset_samples:], at_zero[:, : _SAMPLES - offset_samples])


def test_render_stereo_duplicates_the_mono_voice() -> None:
    """Channels=2 repeats torchsynth's mono output on both channels."""
    audio = _make_renderer(channels=2).render(_ADSR_PATCH, 60, 100, (0.0, 0.5))

    assert audio.shape == (2, _SAMPLES)
    assert np.array_equal(audio[0], audio[1])


def test_render_unknown_param_key_raises_before_rendering() -> None:
    """Keys outside the pinned voice spec fail loudly instead of being dropped."""
    with pytest.raises(KeyError, match="a_filter_1_cutoff"):
        _make_renderer().render({"a_filter_1_cutoff": 0.5}, 60, 100, (0.0, 0.5))


def test_render_zero_length_note_clamps_to_the_voice_minimum_duration() -> None:
    """A degenerate note window still renders instead of tripping torchsynth asserts."""
    audio = _make_renderer().render(_ADSR_PATCH, 60, 100, (0.5, 0.5))

    assert audio.shape == (1, _SAMPLES)
    assert np.isfinite(audio).all()


def test_generate_sample_renders_a_torchsynth_dataset_row() -> None:
    """The dataset sampling loop works end-to-end against the torchsynth backend."""
    sample = generate_sample(
        renderer=_make_renderer(channels=2),
        velocity=100,
        min_loudness=-70.0,
        param_spec=TORCHSYNTH_ADSR_PARAM_SPEC,
        fixed_synth_params=_ADSR_PATCH,
        fixed_note_params={"pitch": 60, "note_start_and_end": (0.0, 0.8)},
    )

    assert sample.audio.shape == (_SAMPLES, 2)
    assert sample.param_array.shape == (len(TORCHSYNTH_ADSR_PARAM_SPEC),)
    assert np.isfinite(sample.param_array).all()
