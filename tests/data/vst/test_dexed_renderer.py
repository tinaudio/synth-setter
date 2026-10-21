"""Real native-engine rendering, with no VST host or mocked synthesizer."""

import numpy as np
import pytest
from dexed import DexedSynth, Preset

from synth_setter.data.vst.dexed_renderer import DexedRenderer
from synth_setter.data.vst.param_spec import ParameterValue


@pytest.fixture
def renderer() -> DexedRenderer:
    """Use a short mono render with an audible initialized patch.

    :returns: A real native renderer with a half-second output window.
    """
    return DexedRenderer(
        plugin_path="dexed",
        sample_rate=44100,
        channels=1,
        signal_duration_seconds=0.5,
    )


def test_default_patch_matches_native_engine_audio(renderer: DexedRenderer) -> None:
    """The adapter preserves the native waveform and gain.

    :param renderer: Native mono renderer fixture.
    """
    native = DexedSynth(sample_rate=44100)
    native.load_preset(Preset())
    expected = native.render(midi_note=60, velocity=100, note_duration=0.25, render_duration=0.5)
    audio = renderer.render({}, 60, 100, (0.0, 0.25))
    assert audio.shape == (1, 22050)
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.01
    np.testing.assert_array_equal(audio[0], expected)


def test_delayed_note_has_silent_prefix(renderer: DexedRenderer) -> None:
    """MIDI onset is placed after the requested silence.

    :param renderer: Native mono renderer fixture.
    """
    audio = renderer.render({}, 60, 100, (0.1, 0.25))
    assert np.count_nonzero(audio[:, :4410]) == 0
    assert np.max(np.abs(audio[:, 4410:])) > 0.01


def test_render_recreates_patch_and_voice_state(renderer: DexedRenderer) -> None:
    """Prior notes and parameter writes cannot contaminate another row.

    :param renderer: Native mono renderer fixture.
    """
    before = renderer.render({}, 60, 100, (0.0, 0.25))
    renderer.render({"algorithm": 31, "feedback": 1.0}, 72, 127, (0.0, 0.5))
    after = renderer.render({}, 60, 100, (0.0, 0.25))
    np.testing.assert_array_equal(before, after)


def test_parameter_change_affects_audio(renderer: DexedRenderer) -> None:
    """Native algorithm selection changes the rendered waveform.

    :param renderer: Native mono renderer fixture.
    """
    first = renderer.render({"algorithm": 0}, 60, 100, (0.0, 0.25))
    second = renderer.render({"algorithm": 31}, 60, 100, (0.0, 0.25))
    assert not np.array_equal(first, second)


@pytest.mark.parametrize(
    "params",
    [
        {"unknown": 0.5},
        {"algorithm": 32},
        {"algorithm": 0.5},
        {"feedback": float("nan")},
        {"feedback": -0.1},
        {"feedback": 1.1},
        {"feedback": True},
        {"feedback": np.array([0.5])},
    ],
)
def test_invalid_parameters_are_rejected(
    renderer: DexedRenderer, params: dict[str, ParameterValue]
) -> None:
    """Malformed patch values never reach the native extension.

    :param renderer: Native mono renderer fixture.
    :param params: Invalid coordinate overrides.
    """
    with pytest.raises(ValueError, match=next(iter(params))):
        renderer.render(params, 60, 100, (0.0, 0.25))


@pytest.mark.parametrize("timing", [(-0.1, 0.2), (0.3, 0.2), (0.0, 0.6), (0.0, float("nan"))])
def test_invalid_note_timing_is_rejected(
    renderer: DexedRenderer, timing: tuple[float, float]
) -> None:
    """Note endpoints must fit the output window.

    :param renderer: Native mono renderer fixture.
    :param timing: Invalid note endpoints in seconds.
    """
    with pytest.raises(ValueError):
        renderer.render({}, 60, 100, timing)


def test_stereo_output_duplicates_native_mono() -> None:
    """Stereo requests preserve the native mono waveform in both channels."""
    renderer = DexedRenderer(
        plugin_path="dexed",
        sample_rate=44100,
        channels=2,
        signal_duration_seconds=0.5,
    )
    audio = renderer.render({}, 60, 100, (0.0, 0.25))
    assert audio.shape == (2, 22050)
    np.testing.assert_array_equal(audio[0], audio[1])


@pytest.mark.parametrize("note,velocity", [(-1, 100), (128, 100), (60, -1), (60, 128)])
def test_invalid_midi_is_rejected(renderer: DexedRenderer, note: int, velocity: int) -> None:
    """Native MIDI input remains in the byte-sized musical domain.

    :param renderer: Native mono renderer fixture.
    :param note: Candidate MIDI pitch.
    :param velocity: Candidate MIDI velocity.
    """
    with pytest.raises(ValueError, match="integer"):
        renderer.render({}, note, velocity, (0.0, 0.25))


@pytest.mark.parametrize("timing", [(0.0, 0.0), (0.5, 0.5)])
def test_empty_note_is_silent(renderer: DexedRenderer, timing: tuple[float, float]) -> None:
    """Empty notes do not create an unintended native attack.

    :param renderer: Native mono renderer fixture.
    :param timing: Coincident note endpoints in seconds.
    """
    assert np.count_nonzero(renderer.render({}, 60, 100, timing)) == 0


def test_zero_velocity_is_silent(renderer: DexedRenderer) -> None:
    """MIDI note-on velocity zero is not an audible note.

    :param renderer: Native mono renderer fixture.
    """
    assert np.count_nonzero(renderer.render({}, 60, 0, (0.0, 0.25))) == 0


def test_editor_warmup_is_rejected(renderer: DexedRenderer) -> None:
    """Headless native rendering cannot honor an editor request.

    :param renderer: Native mono renderer fixture.
    """
    with pytest.raises(ValueError, match="editor"):
        renderer.render({}, 60, 100, (0.0, 0.25), warmup=True)


def test_mismatched_version_is_rejected() -> None:
    """Stored synth identity must match the pinned native implementation."""
    with pytest.raises(ValueError, match="configured=0.0.0"):
        DexedRenderer(
            plugin_path="dexed", sample_rate=44100, channels=1,
            signal_duration_seconds=0.5, synth_version="0.0.0",
        )
