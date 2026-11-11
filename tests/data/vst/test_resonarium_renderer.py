"""Real native Resonarium rendering, with no VST host or mocked synthesizer."""

import numpy as np
import pytest

from synth_setter.data.vst.param_spec import ParameterValue
from synth_setter.data.vst.resonarium_renderer import ResonariumRenderer

pytestmark = pytest.mark.requires_resonarium

# Impulse exciter into one bright string: audible from the first frame after note-on.
_PLUCK: dict[str, ParameterValue] = {
    "impExciter0 enabled": 1.0,
    "enabled wb0r0": 1.0,
    "decayFilterCutoff wb0r0": 0.9,
}


@pytest.fixture
def renderer() -> ResonariumRenderer:
    """Use a short stereo render.

    :returns: A real native renderer with a half-second output window.
    """
    return ResonariumRenderer(
        plugin_path="resonarium", sample_rate=44100, channels=2, signal_duration_seconds=0.5
    )


def test_pluck_renders_finite_audible_float32_stereo(renderer: ResonariumRenderer) -> None:
    """An excited string produces finite audio of the configured geometry.

    :param renderer: Native stereo renderer fixture.
    """
    audio = renderer.render(_PLUCK, 60, 100, (0.0, 0.25))
    assert audio.shape == (2, 22050)
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.01


def test_default_patch_is_silent(renderer: ResonariumRenderer) -> None:
    """Upstream defaults disable every exciter and resonator, so an empty patch is silent.

    :param renderer: Native stereo renderer fixture.
    """
    assert np.count_nonzero(renderer.render({}, 60, 100, (0.0, 0.25))) == 0


def test_render_is_repeatable_across_interleaved_patches(renderer: ResonariumRenderer) -> None:
    """A-B-A: a different patch in between cannot change a repeated render.

    :param renderer: Native stereo renderer fixture.
    """
    noisy = {**_PLUCK, "noiseExciter0enabled": 1.0, "enabled wb1r3": 1.0}
    first = renderer.render(noisy, 60, 100, (0.0, 0.25))
    renderer.render({**_PLUCK, "reverbEnable": 1.0, "dispersion wb0r0": 0.7}, 72, 127, (0.0, 0.5))
    again = renderer.render(noisy, 60, 100, (0.0, 0.25))
    np.testing.assert_array_equal(first, again)


@pytest.mark.parametrize("onset_frames", [13, 4410])
def test_delayed_note_is_exact_frame_shift(
    renderer: ResonariumRenderer, onset_frames: int
) -> None:
    """A later onset shifts the render by exactly its frame offset, across block boundaries.

    :param renderer: Native stereo renderer fixture.
    :param onset_frames: Note-on frame; 13 lands mid-block, 4410 in a later block.
    """
    start = onset_frames / 44100
    immediate = renderer.render(_PLUCK, 60, 100, (0.0, 0.25))
    delayed = renderer.render(_PLUCK, 60, 100, (start, start + 0.25))
    assert np.count_nonzero(delayed[:, :onset_frames]) == 0
    np.testing.assert_array_equal(delayed[:, onset_frames:], immediate[:, : 22050 - onset_frames])


def test_parameter_change_affects_audio(renderer: ResonariumRenderer) -> None:
    """A mapped control reaches the engine.

    :param renderer: Native stereo renderer fixture.
    """
    first = renderer.render(_PLUCK, 60, 100, (0.0, 0.25))
    second = renderer.render({**_PLUCK, "decayFilterCutoff wb0r0": 0.2}, 60, 100, (0.0, 0.25))
    assert not np.array_equal(first, second)


def test_mono_output_is_mean_of_native_stereo() -> None:
    """One channel downmixes the native stereo pair."""
    stereo = ResonariumRenderer(
        plugin_path="resonarium", sample_rate=44100, channels=2, signal_duration_seconds=0.5
    ).render(_PLUCK, 60, 100, (0.0, 0.25))
    mono = ResonariumRenderer(
        plugin_path="resonarium", sample_rate=44100, channels=1, signal_duration_seconds=0.5
    ).render(_PLUCK, 60, 100, (0.0, 0.25))
    np.testing.assert_allclose(mono[0], stereo.mean(axis=0), atol=1e-7)


@pytest.mark.parametrize(
    "params",
    [
        {"unknown": 0.5},
        {"masterGain": 0.5},
        {"decayTime wb0r0": float("nan")},
        {"decayTime wb0r0": -0.1},
        {"decayTime wb0r0": 1.1},
        {"decayTime wb0r0": np.array([0.5])},
    ],
)
def test_invalid_parameters_are_rejected(
    renderer: ResonariumRenderer, params: dict[str, ParameterValue]
) -> None:
    """Unknown names and malformed normalized values never reach the native extension.

    :param renderer: Native stereo renderer fixture.
    :param params: Invalid coordinate overrides.
    """
    with pytest.raises((KeyError, ValueError, TypeError)):
        renderer.render(params, 60, 100, (0.0, 0.25))


@pytest.mark.parametrize("timing", [(-0.1, 0.2), (0.3, 0.2), (0.0, 0.6), (0.0, float("nan"))])
def test_invalid_note_timing_is_rejected(
    renderer: ResonariumRenderer, timing: tuple[float, float]
) -> None:
    """Note endpoints must be ordered and fit the output window.

    :param renderer: Native stereo renderer fixture.
    :param timing: Invalid note endpoints in seconds.
    """
    with pytest.raises(ValueError, match="note"):
        renderer.render({}, 60, 100, timing)


@pytest.mark.parametrize("note,velocity", [(-1, 100), (128, 100), (60, -1), (60, 128)])
def test_invalid_midi_is_rejected(renderer: ResonariumRenderer, note: int, velocity: int) -> None:
    """MIDI input stays in the byte-sized musical domain.

    :param renderer: Native stereo renderer fixture.
    :param note: Candidate MIDI pitch.
    :param velocity: Candidate MIDI velocity.
    """
    with pytest.raises(ValueError, match="MIDI"):
        renderer.render({}, note, velocity, (0.0, 0.25))


@pytest.mark.parametrize("timing", [(0.2, 0.2), (0.2, 0.2 + 1e-6)])
def test_sub_frame_note_gates_for_one_sample(
    renderer: ResonariumRenderer, timing: tuple[float, float]
) -> None:
    """A window inside one frame still plucks: one-sample gate, never an error.

    :param renderer: Native stereo renderer fixture.
    :param timing: Coincident or sub-frame note endpoints in seconds.
    """
    audio = renderer.render(_PLUCK, 60, 100, timing)
    assert np.count_nonzero(audio[:, :8820]) == 0
    assert np.max(np.abs(audio)) > 0.01


def test_note_on_past_last_frame_is_silent(renderer: ResonariumRenderer) -> None:
    """A note starting after the final frame renders silence rather than raising.

    :param renderer: Native stereo renderer fixture.
    """
    assert np.count_nonzero(renderer.render(_PLUCK, 60, 100, (0.5, 0.5))) == 0


def test_zero_velocity_is_silent(renderer: ResonariumRenderer) -> None:
    """MIDI note-on velocity zero is not an audible note.

    :param renderer: Native stereo renderer fixture.
    """
    assert np.count_nonzero(renderer.render(_PLUCK, 60, 0, (0.0, 0.25))) == 0


def test_editor_warmup_is_rejected(renderer: ResonariumRenderer) -> None:
    """Headless native rendering cannot honor an editor request.

    :param renderer: Native stereo renderer fixture.
    """
    with pytest.raises(ValueError, match="editor"):
        renderer.render({}, 60, 100, (0.0, 0.25), warmup=True)


def test_mismatched_version_is_rejected() -> None:
    """Stored synth identity must match the pinned native build."""
    with pytest.raises(ValueError, match="configured=0.0.0"):
        ResonariumRenderer(
            plugin_path="resonarium",
            sample_rate=44100,
            channels=2,
            signal_duration_seconds=0.5,
            synth_version="0.0.0",
        )


def test_effects_start_from_host_steady_state(renderer: ResonariumRenderer) -> None:
    """Effects behave as in a running host: patch applied and processed once, then reset.

    A freshly enabled phaser otherwise ramps its controls from defaults at onset, which the
    Resonarium VST3 never does after a host loads state (verified bit-exact against the plugin).

    :param renderer: Native stereo renderer fixture.
    """
    import resonarium

    from synth_setter.data.vst.resonarium_param_spec import RESONARIUM_CONTROLS

    patch = {**_PLUCK, "phaserEnable": 1.0, "phaserRate": 0.7, "phaserFeedback": 0.95}
    controls = {control.uid: control for control in RESONARIUM_CONTROLS}
    host = resonarium.Resonarium(44100.0, 512)
    for uid, value in patch.items():
        host.set_param(uid, controls[uid].native_value(value))
    host.process_multi_block(host.create_multi_block(1))
    host.reset()
    expected = host.create_multi_block(44)
    host.play_note(0, 60, 100, 0)
    host.process_multi_block(expected)

    audio = renderer.render(patch, 60, 100, (0.0, 0.5))

    np.testing.assert_array_equal(audio, expected[:, :22050])
