"""Behavior tests for the live pyFDN renderer effect."""

from __future__ import annotations

import numpy as np
import pytest

from synth_setter.data.vst.pyfdn_renderer import PyFDNEffectRenderer
from synth_setter.data.vst.renderers import AudioRenderer
from synth_setter.pipeline.schemas.spec import PyFDNEffectConfig

_SAMPLE_RATE = 48_000
_NUM_SAMPLES = 4_096


class _FixedAudioRenderer(AudioRenderer):
    """Return a fixed channel-first waveform through the renderer contract."""

    def __init__(self, audio: np.ndarray):
        """Store channel-first source audio.

        :param audio: Fixed waveform returned by every render.
        """
        super().__init__(
            plugin_path="fixed",
            sample_rate=_SAMPLE_RATE,
            channels=audio.shape[0],
            signal_duration_seconds=audio.shape[1] / _SAMPLE_RATE,
        )
        self.audio = audio

    def render(
        self,
        params: dict[str, float],
        midi_note: int,
        velocity: int,
        note_start_and_end: tuple[float, float],
        *,
        warmup: bool = False,
    ) -> np.ndarray:
        """Return a copy so effect processing cannot mutate the source waveform.

        :param params: Ignored fixed-renderer parameter mapping.
        :param midi_note: Ignored MIDI pitch.
        :param velocity: Ignored MIDI velocity.
        :param note_start_and_end: Ignored note interval.
        :param warmup: Ignored editor lifecycle flag.
        :returns: Copy of the fixed channel-first waveform.
        """
        return self.audio.copy()


def _effect_renderer(
    audio: np.ndarray,
    *,
    decay_seconds: float = 0.5,
    wet_mix: float = 1.0,
) -> PyFDNEffectRenderer:
    """Build the real pyFDN effect around deterministic source audio.

    :param audio: Fixed dry waveform supplied by the inner renderer.
    :param decay_seconds: Homogeneous decay time.
    :param wet_mix: Linear wet-signal proportion.
    :returns: Effect renderer configured with the bundled preset.
    """
    return PyFDNEffectRenderer(
        inner=_FixedAudioRenderer(audio),
        effect=PyFDNEffectConfig(
            package_version="0.4.2",
            preset_name="colorless_N8_d1",
            decay_seconds=decay_seconds,
            wet_mix=wet_mix,
        ),
    )


def _render(renderer: PyFDNEffectRenderer) -> np.ndarray:
    """Hold MIDI inputs fixed so tests isolate effect behavior.

    :param renderer: Effect session under test.
    :returns: Effected channel-first waveform.
    """
    return renderer.render({}, 60, 100, (0.0, 0.1))


def test_pyfdn_effect_package_version_mismatch_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Renderer construction rejects an installed pyFDN version outside provenance.

    :param monkeypatch: Replaces distribution metadata at the package boundary.
    """
    monkeypatch.setattr("synth_setter.data.vst.pyfdn_renderer.version", lambda _name: "9.9.9")
    audio = np.zeros((1, _NUM_SAMPLES), dtype=np.float32)

    with pytest.raises(RuntimeError, match="expected 0.4.2, found 9.9.9"):
        _effect_renderer(audio)


def test_pyfdn_effect_impulse_produces_delayed_wet_audio() -> None:
    """The real bundled FDN turns an impulse into a non-zero delayed response."""
    impulse = np.zeros((1, _NUM_SAMPLES), dtype=np.float32)
    impulse[0, 0] = 1.0

    effected = _render(_effect_renderer(impulse))

    assert effected.shape == impulse.shape
    assert effected.dtype == np.float32
    assert np.isfinite(effected).all()
    assert effected[0, 0] == 0.0
    assert np.max(np.abs(effected[0, 800:])) > 0.01


def test_pyfdn_effect_longer_decay_retains_more_late_tail_energy() -> None:
    """Decay seconds controls late impulse-response energy directionally."""
    impulse = np.zeros((1, _NUM_SAMPLES), dtype=np.float32)
    impulse[0, 0] = 1.0
    short_decay = _render(_effect_renderer(impulse, decay_seconds=0.05))
    long_decay = _render(_effect_renderer(impulse, decay_seconds=1.5))

    short_tail_energy = float(np.sum(np.square(short_decay[0, 2500:])))
    long_tail_energy = float(np.sum(np.square(long_decay[0, 2500:])))

    assert long_tail_energy > short_tail_energy


def test_pyfdn_effect_repeated_render_resets_feedback_state() -> None:
    """Rows are deterministic because every render starts with empty delay/filter state."""
    impulse = np.zeros((1, _NUM_SAMPLES), dtype=np.float32)
    impulse[0, 0] = 1.0
    renderer = _effect_renderer(impulse)

    first = _render(renderer)
    second = _render(renderer)

    assert np.array_equal(first, second)


def test_pyfdn_effect_processes_channels_independently() -> None:
    """A silent channel stays silent when another channel excites its own FDN."""
    stereo = np.zeros((2, _NUM_SAMPLES), dtype=np.float32)
    stereo[0, 0] = 1.0

    effected = _render(_effect_renderer(stereo))

    assert np.max(np.abs(effected[0])) > 0.01
    assert np.count_nonzero(effected[1]) == 0


def test_pyfdn_effect_preserves_overrange_for_pipeline_clipping_gate() -> None:
    """Over-range wet output stays visible so generation can reject the sampled row."""
    bounded_noise = np.random.default_rng(42).uniform(
        -1.0, 1.0, size=(1, _NUM_SAMPLES)
    ).astype(np.float32)

    effected = _render(_effect_renderer(bounded_noise, decay_seconds=1.5))

    assert np.max(np.abs(bounded_noise)) <= 1.0
    assert np.isfinite(effected).all()
    assert np.max(np.abs(effected)) > 1.0


def test_pyfdn_effect_wet_mix_blends_dry_and_wet_audio() -> None:
    """Wet mix scales both sides of the configured linear interpolation."""
    impulse = np.zeros((1, _NUM_SAMPLES), dtype=np.float32)
    impulse[0, 0] = 1.0
    full_wet = _render(_effect_renderer(impulse, wet_mix=1.0))

    mixed = _render(_effect_renderer(impulse, wet_mix=0.25))

    expected = 0.75 * impulse + 0.25 * full_wet
    assert np.array_equal(mixed, expected)
