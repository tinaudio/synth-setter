"""Behavioral tests for the log-mel front end and the frontend/backbone composition."""

import math
from collections.abc import Callable
from functools import partial
from typing import Any

import jaxtyping
import librosa
import numpy as np
import pytest
import torch

from synth_setter.models.components.cnn import MelCNN
from synth_setter.models.components.spec_encoder import (
    CepstrogramFrontend,
    LogMelFrontend,
    SpecEncoder,
)

_frontend = partial(LogMelFrontend, in_dim=4_410, sample_rate=44_100)
# 44.1k samples, 4096-point window, 11025 hop: five centered frames, quefrencies to 800.
_cepstrum = partial(CepstrogramFrontend, in_dim=44_100, n_fft=4_096, hop_length=11_025, q_max=800)


def _echo(delay: int, gain: float) -> torch.Tensor:
    """Build a unit impulse plus one scaled echo at the cepstrum fixture length.

    :param delay: Echo position in samples.
    :param gain: Echo amplitude relative to the unit impulse.
    :returns: One waveform shaped ``(1, 44_100)``.
    """
    audio = torch.zeros(1, 44_100)
    audio[0, 0] = 1.0
    audio[0, delay] = gain
    return audio


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Keep model initialization and synthetic waveforms deterministic."""
    torch.manual_seed(0)


def test_log_mel_frontend_returns_single_channel_spectrogram_grid() -> None:
    """The front end emits the channel axis spectrogram backbones consume."""
    features = _frontend()(torch.zeros(2, 4_410))

    assert features.shape == (2, 1, 128, 11)


def test_log_mel_frontend_matches_dataset_frontend() -> None:
    """All frames preserve the stored-mel frontend's numeric contract."""
    audio = torch.randn(1, 4_410)
    expected = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=audio[0].numpy(),
            sr=44_100,
            n_fft=1_102,
            hop_length=441,
            n_mels=128,
            window="hamming",
        ),
        ref=np.max,
    )

    actual = _frontend()(audio)[0, 0].detach().numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)


def test_log_mel_frontend_power_one_matches_amplitude_decibels() -> None:
    """Magnitude spectrograms use amplitude rather than power decibel scaling."""
    audio = torch.randn(1, 4_410)
    expected = librosa.amplitude_to_db(
        librosa.feature.melspectrogram(
            y=audio[0].numpy(),
            sr=44_100,
            n_fft=1_102,
            hop_length=441,
            n_mels=128,
            window="hamming",
            power=1.0,
        ),
        ref=np.max,
    )

    actual = _frontend(power=1.0)(audio)[0, 0].detach().numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)


def test_log_mel_frontend_htk_scale_matches_dataset_frontend() -> None:
    """The HTK mel scale matches the stored-feature reference."""
    audio = torch.randn(1, 4_410)
    expected = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=audio[0].numpy(),
            sr=44_100,
            n_fft=1_102,
            hop_length=441,
            n_mels=128,
            window="hamming",
            htk=True,
        ),
        ref=np.max,
    )

    actual = _frontend(mel_scale="htk")(audio)[0, 0].numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)


def test_log_mel_frontend_hann_window_matches_dataset_frontend() -> None:
    """The Hann window matches the stored-feature reference."""
    audio = torch.randn(1, 4_410)
    expected = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=audio[0].numpy(),
            sr=44_100,
            n_fft=1_102,
            hop_length=441,
            n_mels=128,
            window="hann",
        ),
        ref=np.max,
    )

    actual = _frontend(window="hann")(audio)[0, 0].numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)


def test_log_mel_frontend_top_db_clips_relative_dynamic_range() -> None:
    """The dynamic-range option clips values relative to each waveform peak."""
    audio = torch.randn(1, 4_410)
    unclipped = _frontend(top_db=None)(audio)
    clipped = _frontend(top_db=10.0)(audio)

    torch.testing.assert_close(clipped, torch.clamp(unclipped, min=-10.0))


def test_log_mel_frontend_sign_inversion_returns_same_spectrogram() -> None:
    """A pi phase shift leaves the magnitude-based features unchanged."""
    frontend = _frontend()
    audio = torch.randn(2, 4_410)

    torch.testing.assert_close(frontend(-audio), frontend(audio))


@pytest.mark.parametrize("audio", [torch.zeros(2, 1, 4_410), torch.zeros(2, 4_409)])
def test_log_mel_frontend_invalid_waveform_shape_raises(audio: torch.Tensor) -> None:
    """Malformed waveform batches fail at the front-end boundary.

    :param audio: Wrong-rank or wrong-length waveform batch.
    """
    with pytest.raises(ValueError, match="Expected waveform shape"):
        _frontend()(audio)


@pytest.mark.parametrize("amin", [0.0, -1.0, float("inf"), float("nan")])
def test_log_mel_frontend_invalid_amin_raises(amin: float) -> None:
    """A non-positive or non-finite logarithm floor is rejected.

    :param amin: Invalid power floor.
    """
    with pytest.raises(ValueError, match="amin"):
        _frontend(amin=amin)


@pytest.mark.parametrize("power", [0.0, -1.0, float("inf"), float("nan")])
def test_log_mel_frontend_invalid_power_raises(power: float) -> None:
    """A non-positive or non-finite magnitude exponent is rejected.

    :param power: Invalid spectrogram exponent.
    """
    with pytest.raises(ValueError, match="power"):
        _frontend(power=power)


@pytest.mark.parametrize("top_db", [-1.0, float("inf"), float("nan")])
def test_log_mel_frontend_invalid_top_db_raises(top_db: float) -> None:
    """A negative or non-finite dynamic range is rejected.

    :param top_db: Invalid dynamic range.
    """
    with pytest.raises(ValueError, match="top_db"):
        _frontend(top_db=top_db)


@pytest.mark.parametrize(
    ("factory", "frequency_name"),
    [
        pytest.param(partial(_frontend, f_min=-1.0), "f_min", id="negative-f-min"),
        pytest.param(partial(_frontend, f_min=float("inf")), "f_min", id="infinite-f-min"),
        pytest.param(partial(_frontend, f_min=float("nan")), "f_min", id="nan-f-min"),
        pytest.param(partial(_frontend, f_min=22_050.0), "f_min", id="f-min-at-nyquist"),
        pytest.param(partial(_frontend, f_max=0.0), "f_max", id="f-max-not-greater-than-f-min"),
        pytest.param(partial(_frontend, f_max=float("inf")), "f_max", id="infinite-f-max"),
        pytest.param(partial(_frontend, f_max=float("nan")), "f_max", id="nan-f-max"),
        pytest.param(partial(_frontend, f_max=22_051.0), "f_max", id="f-max-above-nyquist"),
    ],
)
def test_log_mel_frontend_invalid_frequency_bound_raises(
    factory: Callable[[], LogMelFrontend], frequency_name: str
) -> None:
    """Invalid mel-frequency bounds fail before producing non-finite features.

    :param factory: Front-end factory containing the invalid bound.
    :param frequency_name: Constructor argument receiving the invalid bound.
    """
    with pytest.raises(ValueError, match=frequency_name):
        factory()


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        pytest.param(partial(_frontend, hop_length=0), "hop_length", id="zero-hop-length"),
        pytest.param(partial(_frontend, n_fft=0), "n_fft", id="zero-n-fft"),
        pytest.param(partial(_frontend, n_mels=0), "n_mels", id="zero-n-mels"),
    ],
)
def test_log_mel_frontend_non_positive_geometry_raises(
    factory: Callable[[], LogMelFrontend], field: str
) -> None:
    """Non-positive frontend geometry fails at the configuration boundary.

    :param factory: Front-end factory containing the zero size.
    :param field: Constructor argument receiving the zero size.
    """
    with pytest.raises(ValueError, match=field):
        factory()


def test_log_mel_frontend_unknown_window_raises() -> None:
    """An unsupported Fourier window fails before transform construction."""
    with pytest.raises(jaxtyping.TypeCheckError):
        _frontend(window="blackman")  # type: ignore[arg-type]


def test_spec_encoder_with_cnn_backbone_returns_pooled_embedding() -> None:
    """The CNN backbone reduces the front end's grid to one vector per row."""
    encoder = SpecEncoder(
        frontend=_frontend(),
        backbone=MelCNN(hidden_dim=4, out_dim=5, num_blocks=1, kernel_size=3),
    )

    assert encoder(torch.zeros(2, 4_410)).shape == (2, 5)


def test_spec_encoder_distinct_spectra_return_distinct_embeddings() -> None:
    """The composed encoder responds to spectral content instead of returning a constant."""
    time = torch.arange(4_410) / 44_100
    audio = torch.stack(
        [torch.sin(2 * torch.pi * 220 * time), torch.sin(2 * torch.pi * 1_760 * time)]
    )
    encoder = SpecEncoder(
        frontend=_frontend(),
        backbone=MelCNN(hidden_dim=4, out_dim=5, num_blocks=1, kernel_size=3),
    )
    encoder.eval()

    with torch.no_grad():
        embeddings = encoder(audio)

    assert torch.isfinite(embeddings).all()
    assert not torch.allclose(embeddings[0], embeddings[1])


def test_spec_encoder_distinct_envelopes_return_distinct_embeddings() -> None:
    """The composed encoder preserves temporal-envelope information for one carrier."""
    time = torch.arange(4_410) / 44_100
    carrier = torch.sin(2 * torch.pi * 440 * time)
    audio = torch.stack(
        [carrier * torch.linspace(0, 1, 4_410), carrier * torch.linspace(1, 0, 4_410)]
    )
    encoder = SpecEncoder(
        frontend=_frontend(),
        backbone=MelCNN(hidden_dim=4, out_dim=5, num_blocks=1, kernel_size=3),
    )
    encoder.eval()

    with torch.no_grad():
        embeddings = encoder(audio)

    assert torch.isfinite(embeddings).all()
    assert not torch.allclose(embeddings[0], embeddings[1])


def test_spec_encoder_backward_reaches_the_backbone_and_the_waveform() -> None:
    """Gradients survive the front end, so waveform-side terms stay trainable."""
    encoder = SpecEncoder(
        frontend=_frontend(),
        backbone=MelCNN(hidden_dim=4, out_dim=5, num_blocks=1, kernel_size=3),
    )
    audio = torch.randn(2, 4_410, requires_grad=True)

    encoder(audio).square().mean().backward()

    assert audio.grad is not None
    assert torch.count_nonzero(audio.grad)
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.count_nonzero(parameter.grad), name


def test_cepstrogram_frontend_returns_quefrency_grid() -> None:
    """The front end emits ``(batch, 1, q_max, frames)`` for the backbone."""
    features = _cepstrum()(torch.randn(2, 44_100))

    assert features.shape == (2, 1, 800, 5)


def test_cepstrogram_frontend_echo_peaks_at_delay_row() -> None:
    """One echo at delay ``d`` is a cepstral spike at quefrency row ``d``."""
    features = _cepstrum()(_echo(300, 0.6))

    first_frame = features[0, 0, :, 0]
    assert int(torch.argmax(first_frame[50:])) + 50 == 300


def test_cepstrogram_frontend_rows_above_q_max_are_dropped() -> None:
    """An echo beyond the quefrency window leaves no spike inside it."""
    frontend = _cepstrum(q_max=250)

    first_frame = frontend(_echo(300, 0.6))[0, 0, :, 0]

    assert first_frame.shape == (250,)
    assert first_frame[50:].abs().max() < 1.0


def test_cepstrogram_frontend_gain_invariant() -> None:
    """A positive waveform gain lands in the discarded absolute level only."""
    frontend = _cepstrum()
    audio = torch.randn(2, 44_100)

    torch.testing.assert_close(frontend(3.0 * audio), frontend(audio))


def test_cepstrogram_frontend_bin_zero_drop_tracks_decay_rate() -> None:
    """Doubling an exponential decay doubles the frame-to-frame fall of quefrency 0."""
    frontend = _cepstrum()
    noise = torch.randn(1, 44_100)
    ramp = torch.arange(44_100, dtype=torch.float32)
    slow = frontend(noise * torch.exp(-ramp / 20_000.0))[0, 0, 0]
    fast = frontend(noise * torch.exp(-ramp / 10_000.0))[0, 0, 0]

    slow_drop = float(slow[1] - slow[3])
    fast_drop = float(fast[1] - fast[3])

    assert slow_drop > 0
    assert fast_drop / slow_drop == pytest.approx(2.0, rel=0.1)


def test_cepstrogram_frontend_output_is_zero_mean_per_clip() -> None:
    """Clip-mean subtraction removes the level offset without a per-clip rescale."""
    features = _cepstrum()(torch.randn(3, 44_100))

    torch.testing.assert_close(features.mean(dim=(1, 2, 3)), torch.zeros(3), atol=1e-4, rtol=0)


def test_cepstrogram_frontend_scale_multiplies_output() -> None:
    """``scale`` is a fixed dataset-constant multiplier on the normalized grid."""
    audio = torch.randn(1, 44_100)

    torch.testing.assert_close(_cepstrum(scale=0.25)(audio), 0.25 * _cepstrum()(audio))


@pytest.mark.parametrize("audio", [torch.zeros(2, 1, 44_100), torch.zeros(2, 44_099)])
def test_cepstrogram_frontend_invalid_waveform_shape_raises(audio: torch.Tensor) -> None:
    """Malformed waveform batches fail at the front-end boundary.

    :param audio: Wrong-rank or wrong-length waveform batch.
    """
    with pytest.raises(ValueError, match="Expected waveform shape"):
        _cepstrum()(audio)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"q_max": 0},
        {"q_max": 4_097},
        {"hop_length": 0},
        {"n_fft": 0},
        {"floor_db": -1.0},
        {"floor_db": math.inf},
        {"scale": 0.0},
    ],
)
def test_cepstrogram_frontend_invalid_geometry_raises(kwargs: dict[str, Any]) -> None:
    """Out-of-range quefrency, framing, floor, or scale settings are rejected.

    :param kwargs: One invalid constructor argument.
    """
    with pytest.raises(ValueError):
        _cepstrum(**kwargs)


def test_spec_encoder_cepstrum_backward_reaches_the_waveform() -> None:
    """Gradients survive the cepstral front end's log and inverse transform."""
    encoder = SpecEncoder(
        frontend=_cepstrum(),
        backbone=MelCNN(hidden_dim=4, out_dim=5, num_blocks=1, kernel_size=3),
    )
    audio = torch.randn(2, 44_100, requires_grad=True)

    encoder(audio).square().mean().backward()

    assert audio.grad is not None
    assert torch.count_nonzero(audio.grad)
