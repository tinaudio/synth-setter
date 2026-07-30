"""Behavioral tests for the log-mel front end and the frontend/backbone composition."""

from collections.abc import Callable
from functools import partial

import jaxtyping
import librosa
import numpy as np
import pytest
import torch

from synth_setter.models.components.cnn import MelCNN
from synth_setter.models.components.spec_encoder import LogMelFrontend, SpecEncoder

_frontend = partial(LogMelFrontend, in_dim=4_410, sample_rate=44_100)


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
