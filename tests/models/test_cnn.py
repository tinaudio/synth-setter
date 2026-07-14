"""Behavioral tests for waveform encoders."""

import librosa
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from synth_setter.models.components.cnn import LogMelEncoder
from synth_setter.models.components.residual_mlp import CNNResidualMLP


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Keep model initialization and synthetic waveforms deterministic."""
    torch.manual_seed(0)


def test_log_mel_frontend_four_second_audio_returns_bounded_embedding() -> None:
    """Four-second audio produces predictions without a length-sized linear head."""
    model = CNNResidualMLP(
        in_dim=176_400,
        channels=4,
        encoder_blocks=2,
        trunk_blocks=1,
        hidden_dim=16,
        out_dim=3,
        kernel_size=3,
        norm="bn",
        frontend="log_mel",
        sample_rate=44_100,
    )
    model.eval()

    with torch.no_grad():
        prediction = model(torch.zeros(2, 176_400))

    assert prediction.shape == (2, 3)
    assert sum(parameter.numel() for parameter in model.parameters()) < 100_000


def test_log_mel_frontend_sign_inversion_returns_same_embedding() -> None:
    """A pi phase shift leaves the magnitude-based embedding unchanged."""
    encoder = LogMelEncoder(
        in_dim=4_410,
        hidden_dim=4,
        out_dim=5,
        sample_rate=44_100,
        num_blocks=1,
        kernel_size=3,
    )
    encoder.eval()
    audio = torch.randn(2, 4_410)

    with torch.no_grad():
        original = encoder(audio)
        phase_shifted = encoder(-audio)

    torch.testing.assert_close(phase_shifted, original)


def test_log_mel_frontend_short_smoke_audio_returns_embedding() -> None:
    """A 0.1-second waveform remains valid through every pooling block."""
    encoder = LogMelEncoder(
        in_dim=4_410,
        hidden_dim=4,
        out_dim=5,
        sample_rate=44_100,
        num_blocks=4,
        kernel_size=3,
    )

    embedding = encoder(torch.zeros(2, 4_410))

    assert embedding.shape == (2, 5)


def test_cnn_residual_mlp_unknown_frontend_raises() -> None:
    """An unsupported front-end name fails instead of silently selecting log-mel."""
    with pytest.raises(ValueError, match="Unsupported frontend"):
        CNNResidualMLP(frontend="unknown", sample_rate=44_100)  # type: ignore[arg-type]


def test_cnn_residual_mlp_log_mel_without_sample_rate_raises() -> None:
    """Log-mel configuration fails before constructing an invalid transform."""
    with pytest.raises(ValueError, match="sample_rate is required"):
        CNNResidualMLP(frontend="log_mel")


@pytest.mark.parametrize("audio", [torch.zeros(2, 1, 4_410), torch.zeros(2, 4_409)])
def test_log_mel_frontend_invalid_waveform_shape_raises(audio: torch.Tensor) -> None:
    """Malformed waveform batches fail at the encoder boundary.

    :param audio: Wrong-rank or wrong-length waveform batch.
    """
    encoder = LogMelEncoder(
        in_dim=4_410,
        hidden_dim=4,
        out_dim=5,
        sample_rate=44_100,
        num_blocks=1,
        kernel_size=3,
    )

    with pytest.raises(ValueError, match="Expected waveform shape"):
        encoder(audio)


def test_log_mel_frontend_unknown_norm_raises() -> None:
    """An unsupported normalization name fails instead of selecting GroupNorm."""
    with pytest.raises(ValueError, match="Unsupported norm"):
        LogMelEncoder(
            in_dim=4_410,
            hidden_dim=4,
            out_dim=5,
            sample_rate=44_100,
            norm="unknown",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("amin", [0.0, -1.0, float("inf"), float("nan")])
def test_log_mel_frontend_invalid_amin_raises(amin: float) -> None:
    """A non-positive or non-finite logarithm floor is rejected.

    :param amin: Invalid power floor.
    """
    with pytest.raises(ValueError, match="amin"):
        LogMelEncoder(4_410, 4, 5, sample_rate=44_100, amin=amin)


@pytest.mark.parametrize("power", [0.0, -1.0, float("inf"), float("nan")])
def test_log_mel_frontend_invalid_power_raises(power: float) -> None:
    """A non-positive or non-finite magnitude exponent is rejected.

    :param power: Invalid spectrogram exponent.
    """
    with pytest.raises(ValueError, match="power"):
        LogMelEncoder(4_410, 4, 5, sample_rate=44_100, power=power)


@pytest.mark.parametrize("top_db", [-1.0, float("inf"), float("nan")])
def test_log_mel_frontend_invalid_top_db_raises(top_db: float) -> None:
    """A negative or non-finite dynamic range is rejected.

    :param top_db: Invalid dynamic range.
    """
    with pytest.raises(ValueError, match="top_db"):
        LogMelEncoder(4_410, 4, 5, sample_rate=44_100, top_db=top_db)


def test_log_mel_frontend_unknown_window_raises() -> None:
    """An unsupported Fourier window fails before transform construction."""
    with pytest.raises(ValueError, match="Unsupported window"):
        LogMelEncoder(
            4_410,
            4,
            5,
            sample_rate=44_100,
            window="blackman",  # type: ignore[arg-type]
        )


def test_log_mel_spectrogram_matches_dataset_frontend() -> None:
    """All frames preserve the stored-mel frontend's numeric contract."""
    audio = torch.randn(1, 4_410)
    encoder = LogMelEncoder(
        in_dim=4_410,
        hidden_dim=4,
        out_dim=5,
        sample_rate=44_100,
        num_blocks=1,
        kernel_size=3,
    )
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

    actual = encoder.log_mel_spectrogram(audio)[0].detach().numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)


def test_log_mel_spectrogram_power_one_matches_amplitude_decibels() -> None:
    """Magnitude spectrograms use amplitude rather than power decibel scaling."""
    audio = torch.randn(1, 4_410)
    encoder = LogMelEncoder(
        in_dim=4_410,
        hidden_dim=4,
        out_dim=5,
        sample_rate=44_100,
        power=1.0,
        num_blocks=1,
    )
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

    actual = encoder.log_mel_spectrogram(audio)[0].detach().numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)


def test_log_mel_frontend_supported_alternates_return_finite_embedding() -> None:
    """Supported non-default transform and normalization options remain operational."""
    encoder = LogMelEncoder(
        4_410,
        4,
        5,
        sample_rate=44_100,
        mel_scale="htk",
        norm="ln",
        top_db=None,
        window="hann",
        num_blocks=1,
    )

    embedding = encoder(torch.randn(2, 4_410))

    assert embedding.shape == (2, 5)
    assert torch.isfinite(embedding).all()


def test_log_mel_frontend_distinct_spectra_return_distinct_embeddings() -> None:
    """The encoder responds to spectral content instead of returning a constant."""
    time = torch.arange(4_410) / 44_100
    audio = torch.stack(
        [
            torch.sin(2 * torch.pi * 220 * time),
            torch.sin(2 * torch.pi * 1_760 * time),
        ]
    )
    encoder = LogMelEncoder(
        in_dim=4_410,
        hidden_dim=4,
        out_dim=5,
        sample_rate=44_100,
        num_blocks=1,
        kernel_size=3,
    )
    encoder.eval()

    with torch.no_grad():
        embeddings = encoder(audio)

    assert torch.isfinite(embeddings).all()
    assert not torch.allclose(embeddings[0], embeddings[1])


def test_log_mel_frontend_distinct_envelopes_return_distinct_embeddings() -> None:
    """The encoder preserves temporal-envelope information for one carrier."""
    time = torch.arange(4_410) / 44_100
    carrier = torch.sin(2 * torch.pi * 440 * time)
    audio = torch.stack(
        [carrier * torch.linspace(0, 1, 4_410), carrier * torch.linspace(1, 0, 4_410)]
    )
    encoder = LogMelEncoder(
        in_dim=4_410,
        hidden_dim=4,
        out_dim=5,
        sample_rate=44_100,
        num_blocks=1,
        kernel_size=3,
    )
    encoder.eval()

    with torch.no_grad():
        embeddings = encoder(audio)

    assert torch.isfinite(embeddings).all()
    assert not torch.allclose(embeddings[0], embeddings[1])


def test_log_mel_frontend_eval_prediction_is_independent_of_batch_peers() -> None:
    """An example's inference output does not depend on neighboring examples."""
    model = CNNResidualMLP(
        in_dim=4_410,
        channels=4,
        encoder_blocks=1,
        trunk_blocks=1,
        hidden_dim=8,
        out_dim=2,
        kernel_size=3,
        frontend="log_mel",
        sample_rate=44_100,
    )
    model.eval()
    anchor = torch.randn(4_410)

    with torch.no_grad():
        first = model(torch.stack([anchor, torch.zeros_like(anchor)]))[0]
        second = model(torch.stack([anchor, torch.randn_like(anchor)]))[0]

    torch.testing.assert_close(first, second)


def test_log_mel_frontend_backward_reaches_every_parameter() -> None:
    """A real prediction loss sends finite, non-zero gradients through the network."""
    model = CNNResidualMLP(
        in_dim=4_410,
        channels=4,
        encoder_blocks=1,
        trunk_blocks=1,
        hidden_dim=8,
        out_dim=2,
        kernel_size=3,
        frontend="log_mel",
        sample_rate=44_100,
    )

    F.mse_loss(model(torch.randn(2, 4_410)), torch.rand(2, 2)).backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad), name


@pytest.mark.slow
def test_log_mel_frontend_overfits_fixed_envelope_examples() -> None:
    """The complete frontend and trunk can learn temporal-envelope differences."""
    time = torch.arange(4_410) / 44_100
    carrier = torch.sin(2 * torch.pi * 440 * time)
    audio = torch.stack(
        [
            carrier * torch.linspace(0, 1, 4_410),
            carrier * torch.linspace(1, 0, 4_410),
        ]
    )
    targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    model = CNNResidualMLP(
        in_dim=4_410,
        channels=4,
        encoder_blocks=1,
        trunk_blocks=1,
        hidden_dim=8,
        out_dim=2,
        kernel_size=3,
        frontend="log_mel",
        sample_rate=44_100,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    initial_loss = F.mse_loss(model(audio), targets).item()

    for _ in range(100):
        optimizer.zero_grad()
        loss = F.mse_loss(model(audio), targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_loss = F.mse_loss(model(audio), targets).item()
    assert final_loss < initial_loss / 100
