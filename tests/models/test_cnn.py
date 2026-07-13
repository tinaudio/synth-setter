"""Behavioral tests for waveform encoders."""

import pytest
import torch

from synth_setter.models.components.cnn import LogMelEncoder
from synth_setter.models.components.residual_mlp import CNNResidualMLP


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
    """The 0.1 s CPU smoke-test override survives every pooling block."""
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
