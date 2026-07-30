"""Behavioral tests for the pooled spectrogram CNN and the model that composes it."""

from functools import partial

import jaxtyping
import pytest
import torch
import torch.nn.functional as F

from synth_setter.models.components.cnn import MelCNN
from synth_setter.models.components.residual_mlp import LogMelCNNResidualMLP

_mel_cnn = partial(MelCNN, hidden_dim=4, out_dim=5, num_blocks=1, kernel_size=3)
_log_mel_model = partial(
    LogMelCNNResidualMLP,
    in_dim=4_410,
    channels=4,
    encoder_blocks=1,
    trunk_blocks=1,
    hidden_dim=8,
    out_dim=2,
    kernel_size=3,
    sample_rate=44_100,
)


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Keep model initialization and synthetic waveforms deterministic."""
    torch.manual_seed(0)


def test_mel_cnn_pools_a_spectrogram_grid_to_one_embedding_per_row() -> None:
    """Adaptive pooling makes the embedding width independent of the grid size."""
    assert _mel_cnn()(torch.zeros(2, 1, 128, 11)).shape == (2, 5)


def test_mel_cnn_four_pooling_blocks_survive_a_short_grid() -> None:
    """A 0.1-second waveform's grid remains valid through every pooling block."""
    backbone = MelCNN(hidden_dim=4, out_dim=5, num_blocks=4, kernel_size=3)

    assert backbone(torch.zeros(2, 1, 128, 11)).shape == (2, 5)


def test_mel_cnn_stereo_grid_returns_embedding() -> None:
    """A stored two-channel mel grid feeds the same backbone as the mono online grid."""
    backbone = MelCNN(hidden_dim=4, out_dim=5, input_channels=2, num_blocks=1, kernel_size=3)

    assert backbone(torch.zeros(2, 2, 128, 11)).shape == (2, 5)


def test_mel_cnn_layer_norm_returns_finite_embedding() -> None:
    """The layer-normalized convolution path remains operational."""
    embedding = _mel_cnn(norm="ln")(torch.randn(2, 1, 128, 11))

    assert embedding.shape == (2, 5)
    assert torch.isfinite(embedding).all()


def test_mel_cnn_unknown_norm_raises() -> None:
    """An unsupported normalization name fails instead of silently selecting GroupNorm."""
    with pytest.raises(jaxtyping.TypeCheckError):
        _mel_cnn(norm="unknown")  # type: ignore[arg-type]


def test_mel_cnn_non_positive_kernel_size_raises() -> None:
    """A zero kernel fails at the configuration boundary, not inside the convolution."""
    with pytest.raises(ValueError, match="kernel_size"):
        _mel_cnn(kernel_size=0)


def test_log_mel_model_four_second_audio_returns_bounded_embedding() -> None:
    """Four-second audio produces predictions without a length-sized linear head."""
    model = LogMelCNNResidualMLP(
        in_dim=176_400,
        channels=4,
        encoder_blocks=2,
        trunk_blocks=1,
        hidden_dim=16,
        out_dim=3,
        kernel_size=3,
        norm="bn",
        sample_rate=44_100,
    )
    model.eval()

    with torch.no_grad():
        prediction = model(torch.zeros(2, 176_400))

    assert prediction.shape == (2, 3)
    assert sum(parameter.numel() for parameter in model.parameters()) < 100_000


def test_log_mel_model_predictions_stay_in_normalized_parameter_range() -> None:
    """TorchSynth predictions stay inside the renderer's normalized domain."""
    predictions = _log_mel_model()(torch.randn(16, 4_410))

    assert torch.all((0 <= predictions) & (predictions <= 1))


def test_log_mel_model_initial_predictions_center_normalized_range() -> None:
    """Initial predictions stay near the normalized target mean."""
    predictions = _log_mel_model()(torch.randn(16, 4_410))

    torch.testing.assert_close(predictions.mean(), torch.tensor(0.5), rtol=0, atol=0.1)


def test_log_mel_model_eval_prediction_is_independent_of_batch_peers() -> None:
    """An example's inference output does not depend on neighboring examples."""
    model = _log_mel_model()
    model.eval()
    anchor = torch.randn(4_410)

    with torch.no_grad():
        first = model(torch.stack([anchor, torch.zeros_like(anchor)]))[0]
        second = model(torch.stack([anchor, torch.randn_like(anchor)]))[0]

    torch.testing.assert_close(first, second)


def test_log_mel_model_backward_reaches_every_parameter() -> None:
    """A real prediction loss sends finite, non-zero gradients through the network."""
    model = _log_mel_model()

    F.mse_loss(model(torch.randn(2, 4_410)), torch.rand(2, 2)).backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad), name


@pytest.mark.slow
def test_log_mel_model_overfits_fixed_envelope_examples() -> None:
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
    model = _log_mel_model()
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
