"""Behavioral tests for waveform encoders."""

from collections.abc import Callable
from functools import partial

import librosa
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from synth_setter.models.components.cnn import LogMelEncoder
from synth_setter.models.components.residual_mlp import LogMelCNNResidualMLP

_log_mel_encoder = partial(
    LogMelEncoder,
    in_dim=4_410,
    hidden_dim=4,
    out_dim=5,
    sample_rate=44_100,
    num_blocks=1,
    kernel_size=3,
)
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


def test_log_mel_frontend_four_second_audio_returns_bounded_embedding() -> None:
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


def test_log_mel_frontend_predictions_stay_in_normalized_parameter_range() -> None:
    """TorchSynth predictions stay inside the renderer's normalized domain."""
    predictions = _log_mel_model()(torch.randn(16, 4_410))

    assert torch.all((0 <= predictions) & (predictions <= 1))


def test_log_mel_frontend_initial_predictions_center_normalized_range() -> None:
    """Initial predictions stay near the normalized target mean."""
    predictions = _log_mel_model()(torch.randn(16, 4_410))

    torch.testing.assert_close(predictions.mean(), torch.tensor(0.5), rtol=0, atol=0.1)


def test_log_mel_frontend_sign_inversion_returns_same_embedding() -> None:
    """A pi phase shift leaves the magnitude-based embedding unchanged."""
    encoder = _log_mel_encoder()
    encoder.eval()
    audio = torch.randn(2, 4_410)

    with torch.no_grad():
        original = encoder(audio)
        phase_shifted = encoder(-audio)

    torch.testing.assert_close(phase_shifted, original)


def test_log_mel_frontend_short_smoke_audio_returns_embedding() -> None:
    """A 0.1-second waveform remains valid through every pooling block."""
    encoder = _log_mel_encoder(num_blocks=4)

    embedding = encoder(torch.zeros(2, 4_410))

    assert embedding.shape == (2, 5)


@pytest.mark.parametrize("audio", [torch.zeros(2, 1, 4_410), torch.zeros(2, 4_409)])
def test_log_mel_frontend_invalid_waveform_shape_raises(audio: torch.Tensor) -> None:
    """Malformed waveform batches fail at the encoder boundary.

    :param audio: Wrong-rank or wrong-length waveform batch.
    """
    encoder = _log_mel_encoder()

    with pytest.raises(ValueError, match="Expected waveform shape"):
        encoder(audio)


def test_log_mel_frontend_unknown_norm_raises() -> None:
    """An unsupported normalization name fails instead of selecting GroupNorm."""
    with pytest.raises(ValueError, match="Unsupported norm"):
        _log_mel_encoder(norm="unknown")  # type: ignore[arg-type]


@pytest.mark.parametrize("amin", [0.0, -1.0, float("inf"), float("nan")])
def test_log_mel_frontend_invalid_amin_raises(amin: float) -> None:
    """A non-positive or non-finite logarithm floor is rejected.

    :param amin: Invalid power floor.
    """
    with pytest.raises(ValueError, match="amin"):
        _log_mel_encoder(amin=amin)


@pytest.mark.parametrize("power", [0.0, -1.0, float("inf"), float("nan")])
def test_log_mel_frontend_invalid_power_raises(power: float) -> None:
    """A non-positive or non-finite magnitude exponent is rejected.

    :param power: Invalid spectrogram exponent.
    """
    with pytest.raises(ValueError, match="power"):
        _log_mel_encoder(power=power)


@pytest.mark.parametrize("top_db", [-1.0, float("inf"), float("nan")])
def test_log_mel_frontend_invalid_top_db_raises(top_db: float) -> None:
    """A negative or non-finite dynamic range is rejected.

    :param top_db: Invalid dynamic range.
    """
    with pytest.raises(ValueError, match="top_db"):
        _log_mel_encoder(top_db=top_db)


@pytest.mark.parametrize(
    ("factory", "frequency_name"),
    [
        pytest.param(partial(_log_mel_encoder, f_min=-1.0), "f_min", id="negative-f-min"),
        pytest.param(partial(_log_mel_encoder, f_min=float("inf")), "f_min", id="infinite-f-min"),
        pytest.param(partial(_log_mel_encoder, f_min=float("nan")), "f_min", id="nan-f-min"),
        pytest.param(partial(_log_mel_encoder, f_min=22_050.0), "f_min", id="f-min-at-nyquist"),
        pytest.param(
            partial(_log_mel_encoder, f_max=0.0),
            "f_max",
            id="f-max-not-greater-than-f-min",
        ),
        pytest.param(partial(_log_mel_encoder, f_max=float("inf")), "f_max", id="infinite-f-max"),
        pytest.param(partial(_log_mel_encoder, f_max=float("nan")), "f_max", id="nan-f-max"),
        pytest.param(partial(_log_mel_encoder, f_max=22_051.0), "f_max", id="f-max-above-nyquist"),
    ],
)
def test_log_mel_frontend_invalid_frequency_bound_raises(
    factory: Callable[[], LogMelEncoder], frequency_name: str
) -> None:
    """Invalid mel-frequency bounds fail before producing non-finite features.

    :param factory: Encoder factory containing the invalid bound.
    :param frequency_name: Constructor argument receiving the invalid bound.
    """
    with pytest.raises(ValueError, match=frequency_name):
        factory()


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        pytest.param(partial(_log_mel_encoder, hop_length=0), "hop_length", id="zero-hop-length"),
        pytest.param(
            partial(_log_mel_encoder, kernel_size=0), "kernel_size", id="zero-kernel-size"
        ),
        pytest.param(partial(_log_mel_encoder, n_fft=0), "n_fft", id="zero-n-fft"),
        pytest.param(partial(_log_mel_encoder, n_mels=0), "n_mels", id="zero-n-mels"),
    ],
)
def test_log_mel_frontend_non_positive_geometry_raises(
    factory: Callable[[], LogMelEncoder], field: str
) -> None:
    """Non-positive frontend geometry fails at the configuration boundary.

    :param factory: Encoder factory containing the zero size.
    :param field: Constructor argument receiving the zero size.
    """
    with pytest.raises(ValueError, match=field):
        factory()


def test_log_mel_frontend_unknown_window_raises() -> None:
    """An unsupported Fourier window fails before transform construction."""
    with pytest.raises(ValueError, match="Unsupported window"):
        _log_mel_encoder(window="blackman")  # type: ignore[arg-type]


def test_log_mel_spectrogram_matches_dataset_frontend() -> None:
    """All frames preserve the stored-mel frontend's numeric contract."""
    audio = torch.randn(1, 4_410)
    encoder = _log_mel_encoder()
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
    encoder = _log_mel_encoder(power=1.0)
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


def test_log_mel_spectrogram_htk_scale_matches_dataset_frontend() -> None:
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

    actual = _log_mel_encoder(mel_scale="htk").log_mel_spectrogram(audio)[0].numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)


def test_log_mel_spectrogram_hann_window_matches_dataset_frontend() -> None:
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

    actual = _log_mel_encoder(window="hann").log_mel_spectrogram(audio)[0].numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)


def test_log_mel_spectrogram_top_db_clips_relative_dynamic_range() -> None:
    """The dynamic-range option clips values relative to each waveform peak."""
    audio = torch.randn(1, 4_410)
    unclipped = _log_mel_encoder(top_db=None).log_mel_spectrogram(audio)
    clipped = _log_mel_encoder(top_db=10.0).log_mel_spectrogram(audio)

    torch.testing.assert_close(clipped, torch.clamp(unclipped, min=-10.0))


def test_log_mel_frontend_layer_norm_returns_finite_embedding() -> None:
    """The layer-normalized convolution path remains operational."""
    embedding = _log_mel_encoder(norm="ln")(torch.randn(2, 4_410))

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
    encoder = _log_mel_encoder()
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
    encoder = _log_mel_encoder()
    encoder.eval()

    with torch.no_grad():
        embeddings = encoder(audio)

    assert torch.isfinite(embeddings).all()
    assert not torch.allclose(embeddings[0], embeddings[1])


def test_log_mel_frontend_eval_prediction_is_independent_of_batch_peers() -> None:
    """An example's inference output does not depend on neighboring examples."""
    model = _log_mel_model()
    model.eval()
    anchor = torch.randn(4_410)

    with torch.no_grad():
        first = model(torch.stack([anchor, torch.zeros_like(anchor)]))[0]
        second = model(torch.stack([anchor, torch.randn_like(anchor)]))[0]

    torch.testing.assert_close(first, second)


def test_log_mel_frontend_backward_reaches_every_parameter() -> None:
    """A real prediction loss sends finite, non-zero gradients through the network."""
    model = _log_mel_model()

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
