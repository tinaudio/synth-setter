"""Behaviour of the exponential Fourier value encoder used by the VST flow projection."""

import math

import pytest
import torch

from synth_setter.models.components.fourier_number import (
    ExpoFourierFeatures,
    FourierNumberEmbedder,
)


def test_expo_fourier_features_known_input_returns_hardcoded_cos_sin_values() -> None:
    """A quarter and a half turn produce the documented cosine-then-sine layout."""
    encoder = ExpoFourierFeatures(dim=4, min_freq=0.25, max_freq=0.5)

    out = encoder(torch.tensor([1.0]))

    torch.testing.assert_close(out, torch.tensor([[0.0, -1.0, 1.0, 0.0]]), atol=1e-6, rtol=0)


def test_expo_fourier_features_endpoints_span_requested_frequency_range() -> None:
    """The band includes both requested endpoints rather than approaching them."""
    encoder = ExpoFourierFeatures(dim=8, min_freq=0.5, max_freq=32.0)

    torch.testing.assert_close(encoder.frequencies[0], torch.tensor(0.5))
    torch.testing.assert_close(encoder.frequencies[-1], torch.tensor(32.0))


def test_expo_fourier_features_scalar_input_returns_one_dimensional_output() -> None:
    """A 0-d input gains exactly one feature dimension."""
    encoder = ExpoFourierFeatures(dim=16)

    assert encoder(torch.tensor(0.3)).shape == (16,)


def test_expo_fourier_features_batched_params_appends_feature_dimension() -> None:
    """A (B, P) parameter batch encodes to (B, P, dim)."""
    encoder = ExpoFourierFeatures(dim=16)

    assert encoder(torch.zeros(3, 5)).shape == (3, 5, 16)


def test_expo_fourier_features_value_outside_unit_range_is_not_clipped() -> None:
    """Flow states beyond [-1, 1] keep distinct encodings instead of saturating."""
    encoder = ExpoFourierFeatures(dim=16, min_freq=0.5, max_freq=8.0)

    in_range = encoder(torch.tensor([1.0]))
    out_of_range = encoder(torch.tensor([4.0]))

    assert not torch.allclose(in_range, out_of_range)


def test_expo_fourier_features_bfloat16_input_returns_bfloat16_output() -> None:
    """Internal float32 trigonometry is invisible to the caller's dtype."""
    encoder = ExpoFourierFeatures(dim=16)

    assert encoder(torch.zeros(2, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_expo_fourier_features_has_no_trainable_parameters() -> None:
    """The band is deterministic, so training never moves it."""
    encoder = ExpoFourierFeatures(dim=16)

    assert list(encoder.parameters()) == []


def test_expo_fourier_features_frequencies_absent_from_state_dict() -> None:
    """Checkpoints stay loadable when the band is tuned to a different range."""
    encoder = ExpoFourierFeatures(dim=16)

    assert encoder.state_dict() == {}


def test_expo_fourier_features_odd_dim_raises_value_error() -> None:
    """An odd width cannot split evenly into cosine and sine halves."""
    with pytest.raises(ValueError, match="even"):
        ExpoFourierFeatures(dim=7)


def test_expo_fourier_features_dim_below_four_raises_value_error() -> None:
    """A single-frequency half-width would divide by zero when spacing the band."""
    with pytest.raises(ValueError, match="at least 4"):
        ExpoFourierFeatures(dim=2)


def test_expo_fourier_features_non_positive_min_freq_raises_value_error() -> None:
    """Logarithmic spacing is undefined at or below zero."""
    with pytest.raises(ValueError, match="min_freq"):
        ExpoFourierFeatures(dim=16, min_freq=0.0)


def test_expo_fourier_features_max_freq_below_min_freq_raises_value_error() -> None:
    """An inverted band is rejected instead of silently descending."""
    with pytest.raises(ValueError, match="max_freq"):
        ExpoFourierFeatures(dim=16, min_freq=4.0, max_freq=1.0)


def test_expo_fourier_features_integer_input_raises_type_error() -> None:
    """Integer input would cast the encoding back to integers and lose the phase."""
    encoder = ExpoFourierFeatures(dim=16)

    with pytest.raises(TypeError, match="floating"):
        encoder(torch.tensor([1, 2]))


def test_fourier_number_embedder_projects_params_to_requested_width() -> None:
    """The embedder emits the configured feature width, not the Fourier width."""
    embedder = FourierNumberEmbedder(features=12, dim=16)

    assert embedder(torch.zeros(2, 5)).shape == (2, 5, 12)


def test_fourier_number_embedder_backward_reaches_projection_weight() -> None:
    """The output projection is trainable through the encoding."""
    embedder = FourierNumberEmbedder(features=12, dim=16)

    embedder(torch.rand(2, 5)).sum().backward()

    assert embedder.projection.weight.grad is not None
    assert torch.isfinite(embedder.projection.weight.grad).all()


def test_fourier_number_embedder_distinguishes_values_a_tenth_apart() -> None:
    """The default band separates nearby values instead of acting linearly."""
    embedder = FourierNumberEmbedder(features=32, dim=64, min_freq=0.5, max_freq=32.0)

    encoded = embedder.fourier_features(torch.tensor([0.5, 0.6]))
    similarity = torch.nn.functional.cosine_similarity(encoded[0], encoded[1], dim=0)

    assert similarity < 0.5


def test_fourier_number_embedder_nan_input_propagates_nan() -> None:
    """A non-finite coordinate surfaces rather than being silently absorbed."""
    embedder = FourierNumberEmbedder(features=8, dim=16)

    assert torch.isnan(embedder(torch.tensor([math.nan]))).any()
