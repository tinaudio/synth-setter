"""Behaviour tests for the pluggable audio-distance spaces the feedback term measures in."""

import numpy as np
import pytest
import torch

from synth_setter.evaluation.compute_audio_metrics import (
    MEL_PARAMS,
    compute_mldr_corresponding_channels,
    compute_mldr_mid_side_stereo_only,
)
from synth_setter.models.components.audio_distance import (
    MEL_SCALES,
    LatentMseDistance,
    MultichannelAudioDistance,
    MultiScaleSpectralDistance,
)

_SAMPLE_RATE = 16000
_LENGTH = 8192


def _mss() -> MultiScaleSpectralDistance:
    """Build a multi-scale spectral distance at test geometry.

    :returns: Configured distance module.
    """
    return MultiScaleSpectralDistance(sample_rate=_SAMPLE_RATE)


def test_scales_match_the_reported_evaluation_metric() -> None:
    """Training and reporting must not drift onto different resolutions."""
    assert MEL_SCALES == tuple(tuple(entry) for entry in MEL_PARAMS)


def test_identical_audio_has_zero_distance() -> None:
    """A perfect render is the fixed point the term drives toward."""
    torch.manual_seed(0)
    audio = torch.randn(3, _LENGTH).clamp(-1.0, 1.0)

    assert _mss()(audio, audio).abs().max().item() == pytest.approx(0.0, abs=1e-5)


def test_different_audio_has_positive_distance() -> None:
    """Distinct spectra must be separated, not collapsed."""
    torch.manual_seed(0)
    rendered = torch.randn(3, _LENGTH).clamp(-1.0, 1.0)
    target = torch.zeros(3, _LENGTH)

    assert (_mss()(rendered, target) > 0).all()


def test_distance_is_reported_per_sample() -> None:
    """The caller weights each row by its own flow time."""
    torch.manual_seed(0)
    rendered = torch.randn(4, _LENGTH).clamp(-1.0, 1.0)
    target = torch.randn(4, _LENGTH).clamp(-1.0, 1.0)

    assert _mss()(rendered, target).shape == (4,)


def test_gradient_reaches_the_rendered_waveform() -> None:
    """Without this the term cannot train the field that produced the render."""
    torch.manual_seed(0)
    rendered = torch.randn(2, _LENGTH).clamp(-1.0, 1.0).requires_grad_(True)
    target = torch.zeros(2, _LENGTH)

    _mss()(rendered, target).sum().backward()

    assert rendered.grad is not None
    assert torch.isfinite(rendered.grad).all()
    assert torch.count_nonzero(rendered.grad) > 0


def test_silent_pair_stays_finite() -> None:
    """Silent renders are common online and must not poison the gradient."""
    silent = torch.zeros(2, _LENGTH, requires_grad=True)

    distance = _mss()(silent, torch.zeros(2, _LENGTH))
    distance.sum().backward()

    assert torch.isfinite(distance).all()
    assert silent.grad is not None
    assert torch.isfinite(silent.grad).all()


def _multichannel(
    *,
    spectral_weight: float = 1.0,
    channel_mldr_weight: float = 0.1,
    pair_mldr_weight: float = 0.1,
) -> MultichannelAudioDistance:
    """Build the shipped multichannel distance at inexpensive test geometry.

    :param spectral_weight: Corresponding-channel MSS weight.
    :param channel_mldr_weight: Corresponding-channel MLDR weight.
    :param pair_mldr_weight: Within-signal sum/difference MLDR weight.
    :returns: Configured distance module.
    """
    return MultichannelAudioDistance(
        sample_rate=1_000,
        spectral_weight=spectral_weight,
        channel_mldr_weight=channel_mldr_weight,
        pair_mldr_weight=pair_mldr_weight,
    )


def test_multichannel_distance_accepts_flat_mono_and_reports_rows() -> None:
    """Legacy ``(batch, samples)`` audio canonicalizes to one channel."""
    audio = torch.randn(2, 2_048)

    assert _multichannel()(audio, audio).shape == (2,)


@pytest.mark.parametrize(
    ("rendered", "target"),
    [
        pytest.param(torch.zeros(1, 0), torch.zeros(1, 0), id="empty-samples"),
        pytest.param(torch.zeros(0, 32), torch.zeros(0, 32), id="empty-batch"),
        pytest.param(torch.zeros(1, 2, 32), torch.zeros(1, 3, 32), id="channel-mismatch"),
        pytest.param(torch.zeros(1, 32), torch.zeros(2, 32), id="batch-mismatch"),
        pytest.param(torch.zeros(32), torch.zeros(32), id="missing-batch"),
    ],
)
def test_multichannel_distance_rejects_invalid_geometry(
    rendered: torch.Tensor, target: torch.Tensor
) -> None:
    """Only exact matching nonempty ``(B,T)`` or ``(B,C,T)`` geometry is valid.

    :param rendered: Invalid predicted geometry.
    :param target: Invalid target geometry.
    """
    with pytest.raises(ValueError, match="matching nonempty"):
        _multichannel()(rendered, target)


def test_multichannel_distance_rejects_nonfinite_audio() -> None:
    """NaN audio must fail before contaminating a training loss."""
    rendered = torch.zeros(1, 32)
    rendered[0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        _multichannel()(rendered, torch.zeros_like(rendered))


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        pytest.param((-1.0, 0.0, 0.0), "nonnegative", id="negative"),
        pytest.param((float("inf"), 0.0, 0.0), "finite", id="infinite"),
        pytest.param((0.0, 0.0, 0.0), "positive", id="all-zero"),
    ],
)
def test_multichannel_distance_rejects_invalid_weights(
    weights: tuple[float, float, float], match: str
) -> None:
    """Component weights define a finite nonnegative nonempty objective.

    :param weights: Spectral, corresponding MLDR, and pair MLDR weights.
    :param match: Expected validation message fragment.
    """
    with pytest.raises(ValueError, match=match):
        _multichannel(
            spectral_weight=weights[0],
            channel_mldr_weight=weights[1],
            pair_mldr_weight=weights[2],
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_torch_mldr_matches_numpy_corresponding_channels(dtype: torch.dtype) -> None:
    """Differentiable MLDR preserves PR1's reported metric contract.

    :param dtype: Floating-point precision under test.
    """
    time = torch.arange(3_000, dtype=dtype) / 1_000
    target = torch.stack((torch.sin(2 * torch.pi * 7 * time), torch.sin(2 * torch.pi * 13 * time)))
    pred = target * (1.0 + 0.4 * torch.sin(2 * torch.pi * 3 * time))
    distance = _multichannel(spectral_weight=0.0, channel_mldr_weight=1.0, pair_mldr_weight=0.0)

    actual = distance(pred.unsqueeze(0), target.unsqueeze(0))[0]
    expected = compute_mldr_corresponding_channels(target.numpy(), pred.numpy(), sample_rate=1_000)

    assert actual.dtype == dtype
    assert actual.item() == pytest.approx(expected, rel=2e-4, abs=2e-5)


def test_pair_mldr_matches_numpy_stereo_mid_side() -> None:
    """The sole stereo pair has the same energy-preserving mid/side semantics as PR1."""
    time = torch.arange(3_000, dtype=torch.float64) / 1_000
    target = torch.stack((torch.sin(2 * torch.pi * 7 * time), torch.sin(2 * torch.pi * 11 * time)))
    pred = torch.stack((target[0], target[1] * (1.0 + 0.5 * torch.sin(2 * torch.pi * 2 * time))))
    distance = _multichannel(spectral_weight=0.0, channel_mldr_weight=0.0, pair_mldr_weight=1.0)

    actual = distance(pred.unsqueeze(0), target.unsqueeze(0))[0]
    expected = compute_mldr_mid_side_stereo_only(target.numpy(), pred.numpy(), sample_rate=1_000)

    assert actual.item() == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_pair_mldr_detects_spatial_polarity_error_hidden_by_channel_terms() -> None:
    """Sum/difference pairs detect spatial errors invisible to magnitude and energy terms."""
    signal = torch.sin(torch.arange(3_000) * 0.07)
    target = torch.stack((signal, signal)).unsqueeze(0)
    rendered = torch.stack((signal, -signal)).unsqueeze(0)

    channel_only = _multichannel(
        spectral_weight=1.0, channel_mldr_weight=1.0, pair_mldr_weight=0.0
    )(rendered, target)
    pair_only = _multichannel(spectral_weight=0.0, channel_mldr_weight=0.0, pair_mldr_weight=1.0)(
        rendered, target
    )

    assert channel_only.item() == pytest.approx(0.0, abs=1e-5)
    assert pair_only.item() > 0.0


def test_pair_mldr_averages_all_three_channel_pairs() -> None:
    """Three-channel pair scoring is the mean of pairs 0-1, 0-2, and 1-2."""
    time = torch.arange(3_000, dtype=torch.float64) / 1_000
    target = torch.stack(
        (
            torch.sin(2 * torch.pi * 5 * time),
            torch.sin(2 * torch.pi * 7 * time),
            torch.sin(2 * torch.pi * 11 * time),
        )
    )
    rendered = target.clone()
    rendered[2] *= 1.0 + 0.7 * torch.sin(2 * torch.pi * 2 * time)
    distance = _multichannel(spectral_weight=0.0, channel_mldr_weight=0.0, pair_mldr_weight=1.0)
    scale = 2.0**-0.5

    actual = distance(rendered.unsqueeze(0), target.unsqueeze(0)).item()
    expected = (
        compute_mldr_corresponding_channels(
            np.stack(((target[0] + target[1]) * scale, (target[0] - target[1]) * scale)),
            np.stack(((rendered[0] + rendered[1]) * scale, (rendered[0] - rendered[1]) * scale)),
            sample_rate=1_000,
        )
        + compute_mldr_corresponding_channels(
            np.stack(((target[0] + target[2]) * scale, (target[0] - target[2]) * scale)),
            np.stack(((rendered[0] + rendered[2]) * scale, (rendered[0] - rendered[2]) * scale)),
            sample_rate=1_000,
        )
        + compute_mldr_corresponding_channels(
            np.stack(((target[1] + target[2]) * scale, (target[1] - target[2]) * scale)),
            np.stack(((rendered[1] + rendered[2]) * scale, (rendered[1] - rendered[2]) * scale)),
            sample_rate=1_000,
        )
    ) / 3.0

    assert actual == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_multichannel_distance_is_weighted_sum_of_three_components() -> None:
    """Configured weights scale the three component distances without hidden normalization."""
    time = torch.arange(3_000) / 1_000
    target = torch.stack(
        (torch.sin(2 * torch.pi * 5 * time), torch.sin(2 * torch.pi * 11 * time))
    ).unsqueeze(0)
    rendered = target * (1.0 + 0.4 * torch.sin(2 * torch.pi * 2 * time))

    spectral = _multichannel(spectral_weight=1.0, channel_mldr_weight=0.0, pair_mldr_weight=0.0)(
        rendered, target
    )
    channel_mldr = _multichannel(
        spectral_weight=0.0, channel_mldr_weight=1.0, pair_mldr_weight=0.0
    )(rendered, target)
    pair_mldr = _multichannel(spectral_weight=0.0, channel_mldr_weight=0.0, pair_mldr_weight=1.0)(
        rendered, target
    )

    actual = _multichannel(spectral_weight=0.7, channel_mldr_weight=0.2, pair_mldr_weight=0.1)(
        rendered, target
    )

    torch.testing.assert_close(actual, 0.7 * spectral + 0.2 * channel_mldr + 0.1 * pair_mldr)


def test_multichannel_distance_detects_channel_permutation() -> None:
    """Corresponding-channel terms reject permutation as a different spatial assignment."""
    time = torch.arange(3_000) / 1_000
    target = torch.stack(
        (torch.sin(2 * torch.pi * 5 * time), torch.sin(2 * torch.pi * 17 * time))
    ).unsqueeze(0)

    score = _multichannel()(target.flip(1), target)

    assert score.item() > 0.0


def test_pair_mldr_is_absent_for_mono() -> None:
    """Mono has no unordered channel pair and receives exactly zero pair contribution."""
    rendered = torch.randn(2, 2_048)
    target = torch.randn(2, 2_048)
    distance = _multichannel(spectral_weight=0.0, channel_mldr_weight=0.0, pair_mldr_weight=1.0)

    torch.testing.assert_close(distance(rendered, target), torch.zeros(2))


def test_multichannel_distance_preserves_finite_render_gradients_only() -> None:
    """Every component trains rendered channels while treating targets as detached data."""
    rendered = torch.randn(2, 3, 3_000, requires_grad=True)
    target = torch.randn(2, 3, 3_000, requires_grad=True)

    score = _multichannel()(rendered, target)
    score.sum().backward()

    assert torch.isfinite(score).all()
    assert rendered.grad is not None
    assert torch.isfinite(rendered.grad).all()
    assert torch.count_nonzero(rendered.grad) > 0
    assert target.grad is None


def test_multichannel_silence_has_finite_gradients() -> None:
    """Silence floors remain finite through both envelope scales and pair transforms."""
    rendered = torch.zeros(1, 2, 3_000, requires_grad=True)

    score = _multichannel()(rendered, torch.zeros_like(rendered))
    score.sum().backward()

    assert torch.isfinite(score).all()
    assert rendered.grad is not None
    assert torch.isfinite(rendered.grad).all()


class _ReshapeEncoder(torch.nn.Module):
    """Frozen fake encoder exposing waveform samples as a latent grid."""

    def __init__(self, latent_dim: int) -> None:
        """Fix the latent width the waveform is folded into.

        :param latent_dim: Number of latent channels.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.gain = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Fold each waveform into a latent grid without changing its values.

        :param audio: Waveform batch shaped ``(batch, samples)``.
        :returns: Latents shaped ``(batch, latent_dim, samples / latent_dim)``.
        """
        return (audio * self.gain).reshape(audio.shape[0], self.latent_dim, -1)


def _latent_mse() -> LatentMseDistance:
    """Build a latent distance over the value-preserving fake encoder.

    :returns: Configured distance module.
    """
    return LatentMseDistance(encoder=_ReshapeEncoder(4))


def test_latent_distance_of_identical_audio_is_zero() -> None:
    """A perfect render is the fixed point the term drives toward."""
    torch.manual_seed(0)
    audio = torch.randn(3, _LENGTH).clamp(-1.0, 1.0)

    assert _latent_mse()(audio, audio).abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_latent_distance_separates_different_audio_per_sample() -> None:
    """The caller weights each row by its own flow time, so rows must not merge."""
    torch.manual_seed(0)
    rendered = torch.randn(4, _LENGTH).clamp(-1.0, 1.0)
    target = torch.randn(4, _LENGTH).clamp(-1.0, 1.0)

    distance = _latent_mse()(rendered, target)

    assert distance.shape == (4,)
    assert (distance > 0).all()


def test_latent_distance_is_invariant_to_target_magnitude() -> None:
    """Normalization is what stops loud targets from dominating quiet ones."""
    torch.manual_seed(0)
    target = torch.randn(2, _LENGTH).clamp(-1.0, 1.0)
    rendered = 0.5 * target

    quiet = _latent_mse()(rendered, target)
    loud = _latent_mse()(10.0 * rendered, 10.0 * target)

    torch.testing.assert_close(loud, quiet, rtol=1e-4, atol=1e-6)


def test_latent_distance_gradient_reaches_only_the_rendered_waveform() -> None:
    """The target is data; a gradient into it would train the wrong tensor."""
    torch.manual_seed(0)
    rendered = torch.randn(2, _LENGTH).clamp(-1.0, 1.0).requires_grad_(True)
    target = torch.randn(2, _LENGTH).clamp(-1.0, 1.0).requires_grad_(True)

    _latent_mse()(rendered, target).sum().backward()

    assert rendered.grad is not None
    assert torch.count_nonzero(rendered.grad) > 0
    assert target.grad is None


def test_latent_distance_of_a_constant_target_stays_finite() -> None:
    """Silent targets have zero latent variance and must not divide by it."""
    rendered = torch.full((2, _LENGTH), 0.25, requires_grad=True)

    distance = _latent_mse()(rendered, torch.zeros(2, _LENGTH))
    distance.sum().backward()

    assert torch.isfinite(distance).all()
    assert rendered.grad is not None
    assert torch.isfinite(rendered.grad).all()


def test_latent_distance_rejects_a_trainable_encoder() -> None:
    """A trainable space could shrink the distance without improving the render."""
    with pytest.raises(ValueError, match="frozen"):
        LatentMseDistance(encoder=_ReshapeEncoder(4).requires_grad_(True))
