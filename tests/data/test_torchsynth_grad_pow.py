"""Regression tests for the tensor-exponent pow singularity in torchsynth's ADSR ramp."""

import pytest
import torch
from torchsynth.config import SynthConfig
from torchsynth.module import ADSR

from synth_setter.data.torchsynth_grad_render import finite_tensor_exponent_pow

_BATCH = 2


def _adsr() -> ADSR:
    """Build a tiny batched ADSR whose ramp saturates inside the buffer.

    :returns: ADSR module on CPU.
    """
    config = SynthConfig(
        batch_size=_BATCH, sample_rate=16000, buffer_size_seconds=0.25, reproducible=False
    )
    adsr = ADSR(config)
    # ``forward`` caches alpha as (batch, 1); ``ramp`` is exercised directly here.
    adsr.alpha = torch.full((_BATCH, 1), 0.5)
    return adsr


def test_zero_base_with_fractional_exponent_is_non_finite_without_the_guard() -> None:
    """Pins the upstream defect: an unweighted zero-base element yields ``0 * inf``."""
    base = torch.tensor([0.0, 0.5], requires_grad=True)

    torch.pow(base, torch.tensor([0.5])).backward(torch.tensor([0.0, 1.0]))

    assert base.grad is not None
    assert not torch.isfinite(base.grad).all()


def test_zero_base_with_fractional_exponent_is_finite_under_the_guard() -> None:
    """The guard removes the singularity the audio-feedback backward walks into."""
    base = torch.tensor([0.0, 0.5], requires_grad=True)

    with finite_tensor_exponent_pow():
        torch.pow(base, torch.tensor([0.5])).backward(torch.tensor([0.0, 1.0]))

    assert base.grad is not None
    assert torch.isfinite(base.grad).all()


def test_saturated_inverse_ramp_reaches_the_singular_base() -> None:
    """Ties the primitive defect to the real ADSR path that produces a zero base."""
    ramp = _adsr().ramp(torch.full((_BATCH,), 0.01), inverse=True)

    assert (ramp == 0.0).any()


def test_guard_leaves_the_ramp_bitwise_unchanged() -> None:
    """Stored datasets were rendered without the guard, so the forward must not move."""
    duration = torch.full((_BATCH,), 0.01)

    with torch.no_grad():
        unguarded = _adsr().ramp(duration, inverse=True).clone()
        with finite_tensor_exponent_pow():
            guarded = _adsr().ramp(duration, inverse=True).clone()

    assert torch.equal(guarded, unguarded)


def test_guard_is_removed_on_exit() -> None:
    """A leaked patch would silently change every later render in the process."""
    original = torch.pow

    with finite_tensor_exponent_pow():
        pass

    assert torch.pow is original


def test_scalar_exponent_pow_is_untouched() -> None:
    """Only a tensor exponent reaches the singular derivative; squares must stay exact."""
    values = torch.tensor([0.0, 0.5, 2.0], requires_grad=True)

    with finite_tensor_exponent_pow():
        torch.pow(values, 2.0).sum().backward()

    assert values.grad is not None
    assert values.grad.tolist() == pytest.approx([0.0, 1.0, 4.0])
