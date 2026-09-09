"""Behavior tests for tensor-native differentiable renderers."""

import numpy as np
import pytest
import torch

from synth_setter.data.pyfdn_instrument import PyFDNRenderer
from synth_setter.data.pyfdn_param_spec import PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC
from synth_setter.data.vst.param_spec import ParameterValues
from synth_setter.models.components.differentiable_renderer import (
    FlamoFDNDifferentiableRenderer,
)


def _model_row(seed: int) -> tuple[torch.Tensor, ParameterValues]:
    spec = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC
    native, note = spec.sample(np.random.default_rng(seed))
    encoded = spec.encode(native, note)
    model = torch.tensor(encoded * 2.0 - 1.0, dtype=torch.float32)
    return model, native


def test_flamo_householder_batch_matches_pyfdn_geometry_and_amplitude() -> None:
    """Independent parameter rows preserve pyFDN's timing and unnormalized amplitude."""
    first, first_native = _model_row(2)
    second, second_native = _model_row(3)
    renderer = FlamoFDNDifferentiableRenderer(
        param_spec="pyfdn_n8_mono_householder",
        sample_rate=44_100,
        signal_length=176_400,
        fft_size=524_288,
    )

    actual = renderer(torch.stack((first, second))).detach().numpy()
    expected = np.stack(
        (
            PyFDNRenderer().render(first_native)[0],
            PyFDNRenderer().render(second_native)[0],
        )
    )

    assert actual.shape == (2, 176_400)
    assert np.sqrt(np.mean((actual - expected) ** 2)) < 1.1e-3
    assert np.max(np.abs(actual - expected)) < 1.4e-2


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_flamo_householder_supported_controls_have_finite_nonzero_gradients(device: str) -> None:
    """B/C/D and RT controls carry gradients, while rounded delays do not.

    :param device: Device hosting both renderer and input rows.
    """
    row, _ = _model_row(4)
    row = row.to(device).requires_grad_(True)
    renderer = FlamoFDNDifferentiableRenderer(
        param_spec="pyfdn_n8_mono_householder",
        sample_rate=44_100,
        signal_length=4096,
        fft_size=8192,
    ).to(device)

    renderer(row.unsqueeze(0)).square().mean().backward()

    assert row.grad is not None
    supported = row.grad[8:]
    assert torch.isfinite(supported).all()
    assert torch.count_nonzero(supported).item() == supported.numel()
    assert torch.count_nonzero(row.grad[:8]).item() == 0


def test_flamo_audio_feedback_unclipped_target_has_zero_loss() -> None:
    """FLAMO feedback must not inherit TorchSynth's stored-audio clipping."""
    from synth_setter.models.components.audio_distance import MultiScaleSpectralDistance
    from synth_setter.models.components.audio_feedback import AudioFeedbackLoss

    row = torch.zeros(1, 27)
    row[:, 8:25] = 1.0
    renderer = FlamoFDNDifferentiableRenderer(
        param_spec="pyfdn_n8_mono_householder", sample_rate=44_100, signal_length=4096
    )
    target = renderer(row).detach()
    assert target.abs().max() > 1.0
    loss = AudioFeedbackLoss(
        lambda_audio=1.0,
        t_min=0.0,
        sample_rate=44_100,
        signal_length=4096,
        render_batch_size=1,
        distance=MultiScaleSpectralDistance(sample_rate=44_100),
        renderer=renderer,
    )

    assert loss(row, torch.ones(1, 1), target).item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("width", [26, 28])
def test_flamo_renderer_wrong_parameter_width_raises(width: int) -> None:
    """Reject extra or missing coordinates before entering FLAMO.

    :param width: Encoded width incompatible with the supported spec.
    """
    renderer = FlamoFDNDifferentiableRenderer(
        param_spec="pyfdn_n8_mono_householder", sample_rate=44_100, signal_length=4096
    )
    with pytest.raises(ValueError, match="width 27"):
        renderer(torch.zeros(1, width))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_flamo_renderer_nonfinite_parameters_raise(value: float) -> None:
    """Fail before unstable values reach the recursive DSP solve.

    :param value: Nonfinite model output.
    """
    renderer = FlamoFDNDifferentiableRenderer(
        param_spec="pyfdn_n8_mono_householder", sample_rate=44_100, signal_length=4096
    )
    with pytest.raises(ValueError, match="finite"):
        renderer(torch.full((1, 27), value))


def test_flamo_renderer_empty_batch_preserves_audio_geometry() -> None:
    """No rows still produces a batch-first waveform tensor."""
    renderer = FlamoFDNDifferentiableRenderer(
        param_spec="pyfdn_n8_mono_householder", sample_rate=44_100, signal_length=4096
    )
    assert renderer(torch.empty(0, 27)).shape == (0, 4096)


def test_flamo_renderer_rejects_unsupported_topology() -> None:
    """A different pyFDN spec must not silently use the fixed-Householder mapping."""
    with pytest.raises(ValueError, match="supports only pyfdn_n8_mono_householder"):
        FlamoFDNDifferentiableRenderer(
            param_spec="pyfdn_n8_mono_kronecker",
            sample_rate=44_100,
            signal_length=4096,
        )
