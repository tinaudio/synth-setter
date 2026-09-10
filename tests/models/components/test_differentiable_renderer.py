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


@pytest.mark.parametrize("field", ["sample_rate", "signal_length"])
def test_flamo_experiment_geometry_override_propagates_to_renderer(field: str) -> None:
    """Dependent components inherit the loss's configured geometry rather than stale literals.

    :param field: Shared geometry coordinate to override.
    """
    from pathlib import Path

    from omegaconf import OmegaConf

    path = (
        Path(__file__).parents[3]
        / "src/synth_setter/configs/experiment/pyfdn/flow_audio_flamo.yaml"
    )
    config = OmegaConf.load(path)
    config.model.audio_loss[field] += 1

    assert config.model.audio_loss.renderer[field] == config.model.audio_loss[field]
    if field == "sample_rate":
        assert config.model.audio_loss.distance.sample_rate == config.model.audio_loss.sample_rate


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
    renderer = FlamoFDNDifferentiableRenderer.from_param_spec(
        param_spec="pyfdn_n8_mono_householder",
        sample_rate=44_100,
        signal_length=176_400,
        fft_size=524_288,
    )

    actual = renderer(torch.stack((first, second))).detach().numpy()
    expected = np.stack(
        (
            PyFDNRenderer().render(first_native),
            PyFDNRenderer().render(second_native),
        )
    )

    assert actual.shape == (2, 1, 176_400)
    assert np.sqrt(np.mean((actual - expected) ** 2)) < 1.1e-3
    assert np.max(np.abs(actual - expected)) < 1.4e-2


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_flamo_householder_supported_controls_have_finite_nonzero_gradients(device: str) -> None:
    """B/C/D and RT controls carry gradients, while rounded delays do not.

    :param device: Device hosting both renderer and input rows.
    """
    row, _ = _model_row(4)
    row = row.to(device).requires_grad_(True)
    renderer = FlamoFDNDifferentiableRenderer.from_param_spec(
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


def test_flamo_renderer_same_dtype_conversion_preserves_output() -> None:
    """An explicit same-dtype conversion leaves the FLAMO graph executable."""
    row, _ = _model_row(5)
    renderer = FlamoFDNDifferentiableRenderer.from_param_spec(
        param_spec="pyfdn_n8_mono_householder",
        sample_rate=44_100,
        signal_length=4096,
        fft_size=8192,
    )
    expected = renderer(row.unsqueeze(0))

    renderer.to(dtype=torch.float32)

    torch.testing.assert_close(renderer(row.unsqueeze(0)), expected)


def test_flamo_audio_feedback_unclipped_target_has_zero_loss() -> None:
    """FLAMO feedback must not inherit TorchSynth's stored-audio clipping."""
    from synth_setter.models.components.audio_distance import MultiScaleSpectralDistance
    from synth_setter.models.components.audio_feedback import AudioFeedbackLoss

    row = torch.zeros(1, 27)
    row[:, 8:25] = 1.0
    renderer = FlamoFDNDifferentiableRenderer.from_param_spec(
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
    renderer = FlamoFDNDifferentiableRenderer.from_param_spec(
        param_spec="pyfdn_n8_mono_householder", sample_rate=44_100, signal_length=4096
    )
    with pytest.raises(ValueError, match="width 27"):
        renderer(torch.zeros(1, width))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_flamo_renderer_nonfinite_parameters_raise(value: float) -> None:
    """Fail before unstable values reach the recursive DSP solve.

    :param value: Nonfinite model output.
    """
    renderer = FlamoFDNDifferentiableRenderer.from_param_spec(
        param_spec="pyfdn_n8_mono_householder", sample_rate=44_100, signal_length=4096
    )
    with pytest.raises(ValueError, match="finite"):
        renderer(torch.full((1, 27), value))


def test_flamo_renderer_empty_batch_preserves_audio_geometry() -> None:
    """No rows still produces a batch-first waveform tensor."""
    renderer = FlamoFDNDifferentiableRenderer.from_param_spec(
        param_spec="pyfdn_n8_mono_householder", sample_rate=44_100, signal_length=4096
    )
    assert renderer(torch.empty(0, 27)).shape == (0, 1, 4096)


@pytest.mark.parametrize(
    "param_spec",
    [
        "pyfdn_pitchshift_n8_mono_householder",
        "pyfdn_diffvox",
        "pyfdn_gotz_n8_mono_fixed_delays",
        "pyfdn_gotz_n8_mono_learned_delays",
        "pyfdn_gotz_n8_mono_fixed_delays_givens",
        "pyfdn_gotz_n8_mono_learned_delays_givens",
    ],
)
def test_flamo_renderer_rejects_unsupported_topology(param_spec: str) -> None:
    """An advanced effect cannot silently discard processing outside its FDN build.

    :param param_spec: Advanced effect outside the complete BasicFDN contract.
    """
    with pytest.raises(ValueError, match="unsupported FLAMO topology"):
        FlamoFDNDifferentiableRenderer.from_param_spec(
            param_spec=param_spec,
            sample_rate=44_100,
            signal_length=4096,
        )


@pytest.mark.parametrize(
    "param_spec",
    [
        "pyfdn_n8_mono_householder",
        "pyfdn_n8_mono_householder_vector",
        "pyfdn_n8_mono_kronecker",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_flamo_same_prediction_offline_and_online_audio_agree(
    param_spec: str, dtype: torch.dtype
) -> None:
    """A model prediction retains its sound through either decoding/rendering path.

    :param param_spec: Registered feedback and attenuation parameterization.
    :param dtype: Precision shared by the prediction and online renderer.
    """
    from synth_setter.data.vst.param_spec import decode_model_output
    from synth_setter.data.vst.param_spec_registry import resolve_param_spec
    from synth_setter.param_spec_name import ParamSpecName

    spec = resolve_param_spec(ParamSpecName(param_spec))
    prediction = torch.tensor(
        np.random.default_rng(12).uniform(-0.9, 0.9, spec.encoded_width), dtype=dtype
    )
    native, _ = decode_model_output(prediction.numpy(), spec)
    expected = PyFDNRenderer(param_spec_name=ParamSpecName(param_spec)).render(native)[0]
    renderer = FlamoFDNDifferentiableRenderer.from_param_spec(
        param_spec=param_spec,
        sample_rate=44_100,
        signal_length=8192,
        fft_size=262_144,
    )
    renderer = renderer.double() if dtype == torch.float64 else renderer.float()

    actual = renderer(prediction.unsqueeze(0)).detach().numpy()[0, 0]

    np.testing.assert_allclose(actual, expected[:8192], atol=2e-4, rtol=2e-3)
