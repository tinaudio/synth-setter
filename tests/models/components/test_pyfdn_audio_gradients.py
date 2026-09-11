"""Continuous-control and batch-isolation contracts through the real FLAMO graph."""

import numpy as np
import pytest
import torch

from synth_setter.data.vst.param_spec import DiscreteArrayParameter
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.models.components.differentiable_renderer import FlamoFDNDifferentiableRenderer
from synth_setter.param_spec_name import ParamSpecName


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    "param_spec",
    [
        "pyfdn_n8_mono_householder",
        "pyfdn_n8_mono_householder_vector",
        "pyfdn_n8_mono_kronecker",
    ],
)
def test_audio_gradient_continuous_controls_affect_only_their_own_row(
    param_spec: str, device: str
) -> None:
    """Audio depends on continuous controls without leaking across per-patch graphs.

    :param param_spec: Feedback and attenuation topology to exercise.
    :param device: Device hosting the prediction and FLAMO solve.
    """
    renderer = FlamoFDNDifferentiableRenderer.from_param_spec(
        param_spec=param_spec, sample_rate=44_100, signal_length=4096, fft_size=8192
    ).to(device)
    spec = resolve_param_spec(ParamSpecName(param_spec))
    rows = torch.tensor(
        np.random.default_rng(19).uniform(-0.8, 0.8, (2, spec.encoded_width)),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    audio = renderer(rows)
    audio[0].square().mean().backward()

    assert rows.grad is not None
    assert torch.isfinite(rows.grad).all()
    assert torch.count_nonzero(rows.grad[1]) == 0
    for parameter, span in spec.encoded_slices():
        gradient = rows.grad[0, span]
        if isinstance(parameter, DiscreteArrayParameter):
            assert torch.count_nonzero(gradient) == 0, parameter.name
        else:
            assert torch.count_nonzero(gradient) > 0, parameter.name
