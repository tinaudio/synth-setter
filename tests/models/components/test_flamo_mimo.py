"""Real multichannel FDN responses retain every input/output transfer path."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from pyFDN import FDNBuild
from torch import nn

from synth_setter.data.basic_fdn import BasicFDN
from synth_setter.models.components.differentiable_renderer import FlamoFDNDifferentiableRenderer


class _GainDecoder(nn.Module):
    """Predict each input and direct gain; keep the remaining build fields fixed."""

    def __init__(self, build: FDNBuild) -> None:
        """Retain fixed native fields as movable module buffers.

        :param build: Complete source build.
        """
        super().__init__()
        for name in ("A", "C", "delays", "post_delay", "post_matrix", "post_output"):
            value = getattr(build, name)
            if value is not None:
                self.register_buffer(name, torch.as_tensor(value, dtype=torch.float64))

    def decode_build_fields(self, params: torch.Tensor, *, sample_rate: float) -> dict:
        """Bind all ten independent gain coordinates to their native matrices.

        :param params: Four input gains followed by six direct gains.
        :param sample_rate: Build sample rate; these gains do not depend on it.
        :returns: Complete tensor build fields without sample-rate metadata.
        """
        return {
            **dict(self.named_buffers()),
            "B": params[:4].reshape(2, 2),
            "D": params[4:].reshape(3, 2),
        }


@pytest.fixture(
    params=[
        (),
        ("post_delay",),
        ("post_matrix",),
        ("post_output",),
        ("post_delay", "post_matrix", "post_output"),
    ]
)
def build(request: pytest.FixtureRequest) -> FDNBuild:
    """Create unequal input/output dimensions and distinct hooks.

    :param request: Filter-hook combination to include.
    :returns: Stable two-input, three-output FDN with distinct optional SOS hooks.
    """

    def sos(channels: int) -> np.ndarray:
        coefficients = np.zeros((1, 6, channels))
        coefficients[:, 0, :] = np.linspace(0.6, 0.9, channels)
        coefficients[:, 3, :] = 1.0
        coefficients[:, 4, :] = -0.1
        return coefficients

    return FDNBuild(
        A=np.array([[0.2, 0.1], [-0.1, 0.2]]),
        B=np.ones((2, 2)),
        C=np.array([[1.0, 0.2], [0.3, 0.8], [-0.4, 0.5]]),
        D=np.zeros((3, 2)),
        delays=np.array([17, 29]),
        fs=48_000.0,
        post_delay=sos(2) if "post_delay" in request.param else None,
        post_matrix=sos(2) if "post_matrix" in request.param else None,
        post_output=sos(3) if "post_output" in request.param else None,
    )


def test_flamo_mimo_predictions_preserve_every_transfer_path(build: FDNBuild) -> None:
    """Each input impulse is rendered separately, without mixing or selecting output zero.

    :param build: Complete MIMO template.
    """
    renderer = FlamoFDNDifferentiableRenderer(
        fdn=BasicFDN(build),
        decoder=_GainDecoder(build),
        parameter_width=10,
        signal_length=512,
        fft_size=4096,
        dtype=torch.float64,
    )
    rows = torch.linspace(0.1, 0.9, 20, dtype=torch.float64).reshape(2, 10).requires_grad_()

    actual = renderer(rows)

    assert actual.shape == (2, 6, 512)
    for index, row in enumerate(rows.detach().numpy()):
        native = replace(build, B=row[:4].reshape(2, 2), D=row[4:].reshape(3, 2))
        expected = BasicFDN(native).impulse_response(512).transpose(1, 2, 0).reshape(6, 512)
        np.testing.assert_allclose(actual[index].detach().numpy(), expected, atol=1e-9, rtol=1e-7)


def test_flamo_mimo_nonfirst_output_backpropagates_to_its_direct_input(build: FDNBuild) -> None:
    """The last output/input path is differentiable independently of other rows and gains.

    :param build: Complete MIMO template.
    """
    renderer = FlamoFDNDifferentiableRenderer(
        fdn=BasicFDN(build),
        decoder=_GainDecoder(build),
        parameter_width=10,
        signal_length=512,
        fft_size=4096,
        dtype=torch.float64,
    )
    rows = torch.linspace(0.1, 0.9, 20, dtype=torch.float64).reshape(2, 10).requires_grad_()

    gradient = torch.autograd.grad(renderer(rows)[0, 5, 0], rows)[0]

    torch.testing.assert_close(gradient[0, 9], torch.tensor(1.0, dtype=torch.float64))
    assert torch.count_nonzero(gradient[1]) == 0
    torch.testing.assert_close(gradient[0, 4:9], torch.zeros(5, dtype=torch.float64))
