"""Training consumers retain distinct waveform channels."""

from functools import partial

import numpy as np
import pytest
import torch
from pyFDN import FDNBuild
from torch import nn

from synth_setter.data.basic_fdn import BasicFDN
from synth_setter.models.components.audio_distance import MultichannelAudioDistance
from synth_setter.models.components.differentiable_renderer import FlamoFDNDifferentiableRenderer
from synth_setter.models.components.vector_field import VectorField
from synth_setter.models.vst_flow_finetune_module import ControlMode, VSTFlowFinetuneModule

_SAMPLES = 4096
_RATE = 16_000


class _StereoGainDecoder(nn.Module):
    """Four network coordinates control two input gains and two direct gains."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("A", torch.tensor([[0.2, 0.1], [-0.1, 0.2]]))
        self.register_buffer("C", torch.tensor([[1.0, 0.2], [-0.3, 0.8]]))
        self.register_buffer("delays", torch.tensor([17.0, 29.0]))

    def decode_build_fields(self, params: torch.Tensor, *, sample_rate: float) -> dict:
        """Supply all fields of a filter-free stereo-output build.

        :param params: Four independent gain values.
        :param sample_rate: Build rate; gains do not depend on it.
        :returns: Complete tensor field mapping.
        """
        return {**dict(self.named_buffers()), "B": params[:2, None], "D": params[2:, None]}


class _ParameterlessRenderer(nn.Module):
    """Render stereo audio using the floating-point state owned by the backend."""

    def __init__(self, device: torch.device) -> None:
        """Store a float64 renderer reference on the device under test.

        :param device: Device owning the renderer buffer.
        """
        super().__init__()
        self.reference: torch.Tensor
        self.register_buffer("reference", torch.zeros((), dtype=torch.float64, device=device))

    def validate(self, params: torch.Tensor) -> None:
        """Accept finite rows matching the renderer's geometry and tensor state.

        :param params: Candidate model-space rows.
        :raises ValueError: Rows do not have the required geometry, device, or dtype.
        """
        if params.ndim != 2 or params.shape[1] != 2 or not torch.isfinite(params).all():
            raise ValueError("params must be finite two-coordinate rows")
        if params.device != self.reference.device or params.dtype != self.reference.dtype:
            raise ValueError("params must match renderer tensor state")

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Expand each coordinate into its own constant waveform channel.

        :param params: Model-space rows shaped ``(batch, 2)``.
        :returns: Stereo audio shaped ``(batch, 2, samples)``.
        """
        self.validate(params)
        return (params + self.reference).unsqueeze(-1).expand(-1, -1, _SAMPLES)


class _ParameterlessFlow(nn.Module):
    """Minimal frozen field accepted by the finetune constructor."""

    def __init__(self) -> None:
        super().__init__()
        self.d_model = 1

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, conditioning: torch.Tensor | None
    ) -> torch.Tensor:
        """Return a zero velocity with the requested parameter geometry.

        :param x_t: Parameter trajectory rows.
        :param t: Flow times.
        :param conditioning: Optional conditioning ignored by this field.
        :returns: Zero velocity shaped like ``x_t``.
        """
        del t, conditioning
        return torch.zeros_like(x_t)


@pytest.mark.parametrize(
    "device", [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
)
def test_finetune_width_probe_uses_parameterless_renderer_tensor_state(
    device: torch.device,
) -> None:
    """Learned-control probing follows a parameterless renderer's device and float64 buffer.

    :param device: Available device owning the renderer state.
    """
    model = VSTFlowFinetuneModule(
        encoder=nn.Flatten(start_dim=1),
        vector_field=_ParameterlessFlow(),
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        scheduler=None,
        base_checkpoint=None,
        num_params=2,
        sample_rate=_RATE,
        signal_length=_SAMPLES,
        render_batch_size=1,
        control_mode="learned_audio",
        control_encoder=nn.AdaptiveAvgPool1d(1),
        renderer=_ParameterlessRenderer(device),
        conditioning="audio",
    )

    assert model.control_dim == 2


@pytest.mark.parametrize("mode", ["gradient_spectral", "learned_audio"])
def test_finetune_real_stereo_renderer_preserves_observation_and_control(
    mode: ControlMode,
) -> None:
    """Both simulator-control arms consume stereo without truncating or downmixing.

    :param mode: Production simulator-control arm.
    """
    decoder = _StereoGainDecoder()
    fields = dict(decoder.named_buffers())
    build = FDNBuild(
        A=fields["A"].numpy(),
        B=np.ones((2, 1)),
        C=fields["C"].numpy(),
        D=np.zeros((2, 1)),
        delays=np.array([17, 29]),
        fs=float(_RATE),
    )
    renderer = FlamoFDNDifferentiableRenderer(
        fdn=BasicFDN(build),
        decoder=decoder,
        parameter_width=4,
        signal_length=_SAMPLES,
    )
    cost = MultichannelAudioDistance(
        sample_rate=_RATE,
        spectral_weight=1.0,
        channel_mldr_weight=0.0,
        pair_mldr_weight=0.0,
    )
    model = VSTFlowFinetuneModule(
        base_checkpoint=None,
        control_mode=mode,
        control_t_min=0.0,
        num_params=4,
        sample_rate=_RATE,
        signal_length=_SAMPLES,
        render_batch_size=1,
        renderer=renderer,
        cost=cost if mode == "gradient_spectral" else None,
        control_encoder=nn.Sequential(nn.Flatten(1), nn.Linear(2 * _SAMPLES, 5))
        if mode == "learned_audio"
        else None,
        encoder=nn.Sequential(nn.Flatten(1), nn.Linear(2 * _SAMPLES, 8)),
        vector_field=VectorField(field_dim=4, hidden_dim=8, conditioning_dim=8, num_blocks=1),
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        scheduler=None,
        conditioning="audio",
    )
    rows = torch.tensor([[0.2, 0.4, 0.1, 0.3]])
    target = renderer(rows).detach()
    changed = target.clone()
    changed[:, 1] = 0.0
    model.on_validation_batch_start({"audio": target}, 0)

    assert model._sampling_target is not None
    torch.testing.assert_close(model._sampling_target, target)
    baseline = model._score(rows, target)
    actual = model._score(rows, changed)

    assert actual.shape == (1, 5)
    assert torch.isfinite(actual).all()
    assert not torch.allclose(actual, baseline)
    if mode == "learned_audio":
        actual.sum().backward()
        assert model.control_encoder is not None
        assert any(parameter.grad is not None for parameter in model.control_encoder.parameters())

    model.zero_grad(set_to_none=True)
    step = model._train_step(
        {"params": rows, "noise": torch.full_like(rows, -0.3), "audio": changed}
    )
    assert torch.isfinite(step.loss)
    step.loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.vector_field.control.parameters()
    )
