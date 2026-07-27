"""Shared model, data, and sampling pieces for the simulator-feedback spike (#2553).

Model space is ``[-1, 1]`` (stored torchsynth space ``[0, 1]`` maps via
``x * 2 - 1``, mirroring ``vst_datamodule.prepare_batch``). The base flow is
the repo's :class:`~synth_setter.models.components.vector_field.VectorField`
conditioned on a log-magnitude-spectrum encoder; the control field follows
arXiv 2410.22573 (zero-init output, active only for ``t >= 0.8``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from prototypes.torchsynth_feedback.grad_render import (
    log_spectral_distance,
    render_torchsynth_grad,
)
from synth_setter.data.torchsynth_datamodule import render_torchsynth
from synth_setter.data.vst.torchsynth_param_spec import NUM_PARAMS
from synth_setter.models.components.vector_field import VectorField, VectorFieldBlock

SAMPLE_RATE = 44_100
SIGNAL_LENGTH = 4_410
MIDI_PITCH = 60
CONTROL_T_MIN = 0.8
# Per-sample LSD averages ~10 dB on random patches (Step A); rescale to O(1).
COST_SCALE = 0.1


class SpectrumEncoder(nn.Module):
    """Flattened log-magnitude STFT followed by a two-layer MLP.

    Time-frequency structure (not just the long-window spectrum) is needed to identify envelope and
    LFO parameters.
    """

    def __init__(
        self,
        signal_length: int = SIGNAL_LENGTH,
        output_dim: int = 128,
        n_fft: int = 512,
        hop_length: int = 128,
    ) -> None:
        """Build the MLP over the flattened spectrogram.

        :param signal_length: Audio samples per row (fixes the frame count).
        :param output_dim: Conditioning width consumed by the vector field.
        :param n_fft: STFT window size.
        :param hop_length: STFT hop size.
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        num_frames = signal_length // hop_length + 1
        num_bins = n_fft // 2 + 1
        self.register_buffer("window", torch.hann_window(n_fft))
        self.net = nn.Sequential(
            nn.Linear(num_bins * num_frames, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio into a conditioning vector.

        :param audio: Audio batch shaped ``(batch, samples)``.
        :returns: Conditioning shaped ``(batch, output_dim)``.
        """
        spectrogram = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        ).abs()
        log_spectrogram = torch.log10(spectrogram.clamp_min(1e-8))
        return self.net(log_spectrogram.flatten(start_dim=1))


class ControlField(nn.Module):
    """Control network v^C(t, v, c) with a zero-init output layer.

    Zero init makes ``v + v^C`` start as the frozen base flow, so finetuning
    can only improve on it.
    """

    def __init__(self, num_params: int = NUM_PARAMS, hidden_dim: int = 128) -> None:
        """Build the residual MLP over ``[t, v, c]``.

        :param num_params: Parameter-vector width of the flow.
        :param hidden_dim: Hidden width of the control MLP.
        """
        super().__init__()
        # Input: t (1) + base velocity (num_params) + cost (1) + cost gradient (num_params).
        input_dim = 2 * num_params + 2
        self.input = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([VectorFieldBlock(hidden_dim), VectorFieldBlock(hidden_dim)])
        self.output = nn.Linear(hidden_dim, num_params)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, t: torch.Tensor, velocity: torch.Tensor, control: torch.Tensor
    ) -> torch.Tensor:
        """Predict the velocity correction.

        :param t: Flow time shaped ``(batch, 1)``.
        :param velocity: Base-flow velocity shaped ``(batch, num_params)``.
        :param control: Simulator-feedback signal ``[C; grad C]`` shaped
            ``(batch, num_params + 1)``.
        :returns: Velocity correction shaped ``(batch, num_params)``.
        """
        y = self.input(torch.cat((t, velocity, control), dim=-1))
        for block in self.blocks:
            y = block(y)
        return self.output(y)


def build_base_flow(device: str) -> tuple[SpectrumEncoder, VectorField]:
    """Construct the spike's encoder and base vector field.

    :param device: Torch device string.
    :returns: Encoder and vector field, moved to ``device``.
    """
    encoder = SpectrumEncoder().to(device)
    vector_field = VectorField(
        field_dim=NUM_PARAMS, hidden_dim=256, conditioning_dim=128, num_blocks=4
    ).to(device)
    return encoder, vector_field


def sample_batch(
    batch_size: int, device: str, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw random torchsynth params and render their audio (no grad).

    :param batch_size: Rows to draw; keep fixed per loop (renderer cache #1820).
    :param device: Torch device string.
    :param generator: CPU RNG driving the parameter draw.
    :returns: Params in model space ``[-1, 1]`` shaped ``(batch, NUM_PARAMS)``
        and audio shaped ``(batch, SIGNAL_LENGTH)``, both on ``device``.
    """
    params01 = torch.rand((batch_size, NUM_PARAMS), generator=generator).to(device)
    with torch.no_grad():
        audio = render_torchsynth(
            params01,
            sample_rate=SAMPLE_RATE,
            signal_length=SIGNAL_LENGTH,
            midi_pitch=MIDI_PITCH,
        )
    return params01 * 2 - 1, audio


def control_signal(
    x_t: torch.Tensor,
    t: torch.Tensor,
    base_velocity: torch.Tensor,
    target_audio: torch.Tensor,
) -> torch.Tensor:
    """Render the one-step estimate and return ``[C; grad C]`` (detached).

    The gradient is normalized to unit RMS per sample: Step A showed raw
    magnitudes span ~9 orders of magnitude across parameters.

    :param x_t: Current flow state in model space, shaped ``(batch, NUM_PARAMS)``.
    :param t: Flow time shaped ``(batch, 1)``.
    :param base_velocity: Frozen base-flow velocity at ``(x_t, t)``.
    :param target_audio: Observed audio shaped ``(batch, SIGNAL_LENGTH)``.
    :returns: Control signal shaped ``(batch, NUM_PARAMS + 1)``.
    """
    theta_hat = (x_t + (1 - t) * base_velocity).detach().requires_grad_(True)
    with torch.enable_grad():
        params01 = (theta_hat + 1) / 2
        audio = render_torchsynth_grad(
            params01,
            sample_rate=SAMPLE_RATE,
            signal_length=SIGNAL_LENGTH,
            midi_pitch=MIDI_PITCH,
        )
        cost = log_spectral_distance(audio, target_audio)
        (grad,) = torch.autograd.grad(cost.sum(), theta_hat)
    rms = grad.norm(dim=-1, keepdim=True) / math.sqrt(grad.shape[-1])
    grad = grad / (rms + 1e-6)
    return torch.cat((cost.unsqueeze(-1) * COST_SCALE, grad), dim=-1).detach()


@dataclass
class SampleConfig:
    """Euler-ODE sampling configuration.

    .. attribute :: steps

       Euler steps from t=0 to t=1.

    .. attribute :: feedback

       Whether the control field receives the simulator signal (else zeros,
       the capacity-matched ablation).
    """

    steps: int = 40
    feedback: bool = True


@torch.no_grad()
def sample_ode(
    encoder: SpectrumEncoder,
    vector_field: VectorField,
    target_audio: torch.Tensor,
    noise: torch.Tensor,
    control_field: ControlField | None = None,
    config: SampleConfig | None = None,
) -> torch.Tensor:
    """Integrate the flow with Euler steps, applying the control for t >= 0.8.

    No CFG (conditional field only, cfg_strength equivalent 1.0) — the spike
    sidesteps guidance interactions.

    :param encoder: Audio conditioning encoder.
    :param vector_field: Base flow (frozen or not; evaluated without grad).
    :param target_audio: Observed audio conditioning the flow.
    :param noise: Initial state x0 shaped ``(batch, NUM_PARAMS)``.
    :param control_field: Optional control network v^C.
    :param config: Steps and feedback toggle; defaults to :class:`SampleConfig`.
    :returns: Predicted params in model space ``[-1, 1]``.
    """
    if config is None:
        config = SampleConfig()
    conditioning = encoder(target_audio)
    x = noise
    dt = 1.0 / config.steps
    for step in range(config.steps):
        t = torch.full((x.shape[0], 1), step * dt, device=x.device)
        velocity = vector_field(x, t, conditioning)
        if control_field is not None and float(t[0, 0]) >= CONTROL_T_MIN:
            if config.feedback:
                signal = control_signal(x, t, velocity, target_audio)
            else:
                signal = torch.zeros((x.shape[0], NUM_PARAMS + 1), device=x.device, dtype=x.dtype)
            velocity = velocity + control_field(t, velocity, signal)
        x = x + dt * velocity
    return x
