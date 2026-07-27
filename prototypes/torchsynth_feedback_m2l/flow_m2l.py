"""Shared model, data, and sampling pieces for the m2l simulator-feedback spike (#2557).

Adapted from the #2553 mel-cell spike's ``flow.py``: same base flow
(:class:`~synth_setter.models.components.vector_field.VectorField` conditioned
on a log-magnitude-spectrum encoder) and control-field recipe (arXiv
2410.22573: zero-init output, active only for ``t >= 0.8``), with the
simulator-feedback cost moved into music2latent latent space. Model space is
``[-1, 1]`` (stored torchsynth space ``[0, 1]`` maps via ``x * 2 - 1``,
mirroring ``vst_datamodule.prepare_batch``). Geometry is 0.5 s at 44.1 kHz —
m2l emits one latent frame per 4,096 samples, so the mel cell's 0.1 s clips
would encode to zero frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torchaudio
from torch import nn

from prototypes.torchsynth_feedback_m2l.grad_render import (
    log_spectral_distance,
    render_torchsynth_grad,
)
from prototypes.torchsynth_feedback_m2l.m2l_grad import (
    M2LGradEncoder,
    m2l_framewise_mse,
    m2l_pooled_l2,
)
from synth_setter.data.torchsynth_datamodule import render_torchsynth
from synth_setter.data.vst.torchsynth_param_spec import NUM_PARAMS
from synth_setter.models.components.vector_field import VectorField, VectorFieldBlock

SAMPLE_RATE = 44_100
SIGNAL_LENGTH = 22_050
MIDI_PITCH = 60
CONTROL_T_MIN = 0.8
# Pooled-L2 m2l cost averages ~11.8 on random patch pairs (Step A); rescale to O(1).
COST_SCALE = 0.1


class SpectrumEncoder(nn.Module):
    """Flattened log-magnitude STFT followed by a small MLP.

    Time-frequency structure (not just the long-window spectrum) is needed to identify envelope and
    LFO parameters. Hop 256 (vs #2553's 128) keeps the first linear layer modest at the 5x longer
    signal.
    """

    def __init__(
        self,
        signal_length: int = SIGNAL_LENGTH,
        output_dim: int = 128,
        n_fft: int = 512,
        hop_length: int = 256,
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


class MultiScaleLogMel(nn.Module):
    """Multi-scale log-mel spectral distance (common eval metric across all four cells)."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_ffts: tuple[int, ...] = (2_048, 1_024, 512),
        n_mels: int = 64,
    ) -> None:
        """Build one mel transform per scale.

        :param sample_rate: Audio sample rate in Hz.
        :param n_ffts: STFT sizes; hop is ``n_fft // 4``.
        :param n_mels: Mel bins per scale.
        """
        super().__init__()
        self.transforms = nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate, n_fft=n_fft, hop_length=n_fft // 4, n_mels=n_mels
                )
                for n_fft in n_ffts
            ]
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean absolute log-mel difference, averaged over scales.

        :param pred: Predicted audio shaped ``(batch, samples)``.
        :param target: Target audio shaped ``(batch, samples)``.
        :returns: Per-sample distance shaped ``(batch,)``.
        """
        total = torch.zeros(pred.shape[0], device=pred.device)
        for transform in self.transforms:
            log_pred = torch.log10(transform(pred).clamp_min(1e-7))
            log_target = torch.log10(transform(target).clamp_min(1e-7))
            total = total + (log_pred - log_target).abs().mean(dim=(-2, -1))
        return total / len(self.transforms)


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
    target_latents: torch.Tensor,
    m2l: M2LGradEncoder,
    framewise: bool = False,
    cost_scale: float = COST_SCALE,
) -> torch.Tensor:
    """Render the one-step estimate and return the m2l-space ``[C; grad C]`` (detached).

    The gradient is normalized to unit RMS per sample: Step A showed raw
    magnitudes span ~9 orders of magnitude across parameters.

    :param x_t: Current flow state in model space, shaped ``(batch, NUM_PARAMS)``.
    :param t: Flow time shaped ``(batch, 1)``.
    :param base_velocity: Frozen base-flow velocity at ``(x_t, t)``.
    :param target_latents: m2l latents of the observed audio.
    :param m2l: Grad-enabled m2l encoder.
    :param framewise: Use the unpooled framewise MSE cost instead of pooled L2.
    :param cost_scale: Rescale factor bringing the cost feature to O(1).
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
        distance = m2l_framewise_mse if framewise else m2l_pooled_l2
        cost = distance(m2l(audio), target_latents)
        (grad,) = torch.autograd.grad(cost.sum(), theta_hat)
    rms = grad.norm(dim=-1, keepdim=True) / math.sqrt(grad.shape[-1])
    grad = grad / (rms + 1e-6)
    return torch.cat((cost.unsqueeze(-1) * cost_scale, grad), dim=-1).detach()


@dataclass
class SampleConfig:
    """Euler-ODE sampling configuration.

    .. attribute :: steps

       Euler steps from t=0 to t=1.

    .. attribute :: feedback

       Whether the control field receives the simulator signal (else zeros,
       the capacity-matched ablation).

    .. attribute :: framewise

       Use the unpooled framewise m2l cost for the control signal.

    .. attribute :: cost_scale

       Rescale factor bringing the cost feature to O(1).
    """

    steps: int = 40
    feedback: bool = True
    framewise: bool = False
    cost_scale: float = COST_SCALE


@torch.no_grad()
def sample_ode(
    encoder: SpectrumEncoder,
    vector_field: VectorField,
    target_audio: torch.Tensor,
    noise: torch.Tensor,
    control_field: ControlField | None = None,
    m2l: M2LGradEncoder | None = None,
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
    :param m2l: Grad-enabled m2l encoder; required when ``config.feedback``.
    :param config: Steps and feedback toggle; defaults to :class:`SampleConfig`.
    :returns: Predicted params in model space ``[-1, 1]``.
    :raises ValueError: Feedback sampling requested without an m2l encoder.
    """
    if config is None:
        config = SampleConfig()
    if config.feedback and control_field is not None and m2l is None:
        raise ValueError("feedback sampling needs the m2l encoder")
    conditioning = encoder(target_audio)
    target_latents = None
    if config.feedback and control_field is not None and m2l is not None:
        target_latents = m2l(target_audio)
    x = noise
    dt = 1.0 / config.steps
    for step in range(config.steps):
        t = torch.full((x.shape[0], 1), step * dt, device=x.device)
        velocity = vector_field(x, t, conditioning)
        if control_field is not None and float(t[0, 0]) >= CONTROL_T_MIN:
            if config.feedback and m2l is not None and target_latents is not None:
                signal = control_signal(
                    x,
                    t,
                    velocity,
                    target_latents,
                    m2l,
                    framewise=config.framewise,
                    cost_scale=config.cost_scale,
                )
            else:
                signal = torch.zeros((x.shape[0], NUM_PARAMS + 1), device=x.device, dtype=x.dtype)
            velocity = velocity + control_field(t, velocity, signal)
        x = x + dt * velocity
    return x


def eval_metrics(
    preds: torch.Tensor,
    params: torch.Tensor,
    target_audio: torch.Tensor,
    mel_metric: MultiScaleLogMel,
    m2l: M2LGradEncoder,
) -> dict[str, float]:
    """Compute the common eval protocol: param MSE, multi-scale log-mel, LSD, m2l distance.

    :param preds: Predicted params in model space shaped ``(batch, NUM_PARAMS)``.
    :param params: True params in model space shaped ``(batch, NUM_PARAMS)``.
    :param target_audio: Observed audio shaped ``(batch, SIGNAL_LENGTH)``.
    :param mel_metric: Shared multi-scale log-mel module.
    :param m2l: Grad-enabled m2l encoder (used under no_grad here).
    :returns: Mean ``param_mse``, ``mel``, ``lsd``, and ``m2l`` for the batch.
    """
    with torch.no_grad():
        pred_audio = render_torchsynth(
            ((preds + 1) / 2).clamp(0, 1),
            sample_rate=SAMPLE_RATE,
            signal_length=SIGNAL_LENGTH,
            midi_pitch=MIDI_PITCH,
        )
        return {
            "param_mse": (preds - params).square().mean().item(),
            "mel": mel_metric(pred_audio, target_audio).mean().item(),
            "lsd": log_spectral_distance(pred_audio, target_audio).mean().item(),
            "m2l": m2l_pooled_l2(m2l(pred_audio), m2l(target_audio)).mean().item(),
        }
