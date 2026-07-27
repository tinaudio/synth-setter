"""Shared model, data, and sampling pieces for the audio-loss flow spike.

Model space is ``[-1, 1]`` (stored torchsynth space ``[0, 1]`` maps via
``x * 2 - 1``, mirroring ``vst_datamodule.prepare_batch``). The flow is the
repo's :class:`~synth_setter.models.components.vector_field.VectorField`
conditioned on a log-magnitude-spectrum encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from synth_setter.data.torchsynth_datamodule import render_torchsynth
from synth_setter.data.vst.torchsynth_param_spec import NUM_PARAMS
from synth_setter.models.components.vector_field import VectorField

SAMPLE_RATE = 44_100
SIGNAL_LENGTH = 4_410
MIDI_PITCH = 60


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


@dataclass
class SampleConfig:
    """RK4-ODE sampling configuration.

    .. attribute :: steps

       RK4 steps from t=0 to t=1 (mirrors ``validation_sample_steps``).
    """

    steps: int = 50


@torch.no_grad()
def sample_ode(
    encoder: SpectrumEncoder,
    vector_field: VectorField,
    target_audio: torch.Tensor,
    noise: torch.Tensor,
    config: SampleConfig | None = None,
) -> torch.Tensor:
    """Integrate the flow with fixed-step RK4.

    Mirrors ``VSTFlowMatchingModule._sample`` but with the conditional field
    only — no CFG (cfg_strength equivalent 1.0), a deliberate spike decision.

    :param encoder: Audio conditioning encoder.
    :param vector_field: Flow to integrate (evaluated without grad).
    :param target_audio: Observed audio conditioning the flow.
    :param noise: Initial state x0 shaped ``(batch, NUM_PARAMS)``.
    :param config: Step count; defaults to :class:`SampleConfig`.
    :returns: Predicted params in model space ``[-1, 1]``.
    """
    if config is None:
        config = SampleConfig()
    conditioning = encoder(target_audio)

    def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return vector_field(x, t, conditioning)

    x = noise
    dt = 1.0 / config.steps
    for step in range(config.steps):
        t = torch.full((x.shape[0], 1), step * dt, device=x.device)
        k1 = velocity(x, t)
        k2 = velocity(x + dt * k1 / 2, t + dt / 2)
        k3 = velocity(x + dt * k2 / 2, t + dt / 2)
        k4 = velocity(x + dt * k3, t + dt)
        x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
