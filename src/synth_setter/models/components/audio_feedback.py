"""Audio-domain feedback loss that backpropagates through a differentiable render.

The flow's one-step parameter estimate is rendered with autograd connected and scored
against the target audio, so spectral error reaches the vector field's own weights. The
term is gated to late flow times: below ``t_min`` the estimate is still near noise and
its render carries no usable signal.
"""

from __future__ import annotations

from enum import StrEnum

import torch
import torchaudio
from torch import nn

from synth_setter.data.torchsynth_grad_render import (
    differentiable_decode,
    render_torchsynth_grad,
)


class AudioDistance(StrEnum):
    """Representation the audio term measures error in.

    .. attribute :: MSLM

        Multi-scale log-mel L1 on the raw waveforms.

    .. attribute :: LATENT

        MSE between a frozen encoder's embeddings of both waveforms.
    """

    MSLM = "mslm"
    LATENT = "latent"


def validate_audio_feedback_runtime(*, drop_last: bool, compiled: bool, world_size: int) -> None:
    """Reject runtime configurations the differentiable renderer cannot serve.

    Each condition fails loudly rather than degrading silently — see
    https://github.com/tinaudio/synth-setter/issues/2585.

    :param drop_last: Whether the train loader drops a trailing partial batch.
    :param compiled: Whether the module is wrapped by ``torch.compile``.
    :param world_size: Number of distributed training processes.
    :raises ValueError: Any unsupported condition holds.
    """
    if not drop_last:
        # The renderer caches per (sample_rate, signal_length, batch, device) — #1820. A
        # trailing partial batch changes the batch dim and silently misses that cache.
        raise ValueError(
            "audio feedback requires drop_last=True on the train dataloader; the torchsynth "
            "renderer is cached per batch size and a partial final batch misses the cache "
            "(see https://github.com/tinaudio/synth-setter/issues/2585)"
        )
    if compiled:
        # torch.compile traces through functional_call into torchsynth's Voice, where it
        # graph-breaks or miscompiles; wrong gradients are worse than no run.
        raise ValueError(
            "audio feedback is incompatible with torch.compile; set model.compile=false "
            "(see https://github.com/tinaudio/synth-setter/issues/2585)"
        )
    if world_size > 1:
        # The render runs inside the loss on the training device behind a process-local
        # lock; distributed training is unvalidated and must not run silently.
        raise ValueError(
            f"audio feedback is single-device only, got world_size={world_size} "
            "(see https://github.com/tinaudio/synth-setter/issues/2585)"
        )


class AudioFeedbackLoss(nn.Module):
    """Weighted audio-domain distance on the flow's rendered one-step estimate."""

    def __init__(
        self,
        lambda_audio: float,
        t_min: float,
        sample_rate: int,
        signal_length: int,
        midi_pitch: int,
        distance: AudioDistance = AudioDistance.MSLM,
        n_ffts: tuple[int, ...] = (256, 512, 1024),
        n_mels: int = 64,
        eps: float = 1e-5,
    ) -> None:
        """Configure the render geometry and the distance.

        :param lambda_audio: Audio-term weight at t=1; must be positive.
        :param t_min: Flow time at which the term switches on, in ``[0, 1)``.
        :param sample_rate: Render sample rate in Hz.
        :param signal_length: Rendered samples per row.
        :param midi_pitch: Fixed MIDI note rendered for every row.
        :param distance: Representation the error is measured in.
        :param n_ffts: STFT window sizes for the multi-scale mel distance.
        :param n_mels: Maximum mel bands per scale; capped so no band ends up empty.
        :param eps: Mel-magnitude clamp floor before the log.
        :raises ValueError: Non-positive ``lambda_audio`` or out-of-range ``t_min``.
        """
        super().__init__()
        if lambda_audio <= 0.0:
            raise ValueError(
                f"lambda_audio must be positive, got {lambda_audio}; omit the audio loss "
                "entirely for the no-render control arm"
            )
        if not 0.0 <= t_min < 1.0:
            raise ValueError(f"t_min must lie in [0, 1), got {t_min}")
        self.lambda_audio = lambda_audio
        self.t_min = t_min
        self.sample_rate = sample_rate
        self.signal_length = signal_length
        self.midi_pitch = midi_pitch
        self.distance = AudioDistance(distance)
        self.eps = eps
        self.mels = nn.ModuleList(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=n_fft // 4,
                n_mels=min(n_mels, (n_fft // 2 + 1) // 4),
                f_min=0.0,
                f_max=sample_rate / 2,
                power=1.0,
                norm="slaney",
                mel_scale="slaney",
            )
            for n_fft in n_ffts
        )

    def audio_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Ramp the weight from zero at ``t_min`` to ``lambda_audio`` at t=1.

        :param t: Flow time shaped ``(batch, 1)``.
        :returns: Per-sample weight shaped ``(batch, 1)``.
        """
        return self.lambda_audio * ((t - self.t_min) / (1 - self.t_min)).clamp(min=0.0)

    def multi_scale_log_mel(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-sample L1 between log-mel spectrograms, averaged over the STFT scales.

        :param predicted: Rendered estimate shaped ``(batch, samples)``.
        :param target: Observed audio, same shape.
        :returns: Per-sample distance shaped ``(batch,)``.
        """
        total = torch.zeros(predicted.shape[0], device=predicted.device)
        for mel in self.mels:
            pred_mel, target_mel = (
                mel(signal).clamp_min(self.eps) for signal in (predicted, target)
            )
            total = total + (pred_mel.log10() - target_mel.log10()).abs().mean(dim=(-1, -2))
        return total / len(self.mels)

    def forward(
        self,
        theta_hat: torch.Tensor,
        t: torch.Tensor,
        target_audio: torch.Tensor,
        encoder: nn.Module | None = None,
    ) -> torch.Tensor:
        """Render the estimate and return the weighted distance to the target.

        :param theta_hat: One-step parameter estimate in model space ``[-1, 1]``.
        :param t: Flow time shaped ``(batch, 1)``.
        :param target_audio: Observed audio shaped ``(batch, signal_length)``.
        :param encoder: Encoder defining the latent space; required for
            :attr:`AudioDistance.LATENT`.
        :returns: Scalar weighted audio loss.
        :raises ValueError: The latent distance is selected without an encoder.
        """
        if self.distance is AudioDistance.LATENT and encoder is None:
            raise ValueError("the latent distance needs an encoder to define the space")
        rendered = render_torchsynth_grad(
            differentiable_decode(theta_hat),
            sample_rate=self.sample_rate,
            signal_length=self.signal_length,
            midi_pitch=self.midi_pitch,
        )
        if encoder is not None and self.distance is AudioDistance.LATENT:
            distance = (encoder(rendered) - encoder(target_audio)).square().mean(dim=-1)
        else:
            distance = self.multi_scale_log_mel(rendered, target_audio)
        return (self.audio_weight(t).squeeze(-1) * distance).mean()
