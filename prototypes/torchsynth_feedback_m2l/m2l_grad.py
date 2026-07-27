"""Gradient-enabled music2latent encoding for the simulator-feedback spike (#2557).

music2latent's public ``EncoderDecoder.encode`` runs under ``@torch.no_grad()``
and returns numpy (see ``pipeline.data.add_embeddings.load_m2l_audio_encoder``);
this tensor-in/tensor-out sibling keeps the graph connected from the audio
input to the latent so a simulator-feedback cost can differentiate through the
embedding. Weights hydrate into the shared embedding cache
(:func:`synth_setter.model_cache.embedding_model_dir`).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from synth_setter.model_cache import embedding_model_dir

M2L_SAMPLE_RATE = 44_100
# The m2l frontend crops audio to 3*hop + k*hop*downscaling_factor samples
# (hop=512, downscaling_factor=8); anything below one latent frame crops to
# zero frames and the encoder collapses.
M2L_MIN_SIGNAL_LENGTH = 3 * 512 + 512 * 8
# Magnitude floor for the STFT normalization: |X|**0.65 and angle(X) have NaN
# gradients at X=0 (silent frames of a synth render hit this exactly); the
# guarded form beta*(|X|+eps)**(alpha-1)*X matches the original to ~eps.
_STFT_MAGNITUDE_EPS = 1e-4


def m2l_checkpoint_path() -> Path:
    """Hydrate the music2latent checkpoint into the shared embedding cache.

    :returns: Local path of ``music2latent.pt``.
    """
    from huggingface_hub import hf_hub_download

    target = embedding_model_dir("music2latent")
    path = target / "music2latent.pt"
    if not path.exists():
        target.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id="SonyCSLParis/music2latent",
            filename="music2latent.pt",
            local_dir=str(target),
        )
    return path


class M2LGradEncoder(nn.Module):
    """Frozen music2latent encoder with autograd connected to the audio input.

    Runs in float32: music2latent's own inference path notes fp16 encoding
    can produce NaNs, so no autocast here.
    """

    def __init__(self, device: str) -> None:
        """Load the frozen music2latent encoder onto ``device``.

        :param device: Torch device string.
        """
        super().__init__()
        from music2latent import EncoderDecoder

        encoder_decoder = EncoderDecoder(
            load_path_inference=str(m2l_checkpoint_path()), device=device
        )
        generator = encoder_decoder.gen.eval()
        generator.requires_grad_(False)
        self.encoder = generator.encoder

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode 44.1 kHz mono audio into m2l latents with gradients attached.

        :param audio: Audio shaped ``(batch, samples)`` at 44.1 kHz.
        :returns: Latents shaped ``(batch, 64, T_lat)``, one frame per 4096
            samples, scaled like ``EncoderDecoder.encode`` output.
        :raises ValueError: Audio shorter than ``M2L_MIN_SIGNAL_LENGTH``.
        """
        from music2latent.hparams import freq_downsample_list, hop
        from music2latent.hparams_inference import sigma_rescale

        if audio.shape[-1] < M2L_MIN_SIGNAL_LENGTH:
            raise ValueError(
                f"m2l needs >= {M2L_MIN_SIGNAL_LENGTH} samples, got {audio.shape[-1]}"
            )
        downscaling_factor = 2 ** freq_downsample_list.count(0)
        cropped = (
            (((audio.shape[-1] - 3 * hop) // hop) // downscaling_factor) * hop * downscaling_factor
        ) + 3 * hop
        representation = _safe_representation_encoder(audio[..., :cropped])
        return self.encoder(representation) / sigma_rescale


def _safe_representation_encoder(audio: torch.Tensor) -> torch.Tensor:
    """NaN-safe drop-in for music2latent's ``to_representation_encoder``.

    The original ``normalize_complex`` computes ``|X|**alpha * exp(i*angle(X))``
    whose gradient is NaN at ``X = 0``; this guarded, angle-free form is
    ``beta * (|X| + eps)**(alpha - 1) * X`` — identical up to ``eps`` for
    non-silent bins and smooth everywhere.

    :param audio: Cropped audio shaped ``(batch, samples)`` at 44.1 kHz.
    :returns: Real/imag representation shaped ``(batch, 2, 2*hop, frames)``.
    """
    from music2latent.audio import stft
    from music2latent.hparams import alpha_rescale, beta_rescale, hop

    spectrum = stft(audio, hop_size=hop, device=audio.device)[:, : hop * 2, :]
    magnitude = torch.sqrt(
        spectrum.real.square() + spectrum.imag.square() + _STFT_MAGNITUDE_EPS**2
    )
    scale = beta_rescale * magnitude ** (alpha_rescale - 1)
    return torch.stack((spectrum.real * scale, spectrum.imag * scale), dim=-3)


def m2l_pooled_l2(pred_latents: torch.Tensor, target_latents: torch.Tensor) -> torch.Tensor:
    """Per-sample L2 distance between time-pooled m2l latents.

    Mean-pooling over the latent time axis before the L2 keeps the cost scale
    independent of audio length; alternatives (framewise MSE, cosine) are
    logged in #2557.

    :param pred_latents: Latents shaped ``(batch, dim, T_lat)``.
    :param target_latents: Latents shaped ``(batch, dim, T_lat)``.
    :returns: Distances shaped ``(batch,)``.
    """
    return (pred_latents.mean(dim=-1) - target_latents.mean(dim=-1)).norm(dim=-1)


def m2l_framewise_mse(pred_latents: torch.Tensor, target_latents: torch.Tensor) -> torch.Tensor:
    """Per-sample MSE over the full (dim, T_lat) latent, no time pooling.

    Keeps temporal structure the pooled cost discards; #2557 Step C tests
    whether pooling is why the pooled-L2 control signal underperforms.

    :param pred_latents: Latents shaped ``(batch, dim, T_lat)``.
    :param target_latents: Latents shaped ``(batch, dim, T_lat)``.
    :returns: Distances shaped ``(batch,)``.
    """
    return (pred_latents - target_latents).square().mean(dim=(-2, -1))
