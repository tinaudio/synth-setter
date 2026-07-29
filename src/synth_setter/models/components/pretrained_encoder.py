"""Frozen pretrained audio backbone serving conditioning and a stationary metric space.

The backbone is differentiable from waveform to embedding — no ``torch.no_grad`` and no
numpy hop — so the audio-feedback loss can score a rendered estimate in a space whose
geometry does not move between steps. See
https://github.com/tinaudio/synth-setter/issues/2728.
"""

from __future__ import annotations

from typing import Any, Final, cast

import torch
import torchaudio.functional as audio_fn
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torch.nn import functional

from synth_setter.pipeline.data.add_embeddings import (
    CLAP_SAMPLE_RATE,
    DEFAULT_CLAP_CHECKPOINT,
    _resolve_clap_checkpoint,
)

# Matches ClapFeatureExtractor's ``mel_floor`` and ``power_to_db``'s ``min_value``, which
# together clamp the log-mel at -100 dB.
_MEL_FLOOR: Final = 1e-10
# Power spectrograms convert to dB at 10 dB per decade, not 20.
_POWER_TO_DB: Final = 10.0

_BATCH_AUDIO_SHAPE: Final = "batch samples"
_BATCH_FEATURES_SHAPE: Final = "batch 1 frames mels"
_BATCH_EMBEDDING_SHAPE: Final = "batch embedding"
_BATCH_ANY_SHAPE: Final = "batch ..."


class ClapAudioEncoder(nn.Module):
    """Frozen HF CLAP audio branch, differentiable from waveform to embedding."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        *,
        sample_rate: int,
        checkpoint: str = DEFAULT_CLAP_CHECKPOINT,
        pretrained: bool = True,
        backbone_config: dict[str, Any] | None = None,
    ) -> None:
        """Load the CLAP audio branch and mirror its feature extractor in torch.

        :param sample_rate: Rate of the waveforms handed to this encoder; resampled to
            CLAP's 48 kHz when it differs.
        :param checkpoint: Local directory or HuggingFace CLAP model id.
        :param pretrained: Load checkpoint weights instead of an offline random backbone.
        :param backbone_config: ``ClapConfig`` overrides used in offline mode.
        :raises ValueError: ``backbone_config`` is supplied in pretrained mode, or the
            backbone's mel-bin count disagrees with the feature extractor's.
        """
        super().__init__()
        from transformers import ClapConfig, ClapFeatureExtractor, ClapModel

        if pretrained and backbone_config is not None:
            raise ValueError("backbone_config requires pretrained=False")
        checkpoint_dir = _resolve_clap_checkpoint(checkpoint)
        clap = (
            ClapModel.from_pretrained(checkpoint_dir)
            if pretrained
            else ClapModel(ClapConfig(**(backbone_config or {})))
        )
        # The front-end geometry always comes from the checkpoint even offline: features
        # that diverge from the stored ``clap`` column are worse than no online path.
        extractor = ClapFeatureExtractor.from_pretrained(checkpoint_dir)
        clap_config = cast(Any, clap.config)
        backbone_mel_bins = clap_config.audio_config.num_mel_bins
        if backbone_mel_bins != extractor.feature_size:
            raise ValueError(
                f"backbone num_mel_bins {backbone_mel_bins} does not match feature "
                f"extractor feature_size {extractor.feature_size}"
            )

        self.clap = clap
        self.clap.requires_grad_(False)
        self.clap.eval()

        self.sample_rate = sample_rate
        self.out_dim: int = clap_config.projection_dim
        self.n_fft: int = extractor.fft_window_size
        self.hop_length: int = extractor.hop_length
        self.max_samples: int = extractor.nb_max_samples
        # HF precomputes the slaney-normalised bank; reimplementing it is the main
        # correctness risk, so copy it verbatim.
        self.register_buffer("mel_filters", torch.from_numpy(extractor.mel_filters_slaney).float())

    @jaxtyped(typechecker=beartype)
    def train(self, mode: bool = True) -> ClapAudioEncoder:
        """Keep the backbone in eval mode whatever the surrounding module does.

        ``requires_grad_(False)`` does not stop dropout or BatchNorm running stats, so a
        Lightning epoch boundary would otherwise move the metric space.

        :param mode: Training mode requested for this module's own children.
        :returns: This module.
        """
        super().train(mode)
        self.clap.eval()
        return self

    @jaxtyped(typechecker=beartype)
    def features(
        self, audio: Float[Tensor, _BATCH_AUDIO_SHAPE]
    ) -> Float[Tensor, _BATCH_FEATURES_SHAPE]:
        """Torch reimplementation of ``ClapFeatureExtractor``'s deterministic short path.

        Audio shorter than the extractor's 10 s window is tiled then zero-padded, and a
        single log-mel is taken — the ``np.random`` crops fire only past that window, so
        this path has no randomness to reproduce.

        :param audio: Mono waveform batch at ``sample_rate``.
        :returns: Log-mel in dB shaped ``(batch, 1, frames, mels)``.
        :raises ValueError: Audio is empty or exceeds the extractor's window, where the
            reference implementation crops at random and cannot be reproduced.
        """
        if self.sample_rate != CLAP_SAMPLE_RATE:
            audio = audio_fn.resample(audio, self.sample_rate, CLAP_SAMPLE_RATE)
        length = audio.shape[-1]
        if length == 0:
            raise ValueError("audio waveform cannot be empty")
        if length > self.max_samples:
            raise ValueError(
                f"audio of {length} samples exceeds CLAP's {self.max_samples}-sample "
                "window, past which the reference extractor crops at random"
            )
        if length < self.max_samples:
            audio = audio.repeat(1, self.max_samples // length)
            audio = functional.pad(audio, (0, self.max_samples - audio.shape[-1]))

        half_window = self.n_fft // 2
        centred = functional.pad(
            audio.unsqueeze(1), (half_window, half_window), mode="reflect"
        ).squeeze(1)
        # HF's extractor runs its FFT in float64. In float32 the sub- -60 dB leakage floor
        # diverges by up to 0.08 dB, which would desync this from the stored ``clap`` column.
        centred = centred.double()
        window = torch.hann_window(
            self.n_fft, periodic=True, device=audio.device, dtype=centred.dtype
        )
        spectrum = torch.stft(
            centred,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=False,
            onesided=True,
            return_complex=True,
        )
        # The mel bank sums non-negative bins, so float32 is exact enough past the FFT.
        mel_filters = cast(Tensor, self.mel_filters)
        power = (spectrum.real.square() + spectrum.imag.square()).to(mel_filters.dtype)
        mel = torch.einsum("bft,fm->bmt", power, mel_filters).clamp_min(_MEL_FLOOR)
        return (_POWER_TO_DB * torch.log10(mel)).transpose(1, 2).unsqueeze(1)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, audio: Float[Tensor, _BATCH_AUDIO_SHAPE]
    ) -> Float[Tensor, _BATCH_EMBEDDING_SHAPE]:
        """Embed a waveform batch with gradient intact all the way to the input.

        :param audio: Mono waveform batch at ``sample_rate``.
        :returns: Pooled CLAP audio embedding shaped ``(batch, out_dim)``.
        :raises RuntimeError: The Transformers audio branch returns no pooled embedding.
        """
        output = self.clap.get_audio_features(input_features=self.features(audio))
        embedding = cast(Any, output).pooler_output
        if not isinstance(embedding, Tensor):
            raise RuntimeError("CLAP audio branch returned no pooled embedding")
        return embedding


class PretrainedConditioningEncoder(nn.Module):
    """Frozen backbone with a trainable projection head, tapped at two depths."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, *, backbone: ClapAudioEncoder, head: nn.Module, out_dim: int) -> None:
        """Pair a frozen backbone with the head that adapts it to the flow's width.

        :param backbone: Frozen waveform-in encoder defining the metric space.
        :param head: Trainable module mapping ``backbone.out_dim`` to ``out_dim``.
        :param out_dim: Conditioning width Hydra resolves ``${model.encoder.out_dim}`` to.
        :raises ValueError: The head's input width does not match the backbone's output.
        """
        super().__init__()
        head_input_dim = getattr(head, "input_dim", None)
        if head_input_dim != backbone.out_dim:
            raise ValueError(
                f"head input width {head_input_dim} does not match backbone out_dim "
                f"{backbone.out_dim}"
            )
        self.backbone = backbone
        self.head = head
        self.out_dim = out_dim

    @jaxtyped(typechecker=beartype)
    def embed(
        self, audio: Float[Tensor, _BATCH_AUDIO_SHAPE]
    ) -> Float[Tensor, _BATCH_EMBEDDING_SHAPE]:
        """Embed audio in the frozen backbone's space — the audio loss's metric tap.

        :param audio: Mono waveform batch.
        :returns: Backbone embedding shaped ``(batch, backbone.out_dim)``.
        """
        return self.backbone(audio)

    @jaxtyped(typechecker=beartype)
    def project(
        self, embedding: Float[Tensor, _BATCH_EMBEDDING_SHAPE]
    ) -> Float[Tensor, _BATCH_ANY_SHAPE]:
        """Map a backbone embedding to the vector field's conditioning width.

        :param embedding: Backbone embedding shaped ``(batch, backbone.out_dim)``.
        :returns: Conditioning shaped ``(batch, out_dim)``.
        """
        return self.head(embedding)

    @jaxtyped(typechecker=beartype)
    def forward(self, audio: Float[Tensor, _BATCH_AUDIO_SHAPE]) -> Float[Tensor, _BATCH_ANY_SHAPE]:
        """Encode audio into conditioning through both taps.

        :param audio: Mono waveform batch.
        :returns: Conditioning shaped ``(batch, out_dim)``.
        """
        return self.project(self.embed(audio))
