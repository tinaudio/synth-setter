"""Frozen SAME autoencoder encoder serving a stationary differentiable metric space.

The backbone is differentiable from waveform to latent — no ``torch.no_grad`` and no numpy
hop — so the audio-feedback loss can score a rendered estimate in the same space the stored
``same_s`` conditioning column is written in. See
https://github.com/tinaudio/synth-setter/issues/2741.

Typical usage passes ``SameAudioEncoder.from_pretrained(...)`` to ``LatentMseDistance``.
"""

from __future__ import annotations

from typing import Final, Protocol, cast

import torchaudio.functional as audio_fn
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from synth_setter.model_cache import checkpoint_tree_sha256
from synth_setter.same import (
    DEFAULT_SAME_S_CHECKPOINT,
    SAME_SAMPLE_RATE,
    load_same_autoencoder,
    resolve_same_checkpoint,
)

_BATCH_AUDIO_SHAPE: Final = "batch samples"
_BATCH_STEREO_SHAPE: Final = "batch 2 resampled"
_BATCH_LATENT_SHAPE: Final = "batch latent frames"
_SAME_CHANNELS: Final = 2


class _SameAutoencoder(Protocol):
    """SAME autoencoder surface consumed by the encoder.

    .. attribute :: latent_dim

       Width of each latent frame.

    .. attribute :: pretransform

       Patching front end whose ``enable_grad`` flag gates the waveform gradient path.
    """

    latent_dim: int
    pretransform: nn.Module

    @jaxtyped(typechecker=beartype)
    def encode(
        self, audio: Float[Tensor, _BATCH_STEREO_SHAPE]
    ) -> Float[Tensor, _BATCH_LATENT_SHAPE]:
        """Encode preprocessed stereo audio.

        :param audio: Stereo waveform batch at ``SAME_SAMPLE_RATE``.
        :returns: Latent sequence shaped ``(batch, latent, frames)``.
        """
        ...


class SameAudioEncoder(nn.Module):
    """Frozen SAME audio branch, differentiable from waveform to latent."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, *, sample_rate: int, autoencoder: nn.Module) -> None:
        """Adopt a frozen SAME autoencoder and open its waveform gradient path.

        SAME's patched pretransform encodes under ``torch.no_grad`` by default, which would
        silently zero every waveform gradient; the patching itself carries no parameters.

        :param sample_rate: Source rate of waveforms handed to this encoder.
        :param autoencoder: Frozen SAME autoencoder exposing ``encode`` and ``pretransform``.
        :raises ValueError: The autoencoder is trainable or is not a SAME autoencoder.
        """
        super().__init__()
        trainable = [name for name, p in autoencoder.named_parameters() if p.requires_grad]
        if trainable:
            raise ValueError(
                f"autoencoder must be frozen; {len(trainable)} trainable parameter(s) "
                f"{trainable} would move the space the distance is measured in"
            )
        pretransform = getattr(autoencoder, "pretransform", None)
        latent_dim = getattr(autoencoder, "latent_dim", None)
        if (
            pretransform is None
            or not callable(getattr(autoencoder, "encode", None))
            or not isinstance(latent_dim, int)
        ):
            raise ValueError(
                "autoencoder must expose SAME's encode, latent_dim, and pretransform surface"
            )
        pretransform.enable_grad = True
        self.autoencoder = autoencoder
        self.sample_rate = sample_rate
        self.out_dim = latent_dim

    @classmethod
    @jaxtyped(typechecker=beartype)
    def from_pretrained(
        cls,
        *,
        sample_rate: int,
        checkpoint: str | None = None,
        checkpoint_sha256: str | None = None,
    ) -> SameAudioEncoder:
        """Load frozen SAME weights from a local, R2, or HuggingFace checkpoint.

        :param sample_rate: Source rate of waveforms handed to this encoder.
        :param checkpoint: Checkpoint source, or ``None`` for the shared SAME-S mirror.
        :param checkpoint_sha256: Expected materialized tree digest, or ``None`` to skip
            verification.
        :returns: Frozen differentiable SAME encoder.
        :raises ValueError: The materialized checkpoint digest differs from the expected one.
        """
        checkpoint_dir = resolve_same_checkpoint(checkpoint or DEFAULT_SAME_S_CHECKPOINT)
        if checkpoint_sha256 is not None:
            actual_sha256 = checkpoint_tree_sha256(checkpoint_dir)
            if actual_sha256 != checkpoint_sha256:
                raise ValueError(
                    f"SAME checkpoint digest mismatch: expected {checkpoint_sha256}, "
                    f"got {actual_sha256}"
                )
        return cls(sample_rate=sample_rate, autoencoder=load_same_autoencoder(checkpoint_dir))

    @jaxtyped(typechecker=beartype)
    def train(self, mode: bool = True) -> SameAudioEncoder:
        """Keep the backbone in eval mode so its normalization cannot drift.

        :param mode: Training mode requested for this module's own children.
        :returns: This module.
        """
        super().train(mode)
        self.autoencoder.eval()
        return self

    @jaxtyped(typechecker=beartype)
    def _stereo_at_same_rate(
        self, audio: Float[Tensor, _BATCH_AUDIO_SHAPE]
    ) -> Float[Tensor, _BATCH_STEREO_SHAPE]:
        """Resample and duplicate mono audio into SAME's stereo input contract.

        Duplication rather than a silent second channel, matching the ``same_s`` column the
        stored conditioning is written from.

        :param audio: Mono waveform batch at ``sample_rate``.
        :returns: Stereo waveform batch at ``SAME_SAMPLE_RATE``.
        """
        if self.sample_rate != SAME_SAMPLE_RATE:
            audio = audio_fn.resample(audio, self.sample_rate, SAME_SAMPLE_RATE)
        return audio.unsqueeze(1).expand(-1, _SAME_CHANNELS, -1)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, audio: Float[Tensor, _BATCH_AUDIO_SHAPE]
    ) -> Float[Tensor, _BATCH_LATENT_SHAPE]:
        """Embed a waveform batch with gradient intact all the way to the input.

        SAME zero-pads the tail to a whole latent frame internally, so a rendered estimate and
        its target share a frame grid whenever they share a length.

        :param audio: Mono waveform batch at ``sample_rate``.
        :returns: SAME latents shaped ``(batch, latent, frames)``.
        """
        return cast(_SameAutoencoder, self.autoencoder).encode(self._stereo_at_same_rate(audio))
