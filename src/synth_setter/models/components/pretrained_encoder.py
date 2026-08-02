"""Frozen pretrained audio backbone serving conditioning and a stationary metric space.

The backbone is differentiable from waveform to embedding — no ``torch.no_grad`` and no
numpy hop — so the audio-feedback loss can score a rendered estimate in a space whose
geometry does not move between steps. See
https://github.com/tinaudio/synth-setter/issues/2728.

Typical usage constructs a frozen waveform backbone and passes it to
``PretrainedConditioningEncoder``; call ``embed`` for its pretrained representation and
``forward`` for flow conditioning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Protocol, cast, runtime_checkable

import numpy as np
import torch
import torchaudio.functional as audio_fn
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torch.nn import functional

from synth_setter.clap import (
    DEFAULT_CLAP_TRAINING_CHECKPOINT,
    DEFAULT_CLAP_TRAINING_CHECKPOINT_SHA256,
    clap_checkpoint_sha256,
    resolve_clap_checkpoint,
)

# Matches ClapFeatureExtractor's ``mel_floor`` and ``power_to_db``'s ``min_value``, which
# together clamp the log-mel at -100 dB.
_MEL_FLOOR: Final = 1e-10
# Convert power spectrograms, rather than amplitudes, to decibels.
_POWER_TO_DB: Final = 10.0

_BATCH_AUDIO_INPUT_SHAPE: Final = "batch ... samples"
_BATCH_FEATURES_SHAPE: Final = "batch 1 frames mels"
_BATCH_EMBEDDING_SHAPE: Final = "batch embedding"
_BATCH_ANY_SHAPE: Final = "batch ..."
_OFFLINE_CONFIG_KEYS: Final = frozenset({"audio_config", "projection_dim", "text_config"})
_SUPPORTED_TRUNCATION: Final = "rand_trunc"
_SUPPORTED_PADDING: Final = "repeatpad"


class _ClapAudioConfig(Protocol):
    """Typed CLAP audio configuration fields consumed by the encoder.

    .. attribute :: num_mel_bins

       Number of mel bins consumed by the backbone.
    """

    num_mel_bins: int


class _ClapConfig(Protocol):
    """Typed CLAP configuration fields consumed by the encoder.

    .. attribute :: audio_config

       Audio-branch configuration.

    .. attribute :: projection_dim

       Pooled embedding width.
    """

    audio_config: _ClapAudioConfig
    projection_dim: int


@runtime_checkable
class _ClapFeatureExtractor(Protocol):
    """Feature-extractor fields needed by the differentiable frontend.

    .. attribute :: feature_size

       Number of mel bins.

    .. attribute :: fft_window_size

       FFT window length in samples.

    .. attribute :: hop_length

       STFT hop length in samples.

    .. attribute :: mel_filters_slaney

       Slaney-normalized mel filter bank.

    .. attribute :: nb_max_samples

       Maximum deterministic waveform length.

    .. attribute :: padding

       Short-waveform padding policy.

    .. attribute :: sampling_rate

       Target waveform sample rate.

    .. attribute :: truncation

       Long-waveform truncation policy.
    """

    feature_size: int
    fft_window_size: int
    hop_length: int
    mel_filters_slaney: np.ndarray
    nb_max_samples: int
    padding: str
    sampling_rate: int
    truncation: str


class _ClapAudioModel(Protocol):
    """CLAP audio API used after registering the concrete module."""

    @jaxtyped(typechecker=beartype)
    def get_audio_features(
        self, *, input_features: Float[Tensor, _BATCH_FEATURES_SHAPE]
    ) -> object:
        """Return pooled audio features.

        :param input_features: CLAP log-mel features.
        :returns: Tensor or Transformers model output carrying pooled features.
        """
        ...


@jaxtyped(typechecker=beartype)
def _plain_config_value(value: object) -> object:
    """Convert Hydra containers into Transformers' built-in collection contract.

    :param value: Nested config value from Hydra or a direct caller.
    :returns: Equivalent value containing only built-in dictionaries and lists.
    """
    if isinstance(value, Mapping):
        return {str(key): _plain_config_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_config_value(item) for item in value]
    return value


@jaxtyped(typechecker=beartype)
def _verified_checkpoint_dir(checkpoint: str | None, expected_sha256: str | None) -> str:
    """Resolve and verify a checkpoint without mutating its shared cache.

    :param checkpoint: Explicit checkpoint or ``None`` for the training default.
    :param expected_sha256: Explicit expected digest or ``None`` for default policy.
    :returns: Verified local checkpoint directory.
    :raises ValueError: The materialized checkpoint digest differs from the expected digest.
    """
    if checkpoint is None:
        checkpoint = DEFAULT_CLAP_TRAINING_CHECKPOINT
    if checkpoint == DEFAULT_CLAP_TRAINING_CHECKPOINT and expected_sha256 is None:
        expected_sha256 = DEFAULT_CLAP_TRAINING_CHECKPOINT_SHA256

    checkpoint_dir = resolve_clap_checkpoint(checkpoint)
    if expected_sha256 is None:
        return checkpoint_dir
    actual_sha256 = clap_checkpoint_sha256(Path(checkpoint_dir))
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"CLAP checkpoint digest mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return checkpoint_dir


@jaxtyped(typechecker=beartype)
def _random_clap(backbone_config: Mapping[str, object]) -> nn.Module:
    """Build a random CLAP model from an explicit bounded config.

    :param backbone_config: Required ``ClapConfig`` projection/text/audio mappings.
    :returns: Randomly initialized Transformers CLAP model.
    :raises ValueError: The config has unknown keys or malformed required mappings.
    """
    from transformers import ClapConfig, ClapModel

    unknown_keys = set(backbone_config) - _OFFLINE_CONFIG_KEYS
    if unknown_keys:
        raise ValueError(f"unknown offline CLAP config keys: {sorted(unknown_keys)}")
    config_values = cast(dict[str, object], _plain_config_value(backbone_config))
    projection_dim = config_values.get("projection_dim", 512)
    text_config = config_values.get("text_config")
    audio_config = config_values.get("audio_config")
    if not isinstance(projection_dim, int):
        raise ValueError("offline CLAP projection_dim must be an integer")
    if not isinstance(text_config, dict) or not isinstance(audio_config, dict):
        raise ValueError("offline CLAP text_config and audio_config must be mappings")
    return ClapModel(
        ClapConfig(
            projection_dim=projection_dim,
            text_config=text_config,
            audio_config=audio_config,
        )
    )


@jaxtyped(typechecker=beartype)
def _load_feature_extractor(checkpoint_dir: str) -> _ClapFeatureExtractor:
    """Load and validate the deterministic CLAP frontend contract.

    :param checkpoint_dir: Materialized Transformers checkpoint directory.
    :returns: Validated feature extractor.
    :raises ValueError: The extractor uses unsupported truncation or padding.
    """
    from transformers import ClapFeatureExtractor

    extractor = cast(_ClapFeatureExtractor, ClapFeatureExtractor.from_pretrained(checkpoint_dir))
    if extractor.truncation != _SUPPORTED_TRUNCATION:
        raise ValueError(
            f"CLAP feature extractor truncation must be {_SUPPORTED_TRUNCATION!r}, "
            f"got {extractor.truncation!r}"
        )
    if extractor.padding != _SUPPORTED_PADDING:
        raise ValueError(
            f"CLAP feature extractor padding must be {_SUPPORTED_PADDING!r}, "
            f"got {extractor.padding!r}"
        )
    return extractor


class ClapAudioEncoder(nn.Module):
    """Frozen HF CLAP audio branch, differentiable from waveform to embedding."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        *,
        sample_rate: int,
        clap: nn.Module,
        extractor: _ClapFeatureExtractor,
    ) -> None:
        """Register an already selected CLAP model and validated frontend.

        Call :meth:`from_pretrained` or :meth:`from_random_config`; this constructor only
        performs the construction shared by both explicit policies.

        :param sample_rate: Source rate of waveforms handed to this encoder.
        :param clap: Pretrained or explicitly configured random CLAP model.
        :param extractor: Validated feature extractor from the selected checkpoint.
        :raises ValueError: Backbone and extractor mel-bin counts differ.
        """
        super().__init__()
        clap_config = cast(_ClapConfig, getattr(clap, "config"))
        backbone_mel_bins = clap_config.audio_config.num_mel_bins
        if backbone_mel_bins != extractor.feature_size:
            raise ValueError(
                f"backbone num_mel_bins {backbone_mel_bins} does not match feature "
                f"extractor feature_size {extractor.feature_size}"
            )

        self.clap: nn.Module = clap
        self.clap.requires_grad_(False)
        self.clap.eval()

        self.sample_rate = sample_rate
        self.target_sample_rate = extractor.sampling_rate
        self.out_dim: int = clap_config.projection_dim
        self.n_fft: int = extractor.fft_window_size
        self.hop_length: int = extractor.hop_length
        self.max_samples: int = extractor.nb_max_samples
        self.mel_filters: Tensor
        self.register_buffer("mel_filters", torch.from_numpy(extractor.mel_filters_slaney).float())

    @classmethod
    @jaxtyped(typechecker=beartype)
    def from_pretrained(
        cls,
        *,
        sample_rate: int,
        checkpoint: str | None = None,
        checkpoint_sha256: str | None = None,
    ) -> ClapAudioEncoder:
        """Load frozen CLAP weights from a verified checkpoint.

        :param sample_rate: Source rate of waveforms handed to this encoder.
        :param checkpoint: Local directory, R2 prefix, Hugging Face id, or ``None`` for
            the shared training checkpoint.
        :param checkpoint_sha256: Expected materialized identity; default-policy checkpoints
            use the shared training digest when omitted.
        :returns: Frozen pretrained waveform encoder.
        """
        from transformers import ClapModel

        checkpoint_dir = _verified_checkpoint_dir(checkpoint, checkpoint_sha256)
        extractor = _load_feature_extractor(checkpoint_dir)
        clap = ClapModel.from_pretrained(checkpoint_dir)
        return cls(sample_rate=sample_rate, clap=clap, extractor=extractor)

    @classmethod
    @jaxtyped(typechecker=beartype)
    def from_random_config(
        cls,
        *,
        sample_rate: int,
        backbone_config: Mapping[str, object],
        checkpoint: str | None = None,
        checkpoint_sha256: str | None = None,
    ) -> ClapAudioEncoder:
        """Build a frozen random CLAP backbone from mandatory explicit geometry.

        :param sample_rate: Source rate of waveforms handed to this encoder.
        :param backbone_config: Required random ``ClapConfig`` projection/text/audio config.
        :param checkpoint: Checkpoint supplying the feature-extractor contract, or ``None``
            for the shared training checkpoint.
        :param checkpoint_sha256: Expected materialized identity; default-policy checkpoints
            use the shared training digest when omitted.
        :returns: Frozen randomly initialized waveform encoder.
        """
        checkpoint_dir = _verified_checkpoint_dir(checkpoint, checkpoint_sha256)
        extractor = _load_feature_extractor(checkpoint_dir)
        return cls(
            sample_rate=sample_rate,
            clap=_random_clap(backbone_config),
            extractor=extractor,
        )

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
        self, audio: Float[Tensor, _BATCH_AUDIO_INPUT_SHAPE]
    ) -> Float[Tensor, _BATCH_FEATURES_SHAPE]:
        """Torch reimplementation of ``ClapFeatureExtractor``'s deterministic short path.

        Audio shorter than the extractor's 10 s window is tiled then zero-padded, and a
        single log-mel is taken — the ``np.random`` crops fire only past that window, so
        this path has no randomness to reproduce.

        :param audio: Mono waveform batch at ``sample_rate``.
        :returns: Log-mel in dB shaped ``(batch, 1, frames, mels)``.
        :raises ValueError: Audio is empty, non-finite, outside ``[-1, 1]``, on MPS,
            or exceeds the extractor's deterministic short-audio window.
        """
        if audio.ndim == 3:
            audio = audio.mean(dim=1)
        elif audio.ndim != 2:
            raise ValueError(
                f"audio must have shape (batch, [channels,] samples), got {audio.shape}"
            )
        if audio.device.type == "mps":
            raise ValueError("CLAP online features do not support MPS float64 FFT")
        if audio.shape[-1] == 0:
            raise ValueError("audio waveform cannot be empty")
        if not torch.isfinite(audio).all():
            raise ValueError("audio waveform must contain only finite values")
        if (audio.abs() > 1.0).any():
            raise ValueError("audio waveform values must be in [-1, 1]")

        if self.sample_rate != self.target_sample_rate:
            audio = audio_fn.resample(audio, self.sample_rate, self.target_sample_rate)
        # Stored add_embeddings CLAP features preserve finite resampler overshoot.
        length = audio.shape[-1]
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
        self, audio: Float[Tensor, _BATCH_AUDIO_INPUT_SHAPE]
    ) -> Float[Tensor, _BATCH_EMBEDDING_SHAPE]:
        """Embed a waveform batch with gradient intact all the way to the input.

        :param audio: Mono waveform batch at ``sample_rate``.
        :returns: Pooled CLAP audio embedding shaped ``(batch, out_dim)``.
        :raises RuntimeError: The Transformers audio branch returns no pooled embedding.
        """
        clap = cast(_ClapAudioModel, self.clap)
        output = clap.get_audio_features(input_features=self.features(audio))
        embedding = (
            output if isinstance(output, Tensor) else getattr(output, "pooler_output", None)
        )
        if not isinstance(embedding, Tensor):
            raise RuntimeError("CLAP audio branch returned no pooled embedding")
        return embedding


class PretrainedConditioningEncoder(nn.Module):
    """Frozen backbone with a trainable projection head, tapped at two depths."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, *, backbone: nn.Module, head: nn.Module, out_dim: int) -> None:
        """Pair a frozen backbone with the head that adapts it to the flow's width.

        :param backbone: Frozen waveform-in encoder defining the metric space.
        :param head: Trainable module mapping ``backbone.out_dim`` to ``out_dim``.
        :param out_dim: Conditioning width Hydra resolves ``${model.encoder.out_dim}`` to.
        :raises ValueError: Width metadata is missing or the backbone and head widths differ.
        """
        super().__init__()
        backbone_out_dim = getattr(backbone, "out_dim", None)
        head_input_dim = getattr(head, "input_dim", None)
        if not isinstance(backbone_out_dim, int) or not isinstance(head_input_dim, int):
            raise ValueError("backbone and head must expose integer dimension metadata")
        if head_input_dim != backbone_out_dim:
            raise ValueError(
                f"head input width {head_input_dim} does not match backbone out_dim "
                f"{backbone_out_dim}"
            )
        self.backbone = backbone
        self.head = head
        self.out_dim = out_dim

    @jaxtyped(typechecker=beartype)
    def embed(
        self, audio: Float[Tensor, _BATCH_AUDIO_INPUT_SHAPE]
    ) -> Float[Tensor, _BATCH_ANY_SHAPE]:
        """Embed audio in the frozen backbone's representation.

        :param audio: Mono waveform batch.
        :returns: Backbone representation with a leading batch axis.
        """
        return self.backbone(audio)

    @jaxtyped(typechecker=beartype)
    def project(
        self, embedding: Float[Tensor, _BATCH_ANY_SHAPE]
    ) -> Float[Tensor, _BATCH_ANY_SHAPE]:
        """Map a backbone representation to the flow's conditioning width.

        :param embedding: Backbone representation with a leading batch axis.
        :returns: Conditioning shaped ``(batch, out_dim)``.
        """
        return self.head(embedding)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, audio: Float[Tensor, _BATCH_AUDIO_INPUT_SHAPE]
    ) -> Float[Tensor, _BATCH_ANY_SHAPE]:
        """Encode audio into conditioning through both taps.

        :param audio: Mono waveform batch.
        :returns: Conditioning shaped ``(batch, out_dim)``.
        """
        return self.project(self.embed(audio))
