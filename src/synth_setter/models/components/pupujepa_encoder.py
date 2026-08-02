"""Frozen PupuJEPA teachers from waveform to frequency-concatenated sequences.

The architecture is adapted from PupuJEPA commit
``54a621e9f879be7659d81b6a3c493bba855cc85f`` under the MIT license retained in
``LICENSES/PupuJEPA-MIT.txt``. Only the patch embed and EMA teacher inference path are included.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np
import torch
import torch.nn.functional as functional
import torchaudio.functional as audio_functional
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from synth_setter.pupujepa import (
    DEFAULT_PUPUJEPA_TINY_CHECKPOINT,
    PUPUJEPA_CHECKPOINT_REVISION,
    PUPUJEPA_CHECKPOINT_SPECS,
    PUPUJEPA_TINY_CONFIG,
    PupuJepaConfig,
    PupuJepaVariant,
    load_pupujepa_config,
    pupujepa_checkpoint_files,
    pupujepa_num_time_patches,
    resolve_pupujepa_checkpoint,
)

_BATCH_AUDIO = "batch ... samples"
_BATCH_MEL = "batch 1 frames mel"
_BATCH_PATCHES = "batch patches hidden"
_BATCH_SEQUENCE = "batch embedding time_patches"
_MEL_FILTER = "mel frequency"
_WINDOW = "window"


class _RotaryEmbedding(Protocol):
    """RoPE surface consumed from the pinned timm implementation."""

    @jaxtyped(typechecker=beartype)
    def get_embed(self, shape: Sequence[int]) -> Float[Tensor, "patches rope_dim"]:
        """Build a rotary embedding for a two-dimensional patch grid.

        :param shape: Time and frequency patch counts.
        :returns: Concatenated sine/cosine rotary values per patch.
        """
        ...


class PupuJepaMelFrontend(nn.Module):
    """Implement the pinned librosa-mel frontend with waveform gradients.

    .. attribute :: mel_filter

        Librosa-compatible mel projection matrix.

    .. attribute :: window

        Hann STFT window.
    """

    mel_filter: Float[Tensor, _MEL_FILTER]
    window: Float[Tensor, _WINDOW]

    @jaxtyped(typechecker=beartype)
    def __init__(self, config: PupuJepaConfig = PUPUJEPA_TINY_CONFIG) -> None:
        """Build fixed Hann-window and librosa mel-filter buffers.

        :param config: Shape and normalization contract used by the checkpoint.
        """
        super().__init__()
        from librosa.filters import mel

        mel_filter = mel(
            sr=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.fmin,
            fmax=config.fmax,
            dtype=np.float32,
        )
        self.config = config
        self.register_buffer("mel_filter", torch.from_numpy(mel_filter), persistent=False)
        self.register_buffer(
            "window",
            torch.hann_window(config.win_length),
            persistent=False,
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, audio: Float[Tensor, _BATCH_AUDIO]) -> Float[Tensor, _BATCH_MEL]:
        """Convert native-rate mono audio to normalized time-major log-mel features.

        :param audio: Mono waveform batch at ``config.sample_rate``.
        :returns: Features shaped ``(batch, 1, mel_frames, n_mels)``.
        :raises ValueError: Audio is empty, too short, or contains non-finite values.
        """
        config = self.config
        pupujepa_num_time_patches(audio.shape[-1], config.sample_rate, config)
        if len(audio) < 1:
            raise ValueError("PupuJEPA expects a non-empty batch")
        if not torch.isfinite(audio).all():
            raise ValueError("PupuJEPA input audio contains non-finite values")

        waveform = functional.pad(
            audio.float().unsqueeze(1),
            (config.reflection_padding, config.reflection_padding),
            mode="reflect",
        ).squeeze(1)
        spectrum = torch.stft(
            waveform,
            config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            window=self.window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        magnitude = spectrum.abs()
        mel_spectrum = torch.matmul(self.mel_filter, magnitude)
        log_mel = torch.log(torch.clamp(mel_spectrum, min=1e-5))
        normalized = (log_mel - config.mel_mean) / (config.mel_std + 1e-8)
        return normalized.transpose(-2, -1).unsqueeze(1)


class _PupuJepaPatchEmbed(nn.Module):
    """Patch time-major log-mel images in upstream token order."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, config: PupuJepaConfig) -> None:
        """Construct the checkpoint-compatible convolution and layer norm.

        :param config: Patch geometry and hidden width.
        """
        super().__init__()
        patch_size = (config.patch_time, config.patch_frequency)
        self.proj = nn.Conv2d(1, config.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm_layer = nn.LayerNorm(config.embed_dim)

    @jaxtyped(typechecker=beartype)
    def forward(self, features: Float[Tensor, _BATCH_MEL]) -> Float[Tensor, _BATCH_PATCHES]:
        """Return flattened time-major then frequency-major patch tokens.

        :param features: Time-major log-mel image batch.
        :returns: Patch tokens shaped ``(batch, patches, hidden)``.
        """
        patches = self.proj(features).permute(0, 2, 3, 1)
        return self.norm_layer(patches.reshape(len(features), -1, patches.shape[-1]))


class _PupuJepaTransformer(nn.Module):
    """Minimal checkpoint-compatible PupuJEPA teacher transformer."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, config: PupuJepaConfig) -> None:
        """Construct the EMA teacher's EVA blocks.

        :param config: Hidden width, depth, attention, and MLP settings.
        """
        super().__init__()
        from timm.models.eva import EvaBlock

        self.blocks = nn.ModuleList(
            [
                EvaBlock(
                    dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                    attn_type="rope",
                    num_prefix_tokens=0,
                    drop_path=0.0,
                    swiglu_mlp=config.use_swiglu,
                    init_values=None,
                    qk_norm=config.qk_norm,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        patches: Float[Tensor, _BATCH_PATCHES],
        rope: Float[Tensor, "patches rope_dim"],
    ) -> Float[Tensor, _BATCH_PATCHES]:
        """Encode every unmasked patch with the frozen teacher.

        :param patches: Full patch sequence.
        :param rope: Two-dimensional rotary embedding for the patch grid.
        :returns: Teacher hidden states in the input token order.
        """
        hidden = patches
        for block in self.blocks:
            hidden = block(hidden, rope=rope)
        return self.norm(hidden)


class _PupuJepaTeacherModel(nn.Module):
    """Checkpoint-owned patch embed and EMA teacher modules."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, config: PupuJepaConfig) -> None:
        """Build the exact inference subset represented in the checkpoint.

        :param config: PupuJEPA architecture.
        """
        super().__init__()
        from timm.layers.pos_embed_sincos import create_rope_embed

        self.config = config
        self.patch_embed = _PupuJepaPatchEmbed(config)
        self.teacher = _PupuJepaTransformer(config)
        self.rope = cast(
            "_RotaryEmbedding",
            create_rope_embed(
                rope_type="cat",
                dim=config.embed_dim,
                num_heads=config.num_heads,
                feat_shape=None,
            ),
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, features: Float[Tensor, _BATCH_MEL]) -> Float[Tensor, _BATCH_SEQUENCE]:
        """Return frequency-concatenated teacher states per time patch.

        :param features: Time-major log-mel image batch.
        :returns: Sequence shaped ``(batch, frequency_patches * hidden, time_patches)``.
        """
        config = self.config
        time_patches = features.shape[-2] // config.patch_time
        grid_size: Sequence[int] = (time_patches, config.frequency_patches)
        patches = self.patch_embed(features)
        rope = self.rope.get_embed(grid_size).to(device=patches.device, dtype=patches.dtype)
        hidden = self.teacher(patches, rope)
        grouped = hidden.reshape(
            len(features),
            time_patches,
            config.frequency_patches * config.embed_dim,
        )
        return grouped.transpose(1, 2).contiguous()


class PupuJepaAudioEncoder(nn.Module):
    """Frozen PupuJEPA teacher shared by online and offline conditioning."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        *,
        sample_rate: int,
        config: PupuJepaConfig = PUPUJEPA_TINY_CONFIG,
    ) -> None:
        """Build a frozen teacher for waveforms arriving at ``sample_rate``.

        :param sample_rate: Default source waveform rate in Hz.
        :param config: Explicit PupuJEPA frontend and teacher geometry.
        :raises ValueError: The source rate is non-positive.
        """
        super().__init__()
        if sample_rate < 1:
            raise ValueError(f"PupuJEPA needs a positive sample_rate, got {sample_rate}")
        self.sample_rate = sample_rate
        self.config = config
        self.out_dim = config.output_dim
        self.frontend = PupuJepaMelFrontend(config)
        self.teacher_model = _PupuJepaTeacherModel(config)
        self.requires_grad_(False)
        self.eval()

    @classmethod
    @jaxtyped(typechecker=beartype)
    def from_pretrained(
        cls,
        *,
        sample_rate: int,
        checkpoint: str = DEFAULT_PUPUJEPA_TINY_CHECKPOINT,
        revision: str = PUPUJEPA_CHECKPOINT_REVISION,
        variant: PupuJepaVariant = "tiny",
    ) -> PupuJepaAudioEncoder:
        """Load only the patch embed and teacher from the pinned safetensors file.

        :param sample_rate: Default source waveform rate in Hz.
        :param checkpoint: Canonical Hugging Face repo id or local checkpoint directory.
        :param revision: Immutable Hugging Face commit required for remote loading.
        :param variant: Released teacher size to load.
        :returns: Frozen eval-mode PupuJEPA audio encoder.
        :raises RuntimeError: Teacher state is missing, unexpected, or shape-incompatible.
        :raises ValueError: Checkpoint geometry differs from the selected variant.
        """
        from safetensors import safe_open

        checkpoint_dir = resolve_pupujepa_checkpoint(checkpoint, revision, variant)
        config = load_pupujepa_config(checkpoint_dir, variant)
        expected_config = PUPUJEPA_CHECKPOINT_SPECS[variant].config
        if config != expected_config:
            raise ValueError(
                f"checkpoint is not the pinned PupuJEPA {variant} architecture: {config}"
            )
        encoder = cls(sample_rate=sample_rate, config=config)
        _, weights_path = pupujepa_checkpoint_files(checkpoint_dir, variant)
        with safe_open(weights_path, framework="pt", device="cpu") as checkpoint_file:
            state = {
                key: checkpoint_file.get_tensor(key)
                for key in checkpoint_file.keys()
                if key.startswith("patch_embed.") or key.startswith("teacher.")
            }
        try:
            encoder.teacher_model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(f"strict PupuJEPA teacher state is incompatible: {exc}") from exc
        return encoder.eval()

    @jaxtyped(typechecker=beartype)
    def train(self, mode: bool = True) -> PupuJepaAudioEncoder:
        """Keep the frozen teacher in eval mode regardless of its parent module.

        :param mode: Parent training mode request, ignored for the frozen backbone.
        :returns: This encoder in eval mode.
        """
        del mode
        super().train(False)
        return self

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        audio: Float[Tensor, _BATCH_AUDIO],
        sample_rate: int | None = None,
    ) -> Float[Tensor, _BATCH_SEQUENCE]:
        """Embed mono or channel-first waveforms while preserving input gradients.

        :param audio: Waveform batch shaped ``(B, T)`` or ``(B, C, T)``.
        :param sample_rate: Source rate override used by the offline adapter.
        :returns: Teacher sequence shaped ``(batch, out_dim, time_patches)``.
        :raises ValueError: Input shape, values, rate, or output violates the contract.
        """
        source_rate = self.sample_rate if sample_rate is None else sample_rate
        if source_rate < 1:
            raise ValueError(f"PupuJEPA needs a positive sample_rate, got {source_rate}")
        if len(audio) < 1:
            raise ValueError("PupuJEPA expects a non-empty batch")
        if audio.ndim == 3:
            audio = audio.mean(dim=1)
        elif audio.ndim != 2:
            raise ValueError(
                f"PupuJEPA audio must have shape (B, T) or (B, C, T), got {tuple(audio.shape)}"
            )
        expected_patches = pupujepa_num_time_patches(audio.shape[-1], source_rate, self.config)
        with torch.autocast(device_type=audio.device.type, enabled=False):
            waveform = audio.float()
            if source_rate != self.config.sample_rate:
                waveform = audio_functional.resample(
                    waveform,
                    source_rate,
                    self.config.sample_rate,
                )
            sequence = self.teacher_model(self.frontend(waveform))
        expected_shape = (len(audio), self.out_dim, expected_patches)
        if tuple(sequence.shape) != expected_shape:
            raise ValueError(
                f"PupuJEPA teacher produced shape {tuple(sequence.shape)}, expected "
                f"{expected_shape}"
            )
        if not torch.isfinite(sequence).all():
            raise ValueError("PupuJEPA teacher produced non-finite values")
        return sequence
