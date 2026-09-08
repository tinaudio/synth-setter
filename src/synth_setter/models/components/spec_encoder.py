"""Waveform-to-spectrogram front end and the encoder pairing it with a backbone.

Online-render synths have no stored mel column, so their conditioning encoder computes
features from the waveform. Keeping the front end separate from the backbone lets the
same spectrogram-in backbones serve both the stored-mel and the online path.

Example::

    encoder = SpecEncoder(
        frontend=LogMelFrontend(176_400, sample_rate=44_100),
        backbone=MelCNN(hidden_dim=16, out_dim=512),
    )
"""

import math
from collections.abc import Mapping
from typing import Any, Final, Literal

import torch
import torch.nn as nn
import torchaudio
from beartype import beartype
from jaxtyping import Float, Shaped, jaxtyped
from torch import Tensor

from synth_setter.data.vst.shapes import MEL_N_MELS, mel_hop_length, mel_n_fft
from synth_setter.data.vst_datamodule import load_mel_statistics

# Permissive so wrong-rank batches reach this module's own shape error rather than
# beartype's, which cannot report the expected sample count.
_BATCH_AUDIO_SHAPE: Final = "batch ... samples"
_BATCH_GRID_SHAPE: Final = "batch 1 mels frames"
_BATCH_ANY_SHAPE: Final = "batch ..."


class LogMelFrontend(nn.Module):
    """Convert fixed-length waveforms into the log-mel grid the dataset writers store."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        in_dim: int,
        *,
        sample_rate: int,
        center: bool = True,
        f_min: float = 0.0,
        f_max: float | None = None,
        n_fft: int | None = None,
        hop_length: int | None = None,
        n_mels: int = MEL_N_MELS,
        pad_mode: Literal["constant", "reflect"] = "constant",
        power: float = 2.0,
        mel_norm: Literal["slaney"] | None = "slaney",
        mel_scale: Literal["htk", "slaney"] = "slaney",
        window: Literal["hamming", "hann"] = "hamming",
        amin: float = 1e-10,
        top_db: float | None = 80.0,
        normalization_mean: float | list | None = None,
        normalization_std: float | list | None = None,
        normalization_stats_path: str | None = None,
    ) -> None:
        """Build the mel transform and the decibel scaling applied to its output.

        :param in_dim: Expected waveform length in samples.
        :param sample_rate: Waveform sample rate in Hz.
        :param center: Whether to pad waveforms so frames are centered on timestamps.
        :param f_min: Lowest frequency included in the mel filter bank, in Hz.
        :param f_max: Highest included frequency, in Hz; ``None`` selects Nyquist.
        :param n_fft: Fourier transform size; defaults to 25 ms of audio.
        :param hop_length: Frame stride; defaults to 100 frames per second.
        :param n_mels: Number of mel-frequency bins.
        :param pad_mode: Waveform padding mode used when ``center`` is enabled.
        :param power: Exponent applied to the magnitude spectrogram.
        :param mel_norm: Area normalization applied to mel filter-bank weights.
        :param mel_scale: Mel-frequency conversion formula.
        :param window: Window function applied before each Fourier transform.
        :param amin: Lower power bound used before converting to decibels.
        :param top_db: Dynamic range limit in decibels; ``None`` disables clipping.
        :param normalization_mean: Inline post-decibel mean, paired with ``normalization_std``.
        :param normalization_std: Inline post-decibel standard deviation, paired with the mean.
        :param normalization_stats_path: Local ``stats.npz`` used instead of inline statistics.
        :raises ValueError: If numeric bounds or normalization statistics are invalid.
        """
        super().__init__()
        if not math.isfinite(amin) or amin <= 0:
            raise ValueError(f"amin must be positive and finite, got {amin}")
        nyquist = sample_rate / 2
        if not math.isfinite(f_min) or not 0 <= f_min < nyquist:
            raise ValueError(f"f_min must be finite and below Nyquist, got {f_min}")
        if f_max is not None and (not math.isfinite(f_max) or not f_min < f_max <= nyquist):
            raise ValueError(
                f"f_max must be finite, above f_min, and no greater than Nyquist, got {f_max}"
            )
        for name, value in (("hop_length", hop_length), ("n_fft", n_fft), ("n_mels", n_mels)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not math.isfinite(power) or power <= 0:
            raise ValueError(f"power must be positive and finite, got {power}")
        if top_db is not None and (not math.isfinite(top_db) or top_db < 0):
            raise ValueError(f"top_db must be non-negative and finite, got {top_db}")
        window_fn = {"hamming": torch.hamming_window, "hann": torch.hann_window}[window]

        resolved_n_fft = n_fft if n_fft is not None else mel_n_fft(sample_rate)
        resolved_hop_length = hop_length if hop_length is not None else mel_hop_length(sample_rate)
        self.in_dim = in_dim
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            center=center,
            f_min=f_min,
            f_max=f_max,
            n_fft=resolved_n_fft,
            hop_length=resolved_hop_length,
            n_mels=n_mels,
            pad_mode=pad_mode,
            window_fn=window_fn,
            power=power,
            norm=mel_norm,
            mel_scale=mel_scale,
        )
        self.amin = amin
        self.db_multiplier = 20.0 / power
        self.top_db = top_db
        centered_padding = 2 * (resolved_n_fft // 2) if center else 0
        frame_count = 1 + (in_dim + centered_padding - resolved_n_fft) // resolved_hop_length
        self._normalization_shape = (1, n_mels, frame_count)
        buffer_options = {
            "device": self.mel.mel_scale.fb.device,
            "dtype": self.mel.mel_scale.fb.dtype,
        }
        self.register_buffer("normalization_mean", torch.empty(0, **buffer_options))
        self.register_buffer("normalization_std", torch.empty(0, **buffer_options))

        has_inline_mean = normalization_mean is not None
        has_inline_std = normalization_std is not None
        if has_inline_mean != has_inline_std:
            raise ValueError("normalization_mean and normalization_std must be provided together")
        if normalization_stats_path is not None and has_inline_mean:
            raise ValueError("normalization statistics must use either inline values or a file")
        if normalization_stats_path is not None:
            file_mean, file_std = load_mel_statistics(normalization_stats_path)
            self.set_normalization_statistics(
                torch.as_tensor(file_mean), torch.as_tensor(file_std)
            )
        elif has_inline_mean:
            self.set_normalization_statistics(
                torch.as_tensor(normalization_mean), torch.as_tensor(normalization_std)
            )

    @property
    @jaxtyped(typechecker=beartype)
    def normalization_enabled(self) -> bool:
        """Whether post-decibel standardization is configured."""
        return self.normalization_mean.numel() > 0

    @jaxtyped(typechecker=beartype)
    def set_normalization_statistics(
        self, mean: Shaped[Tensor, "..."], std: Shaped[Tensor, "..."]
    ) -> None:
        """Validate and store broadcastable post-decibel statistics.

        :param mean: Finite mean broadcastable to one raw feature grid.
        :param std: Finite positive standard deviation with compatible geometry.
        """
        mean, std = self._validated_normalization_statistics(mean, std)
        self.normalization_mean = mean
        self.normalization_std = std

    @jaxtyped(typechecker=beartype)
    def _validated_normalization_statistics(
        self, mean: Shaped[Tensor, "..."], std: Shaped[Tensor, "..."]
    ) -> tuple[Float[Tensor, "..."], Float[Tensor, "..."]]:
        """Return detached statistics compatible with this frontend's output geometry.

        :param mean: Candidate mean tensor.
        :param std: Candidate standard-deviation tensor.
        :returns: Statistics converted to the frontend's device and dtype.
        :raises ValueError: If values or shapes cannot safely normalize the feature grid.
        """
        if not torch.isfinite(mean).all():
            raise ValueError("normalization mean must contain only finite values")
        if not torch.isfinite(std).all():
            raise ValueError("normalization std must contain only finite values")
        if torch.any(std <= 0):
            raise ValueError("normalization std values must be positive")
        for name, value in (("mean", mean), ("std", std)):
            try:
                broadcast_shape = torch.broadcast_shapes(value.shape, self._normalization_shape)
            except RuntimeError as error:
                raise ValueError(
                    f"normalization {name} shape {tuple(value.shape)} cannot broadcast to "
                    f"{self._normalization_shape}"
                ) from error
            if broadcast_shape != self._normalization_shape:
                raise ValueError(
                    f"normalization {name} shape {tuple(value.shape)} cannot broadcast to "
                    f"{self._normalization_shape}"
                )
        options = {
            "device": self.mel.mel_scale.fb.device,
            "dtype": self.mel.mel_scale.fb.dtype,
        }
        stored_mean = mean.detach().to(**options).clone()
        stored_std = std.detach().to(**options).clone()
        if not torch.isfinite(stored_mean).all():
            raise ValueError("normalization mean must remain finite in the frontend dtype")
        if not torch.isfinite(stored_std).all():
            raise ValueError("normalization std must remain finite in the frontend dtype")
        if torch.any(stored_std <= 0):
            raise ValueError("normalization std must remain positive in the frontend dtype")
        return stored_mean, stored_std

    @jaxtyped(typechecker=beartype)
    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Restore variable-shape statistics while accepting legacy checkpoints.

        :param state_dict: Full checkpoint state mapping.
        :param prefix: Key prefix for this nested module.
        :param local_metadata: PyTorch load metadata for this module.
        :param strict: Whether incompatible keys are collected as load errors.
        :param missing_keys: Mutable missing-key collection.
        :param unexpected_keys: Mutable unexpected-key collection.
        :param error_msgs: Mutable tensor-copy error collection.
        :raises ValueError: If checkpoint statistics are incomplete or invalid.
        """
        mean_key = prefix + "normalization_mean"
        std_key = prefix + "normalization_std"
        has_mean = mean_key in state_dict
        has_std = std_key in state_dict
        if has_mean != has_std:
            raise ValueError("checkpoint normalization mean and std must both be present")
        if has_mean:
            loaded_mean = state_dict[mean_key]
            loaded_std = state_dict[std_key]
            if loaded_mean.numel() == 0 and loaded_std.numel() == 0:
                self.normalization_mean = self.normalization_mean.new_empty(0)
                self.normalization_std = self.normalization_std.new_empty(0)
            else:
                self.set_normalization_statistics(loaded_mean, loaded_std)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if not has_mean:
            self.normalization_mean = self.normalization_mean.new_empty(0)
            self.normalization_std = self.normalization_std.new_empty(0)
            missing_keys.remove(mean_key)
            missing_keys.remove(std_key)

    @jaxtyped(typechecker=beartype)
    def forward_raw(
        self, x: Float[Tensor, _BATCH_AUDIO_SHAPE]
    ) -> Float[Tensor, _BATCH_GRID_SHAPE]:
        """Return decibel-scaled mel grids before optional standardization.

        :param x: Waveforms shaped ``(batch, samples)``.
        :returns: Raw mel grids shaped ``(batch, 1, mels, frames)``.
        :raises ValueError: If the waveform shape differs from the configured input length.
        """
        if x.ndim != 2 or x.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected waveform shape (batch, {self.in_dim}), got {tuple(x.shape)}"
            )
        log_mel = self.db_multiplier * torch.log10(torch.clamp(self.mel(x), min=self.amin))
        log_mel = log_mel - log_mel.amax(dim=(-2, -1), keepdim=True)
        if self.top_db is not None:
            log_mel = torch.clamp(log_mel, min=-self.top_db)
        return log_mel.unsqueeze(1)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, _BATCH_AUDIO_SHAPE]) -> Float[Tensor, _BATCH_GRID_SHAPE]:
        """Return raw or standardized log-mel features for each waveform.

        :param x: Waveforms shaped ``(batch, samples)``.
        :returns: Mel grids shaped ``(batch, 1, mels, frames)``.
        """
        features = self.forward_raw(x)
        if not self.normalization_enabled:
            return features
        return (features - self.normalization_mean) / self.normalization_std


class SpecEncoder(nn.Module):
    """Encode waveforms by running a feature front end into a spectrogram-in backbone."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, *, frontend: nn.Module, backbone: nn.Module) -> None:
        """Pair a front end with the backbone that consumes its feature grid.

        :param frontend: Waveform-in module emitting ``(batch, channels, mels, frames)``.
        :param backbone: Spectrogram-in module producing the conditioning tensor.
        """
        super().__init__()
        self.frontend = frontend
        self.backbone = backbone

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, _BATCH_AUDIO_SHAPE]) -> Float[Tensor, _BATCH_ANY_SHAPE]:
        """Encode a mono waveform batch into the backbone's conditioning tensor.

        :param x: Waveforms shaped ``(batch, samples)``.
        :returns: Whatever the backbone emits for the front end's feature grid.
        """
        return self.backbone(self.frontend(x))
