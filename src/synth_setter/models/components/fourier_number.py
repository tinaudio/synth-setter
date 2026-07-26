"""Exponentially spaced Fourier encoding of scalar parameter values.

Adapted from Stable Audio 3's ``ExpoFourierFeatures`` / ``NumberEmbedder``
(https://github.com/Stability-AI/stable-audio-3, commit 124e8a79, MIT licence,
Copyright (c) 2026 Stability AI), whose lineage runs through
``stable-audio-tools`` to ``archinetai/audio-diffusion-pytorch`` (MIT,
Copyright (c) 2022 archinet.ai).

Upstream feeds a clipped ``[0, 1]`` duration and spans frequencies up to
10 kHz. Flow parameter coordinates are unclipped and roughly ``[-4, 4]``, so
the default band stops far lower — see :class:`ExpoFourierFeatures`.
"""

import math

import torch
from torch import nn

# Upstream's 10 kHz ceiling decorrelates values 1e-3 apart, which makes the
# vector field stiff in x_t and degrades the fixed-step RK4 sample path.
DEFAULT_MAX_FREQ = 32.0
DEFAULT_MIN_FREQ = 0.5
DEFAULT_DIM = 256


class ExpoFourierFeatures(nn.Module):
    """Map scalars to cosine/sine features on a logarithmically spaced band.

    .. attribute :: frequencies

        Non-trainable band in cycles per unit input, ascending from ``min_freq``
        through ``max_freq`` inclusive.
    """

    # Declared so the registered buffer types as a Tensor rather than
    # nn.Module's `Tensor | Module` attribute union.
    frequencies: torch.Tensor

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        min_freq: float = DEFAULT_MIN_FREQ,
        max_freq: float = DEFAULT_MAX_FREQ,
    ) -> None:
        """Build the deterministic frequency band.

        :param dim: Output feature width, split evenly between cosine and sine.
        :param min_freq: Lowest frequency in cycles per unit input.
        :param max_freq: Highest frequency in cycles per unit input.
        :raises ValueError: If ``dim`` is odd or below 4, ``min_freq`` is not
            positive, or ``max_freq`` is below ``min_freq``.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        # Half-widths below 2 leave no interval to space frequencies across.
        if dim < 4:
            raise ValueError(f"dim must be at least 4, got {dim}")
        if min_freq <= 0.0:
            raise ValueError(f"min_freq must be positive, got {min_freq}")
        if max_freq < min_freq:
            raise ValueError(f"max_freq must be >= min_freq {min_freq}, got {max_freq}")

        half = dim // 2
        ramp = torch.arange(half, dtype=torch.float32) / (half - 1)
        frequencies = torch.exp(
            math.log(min_freq) + ramp * (math.log(max_freq) - math.log(min_freq))
        )
        # Non-persistent: the band is config-derived, so checkpoints stay
        # loadable when it is tuned to a different range.
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Encode every scalar in ``values`` independently.

        :param values: Floating-point tensor of any shape ``S``.
        :returns: Features of shape ``(*S, dim)``, cosine half then sine half,
            in the dtype and on the device of ``values``.
        :raises TypeError: If ``values`` is not a floating-point tensor.
        """
        if not values.is_floating_point():
            raise TypeError(f"values must be a floating-point tensor, got {values.dtype}")
        # Phase is evaluated in float32: bf16 rounding at the top of the band
        # exceeds the value spacing the encoding exists to resolve.
        angles = 2 * math.pi * values.to(torch.float32).unsqueeze(-1) * self.frequencies
        return torch.cat([angles.cos(), angles.sin()], dim=-1).to(values.dtype)


class FourierNumberEmbedder(nn.Module):
    """Project Fourier value features into a learned embedding width.

    .. attribute :: fourier_features

        Deterministic :class:`ExpoFourierFeatures` band.

    .. attribute :: projection

        The module's only trainable tensor.
    """

    fourier_features: ExpoFourierFeatures
    projection: nn.Linear

    def __init__(
        self,
        features: int,
        dim: int = DEFAULT_DIM,
        min_freq: float = DEFAULT_MIN_FREQ,
        max_freq: float = DEFAULT_MAX_FREQ,
    ) -> None:
        """Compose a frequency band with a learned output projection.

        :param features: Embedding width the projection emits.
        :param dim: Fourier feature width consumed by the projection.
        :param min_freq: Lowest frequency in cycles per unit input.
        :param max_freq: Highest frequency in cycles per unit input.
        """
        super().__init__()
        self.fourier_features = ExpoFourierFeatures(dim=dim, min_freq=min_freq, max_freq=max_freq)
        self.projection = nn.Linear(dim, features)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Embed every scalar in ``values`` independently.

        :param values: Floating-point tensor of any shape ``S``.
        :returns: Embeddings of shape ``(*S, features)``.
        """
        return self.projection(self.fourier_features(values))
