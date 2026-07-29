"""Zero-initialized sketch-control tokenizer for concat conditioning (#2612)."""

import math

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from torch import nn

from synth_setter.conditioning import (
    NUM_SKETCH_CONTROLS,
    SKETCH_CENTROID_ROW,
    SKETCH_LOUDNESS_ROW,
    SKETCH_PITCH_BINS,
    SKETCH_PITCH_SLICE,
)

# Drop-mask column order; each name keys one projection and one channel group.
CONTROL_GROUPS = ("loudness", "centroid", "pitch")

_CHANNEL_SLICES = {
    "loudness": slice(SKETCH_LOUDNESS_ROW, SKETCH_LOUDNESS_ROW + 1),
    "centroid": slice(SKETCH_CENTROID_ROW, SKETCH_CENTROID_ROW + 1),
    "pitch": SKETCH_PITCH_SLICE,
}


def _sinusoidal_positional_encoding(num_positions: int, d_model: int) -> torch.Tensor:
    """Build the fixed transformer sinusoidal encoding.

    :param num_positions: Sequence length to encode.
    :param d_model: Embedding width; odd widths leave the last channel zero.
    :returns: ``(1, num_positions, d_model)`` encoding.
    """
    position = torch.arange(num_positions, dtype=torch.float32)[:, None]
    frequency = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(num_positions, d_model)
    pe[:, 0::2] = torch.sin(position * frequency)
    pe[:, 1::2] = torch.cos(position * frequency[: d_model // 2])
    return pe[None]


class SketchControlTokens(nn.Module):
    """Resample sketch controls to control tokens carrying a temporal PE.

    Each control group is zeroed when dropped, linearly resampled from the
    stored frame grid to ``num_ctrl_tokens`` positions, and projected by a
    zero-initialized bias-free linear layer (FlashFoley ``input_add``): at
    initialization — and forever for a dropped control — the projections
    contribute exactly nothing, so training starts at the unconditioned model.
    The fixed sinusoidal encoding lives on the control tokens only, keeping
    the field's parameter tokens permutation-symmetric.

    .. attribute :: positional_encoding

        Fixed ``(1, num_ctrl_tokens, d_model)`` sinusoidal temporal encoding buffer.
    """

    positional_encoding: torch.Tensor

    def __init__(self, d_model: int, num_ctrl_tokens: int = 32):
        """Build the per-control projections and the fixed temporal encoding.

        :param d_model: Vector-field token width the controls project into.
        :param num_ctrl_tokens: Control tokens the time axis is resampled to.
        """
        super().__init__()
        projections = {
            "loudness": nn.Linear(1, d_model, bias=False),
            "centroid": nn.Linear(1, d_model, bias=False),
            "pitch": nn.Linear(SKETCH_PITCH_BINS, d_model, bias=False),
        }
        for projection in projections.values():
            nn.init.zeros_(projection.weight)
        self.projections = nn.ModuleDict(projections)
        self.register_buffer(
            "positional_encoding",
            _sinusoidal_positional_encoding(num_ctrl_tokens, d_model),
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        controls: Float[torch.Tensor, f"batch {NUM_SKETCH_CONTROLS} frames"],
        drop_mask: Bool[torch.Tensor, "batch 3"],
    ) -> Float[torch.Tensor, "batch tokens d_model"]:
        """Tokenize a stored sketch-control batch.

        :param controls: Loudness, centroid, and pitch rows on the mel grid.
        :param drop_mask: Per-control CFG drop flags in ``CONTROL_GROUPS`` order;
            a dropped control's channels are zeroed before projection.
        :returns: Control tokens with the temporal encoding added.
        """
        keep = (~drop_mask).to(controls.dtype)
        num_tokens = self.positional_encoding.shape[1]
        tokens = self.positional_encoding.expand(controls.shape[0], -1, -1)
        for group_index, group in enumerate(CONTROL_GROUPS):
            channels = controls[:, _CHANNEL_SLICES[group]]
            channels = channels * keep[:, group_index, None, None]
            resampled = F.interpolate(channels, size=num_tokens, mode="linear", align_corners=True)
            tokens = tokens + self.projections[group](resampled.transpose(1, 2))
        return tokens
