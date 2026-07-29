"""Zero-initialized sketch-control tokenizer for concat conditioning (#2612)."""

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from torch import nn

from synth_setter.conditioning import (
    NUM_SKETCH_CONTROLS,
    SKETCH_CENTROID_ROW,
    SKETCH_LOUDNESS_ROW,
    SKETCH_PITCH_SLICE,
)
from synth_setter.models.components.embed_pool import make_sin_pos_enc

_CONTROL_CHANNELS = {
    "loudness": slice(SKETCH_LOUDNESS_ROW, SKETCH_LOUDNESS_ROW + 1),
    "centroid": slice(SKETCH_CENTROID_ROW, SKETCH_CENTROID_ROW + 1),
    "pitch": SKETCH_PITCH_SLICE,
}
# Drop-mask column order; each name keys one projection and one channel group.
CONTROL_GROUPS = tuple(_CONTROL_CHANNELS)
# Pooling covers every stored frame (point-sampling would skip sub-stride
# transients); max for near-impulsive pitch activations, mean elsewhere.
_POOLING = {
    "loudness": F.adaptive_avg_pool1d,
    "centroid": F.adaptive_avg_pool1d,
    "pitch": F.adaptive_max_pool1d,
}


class SketchControlTokens(nn.Module):
    """Resample sketch controls to control tokens carrying a temporal PE.

    Each control group is zeroed when dropped, adaptively pooled from the
    stored frame grid to ``num_ctrl_tokens`` bins, and projected by a
    zero-initialized bias-free linear layer (FlashFoley ``input_add``): at
    initialization — and forever for a dropped control — the projections
    contribute exactly nothing, so training starts at the unconditioned model.
    The fixed sinusoidal encoding lives on the control tokens only, keeping
    the field's parameter tokens permutation-symmetric.

    .. attribute :: positional_encoding

        Fixed ``(1, num_ctrl_tokens, d_model)`` sinusoidal temporal encoding buffer.
    """

    positional_encoding: torch.Tensor

    @jaxtyped(typechecker=beartype)
    def __init__(self, d_model: int, num_ctrl_tokens: int = 32):
        """Build the per-control projections and the fixed temporal encoding.

        :param d_model: Vector-field token width the controls project into.
        :param num_ctrl_tokens: Control tokens the time axis is resampled to.
        """
        super().__init__()
        projections = {
            group: nn.Linear(channels.stop - channels.start, d_model, bias=False)
            for group, channels in _CONTROL_CHANNELS.items()
        }
        for projection in projections.values():
            nn.init.zeros_(projection.weight)
        self.projections = nn.ModuleDict(projections)
        self.register_buffer("positional_encoding", make_sin_pos_enc(num_ctrl_tokens, d_model))

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        controls: Float[torch.Tensor, f"batch {NUM_SKETCH_CONTROLS} frames"],
        drop_mask: Bool[torch.Tensor, f"batch {len(CONTROL_GROUPS)}"],
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
            channels = controls[:, _CONTROL_CHANNELS[group]]
            channels = channels * keep[:, group_index, None, None]
            resampled = _POOLING[group](channels, num_tokens)
            tokens = tokens + self.projections[group](resampled.transpose(1, 2))
        return tokens
