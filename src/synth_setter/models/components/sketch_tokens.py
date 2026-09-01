"""Zero-initialized sketch-control tokenizer for concat conditioning (#2612)."""

import torch
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
from synth_setter.sketch import pool_sketch_controls

_CONTROL_CHANNELS = {
    "loudness": slice(SKETCH_LOUDNESS_ROW, SKETCH_LOUDNESS_ROW + 1),
    "centroid": slice(SKETCH_CENTROID_ROW, SKETCH_CENTROID_ROW + 1),
    "pitch": SKETCH_PITCH_SLICE,
}
# Drop-mask column order; each name keys one projection and one channel group.
CONTROL_GROUPS = tuple(_CONTROL_CHANNELS)


class SketchControlTokens(nn.Module):
    """Resample sketch controls to control tokens carrying a temporal PE.

    Each control group is zeroed unless kept, pooled to ``num_control_tokens``
    bins, and passed through a zero-initialized bias-free projection. Control
    extraction and per-group projection follow FlashFoley/Sketch2Sound; the
    concat-and-slice injection is U-ViT-style in-context conditioning. The fixed
    temporal encoding stays on control tokens so parameter tokens remain
    permutation-symmetric.

    .. attribute :: positional_encoding

        Fixed ``(1, num_control_tokens, d_model)`` sinusoidal temporal encoding buffer.
    """

    positional_encoding: torch.Tensor

    @jaxtyped(typechecker=beartype)
    def __init__(self, d_model: int, num_control_tokens: int = 32) -> None:
        """Build the per-control projections and the fixed temporal encoding.

        :param d_model: Vector-field token width the controls project into.
        :param num_control_tokens: Control tokens the time axis is resampled to.
        """
        super().__init__()
        projections = {
            group: nn.Linear(channels.stop - channels.start, d_model, bias=False)
            for group, channels in _CONTROL_CHANNELS.items()
        }
        for projection in projections.values():
            nn.init.zeros_(projection.weight)
        self.projections = nn.ModuleDict(projections)
        self.register_buffer("positional_encoding", make_sin_pos_enc(num_control_tokens, d_model))

    @jaxtyped(typechecker=beartype)
    def unconditional(self, batch_size: int) -> Float[torch.Tensor, "batch tokens d_model"]:
        """Return the PE-only control sequence used by the CFG unconditional branch.

        :param batch_size: Number of rows requiring unconditional tokens.
        :returns: Expanded ``(batch_size, num_control_tokens, d_model)`` positional encoding.
        """
        return self.positional_encoding.expand(batch_size, -1, -1)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        controls: Float[torch.Tensor, f"batch {NUM_SKETCH_CONTROLS} frames"],
        keep: Bool[torch.Tensor, f"batch {len(CONTROL_GROUPS)}"],
    ) -> Float[torch.Tensor, "batch tokens d_model"]:
        """Tokenize a stored sketch-control batch.

        :param controls: Loudness, centroid, and pitch rows on the mel grid.
        :param keep: Positive per-group keep state in ``CONTROL_GROUPS`` order;
            an absent group's channels are zeroed before projection.
        :returns: Control tokens with the temporal encoding added.
        """
        keep_values = keep.to(controls.dtype)
        pooled = pool_sketch_controls(controls, self.positional_encoding.shape[1])
        tokens = self.unconditional(controls.shape[0])
        for group_index, group in enumerate(CONTROL_GROUPS):
            channels = pooled[:, _CONTROL_CHANNELS[group]]
            channels = channels * keep_values[:, group_index, None, None]
            tokens = tokens + self.projections[group](channels.transpose(1, 2))
        return tokens
