"""Zero-initialized sketch-control tokenizer for concat conditioning (#2612)."""

import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from torch import nn

from synth_setter.conditioning import (
    SKETCH_STORAGE_FRAMES,
    SketchControlLayout,
    SketchControlProfile,
    sketch_control_layout,
)
from synth_setter.models.components.embed_pool import make_sin_pos_enc
from synth_setter.sketch import pool_sketch_controls

_MUSIC_LAYOUT = sketch_control_layout("music")
# Compatibility export for callers that construct music-profile masks directly.
CONTROL_GROUPS = _MUSIC_LAYOUT.group_names


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
    def __init__(
        self,
        d_model: int,
        num_control_tokens: int = 32,
        profile: SketchControlProfile = "music",
    ) -> None:
        """Build the profile projections and fixed temporal encoding.

        :param d_model: Vector-field token width the controls project into.
        :param num_control_tokens: Control tokens retained along the time axis.
        :param profile: Channel grouping and temporal processing contract.
        :raises ValueError: The reverb profile receives a noncanonical token count.
        """
        super().__init__()
        self.layout: SketchControlLayout = sketch_control_layout(profile)
        if profile == "pyfdn_reverb" and num_control_tokens != SKETCH_STORAGE_FRAMES:
            raise ValueError(
                f"pyfdn_reverb sketch requires {SKETCH_STORAGE_FRAMES} control tokens"
            )
        projections = {
            group: nn.Linear(width, d_model, bias=False)
            for group, width in zip(self.layout.group_names, self.layout.group_widths, strict=True)
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
        controls: Float[torch.Tensor, "batch channels frames"],
        keep: Bool[torch.Tensor, "batch groups"],
    ) -> Float[torch.Tensor, "batch tokens d_model"]:
        """Tokenize a stored sketch-control batch.

        :param controls: Profile channels arranged as ``(batch, channels, frames)``.
        :param keep: Positive per-group keep state in layout order.
        :returns: Control tokens with the temporal encoding added.
        """
        self._validate_inputs(controls, keep)
        keep_values = keep.to(controls.dtype)
        temporal_controls = (
            pool_sketch_controls(controls, self.positional_encoding.shape[1])
            if self.layout.profile == "music"
            else controls
        )
        tokens = self.unconditional(controls.shape[0])
        for group_index, (group, channel_slice) in enumerate(
            zip(self.layout.group_names, self.layout.group_slices, strict=True)
        ):
            channels = temporal_controls[:, channel_slice]
            channels = channels * keep_values[:, group_index, None, None]
            tokens = tokens + self.projections[group](channels.transpose(1, 2))
        return tokens

    @jaxtyped(typechecker=beartype)
    def _validate_inputs(
        self,
        controls: Float[torch.Tensor, "batch channels frames"],
        keep: Bool[torch.Tensor, "batch groups"],
    ) -> None:
        """Reject inputs that do not match the selected profile layout.

        :param controls: Profile control tensor.
        :param keep: Per-group positive keep state.
        :raises ValueError: Channel, group, dtype, or temporal dimensions violate the profile.
        """
        expected_channels = self.layout.num_controls
        if controls.shape[1] != expected_channels:
            raise ValueError(
                f"{self.layout.profile} sketch requires {expected_channels} channels, "
                f"got {controls.shape[1]}"
            )
        if keep.shape[1] != len(self.layout.group_names):
            raise ValueError(
                f"{self.layout.profile} sketch requires {len(self.layout.group_names)} "
                f"keep groups, got {keep.shape[1]}"
            )
        if self.layout.profile == "pyfdn_reverb":
            expected_frames = self.positional_encoding.shape[1]
            if controls.dtype != torch.float32:
                raise ValueError("pyfdn_reverb sketch controls must be float32")
            if controls.shape[2] != expected_frames:
                raise ValueError(
                    f"pyfdn_reverb sketch requires {expected_frames} frames, "
                    f"got {controls.shape[2]}"
                )
