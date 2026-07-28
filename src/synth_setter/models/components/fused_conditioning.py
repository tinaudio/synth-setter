"""Layerwise fusion of pooled m2l content with per-layer sketch control tokens."""

import math

import torch
import torch.nn as nn

from synth_setter.models.components.embed_pool import PosEnc


class SketchLayerwiseEncoder(nn.Module):
    """Encode a (B, num_controls, n_frames) control matrix into one token per layer.

    Patchifies the control tracks along time with a strided 1D convolution, adds
    a fixed sinusoidal positional encoding, and reads out ``num_layers`` learned
    query tokens through a single multi-head attention.
    """

    def __init__(
        self,
        num_controls: int,
        num_frames: int,
        d_model: int,
        num_layers: int,
        num_heads: int = 4,
        patch_size: int = 25,
    ):
        """Build the patchify, positional-encoding, and readout submodules.

        :param num_controls: Control tracks per row.
        :param num_frames: Maximum control frames covered by the positional encoding.
        :param d_model: Token width produced per layer slot.
        :param num_layers: Query tokens, one per vector-field layer.
        :param num_heads: Attention heads in the readout.
        :param patch_size: Frames per convolutional patch (also the stride).
        """
        super().__init__()
        self.patchify = nn.Conv1d(num_controls, d_model, patch_size, stride=patch_size)
        num_patches = max(1, (num_frames - patch_size) // patch_size + 1)
        self.positional_encoding = PosEnc(d_model, num_patches, "sin")
        self.norm = nn.LayerNorm(d_model)
        self.queries = nn.Parameter(torch.randn(1, num_layers, d_model) / math.sqrt(d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, controls: torch.Tensor) -> torch.Tensor:
        """Read layerwise tokens out of one control matrix.

        :param controls: Sketch controls of shape ``(batch, num_controls, n_frames)``.
        :returns: Tokens of shape ``(batch, num_layers, d_model)``.
        """
        tokens = self.patchify(controls).permute(0, 2, 1)
        tokens = self.norm(self.positional_encoding(tokens))
        queries = self.queries.expand(controls.shape[0], -1, -1)
        readout, _ = self.attn(queries, tokens, tokens)
        return readout


class FusedConditioningEncoder(nn.Module):
    """Fuse pooled m2l content with per-layer sketch tokens for layerwise adaLN.

    The m2l vector is broadcast to every layer slot and concatenated with the
    sketch encoder's per-layer tokens, yielding rank-3 ``(batch, num_layers,
    d_m2l + d_sketch)`` conditioning. Classifier-free-guidance dropout happens
    here (per modality, Sketch2Sound scheme), so the flow module must not apply
    the vector field's rank-2 dropout on top.

    .. attribute :: applies_cfg_dropout

        Read by ``VSTFlowMatchingModule`` to skip the vector field's rank-2
        CFG dropout.
    """

    applies_cfg_dropout = True

    def __init__(
        self,
        m2l_encoder: nn.Module,
        sketch_encoder: SketchLayerwiseEncoder,
        d_m2l: int,
        d_sketch: int,
        num_layers: int,
        modality_dropout_rate: float = 0.2,
        joint_dropout_rate: float = 0.2,
    ):
        """Wire both modality encoders and their learned null tokens.

        :param m2l_encoder: Encoder pooling the stored embedding to ``(batch, d_m2l)``.
        :param sketch_encoder: Encoder producing ``(batch, num_layers, d_sketch)`` tokens.
        :param d_m2l: Pooled m2l width.
        :param d_sketch: Per-layer sketch token width.
        :param num_layers: Layer slots; must match the vector field's depth.
        :param modality_dropout_rate: Independent per-modality drop probability.
        :param joint_dropout_rate: Probability of dropping both modalities together.
        """
        super().__init__()
        self.m2l_encoder = m2l_encoder
        self.sketch_encoder = sketch_encoder
        self.d_m2l = d_m2l
        self.d_sketch = d_sketch
        self.num_layers = num_layers
        self.modality_dropout_rate = modality_dropout_rate
        self.joint_dropout_rate = joint_dropout_rate
        self.m2l_null = nn.Parameter(torch.randn(1, d_m2l) / math.sqrt(d_m2l))
        self.sketch_null = nn.Parameter(torch.randn(1, 1, d_sketch) / math.sqrt(d_sketch))

    def _drop_modalities(
        self, m2l: torch.Tensor, sketch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace dropped modality rows with their learned null tokens.

        :param m2l: Pooled content of shape ``(batch, d_m2l)``.
        :param sketch: Layer tokens of shape ``(batch, num_layers, d_sketch)``.
        :returns: The pair with dropped rows nulled.
        """
        batch = m2l.shape[0]
        joint = torch.rand(batch, 1, device=m2l.device) < self.joint_dropout_rate
        drop_m2l = joint | (torch.rand(batch, 1, device=m2l.device) < self.modality_dropout_rate)
        drop_sketch = joint | (
            torch.rand(batch, 1, device=m2l.device) < self.modality_dropout_rate
        )
        m2l = torch.where(drop_m2l, self.m2l_null, m2l)
        sketch = torch.where(drop_sketch.unsqueeze(-1), self.sketch_null, sketch)
        return m2l, sketch

    def forward(self, conditioning: torch.Tensor, sketch_ctrl: torch.Tensor) -> torch.Tensor:
        """Fuse both modalities into rank-3 layerwise conditioning.

        :param conditioning: Stored embedding consumed by the m2l encoder.
        :param sketch_ctrl: Sketch controls of shape ``(batch, num_controls, n_frames)``.
        :returns: Conditioning of shape ``(batch, num_layers, d_m2l + d_sketch)``.
        :raises ValueError: If the sketch encoder's layer slots mismatch ``num_layers``.
        """
        m2l = self.m2l_encoder(conditioning)
        sketch = self.sketch_encoder(sketch_ctrl)
        if sketch.shape[1] != self.num_layers:
            raise ValueError(
                f"sketch encoder produced {sketch.shape[1]} layer slots, "
                f"expected {self.num_layers}"
            )
        if self.training:
            m2l, sketch = self._drop_modalities(m2l, sketch)
        broadcast = m2l.unsqueeze(1).expand(-1, self.num_layers, -1)
        return torch.cat((broadcast, sketch), dim=-1)

    def drop_sketch(self, fused: torch.Tensor) -> torch.Tensor:
        """Replace the sketch slice of fused conditioning with its null token.

        :param fused: Conditioning of shape ``(batch, num_layers, d_m2l + d_sketch)``.
        :returns: Conditioning with a nulled sketch slice.
        """
        sketch = self.sketch_null.expand(fused.shape[0], self.num_layers, -1)
        return torch.cat((fused[..., : self.d_m2l], sketch), dim=-1)

    def null_conditioning(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Build the all-null conditioning used by the unconditional CFG branch.

        :param batch_size: Rows to produce.
        :param device: Target device.
        :returns: Conditioning of shape ``(batch_size, num_layers, d_m2l + d_sketch)``.
        """
        m2l = self.m2l_null.to(device).expand(batch_size, -1)
        sketch = self.sketch_null.to(device).expand(batch_size, self.num_layers, -1)
        return torch.cat((m2l.unsqueeze(1).expand(-1, self.num_layers, -1), sketch), dim=-1)
