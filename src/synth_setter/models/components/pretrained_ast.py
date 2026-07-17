"""Adapt a Hugging Face AST backbone to the flow-conditioning contract.

The input bridge maps normalized stereo mel to HF's mono time-major layout; resized position
embeddings avoid padding every sample to 1024 frames. Learned cross-attention queries produce one
conditioning token per vector-field layer.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from transformers import ASTModel

DEFAULT_CHECKPOINT: Final = "MIT/ast-finetuned-audioset-10-10-0.4593"


def interpolate_time_position_embeddings(backbone: ASTModel, max_length: int) -> None:
    """Resize the time position grid while preserving cls/distillation embeddings.

    :param backbone: Hugging Face AST model to adapt in place.
    :param max_length: Frame count the adapted backbone should accept.
    """
    config = backbone.config
    f_dim, t_dim_old = backbone.embeddings.get_shape(config)
    config.max_length = max_length
    _, t_dim_new = backbone.embeddings.get_shape(config)

    pos = backbone.embeddings.position_embeddings
    special, patches = pos[:, :2], pos[:, 2:]
    grid = patches.reshape(1, f_dim, t_dim_old, -1).permute(0, 3, 1, 2)
    grid = nn.functional.interpolate(
        grid, size=(f_dim, t_dim_new), mode="bilinear", align_corners=False
    )
    patches = grid.permute(0, 2, 3, 1).reshape(1, f_dim * t_dim_new, -1)
    backbone.embeddings.position_embeddings = nn.Parameter(torch.cat([special, patches], dim=1))


class PretrainedASTEncoder(nn.Module):
    """Flow-conditioning encoder over a (optionally pretrained) HF AST backbone."""

    def __init__(
        self,
        d_model: int = 768,
        n_conditioning_outputs: int = 8,
        n_pool_heads: int = 8,
        checkpoint: str = DEFAULT_CHECKPOINT,
        pretrained: bool = True,
        freeze: bool = False,
        spec_shape: tuple[int, int] = (128, 401),
        backbone_config: dict | None = None,
    ):
        """Build the backbone and adaptation layers.

        :param d_model: Backbone width exposed to Hydra's ``conditioning_dim``.
        :param n_conditioning_outputs: Number of vector-field layers to condition.
        :param n_pool_heads: Attention heads in the output query pool.
        :param checkpoint: Hugging Face checkpoint loaded in pretrained mode.
        :param pretrained: Load checkpoint weights instead of an offline random backbone.
        :param freeze: Freeze only the backbone, leaving adaptation layers trainable.
        :param spec_shape: Input geometry as ``(n_mels, n_frames)``.
        :param backbone_config: ``ASTConfig`` overrides used in offline mode.
        :raises ValueError: Backbone geometry, width, or mel bins violate the input contract.
        """
        super().__init__()
        from transformers import ASTConfig, ASTModel

        n_mels, n_frames = spec_shape
        if pretrained:
            if backbone_config is not None:
                raise ValueError("backbone_config requires pretrained=False")
            backbone = ASTModel.from_pretrained(checkpoint)
            interpolate_time_position_embeddings(backbone, max_length=n_frames)
        else:
            config_overrides = backbone_config or {}
            reserved_keys = sorted({"max_length", "num_mel_bins"} & config_overrides.keys())
            if reserved_keys:
                names = ", ".join(reserved_keys)
                raise ValueError(f"backbone_config keys {names} are derived from spec_shape")
            config = ASTConfig(
                num_mel_bins=n_mels,
                max_length=n_frames,
                **config_overrides,
            )
            backbone = ASTModel(config)

        if backbone.config.hidden_size != d_model:
            raise ValueError(
                f"d_model {d_model} does not match backbone hidden size "
                f"{backbone.config.hidden_size}"
            )
        if backbone.config.num_mel_bins != n_mels:
            raise ValueError(
                f"spec_shape mel bins {n_mels} do not match backbone "
                f"num_mel_bins {backbone.config.num_mel_bins}"
            )

        self.backbone = backbone
        if freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

        self.input_scale = nn.Parameter(torch.ones(1))
        self.input_shift = nn.Parameter(torch.zeros(1))
        self.queries = nn.Parameter(
            torch.randn(1, n_conditioning_outputs, d_model) / math.sqrt(d_model)
        )
        self.pool = nn.MultiheadAttention(d_model, n_pool_heads, batch_first=True)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode a mel batch into per-layer conditioning tokens.

        :param mel: Stats-normalized dB mel, ``(B, 2, n_mels, n_frames)``.
        :returns: Conditioning tokens ``(B, n_conditioning_outputs, d_model)``.
        """
        x = mel.mean(dim=1).transpose(1, 2)
        x = x * self.input_scale + self.input_shift

        hidden = self.backbone(input_values=x).last_hidden_state

        queries = self.queries.expand(x.shape[0], -1, -1)
        pooled, _ = self.pool(queries, hidden, hidden)
        return pooled
