"""Siamese arm and BYOL loss from the SLAP reference implementation.

Source: Pliploop/SLAP commit b49290186ee354d34798f9947110a375f9e3f5a7.
Paper: https://arxiv.org/abs/2506.17815.
"""

from __future__ import annotations

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torch.nn import functional

_BATCH_FEATURES = "batch features"
_BATCH_REPRESENTATION = "batch representation"


class SiameseArm(nn.Module):
    """Compose a backbone, projector, and online prediction transform."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module | None = None,
        transform: nn.Module | None = None,
        *,
        normalize_representations: bool = False,
        normalize_projections: bool = False,
        freeze_encoder: bool = False,
    ) -> None:
        """Build one online or target arm.

        :param encoder: Backbone mapping one modality into its latent space.
        :param projector: Optional latent-to-shared-space projection.
        :param transform: Optional online predictor applied after projection.
        :param normalize_representations: Whether to L2-normalize backbone outputs.
        :param normalize_projections: Whether to L2-normalize projections and predictions.
        :param freeze_encoder: Whether to exclude the backbone from gradient updates.
        """
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.transform = transform or nn.Identity()
        self.normalize_representations = normalize_representations
        self.normalize_projections = normalize_projections
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.encoder.requires_grad_(False)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, inputs: Float[Tensor, "batch ..."]
    ) -> tuple[
        Float[Tensor, _BATCH_REPRESENTATION],
        Float[Tensor, _BATCH_FEATURES] | None,
        Float[Tensor, _BATCH_FEATURES] | None,
    ]:
        """Return backbone, projected, and predicted representations.

        :param inputs: Batch accepted by the configured modality backbone.
        :returns: Latent representation followed by optional projection and prediction.
        """
        if self.freeze_encoder:
            self.encoder.eval()
        representation = self.encoder(inputs)
        if self.projector is None:
            return representation, None, None

        projection = self.projector(representation)
        prediction = self.transform(projection)
        if self.normalize_representations:
            representation = functional.normalize(representation)
        if self.normalize_projections:
            projection = functional.normalize(projection)
            prediction = functional.normalize(prediction)
        return representation, projection, prediction


class BYOLLoss(nn.Module):
    """Combine cross-modal and intra-modal cosine prediction losses."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        out_key: str = "all_loss",
        *,
        unimodal: bool = True,
        ssl_weight: float = 0.5,
    ) -> None:
        """Select the reference loss composition.

        :param out_key: Computed loss term exposed as ``total_loss``.
        :param unimodal: Whether to include within-modality prediction losses.
        :param ssl_weight: Weight assigned to cross-modal rather than within-modal loss.
        """
        super().__init__()
        self.out_key = out_key
        self.unimodal = unimodal
        self.ssl_weight = ssl_weight

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def forward_single(
        query: Float[Tensor, _BATCH_FEATURES],
        target: Float[Tensor, _BATCH_FEATURES],
    ) -> Float[Tensor, ""]:
        """Return mean normalized cosine prediction distance.

        :param query: Online normalized prediction vectors.
        :param target: Moving-average normalized projection vectors.
        :returns: Mean cosine distance across the batch.
        """
        return (2 - 2 * (query * target).sum(dim=-1)).mean()

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        q_a: Float[Tensor, _BATCH_FEATURES],
        q_t: Float[Tensor, _BATCH_FEATURES],
        z_a_ema: Float[Tensor, _BATCH_FEATURES],
        z_t_ema: Float[Tensor, _BATCH_FEATURES],
    ) -> dict[str, Float[Tensor, ""]]:
        """Return the reference SLAP loss terms and selected total.

        :param q_a: Online predictions from the first modality.
        :param q_t: Online predictions from the second modality.
        :param z_a_ema: Target projections from the first modality.
        :param z_t_ema: Target projections from the second modality.
        :returns: Cross-modal, within-modal, combined, and selected losses.
        """
        q_a = functional.normalize(q_a)
        q_t = functional.normalize(q_t)
        z_a_ema = functional.normalize(z_a_ema)
        z_t_ema = functional.normalize(z_t_ema)

        a_t_loss = self.forward_single(q_a, z_t_ema)
        t_a_loss = self.forward_single(q_t, z_a_ema)
        if self.unimodal:
            a_a_loss = self.forward_single(q_a, z_a_ema)
            t_t_loss = self.forward_single(q_t, z_t_ema)
        else:
            a_a_loss = torch.zeros_like(a_t_loss)
            t_t_loss = torch.zeros_like(t_a_loss)

        multimodal_loss = (a_t_loss + t_a_loss) / 2
        unimodal_loss = (a_a_loss + t_t_loss) / 2
        all_loss = self.ssl_weight * multimodal_loss + (1 - self.ssl_weight) * unimodal_loss
        losses = {
            "a_t_loss": a_t_loss,
            "t_a_loss": t_a_loss,
            "a_a_loss": a_a_loss,
            "t_t_loss": t_t_loss,
            "all_loss": all_loss,
            "multimodal_loss": multimodal_loss,
            "unimodal_loss": unimodal_loss,
        }
        losses["total_loss"] = losses[self.out_key]
        return losses
