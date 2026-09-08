"""Parameter-vector embeddings with the feed-forward AST projection head.

Example::

    projection = LearntProjection(768, 768, num_params=300, num_tokens=128)
    encoder = VSTFeedForwardParamEncoder(projection)
    embeddings = encoder(params)
"""

from beartype import beartype
from jaxtyping import jaxtyped

from synth_setter.models.components.transformer import (
    ASTWithProjectionHead,
    LearntProjection,
    ParamTokenEmbed,
)


class VSTFeedForwardParamEncoder(ASTWithProjectionHead):
    """Encode flat VST parameters into one embedding per example."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        projection: LearntProjection,
        d_model: int = 768,
        d_out: int = 512,
        n_heads: int = 16,
        n_layers: int = 12,
    ) -> None:
        """Use the projection's encoder half and freeze its unused decoder.

        :param projection: Parameter-to-token assignment producing ``d_model``-wide tokens.
        :param d_model: Shared token, transformer, and hidden head width.
        :param d_out: Output embedding width.
        :param n_heads: Attention heads per layer; must divide ``d_model``.
        :param n_layers: Transformer depth.
        """
        super().__init__(
            d_model=d_model,
            d_out=d_out,
            n_heads=n_heads,
            n_layers=n_layers,
            token_embed=ParamTokenEmbed(projection),
        )
