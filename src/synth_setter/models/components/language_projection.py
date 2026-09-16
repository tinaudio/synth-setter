"""Grouped parameter tokens with a shared, zero-initialized language correction."""

import hashlib
from pathlib import Path
from typing import Literal

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from synth_setter.models.components.transformer import GroupedParameterProjection
from synth_setter.pipeline.data.param_language import (
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    describe_fields,
    load_param_name_embeddings,
)


class LanguageParameterProjection(GroupedParameterProjection):
    """Fuse static field meanings with numeric tokens without constraining decoded velocities.

    .. attribute :: language_embeddings

        Frozen field-major vectors persisted with the checkpoint.
    """

    language_embeddings: Float[Tensor, "num_tokens embedding_dim"]
    _language_ready: bool

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        d_model: int,
        param_spec_name: str,
        synth_name: str,
        *,
        embedding_dim: int = 128,
        embedding_path: str | None = None,
        embedding_source: Literal["language", "random", "learned"] = "language",
        embedding_seed: int = 0,
    ) -> None:
        """Build grouped numeric heads and shared residual language fusion.

        :param d_model: Transformer token width.
        :param param_spec_name: Registered field layout.
        :param synth_name: Dataset synth identity.
        :param embedding_dim: Finalized metadata width.
        :param embedding_path: Local artifact consumed by explicit initialization before forward.
        :param embedding_source: Language vectors or matched frozen/trainable identity controls.
        :param embedding_seed: Independent control-vector seed, without changing backbone RNG.
        :raises ValueError: Embedding width is unsupported.
        """
        super().__init__(d_model, param_spec_name)
        if embedding_dim not in (128, 256, 512, 768):
            raise ValueError("unsupported parameter language embedding dimension")
        self.param_spec_name = param_spec_name
        self.synth_name = synth_name
        self.embedding_dim = embedding_dim
        self.embedding_path = embedding_path
        self.embedding_source = embedding_source
        self.embedding_seed = embedding_seed
        vectors = torch.zeros(self.num_tokens, embedding_dim)
        if embedding_source != "language":
            generator = torch.Generator().manual_seed(embedding_seed)
            vectors = nn.functional.normalize(
                torch.randn(vectors.shape, generator=generator), dim=-1
            )
        if embedding_source == "learned":
            self.language_embeddings = nn.Parameter(vectors)
        else:
            self.register_buffer("language_embeddings", vectors)
        self._language_ready = embedding_source != "language"
        # Preserve identical downstream backbone initialization across ablation treatments.
        with torch.random.fork_rng(devices=[]):
            self.text_adapter = nn.Linear(embedding_dim, d_model)
            residual_output = nn.Linear(d_model, d_model)
            nn.init.zeros_(residual_output.weight)
            nn.init.zeros_(residual_output.bias)
            self.fusion = nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.GELU(), residual_output
            )

    @jaxtyped(typechecker=beartype)
    def get_extra_state(self) -> dict[str, str]:
        """Bind checkpoint vectors to the current field semantics and dimension.

        :returns: Identity checked before checkpoint field vectors can be consumed.
        """
        descriptions = describe_fields(self.param_spec_name, self.synth_name)
        return {
            "description_sha256": hashlib.sha256("\n".join(descriptions).encode()).hexdigest(),
            "embedding_dim": str(self.embedding_dim),
            "model": EMBEDDING_MODEL,
            "revision": EMBEDDING_REVISION,
            "initialized": str(self._language_ready),
            "embedding_source": self.embedding_source,
            "embedding_seed": str(self.embedding_seed),
        }

    @jaxtyped(typechecker=beartype)
    def set_extra_state(self, state: dict[str, str]) -> None:
        """Reject checkpoints whose field semantics differ despite compatible tensor shapes.

        :param state: Saved semantic identity.
        :raises ValueError: Saved descriptions or dimension differ from this projection.
        """
        identity = dict(state)
        initialized = identity.pop("initialized", None)
        expected = self.get_extra_state()
        expected.pop("initialized")
        if identity != expected or initialized not in {"True", "False"}:
            raise ValueError("parameter language checkpoint does not match the current spec")
        self._language_ready = initialized == "True"

    @jaxtyped(typechecker=beartype)
    def initialize_embeddings(self) -> None:
        """Load and validate metadata at the caller's initialization boundary, never in forward.

        :raises ValueError: No artifact is supplied or its width differs from the configured width.
        """
        if self._language_ready:
            return
        if self.embedding_path is None:
            raise ValueError("fresh language projection requires a finalized embedding_path")
        embeddings, metadata = load_param_name_embeddings(
            Path(self.embedding_path), self.param_spec_name, self.synth_name
        )
        if metadata.dimension != self.embedding_dim:
            raise ValueError("artifact dimension does not match projection embedding_dim")
        self.language_embeddings.copy_(torch.from_numpy(embeddings).to(self.language_embeddings))
        self._language_ready = True

    @torch.compiler.disable()
    @jaxtyped(typechecker=beartype)
    def param_to_token(
        self, params: Float[Tensor, "batch num_params"]
    ) -> Float[Tensor, "batch num_tokens d_model"]:
        """Add a learned language-conditioned correction to each grouped numeric token.

        :param params: Current flat parameter states, including unconstrained flow states.
        :returns: One token per logical field.
        :raises ValueError: Initialization or checkpoint restoration has not supplied embeddings.
        """
        if not self._language_ready:
            raise ValueError("call initialize_embeddings with an embedding_path before forward")
        values = super().param_to_token(params)
        semantics = self.text_adapter(self.language_embeddings).unsqueeze(0).expand_as(values)
        return values + self.fusion(torch.cat((values, semantics), dim=-1))
