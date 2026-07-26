"""Behavioral tests for generic embedding encoders and model routing."""

from collections.abc import Callable
from functools import partial

import pytest
import torch

from synth_setter.conditioning import ConditioningMode, EmbeddingConditioningSpec
from synth_setter.models.components.embed_pool import EmbeddingPool
from synth_setter.models.components.vector_projection import VectorProjection
from synth_setter.models.vst_ff_module import VSTFeedForwardModule
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule

_ModelFactory = Callable[[], VSTFeedForwardModule | VSTFlowMatchingModule]
_BatchFactory = Callable[[], dict[str, torch.Tensor]]


def _flow_module(
    conditioning: ConditioningMode | EmbeddingConditioningSpec,
) -> VSTFlowMatchingModule:
    """Build a tiny module for conditioning-key selection tests.

    :param conditioning: Legacy literal or embedding spec under test.
    :returns: Flow module with inert child networks.
    """
    return VSTFlowMatchingModule(
        encoder=torch.nn.Identity(),
        vector_field=torch.nn.Identity(),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=1,
        conditioning=conditioning,
    )


class _TinyVectorField(torch.nn.Module):
    """Minimal differentiable field for exercising flow training."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(7, 2)

    def apply_dropout(self, conditioning: torch.Tensor, dropout_rate: float) -> torch.Tensor:
        """Return conditioning unchanged for deterministic training.

        :param conditioning: Encoded conditioning vectors.
        :param dropout_rate: Unused classifier-free guidance dropout probability.
        :returns: The unchanged vectors.
        """
        return conditioning

    def forward(
        self, params: torch.Tensor, time: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        """Predict a field from parameters, time, and conditioning.

        :param params: Noisy parameter vectors.
        :param time: Sampled flow times.
        :param conditioning: Encoded conditioning vectors.
        :returns: Predicted parameter-space field.
        """
        return self.projection(torch.cat((params, time, conditioning), dim=1))

    def penalty(self) -> torch.Tensor:
        """Return a differentiable zero regularization penalty.

        :returns: Scalar zero connected to the field parameters.
        """
        return self.projection.weight.sum() * 0


def _ff_embedding_module() -> VSTFeedForwardModule:
    """Build a feed-forward module over a cached sequence embedding.

    :returns: Feed-forward module configured for generic conditioning.
    """
    return VSTFeedForwardModule(
        net=EmbeddingPool(embed_dim=4, d_model=2, num_heads=1, max_seq_len=3),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        conditioning=EmbeddingConditioningSpec(column="cached", input_shape=(4, 3)),
    )


def _flow_embedding_module() -> VSTFlowMatchingModule:
    """Build a flow module over a cached sequence embedding.

    :returns: Flow module configured for generic conditioning.
    """
    return VSTFlowMatchingModule(
        encoder=EmbeddingPool(embed_dim=4, d_model=4, num_heads=1, max_seq_len=3),
        vector_field=_TinyVectorField(),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=2,
        conditioning=EmbeddingConditioningSpec(column="cached", input_shape=(4, 3)),
    )


def _ff_embedding_batch() -> dict[str, torch.Tensor]:
    """Build a non-mel feed-forward training batch.

    :returns: Batch containing cached conditioning and target parameters.
    """
    return {
        "conditioning": torch.randn(2, 4, 3),
        "params": torch.randn(2, 2),
    }


def _flow_embedding_batch() -> dict[str, torch.Tensor]:
    """Build a non-mel flow training batch.

    :returns: Batch containing cached conditioning, targets, and flow noise.
    """
    return {
        "conditioning": torch.randn(2, 4, 3),
        "noise": torch.randn(2, 2),
        "params": torch.randn(2, 2),
    }


@pytest.mark.parametrize(
    ("module_factory", "batch_factory"),
    [
        (_flow_embedding_module, _flow_embedding_batch),
        (_ff_embedding_module, _ff_embedding_batch),
    ],
    ids=["flow", "feed_forward"],
)
def test_vst_module_training_step_cached_embedding_returns_finite_loss(
    module_factory: _ModelFactory,
    batch_factory: _BatchFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both inversion paths train directly from a cached non-mel tensor.

    :param module_factory: Flow or feed-forward module factory under test.
    :param batch_factory: Matching cached-conditioning batch factory.
    :param monkeypatch: Pytest fixture used to detach Lightning logging from a Trainer.
    """
    module = module_factory()
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)

    loss = module.training_step(batch_factory(), batch_idx=0)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in module.parameters())


def test_model_embedding_spec_reads_generic_conditioning_key() -> None:
    """Spec-driven models consume the canonical embedding tensor."""
    module = _flow_module(EmbeddingConditioningSpec(column="clap", input_shape=(5,)))
    expected = torch.randn(2, 5)

    actual = module._get_conditioning_from_batch(  # noqa: SLF001
        {"conditioning": expected, "mel_spec": torch.randn(2, 1)}
    )

    assert actual is expected


def test_model_legacy_m2l_hparams_stay_string_while_routing_generic() -> None:
    """Old checkpoint hparams remain m2l while selecting the canonical tensor."""
    module = _flow_module("m2l")
    expected = torch.randn(2, 128, 42)

    actual = module._get_conditioning_from_batch(  # noqa: SLF001
        {"conditioning": expected, "m2l": torch.randn_like(expected)}
    )

    assert module.hparams["conditioning"] == "m2l"
    assert actual is expected


def test_vector_projection_maps_fixed_vectors_to_output_width() -> None:
    """CLAP-style vectors retain their batch axis and receive the configured width."""
    encoder = VectorProjection(input_dim=7, d_model=11)

    output = encoder(torch.randn(3, 7))

    assert output.shape == (3, 11)


def test_vector_projection_wrong_input_width_raises() -> None:
    """A configured vector width mismatch fails with the shape in the message."""
    encoder = VectorProjection(input_dim=7, d_model=11)

    with pytest.raises(ValueError, match=r"expected .*7.*got .*8"):
        encoder(torch.randn(3, 8))


def test_embedding_pool_seq_len_configurable() -> None:
    """A fixed sequence longer than the legacy default pools when configured."""
    encoder = EmbeddingPool(
        embed_dim=8,
        d_model=12,
        num_heads=3,
        max_seq_len=64,
    )

    assert encoder(torch.randn(2, 8, 64)).shape == (2, 12)


def test_embedding_pool_default_42_unchanged() -> None:
    """Omitting max_seq_len preserves the legacy 42-position contract."""
    encoder = EmbeddingPool(embed_dim=8, d_model=12, num_heads=3)

    assert encoder(torch.randn(2, 8, 42)).shape == (2, 12)
    with pytest.raises(ValueError, match=r"sequence length 43 exceeds max_seq_len 42"):
        encoder(torch.randn(2, 8, 43))
