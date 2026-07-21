"""Behavioral tests for generic embedding encoders and model routing."""

from functools import partial

import pytest
import torch

from synth_setter.conditioning import ConditioningMode, EmbeddingConditioningSpec
from synth_setter.models.components.embed_pool import EmbeddingPool
from synth_setter.models.components.vector_projection import VectorProjection
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule


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
