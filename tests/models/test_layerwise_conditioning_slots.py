"""Per-field-layer conditioning slots on the cached-embedding encoders."""

from collections.abc import Callable

import pytest
import torch
from torch import nn

from synth_setter.models.components.embed_pool import EmbeddingPool
from synth_setter.models.components.vector_projection import VectorProjection


def _pool(slots: int) -> EmbeddingPool:
    """Build a pooling head emitting the requested slot count.

    :param slots: Conditioning slots to allocate.
    :returns: Attention-pooling head.
    """
    return EmbeddingPool(
        embed_dim=4, d_model=6, num_heads=2, max_seq_len=3, n_conditioning_outputs=slots
    )


def _projection(slots: int) -> VectorProjection:
    """Build a vector projection emitting the requested slot count.

    :param slots: Conditioning slots to allocate.
    :returns: Fixed-width projection head.
    """
    return VectorProjection(input_dim=4, d_model=6, n_conditioning_outputs=slots)


@pytest.mark.parametrize(
    ("build", "sample"),
    [(_pool, torch.randn(2, 4, 3)), (_projection, torch.randn(2, 4))],
    ids=["embedding-pool", "vector-projection"],
)
def test_encoder_with_one_slot_keeps_the_pooled_rank_two_output(
    build: Callable[[int], nn.Module], sample: torch.Tensor
) -> None:
    """The single-slot default stays rank 2, which is what every current run consumes.

    :param build: Encoder factory under test.
    :param sample: Input batch shaped for that encoder.
    """
    assert build(1)(sample).shape == (2, 6)


@pytest.mark.parametrize(
    ("build", "sample"),
    [(_pool, torch.randn(2, 4, 3)), (_projection, torch.randn(2, 4))],
    ids=["embedding-pool", "vector-projection"],
)
def test_encoder_with_many_slots_emits_one_conditioning_row_per_slot(
    build: Callable[[int], nn.Module], sample: torch.Tensor
) -> None:
    """Layerwise output carries a slot axis the field indexes one layer at a time.

    :param build: Encoder factory under test.
    :param sample: Input batch shaped for that encoder.
    """
    assert build(5)(sample).shape == (2, 5, 6)


@pytest.mark.parametrize(
    ("build", "sample"),
    [(_pool, torch.randn(2, 4, 3)), (_projection, torch.randn(2, 4))],
    ids=["embedding-pool", "vector-projection"],
)
def test_encoder_slots_are_free_to_differ_from_one_another(
    build: Callable[[int], nn.Module], sample: torch.Tensor
) -> None:
    """Slots must not be tied copies, or the extra conditioning rows carry no signal.

    :param build: Encoder factory under test.
    :param sample: Input batch shaped for that encoder.
    """
    torch.manual_seed(0)
    slots = build(3)(sample)

    assert not torch.allclose(slots[:, 0], slots[:, 1])


@pytest.mark.parametrize(
    "build", [_pool, _projection], ids=["embedding-pool", "vector-projection"]
)
def test_encoder_rejects_a_non_positive_slot_count(build: Callable[[int], nn.Module]) -> None:
    """A zero or negative slot count cannot address a field layer.

    :param build: Encoder factory under test.
    """
    with pytest.raises(ValueError, match="n_conditioning_outputs"):
        build(0)


def test_single_slot_pool_keeps_the_pooled_query_shape() -> None:
    """Pooled checkpoints must load unadapted, so the slot-1 query shape cannot move."""
    assert tuple(_pool(1).state_dict()["query"].shape) == (1, 1, 6)


def test_single_slot_projection_keeps_the_pooled_weight_shapes() -> None:
    """Pooled checkpoints must load unadapted, so the slot-1 weight shapes cannot move."""
    state = _projection(1).state_dict()

    assert tuple(state["projection.weight"].shape) == (6, 4)
    assert tuple(state["projection.bias"].shape) == (6,)
