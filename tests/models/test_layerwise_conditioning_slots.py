"""Per-field-layer conditioning slots on the cached-embedding encoders."""

import math
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


_ENCODER_CASES = (
    pytest.param(_pool, (2, 4, 3), id="embedding-pool"),
    pytest.param(_projection, (2, 4), id="vector-projection"),
)


@pytest.mark.parametrize(("build", "sample_shape"), _ENCODER_CASES)
def test_encoder_with_one_slot_keeps_the_pooled_rank_two_output(
    build: Callable[[int], nn.Module], sample_shape: tuple[int, ...]
) -> None:
    """The single-slot default stays rank 2, which is what every current run consumes.

    :param build: Encoder factory under test.
    :param sample_shape: Input batch shape accepted by the encoder.
    """
    assert build(1)(torch.zeros(sample_shape)).shape == (2, 6)


def test_vector_projection_legacy_checkpoint_without_slot_metadata_keeps_rank_two() -> None:
    """Pre-slot checkpoints retain their original single-output projection contract."""
    encoder = _projection(1)
    del encoder.n_conditioning_outputs
    del encoder.d_model

    assert encoder(torch.zeros(2, 4)).shape == (2, 6)


@pytest.mark.parametrize(("build", "sample_shape"), _ENCODER_CASES)
def test_encoder_with_many_slots_emits_one_conditioning_row_per_slot(
    build: Callable[[int], nn.Module], sample_shape: tuple[int, ...]
) -> None:
    """Layerwise output carries a slot axis the field indexes one layer at a time.

    :param build: Encoder factory under test.
    :param sample_shape: Input batch shape accepted by the encoder.
    """
    assert build(5)(torch.zeros(sample_shape)).shape == (2, 5, 6)


@pytest.mark.parametrize(("build", "sample_shape"), _ENCODER_CASES)
def test_encoder_slots_are_free_to_differ_from_one_another(
    build: Callable[[int], nn.Module], sample_shape: tuple[int, ...]
) -> None:
    """Slots must not be tied copies, or the extra conditioning rows carry no signal.

    :param build: Encoder factory under test.
    :param sample_shape: Input batch shape accepted by the encoder.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        encoder = build(3)
    slots = encoder(torch.zeros(sample_shape))

    assert not torch.equal(slots[:, 0], slots[:, 1])


@pytest.mark.parametrize(("build", "sample_shape"), _ENCODER_CASES)
@pytest.mark.parametrize("slot_index", range(3))
def test_every_encoder_slot_depends_on_the_input(
    build: Callable[[int], nn.Module], sample_shape: tuple[int, ...], slot_index: int
) -> None:
    """Every conditioning slot carries input-dependent signal.

    :param build: Encoder factory under test.
    :param sample_shape: Input shape accepted by the encoder.
    :param slot_index: Conditioning slot whose dependency is checked.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        encoder = build(3).eval()
    single_row_shape = (1, *sample_shape[1:])
    sample = torch.linspace(-0.5, 0.5, math.prod(single_row_shape)).reshape(single_row_shape)

    jacobian = torch.autograd.functional.jacobian(
        lambda value: encoder(value)[0, slot_index], sample
    )

    assert torch.count_nonzero(jacobian).item() > 0


@pytest.mark.parametrize(
    ("build", "slots"),
    [
        pytest.param(_pool, 0, id="embedding-pool-zero"),
        pytest.param(_pool, -1, id="embedding-pool-negative"),
        pytest.param(_projection, 0, id="vector-projection-zero"),
        pytest.param(_projection, -1, id="vector-projection-negative"),
    ],
)
def test_encoder_non_positive_slot_count_raises_value_error(
    build: Callable[[int], nn.Module], slots: int
) -> None:
    """A zero or negative slot count cannot address a field layer.

    :param build: Encoder factory under test.
    :param slots: Invalid conditioning-slot count.
    """
    with pytest.raises(ValueError, match="n_conditioning_outputs"):
        build(slots)


def test_single_slot_pool_keeps_the_pooled_query_shape() -> None:
    """Pooled checkpoints must load unadapted, so the slot-1 query shape cannot move."""
    assert tuple(_pool(1).state_dict()["query"].shape) == (1, 1, 6)


def test_single_slot_projection_keeps_the_pooled_weight_shapes() -> None:
    """Pooled checkpoints must load unadapted, so the slot-1 weight shapes cannot move."""
    state = _projection(1).state_dict()

    assert tuple(state["projection.weight"].shape) == (6, 4)
    assert tuple(state["projection.bias"].shape) == (6,)
