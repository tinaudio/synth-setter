"""Tests for conditioning configuration contracts."""

import pytest
from pydantic import ValidationError

from synth_setter.conditioning import (
    EmbeddingConditioningSpec,
    resolve_embedding_conditioning,
)


def test_embedding_conditioning_spec_accepts_fixed_shape() -> None:
    """A column and positive per-row dimensions form an immutable strict spec."""
    spec = EmbeddingConditioningSpec(column="clap", input_shape=(512,))

    assert spec.column == "clap"
    assert spec.input_shape == (512,)


def test_embedding_conditioning_spec_rejects_extra_fields() -> None:
    """Unknown configuration cannot silently cross the conditioning boundary."""
    with pytest.raises(ValidationError, match="unexpected"):
        EmbeddingConditioningSpec(
            column="clap",
            input_shape=(512,),
            unexpected=True,  # type: ignore[call-arg]
        )


@pytest.mark.parametrize("input_shape", [(), (0,), (-1,), (128, 42.0)])
def test_embedding_conditioning_spec_rejects_invalid_shape(
    input_shape: tuple[object, ...],
) -> None:
    """Shapes must contain one or more strictly typed positive integers.

    :param input_shape: Empty, non-positive, or non-integer shape under test.
    """
    with pytest.raises(ValidationError):
        EmbeddingConditioningSpec.model_validate(
            {"column": "embedding", "input_shape": input_shape}
        )


def test_resolve_embedding_conditioning_m2l_returns_legacy_spec() -> None:
    """The legacy m2l literal resolves without changing its public spelling."""
    spec = resolve_embedding_conditioning("m2l")

    assert spec == EmbeddingConditioningSpec(column="music2latent", input_shape=(128, 42))


def test_resolve_embedding_conditioning_hydra_mapping_accepts_list_shape() -> None:
    """Hydra's list-shaped container is normalized before strict validation."""
    spec = resolve_embedding_conditioning({"column": "clap", "input_shape": [512]})

    assert spec == EmbeddingConditioningSpec(column="clap", input_shape=(512,))


def test_resolve_embedding_conditioning_mel_returns_none() -> None:
    """Legacy mel remains outside generic embedding routing."""
    assert resolve_embedding_conditioning("mel") is None


def test_resolve_embedding_conditioning_unknown_literal_raises() -> None:
    """Unsupported legacy literals fail at the strict routing boundary."""
    with pytest.raises(ValueError, match="unknown conditioning mode 'clap'"):
        resolve_embedding_conditioning("clap")  # type: ignore[arg-type]
