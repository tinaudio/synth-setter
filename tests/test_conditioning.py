"""Tests for conditioning configuration contracts."""

from typing import cast, get_args

import pytest
import torch
from pydantic import ValidationError

from synth_setter.conditioning import (
    EMBEDDING_BATCH_KEY,
    RAW_CONDITIONING_MODES,
    ConditioningMode,
    EmbeddingConditioningSpec,
    conditioning_batch_key,
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


def test_audio_mode_resolves_to_no_embedding_spec() -> None:
    """Raw audio is not a Lance embedding column."""
    assert resolve_embedding_conditioning("audio") is None


@pytest.mark.parametrize("mode", sorted(RAW_CONDITIONING_MODES))
def test_raw_conditioning_mode_selects_the_batch_entry_it_names(mode: str) -> None:
    """A raw mode reads the batch entry spelled exactly like the mode.

    :param mode: Raw conditioning mode under test.
    """
    batch = {
        name: torch.full((2, 3), float(index))
        for index, name in enumerate(sorted(RAW_CONDITIONING_MODES))
    }

    selected = batch[conditioning_batch_key(cast(ConditioningMode, mode))]

    assert torch.equal(selected, batch[mode])


@pytest.mark.parametrize("mode", sorted(RAW_CONDITIONING_MODES))
def test_raw_conditioning_mode_is_a_declared_conditioning_literal(mode: str) -> None:
    """No raw mode may name a batch key the configuration cannot spell.

    :param mode: Raw conditioning mode under test.
    """
    assert mode in get_args(ConditioningMode)


@pytest.mark.parametrize(
    "conditioning",
    [
        "m2l",
        EmbeddingConditioningSpec(column="clap", input_shape=(512,)),
        {"column": "same_s", "input_shape": [512]},
    ],
)
def test_embedding_conditioning_selects_the_shared_embedding_key(
    conditioning: object,
) -> None:
    """Every cached embedding is read from one key, never from a raw mode's entry.

    :param conditioning: Embedding configuration under test.
    """
    key = conditioning_batch_key(conditioning)  # type: ignore[arg-type]

    assert key == EMBEDDING_BATCH_KEY
    assert key not in RAW_CONDITIONING_MODES


def test_unknown_conditioning_mode_cannot_fall_back_to_a_raw_key() -> None:
    """An unrecognized literal raises instead of silently resolving to a raw entry."""
    with pytest.raises(ValueError, match="unknown conditioning mode 'clap'"):
        conditioning_batch_key("clap")  # type: ignore[arg-type]
