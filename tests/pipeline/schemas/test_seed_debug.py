"""Tests for typed row-level seed debug documents."""

import json

import pytest
from pydantic import ValidationError

from synth_setter.pipeline.schemas.seed_debug import SeedDebugDocument


def test_seed_debug_document_json_round_trip_omits_absent_parameter_fields() -> None:
    """Natural JSON serialization preserves the document without null optional fields."""
    document = SeedDebugDocument(
        seed=17,
        master_seed=42,
        sample_idx=9,
        attempt=2,
        shard_id=7,
        parameter_source="sampled",
    )

    encoded = document.model_dump_json(exclude_none=True)

    assert SeedDebugDocument.model_validate_json(encoded) == document
    assert "parameter_seed" not in encoded


def test_seed_debug_document_without_consumed_seed_omits_seed() -> None:
    """Rows rendered from fixed parameters have no concrete sampler seed."""
    document = SeedDebugDocument(
        master_seed=42,
        sample_idx=9,
        attempt=0,
        shard_id=7,
        parameter_source="fixed",
    )

    encoded = document.model_dump_json(exclude_none=True)

    assert SeedDebugDocument.model_validate_json(encoded) == document
    assert "seed" not in json.loads(encoded)


def test_seed_debug_document_invalid_parameter_source_raises_validation_error() -> None:
    """Parameter provenance accepts only the supported source categories."""
    with pytest.raises(ValidationError, match="parameter_source"):
        SeedDebugDocument.model_validate(
            {
                "seed": 17,
                "master_seed": 42,
                "sample_idx": 9,
                "attempt": 2,
                "shard_id": 7,
                "parameter_source": "unknown",
            }
        )


def test_seed_debug_document_partial_parameter_provenance_raises_validation_error() -> None:
    """Reused-parameter provenance is either complete or absent."""
    with pytest.raises(ValidationError, match="must be provided together"):
        SeedDebugDocument(
            seed=17,
            master_seed=42,
            sample_idx=10,
            attempt=1,
            shard_id=7,
            parameter_source="sampled",
            parameter_seed=23,
            parameter_sample_idx=9,
        )
