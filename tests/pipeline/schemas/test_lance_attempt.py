"""Trust-boundary tests for the Lance staging/audit Pydantic contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synth_setter.pipeline.schemas.lance_attempt import (
    EmbeddingProvenance,
    EmbeddingSplitProvenance,
    LanceDatasetCard,
    LanceFragmentSidecar,
    SelectedLanceAttempt,
)


def _card() -> LanceDatasetCard:
    """Build a minimal one-shard card for round-trip and mutation tests.

    :returns: Card with one selected attempt.
    """
    return LanceDatasetCard(
        schema_version=1,
        run_id="run-1",
        finalized_at="2026-07-09T00:00:00+00:00",
        selected_attempts=(
            SelectedLanceAttempt(shard_id=0, attempt="pod-a-u0", valid_key="k/.valid"),
        ),
    )


def test_sidecar_rejects_unknown_fields() -> None:
    """Extra keys fail the sidecar's ``extra=\"forbid\"`` boundary."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        LanceFragmentSidecar.model_validate(
            {"schema_version": 1, "fragment_json": "{}", "shard_id": 3}
        )


def test_sidecar_rejects_unknown_schema_version() -> None:
    """Only the literal current schema version validates."""
    with pytest.raises(ValidationError, match="schema_version"):
        LanceFragmentSidecar.model_validate({"schema_version": 2, "fragment_json": "{}"})


def test_sidecar_rejects_non_string_fragment_json_strictly() -> None:
    """Strict mode rejects a non-string ``fragment_json`` (no int coercion)."""
    with pytest.raises(ValidationError, match="fragment_json"):
        LanceFragmentSidecar.model_validate({"schema_version": 1, "fragment_json": 42})


def test_sidecar_is_frozen() -> None:
    """Sidecar instances are immutable."""
    sidecar = LanceFragmentSidecar(schema_version=1, fragment_json="{}")
    with pytest.raises(ValidationError, match="frozen"):
        sidecar.fragment_json = "{...}"  # type: ignore[misc]


def test_dataset_card_with_duplicate_embedding_names_raises() -> None:
    """A v2 card cannot hide one embedding record behind another."""
    split = EmbeddingSplitProvenance(
        split="train", dataset_version=2, row_count=4, index_built=False, complete=True
    )
    embedding = EmbeddingProvenance(
        name="tinymu",
        columns=("tinymu", "tinymu_vec"),
        checkpoint="checkpoint",
        producer_git_sha="producer-sha",
        producer_transform_sha256="transform-sha",
        splits=(split,),
    )

    with pytest.raises(ValidationError, match="duplicate embedding names"):
        LanceDatasetCard(
            schema_version=2,
            run_id="run-1",
            finalized_at="2026-07-09T00:00:00+00:00",
            selected_attempts=(),
            embeddings=(embedding, embedding),
        )


def test_embedding_provenance_with_duplicate_splits_raises() -> None:
    """One embedding cannot claim the same split twice."""
    split = EmbeddingSplitProvenance(
        split="train", dataset_version=2, row_count=4, index_built=False, complete=True
    )

    with pytest.raises(ValidationError, match="duplicate splits"):
        EmbeddingProvenance(
            name="tinymu",
            columns=("tinymu", "tinymu_vec"),
            checkpoint="checkpoint",
            producer_git_sha="producer-sha",
            producer_transform_sha256="transform-sha",
            splits=(split, split),
        )


def test_dataset_card_rejects_unknown_fields() -> None:
    """Extra keys fail the card's ``extra=\"forbid\"`` boundary."""
    payload = _card().model_dump()
    payload["stats"] = {}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        LanceDatasetCard.model_validate(payload)


def test_selected_attempt_rejects_unknown_fields() -> None:
    """Extra keys fail the selected-attempt ``extra=\"forbid\"`` boundary."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SelectedLanceAttempt.model_validate(
            {"shard_id": 0, "attempt": "a", "valid_key": "k", "split": "train"}
        )
