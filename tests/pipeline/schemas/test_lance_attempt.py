"""Trust-boundary tests for the Lance staging/audit Pydantic contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synth_setter.pipeline.schemas.lance_attempt import (
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
    with pytest.raises(ValidationError, match="extra_forbidden"):
        LanceFragmentSidecar.model_validate(
            {"schema_version": 1, "fragment_json": "{}", "shard_id": 3}
        )


def test_sidecar_rejects_unknown_schema_version() -> None:
    with pytest.raises(ValidationError, match="schema_version"):
        LanceFragmentSidecar.model_validate({"schema_version": 2, "fragment_json": "{}"})


def test_sidecar_rejects_non_string_fragment_json_strictly() -> None:
    with pytest.raises(ValidationError, match="fragment_json"):
        LanceFragmentSidecar.model_validate({"schema_version": 1, "fragment_json": 42})


def test_sidecar_is_frozen() -> None:
    sidecar = LanceFragmentSidecar(schema_version=1, fragment_json="{}")
    with pytest.raises(ValidationError, match="frozen"):
        sidecar.fragment_json = "{...}"  # type: ignore[misc]


def test_dataset_card_round_trips_through_json() -> None:
    card = _card()
    assert LanceDatasetCard.model_validate_json(card.model_dump_json()) == card


def test_dataset_card_rejects_unknown_fields() -> None:
    payload = _card().model_dump()
    payload["stats"] = {}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        LanceDatasetCard.model_validate(payload)


def test_selected_attempt_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SelectedLanceAttempt.model_validate(
            {"shard_id": 0, "attempt": "a", "valid_key": "k", "split": "train"}
        )
