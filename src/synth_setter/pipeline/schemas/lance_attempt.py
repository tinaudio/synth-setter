"""Strict Pydantic contracts for Lance staging metadata and dataset audit records."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class SelectedLanceAttempt(BaseModel):
    """One shard's winning attempt as recorded in the ``dataset.json`` audit record.

    .. attribute :: model_config

        Pydantic model config sentinel — see ``ConfigDict(...)`` below for active settings.

    .. attribute :: shard_id

        Logical shard the attempt rendered.

    .. attribute :: attempt

        Attempt name (``{worker_id}-{attempt_uuid}``) from the staging filenames.

    .. attribute :: valid_key

        Full object key of the winning ``.valid`` marker — the exact object
        whose ``LastModified`` won selection.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    shard_id: int
    attempt: str
    valid_key: str


class LanceDatasetCard(BaseModel):
    """Provenance audit record finalize writes to ``dataset.json``.

    .. attribute :: model_config

        Pydantic model config sentinel — see ``ConfigDict(...)`` below for active settings.

    .. attribute :: schema_version

        Card schema version; bump on any layout change.

    .. attribute :: run_id

        The finalized run's id.

    .. attribute :: finalized_at

        ISO 8601 UTC timestamp of the finalize pass that sealed the dataset.

    .. attribute :: selected_attempts

        The winning attempt per shard, in ``shard_id`` order.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    schema_version: Literal[1]
    run_id: str
    finalized_at: str
    selected_attempts: tuple[SelectedLanceAttempt, ...]


class LanceFragmentSidecar(BaseModel):
    """Per-attempt Lance fragment sidecar (``{worker}-{attempt}.fragment.json``).

    .. attribute :: model_config

        Pydantic model config sentinel — see ``ConfigDict(...)`` below for active settings.

    .. attribute :: schema_version

        Sidecar schema version; bump on any layout change.

    .. attribute :: fragment_json

        ``json.dumps`` of Lance's ``FragmentMetadata.to_json()`` dict — an
        opaque Lance-owned string that finalize re-parses with
        ``FragmentMetadata.from_json``.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    schema_version: Literal[1]
    fragment_json: str
