"""Strict Pydantic contracts for Lance staging metadata and dataset audit records."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class EmbeddingSplitProvenance(BaseModel):
    """One split commit carrying a persisted embedding.

    .. attribute :: model_config

        Pydantic model config sentinel.

    .. attribute :: split

        Canonical dataset split.

    .. attribute :: dataset_version

        Lance version after embedding columns and indexes were committed.

    .. attribute :: row_count

        Rows preserved by the augmentation.

    .. attribute :: index_built

        Whether the embedding's declared vector index exists.

    .. attribute :: complete

        Whether columns and requested index work are committed.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    split: Literal["train", "val", "test"]
    dataset_version: PositiveInt
    row_count: PositiveInt
    index_built: bool
    complete: bool


class EmbeddingProvenance(BaseModel):
    """Identity and committed splits for one post-finalization embedding.

    .. attribute :: model_config

        Pydantic model config sentinel.

    .. attribute :: name

        Embedding registry key.

    .. attribute :: columns

        Lance columns emitted by the registry policy.

    .. attribute :: checkpoint

        Configured checkpoint URI, model id, or local identity.

    .. attribute :: producer_git_sha

        synth-setter revision defining preprocessing, pooling, and integration behavior.

    .. attribute :: producer_transform_sha256

        Digest of synth-setter modules defining the persisted MATPAC representation.

    .. attribute :: source_commit

        Pinned upstream package or source commit, when applicable.

    .. attribute :: checkpoint_revision

        Pinned checkpoint revision, when applicable.

    .. attribute :: checkpoint_sha256

        Strong checkpoint digest, when applicable.

    .. attribute :: param_spec_name

        Parameter specification used by parameter-sourced embeddings.

    .. attribute :: param_text_normalizer

        Caption normalizer used by parameter-sourced embeddings.

    .. attribute :: index_requested

        Whether the registry index was requested.

    .. attribute :: num_partitions

        Configured IVF partition count, or ``None`` for row-derived selection.

    .. attribute :: num_sub_vectors

        Configured PQ sub-vector count.

    .. attribute :: metric

        Configured vector distance metric.

    .. attribute :: splits

        Intended or successfully committed split results.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    name: str
    columns: tuple[str, ...]
    checkpoint: str
    producer_git_sha: str
    producer_transform_sha256: str
    source_commit: str | None = None
    checkpoint_revision: str | None = None
    checkpoint_sha256: str | None = None
    param_spec_name: str | None = None
    param_text_normalizer: str | None = None
    index_requested: bool | None = None
    num_partitions: int | None = None
    num_sub_vectors: PositiveInt | None = None
    metric: str | None = None
    splits: tuple[EmbeddingSplitProvenance, ...] = ()

    @model_validator(mode="after")
    def _split_names_are_unique(self) -> EmbeddingProvenance:
        """Reject ambiguous repeated split records.

        :returns: Validated provenance unchanged.
        :raises ValueError: A canonical split appears more than once.
        """
        names = [split.split for split in self.splits]
        if len(names) != len(set(names)):
            raise ValueError(f"embedding {self.name!r} has duplicate splits")
        return self


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

    .. attribute :: embeddings

        Post-finalization embedding identities and committed split results.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    schema_version: Literal[1, 2]
    run_id: str
    finalized_at: str
    selected_attempts: tuple[SelectedLanceAttempt, ...]
    embeddings: tuple[EmbeddingProvenance, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def _embedding_layout_matches_schema_version(self) -> LanceDatasetCard:
        """Preserve the v1 wire layout and reject ambiguous v2 records.

        :returns: Validated card unchanged.
        :raises ValueError: V1 carries embeddings or v2 repeats a registry key.
        """
        if self.schema_version == 1 and self.embeddings:
            raise ValueError("schema_version=1 cannot carry embedding provenance")
        names = [embedding.name for embedding in self.embeddings]
        if len(names) != len(set(names)):
            raise ValueError("dataset card has duplicate embedding names")
        return self


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
