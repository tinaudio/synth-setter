"""Stream SLAP EMA projections into split retrieval datasets.

Typical use is ``export_slap(ExportSlapConfig.from_hydra_cfg(cfg))``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import hydra
import lance
import numpy as np
import pyarrow as pa
import structlog
import torch
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
    audio_dataset_shape,
    mel_dataset_shape,
)
from synth_setter.data.vst_datamodule import RawBatch, load_mel_statistics, prepare_batch
from synth_setter.models.slap_module import SLAPModule
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_materialize import (
    _is_retryable_lance_read_error,
    retry_lance_read,
)
from synth_setter.pipeline.data.lance_shard import (
    LANCE_DATA_STORAGE_VERSION,
    read_shard_metadata,
)
from synth_setter.pipeline.file_uri import file_uri_to_path, is_file_uri
from synth_setter.pipeline.schemas.export_slap_config import ExportSlapConfig

logger = structlog.get_logger(__name__)

ROW_UUID_FIELD = "row_uuid"
IS_PARAM_EMBEDDING_FIELD = "is_param_embedding"
SLAP_FIELD = "slap"
SLAP_EXPORT_METADATA_KEY = b"synth_setter.slap_export"
SLAP_SOURCE_POINTER_KEY = b"synth_setter.slap_retrieval"
_UUID_MIGRATION_METADATA_KEY = b"synth_setter.row_uuid_migration"
_SCHEMA_VERSION = 1
_POLICY_VERSION = 2


class _ExportMetadata(BaseModel):
    """Validate persisted SLAP export provenance.

    .. attribute :: model_config
    .. attribute :: schema_version
    .. attribute :: policy_version
    .. attribute :: completed
    .. attribute :: request_hash
    .. attribute :: source_uri
    .. attribute :: split
    .. attribute :: requested_source_version
    .. attribute :: requested_source_transaction_uuid
    .. attribute :: source_version
    .. attribute :: source_transaction_uuid
    .. attribute :: checkpoint_sha256
    .. attribute :: resolved_model_config
    .. attribute :: input_policy
    .. attribute :: projection_policy
    .. attribute :: normalization
    .. attribute :: parameter_preprocessing
    .. attribute :: mel_preprocessing
    .. attribute :: mel_stats_sha256
    .. attribute :: vector_dimension
    .. attribute :: source_row_count
    .. attribute :: output_row_count
    .. attribute :: build_index
    .. attribute :: metric
    .. attribute :: requested_num_partitions
    .. attribute :: num_partitions
    .. attribute :: num_sub_vectors
    .. attribute :: index_built
    .. attribute :: output_version
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    schema_version: int
    policy_version: int
    completed: bool
    request_hash: str
    source_uri: str
    split: str
    requested_source_version: int
    requested_source_transaction_uuid: str
    source_version: int
    source_transaction_uuid: str
    checkpoint_sha256: str
    resolved_model_config: dict[str, object]
    input_policy: str
    projection_policy: str
    normalization: str
    parameter_preprocessing: str
    mel_preprocessing: str
    mel_stats_sha256: str | None
    vector_dimension: int
    source_row_count: int
    output_row_count: int
    build_index: bool
    metric: str
    requested_num_partitions: int | None
    num_partitions: int
    num_sub_vectors: int
    index_built: bool
    output_version: int | None


class _UuidMigration(BaseModel):
    """Validate the identity attached atomically to a UUID migration field.

    .. attribute :: model_config
    .. attribute :: schema_version
    .. attribute :: policy_version
    .. attribute :: requested_source_version
    .. attribute :: requested_source_transaction_uuid
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    schema_version: int
    policy_version: int
    requested_source_version: int
    requested_source_transaction_uuid: str


class _SourcePointer(BaseModel):
    """Validate the source-side pointer to one completed export.

    .. attribute :: model_config
    .. attribute :: schema_version
    .. attribute :: output_uri
    .. attribute :: output_version
    .. attribute :: input_source_version
    .. attribute :: request_hash
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    schema_version: int
    output_uri: str
    output_version: int
    input_source_version: int
    request_hash: str


@dataclass(frozen=True)
class SlapExportResult:
    """Identify one completed split export.

    .. attribute :: split
    .. attribute :: output_uri
    .. attribute :: output_version
    .. attribute :: source_version
    .. attribute :: source_transaction_uuid
    .. attribute :: source_row_count
    .. attribute :: output_row_count
    """

    split: str
    output_uri: str
    output_version: int
    source_version: int
    source_transaction_uuid: str
    source_row_count: int
    output_row_count: int


@dataclass(frozen=True)
class _Preprocessing:
    """Freeze model-input transforms and optional mel statistics for one export.

    .. attribute :: mean
    .. attribute :: std
    .. attribute :: parameter_policy
    .. attribute :: mel_policy
    .. attribute :: mel_stats_sha256
    """

    mean: np.ndarray | None
    std: np.ndarray | None
    parameter_policy: str
    mel_policy: str
    mel_stats_sha256: str | None


def _split_uri(root: str, split: str) -> str:
    """Return one split dataset URI.

    :param root: Dataset root URI.
    :param split: Split name.
    :returns: URI ending in ``<split>.lance``.
    """
    return f"{root.rstrip('/')}/{split}.lance"


def _lance_target(uri: str) -> tuple[str, dict[str, str] | None]:
    """Resolve a supported local or R2-compatible URI for Lance.

    :param uri: Dataset URI.
    :returns: Lance target and optional storage options.
    :raises ValueError: The URI scheme is unsupported.
    """
    if r2_io.is_r2_uri(uri):
        return r2_io.lance_target(uri)
    if uri.startswith("s3://"):
        return r2_io.lance_target(r2_io.from_s3_uri(uri))
    if is_file_uri(uri):
        return str(file_uri_to_path(uri)), None
    if "://" in uri:
        raise ValueError(f"unsupported dataset URI scheme: {uri!r}")
    return uri, None


def _open(uri: str) -> lance.LanceDataset:
    """Open a local or R2 Lance dataset.

    :param uri: Dataset URI.
    :returns: Open dataset.
    """
    target, storage_options = _lance_target(uri)
    return retry_lance_read(
        "slap_dataset_open", lambda: lance.dataset(target, storage_options=storage_options)
    )


def _transaction_uuid(dataset: lance.LanceDataset, version: int) -> str:
    """Read the transaction UUID for an exact version.

    :param dataset: Lance dataset.
    :param version: Exact version.
    :returns: Transaction UUID.
    :raises ValueError: The version has no transaction.
    """
    transaction = retry_lance_read(
        "slap_transaction_read", lambda: dataset.read_transaction(version)
    )
    if transaction is None:
        raise ValueError(f"Lance version {version} has no transaction UUID")
    return transaction.uuid


def _file_sha256(path: Path) -> str:
    """Hash local file content.

    :param path: Local file.
    :returns: SHA-256 hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_sha256(path: Path) -> str:
    """Hash checkpoint content.

    :param path: Checkpoint file.
    :returns: SHA-256 hexadecimal digest.
    """
    return _file_sha256(path)


def _load_preprocessing(config: ExportSlapConfig, model: SLAPModule) -> _Preprocessing:
    """Load and freeze the production preprocessing policy for one export.

    :param config: Export configuration.
    :param model: Loaded model selecting the audio input modality.
    :returns: Immutable preprocessing state shared by all splits.
    :raises ValueError: Required statistics are absent or change while loading.
    """
    parameter_policy = "stored_[0,1]_to_model_[-1,1]"
    if model.audio_input_key == "audio":
        return _Preprocessing(None, None, parameter_policy, "not_applicable", None)
    if not config.use_saved_mean_and_variance:
        return _Preprocessing(None, None, parameter_policy, "stored_mel_unnormalized", None)
    if config.mel_stats_path is None:
        raise ValueError(
            "mel_stats_path is required for mel models when use_saved_mean_and_variance is true"
        )

    digest_before = _file_sha256(config.mel_stats_path)
    mean, std = load_mel_statistics(config.mel_stats_path)
    digest_after = _file_sha256(config.mel_stats_path)
    if digest_after != digest_before:
        raise ValueError("mel statistics changed while loading; use an immutable local file")
    frozen_mean = np.array(mean, copy=True)
    frozen_std = np.array(std, copy=True)
    frozen_mean.setflags(write=False)
    frozen_std.setflags(write=False)
    return _Preprocessing(
        frozen_mean,
        frozen_std,
        parameter_policy,
        "normalize_with_saved_mean_and_variance",
        digest_before,
    )


def _canonical_model_config(config: Mapping[str, object]) -> dict[str, object]:
    """Return JSON-canonical model configuration with compilation disabled.

    :param config: Resolved model configuration.
    :returns: Canonical JSON mapping.
    """
    resolved = dict(config)
    resolved["compile"] = False
    return json.loads(json.dumps(resolved, sort_keys=True, separators=(",", ":")))


def _request_hash(payload: Mapping[str, object]) -> str:
    """Hash one canonical request payload.

    :param payload: JSON-compatible request fields.
    :returns: SHA-256 hexadecimal digest.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _static_request(
    config: ExportSlapConfig,
    preprocessing: _Preprocessing,
    *,
    split: str,
    checkpoint_sha256: str,
    requested_source_transaction_uuid: str,
) -> dict[str, object]:
    """Build request fields independent of source UUID migration.

    :param config: Export configuration.
    :param preprocessing: Frozen production input transforms.
    :param split: Split name.
    :param checkpoint_sha256: Checkpoint content hash.
    :param requested_source_transaction_uuid: Configured snapshot identity.
    :returns: Static provenance fields.
    """
    return {
        "schema_version": _SCHEMA_VERSION,
        "policy_version": _POLICY_VERSION,
        "source_uri": _split_uri(config.source_root_uri, split),
        "split": split,
        "requested_source_version": config.source_versions[split],
        "requested_source_transaction_uuid": requested_source_transaction_uuid,
        "checkpoint_sha256": checkpoint_sha256,
        "resolved_model_config": _canonical_model_config(config.model),
        "input_policy": "stored_audio"
        if config.model.get("audio_input_key", "audio") == "audio"
        else "stored_mel",
        "projection_policy": "audio_ema[1]+text_ema[1]",
        "normalization": "l2_float32",
        "parameter_preprocessing": preprocessing.parameter_policy,
        "mel_preprocessing": preprocessing.mel_policy,
        "mel_stats_sha256": preprocessing.mel_stats_sha256,
        "build_index": config.build_index,
        "metric": config.metric,
        "requested_num_partitions": config.num_partitions,
        "num_sub_vectors": config.num_sub_vectors,
    }


def _load_metadata(dataset: lance.LanceDataset) -> _ExportMetadata | None:
    """Parse output metadata when present.

    :param dataset: Output dataset.
    :returns: Strict metadata or ``None``.
    """
    raw = (dataset.schema.metadata or {}).get(SLAP_EXPORT_METADATA_KEY)
    return None if raw is None else _ExportMetadata.model_validate_json(raw)


def _existing_completed(
    output_uri: str, request_hash: str
) -> tuple[lance.LanceDataset, _ExportMetadata] | None:
    """Open a matching output or reject conflicting ownership.

    :param output_uri: Output dataset URI.
    :param request_hash: Expected request identity.
    :returns: Existing output and metadata, or ``None``.
    :raises ValueError: Existing output metadata conflicts or is inconsistent.
    """
    try:
        dataset = _open(output_uri)
    except FileNotFoundError:
        return None
    except ValueError as exc:
        message = str(exc)
        if message.startswith("Dataset at path ") and " was not found: Not found: " in message:
            return None
        raise
    metadata = _load_metadata(dataset)
    if metadata is None or metadata.request_hash != request_hash:
        raise ValueError(f"output {output_uri} already exists for a conflicting request")
    if not metadata.completed:
        return dataset, metadata
    if metadata.output_version != dataset.version:
        raise ValueError(f"output {output_uri} completion version does not match its head")
    return dataset, metadata


def _scan_batches(
    dataset: lance.LanceDataset,
    *,
    columns: list[str],
    batch_size: int | None = None,
) -> Iterator[pa.RecordBatch]:
    """Stream a pinned dataset, resuming transient failures after yielded rows.

    :yields pa.RecordBatch: Each source row exactly once and in scan order.
    :param dataset: Explicitly pinned Lance dataset.
    :param columns: Projected source columns.
    :param batch_size: Optional scanner batch size.
    :raises Exception: A nonretryable scan failure.
    """
    yielded_rows = 0
    batches = retry_lance_read(
        "slap_batch_scan_open",
        lambda: dataset.to_batches(columns=columns, batch_size=batch_size),
    )
    while True:
        try:
            batch = next(batches)
        except StopIteration:
            return
        except Exception as exc:
            if not _is_retryable_lance_read_error(exc):
                raise

            def resume() -> tuple[Iterator[pa.RecordBatch], pa.RecordBatch | None]:
                resumed = dataset.to_batches(
                    columns=columns,
                    batch_size=batch_size,
                    offset=yielded_rows,
                )
                return resumed, next(resumed, None)

            batches, batch = retry_lance_read("slap_batch_scan", resume)
            if batch is None:
                return
        yield batch
        yielded_rows += batch.num_rows


def _validate_row_uuids(dataset: lance.LanceDataset) -> None:
    """Require canonical, nonnull, unique source UUIDs.

    :param dataset: Source snapshot.
    :raises ValueError: Any UUID is invalid or repeated.
    """
    seen: set[str] = set()
    for batch in _scan_batches(dataset, columns=[ROW_UUID_FIELD]):
        for value in batch.column(0).to_pylist():
            if value is None:
                raise ValueError("row_uuid must be nonnull")
            try:
                canonical = str(UUID(value))
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"row_uuid is invalid: {value!r}") from exc
            if canonical != value:
                raise ValueError(f"row_uuid is not canonical: {value!r}")
            if value in seen:
                raise ValueError(f"row_uuid is not unique: {value}")
            seen.add(value)


def _uuid_migration_field(migration: _UuidMigration) -> pa.Field:
    """Build the UUID field carrying its source-snapshot identity.

    :param migration: Original source identity and migration policy.
    :returns: Nullable UUID field for Lance column creation.
    """
    return pa.field(
        ROW_UUID_FIELD,
        pa.string(),
        nullable=True,
        metadata={_UUID_MIGRATION_METADATA_KEY: migration.model_dump_json().encode()},
    )


def _add_uuid_column(batch: pa.RecordBatch, *, field: pa.Field) -> pa.RecordBatch:
    """Generate one UUID batch with migration field metadata.

    :param batch: Source batch determining output cardinality.
    :param field: UUID field carrying migration identity.
    :returns: UUID-only record batch.
    """
    values = pa.array([str(uuid4()) for _ in range(batch.num_rows)], type=pa.string())
    return pa.RecordBatch.from_arrays([values], schema=pa.schema([field]))


def _attributable_uuid_migration(
    head: lance.LanceDataset, *, requested_version: int, migration: _UuidMigration
) -> lance.LanceDataset | None:
    """Return only the exact next commit carrying the requested migration identity.

    :param head: Current source head.
    :param requested_version: Original UUID-less source version.
    :param migration: Expected migration provenance.
    :returns: Attributable migration snapshot, or ``None``.
    """
    migration_version = requested_version + 1
    if migration_version > head.version:
        return None
    candidate = head.checkout_version(migration_version)
    if ROW_UUID_FIELD not in candidate.schema.names:
        return None
    raw = (candidate.schema.field(ROW_UUID_FIELD).metadata or {}).get(_UUID_MIGRATION_METADATA_KEY)
    if raw is None or _UuidMigration.model_validate_json(raw) != migration:
        return None
    _validate_row_uuids(candidate)
    return candidate


def _ensure_row_uuids(
    source_uri: str, requested_version: int, batch_size: int
) -> lance.LanceDataset:
    """Return a pinned source, adding UUIDs only to the expected head.

    :param source_uri: Source dataset URI.
    :param requested_version: Explicit source pin.
    :param batch_size: UUID UDF batch size.
    :returns: UUID-bearing pinned source.
    :raises ValueError: The pin is absent or historical without an attributable migration.
    :raises RuntimeError: The committed migration provenance cannot be verified.
    """
    head = _open(source_uri)
    if requested_version > head.version:
        raise ValueError(f"source version {requested_version} does not exist")
    pinned = head.checkout_version(requested_version)
    requested_transaction_uuid = _transaction_uuid(pinned, requested_version)
    if ROW_UUID_FIELD in pinned.schema.names:
        _validate_row_uuids(pinned)
        return pinned

    migration = _UuidMigration(
        schema_version=_SCHEMA_VERSION,
        policy_version=_POLICY_VERSION,
        requested_source_version=requested_version,
        requested_source_transaction_uuid=requested_transaction_uuid,
    )
    attributable = _attributable_uuid_migration(
        head, requested_version=requested_version, migration=migration
    )
    if attributable is not None:
        return attributable
    if requested_version != head.version:
        raise ValueError("historical pinned source lacks row_uuid; refusing to mutate newer head")

    field = _uuid_migration_field(migration)
    if head.count_rows() == 0:
        head.add_columns(field)
    else:
        head.add_columns(
            partial(_add_uuid_column, field=field),
            read_columns=[],
            batch_size=batch_size,
        )
    migrated = head.checkout_version(requested_version + 1)
    raw = (migrated.schema.field(ROW_UUID_FIELD).metadata or {}).get(_UUID_MIGRATION_METADATA_KEY)
    if raw is None or _UuidMigration.model_validate_json(raw) != migration:
        raise RuntimeError("UUID migration commit has unexpected provenance")
    _validate_row_uuids(migrated)
    return migrated


def _input_field(model: SLAPModule) -> str:
    """Return the stored source field consumed by the configured audio arm.

    :param model: Loaded SLAP module.
    :returns: Audio or mel source field name.
    """
    return AUDIO_FIELD if model.audio_input_key == "audio" else MEL_SPEC_FIELD


def _decoded(column: pa.Array, *, field: str) -> np.ndarray:
    """Decode one supported fixed-shape tensor column.

    :param column: Arrow array.
    :param field: Source field name for error context.
    :returns: NumPy tensor rows.
    :raises ValueError: The column is not a fixed-shape tensor extension array.
    """
    if not isinstance(column, pa.FixedShapeTensorArray):
        raise ValueError(f"source field {field!r} must be a FixedShapeTensor column")
    return column.to_numpy_ndarray()


def _validate_source_inputs(
    source: lance.LanceDataset,
    *,
    input_field: str,
    preprocessing: _Preprocessing,
) -> None:
    """Validate stored model inputs before any source mutation or inference.

    :param source: Explicitly pinned source snapshot.
    :param input_field: Audio or mel source field selected by the model.
    :param preprocessing: Frozen production input transforms.
    :raises ValueError: A required field, tensor type, finite value, or audio bound is invalid.
    """
    metadata = read_shard_metadata(source.schema)
    shape_fn = audio_dataset_shape if input_field == AUDIO_FIELD else mel_dataset_shape
    expected_shape = shape_fn(
        1, metadata.channels, metadata.sample_rate, metadata.signal_duration_seconds
    )[1:]
    required_fields = (input_field, PARAM_ARRAY_FIELD)
    for field in required_fields:
        if field not in source.schema.names:
            raise ValueError(f"source is missing required field {field!r}")
        if not isinstance(source.schema.field(field).type, pa.FixedShapeTensorType):
            raise ValueError(f"source field {field!r} must be a FixedShapeTensor column")
    if tuple(source.schema.field(input_field).type.shape) != expected_shape:
        raise ValueError(f"source field {input_field!r} shape disagrees with shard metadata")
    param_shape = tuple(source.schema.field(PARAM_ARRAY_FIELD).type.shape)
    if len(param_shape) != 1 or param_shape[0] < 1:
        raise ValueError(f"source field {PARAM_ARRAY_FIELD!r} has invalid shape {param_shape}")
    if preprocessing.mean is not None and preprocessing.std is not None:
        try:
            normalized_shape = np.broadcast_shapes(
                expected_shape, preprocessing.mean.shape, preprocessing.std.shape
            )
        except ValueError as exc:
            raise ValueError("mel statistics cannot broadcast to stored mel geometry") from exc
        if normalized_shape != expected_shape:
            raise ValueError("mel statistics would expand the stored mel geometry")
    for batch in _scan_batches(source, columns=list(required_fields)):
        for field in required_fields:
            values = _decoded(batch.column(field), field=field)
            if not np.isfinite(values).all():
                raise ValueError(f"source field {field!r} contains nonfinite values")
            if field == AUDIO_FIELD and np.any(np.abs(values) > 1.0):
                raise ValueError("source audio values must be within [-1, 1]")
            if field == PARAM_ARRAY_FIELD and np.any((values < 0) | (values > 1)):
                raise ValueError("source param_array values must be within [0, 1]")


def _normalize(vectors: torch.Tensor, modality: str) -> np.ndarray:
    """Normalize finite projection tensors shaped ``(batch, dimension)``.

    :param vectors: Projection tensor with one nonempty row per source row.
    :param modality: Error-context label.
    :returns: Contiguous float32 unit vectors.
    :raises ValueError: Rank, width, values, or norms are invalid.
    """
    values = vectors.detach().float().cpu().numpy()
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(f"{modality} projection has invalid shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{modality} projection contains nonfinite values")
    stable_values = values.astype(np.float64)
    norms = np.linalg.norm(stable_values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError(f"{modality} projection contains zero or invalid vectors")
    return np.ascontiguousarray(stable_values / norms, dtype=np.float32)


def _validate_projection_pair(
    audio_projection: torch.Tensor,
    param_projection: torch.Tensor,
    *,
    batch_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate matching ``(batch, dimension)`` EMA projections.

    :param audio_projection: Audio-arm EMA projection.
    :param param_projection: Parameter-arm EMA projection.
    :param batch_rows: Source record-batch cardinality.
    :returns: Normalized audio and parameter vectors.
    :raises ValueError: Rank, width, cardinality, values, or norms differ from the contract.
    """
    audio_vectors = _normalize(audio_projection, "audio EMA")
    param_vectors = _normalize(param_projection, "parameter EMA")
    if audio_vectors.shape[0] != batch_rows or param_vectors.shape[0] != batch_rows:
        raise ValueError("SLAP EMA projection batch cardinality differs from source batch")
    if audio_vectors.shape[1] != param_vectors.shape[1]:
        raise ValueError(
            "SLAP EMA projection dimensions differ: "
            f"audio {audio_vectors.shape}, params {param_vectors.shape}"
        )
    return audio_vectors, param_vectors


def _prepare_inputs(
    raw: Mapping[str, np.ndarray],
    *,
    model_input_key: str,
    preprocessing: _Preprocessing,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the production batch frontend and select required SLAP tensors.

    :param raw: Stored model columns.
    :param model_input_key: Prepared audio or mel key.
    :param preprocessing: Frozen production input transforms.
    :returns: Model-ready audio-arm and parameter-arm inputs.
    :raises ValueError: Batch preparation omits a required input.
    """
    prepared = prepare_batch(
        cast(RawBatch, raw),
        mean=preprocessing.mean,
        std=preprocessing.std,
        rescale_params=True,
        ot=False,
        generator=torch.Generator().manual_seed(0),
    )
    audio = prepared[model_input_key]
    params = prepared["params"]
    if audio is None or params is None:
        raise ValueError("production batch preparation omitted a required SLAP input")
    return audio, params


def _project_batch(
    model: SLAPModule,
    batch: pa.RecordBatch,
    *,
    device: torch.device,
    preprocessing: _Preprocessing,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Project one source batch through both EMA arms.

    :param model: Loaded SLAP module.
    :param batch: Source rows.
    :param device: Inference device.
    :param preprocessing: Frozen production input transforms.
    :returns: Audio vectors, parameter vectors, and source UUIDs.
    :raises ValueError: EMA outputs are absent or incompatible.
    """
    input_field = _input_field(model)
    audio_values = np.array(
        _decoded(batch.column(input_field), field=input_field),
        dtype=np.float32,
        order="C",
        copy=True,
    )
    param_values = np.array(
        _decoded(batch.column(PARAM_ARRAY_FIELD), field=PARAM_ARRAY_FIELD),
        dtype=np.float32,
        order="C",
        copy=True,
    )
    audio, params = _prepare_inputs(
        {input_field: audio_values, PARAM_ARRAY_FIELD: param_values},
        model_input_key=model.audio_input_key,
        preprocessing=preprocessing,
    )
    with torch.inference_mode():
        _, audio_projection, _ = model.audio_ema(audio.to(device))
        _, param_projection, _ = model.text_ema(params.to(device))
    if audio_projection is None or param_projection is None:
        raise ValueError("SLAP EMA arms must provide projection outputs")
    audio_vectors, param_vectors = _validate_projection_pair(
        audio_projection, param_projection, batch_rows=batch.num_rows
    )
    return audio_vectors, param_vectors, batch.column(ROW_UUID_FIELD).to_pylist()


def _output_batches(
    source: lance.LanceDataset,
    *,
    model: SLAPModule,
    device: torch.device,
    batch_size: int,
    vector_dimension: int,
    preprocessing: _Preprocessing,
) -> Iterator[pa.RecordBatch]:
    """Yield interleaved parameter and audio retrieval rows.

    :yields pa.RecordBatch: Retrieval batches.
    :param source: Pinned source snapshot.
    :param model: Loaded SLAP module.
    :param device: Inference device.
    :param batch_size: Source scan batch size.
    :param vector_dimension: Required output width.
    :param preprocessing: Frozen production input transforms.
    :raises ValueError: Projection dimensions change.
    """
    input_field = _input_field(model)
    columns = [ROW_UUID_FIELD, input_field, PARAM_ARRAY_FIELD]
    for batch in _scan_batches(source, columns=columns, batch_size=batch_size):
        audio, params, row_uuids = _project_batch(
            model,
            batch,
            device=device,
            preprocessing=preprocessing,
        )
        if audio.shape[1] != vector_dimension:
            raise ValueError("SLAP projection dimension changed between batches")
        vectors = np.stack((params, audio), axis=1).reshape(-1, vector_dimension)
        repeated_uuids = np.repeat(row_uuids, 2).tolist()
        yield pa.record_batch(
            [
                pa.array(repeated_uuids, pa.string()),
                pa.array(np.tile([True, False], len(row_uuids)), pa.bool_()),
                pa.FixedSizeListArray.from_arrays(pa.array(vectors.reshape(-1)), vector_dimension),
            ],
            names=[ROW_UUID_FIELD, IS_PARAM_EMBEDDING_FIELD, SLAP_FIELD],
        )


def _load_model(config: ExportSlapConfig) -> tuple[SLAPModule, torch.device]:
    """Instantiate SLAP and strictly load checkpoint state.

    :param config: Export configuration.
    :returns: Evaluation model and device.
    :raises ValueError: Config or checkpoint does not describe SLAP.
    """
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_config = dict(config.model)
    model_config["compile"] = False
    model = hydra.utils.instantiate(OmegaConf.create(model_config))
    if not isinstance(model, SLAPModule):
        raise ValueError("model config must instantiate SLAPModule")
    checkpoint = torch.load(config.ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("checkpoint must contain a Lightning state_dict")
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, device


def _infer_vector_dimension(
    source: lance.LanceDataset,
    model: SLAPModule,
    *,
    device: torch.device,
    preprocessing: _Preprocessing,
) -> int:
    """Infer and validate the shared EMA projection width.

    :param source: Source snapshot providing input shapes.
    :param model: Loaded SLAP module.
    :param device: Inference device.
    :param preprocessing: Frozen production input transforms.
    :returns: Shared vector dimension.
    :raises ValueError: EMA projection outputs are absent or incompatible.
    """
    input_field = _input_field(model)
    audio_shape = tuple(source.schema.field(input_field).type.shape)
    param_shape = tuple(source.schema.field(PARAM_ARRAY_FIELD).type.shape)
    audio, params = _prepare_inputs(
        {
            input_field: np.zeros((1, *audio_shape), dtype=np.float32),
            PARAM_ARRAY_FIELD: np.zeros((1, *param_shape), dtype=np.float32),
        },
        model_input_key=model.audio_input_key,
        preprocessing=preprocessing,
    )
    with torch.inference_mode():
        try:
            _, audio_projection, _ = model.audio_ema(audio.to(device))
        except RuntimeError as exc:
            raise ValueError(
                f"source {input_field} shape {audio_shape} is incompatible with loaded SLAP model"
            ) from exc
        try:
            _, param_projection, _ = model.text_ema(params.to(device))
        except RuntimeError as exc:
            raise ValueError(
                f"source {PARAM_ARRAY_FIELD} shape {param_shape} is incompatible with loaded SLAP model"
            ) from exc
    if audio_projection is None or param_projection is None:
        raise ValueError("SLAP EMA arms must provide projection outputs")
    if audio_projection.ndim != 2 or param_projection.ndim != 2:
        raise ValueError("SLAP EMA projections must be rank-two")
    if audio_projection.shape[1] != param_projection.shape[1]:
        raise ValueError("SLAP EMA projection dimensions differ")
    if audio_projection.shape[1] < 1:
        raise ValueError("SLAP EMA projection dimension must be positive")
    return audio_projection.shape[1]


def _schema(vector_dimension: int, metadata: _ExportMetadata) -> pa.Schema:
    """Build the exact retrieval schema and provenance.

    :param vector_dimension: SLAP vector width.
    :param metadata: Incomplete export provenance.
    :returns: Three-column Arrow schema.
    """
    return pa.schema(
        [
            pa.field(ROW_UUID_FIELD, pa.string(), nullable=False),
            pa.field(IS_PARAM_EMBEDDING_FIELD, pa.bool_(), nullable=False),
            pa.field(SLAP_FIELD, pa.list_(pa.float32(), vector_dimension), nullable=False),
        ],
        metadata={SLAP_EXPORT_METADATA_KEY: metadata.model_dump_json().encode()},
    )


def _index_partitions(config: ExportSlapConfig, output_rows: int) -> int:
    """Resolve the IVF partition count.

    :param config: Export configuration.
    :param output_rows: Retrieval row count.
    :returns: Positive partition count.
    """
    return config.num_partitions or max(1, round(output_rows**0.5))


def _validate_index(dataset: lance.LanceDataset, metadata: _ExportMetadata) -> bool:
    """Validate an existing IVF_PQ index against provenance.

    :param dataset: Output dataset.
    :param metadata: Expected index policy.
    :returns: Whether the index exists.
    :raises ValueError: Existing index configuration differs.
    """
    indices = dataset.list_indices()
    if not indices:
        return False
    if len(indices) != 1 or indices[0]["fields"] != [SLAP_FIELD]:
        raise ValueError("existing output has an incompatible index")
    statistics = dataset.index_statistics(indices[0]["name"])["indices"][0]
    sub_index = statistics["sub_index"]
    matches = (
        indices[0]["type"] == "IVF_PQ"
        and statistics["metric_type"] == metadata.metric
        and statistics["num_partitions"] == metadata.num_partitions
        and sub_index["num_sub_vectors"] == metadata.num_sub_vectors
    )
    if not matches:
        raise ValueError("existing output index configuration differs from the export request")
    return True


def _complete_output(
    dataset: lance.LanceDataset,
    metadata: _ExportMetadata,
    *,
    config: ExportSlapConfig,
    output_uri: str,
) -> tuple[lance.LanceDataset, _ExportMetadata]:
    """Build any requested index and commit completion metadata.

    :param dataset: Committed output data.
    :param metadata: Incomplete provenance.
    :param config: Export configuration.
    :param output_uri: Stable output URI.
    :returns: Completed dataset and provenance.
    :raises ValueError: Existing index configuration differs.
    :raises RuntimeError: Completion commits an unexpected version.
    """
    has_index = _validate_index(dataset, metadata)
    if not config.build_index and has_index:
        raise ValueError("existing output has an index not requested by this export")
    if config.build_index and dataset.count_rows() > 0:
        if not has_index:
            dataset.create_index(
                SLAP_FIELD,
                index_type="IVF_PQ",
                metric=config.metric,
                num_partitions=metadata.num_partitions,
                num_sub_vectors=config.num_sub_vectors,
            )
            dataset = _open(output_uri)
    published_version = dataset.version + 1
    completed = metadata.model_copy(
        update={
            "completed": True,
            "index_built": _validate_index(dataset, metadata),
            "output_version": published_version,
        }
    )
    dataset.update_schema_metadata(
        {SLAP_EXPORT_METADATA_KEY.decode(): completed.model_dump_json()}, replace=False
    )
    dataset = _open(output_uri)
    if dataset.version != published_version:
        raise RuntimeError("output completion committed an unexpected Lance version")
    return dataset, completed


def _validate_source_identity(
    source: lance.LanceDataset, metadata: _ExportMetadata
) -> lance.LanceDataset:
    """Validate both requested and selected snapshots in the current source history.

    :param source: Current source dataset.
    :param metadata: Export provenance naming exact source transactions.
    :returns: Selected source snapshot.
    :raises ValueError: Either source transaction or selected cardinality differs.
    """
    requested = source.checkout_version(metadata.requested_source_version)
    requested_uuid = _transaction_uuid(requested, metadata.requested_source_version)
    if requested_uuid != metadata.requested_source_transaction_uuid:
        raise ValueError("requested source transaction identity differs from output provenance")
    selected = source.checkout_version(metadata.source_version)
    selected_uuid = _transaction_uuid(selected, metadata.source_version)
    if selected_uuid != metadata.source_transaction_uuid:
        raise ValueError("source transaction identity differs from output provenance")
    if selected.count_rows() != metadata.source_row_count:
        raise ValueError("source row count differs from output provenance")
    _validate_row_uuids(selected)
    read_shard_metadata(selected.schema)
    return selected


def _update_source_pointer(
    source_uri: str,
    metadata: _ExportMetadata,
    *,
    output_uri: str,
    output_version: int,
) -> None:
    """Publish a pointer to an exact historical or current source snapshot.

    :param source_uri: Source dataset URI.
    :param metadata: Completed export provenance.
    :param output_uri: Completed output URI.
    :param output_version: Exact completed output version.
    """
    source = _open(source_uri)
    _validate_source_identity(source, metadata)
    pointer = _SourcePointer(
        schema_version=_SCHEMA_VERSION,
        output_uri=output_uri,
        output_version=output_version,
        input_source_version=metadata.source_version,
        request_hash=metadata.request_hash,
    )
    raw = (source.schema.metadata or {}).get(SLAP_SOURCE_POINTER_KEY)
    existing_pointer = None if raw is None else _SourcePointer.model_validate_json(raw)
    if existing_pointer == pointer:
        return
    source.update_schema_metadata(
        {SLAP_SOURCE_POINTER_KEY.decode(): pointer.model_dump_json()}, replace=False
    )


def _validate_reusable_output(
    source_uri: str,
    output: lance.LanceDataset,
    metadata: _ExportMetadata,
) -> None:
    """Validate source lineage and output shape before reuse.

    :param source_uri: Configured source dataset URI.
    :param output: Existing output dataset.
    :param metadata: Persisted output provenance.
    :raises ValueError: Source identity, row counts, or output schema differ.
    """
    expected_schema = _schema(metadata.vector_dimension, metadata).remove_metadata()
    if not output.schema.remove_metadata().equals(expected_schema):
        raise ValueError("existing output schema differs from completed provenance")
    if output.count_rows() != metadata.output_row_count:
        raise ValueError("existing output row count differs from completed provenance")

    _validate_source_identity(_open(source_uri), metadata)


def _result(metadata: _ExportMetadata, output_uri: str) -> SlapExportResult:
    """Project persisted provenance onto the public result.

    :param metadata: Completed export provenance.
    :param output_uri: Output dataset URI.
    :returns: Public split result.
    :raises ValueError: Completed provenance lacks an output version.
    """
    if metadata.output_version is None:
        raise ValueError("completed export metadata is missing output_version")
    return SlapExportResult(
        split=metadata.split,
        output_uri=output_uri,
        output_version=metadata.output_version,
        source_version=metadata.source_version,
        source_transaction_uuid=metadata.source_transaction_uuid,
        source_row_count=metadata.source_row_count,
        output_row_count=metadata.output_row_count,
    )


def _resume_output(
    existing: tuple[lance.LanceDataset, _ExportMetadata],
    *,
    config: ExportSlapConfig,
    source_uri: str,
    output_uri: str,
) -> SlapExportResult:
    """Validate and complete or reuse one existing output.

    :param existing: Matching output dataset and persisted provenance.
    :param config: Export configuration.
    :param source_uri: Source split URI.
    :param output_uri: Output split URI.
    :returns: Completed split identity.
    """
    output, metadata = existing
    _validate_reusable_output(source_uri, output, metadata)
    index_missing = metadata.build_index and output.count_rows() > 0 and not output.list_indices()
    if not metadata.completed or index_missing:
        resumable = metadata.model_copy(
            update={"completed": False, "index_built": False, "output_version": None}
        )
        output, metadata = _complete_output(
            output, resumable, config=config, output_uri=output_uri
        )
    else:
        _validate_index(output, metadata)
    _update_source_pointer(
        source_uri, metadata, output_uri=output_uri, output_version=output.version
    )
    return _result(metadata, output_uri)


def _new_export_metadata(
    source: lance.LanceDataset,
    *,
    config: ExportSlapConfig,
    model: SLAPModule,
    device: torch.device,
    preprocessing: _Preprocessing,
    static: Mapping[str, object],
    request_hash: str,
) -> _ExportMetadata:
    """Build incomplete provenance for one validated source snapshot.

    :param source: UUID-bearing source snapshot.
    :param config: Export configuration.
    :param model: Loaded SLAP module.
    :param device: Inference device.
    :param preprocessing: Frozen production input transforms.
    :param static: Request fields independent of migration.
    :param request_hash: Canonical request identity.
    :returns: Incomplete output provenance.
    :raises ValueError: Requested PQ subdivisions cannot represent the vector width.
    """
    source_rows = source.count_rows()
    vector_dimension = _infer_vector_dimension(
        source,
        model,
        device=device,
        preprocessing=preprocessing,
    )
    if config.build_index and source_rows > 0 and vector_dimension % config.num_sub_vectors != 0:
        raise ValueError("num_sub_vectors must divide the SLAP vector dimension")
    return _ExportMetadata.model_validate(
        {
            **static,
            "completed": False,
            "request_hash": request_hash,
            "source_version": source.version,
            "source_transaction_uuid": _transaction_uuid(source, source.version),
            "vector_dimension": vector_dimension,
            "source_row_count": source_rows,
            "output_row_count": source_rows * 2,
            "num_partitions": _index_partitions(config, source_rows * 2),
            "index_built": False,
            "output_version": None,
        }
    )


def _write_new_output(
    source: lance.LanceDataset,
    metadata: _ExportMetadata,
    *,
    config: ExportSlapConfig,
    model: SLAPModule,
    device: torch.device,
    preprocessing: _Preprocessing,
    output_uri: str,
) -> tuple[lance.LanceDataset, _ExportMetadata]:
    """Write retrieval rows and commit completion metadata.

    :param source: UUID-bearing source snapshot.
    :param metadata: Incomplete export provenance.
    :param config: Export configuration.
    :param model: Loaded SLAP module.
    :param device: Inference device.
    :param preprocessing: Frozen production input transforms.
    :param output_uri: Output split URI.
    :returns: Completed output dataset and provenance.
    """
    schema = _schema(metadata.vector_dimension, metadata)
    reader = pa.RecordBatchReader.from_batches(
        schema,
        _output_batches(
            source,
            model=model,
            device=device,
            batch_size=config.batch_size,
            vector_dimension=metadata.vector_dimension,
            preprocessing=preprocessing,
        ),
    )
    target, storage_options = _lance_target(output_uri)
    output = lance.write_dataset(
        reader,
        target,
        schema=schema,
        mode="create",
        storage_options=storage_options,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
    )
    return _complete_output(output, metadata, config=config, output_uri=output_uri)


def _create_output(
    *,
    config: ExportSlapConfig,
    split: str,
    model: SLAPModule,
    device: torch.device,
    preprocessing: _Preprocessing,
    source_uri: str,
    output_uri: str,
    requested_version: int,
    requested_transaction_uuid: str,
    static: Mapping[str, object],
    request_hash: str,
) -> SlapExportResult:
    """Create and publish one previously absent split output.

    :param config: Export configuration.
    :param split: Split name.
    :param model: Loaded SLAP module.
    :param device: Inference device.
    :param preprocessing: Frozen production input transforms.
    :param source_uri: Source split URI.
    :param output_uri: Output split URI.
    :param requested_version: Explicit original source pin.
    :param requested_transaction_uuid: Original source transaction identity.
    :param static: Request fields independent of migration.
    :param request_hash: Canonical request identity.
    :returns: Completed split identity.
    :raises ValueError: The requested transaction changes during migration.
    """
    source = _ensure_row_uuids(source_uri, requested_version, config.batch_size)
    requested = source.checkout_version(requested_version)
    if _transaction_uuid(requested, requested_version) != requested_transaction_uuid:
        raise ValueError("requested source transaction changed during UUID migration")
    read_shard_metadata(source.schema)
    metadata = _new_export_metadata(
        source,
        config=config,
        model=model,
        device=device,
        preprocessing=preprocessing,
        static=static,
        request_hash=request_hash,
    )
    output, completed = _write_new_output(
        source,
        metadata,
        config=config,
        model=model,
        device=device,
        preprocessing=preprocessing,
        output_uri=output_uri,
    )
    _update_source_pointer(
        source_uri, completed, output_uri=output_uri, output_version=output.version
    )
    logger.info(
        "slap_split_exported",
        split=split,
        rows=metadata.source_row_count,
        output_uri=output_uri,
    )
    return _result(completed, output_uri)


def _preflight_sources(
    config: ExportSlapConfig,
    model: SLAPModule,
    *,
    device: torch.device,
    preprocessing: _Preprocessing,
) -> None:
    """Validate every pinned split before any source can be mutated.

    :param config: Export configuration.
    :param model: Loaded model selecting the stored input field.
    :param device: Inference device.
    :param preprocessing: Frozen production input transforms.
    """
    input_field = _input_field(model)
    for split in config.splits:
        source = _open(_split_uri(config.source_root_uri, split)).checkout_version(
            config.source_versions[split]
        )
        _validate_source_inputs(
            source,
            input_field=input_field,
            preprocessing=preprocessing,
        )
        _infer_vector_dimension(
            source,
            model,
            device=device,
            preprocessing=preprocessing,
        )


def _export_split(
    config: ExportSlapConfig,
    split: str,
    *,
    model: SLAPModule,
    device: torch.device,
    preprocessing: _Preprocessing,
    checkpoint_sha256: str,
) -> SlapExportResult:
    """Export or resume one split.

    :param config: Export configuration.
    :param split: Split name.
    :param model: Loaded SLAP module.
    :param device: Inference device.
    :param preprocessing: Frozen production input transforms.
    :param checkpoint_sha256: Checkpoint content hash.
    :returns: Completed split identity.
    """
    source_uri = _split_uri(config.source_root_uri, split)
    output_uri = _split_uri(config.output_root_uri, split)
    requested_version = config.source_versions[split]
    requested_source = _open(source_uri).checkout_version(requested_version)
    _validate_source_inputs(
        requested_source,
        input_field=_input_field(model),
        preprocessing=preprocessing,
    )
    requested_transaction_uuid = _transaction_uuid(requested_source, requested_version)
    static = _static_request(
        config,
        preprocessing,
        split=split,
        checkpoint_sha256=checkpoint_sha256,
        requested_source_transaction_uuid=requested_transaction_uuid,
    )
    request_hash = _request_hash(static)
    existing = _existing_completed(output_uri, request_hash)
    if existing is not None:
        return _resume_output(
            existing, config=config, source_uri=source_uri, output_uri=output_uri
        )
    return _create_output(
        config=config,
        split=split,
        model=model,
        device=device,
        preprocessing=preprocessing,
        source_uri=source_uri,
        output_uri=output_uri,
        requested_version=requested_version,
        requested_transaction_uuid=requested_transaction_uuid,
        static=static,
        request_hash=request_hash,
    )


def export_slap(config: ExportSlapConfig) -> dict[str, SlapExportResult]:
    """Export selected source splits through the checkpoint's SLAP EMA arms.

    :param config: Validated split, model, checkpoint, and index policy.
    :returns: Completed output identity keyed by split.
    :raises ValueError: The checkpoint changes while loading the model.
    """
    checkpoint_sha256 = _checkpoint_sha256(config.ckpt_path)
    model, device = _load_model(config)
    if _checkpoint_sha256(config.ckpt_path) != checkpoint_sha256:
        raise ValueError("checkpoint changed while loading; use an immutable checkpoint file")
    preprocessing = _load_preprocessing(config, model)
    _preflight_sources(
        config,
        model,
        device=device,
        preprocessing=preprocessing,
    )
    return {
        split: _export_split(
            config,
            split,
            model=model,
            device=device,
            preprocessing=preprocessing,
            checkpoint_sha256=checkpoint_sha256,
        )
        for split in config.splits
    }
