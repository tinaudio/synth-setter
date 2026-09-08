"""Stream SLAP EMA projections into split retrieval datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import hydra
import lance
import numpy as np
import pyarrow as pa
import structlog
import torch
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict

from synth_setter.data.vst.shapes import AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD
from synth_setter.models.slap_module import SLAPModule
from synth_setter.pipeline import r2_io
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
_SCHEMA_VERSION = 1
_POLICY_VERSION = 1


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
    return lance.dataset(target, storage_options=storage_options)


def _transaction_uuid(dataset: lance.LanceDataset, version: int) -> str:
    """Read the transaction UUID for an exact version.

    :param dataset: Lance dataset.
    :param version: Exact version.
    :returns: Transaction UUID.
    :raises ValueError: The version has no transaction.
    """
    transaction = dataset.read_transaction(version)
    if transaction is None:
        raise ValueError(f"Lance version {version} has no transaction UUID")
    return transaction.uuid


def _checkpoint_sha256(path: Path) -> str:
    """Hash checkpoint content.

    :param path: Checkpoint file.
    :returns: SHA-256 hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    split: str,
    checkpoint_sha256: str,
    requested_source_transaction_uuid: str,
) -> dict[str, object]:
    """Build request fields independent of source UUID migration.

    :param config: Export configuration.
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
    :raises OSError: Existing output cannot be opened for a reason other than absence.
    :raises ValueError: Existing output metadata conflicts or is inconsistent.
    """
    try:
        dataset = _open(output_uri)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        if "not found" in str(exc).casefold():
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


def _validate_row_uuids(dataset: lance.LanceDataset) -> None:
    """Require canonical, nonnull, unique source UUIDs.

    :param dataset: Source snapshot.
    :raises ValueError: Any UUID is invalid or repeated.
    """
    seen: set[str] = set()
    for batch in dataset.to_batches(columns=[ROW_UUID_FIELD]):
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


def _ensure_row_uuids(
    source_uri: str, requested_version: int, batch_size: int
) -> lance.LanceDataset:
    """Return a pinned source, adding UUIDs only to the expected head.

    :param source_uri: Source dataset URI.
    :param requested_version: Explicit source pin.
    :param batch_size: UUID UDF batch size.
    :returns: UUID-bearing pinned source.
    :raises ValueError: The pin is absent or historical without UUIDs.
    """
    head = _open(source_uri)
    if requested_version > head.version:
        raise ValueError(f"source version {requested_version} does not exist")
    pinned = _open(source_uri).checkout_version(requested_version)
    if ROW_UUID_FIELD in pinned.schema.names:
        _validate_row_uuids(pinned)
        return pinned
    if requested_version != head.version:
        raise ValueError("historical pinned source lacks row_uuid; refusing to mutate newer head")

    if head.count_rows() == 0:
        head.add_columns(pa.field(ROW_UUID_FIELD, pa.string(), nullable=True))
    else:

        def add_uuid(batch: pa.RecordBatch) -> pa.RecordBatch:
            return pa.record_batch(
                {ROW_UUID_FIELD: pa.array([str(uuid4()) for _ in range(batch.num_rows)])}
            )

        head.add_columns(add_uuid, read_columns=[], batch_size=batch_size)
    _validate_row_uuids(head)
    return head


def _decoded(column: pa.Array) -> np.ndarray:
    """Decode an Arrow column while preserving tensor shape.

    :param column: Arrow array.
    :returns: NumPy rows.
    """
    if isinstance(column, pa.FixedShapeTensorArray):
        return column.to_numpy_ndarray()
    return column.to_numpy(zero_copy_only=False)


def _normalize(vectors: torch.Tensor, modality: str) -> np.ndarray:
    """Validate and L2-normalize projection rows.

    :param vectors: Projection tensor.
    :param modality: Error-context label.
    :returns: Contiguous float32 unit vectors.
    :raises ValueError: Shape, values, or norms are invalid.
    """
    values = vectors.detach().float().cpu().numpy()
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(f"{modality} projection has invalid shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{modality} projection contains nonfinite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError(f"{modality} projection contains zero or invalid vectors")
    return np.ascontiguousarray(values / norms, dtype=np.float32)


def _project_batch(
    model: SLAPModule, batch: pa.RecordBatch, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Project one source batch through both EMA arms.

    :param model: Loaded SLAP module.
    :param batch: Source rows.
    :param device: Inference device.
    :returns: Audio vectors, parameter vectors, and source UUIDs.
    :raises ValueError: EMA outputs are absent or incompatible.
    """
    input_field = AUDIO_FIELD if model.audio_input_key == "audio" else MEL_SPEC_FIELD
    audio_values = np.array(
        _decoded(batch.column(input_field)), dtype=np.float32, order="C", copy=True
    )
    param_values = np.array(
        _decoded(batch.column(PARAM_ARRAY_FIELD)), dtype=np.float32, order="C", copy=True
    )
    audio = torch.from_numpy(audio_values).to(device)
    params = torch.from_numpy(param_values).to(device)
    with torch.inference_mode():
        _, audio_projection, _ = model.audio_ema(audio)
        _, param_projection, _ = model.text_ema(params)
    if audio_projection is None or param_projection is None:
        raise ValueError("SLAP EMA arms must provide projection outputs")
    audio_vectors = _normalize(audio_projection, "audio EMA")
    param_vectors = _normalize(param_projection, "parameter EMA")
    if audio_vectors.shape != param_vectors.shape:
        raise ValueError(
            f"SLAP EMA projection dimensions differ: audio {audio_vectors.shape}, params {param_vectors.shape}"
        )
    return audio_vectors, param_vectors, batch.column(ROW_UUID_FIELD).to_pylist()


def _output_batches(
    source: lance.LanceDataset,
    model: SLAPModule,
    device: torch.device,
    batch_size: int,
    vector_dimension: int,
) -> Iterator[pa.RecordBatch]:
    """Yield interleaved parameter and audio retrieval rows.

    :yields pa.RecordBatch: Retrieval batches.
    :param source: Pinned source snapshot.
    :param model: Loaded SLAP module.
    :param device: Inference device.
    :param batch_size: Source scan batch size.
    :param vector_dimension: Required output width.
    :raises ValueError: Projection dimensions change.
    """
    input_field = AUDIO_FIELD if model.audio_input_key == "audio" else MEL_SPEC_FIELD
    columns = [ROW_UUID_FIELD, input_field, PARAM_ARRAY_FIELD]
    for batch in source.to_batches(columns=columns, batch_size=batch_size):
        audio, params, row_uuids = _project_batch(model, batch, device)
        if audio.shape[1] != vector_dimension:
            raise ValueError("SLAP projection dimension changed between batches")
        vectors = np.stack((params, audio), axis=1).reshape(-1, vector_dimension)
        repeated_uuids = [value for value in row_uuids for _ in range(2)]
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
    source: lance.LanceDataset, model: SLAPModule, device: torch.device
) -> int:
    """Infer and validate the shared EMA projection width.

    :param source: Source snapshot providing input shapes.
    :param model: Loaded SLAP module.
    :param device: Inference device.
    :returns: Shared vector dimension.
    :raises ValueError: EMA projection outputs are absent or incompatible.
    """
    input_field = AUDIO_FIELD if model.audio_input_key == "audio" else MEL_SPEC_FIELD
    audio_shape = tuple(source.schema.field(input_field).type.shape)
    param_shape = tuple(source.schema.field(PARAM_ARRAY_FIELD).type.shape)
    audio = torch.zeros((1, *audio_shape), dtype=torch.float32, device=device)
    params = torch.zeros((1, *param_shape), dtype=torch.float32, device=device)
    with torch.inference_mode():
        _, audio_projection, _ = model.audio_ema(audio)
        _, param_projection, _ = model.text_ema(params)
    if audio_projection is None or param_projection is None:
        raise ValueError("SLAP EMA arms must provide projection outputs")
    if audio_projection.ndim != 2 or param_projection.ndim != 2:
        raise ValueError("SLAP EMA projections must be rank-two")
    if audio_projection.shape[1] != param_projection.shape[1]:
        raise ValueError("SLAP EMA projection dimensions differ")
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


def _update_source_pointer(
    source_uri: str, metadata: _ExportMetadata, output_uri: str, output_version: int
) -> None:
    """Publish or repair a source pointer after output completion.

    :param source_uri: Source dataset URI.
    :param metadata: Completed export provenance.
    :param output_uri: Completed output URI.
    :param output_version: Exact completed output version.
    :raises RuntimeError: Source data advanced before pointer publication.
    """
    source = _open(source_uri)
    raw = (source.schema.metadata or {}).get(SLAP_SOURCE_POINTER_KEY)
    pointer = _SourcePointer(
        schema_version=_SCHEMA_VERSION,
        output_uri=output_uri,
        output_version=output_version,
        input_source_version=metadata.source_version,
        request_hash=metadata.request_hash,
    )
    existing_pointer = None if raw is None else _SourcePointer.model_validate_json(raw)
    if existing_pointer == pointer:
        return
    repairs_prior_pointer = (
        existing_pointer is not None
        and existing_pointer.request_hash == pointer.request_hash
        and existing_pointer.output_uri == pointer.output_uri
        and existing_pointer.input_source_version == pointer.input_source_version
    )
    if source.version != metadata.source_version and not repairs_prior_pointer:
        raise RuntimeError(
            "source advanced after export; completed output retained without publishing pointer"
        )
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

    source_head = _open(source_uri)
    requested = source_head.checkout_version(metadata.requested_source_version)
    requested_uuid = _transaction_uuid(requested, metadata.requested_source_version)
    if requested_uuid != metadata.requested_source_transaction_uuid:
        raise ValueError("requested source transaction identity differs from output provenance")
    pinned = source_head.checkout_version(metadata.source_version)
    source_uuid = _transaction_uuid(pinned, metadata.source_version)
    if source_uuid != metadata.source_transaction_uuid:
        raise ValueError("source transaction identity differs from output provenance")
    if pinned.count_rows() != metadata.source_row_count:
        raise ValueError("source row count differs from output provenance")
    _validate_row_uuids(pinned)
    read_shard_metadata(pinned.schema)


def _result(metadata: _ExportMetadata, output_uri: str) -> SlapExportResult:
    """Project persisted provenance onto the public result.

    :param metadata: Completed export provenance.
    :param output_uri: Output dataset URI.
    :returns: Public split result.
    """
    assert metadata.output_version is not None
    return SlapExportResult(
        split=metadata.split,
        output_uri=output_uri,
        output_version=metadata.output_version,
        source_version=metadata.source_version,
        source_transaction_uuid=metadata.source_transaction_uuid,
        source_row_count=metadata.source_row_count,
        output_row_count=metadata.output_row_count,
    )


def _export_split(
    config: ExportSlapConfig,
    split: str,
    model: SLAPModule,
    device: torch.device,
    checkpoint_sha256: str,
) -> SlapExportResult:
    """Export or resume one split.

    :param config: Export configuration.
    :param split: Split name.
    :param model: Loaded SLAP module.
    :param device: Inference device.
    :param checkpoint_sha256: Checkpoint content hash.
    :returns: Completed split identity.
    :raises ValueError: Persisted source or output identity is inconsistent.
    """
    source_uri = _split_uri(config.source_root_uri, split)
    output_uri = _split_uri(config.output_root_uri, split)
    requested_version = config.source_versions[split]
    requested_source = _open(source_uri).checkout_version(requested_version)
    requested_transaction_uuid = _transaction_uuid(requested_source, requested_version)
    static = _static_request(config, split, checkpoint_sha256, requested_transaction_uuid)
    request_hash = _request_hash(static)
    existing = _existing_completed(output_uri, request_hash)
    if existing is not None and existing[1].completed:
        output, completed = existing
        _validate_reusable_output(source_uri, output, completed)
        index_missing = (
            completed.build_index and output.count_rows() > 0 and not output.list_indices()
        )
        if index_missing:
            resumable = completed.model_copy(
                update={"completed": False, "index_built": False, "output_version": None}
            )
            output, completed = _complete_output(output, resumable, config, output_uri)
        else:
            _validate_index(output, completed)
        _update_source_pointer(source_uri, completed, output_uri, output.version)
        return _result(completed, output_uri)

    if existing is not None:
        output, persisted = existing
        _validate_reusable_output(source_uri, output, persisted)
        output, completed = _complete_output(output, persisted, config, output_uri)
        _update_source_pointer(source_uri, completed, output_uri, output.version)
        return _result(completed, output_uri)

    source = _ensure_row_uuids(source_uri, requested_version, config.batch_size)
    requested_after_migration = source.checkout_version(requested_version)
    if _transaction_uuid(requested_after_migration, requested_version) != requested_transaction_uuid:
        raise ValueError("requested source transaction changed during UUID migration")
    read_shard_metadata(source.schema)
    source_version = source.version
    source_transaction_uuid = _transaction_uuid(source, source_version)
    source_rows = source.count_rows()
    vector_dimension = _infer_vector_dimension(source, model, device)
    partitions = _index_partitions(config, source_rows * 2)
    if config.build_index and source_rows > 0 and vector_dimension % config.num_sub_vectors != 0:
        raise ValueError("num_sub_vectors must divide the SLAP vector dimension")
    metadata = _ExportMetadata.model_validate(
        {
            **static,
            "completed": False,
            "request_hash": request_hash,
            "source_version": source_version,
            "source_transaction_uuid": source_transaction_uuid,
            "vector_dimension": vector_dimension,
            "source_row_count": source_rows,
            "output_row_count": source_rows * 2,
            "num_partitions": partitions,
            "index_built": False,
            "output_version": None,
        }
    )
    schema = _schema(vector_dimension, metadata)
    reader = pa.RecordBatchReader.from_batches(
        schema,
        _output_batches(source, model, device, config.batch_size, vector_dimension),
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
    output, completed = _complete_output(output, metadata, config, output_uri)
    _update_source_pointer(source_uri, completed, output_uri, output.version)
    logger.info("slap_split_exported", split=split, rows=source_rows, output_uri=output_uri)
    return _result(completed, output_uri)


def export_slap(config: ExportSlapConfig) -> dict[str, SlapExportResult]:
    """Export selected source splits through the checkpoint's SLAP EMA arms.

    :param config: Validated split, model, checkpoint, and index policy.
    :returns: Completed output identity keyed by split.
    """
    checkpoint_sha256 = _checkpoint_sha256(config.ckpt_path)
    model, device = _load_model(config)
    return {
        split: _export_split(config, split, model, device, checkpoint_sha256)
        for split in config.splits
    }
