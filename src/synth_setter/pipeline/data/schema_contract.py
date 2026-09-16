"""Immutable physical-schema contracts shared by dataset writers."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, ValidationError

CONTRACT_ID_METADATA_KEY = b"synth_setter.schema_contract.id"
CONTRACT_SAMPLE_RATE_METADATA_KEY = b"synth_setter.schema_contract.sample_rate"
CONTRACT_CHANNELS_METADATA_KEY = b"synth_setter.schema_contract.channels"
CONTRACT_NUM_SAMPLES_METADATA_KEY = b"synth_setter.schema_contract.num_samples"
CONTRACT_SCHEMA_VERSION_METADATA_KEY = b"synth_setter.schema_contract.schema_version"
EMBEDDING_NAME_METADATA_KEY = b"synth_setter.embedding.name"
EMBEDDING_ARTIFACT_METADATA_KEY = b"synth_setter.embedding.artifact"
SHARD_METADATA_SCHEMA_KEY = b"synth_setter.shard_metadata"

_CONTRACT_BINDING_METADATA_KEYS = frozenset(
    {
        CONTRACT_CHANNELS_METADATA_KEY,
        CONTRACT_ID_METADATA_KEY,
        CONTRACT_NUM_SAMPLES_METADATA_KEY,
        CONTRACT_SAMPLE_RATE_METADATA_KEY,
        CONTRACT_SCHEMA_VERSION_METADATA_KEY,
    }
)
_NON_CONTRACT_METADATA_KEYS = frozenset({SHARD_METADATA_SCHEMA_KEY})
_UNSUPPORTED_UNIFIED_EMBEDDINGS = frozenset({"param_shift", "t5gemma"})


class _SerializedAudioGeometry(BaseModel):
    """Strict geometry payload at the registry trust boundary.

    .. attribute :: model_config

        Pydantic model configuration.

    .. attribute :: sample_rate

        Stored waveform sample rate in Hz.

    .. attribute :: channels

        Stored waveform channel count.

    .. attribute :: num_samples

        Stored samples per waveform channel.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    sample_rate: int
    channels: int
    num_samples: int


class _SerializedContractDocument(BaseModel):
    """Strict contract payload at the registry trust boundary.

    .. attribute :: model_config

        Pydantic model configuration.

    .. attribute :: arrow_schema_ipc

        Base64-encoded Arrow IPC schema.

    .. attribute :: audio_geometry

        Stored waveform geometry.

    .. attribute :: contract_id

        Canonical contract digest.

    .. attribute :: logical_embeddings

        Logical embedding names mapped to ordered physical fields.

    .. attribute :: parent_contract_id

        Parent contract digest, when this contract evolves another.

    .. attribute :: schema_version

        Contract serialization version.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    arrow_schema_ipc: str
    audio_geometry: _SerializedAudioGeometry
    contract_id: str
    logical_embeddings: dict[str, list[str]]
    parent_contract_id: str | None
    schema_version: int


@dataclass(frozen=True)
class AudioGeometry:
    """Describe the stored waveform shape used by geometry-dependent fields.

    .. attribute :: sample_rate

        Stored waveform sample rate in Hz.

    .. attribute :: channels

        Stored waveform channel count.

    .. attribute :: num_samples

        Stored samples per waveform channel.
    """

    sample_rate: int
    channels: int
    num_samples: int

    def __post_init__(self) -> None:
        """Reject geometry that cannot describe stored audio.

        :raises TypeError: A geometry component is not an integer.
        :raises ValueError: A geometry component is not positive.
        """
        for name, value in (
            ("sample_rate", self.sample_rate),
            ("channels", self.channels),
            ("num_samples", self.num_samples),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class DatasetSchemaContract:
    """Bind one exact Arrow schema to geometry, provenance, and lineage.

    .. attribute :: contract_id

        SHA-256 digest of canonical contract content.

    .. attribute :: parent_contract_id

        Parent contract for schema evolution, if any.

    .. attribute :: arrow_schema

        Exact physical schema with contract metadata.

    .. attribute :: audio_geometry

        Geometry used to derive physical embedding fields.

    .. attribute :: logical_embeddings

        Sorted logical names and their ordered physical fields.

    .. attribute :: schema_version

        Contract serialization version.
    """

    contract_id: str
    parent_contract_id: str | None
    arrow_schema: pa.Schema
    audio_geometry: AudioGeometry
    logical_embeddings: tuple[tuple[str, tuple[str, ...]], ...]
    schema_version: int

    @classmethod
    def create(
        cls,
        *,
        arrow_schema: pa.Schema,
        audio_geometry: AudioGeometry,
        logical_embeddings: Mapping[str, Sequence[str]],
        parent_contract_id: str | None = None,
        schema_version: int = 1,
    ) -> DatasetSchemaContract:
        """Create a contract and bind its digest into schema metadata.

        The digest omits only its self-referential metadata value. Shard-specific metadata is
        outside the contract and is removed before hashing.

        :param arrow_schema: Complete physical schema in authoritative field order.
        :param audio_geometry: Stored waveform geometry.
        :param logical_embeddings: Logical embedding names mapped to physical fields.
        :param parent_contract_id: Parent digest for schema evolution.
        :param schema_version: Positive contract format version.
        :returns: Frozen contract carrying a self-consistent schema binding.
        :raises TypeError: Inputs do not use the declared domain types.
        :raises ValueError: Lineage, mappings, or schema content is invalid.
        """
        if not isinstance(arrow_schema, pa.Schema):
            raise TypeError("arrow_schema must be a pyarrow.Schema")
        if not isinstance(audio_geometry, AudioGeometry):
            raise TypeError("audio_geometry must be an AudioGeometry")
        if type(schema_version) is not int:
            raise TypeError("schema_version must be an integer")
        if schema_version < 1:
            raise ValueError("schema_version must be positive")
        _validate_optional_contract_id(parent_contract_id, name="parent_contract_id")

        logical = _freeze_logical_embeddings(logical_embeddings, arrow_schema)
        unsupported = sorted(
            name for name, _fields in logical if name in _UNSUPPORTED_UNIFIED_EMBEDDINGS
        )
        if unsupported:
            raise ValueError(
                f"embedding {unsupported[0]!r} is not supported by unified schema contracts"
            )

        metadata = _contract_metadata(
            arrow_schema.metadata,
            audio_geometry=audio_geometry,
            schema_version=schema_version,
        )
        unbound_schema = _canonical_schema(arrow_schema, metadata=metadata)
        contract_id = _content_digest(
            arrow_schema=unbound_schema,
            audio_geometry=audio_geometry,
            logical_embeddings=logical,
            parent_contract_id=parent_contract_id,
            schema_version=schema_version,
        )
        bound_metadata = dict(unbound_schema.metadata or {})
        bound_metadata[CONTRACT_ID_METADATA_KEY] = contract_id.encode("ascii")
        bound_schema = _canonical_schema(unbound_schema, metadata=bound_metadata)
        return cls(
            contract_id=contract_id,
            parent_contract_id=parent_contract_id,
            arrow_schema=bound_schema,
            audio_geometry=audio_geometry,
            logical_embeddings=logical,
            schema_version=schema_version,
        )

    def serialize(self) -> bytes:
        """Serialize the complete contract into canonical bytes.

        :returns: Stable JSON bytes containing the exact Arrow IPC schema.
        """
        return _canonical_json(
            {
                "arrow_schema_ipc": _schema_ipc_base64(self.arrow_schema),
                "audio_geometry": _geometry_document(self.audio_geometry),
                "contract_id": self.contract_id,
                "logical_embeddings": {
                    name: list(fields) for name, fields in self.logical_embeddings
                },
                "parent_contract_id": self.parent_contract_id,
                "schema_version": self.schema_version,
            }
        )

    @classmethod
    def deserialize(cls, payload: bytes) -> DatasetSchemaContract:
        """Validate and restore one serialized contract.

        :param payload: Bytes returned by :meth:`serialize`.
        :returns: Restored self-consistent contract.
        :raises ValueError: The payload is malformed or its digest does not match.
        """
        try:
            document = _SerializedContractDocument.model_validate_json(payload, strict=True)
            schema_bytes = base64.b64decode(document.arrow_schema_ipc, validate=True)
            schema = pa.ipc.read_schema(pa.BufferReader(schema_bytes))
            geometry = AudioGeometry(
                sample_rate=document.audio_geometry.sample_rate,
                channels=document.audio_geometry.channels,
                num_samples=document.audio_geometry.num_samples,
            )
            logical = {name: tuple(fields) for name, fields in document.logical_embeddings.items()}
            serialized_contract_id = document.contract_id
            schema_contract_id = (
                (schema.metadata or {}).get(CONTRACT_ID_METADATA_KEY, b"").decode("ascii")
            )
            restored = cls.create(
                arrow_schema=schema,
                audio_geometry=geometry,
                logical_embeddings=logical,
                parent_contract_id=document.parent_contract_id,
                schema_version=document.schema_version,
            )
        except (
            TypeError,
            UnicodeDecodeError,
            ValidationError,
            ValueError,
            pa.ArrowException,
        ) as error:
            raise ValueError("invalid dataset schema contract payload") from error
        if (
            serialized_contract_id != restored.contract_id
            or schema_contract_id != restored.contract_id
        ):
            raise ValueError("contract_id does not match canonical content")
        serialized_schema = _canonical_schema(schema, metadata=schema.metadata)
        if not serialized_schema.equals(restored.arrow_schema, check_metadata=True):
            raise ValueError("serialized schema does not match canonical content")
        return restored

    def validate_source(self, source_schema: pa.Schema) -> None:
        """Validate the non-embedding fields supplied to this contract.

        :param source_schema: Rendered schema before embedding augmentation.
        """
        embedding_fields = {
            field
            for _name, physical_fields in self.logical_embeddings
            for field in physical_fields
        }
        expected_fields = [
            field for field in self.arrow_schema if field.name not in embedding_fields
        ]
        expected_metadata = {
            key: value
            for key, value in (self.arrow_schema.metadata or {}).items()
            if key not in _CONTRACT_BINDING_METADATA_KEYS
        }
        expected = _canonical_schema(pa.schema(expected_fields), metadata=expected_metadata)
        actual = _canonical_schema(
            source_schema,
            metadata=_without_metadata_keys(
                source_schema.metadata,
                _CONTRACT_BINDING_METADATA_KEYS | _NON_CONTRACT_METADATA_KEYS,
            ),
        )
        _raise_schema_mismatch(actual, expected, context="source schema")

    def validate_dataset(self, dataset_schema: pa.Schema) -> None:
        """Validate a complete physical schema against this contract.

        :param dataset_schema: Candidate fragment or committed dataset schema.
        """
        actual = _canonical_schema(
            dataset_schema,
            metadata=_without_metadata_keys(dataset_schema.metadata, _NON_CONTRACT_METADATA_KEYS),
        )
        _raise_schema_mismatch(actual, self.arrow_schema, context="dataset schema")


def _validate_optional_contract_id(value: str | None, *, name: str) -> None:
    """Reject a non-canonical optional contract digest.

    :param value: Candidate lowercase SHA-256 digest or ``None``.
    :param name: Input name used in diagnostics.
    :raises ValueError: The value is not a lowercase SHA-256 digest.
    """
    if value is None:
        return
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _freeze_logical_embeddings(
    logical_embeddings: Mapping[str, Sequence[str]], schema: pa.Schema
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Validate and freeze logical-to-physical embedding ownership.

    :param logical_embeddings: Logical names mapped to ordered physical fields.
    :param schema: Physical schema whose embedding fields carry provenance.
    :returns: Name-sorted immutable ownership mappings.
    :raises ValueError: Ownership or field provenance is incomplete or conflicting.
    """
    owners: dict[str, str] = {}
    frozen: list[tuple[str, tuple[str, ...]]] = []
    available = set(schema.names)
    for logical_name in sorted(logical_embeddings):
        if not logical_name:
            raise ValueError("logical embedding names must be nonblank")
        physical_fields = tuple(logical_embeddings[logical_name])
        if not physical_fields:
            raise ValueError(f"logical embedding {logical_name!r} must own at least one field")
        for field_name in physical_fields:
            if field_name not in available:
                raise ValueError(
                    f"logical embedding {logical_name!r} references unknown field {field_name!r}"
                )
            if field_name in owners:
                raise ValueError(
                    f"field {field_name!r} belongs to more than one logical embedding"
                )
            owners[field_name] = logical_name
        frozen.append((logical_name, physical_fields))

    for field in schema:
        field_metadata = field.metadata or {}
        has_embedding_provenance = bool(
            field_metadata.keys()
            & {EMBEDDING_NAME_METADATA_KEY, EMBEDDING_ARTIFACT_METADATA_KEY}
        )
        if has_embedding_provenance and field.name not in owners:
            raise ValueError(f"field {field.name!r} has embedding provenance but no owner")

    for field_name, logical_name in owners.items():
        field_metadata = schema.field(field_name).metadata or {}
        declared_name = field_metadata.get(EMBEDDING_NAME_METADATA_KEY)
        if declared_name is None:
            raise ValueError(f"field {field_name!r} lacks embedding name metadata")
        if declared_name != logical_name.encode("utf-8"):
            raise ValueError(
                f"field {field_name!r} declares embedding name "
                f"{declared_name.decode('utf-8', errors='replace')!r}, "
                f"expected {logical_name!r}"
            )
        artifact = field_metadata.get(EMBEDDING_ARTIFACT_METADATA_KEY)
        if not artifact:
            raise ValueError(f"field {field_name!r} lacks embedding artifact metadata")
    return tuple(frozen)


def _contract_metadata(
    metadata: Mapping[bytes, bytes] | None,
    *,
    audio_geometry: AudioGeometry,
    schema_version: int,
) -> dict[bytes, bytes]:
    result = _without_metadata_keys(
        metadata,
        _NON_CONTRACT_METADATA_KEYS | {CONTRACT_ID_METADATA_KEY},
    )
    result.update(
        {
            CONTRACT_CHANNELS_METADATA_KEY: str(audio_geometry.channels).encode("ascii"),
            CONTRACT_NUM_SAMPLES_METADATA_KEY: str(audio_geometry.num_samples).encode("ascii"),
            CONTRACT_SAMPLE_RATE_METADATA_KEY: str(audio_geometry.sample_rate).encode("ascii"),
            CONTRACT_SCHEMA_VERSION_METADATA_KEY: str(schema_version).encode("ascii"),
        }
    )
    return result


def _without_metadata_keys(
    metadata: Mapping[bytes, bytes] | None, keys: frozenset[bytes] | set[bytes]
) -> dict[bytes, bytes]:
    return {key: value for key, value in (metadata or {}).items() if key not in keys}


def _canonical_schema(schema: pa.Schema, *, metadata: Mapping[bytes, bytes] | None) -> pa.Schema:
    """Return a schema with recursively ordered metadata.

    :param schema: Schema whose field order remains authoritative.
    :param metadata: Schema metadata to order and bind.
    :returns: Canonical schema preserving field order and physical types.
    """
    fields = [_canonical_field(field) for field in schema]
    ordered_metadata = dict(sorted((metadata or {}).items()))
    return pa.schema(fields, metadata=ordered_metadata or None)


def _canonical_field(field: pa.Field) -> pa.Field:
    """Canonicalize one field and every nested field-bearing type.

    :param field: Arrow field to canonicalize.
    :returns: Equivalent field with ordered metadata.
    """
    return pa.field(
        field.name,
        _canonical_type(field.type),
        nullable=field.nullable,
        metadata=dict(sorted((field.metadata or {}).items())) or None,
    )


def _canonical_type(data_type: pa.DataType) -> pa.DataType:
    """Canonicalize nested fields while preserving Arrow type semantics.

    :param data_type: Arrow type to canonicalize recursively.
    :returns: Equivalent type with canonical nested fields.
    :raises ValueError: The extension type has no deterministic generic rebuilder.
    """
    if pa.types.is_struct(data_type):
        return pa.struct([_canonical_field(field) for field in data_type])
    if pa.types.is_map(data_type):
        return pa.map_(
            _canonical_field(data_type.key_field),
            _canonical_field(data_type.item_field),
            keys_sorted=data_type.keys_sorted,
        )
    if pa.types.is_fixed_size_list(data_type):
        return pa.list_(_canonical_field(data_type.value_field), data_type.list_size)
    if pa.types.is_list(data_type):
        return pa.list_(_canonical_field(data_type.value_field))
    if pa.types.is_large_list(data_type):
        return pa.large_list(_canonical_field(data_type.value_field))
    if pa.types.is_list_view(data_type):
        return pa.list_view(_canonical_field(data_type.value_field))
    if pa.types.is_large_list_view(data_type):
        return pa.large_list_view(_canonical_field(data_type.value_field))
    if pa.types.is_union(data_type):
        return pa.union(
            [_canonical_field(field) for field in data_type],
            mode=data_type.mode,
            type_codes=data_type.type_codes,
        )
    if pa.types.is_dictionary(data_type):
        return pa.dictionary(
            data_type.index_type,
            _canonical_type(data_type.value_type),
            ordered=data_type.ordered,
        )
    if pa.types.is_run_end_encoded(data_type):
        return pa.run_end_encoded(
            data_type.run_end_type,
            _canonical_type(data_type.value_type),
        )
    if isinstance(data_type, pa.FixedShapeTensorType):
        return data_type
    if isinstance(data_type, pa.BaseExtensionType):
        raise ValueError(f"unsupported Arrow extension type {str(data_type)!r}")
    return data_type


def _content_digest(
    *,
    arrow_schema: pa.Schema,
    audio_geometry: AudioGeometry,
    logical_embeddings: tuple[tuple[str, tuple[str, ...]], ...],
    parent_contract_id: str | None,
    schema_version: int,
) -> str:
    document = {
        "arrow_schema_ipc": _schema_ipc_base64(arrow_schema),
        "audio_geometry": _geometry_document(audio_geometry),
        "logical_embeddings": {name: list(fields) for name, fields in logical_embeddings},
        "parent_contract_id": parent_contract_id,
        "schema_version": schema_version,
    }
    return hashlib.sha256(_canonical_json(document)).hexdigest()


def _schema_ipc_base64(schema: pa.Schema) -> str:
    return base64.b64encode(schema.serialize().to_pybytes()).decode("ascii")


def _geometry_document(geometry: AudioGeometry) -> dict[str, int]:
    return {
        "channels": geometry.channels,
        "num_samples": geometry.num_samples,
        "sample_rate": geometry.sample_rate,
    }


def _canonical_json(document: Mapping[str, Any]) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _raise_schema_mismatch(actual: pa.Schema, expected: pa.Schema, *, context: str) -> None:
    """Raise the first actionable exact-schema mismatch.

    :param actual: Candidate schema.
    :param expected: Authoritative contract schema.
    :param context: Schema role used in diagnostics.
    :raises ValueError: Fields or schema metadata differ.
    """
    if actual.equals(expected, check_metadata=True):
        return

    actual_names = actual.names
    expected_names = expected.names
    for index, expected_field in enumerate(expected):
        if index >= len(actual):
            raise ValueError(f"{context} is missing field {expected_field.name!r}")
        actual_field = actual.field(index)
        if actual_field.name != expected_field.name:
            if expected_field.name not in actual_names:
                raise ValueError(f"{context} is missing field {expected_field.name!r}")
            if actual_field.name not in expected_names:
                raise ValueError(f"{context} has unexpected field {actual_field.name!r}")
            raise ValueError(
                f"{context} field {index} is {actual_field.name!r}, "
                f"expected {expected_field.name!r}"
            )
        if not actual_field.equals(expected_field, check_metadata=True):
            raise ValueError(
                f"{context} field {expected_field.name!r} differs: "
                f"got {actual_field}, expected {expected_field}"
            )
    if len(actual) > len(expected):
        raise ValueError(f"{context} has unexpected field {actual.field(len(expected)).name!r}")
    raise ValueError(
        f"{context} metadata differs: got {actual.metadata}, expected {expected.metadata}"
    )
