"""Behavioral tests for immutable dataset schema contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import pyarrow as pa
import pytest

from synth_setter.pipeline.data.schema_contract import (
    CONTRACT_CHANNELS_METADATA_KEY,
    CONTRACT_ID_METADATA_KEY,
    CONTRACT_NUM_SAMPLES_METADATA_KEY,
    CONTRACT_SAMPLE_RATE_METADATA_KEY,
    CONTRACT_SCHEMA_VERSION_METADATA_KEY,
    AudioGeometry,
    DatasetSchemaContract,
)

_SHARD_METADATA_KEY = b"synth_setter.shard_metadata"
_EMBEDDING_NAME_KEY = b"synth_setter.embedding.name"
_EMBEDDING_ARTIFACT_KEY = b"synth_setter.embedding.artifact"
_GEOMETRY = AudioGeometry(sample_rate=44_100, channels=2, num_samples=176_400)


def _embedding_field(
    name: str = "clap",
    *,
    width: int = 4,
    nullable: bool = False,
    artifact: bytes = b"clap:artifact-a",
    embedding_name: str | None = None,
) -> pa.Field:
    return pa.field(
        name,
        pa.list_(pa.float32(), width),
        nullable=nullable,
        metadata={
            _EMBEDDING_NAME_KEY: (embedding_name or name).encode(),
            _EMBEDDING_ARTIFACT_KEY: artifact,
        },
    )


def _physical_schema(
    *embedding_fields: pa.Field,
    metadata: dict[bytes, bytes] | None = None,
) -> pa.Schema:
    return pa.schema(
        [
            pa.field("audio", pa.fixed_shape_tensor(pa.float16(), (2, 176_400)), nullable=False),
            pa.field("params", pa.list_(pa.float32(), 3), nullable=False),
            *embedding_fields,
        ],
        metadata=metadata,
    )


def _contract(
    *,
    schema: pa.Schema | None = None,
    geometry: AudioGeometry = _GEOMETRY,
    logical_embeddings: dict[str, tuple[str, ...]] | None = None,
    parent_contract_id: str | None = None,
    schema_version: int = 1,
) -> DatasetSchemaContract:
    return DatasetSchemaContract.create(
        arrow_schema=_physical_schema(_embedding_field()) if schema is None else schema,
        audio_geometry=geometry,
        logical_embeddings={"clap": ("clap",)}
        if logical_embeddings is None
        else logical_embeddings,
        parent_contract_id=parent_contract_id,
        schema_version=schema_version,
    )


def test_contract_create_binds_identity_and_geometry_to_schema_metadata() -> None:
    """Contract metadata carries the digest and stored audio geometry."""
    contract = _contract()

    assert len(contract.contract_id) == 64
    assert contract.arrow_schema.metadata == {
        CONTRACT_CHANNELS_METADATA_KEY: b"2",
        CONTRACT_ID_METADATA_KEY: contract.contract_id.encode(),
        CONTRACT_NUM_SAMPLES_METADATA_KEY: b"176400",
        CONTRACT_SAMPLE_RATE_METADATA_KEY: b"44100",
        CONTRACT_SCHEMA_VERSION_METADATA_KEY: b"1",
    }


def test_contract_create_with_mapping_and_metadata_order_changes_preserves_id() -> None:
    """Semantically unordered mappings do not alter canonical bytes."""
    clap = _embedding_field()
    m2l = _embedding_field("m2l", artifact=b"m2l:artifact-a")
    first = DatasetSchemaContract.create(
        arrow_schema=_physical_schema(
            clap.with_metadata(
                {
                    _EMBEDDING_NAME_KEY: b"clap",
                    _EMBEDDING_ARTIFACT_KEY: b"clap:artifact-a",
                }
            ),
            m2l,
            metadata={b"z": b"last", b"a": b"first"},
        ),
        audio_geometry=_GEOMETRY,
        logical_embeddings={"m2l": ("m2l",), "clap": ("clap",)},
    )
    second = DatasetSchemaContract.create(
        arrow_schema=_physical_schema(
            clap.with_metadata(
                {
                    _EMBEDDING_ARTIFACT_KEY: b"clap:artifact-a",
                    _EMBEDDING_NAME_KEY: b"clap",
                }
            ),
            m2l,
            metadata={b"a": b"first", b"z": b"last"},
        ),
        audio_geometry=_GEOMETRY,
        logical_embeddings={"clap": ("clap",), "m2l": ("m2l",)},
    )

    assert first.contract_id == second.contract_id
    assert first.serialize() == second.serialize()


@pytest.mark.parametrize(
    ("changed", "logical_embeddings", "geometry", "schema_version"),
    [
        (
            _physical_schema(_embedding_field(width=8)),
            {"clap": ("clap",)},
            _GEOMETRY,
            1,
        ),
        (
            pa.schema(list(reversed(_physical_schema(_embedding_field())))),
            {"clap": ("clap",)},
            _GEOMETRY,
            1,
        ),
        (
            _physical_schema(_embedding_field(nullable=True)),
            {"clap": ("clap",)},
            _GEOMETRY,
            1,
        ),
        (
            _physical_schema(_embedding_field(artifact=b"clap:artifact-b")),
            {"clap": ("clap",)},
            _GEOMETRY,
            1,
        ),
        (
            _physical_schema(_embedding_field(embedding_name="audio_embedding")),
            {"audio_embedding": ("clap",)},
            _GEOMETRY,
            1,
        ),
        (
            _physical_schema(_embedding_field()),
            {"clap": ("clap",)},
            AudioGeometry(sample_rate=48_000, channels=2, num_samples=176_400),
            1,
        ),
        (
            _physical_schema(_embedding_field()),
            {"clap": ("clap",)},
            _GEOMETRY,
            2,
        ),
        (
            _physical_schema(_embedding_field(), metadata={b"dataset.policy": b"changed"}),
            {"clap": ("clap",)},
            _GEOMETRY,
            1,
        ),
    ],
)
def test_contract_id_when_contract_content_changes_changes(
    changed: pa.Schema,
    logical_embeddings: dict[str, tuple[str, ...]],
    geometry: AudioGeometry,
    schema_version: int,
) -> None:
    """Every contract-owned physical or semantic change alters identity.

    :param changed: Candidate physical schema.
    :param logical_embeddings: Candidate logical-to-physical mapping.
    :param geometry: Candidate stored audio geometry.
    :param schema_version: Candidate contract format version.
    """
    baseline = _contract()
    modified = DatasetSchemaContract.create(
        arrow_schema=changed,
        audio_geometry=geometry,
        logical_embeddings=logical_embeddings,
        schema_version=schema_version,
    )

    assert modified.contract_id != baseline.contract_id


def test_contract_id_with_parent_lineage_change_changes() -> None:
    """Schema-evolution lineage participates in contract identity."""
    baseline = _contract()
    child = _contract(parent_contract_id="a" * 64)

    assert child.contract_id != baseline.contract_id


def test_contract_id_with_nested_metadata_order_change_stays_stable() -> None:
    """Nested Arrow field metadata is canonicalized recursively."""
    first_map = pa.map_(
        pa.string(),
        pa.field("value", pa.int32(), metadata={b"z": b"last", b"a": b"first"}),
    )
    second_map = pa.map_(
        pa.string(),
        pa.field("value", pa.int32(), metadata={b"a": b"first", b"z": b"last"}),
    )
    first = _contract(schema=_physical_schema(_embedding_field(), pa.field("tags", first_map)))
    second = _contract(schema=_physical_schema(_embedding_field(), pa.field("tags", second_map)))

    assert first.contract_id == second.contract_id
    assert first.serialize() == second.serialize()


def test_contract_id_with_dictionary_metadata_order_change_stays_stable() -> None:
    """Dictionary value metadata is canonicalized recursively."""
    first_value = pa.struct(
        [pa.field("value", pa.int32(), metadata={b"z": b"last", b"a": b"first"})]
    )
    second_value = pa.struct(
        [pa.field("value", pa.int32(), metadata={b"a": b"first", b"z": b"last"})]
    )
    first = _contract(
        schema=_physical_schema(
            _embedding_field(),
            pa.field("tags", pa.dictionary(pa.int16(), first_value)),
        )
    )
    second = _contract(
        schema=_physical_schema(
            _embedding_field(),
            pa.field("tags", pa.dictionary(pa.int16(), second_value)),
        )
    )

    assert first.contract_id == second.contract_id
    assert first.serialize() == second.serialize()


def test_contract_id_with_run_end_metadata_order_change_stays_stable() -> None:
    """Run-end value metadata is canonicalized recursively."""
    first_value = pa.struct(
        [pa.field("value", pa.int32(), metadata={b"z": b"last", b"a": b"first"})]
    )
    second_value = pa.struct(
        [pa.field("value", pa.int32(), metadata={b"a": b"first", b"z": b"last"})]
    )
    first = _contract(
        schema=_physical_schema(
            _embedding_field(),
            pa.field("runs", pa.run_end_encoded(pa.int16(), first_value)),
        )
    )
    second = _contract(
        schema=_physical_schema(
            _embedding_field(),
            pa.field("runs", pa.run_end_encoded(pa.int16(), second_value)),
        )
    )

    assert first.contract_id == second.contract_id
    assert first.serialize() == second.serialize()


def test_contract_create_with_unsupported_extension_type_raises() -> None:
    """Extension types without a canonical rebuilder are rejected."""
    schema = _physical_schema(_embedding_field(), pa.field("json", pa.json_()))

    with pytest.raises(ValueError, match="unsupported Arrow extension type 'extension<arrow.json>'"):
        _contract(schema=schema)


def test_contract_id_with_shard_metadata_change_stays_stable() -> None:
    """Per-shard provenance is the sole normalized schema metadata."""
    first = _contract(
        schema=_physical_schema(
            _embedding_field(), metadata={_SHARD_METADATA_KEY: b'{"sample_offset":0}'}
        )
    )
    second = _contract(
        schema=_physical_schema(
            _embedding_field(), metadata={_SHARD_METADATA_KEY: b'{"sample_offset":100}'}
        )
    )

    assert first.contract_id == second.contract_id
    assert _SHARD_METADATA_KEY not in first.arrow_schema.metadata


def test_schema_contract_import_does_not_load_lance_or_model_packages() -> None:
    """The pure contract module does not acquire writer or model dependencies."""
    script = """
import sys
import synth_setter.pipeline.data.schema_contract
for name in ("lance", "torch", "transformers"):
    assert name not in sys.modules, f"{name} leaked into the schema contract module"
"""

    subprocess.run(  # noqa: S603 — interpreter and program are fixed
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_contract_id_is_stable_in_fresh_python_process() -> None:
    """Canonical content hashes identically outside the current process."""
    local_id = _contract().contract_id
    script = """
import pyarrow as pa
from synth_setter.pipeline.data.schema_contract import AudioGeometry, DatasetSchemaContract
field = pa.field(
    "clap",
    pa.list_(pa.float32(), 4),
    nullable=False,
    metadata={
        b"synth_setter.embedding.artifact": b"clap:artifact-a",
        b"synth_setter.embedding.name": b"clap",
    },
)
schema = pa.schema([
    pa.field("audio", pa.fixed_shape_tensor(pa.float16(), (2, 176400)), nullable=False),
    pa.field("params", pa.list_(pa.float32(), 3), nullable=False),
    field,
])
contract = DatasetSchemaContract.create(
    arrow_schema=schema,
    audio_geometry=AudioGeometry(sample_rate=44100, channels=2, num_samples=176400),
    logical_embeddings={"clap": ("clap",)},
)
print(contract.contract_id)
"""

    completed = subprocess.run(  # noqa: S603 — interpreter and program are fixed
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == local_id


def test_contract_logical_embedding_can_own_ordered_physical_fields() -> None:
    """One logical embedding preserves its physical field ordering."""
    sequence = pa.field(
        "m2l",
        pa.fixed_shape_tensor(pa.float32(), (16, 8)),
        nullable=False,
        metadata={
            _EMBEDDING_NAME_KEY: b"m2l",
            _EMBEDDING_ARTIFACT_KEY: b"m2l:artifact-a",
        },
    )
    vector = _embedding_field(
        "m2l_vec",
        width=16,
        artifact=b"m2l:artifact-a",
        embedding_name="m2l",
    )

    contract = DatasetSchemaContract.create(
        arrow_schema=_physical_schema(sequence, vector),
        audio_geometry=_GEOMETRY,
        logical_embeddings={"m2l": ("m2l", "m2l_vec")},
    )

    assert contract.logical_embeddings == (("m2l", ("m2l", "m2l_vec")),)


def test_contract_serialize_round_trip_preserves_exact_contract() -> None:
    """Canonical serialization restores exact Arrow and domain content."""
    contract = _contract(parent_contract_id="b" * 64)

    restored = DatasetSchemaContract.deserialize(contract.serialize())

    assert restored == contract
    assert restored.arrow_schema.equals(contract.arrow_schema, check_metadata=True)
    assert restored.serialize() == contract.serialize()


def test_contract_when_mutated_raises_frozen_instance_error() -> None:
    """Published contracts are immutable domain values."""
    contract = _contract()

    with pytest.raises(FrozenInstanceError):
        contract.schema_version = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("sample_rate", True),
        ("sample_rate", 44_100.0),
        ("channels", True),
        ("channels", 1.0),
        ("num_samples", True),
        ("num_samples", 16_000.0),
    ],
)
def test_audio_geometry_with_non_integer_component_raises(
    field_name: str, invalid_value: object
) -> None:
    """Geometry rejects values that strict serialized contracts cannot restore.

    :param field_name: Geometry component under test.
    :param invalid_value: Non-integer candidate for that component.
    """
    values: dict[str, object] = {
        "sample_rate": 44_100,
        "channels": 1,
        "num_samples": 16_000,
    }
    values[field_name] = invalid_value

    with pytest.raises(TypeError, match=rf"{field_name} must be an integer"):
        AudioGeometry(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_contract_create_with_non_integer_schema_version_raises(
    schema_version: object,
) -> None:
    """Contract versions use the same strict integer domain as deserialization.

    :param schema_version: Non-integer candidate contract version.
    """
    with pytest.raises(TypeError, match="schema_version must be an integer"):
        _contract(schema_version=schema_version)  # type: ignore[arg-type]


def test_contract_validate_source_with_exact_base_fields_accepts() -> None:
    """Shard-specific metadata does not invalidate exact source fields."""
    contract = _contract()
    source = _physical_schema(metadata={_SHARD_METADATA_KEY: b'{"sample_offset":0}'})

    contract.validate_source(source)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (pa.schema([_physical_schema().field("audio")]), "missing field 'params'"),
        (
            pa.schema(list(reversed(_physical_schema()))),
            "field 0 is 'params', expected 'audio'",
        ),
        (
            pa.schema(
                [
                    _physical_schema().field("audio"),
                    pa.field("params", pa.list_(pa.float64(), 3), nullable=False),
                ]
            ),
            "field 'params' differs",
        ),
        (
            _physical_schema(pa.field("unexpected", pa.int64())),
            "unexpected field 'unexpected'",
        ),
    ],
)
def test_contract_validate_source_with_schema_drift_raises(
    source: pa.Schema, message: str
) -> None:
    """Source validation identifies its first contract mismatch.

    :param source: Candidate rendered source schema.
    :param message: Expected actionable mismatch fragment.
    """
    contract = _contract()

    with pytest.raises(ValueError, match=message):
        contract.validate_source(source)


def test_contract_validate_dataset_with_exact_schema_accepts() -> None:
    """The authoritative bound schema validates without normalization."""
    contract = _contract()

    contract.validate_dataset(contract.arrow_schema)


def test_contract_validate_dataset_with_allowed_shard_metadata_difference_accepts() -> None:
    """Only explicitly non-contract shard metadata may vary."""
    contract = _contract()
    candidate = contract.arrow_schema.with_metadata(
        {**contract.arrow_schema.metadata, _SHARD_METADATA_KEY: b'{"sample_offset":100}'}
    )

    contract.validate_dataset(candidate)


def test_contract_validate_dataset_with_field_metadata_drift_raises() -> None:
    """Embedding artifact drift fails complete-schema validation."""
    contract = _contract()
    index = contract.arrow_schema.get_field_index("clap")
    drifted = contract.arrow_schema.set(
        index,
        contract.arrow_schema.field(index).with_metadata(
            {
                _EMBEDDING_NAME_KEY: b"clap",
                _EMBEDDING_ARTIFACT_KEY: b"clap:artifact-b",
            }
        ),
    )

    with pytest.raises(ValueError, match="field 'clap' differs"):
        contract.validate_dataset(drifted)


@pytest.mark.parametrize("embedding", ["param_shift", "t5gemma"])
def test_contract_create_with_unsupported_unified_embedding_raises(embedding: str) -> None:
    """Separate-path transforms cannot enter unified contracts.

    :param embedding: Unsupported logical embedding name.
    """
    with pytest.raises(ValueError, match=f"{embedding!r} is not supported"):
        DatasetSchemaContract.create(
            arrow_schema=_physical_schema(_embedding_field(embedding)),
            audio_geometry=_GEOMETRY,
            logical_embeddings={embedding: (embedding,)},
        )


def test_contract_create_with_unknown_physical_field_raises() -> None:
    """Logical mappings cannot reference absent physical fields."""
    with pytest.raises(
        ValueError, match="logical embedding 'clap' references unknown field 'missing'"
    ):
        DatasetSchemaContract.create(
            arrow_schema=_physical_schema(_embedding_field()),
            audio_geometry=_GEOMETRY,
            logical_embeddings={"clap": ("missing",)},
        )


def test_contract_create_with_missing_embedding_provenance_raises() -> None:
    """Every embedding-owned field carries complete frozen provenance."""
    field = _embedding_field().with_metadata({_EMBEDDING_NAME_KEY: b"clap"})

    with pytest.raises(ValueError, match="field 'clap' lacks embedding artifact metadata"):
        DatasetSchemaContract.create(
            arrow_schema=_physical_schema(field),
            audio_geometry=_GEOMETRY,
            logical_embeddings={"clap": ("clap",)},
        )


def test_contract_create_with_unowned_embedding_provenance_raises() -> None:
    """Every provenance-marked physical field has one logical owner."""
    with pytest.raises(ValueError, match="field 'clap' has embedding provenance but no owner"):
        DatasetSchemaContract.create(
            arrow_schema=_physical_schema(_embedding_field()),
            audio_geometry=_GEOMETRY,
            logical_embeddings={},
        )


def test_contract_create_with_wrong_embedding_owner_metadata_raises() -> None:
    """Physical field provenance names its logical embedding owner."""
    field = _embedding_field(embedding_name="other")

    with pytest.raises(ValueError, match="field 'clap' declares embedding name 'other'"):
        DatasetSchemaContract.create(
            arrow_schema=_physical_schema(field),
            audio_geometry=_GEOMETRY,
            logical_embeddings={"clap": ("clap",)},
        )


def test_contract_create_with_duplicate_physical_owner_raises() -> None:
    """A physical field has exactly one logical embedding owner."""
    with pytest.raises(
        ValueError, match="field 'clap' belongs to more than one logical embedding"
    ):
        DatasetSchemaContract.create(
            arrow_schema=_physical_schema(_embedding_field()),
            audio_geometry=_GEOMETRY,
            logical_embeddings={"audio_embedding": ("clap",), "clap": ("clap",)},
        )


def test_contract_deserialize_with_tampered_contract_id_raises() -> None:
    """Deserialization rejects a forged self-binding digest."""
    contract = _contract()
    tampered_schema = contract.arrow_schema.with_metadata(
        {**contract.arrow_schema.metadata, CONTRACT_ID_METADATA_KEY: b"0" * 64}
    )
    tampered = replace(contract, contract_id="0" * 64, arrow_schema=tampered_schema)

    with pytest.raises(ValueError, match="contract_id does not match canonical content"):
        DatasetSchemaContract.deserialize(tampered.serialize())


def test_contract_deserialize_with_tampered_geometry_metadata_raises() -> None:
    """Deserialization rejects disagreement between metadata and geometry."""
    contract = _contract()
    tampered_schema = contract.arrow_schema.with_metadata(
        {**contract.arrow_schema.metadata, CONTRACT_SAMPLE_RATE_METADATA_KEY: b"48000"}
    )
    tampered = replace(contract, arrow_schema=tampered_schema)

    with pytest.raises(ValueError, match="serialized schema does not match canonical content"):
        DatasetSchemaContract.deserialize(tampered.serialize())


def test_contract_deserialize_with_coerced_numeric_string_raises() -> None:
    """Registry payloads are parsed strictly instead of coercing field types."""
    document = json.loads(_contract().serialize())
    document["schema_version"] = "1"

    with pytest.raises(ValueError, match="invalid dataset schema contract payload"):
        DatasetSchemaContract.deserialize(json.dumps(document).encode())


def test_contract_deserialize_with_unknown_payload_field_raises() -> None:
    """Registry payloads reject unknown fields at the trust boundary."""
    document = json.loads(_contract().serialize())
    document["dataset_uri"] = "r2://bucket/data.lance"

    with pytest.raises(ValueError, match="invalid dataset schema contract payload"):
        DatasetSchemaContract.deserialize(json.dumps(document).encode())
