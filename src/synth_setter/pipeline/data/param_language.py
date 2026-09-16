"""Canonical parameter descriptions and their dataset-level Lance embeddings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline import r2_io

if TYPE_CHECKING:
    from lance import LanceDataset

PARAM_NAME_DATASET = "params.lance"
PARAM_NAME_COMPLETE = "params.lance.complete"
PARAM_FIELD_INDEX_FIELD = "field_index"
PARAM_FIELD_NAME_FIELD = "field_name"
PARAM_DESCRIPTION_FIELD = "description"
PARAM_NAME_EMBEDDING_FIELD = "param_name_embedding"
PARAM_NAME_EMBEDDING_REGISTRY_KEY = "param_name"
EMBEDDING_MODEL = "google/embeddinggemma-300m"
EMBEDDING_REVISION = "57c266a740f537b4dc058e1b0cda161fd15afa75"
_PARAM_NAME_METADATA_KEY = b"synth_setter:param_name_metadata"
_SUPPORTED_DIMENSIONS = (128, 256, 512, 768)


class ParamNameDatasetMetadata(BaseModel):
    """Strict identity for a field-description Lance dataset.

    .. attribute :: model_config

        Frozen JSON boundary rejecting unknown metadata.

    .. attribute :: version

        Dataset schema version.

    .. attribute :: model

        Pinned embedding model identifier.

    .. attribute :: revision

        Immutable model revision.

    .. attribute :: extraction

        Embedding and Matryoshka extraction contract.

    .. attribute :: dimension

        Selected embedding width.

    .. attribute :: param_spec_name

        Registered logical field layout.

    .. attribute :: synth_name

        Synth identity included in descriptions.

    .. attribute :: descriptions_sha256

        Digest of descriptions in logical-field order.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    version: Literal[1] = 1
    model: Literal["google/embeddinggemma-300m"] = EMBEDDING_MODEL
    revision: Literal["57c266a740f537b4dc058e1b0cda161fd15afa75"] = EMBEDDING_REVISION
    extraction: Literal["encode_document;truncate;l2_normalize"] = (
        "encode_document;truncate;l2_normalize"
    )
    dimension: Literal[128, 256, 512, 768]
    param_spec_name: str
    synth_name: str
    descriptions_sha256: str


def describe_fields(param_spec_name: str, synth_name: str) -> list[str]:
    """Describe fields without interpreting renderer-native ranges as physical units.

    :param param_spec_name: Registered parameter specification.
    :param synth_name: Synth identity recorded with the dataset.
    :returns: Deterministic descriptions in encoded field order.
    """
    spec = resolve_param_spec(ParamSpecName(param_spec_name))
    descriptions = []
    for field, span in spec.encoded_slices():
        metadata = {
            "synth": synth_name,
            "spec": param_spec_name,
            "name": field.name,
            "type": type(field).__name__,
            "encoded_span": [span.start, span.stop],
        }
        for attribute in ("encoding", "values", "min", "max", "shape"):
            if hasattr(field, attribute):
                metadata[attribute] = getattr(field, attribute)
        descriptions.append(json.dumps(metadata, sort_keys=True, ensure_ascii=False))
    return descriptions


def matryoshka_vectors(embeddings: np.ndarray, dimension: int) -> np.ndarray:
    """Truncate EmbeddingGemma vectors and normalize retained coordinates.

    :param embeddings: Native-width float32 vectors shaped ``(fields, 768)``.
    :param dimension: Supported Matryoshka width.
    :returns: Unit-normalized float32 matrix shaped ``(fields, dimension)``.
    :raises ValueError: Width, values, or prefix norms are invalid.
    """
    if dimension not in _SUPPORTED_DIMENSIONS:
        raise ValueError("unsupported EmbeddingGemma Matryoshka dimension")
    if embeddings.ndim != 2:
        raise ValueError("parameter name embeddings must be a matrix")
    _validate_vectors(embeddings, embeddings.shape[0], 768)
    prefix = embeddings[:, :dimension].copy()
    norms = np.linalg.norm(prefix.astype(np.float64), axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Matryoshka prefixes must have nonzero norms")
    return (prefix / norms).astype(np.float32)


def _description_digest(descriptions: list[str]) -> str:
    """Hash ordered descriptions without ambiguous concatenation.

    :param descriptions: Canonical description rows.
    :returns: SHA-256 of their JSON representation.
    """
    return hashlib.sha256(json.dumps(descriptions, ensure_ascii=False).encode()).hexdigest()


def _validate_vectors(embeddings: np.ndarray, count: int, dimension: int) -> None:
    """Reject malformed field embeddings.

    :param embeddings: Field-major matrix.
    :param count: Expected logical field count.
    :param dimension: Expected vector width.
    :raises ValueError: Shape, dtype, or values violate the contract.
    """
    if embeddings.shape != (count, dimension) or embeddings.dtype != np.float32:
        raise ValueError("parameter name embeddings require aligned float32 field vectors")
    if not np.isfinite(embeddings).all():
        raise ValueError("parameter name embeddings must be finite")


def _validate_unit_norm(embeddings: np.ndarray) -> None:
    """Reject vectors outside the normalized embedding contract.

    :param embeddings: Validated field-major matrix.
    :raises ValueError: A row is not unit normalized.
    """
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError("parameter name embeddings must have unit norm")


def write_param_name_dataset(
    path: Path, param_spec_name: str, synth_name: str, *, dimension: int = 128
) -> None:
    """Write one minimal Lance row per logical parameter field.

    :param path: Destination ``params.lance`` directory.
    :param param_spec_name: Registered logical field layout.
    :param synth_name: Synth identity included in every canonical description.
    :param dimension: Selected Matryoshka embedding width.
    :raises ValueError: The selected dimension is unsupported.
    """
    from synth_setter.pipeline.data.lance_shard import write_lance_dataset

    if dimension not in _SUPPORTED_DIMENSIONS:
        raise ValueError("unsupported EmbeddingGemma Matryoshka dimension")
    spec = resolve_param_spec(ParamSpecName(param_spec_name))
    descriptions = describe_fields(param_spec_name, synth_name)
    names = [field.name for field, _ in spec.encoded_slices()]
    metadata = ParamNameDatasetMetadata(
        dimension=dimension,
        param_spec_name=param_spec_name,
        synth_name=synth_name,
        descriptions_sha256=_description_digest(descriptions),
    )
    schema = pa.schema(
        [
            pa.field(PARAM_FIELD_INDEX_FIELD, pa.int32(), nullable=False),
            pa.field(PARAM_FIELD_NAME_FIELD, pa.string(), nullable=False),
            pa.field(PARAM_DESCRIPTION_FIELD, pa.string(), nullable=False),
        ],
        metadata={_PARAM_NAME_METADATA_KEY: metadata.model_dump_json().encode()},
    )
    table = pa.Table.from_arrays(
        [pa.array(range(len(names))), pa.array(names), pa.array(descriptions)], schema=schema
    )
    write_lance_dataset(path, schema, table.to_batches())


def _open_param_name_dataset(path: Path | str) -> LanceDataset:
    """Open a local or credentialed R2 parameter-name dataset.

    :param path: Local path or R2 URI.
    :returns: Open Lance dataset.
    """
    import lance

    uri = str(path)
    if r2_io.is_r2_uri(uri):
        r2_io.ensure_r2_env_loaded()
        return lance.dataset(r2_io.to_s3_uri(uri), storage_options=r2_io.r2_storage_options())
    return lance.dataset(uri)


def param_name_dataset_metadata(schema: pa.Schema) -> ParamNameDatasetMetadata:
    """Validate and return parameter-name provenance from an Arrow schema.

    :param schema: Field dataset schema.
    :returns: Strict model, revision, dimension, and field-layout identity.
    :raises ValueError: Provenance metadata is absent.
    """
    encoded = (schema.metadata or {}).get(_PARAM_NAME_METADATA_KEY)
    if encoded is None:
        raise ValueError("parameter name dataset lacks provenance metadata")
    return ParamNameDatasetMetadata.model_validate_json(encoded)


def load_param_name_embeddings(
    path: Path | str, param_spec_name: str, synth_name: str
) -> tuple[np.ndarray, ParamNameDatasetMetadata]:
    """Load field vectors in explicit logical-field-index order.

    :param path: Local or R2-backed ``params.lance`` directory.
    :param param_spec_name: Expected registered logical field layout.
    :param synth_name: Expected synth identity.
    :returns: Unit-normalized float32 field matrix and validated provenance.
    :raises ValueError: Schema, identity, row order, descriptions, or vectors mismatch.
    """
    dataset = _open_param_name_dataset(path)
    metadata = param_name_dataset_metadata(dataset.schema)
    expected_descriptions = describe_fields(param_spec_name, synth_name)
    spec = resolve_param_spec(ParamSpecName(param_spec_name))
    expected_names = [field.name for field, _ in spec.encoded_slices()]
    if (
        metadata.param_spec_name != param_spec_name
        or metadata.synth_name != synth_name
        or metadata.descriptions_sha256 != _description_digest(expected_descriptions)
    ):
        raise ValueError("parameter name dataset does not match the current spec")
    required = {
        PARAM_FIELD_INDEX_FIELD,
        PARAM_FIELD_NAME_FIELD,
        PARAM_DESCRIPTION_FIELD,
        PARAM_NAME_EMBEDDING_FIELD,
    }
    if not required.issubset(dataset.schema.names):
        raise ValueError("parameter name dataset lacks required fields")
    from synth_setter.pipeline.data.add_embeddings import embedding_field_metadata
    from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig

    identity = embedding_field_metadata(
        PARAM_NAME_EMBEDDING_REGISTRY_KEY,
        AddEmbeddingsConfig(
            lance_uri=str(path),
            embeddings=(PARAM_NAME_EMBEDDING_REGISTRY_KEY,),
            build_index=False,
            param_name_embedding_dimension=metadata.dimension,
        ),
    )
    field_metadata = dataset.schema.field(PARAM_NAME_EMBEDDING_FIELD).metadata or {}
    if any(field_metadata.get(key) != value for key, value in identity.items()):
        raise ValueError("parameter name embedding provenance does not match the registry")
    table = dataset.to_table(columns=sorted(required)).sort_by(PARAM_FIELD_INDEX_FIELD)
    if table[PARAM_FIELD_INDEX_FIELD].to_pylist() != list(range(len(expected_names))):
        raise ValueError("parameter name dataset field indices are not complete")
    if table[PARAM_FIELD_NAME_FIELD].to_pylist() != expected_names:
        raise ValueError("parameter name dataset field names do not match the current spec")
    if table[PARAM_DESCRIPTION_FIELD].to_pylist() != expected_descriptions:
        raise ValueError("parameter name dataset descriptions do not match the current spec")
    vectors = table[PARAM_NAME_EMBEDDING_FIELD].combine_chunks()
    if (
        not pa.types.is_fixed_size_list(vectors.type)
        or vectors.type.list_size != metadata.dimension
    ):
        raise ValueError("parameter name embedding column has the wrong width")
    embeddings = np.asarray(
        vectors.values.to_numpy().reshape(len(expected_names), metadata.dimension),
        dtype=np.float32,
    )
    _validate_vectors(embeddings, len(expected_names), metadata.dimension)
    _validate_unit_norm(embeddings)
    return embeddings, metadata


def prepare_param_name_embeddings(
    work_dir: Path,
    param_spec_name: str,
    synth_name: str,
    *,
    dimension: int,
    device: str = "cpu",
) -> Path:
    """Build a minimal field dataset and augment it through ``add_embeddings``.

    :param work_dir: Finalizer scratch directory receiving ``params.lance``.
    :param param_spec_name: Registered logical field layout.
    :param synth_name: Synth identity included in canonical descriptions.
    :param dimension: Selected Matryoshka embedding width.
    :param device: Torch device used by the shared embedding API.
    :returns: Validated local ``params.lance`` path.
    """
    from synth_setter.pipeline.data.add_embeddings import add_embeddings
    from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig

    path = work_dir / PARAM_NAME_DATASET
    write_param_name_dataset(path, param_spec_name, synth_name, dimension=dimension)
    add_embeddings(
        AddEmbeddingsConfig(
            lance_uri=str(path),
            embeddings=(PARAM_NAME_EMBEDDING_REGISTRY_KEY,),
            device=device,
            build_index=False,
            param_name_embedding_dimension=dimension,
        )
    )
    load_param_name_embeddings(path, param_spec_name, synth_name)
    return path
