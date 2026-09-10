"""Offline language descriptions in logical parameter-token order."""

import hashlib
import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
import structlog
from pydantic import BaseModel, ConfigDict

from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.model_cache import retry_external_io
from synth_setter.param_spec_name import ParamSpecName

PARAM_LANGUAGE_FILENAME = "param_language.npz"
EMBEDDING_MODEL = "google/embeddinggemma-300m"
EMBEDDING_REVISION = "57c266a740f537b4dc058e1b0cda161fd15afa75"

logger = structlog.get_logger(__name__)


def describe_fields(param_spec_name: str, synth_name: str) -> list[str]:
    """Describe spec metadata without interpreting renderer-native ranges as physical units.

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
    """Truncate EmbeddingGemma vectors and renormalize their retained coordinates.

    :param embeddings: Native-width float32 vectors shaped ``(fields, 768)``.
    :param dimension: Supported Matryoshka width.
    :returns: Unit-normalized float32 matrix shaped ``(fields, dimension)``.
    :raises ValueError: Width, values, or prefix norms are invalid.
    """
    if dimension not in (128, 256, 512, 768):
        raise ValueError("unsupported EmbeddingGemma Matryoshka dimension")
    if embeddings.ndim != 2:
        raise ValueError("parameter language embeddings must be a matrix")
    _validate_vectors(embeddings, embeddings.shape[0], 768)
    prefix = embeddings[:, :dimension].copy()
    norms = np.linalg.norm(prefix.astype(np.float64), axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Matryoshka prefixes must have nonzero norms")
    return (prefix / norms).astype(np.float32)


def encode_param_language(
    param_spec_name: str, synth_name: str, *, device: str = "cpu", batch_size: int = 16
) -> np.ndarray:
    """Encode static descriptions with the pinned EmbeddingGemma document pipeline.

    :param param_spec_name: Registered parameter specification.
    :param synth_name: Dataset synth identity.
    :param device: Torch inference device.
    :param batch_size: Number of descriptions per inference batch.
    :returns: Native-width float32 embeddings shaped ``(fields, 768)``.
    """
    from httpx import TransportError
    from sentence_transformers import SentenceTransformer

    @retry_external_io(retry_exceptions=(OSError, TransportError))
    def load_model() -> SentenceTransformer:
        """Load the pinned encoder with bounded retries for transport failures.

        :returns: Encoder loaded from the pinned checkpoint.
        """
        return SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device=device)

    model = load_model()
    model.eval()
    model.requires_grad_(False)
    descriptions = describe_fields(param_spec_name, synth_name)
    embeddings = model.encode_document(
        descriptions, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
    )
    return matryoshka_vectors(np.asarray(embeddings, dtype=np.float32), 768)


class ParamLanguageMetadata(BaseModel):
    """Identity and extraction contract for a dataset-level field embedding table.

    .. attribute :: model_config

        Strict, frozen JSON boundary with unknown keys rejected.

    .. attribute :: version

        Artifact schema version.

    .. attribute :: model

        Matryoshka-trained encoder identifier.

    .. attribute :: revision

        Immutable encoder and pooling configuration revision.

    .. attribute :: extraction

        Document prompt, checkpoint pooling, prefix selection, and normalization contract.

    .. attribute :: dimension

        Retained Matryoshka width.

    .. attribute :: param_spec_name

        Registered numeric parameter specification.

    .. attribute :: synth_name

        Dataset synth identity.

    .. attribute :: descriptions

        Canonical metadata in logical-field order.

    .. attribute :: descriptions_sha256

        Digest of ordered descriptions.

    .. attribute :: embeddings_sha256

        Digest of canonical float32 tensor bytes.
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
    descriptions: list[str]
    descriptions_sha256: str
    embeddings_sha256: str


def _description_digest(descriptions: list[str]) -> str:
    """Hash ordered descriptions without ambiguous string concatenation.

    :param descriptions: Canonical per-field descriptions.
    :returns: SHA256 of their JSON representation.
    """
    return hashlib.sha256(json.dumps(descriptions, ensure_ascii=False).encode()).hexdigest()


def _embedding_digest(embeddings: np.ndarray) -> str:
    """Hash the canonical little-endian float32 representation.

    :param embeddings: Field-major matrix.
    :returns: SHA256 of contiguous tensor bytes.
    """
    return hashlib.sha256(embeddings.astype("<f4").tobytes()).hexdigest()


def _validate_vectors(embeddings: np.ndarray, count: int, dimension: int) -> None:
    """Reject malformed or nonfinite field embeddings before publication or consumption.

    :param embeddings: Field-major matrix.
    :param count: Expected logical field count.
    :param dimension: Expected embedding width.
    :raises ValueError: Shape, dtype, or values violate the artifact contract.
    """
    if embeddings.shape != (count, dimension) or embeddings.dtype != np.float32:
        raise ValueError("parameter language embeddings require aligned float32 field vectors")
    if not np.isfinite(embeddings).all():
        raise ValueError("parameter language embeddings must be finite")


def _validate_unit_norm(embeddings: np.ndarray) -> None:
    """Reject field embeddings whose rows are not unit normalized.

    :param embeddings: Validated field-major matrix.
    :raises ValueError: Any row does not have unit norm within artifact tolerance.
    """
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError("parameter language embeddings must have unit norm")


def save_param_language(
    path: Path, embeddings: np.ndarray, param_spec_name: str, synth_name: str
) -> None:
    """Write a pickle-free embedding table with its spec and encoder identity.

    :param path: Destination NPZ path.
    :param embeddings: Float32 field-major matrix from the pinned encoder.
    :param param_spec_name: Registered parameter specification.
    :param synth_name: Dataset synth identity.
    :raises ValueError: The embedding matrix is malformed.
    """
    descriptions = describe_fields(param_spec_name, synth_name)
    if embeddings.ndim != 2:
        raise ValueError("parameter language embeddings must be a matrix")
    _validate_vectors(embeddings, len(descriptions), embeddings.shape[1])
    _validate_unit_norm(embeddings)
    metadata = ParamLanguageMetadata(
        dimension=embeddings.shape[1],
        param_spec_name=param_spec_name,
        synth_name=synth_name,
        descriptions=descriptions,
        descriptions_sha256=_description_digest(descriptions),
        embeddings_sha256=_embedding_digest(embeddings),
    )
    with TemporaryDirectory(dir=path.parent) as temporary_directory:
        temporary_path = Path(temporary_directory) / PARAM_LANGUAGE_FILENAME
        np.savez(
            temporary_path,
            embeddings=embeddings,
            metadata=np.array(metadata.model_dump_json()),
        )
        temporary_path.replace(path)


def load_param_language(
    path: Path, param_spec_name: str, synth_name: str
) -> tuple[np.ndarray, ParamLanguageMetadata]:
    """Load an artifact only when it agrees with the current spec and extraction contract.

    :param path: Pickle-free NPZ artifact.
    :param param_spec_name: Expected registered parameter specification.
    :param synth_name: Expected synth identity.
    :returns: Float32 field matrix and validated provenance.
    :raises ValueError: Metadata, vectors, or their fingerprints do not match.
    """
    with np.load(path, allow_pickle=False) as archive:
        metadata = ParamLanguageMetadata.model_validate_json(str(archive["metadata"].item()))
        embeddings = archive["embeddings"]
    expected = describe_fields(param_spec_name, synth_name)
    if (
        metadata.param_spec_name != param_spec_name
        or metadata.synth_name != synth_name
        or metadata.descriptions != expected
        or metadata.descriptions_sha256 != _description_digest(expected)
    ):
        raise ValueError("parameter language artifact does not match the current spec")
    _validate_vectors(embeddings, len(expected), metadata.dimension)
    _validate_unit_norm(embeddings)
    if metadata.embeddings_sha256 != _embedding_digest(embeddings):
        raise ValueError("parameter language embedding checksum mismatch")
    return embeddings, metadata


def prepare_param_language(
    work_dir: Path, param_spec_name: str, synth_name: str, *, dimension: int
) -> Path:
    """Cache full-width embeddings locally and stage the selected width for finalization.

    :param work_dir: Existing finalizer scratch directory.
    :param param_spec_name: Registered parameter specification.
    :param synth_name: Dataset synth identity.
    :param dimension: Supported output width.
    :returns: Validated, staged dataset-level NPZ path.
    :raises ValueError: Requested width or freshly encoded vectors are invalid.
    """
    cache_path = work_dir / "param_language_full.npz"
    embeddings = None
    if cache_path.exists():
        try:
            embeddings, metadata = load_param_language(cache_path, param_spec_name, synth_name)
            if metadata.dimension != 768:
                raise ValueError("parameter language full cache requires native width")
        except (OSError, ValueError, EOFError, zipfile.BadZipFile, KeyError):
            embeddings = None
            logger.warning("param_language_cache_invalid", path=str(cache_path))
    if embeddings is None:
        embeddings = encode_param_language(param_spec_name, synth_name)
        save_param_language(cache_path, embeddings, param_spec_name, synth_name)
    output = work_dir / PARAM_LANGUAGE_FILENAME
    selected = matryoshka_vectors(embeddings, dimension)
    save_param_language(output, selected, param_spec_name, synth_name)
    return output
