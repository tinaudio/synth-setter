"""Per-field language metadata preserves the numeric parameter contract."""

import json
from pathlib import Path

import numpy as np
import pytest
from huggingface_hub import get_token

from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.data import param_language
from synth_setter.pipeline.data.param_language import (
    describe_fields,
    encode_param_language,
    load_param_language,
    matryoshka_vectors,
    prepare_param_language,
    save_param_language,
)


def test_describe_fields_surge_matches_encoded_field_order() -> None:
    """Ensure Surge descriptions follow encoded field order."""
    spec = resolve_param_spec(ParamSpecName("surge_simple"))
    descriptions = describe_fields("surge_simple", "surge_xt")
    assert [json.loads(text)["name"] for text in descriptions] == [
        field.name for field, _ in spec.encoded_slices()
    ]


def test_describe_fields_uses_persisted_note_timing_semantics() -> None:
    """Timing metadata changes descriptions without changing the registry identity."""
    legacy = describe_fields("surge_4", "surge_4")
    current = describe_fields("surge_4", "surge_4", "onset_duration")

    assert json.loads(legacy[-1])["type"] == "LegacyEndpointNoteDurationParameter"
    assert json.loads(current[-1])["type"] == "NoteDurationParameter"


def test_matryoshka_truncation_renormalizes_prefix() -> None:
    """Ensure truncation renormalizes each embedding prefix."""
    full = np.ones((2, 768), dtype=np.float32)
    full[:, :128] = 2.0
    actual = matryoshka_vectors(full, 128)
    assert actual.shape == (2, 128)
    np.testing.assert_allclose(actual, 1 / np.sqrt(128), rtol=1e-6)


@pytest.mark.slow
def test_real_encoder_artifact_load_preserves_supported_dimensions(tmp_path: Path) -> None:
    """Ensure a real encoded artifact preserves supported dimensions.

    :param tmp_path: Temporary directory for the artifact.
    """
    if get_token() is None:
        pytest.skip("requires google/embeddinggemma-300m license/HF_TOKEN")
    full = encode_param_language("surge_simple", "surge_xt", device="cpu")
    descriptions = describe_fields("surge_simple", "surge_xt")
    assert full.shape == (len(descriptions), 768)
    assert not np.allclose(full[0], full[1])
    path = tmp_path / "language.npz"
    save_param_language(path, matryoshka_vectors(full, 128), "surge_simple", "surge_xt")
    restored, _ = load_param_language(path, "surge_simple", "surge_xt")
    np.testing.assert_allclose(np.linalg.norm(restored, axis=1), 1, atol=1e-6)


def test_artifact_round_trip_preserves_vectors(tmp_path: Path) -> None:
    """Ensure an artifact round trip preserves vectors.

    :param tmp_path: Temporary directory for the artifact.
    """
    descriptions = describe_fields("surge_simple", "surge_xt")
    embeddings = matryoshka_vectors(
        np.random.default_rng(7).normal(size=(len(descriptions), 768)).astype(np.float32), 768
    )
    path = tmp_path / "language.npz"
    save_param_language(path, embeddings, "surge_simple", "surge_xt")
    actual, metadata = load_param_language(path, "surge_simple", "surge_xt")
    np.testing.assert_array_equal(actual, embeddings)
    assert metadata.descriptions == descriptions


def test_artifact_wrong_spec_rejected(tmp_path: Path) -> None:
    """Ensure loading an artifact with the wrong spec fails.

    :param tmp_path: Temporary directory for the artifact.
    """
    descriptions = describe_fields("surge_simple", "surge_xt")
    path = tmp_path / "language.npz"
    embeddings = np.full((len(descriptions), 768), 1 / np.sqrt(768), dtype=np.float32)
    save_param_language(path, embeddings, "surge_simple", "surge_xt")
    with pytest.raises(ValueError, match="spec"):
        load_param_language(path, "surge_4", "surge_xt")


def test_artifact_non_unit_vectors_rejected_on_save(tmp_path: Path) -> None:
    """An artifact cannot publish vectors with non-unit row norms.

    :param tmp_path: Isolated artifact directory.
    """
    count = len(describe_fields("surge_simple", "surge_xt"))
    embeddings = np.ones((count, 768), dtype=np.float32)

    with pytest.raises(ValueError, match="unit norm"):
        save_param_language(tmp_path / "language.npz", embeddings, "surge_simple", "surge_xt")


def test_artifact_nonfinite_vectors_rejected(tmp_path: Path) -> None:
    """Ensure saving nonfinite artifact vectors fails.

    :param tmp_path: Temporary directory for the artifact.
    """
    descriptions = describe_fields("surge_simple", "surge_xt")
    embeddings = np.ones((len(descriptions), 768), dtype=np.float32)
    embeddings[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        save_param_language(tmp_path / "language.npz", embeddings, "surge_simple", "surge_xt")


def test_prepare_cached_full_vectors_selects_requested_width(tmp_path: Path) -> None:
    """Cached native vectors supply normalized metadata without running an encoder.

    :param tmp_path: Finalizer work directory with a native-width cache.
    """
    count = len(describe_fields("surge_4", "surge_4"))
    full = matryoshka_vectors(
        np.random.default_rng(4).normal(size=(count, 768)).astype(np.float32), 768
    )
    save_param_language(tmp_path / "param_language_full.npz", full, "surge_4", "surge_4")
    path = prepare_param_language(tmp_path, "surge_4", "surge_4", dimension=128)
    vectors, _ = load_param_language(path, "surge_4", "surge_4")
    np.testing.assert_allclose(vectors, matryoshka_vectors(full, 128))


def test_artifact_modified_vector_checksum_rejected(tmp_path: Path) -> None:
    """A changed tensor cannot retain its original artifact identity.

    :param tmp_path: Isolated artifact directory.
    """
    count = len(describe_fields("surge_4", "surge_4"))
    path = tmp_path / "language.npz"
    vectors = np.full((count, 128), 1 / np.sqrt(128), dtype=np.float32)
    save_param_language(path, vectors, "surge_4", "surge_4")
    with np.load(path) as archive:
        metadata = archive["metadata"]
    vectors[0, 0] *= -1
    np.savez(path, embeddings=vectors, metadata=metadata)
    with pytest.raises(ValueError, match="checksum"):
        load_param_language(path, "surge_4", "surge_4")


@pytest.mark.parametrize("dimension", [0, 127, 769])
def test_matryoshka_unsupported_width_rejected(dimension: int) -> None:
    """Only trained Matryoshka widths may be requested.

    :param dimension: Unsupported embedding width.
    """
    with pytest.raises(ValueError, match="dimension"):
        matryoshka_vectors(np.ones((2, 768), dtype=np.float32), dimension)


def test_matryoshka_zero_prefix_rejected() -> None:
    """Zero-length directions cannot produce normalized field vectors."""
    with pytest.raises(ValueError, match="nonzero"):
        matryoshka_vectors(np.zeros((2, 768), dtype=np.float32), 128)


def test_matryoshka_nonmatrix_rejected() -> None:
    """A single unbatched vector is not a field table."""
    with pytest.raises(ValueError, match="matrix"):
        matryoshka_vectors(np.ones(768, dtype=np.float32), 128)


def test_artifact_non_unit_vectors_rejected_on_load(tmp_path: Path) -> None:
    """An artifact with changed row magnitude cannot be consumed.

    :param tmp_path: Isolated artifact directory.
    """
    count = len(describe_fields("surge_4", "surge_4"))
    path = tmp_path / "language.npz"
    vectors = np.full((count, 128), 1 / np.sqrt(128), dtype=np.float32)
    save_param_language(path, vectors, "surge_4", "surge_4")
    with np.load(path) as archive:
        metadata = archive["metadata"]
    vectors[0] *= 2
    np.savez(path, embeddings=vectors, metadata=metadata)

    with pytest.raises(ValueError, match="unit norm"):
        load_param_language(path, "surge_4", "surge_4")


def test_save_failure_preserves_existing_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed temporary write leaves the complete destination unchanged.

    :param tmp_path: Isolated artifact directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    count = len(describe_fields("surge_4", "surge_4"))
    path = tmp_path / "language.npz"
    vectors = np.full((count, 128), 1 / np.sqrt(128), dtype=np.float32)
    save_param_language(path, vectors, "surge_4", "surge_4")
    original = path.read_bytes()

    def fail_save(*args, **kwargs) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(param_language.np, "savez", fail_save)
        with pytest.raises(OSError, match="disk full"):
            save_param_language(path, vectors, "surge_4", "surge_4")

    assert path.read_bytes() == original
    restored, _ = load_param_language(path, "surge_4", "surge_4")
    np.testing.assert_array_equal(restored, vectors)


def test_prepare_truncated_cache_regenerates_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truncated native cache is regenerated and remains loadable.

    :param tmp_path: Isolated artifact directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    cache_path = tmp_path / "param_language_full.npz"
    cache_path.write_bytes(b"truncated")
    count = len(describe_fields("surge_4", "surge_4"))
    full = np.full((count, 768), 1 / np.sqrt(768), dtype=np.float32)
    monkeypatch.setattr(param_language, "encode_param_language", lambda *args, **kwargs: full)

    output = prepare_param_language(tmp_path, "surge_4", "surge_4", dimension=128)

    cached, _ = load_param_language(cache_path, "surge_4", "surge_4")
    selected, _ = load_param_language(output, "surge_4", "surge_4")
    np.testing.assert_array_equal(cached, full)
    np.testing.assert_allclose(selected, matryoshka_vectors(full, 128))


def test_prepare_non_native_cache_regenerates_full_width(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid selected-width artifact cannot stand in for the native cache.

    :param tmp_path: Isolated artifact directory.
    :param monkeypatch: Replaces only expensive external encoding.
    """
    count = len(describe_fields("surge_4", "surge_4"))
    cache = tmp_path / "param_language_full.npz"
    selected = np.full((count, 128), 1 / np.sqrt(128), dtype=np.float32)
    save_param_language(cache, selected, "surge_4", "surge_4")
    full = np.full((count, 768), 1 / np.sqrt(768), dtype=np.float32)
    monkeypatch.setattr(param_language, "encode_param_language", lambda *args, **kwargs: full)
    prepare_param_language(tmp_path, "surge_4", "surge_4", dimension=128)
    _, metadata = load_param_language(cache, "surge_4", "surge_4")
    assert metadata.dimension == 768


def test_artifact_wrong_field_count_rejected(tmp_path: Path) -> None:
    """Partial field coverage cannot be published as a complete spec.

    :param tmp_path: Isolated artifact directory.
    """
    with pytest.raises(ValueError, match="aligned"):
        save_param_language(
            tmp_path / "language.npz", np.ones((1, 128), dtype=np.float32), "surge_4", "surge_4"
        )
