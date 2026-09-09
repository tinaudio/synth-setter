"""Per-field language metadata preserves the numeric parameter contract."""

import json
from pathlib import Path

import numpy as np
import pytest

from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.data.param_language import (
    describe_fields,
    encode_param_language,
    load_param_language,
    matryoshka_vectors,
    save_param_language,
)


def test_describe_fields_surge_matches_encoded_field_order() -> None:
    """Ensure Surge descriptions follow encoded field order."""
    spec = resolve_param_spec(ParamSpecName("surge_simple"))
    descriptions = describe_fields("surge_simple", "surge_xt")
    assert [json.loads(text)["name"] for text in descriptions] == [
        field.name for field, _ in spec.encoded_slices()
    ]


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
    embeddings = np.random.default_rng(7).normal(size=(len(descriptions), 768)).astype(np.float32)
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
    save_param_language(
        path, np.ones((len(descriptions), 768), dtype=np.float32), "surge_simple", "surge_xt"
    )
    with pytest.raises(ValueError, match="spec"):
        load_param_language(path, "surge_4", "surge_xt")


def test_artifact_nonfinite_vectors_rejected(tmp_path: Path) -> None:
    """Ensure saving nonfinite artifact vectors fails.

    :param tmp_path: Temporary directory for the artifact.
    """
    descriptions = describe_fields("surge_simple", "surge_xt")
    embeddings = np.ones((len(descriptions), 768), dtype=np.float32)
    embeddings[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        save_param_language(tmp_path / "language.npz", embeddings, "surge_simple", "surge_xt")
