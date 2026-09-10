"""Canonical field descriptions and EmbeddingGemma Lance artifacts stay aligned."""

from pathlib import Path

import numpy as np
import pytest
from huggingface_hub import get_token

from synth_setter.pipeline.data.param_language import (
    describe_fields,
    load_param_name_embeddings,
    matryoshka_vectors,
    prepare_param_name_embeddings,
    write_param_name_dataset,
)


def test_descriptions_capture_type_range_categories_and_encoded_span() -> None:
    """Canonical text retains the field semantics selected for the experiment."""
    descriptions = describe_fields("surge_4", "surge_xt")
    joined = "\n".join(descriptions)
    assert '"type": "ContinuousParameter"' in joined
    assert '"min":' in joined
    assert '"max":' in joined
    assert '"values":' in "\n".join(describe_fields("faust_bubble", "faust_bubble"))
    assert '"encoded_span":' in joined
    assert '"synth": "surge_xt"' in joined


def test_descriptions_are_deterministic_in_logical_field_order() -> None:
    """Repeated extraction returns byte-identical logical field descriptions."""
    assert describe_fields("surge_4", "surge_4") == describe_fields("surge_4", "surge_4")


@pytest.mark.parametrize("dimension", [128, 256, 512, 768])
def test_matryoshka_vectors_supported_width_has_unit_rows(dimension: int) -> None:
    """Every trained prefix width is selected and normalized.

    :param dimension: Supported output width.
    """
    vectors = np.arange(2 * 768, dtype=np.float32).reshape(2, 768) + 1
    selected = matryoshka_vectors(vectors, dimension)
    assert selected.shape == (2, dimension)
    np.testing.assert_allclose(np.linalg.norm(selected, axis=1), 1, atol=1e-6)


def test_matryoshka_vectors_unsupported_width_rejected() -> None:
    """Widths outside the model contract fail before encoding or publication."""
    with pytest.raises(ValueError, match="unsupported"):
        matryoshka_vectors(np.ones((2, 768), dtype=np.float32), 64)


def test_load_param_name_embeddings_wrong_synth_rejected(tmp_path: Path) -> None:
    """A same-shape dataset cannot silently cross synth identities.

    :param tmp_path: Isolated field dataset root.
    """
    path = tmp_path / "params.lance"
    write_param_name_dataset(path, "surge_4", "surge_4")
    with pytest.raises(ValueError, match="current spec"):
        load_param_name_embeddings(path, "surge_4", "surge_xt")


@pytest.mark.slow
@pytest.mark.parametrize("dimension", [128, 768])
def test_real_add_embeddings_api_publishes_consumable_field_vectors(
    dimension: int, tmp_path: Path
) -> None:
    """The real registry encoder writes normalized vectors loadable by consumers.

    :param dimension: Native or default Matryoshka width.
    :param tmp_path: Isolated field dataset root.
    """
    if get_token() is None:
        pytest.skip("requires accepted embedding-model license and HF_TOKEN authentication")
    work_dir = tmp_path / str(dimension)
    work_dir.mkdir()
    path = prepare_param_name_embeddings(
        work_dir, "surge_4", "surge_4", dimension=dimension, device="cpu"
    )
    embeddings, metadata = load_param_name_embeddings(path, "surge_4", "surge_4")
    assert embeddings.shape == (6, dimension)
    assert metadata.dimension == dimension
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-6)
    assert not np.allclose(embeddings[0], embeddings[1])
