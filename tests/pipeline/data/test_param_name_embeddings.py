"""Parameter descriptions use the shared Lance embedding augmentation path."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from synth_setter.pipeline.data import add_embeddings as add_embeddings_module
from synth_setter.pipeline.data.add_embeddings import (
    EMBEDDING_REGISTRY,
    embedding_field_metadata,
)
from synth_setter.pipeline.data.param_language import (
    PARAM_DESCRIPTION_FIELD,
    PARAM_FIELD_INDEX_FIELD,
    PARAM_FIELD_NAME_FIELD,
    PARAM_NAME_EMBEDDING_FIELD,
    PARAM_NAME_EMBEDDING_REGISTRY_KEY,
    describe_fields,
    load_param_name_embeddings,
    write_param_name_dataset,
)
from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig


def test_write_param_name_dataset_contains_only_field_identity_and_description(
    tmp_path: Path,
) -> None:
    """The pre-embedding dataset omits per-example parameter values and audio.

    :param tmp_path: Isolated Lance dataset root.
    """
    import lance

    path = tmp_path / "params.lance"
    write_param_name_dataset(path, "surge_4", "surge_4")

    dataset = lance.dataset(path)
    assert dataset.schema.names == [
        PARAM_FIELD_INDEX_FIELD,
        PARAM_FIELD_NAME_FIELD,
        PARAM_DESCRIPTION_FIELD,
    ]
    assert dataset.count_rows() == len(describe_fields("surge_4", "surge_4"))


def test_add_embeddings_rejects_dimension_different_from_dataset_provenance(
    tmp_path: Path,
) -> None:
    """CLI width cannot silently disagree with the field dataset contract.

    :param tmp_path: Isolated Lance dataset root.
    """
    path = tmp_path / "params.lance"
    write_param_name_dataset(path, "surge_4", "surge_4", dimension=128)
    with pytest.raises(ValueError, match="dataset provenance"):
        add_embeddings_module.add_embeddings(
            AddEmbeddingsConfig(
                lance_uri=str(path),
                embeddings=(PARAM_NAME_EMBEDDING_REGISTRY_KEY,),
                build_index=False,
                param_name_embedding_dimension=768,
            )
        )


def test_add_embeddings_text_dataset_does_not_require_audio_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public augmentation API embeds a text-only Lance dataset without sample-rate metadata.

    :param tmp_path: Isolated Lance dataset root.
    :param monkeypatch: Replaces only the external pretrained encoder.
    """
    path = tmp_path / "params.lance"
    write_param_name_dataset(path, "surge_4", "surge_4")
    selected = EMBEDDING_REGISTRY["param_name"]

    def load_encoder(checkpoint: str, config: AddEmbeddingsConfig):
        del checkpoint, config

        def encode(descriptions: np.ndarray) -> np.ndarray:
            rows = np.arange(len(descriptions) * 128, dtype=np.float32).reshape(-1, 128) + 1
            return rows / np.linalg.norm(rows, axis=1, keepdims=True)

        return encode

    monkeypatch.setitem(
        EMBEDDING_REGISTRY, "param_name", replace(selected, load_encoder=load_encoder)
    )
    add_embeddings_module.add_embeddings(
        AddEmbeddingsConfig(
            lance_uri=str(path),
            embeddings=("param_name",),
            build_index=False,
            device="cpu",
            param_name_embedding_dimension=128,
        )
    )

    embeddings, metadata = load_param_name_embeddings(path, "surge_4", "surge_4")
    assert embeddings.shape == (len(describe_fields("surge_4", "surge_4")), 128)
    assert metadata.dimension == 128
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-6)
    import lance

    assert lance.dataset(path).schema.names == [
        PARAM_FIELD_INDEX_FIELD,
        PARAM_FIELD_NAME_FIELD,
        PARAM_DESCRIPTION_FIELD,
        PARAM_NAME_EMBEDDING_FIELD,
    ]


def test_materialize_splits_hydrates_parameter_name_lance_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Root hydration copies the optional field dataset through its real consumer.

    :param tmp_path: Isolated source and destination roots.
    :param monkeypatch: Replaces only the external pretrained encoder.
    """
    from synth_setter.pipeline.data.lance_materialize import materialize_splits
    from synth_setter.pipeline.data.param_language import PARAM_NAME_COMPLETE

    source = tmp_path / "source"
    source.mkdir()
    path = source / "params.lance"
    write_param_name_dataset(path, "surge_4", "surge_4")
    selected = EMBEDDING_REGISTRY[PARAM_NAME_EMBEDDING_REGISTRY_KEY]

    def load_encoder(checkpoint: str, config: AddEmbeddingsConfig):
        del checkpoint, config
        return lambda descriptions: np.eye(len(descriptions), 128, dtype=np.float32)

    monkeypatch.setitem(
        EMBEDDING_REGISTRY,
        PARAM_NAME_EMBEDDING_REGISTRY_KEY,
        replace(selected, load_encoder=load_encoder),
    )
    add_embeddings_module.add_embeddings(
        AddEmbeddingsConfig(
            lance_uri=str(path),
            embeddings=(PARAM_NAME_EMBEDDING_REGISTRY_KEY,),
            build_index=False,
        )
    )
    (source / PARAM_NAME_COMPLETE).touch()
    (source / "dataset.complete").touch()

    destination = tmp_path / "destination"
    materialize_splits(
        source.as_uri(),
        destination,
        txids={},
        projection={},
        row_limit=None,
        shard_suffix=".lance",
    )
    restored, _ = load_param_name_embeddings(
        destination / "params.lance", "surge_4", "surge_4"
    )
    assert restored.shape == (6, 128)


def test_load_param_name_embeddings_rejects_unpinned_encoder_identity(tmp_path: Path) -> None:
    """Normalized vectors cannot bypass registry provenance validation.

    :param tmp_path: Isolated Lance dataset root.
    """
    import lance
    import pyarrow as pa

    path = tmp_path / "params.lance"
    write_param_name_dataset(path, "surge_4", "surge_4")
    rows = len(describe_fields("surge_4", "surge_4"))
    vectors = np.eye(rows, 128, dtype=np.float32)
    field = pa.field(PARAM_NAME_EMBEDDING_FIELD, pa.list_(pa.float32(), 128))
    table = lance.dataset(path).to_table().append_column(
        field, pa.FixedSizeListArray.from_arrays(pa.array(vectors.reshape(-1)), 128)
    )
    lance.write_dataset(table, path, mode="overwrite")
    with pytest.raises(ValueError, match="provenance"):
        load_param_name_embeddings(path, "surge_4", "surge_4")


def test_load_param_name_embeddings_orders_rows_by_field_index(tmp_path: Path) -> None:
    """Consumer order follows explicit indices rather than physical Lance row order.

    :param tmp_path: Isolated Lance dataset root.
    """
    import lance
    import pyarrow as pa

    path = tmp_path / "params.lance"
    write_param_name_dataset(path, "surge_4", "surge_4")
    dataset = lance.dataset(path)
    table = dataset.to_table().take(pa.array(list(reversed(range(dataset.count_rows())))))
    embeddings = np.eye(dataset.count_rows(), 128, dtype=np.float32)[::-1]
    metadata = embedding_field_metadata(
        PARAM_NAME_EMBEDDING_REGISTRY_KEY,
        AddEmbeddingsConfig(
            lance_uri=str(path),
            embeddings=(PARAM_NAME_EMBEDDING_REGISTRY_KEY,),
            build_index=False,
        ),
    )
    field = pa.field(
        PARAM_NAME_EMBEDDING_FIELD, pa.list_(pa.float32(), 128), metadata=metadata
    )
    table = table.append_column(
        field, pa.FixedSizeListArray.from_arrays(pa.array(embeddings.reshape(-1)), 128)
    )
    lance.write_dataset(table, path, mode="overwrite")

    restored, _ = load_param_name_embeddings(path, "surge_4", "surge_4")
    np.testing.assert_array_equal(restored, np.eye(dataset.count_rows(), 128, dtype=np.float32))
