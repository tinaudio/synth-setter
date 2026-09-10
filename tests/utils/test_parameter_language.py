"""Lightning lifecycle initialization makes parameter transforms self-contained."""

import shutil
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
import torch
from lightning import LightningModule, Trainer

from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.models.components.language_projection import LanguageParameterProjection
from synth_setter.pipeline.data.add_embeddings import embedding_field_metadata
from synth_setter.pipeline.data.param_language import (
    PARAM_NAME_EMBEDDING_FIELD,
    PARAM_NAME_EMBEDDING_REGISTRY_KEY,
    describe_fields,
    write_param_name_dataset,
)
from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig
from synth_setter.utils.parameter_language import ParameterLanguageInitializer


@pytest.mark.parametrize(
    "hook", ["on_fit_start", "on_validation_start", "on_test_start", "on_predict_start"]
)
def test_lifecycle_initialization_supplies_vectors_before_forward(
    hook: str, tmp_path: Path
) -> None:
    """Each supported lifecycle entry makes numeric projection independent of its source file.

    :param hook: Lightning boundary under test.
    :param tmp_path: Isolated source artifact directory.
    """
    count = len(describe_fields("surge_4", "surge_4"))
    path = tmp_path / "params.lance"
    vectors = np.full((count, 128), 1 / np.sqrt(128), dtype=np.float32)
    write_param_name_dataset(path, "surge_4", "surge_4")
    metadata = embedding_field_metadata(
        PARAM_NAME_EMBEDDING_REGISTRY_KEY,
        AddEmbeddingsConfig(
            lance_uri=str(path),
            embeddings=(PARAM_NAME_EMBEDDING_REGISTRY_KEY,),
            build_index=False,
        ),
    )
    field = pa.field(PARAM_NAME_EMBEDDING_FIELD, pa.list_(pa.float32(), 128), metadata=metadata)
    table = (
        lance.dataset(path)
        .to_table()
        .append_column(
            field,
            pa.FixedSizeListArray.from_arrays(pa.array(vectors.reshape(-1)), 128),
        )
    )
    lance.write_dataset(table, path, mode="overwrite")
    projection = LanguageParameterProjection(16, "surge_4", "surge_4", embedding_path=str(path))
    model = LightningModule()
    model.add_module("projection", projection)
    trainer = Trainer(logger=False, enable_checkpointing=False, accelerator="cpu")
    callback = ParameterLanguageInitializer()
    getattr(callback, hook)(trainer, model)
    shutil.rmtree(path)
    tokens = projection.param_to_token(torch.zeros(2, param_specs["surge_4"].encoded_width))
    assert torch.isfinite(tokens).all()
    assert tokens.shape == (2, count, 16)
