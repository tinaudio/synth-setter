"""Boundary tests opening the published RIR corpora exactly as their eval configs do."""

from __future__ import annotations

import hydra
import lance
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module

from synth_setter.pipeline import r2_io

_REQUIRED_COLUMNS = {"source_bytes", "source_path", "audio_decodable"}
_CORPORA = ["mit_ir_survey", "echothief", "ashir", "openair", "thkoeln_omni", "arni"]


def _rir_datamodule_cfg(corpus: str):  # noqa: ANN202 — Hydra DictConfig
    """Compose the production eval config for one RIR corpus with a one-row budget.

    :param corpus: Corpus config under ``datamodule/third_party/rir``.
    :returns: The composed datamodule node.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            overrides=[
                "experiment=pyfdn/eval_flow_rir",
                f"datamodule=third_party/rir/{corpus}",
                "trainer=cpu",
                "ckpt_path=/tmp/none.ckpt",
                "datamodule.row_limit=1",
                "datamodule.batch_size=1",
                "datamodule.num_workers=0",
                "datamodule.use_saved_mean_and_variance=false",
                "datamodule.mel_stats_uri=null",
                "datamodule.mel_stats_sha256=null",
                "paths.output_dir=/tmp/synth-setter-test",
            ],
        )
    return cfg.datamodule


@pytest.mark.slow
@pytest.mark.integration_r2
@pytest.mark.r2
@pytest.mark.parametrize("corpus", _CORPORA)
def test_published_rir_corpus_serves_one_normalized_mono_clip(corpus: str) -> None:
    """The pinned R2 snapshot carries the filtered columns and yields a servable batch.

    :param corpus: Corpus config under ``datamodule/third_party/rir``.
    """
    node = _rir_datamodule_cfg(corpus)
    r2_io.ensure_r2_env_loaded()
    uri, storage_options = r2_io.lance_target(node.dataset_uri)
    schema = lance.dataset(uri, version=node.dataset_version, storage_options=storage_options).schema
    assert _REQUIRED_COLUMNS <= set(schema.names)

    datamodule = hydra.utils.instantiate(node)
    datamodule.setup("predict")
    batch = next(iter(datamodule.predict_dataloader()))

    audio = batch["audio"]
    assert audio.shape == (1, node.channels, datamodule.num_samples)
    assert torch.isfinite(audio).all()
    assert float(audio.abs().max()) == pytest.approx(1.0, abs=1e-3)
    assert np.isfinite(batch["mel"].numpy()).all()
