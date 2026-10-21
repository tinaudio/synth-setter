"""End-to-end coverage for the in-process KR106-native dataset and eval paths."""

from __future__ import annotations

import math
from pathlib import Path

import lance
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from pedalboard.io import AudioFile

from synth_setter.cli.eval import evaluate
from synth_setter.cli.finalize_dataset import finalize_lance
from synth_setter.cli.generate_dataset import from_hydra, spec_from_cfg
from synth_setter.cli.train import train
from synth_setter.data.vst import param_specs
from synth_setter.data.vst.kr106_native_runtime import import_kr106_native
from synth_setter.data.vst.shapes import AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD
from synth_setter.pipeline.ci.validate_shard import validate_all_shards_from_r2
from synth_setter.pipeline.data.lance_staging import shard_has_complete_attempt
from synth_setter.workspace import operator_workspace
from tests.conftest import build_surge_xt_embedding_train_cfg

_NATIVE_PARAM_SPEC = "ultramaster_kr106"
_NATIVE_TRAIN_ROWS = 2
_NATIVE_VAL_ROWS = 2
_NATIVE_TEST_ROWS = 2


def _compose_native_dataset_cfg(tmp_path: Path) -> DictConfig:
    """Compose the native smoke operator with test-scoped storage paths.

    :param tmp_path: Root for Hydra work directories and the fake R2 prefix.
    :returns: Native dataset generation configuration.
    """
    output_dir = tmp_path / "generate"
    output_dir.mkdir()
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="dataset",
            overrides=["experiment=generate_dataset/ultramaster-kr106-native-lance-smoke"],
        )
    with open_dict(cfg):
        cfg.paths.root_dir = str(operator_workspace())
        cfg.paths.output_dir = str(output_dir)
        cfg.paths.work_dir = str(output_dir)
        cfg.paths.log_dir = str(output_dir)
        cfg.train_val_test_sizes = [_NATIVE_TRAIN_ROWS, _NATIVE_VAL_ROWS, _NATIVE_TEST_ROWS]
        cfg.render.samples_per_render_batch = 1
        cfg.render.samples_per_shard = _NATIVE_TRAIN_ROWS
        cfg.r2.prefix = "fake-r2/kr106-native-pipeline/"
        cfg.logger = None
    return cfg


def _compose_native_predict_cfg(tmp_path: Path, dataset_root: Path) -> DictConfig:
    """Compose one-batch native prediction and post-processing from a trained checkpoint.

    :param tmp_path: Shared train/eval output root.
    :param dataset_root: Finalized native Lance splits and normalization statistics.
    :returns: Predict-mode evaluation configuration with native rendering enabled.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=surge/flow_simple",
                "synth=ultramaster_kr106_native",
                "render=kr106_native",
                "conditioning=cqt_online",
                "trainer=cpu",
                "datamodule=surge_lance",
                "callbacks=prediction_writer",
            ],
        )
    with open_dict(cfg):
        cfg.paths.root_dir = str(operator_workspace())
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.datamodule.fake = False
        cfg.datamodule.dataset_root = str(dataset_root)
        cfg.datamodule.predict_file = str(dataset_root / "test.lance")
        cfg.datamodule.batch_size = 1
        cfg.datamodule.num_workers = 0
        cfg.datamodule.pin_memory = False
        cfg.datamodule.ot = False
        cfg.datamodule.use_saved_mean_and_variance = True
        cfg.model.compile = False
        cfg.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg.mode = "predict"
        cfg.trainer.limit_predict_batches = 1
        cfg.trainer.enable_model_summary = False
        cfg.evaluation = {
            "render_vst": True,
            "compute_metrics": True,
            "rerender_target": True,
            "num_workers": 1,
        }
        cfg.logger = None
    return cfg


@pytest.mark.slow
def test_native_kr106_hydra_corpus_trains_and_renders_prediction_end_to_end(
    tmp_path: Path,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate native Lance splits, train one step, then render a predicted native clip.

    :param tmp_path: Isolated generation, training, and evaluation workspace.
    :param fake_r2_remote: Local filesystem backing the real rclone R2 transport.
    :param monkeypatch: Pins the generation worker to one local process.
    """
    native = import_kr106_native()
    assert native.get_version() == "2.5.13"
    monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
    monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")

    cfg_dataset = _compose_native_dataset_cfg(tmp_path)
    spec = spec_from_cfg(cfg_dataset)
    from_hydra(cfg_dataset)

    assert spec.render.renderer_backend == "kr106_native"
    assert spec.render.block_size == 512
    assert all(shard_has_complete_attempt(spec, shard.shard_id) for shard in spec.shards)
    assert validate_all_shards_from_r2(spec) == []

    finalize_dir = tmp_path / "finalize"
    finalize_dir.mkdir()
    finalize_lance(spec, finalize_dir)
    dataset_root = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    for split, expected_rows in (
        ("train", _NATIVE_TRAIN_ROWS),
        ("val", _NATIVE_VAL_ROWS),
        ("test", _NATIVE_TEST_ROWS),
    ):
        table = lance.dataset(dataset_root / f"{split}.lance").to_table(
            columns=[AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD]
        )
        assert table.num_rows == expected_rows
        for column in (AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD):
            assert np.isfinite(table.column(column).combine_chunks().to_numpy_ndarray()).all()
    train_root = tmp_path / "train"
    cfg_train = build_surge_xt_embedding_train_cfg(
        train_root,
        dataset_root,
        param_spec_name=_NATIVE_PARAM_SPEC,
        conditioning="cqt_online",
    )
    HydraConfig().set_config(cfg_train)
    _, train_objects = train(cfg_train)
    checkpoint_path = train_root / "checkpoints" / "last.ckpt"
    assert checkpoint_path.is_file()
    assert train_objects["trainer"].global_step == 1

    cfg_eval = _compose_native_predict_cfg(train_root, dataset_root)
    HydraConfig().set_config(cfg_eval)
    try:
        metrics, _ = evaluate(cfg_eval)
    finally:
        GlobalHydra.instance().clear()

    audio_metric_means = {
        key: value
        for key, value in metrics.items()
        if key.startswith("audio/") and key.endswith("_mean")
    }
    assert audio_metric_means
    assert all(math.isfinite(float(value)) for value in audio_metric_means.values())
    prediction = torch.load(train_root / "predictions" / "pred-0.pt", weights_only=True)
    assert prediction.shape == (1, len(param_specs[_NATIVE_PARAM_SPEC]))
    with AudioFile(str(train_root / "audio" / "sample_0" / "pred.wav")) as audio_file:
        rendered = audio_file.read(audio_file.frames)
    assert rendered.shape == (2, 176_400)
    assert np.isfinite(rendered).all()
    with AudioFile(str(train_root / "audio" / "sample_0" / "target.wav")) as audio_file:
        target = audio_file.read(audio_file.frames)
    assert target.shape == (2, 176_400)
    assert np.isfinite(target).all()
    assert np.max(np.abs(target)) > 0.0
