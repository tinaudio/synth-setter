"""Tests for the ``synth-setter-eval`` CLI entrypoint.

Every test composes a Hydra ``cfg`` and drives the in-process ``evaluate(cfg)``
entrypoint. Helper-level unit tests live in the sibling ``test_eval_*`` modules:
postprocessing argv in ``test_eval_postprocessing``, metric IO in
``test_eval_metrics``, and R2 upload / CLI e2e in ``test_eval_upload``.
``tests/_meta/test_entrypoint_test_modules.py`` enforces that no private
``synth_setter.cli`` helper is imported here.
"""

import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Literal, NamedTuple, cast
from unittest.mock import patch

import pytest
import torch
import wandb
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, open_dict
from omegaconf.errors import InterpolationResolutionError
from pedalboard.io import AudioFile

from synth_setter.cli.eval import evaluate
from synth_setter.cli.train import train
from synth_setter.data.vst import plugin_state_paths
from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.pipeline.spec_io import write_spec_to_path
from synth_setter.utils.utils import register_resolvers
from synth_setter.workspace import operator_workspace
from tests.conftest import REAL_VST_VARIANTS, assert_log_per_param_mse_wired
from tests.helpers.eval_fakes import (
    FAKE_METRICS_CSV,
    fake_postprocessing_subprocess,
)
from tests.helpers.recording_wandb_logger import RecordingWandbLogger as _RecordingWandbLogger
from tests.helpers.run_if import RunIf
from tests.helpers.wandb_artifacts import publish_checkpoint_artifact


class _FakeOracleDataset(NamedTuple):
    name: str
    fixture: str
    datamodule_group: str | None


class _AudioPredictionCase(NamedTuple):
    experiment: str
    datamodule: str
    filename: str
    model_overrides: tuple[str, ...]


_AUDIO_PREDICTION_DURATION_SECONDS = 4.0
_AUDIO_PREDICTION_SAMPLE_RATE = 44_100
_AUDIO_PREDICTION_SAMPLE_COUNT = int(
    _AUDIO_PREDICTION_DURATION_SECONDS * _AUDIO_PREDICTION_SAMPLE_RATE
)
_SURGE_XT_PREDICTION_WIDTH = 300
_AUDIO_PREDICTION_CASES = (
    _AudioPredictionCase(
        "ffn_full",
        "fsd",
        "FSD50K_000001.wav",
        (
            "model.net.d_model=32",
            "model.net.n_heads=2",
            "model.net.n_layers=1",
            "model.net.patch_size=16",
            "model.net.patch_stride=15",
            "model.compile=false",
        ),
    ),
    _AudioPredictionCase(
        "flow_full",
        "nsynth",
        "bass_electronic_000-025-050.wav",
        (
            "model.encoder.d_model=8",
            "model.encoder.n_heads=1",
            "model.encoder.n_layers=1",
            "model.encoder.n_conditioning_outputs=1",
            "model.encoder.patch_stride=15",
            "model.vector_field.d_model=8",
            "model.vector_field.num_heads=1",
            "model.vector_field.num_layers=1",
            "model.vector_field.d_ff=8",
            "model.vector_field.projection.num_tokens=4",
            "model.test_sample_steps=1",
            "model.compile=false",
        ),
    ),
    _AudioPredictionCase(
        "flow_mlp_full",
        "fsd",
        "FSD50K_000002.wav",
        (
            "model.encoder.d_model=8",
            "model.encoder.n_heads=1",
            "model.encoder.n_layers=1",
            "model.encoder.n_conditioning_outputs=1",
            "model.encoder.patch_stride=15",
            "model.vector_field.d_model=8",
            "model.vector_field.d_enc=4",
            "model.vector_field.num_layers=1",
            "model.test_sample_steps=1",
            "model.compile=false",
        ),
    ),
    _AudioPredictionCase(
        "vae_full",
        "nsynth",
        "guitar_acoustic_001-060-075.wav",
        (
            "+model.net.latent_flow_hidden_dim=16",
            "+model.net.latent_flow_num_layers=2",
            "+model.net.latent_flow_num_blocks=1",
            "+model.net.regression_flow_hidden_dim=16",
            "+model.net.regression_flow_num_layers=2",
            "+model.net.regression_flow_num_blocks=1",
            "model.compile=false",
        ),
    ),
)
_TORCHSYNTH_MIN_RELATIVE_VAL_IMPROVEMENT = 0.05


def _write_audio_prediction_fixture(path: Path) -> None:
    """Write a stereo tone with the production prediction duration and rate.

    :param path: Destination WAV path.
    """
    time = torch.arange(_AUDIO_PREDICTION_SAMPLE_COUNT) / _AUDIO_PREDICTION_SAMPLE_RATE
    tone = 0.4 * torch.sin(2 * torch.pi * 220.0 * time)
    stereo = torch.stack([tone, 0.5 * tone]).numpy()
    with AudioFile(
        str(path), "w", samplerate=_AUDIO_PREDICTION_SAMPLE_RATE, num_channels=2
    ) as audio_file:
        audio_file.write(stereo)


def _audio_prediction_cli_args(
    case: _AudioPredictionCase,
    *,
    checkpoint: Path,
    audio_root: Path,
    output_dir: Path,
) -> list[str]:
    """Build the public eval CLI invocation for a tiny checkpoint fixture.

    :param case: Model and audio datamodule variant under test.
    :param checkpoint: Real checkpoint to load.
    :param audio_root: Directory containing one prediction WAV.
    :param output_dir: Directory where PredictionWriter emits artifacts.
    :returns: Complete subprocess argv.
    """
    return [
        sys.executable,
        "-m",
        "synth_setter.cli.eval",
        f"experiment=surge/wandb_checkpoint/{case.experiment}",
        f"datamodule={case.datamodule}",
        "callbacks=eval_surge",
        "mode=predict",
        "trainer=cpu",
        "logger=wandb",
        "logger.wandb.offline=true",
        f"ckpt_path={checkpoint}",
        f"datamodule.root={audio_root}",
        "datamodule.stats_file=null",
        "datamodule.batch_size=1",
        "datamodule.num_workers=0",
        "datamodule.shuffle=false",
        *case.model_overrides,
        f"paths.output_dir={output_dir}",
        "hydra.job.chdir=false",
        "+trainer.enable_progress_bar=false",
        "+trainer.enable_model_summary=false",
    ]


def _save_audio_prediction_checkpoint(case: _AudioPredictionCase, path: Path) -> None:
    """Save a real checkpoint from the case's shipped Hydra model config.

    :param case: Model experiment and tiny architecture overrides.
    :param path: Destination checkpoint path.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            overrides=[
                f"experiment=surge/wandb_checkpoint/{case.experiment}",
                f"datamodule={case.datamodule}",
                "trainer=cpu",
                *case.model_overrides,
            ],
        )
    model = instantiate(cfg.model)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.strategy.connect(model)
    trainer.save_checkpoint(path)


def _assert_audio_prediction_artifacts(output_dir: Path) -> None:
    """Assert PredictionWriter emitted finite, correctly shaped tensors.

    :param output_dir: Eval output root containing ``predictions/``.
    """
    prediction = torch.load(
        output_dir / "predictions" / "pred-0.pt", map_location="cpu", weights_only=True
    )
    target_audio = torch.load(
        output_dir / "predictions" / "target-audio-0.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert prediction.shape == (1, _SURGE_XT_PREDICTION_WIDTH)
    assert torch.isfinite(prediction).all()
    assert target_audio.shape == (1, 2, _AUDIO_PREDICTION_SAMPLE_COUNT)


@pytest.mark.slow
@pytest.mark.parametrize("case", _AUDIO_PREDICTION_CASES, ids=lambda case: case.experiment)
def test_audio_dataset_predict_entrypoint_writes_artifacts(
    tmp_path: Path, case: _AudioPredictionCase
) -> None:
    """Every audio checkpoint family predicts through FSD50K or NSynth.

    :param tmp_path: Pytest fixture providing isolated input and output directories.
    :param case: Shipped model checkpoint and audio datamodule pairing under test.
    """
    audio_root = tmp_path / case.datamodule
    audio_root.mkdir()
    _write_audio_prediction_fixture(audio_root / case.filename)
    checkpoint = tmp_path / f"{case.experiment}.ckpt"
    _save_audio_prediction_checkpoint(case, checkpoint)
    output_dir = tmp_path / f"{case.experiment}-output"

    result = subprocess.run(  # noqa: S603 — argv contains only test-owned paths
        _audio_prediction_cli_args(
            case,
            checkpoint=checkpoint,
            audio_root=audio_root,
            output_dir=output_dir,
        ),
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
    _assert_audio_prediction_artifacts(output_dir)


def _compose_torchsynth_overfit_cfg(tmp_path: Path) -> DictConfig:
    """Compose the deterministic TorchSynth checkpoint smoke run.

    :param tmp_path: Pinned Hydra output and log directory.
    :returns: Ready-to-run training configuration.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        train_cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=torchsynth/ffn",
                "trainer=cpu",
                "+trainer.max_epochs=10",
                "+trainer.limit_train_batches=1",
                "+trainer.limit_val_batches=1",
                "trainer.val_check_interval=1.0",
                "trainer.check_val_every_n_epoch=1",
                "datamodule.resample_train_per_epoch=false",
                "datamodule.train_val_test_sizes=[1,1,1]",
                "datamodule.batch_size=1",
                "datamodule.num_workers=0",
                "model.compile=false",
                "model.optimizer.lr=0.0001",
                "logger=csv",
            ],
        )
    with open_dict(train_cfg):
        train_cfg.paths.root_dir = str(operator_workspace())
        train_cfg.paths.output_dir = str(tmp_path)
        train_cfg.paths.log_dir = str(tmp_path)
        train_cfg.test = False
        train_cfg.seed = 123
    return train_cfg


def _torchsynth_initial_loss(
    cfg: DictConfig,
    *,
    stage: Literal["fit", "validate"],
    split: Literal["train", "val"],
) -> float:
    """Return an untrained model's loss on a fixed TorchSynth batch.

    :param cfg: TorchSynth train or evaluation configuration.
    :param stage: Datamodule setup stage.
    :param split: Dataloader split used for the baseline batch.
    :returns: Initial MSE for the fixed batch.
    """
    baseline_datamodule = instantiate(cfg.datamodule)
    baseline_datamodule.setup(stage)
    if split == "train":
        dataloader = baseline_datamodule.train_dataloader()
    else:
        dataloader = baseline_datamodule.val_dataloader()
    # Seed exactly as train() does (L.seed_everything, covering torch/numpy/python RNG)
    # so this "initial" model matches training's start regardless of what model init draws.
    seed_everything(cfg.seed, workers=True)
    baseline_model = instantiate(cfg.model)
    if split == "val":
        baseline_model.eval()
    total_squared_error = 0.0
    total_elements = 0
    with torch.no_grad():
        for baseline_audio, baseline_params, *_ in dataloader:
            squared_error = torch.nn.functional.mse_loss(
                baseline_model(baseline_audio), baseline_params, reduction="sum"
            )
            total_squared_error += squared_error.item()
            total_elements += baseline_params.numel()
    return total_squared_error / total_elements


def _compose_torchsynth_eval_cfg(tmp_path: Path, checkpoint: Path) -> DictConfig:
    """Compose validation against the trained TorchSynth checkpoint.

    :param tmp_path: Pinned Hydra output and log directory.
    :param checkpoint: Trained checkpoint path.
    :returns: Ready-to-run evaluation configuration.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=torchsynth/eval_ffn",
                "trainer=cpu",
                "datamodule.train_val_test_sizes=[2,32,2]",
                "datamodule.batch_size=1",
                "datamodule.num_workers=0",
            ],
        )
    with open_dict(cfg):
        cfg.paths.root_dir = str(operator_workspace())
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.ckpt_path = str(checkpoint)
        cfg.seed = 123
    return cfg


@pytest.mark.slow
def test_eval_torchsynth_experiment_validates_checkpoint(tmp_path: Path) -> None:
    """Train and validate TorchSynth using audio rendered on the local machine.

    :param tmp_path: Shared training, checkpoint, and evaluation directory.
    """
    train_cfg = _compose_torchsynth_overfit_cfg(tmp_path)
    initial_loss = _torchsynth_initial_loss(train_cfg, stage="fit", split="train")
    HydraConfig().set_config(train_cfg)
    try:
        train_metrics, train_objects = train(train_cfg)
    finally:
        GlobalHydra.instance().clear()

    overfit_loss = train_metrics["train/loss_epoch"].item()
    assert math.isfinite(overfit_loss)
    assert overfit_loss < initial_loss

    checkpoint = Path(train_objects["trainer"].checkpoint_callback.best_model_path)
    assert checkpoint.is_file()
    eval_cfg = _compose_torchsynth_eval_cfg(tmp_path, checkpoint)
    initial_val_loss = _torchsynth_initial_loss(eval_cfg, stage="validate", split="val")
    HydraConfig().set_config(eval_cfg)
    try:
        metric_dict, eval_objects = evaluate(eval_cfg)
    finally:
        GlobalHydra.instance().clear()

    val_loss = metric_dict["val/loss"]
    assert torch.isfinite(val_loss)
    assert val_loss < initial_val_loss * (1 - _TORCHSYNTH_MIN_RELATIVE_VAL_IMPROVEMENT)
    eval_batch = next(iter(eval_objects["datamodule"].val_dataloader()))
    assert torch.isfinite(eval_batch[0]).all()


_FAKE_ORACLE_DATASETS = [
    pytest.param(
        _FakeOracleDataset("lance", "fake_surge_smoke_datasets", "surge_lance"),
        id="lance",
    )
]


@pytest.mark.requires_vst
@pytest.mark.slow
def test_evaluate_runs_oracle_with_null_ckpt_path(
    tmp_path: Path,
    surge_xt_smoke_datasets: Path,
    dataset_spec_factory: Callable[..., DatasetSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fake oracle returns ``batch["params"]`` verbatim, so ``test/param_mse`` is exactly zero.

    The load-bearing invariant is that ``ckpt_path=null`` survives Hydra
    composition into ``evaluate()`` and the oracle's exact-zero MSE reaches
    the metric dict.

    :param tmp_path: Pinned as Hydra ``paths.output_dir`` / ``paths.log_dir``.
    :param surge_xt_smoke_datasets: Holds ``{train,val,test}.lance`` + ``stats.npz``.
    :param dataset_spec_factory: Factory producing the frozen dataset provenance.
    :param monkeypatch: Replaces the external W&B logger boundary.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=surge/test-mps-fake-oracle",
                "trainer=cpu",
                # The experiment defaults to mode=predict; this invariant is test-mode.
                "mode=test",
                "datamodule.param_spec_name=surge_4",
            ],
        )

    with open_dict(cfg):
        cfg.paths.root_dir = str(operator_workspace())
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.datamodule.dataset_root = str(surge_xt_smoke_datasets)
        cfg.datamodule.predict_file = str(surge_xt_smoke_datasets / "test.lance")
        cfg.datamodule.batch_size = 1
        cfg.datamodule.num_workers = 0
        cfg.ckpt_path = None

    write_spec_to_path(
        dataset_spec_factory(
            task_name="lineage-eval",
            train_val_test_sizes=[4, 4, 0],
            r2={"bucket": "intermediate-data"},
            render={"samples_per_shard": 4},
        ),
        surge_xt_smoke_datasets / "input_spec.json",
    )
    HydraConfig().set_config(cfg)
    logger = _RecordingWandbLogger()
    try:
        with patch("synth_setter.cli.eval.instantiate_loggers", return_value=[logger]):
            metric_dict, _ = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    param_mse = metric_dict["test/param_mse"]
    assert isinstance(param_mse, torch.Tensor)
    assert param_mse.numel() == 1
    assert param_mse.dtype.is_floating_point
    assert torch.isfinite(param_mse), f"oracle test/param_mse must be finite; got {param_mse!r}"
    assert param_mse.item() == 0.0
    assert logger.experiment.config["ckpt_path"] is None
    assert logger.experiment.config.allow_val_change_calls == [True]
    assert logger.used_artifacts == ["data-lineage-eval:lineage-eval-20260520T000000000Z"]


_FLOW_LAD_EVAL_OVERRIDES = {
    "flow_simple": (
        "model.vector_field.d_model=8",
        "model.vector_field.num_heads=1",
        "model.vector_field.num_layers=1",
        "model.vector_field.d_ff=8",
        "model.vector_field.projection.num_tokens=4",
    ),
    "flow_mlp_simple": (
        "model.vector_field.d_model=8",
        "model.vector_field.d_enc=4",
        "model.vector_field.num_layers=1",
    ),
}


@pytest.mark.parametrize("experiment", sorted(_FLOW_LAD_EVAL_OVERRIDES))
def test_evaluate_flow_simple_test_mode_logs_param_lad(tmp_path: Path, experiment: str) -> None:
    """``mode=test`` through both flow configs logs ``test/param_lad`` beside the MSE.

    Pins the production ``model.param_spec_name`` wiring end-to-end: surge_simple
    has interchangeable blocks, so the eval entrypoint must emit the metric.

    :param tmp_path: Pinned as Hydra ``paths.output_dir`` / ``paths.log_dir``.
    :param experiment: Surge flow experiment variant under test.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                f"experiment=surge/{experiment}",
                "trainer=cpu",
                "mode=test",
                "model.encoder.d_model=8",
                "model.encoder.n_heads=1",
                "model.encoder.n_layers=1",
                "model.encoder.n_conditioning_outputs=1",
                "model.encoder.patch_stride=15",
                *_FLOW_LAD_EVAL_OVERRIDES[experiment],
                "model.test_sample_steps=1",
                "model.compile=false",
            ],
        )

    with open_dict(cfg):
        cfg.paths.root_dir = str(operator_workspace())
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.datamodule.fake = True
        cfg.datamodule.batch_size = 2
        cfg.datamodule.num_workers = 0
        cfg.datamodule.use_saved_mean_and_variance = False
        cfg.ckpt_path = None
        cfg.logger = None

    HydraConfig().set_config(cfg)
    try:
        metric_dict, _ = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert "test/param_mse" in metric_dict
    param_lad = metric_dict["test/param_lad"]
    assert torch.isfinite(param_lad)
    assert param_lad.item() <= metric_dict["test/param_mse"].item() + 1e-6


@pytest.mark.requires_vst
@pytest.mark.slow
@pytest.mark.parametrize("surge_smoke_variant", REAL_VST_VARIANTS, indirect=True)
def test_evaluate_predict_explicit_shuffle_seed_rejects_nonuniform_params_via_subprocess(
    cfg_surge_real_train: DictConfig,
    cfg_surge_real_eval: DictConfig,
) -> None:
    """Non-zero ``shuffle_seed`` with non-uniform params causes the metrics subprocess to fail, for both dataset formats.

    Drives the real train→eval roundtrip end-to-end with ``shuffle_seed=7``,
    exercising the ``evaluate()`` → ``_run_predict_postprocessing`` →
    metrics-subprocess wiring. The smoke dataset renders distinct params per
    sample, so the uniform-params guard inside ``compute_audio_metrics`` raises
    ``ValueError`` (non-zero seed + non-uniform = misconfiguration), the
    subprocess exits non-zero, and ``CalledProcessError`` surfaces at the
    ``evaluate()`` boundary — confirming the gate is wired through the real
    entrypoint (#489).

    :param cfg_surge_real_train: Surge XT smoke-test training config (Lance).
    :param cfg_surge_real_eval: Matching predict-mode eval config (render + metrics on),
        sharing ``tmp_path`` so eval reads the checkpoint training writes.
    """
    HydraConfig().set_config(cfg_surge_real_train)
    train(cfg_surge_real_train)
    assert Path(cfg_surge_real_eval.ckpt_path).exists()

    with open_dict(cfg_surge_real_eval):
        cfg_surge_real_eval.evaluation.shuffle_seed = 7

    HydraConfig().set_config(cfg_surge_real_eval)
    with pytest.raises(subprocess.CalledProcessError):
        evaluate(cfg_surge_real_eval)


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_eval(tmp_path: Path, cfg_train: DictConfig, cfg_eval: DictConfig) -> None:
    """Train for 1 epoch with ``train.py`` then evaluate the resulting checkpoint with ``eval.py``.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    :param cfg_eval: A DictConfig containing a valid evaluation configuration.
    """
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.test = True
    with open_dict(cfg_eval):
        cfg_eval.trainer.accelerator = "gpu"

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")

    HydraConfig().set_config(cfg_eval)
    test_metric_dict, _ = evaluate(cfg_eval)

    assert math.isfinite(test_metric_dict["test/loss"].item())
    assert (
        abs(train_metric_dict["test/loss"].item() - test_metric_dict["test/loss"].item()) < 0.001
    )


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_validate(tmp_path: Path, cfg_train: DictConfig, cfg_eval: DictConfig) -> None:
    """Train one epoch then validate the checkpoint via ``eval.py`` ``mode=validate``.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    :param cfg_eval: A DictConfig containing a valid evaluation configuration.
    """
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.test = False
    with open_dict(cfg_eval):
        cfg_eval.trainer.accelerator = "gpu"

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_eval.mode = "validate"

    HydraConfig().set_config(cfg_eval)
    val_metric_dict, _ = evaluate(cfg_eval)

    assert math.isfinite(val_metric_dict["val/loss"].item())
    assert abs(train_metric_dict["val/loss"].item() - val_metric_dict["val/loss"].item()) < 0.001


def _compose_fake_oracle_eval_cfg(
    tmp_path: Path,
    dataset_root: Path,
    *,
    mode: str,
    param_spec_name: str = "surge_4",
    datamodule: str | None = None,
) -> DictConfig:
    """Compose ``eval.yaml`` with the CPU ``surge/fake_oracle`` experiment, pinned to a dataset.

    Drives the CPU production oracle config (``experiment/surge/fake_oracle.yaml``)
    rather than its MPS smoke sibling, so this composition is itself coverage of
    that config. ``param_spec_name`` selects the datamodule schema and matching render
    group.

    :param tmp_path: Pinned as ``paths.output_dir`` / ``paths.log_dir``; the
        predict-mode ``PredictionWriter`` writes ``predictions/`` beneath it.
    :param dataset_root: Holds the ``{train,val,test}.lance`` splits + ``stats.npz``.
    :param mode: ``cfg.mode`` under test (``test`` / ``validate`` / ``val`` /
        ``predict`` / an unknown spelling).
    :param param_spec_name: Param spec selecting the dataset schema and render group.
    :param datamodule: Optional datamodule group override (e.g. ``surge_lance``);
        ``None`` keeps the experiment's default ``surge`` group.
    :returns: Composed eval ``DictConfig`` ready for ``evaluate``.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=["experiment=surge/fake_oracle", f"mode={mode}"]
            + ([f"datamodule={datamodule}"] if datamodule else []),
        )
    with open_dict(cfg):
        cfg.paths.root_dir = str(operator_workspace())
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.datamodule.dataset_root = str(dataset_root)
        cfg.datamodule.param_spec_name = param_spec_name
        # None lets the datamodule derive ``test.<its shard suffix>`` under dataset_root.
        cfg.datamodule.predict_file = None
        cfg.datamodule.batch_size = 1
        cfg.datamodule.num_workers = 0
        cfg.datamodule.use_saved_mean_and_variance = True
        cfg.ckpt_path = None
        # surge/base enables the wandb logger; null it so the fast loop never hits
        # wandb init/network/login (these tests don't assert on logging).
        cfg.logger = None
        # Pin the full split because surge/base bounds validation by batch count.
        # mode=val/validate must see every fixture row.
        cfg.trainer.limit_val_batches = 1.0
        # Render group is null on fake_oracle; set it inline to the dataset's spec.
        cfg.render = {
            "param_spec_name": param_spec_name,
            "plugin_state_path": str(plugin_state_paths[param_spec_name]),
            "plugin_path": "plugins/fake.vst3",
        }
    return cfg


def _compose_parametrized_fake_oracle_eval_cfg(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    dataset_variant: _FakeOracleDataset,
    *,
    mode: str,
) -> DictConfig:
    """Compose the fake-oracle eval cfg for the parametrized Lance dataset.

    :param tmp_path: Pinned as ``paths.output_dir`` / ``paths.log_dir``.
    :param request: Fetches the parametrized dataset fixture.
    :param dataset_variant: Dataset fixture and datamodule override under test.
    :param mode: Eval mode to compose.
    :returns: Composed eval ``DictConfig`` ready for ``evaluate``.
    """
    dataset_root = request.getfixturevalue(dataset_variant.fixture)
    return _compose_fake_oracle_eval_cfg(
        tmp_path,
        dataset_root,
        mode=mode,
        datamodule=dataset_variant.datamodule_group,
    )


@pytest.mark.fake_vst
@pytest.mark.parametrize("dataset_variant", _FAKE_ORACLE_DATASETS)
def test_evaluate_predict_mode_merges_audio_metrics_into_metric_dict(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    dataset_variant: _FakeOracleDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mode=predict`` runs the oracle's predict + postprocessing and merges audio metrics.

    Exercises the predict branch of ``evaluate`` end-to-end on the fast loop: the
    fake-plugin dataset feeds ``trainer.predict`` (the ``PredictionWriter`` writes
    ``predictions/``), then the render + metrics subprocesses are faked so the
    aggregated ``audio/*`` values land in ``trainer.callback_metrics`` via
    ``metric_dict.update(audio_metrics)``. Pins that the rank-zero gate fires and
    the float audio metrics reach the returned dict alongside any tensor metrics.

    :param tmp_path: Hydra ``output_dir``; ``predictions/`` / ``audio/`` / ``metrics/``
        are derived beneath it.
    :param request: Fetches the parametrized dataset fixture.
    :param dataset_variant: Dataset fixture and datamodule override under test.
    :param monkeypatch: Stubs the render/metrics subprocesses and the headless
        wrapper extraction so no real VST host or Python subprocess launches.
    """
    cfg = _compose_parametrized_fake_oracle_eval_cfg(
        tmp_path, request, dataset_variant, mode="predict"
    )
    monkeypatch.setattr(
        "synth_setter.cli.eval.subprocess.run",
        fake_postprocessing_subprocess(),
    )
    monkeypatch.setattr("synth_setter.cli.eval.vst_headless_wrapper", lambda: object())
    monkeypatch.setattr(
        "synth_setter.cli.eval.as_file",
        lambda _traversable: nullcontext(Path("/fake/headless-wrapper")),
    )

    HydraConfig().set_config(cfg)
    try:
        metric_dict, _ = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert metric_dict["audio/mss_mean"] == pytest.approx(0.5)
    assert metric_dict["audio/rms_std"] == pytest.approx(0.01)
    for key in ("mss", "wmfcc", "sot", "rms"):
        for stat in ("mean", "std"):
            value = metric_dict[f"audio/{key}_{stat}"]
            assert isinstance(value, float) and math.isfinite(value)


@pytest.mark.fake_vst
@pytest.mark.parametrize("dataset_variant", _FAKE_ORACLE_DATASETS)
def test_evaluate_predict_mode_logs_per_sample_metrics_table_to_wandb(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    dataset_variant: _FakeOracleDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mode=predict`` with an active wandb run uploads ``metrics.csv`` as a wandb.Table.

    Exercises the ``_log_metrics_csv_to_wandb`` call-through via the real
    ``evaluate`` entrypoint: the fake metrics subprocess writes both
    ``aggregated_metrics.csv`` and ``metrics.csv``; a spy on ``wandb.run.log``
    verifies the per-sample Table arrives under ``audio/per_sample_metrics``.

    :param tmp_path: Hydra ``output_dir``; the fake subprocess writes CSVs beneath it.
    :param request: Fetches the parametrized dataset fixture.
    :param dataset_variant: Dataset fixture and datamodule override under test.
    :param monkeypatch: Stubs subprocesses, headless wrapper, and ``wandb.run``.
    """
    logged: list[dict[str, object]] = []

    class _Spy:
        """Stand-in for ``wandb.run`` that records ``log`` payloads; no-ops SDK lifecycle calls.

        ``__getattr__`` absorbs wandb SDK cleanup methods (e.g. ``finish``,
        ``summary``) that Lightning triggers after predict — they are irrelevant to
        this test's contract.
        """

        def log(self, payload: dict[str, object]) -> None:
            """Append payload to the capture list.

            :param payload: The wandb log payload to record.
            """
            logged.append(payload)

        def __getattr__(self, _name: str) -> object:
            """Return a no-op callable for any wandb SDK method not explicitly defined.

            :param _name: Attribute name; unused — any undeclared attribute gets a no-op.
            :returns: A callable that accepts any arguments and returns ``None``.
            """
            return lambda *_args, **_kwargs: None

    cfg = _compose_parametrized_fake_oracle_eval_cfg(
        tmp_path, request, dataset_variant, mode="predict"
    )
    monkeypatch.setattr(
        "synth_setter.cli.eval.subprocess.run",
        fake_postprocessing_subprocess(per_sample_metrics_csv=FAKE_METRICS_CSV),
    )
    monkeypatch.setattr("synth_setter.cli.eval.vst_headless_wrapper", lambda: object())
    monkeypatch.setattr(
        "synth_setter.cli.eval.as_file",
        lambda _traversable: nullcontext(Path("/fake/headless-wrapper")),
    )
    monkeypatch.setattr(wandb, "run", _Spy())
    # task_wrapper's teardown calls module-level wandb.finish() while wandb.run is
    # truthy (utils.py); stub it so the spy run is the only wandb surface exercised.
    monkeypatch.setattr(wandb, "finish", lambda *_args, **_kwargs: None)

    HydraConfig().set_config(cfg)
    try:
        evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    table_payloads = [p for p in logged if "audio/per_sample_metrics" in p]
    assert len(table_payloads) == 1
    assert isinstance(table_payloads[0]["audio/per_sample_metrics"], wandb.Table)


@pytest.mark.fake_vst
@pytest.mark.parametrize("dataset_variant", _FAKE_ORACLE_DATASETS)
def test_evaluate_predict_mode_logs_shuffle_permutation_table_to_wandb(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    dataset_variant: _FakeOracleDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mode=predict`` uploads the render-order probe permutation as a ``shuffle/permutation`` Table.

    Exercises the ``_log_shuffle_permutation_to_wandb`` call-through via the real
    ``evaluate`` entrypoint: the fake metrics subprocess writes ``aggregated_metrics.csv``
    and ``shuffle_permutation.csv``; a spy on ``wandb.run.log`` verifies the permutation
    Table arrives under ``shuffle/permutation`` (#1669).

    :param tmp_path: Hydra ``output_dir``; the fake subprocess writes CSVs beneath it.
    :param request: Fetches the parametrized dataset fixture.
    :param dataset_variant: Dataset fixture and datamodule override under test.
    :param monkeypatch: Stubs subprocesses, headless wrapper, and ``wandb.run``.
    """
    permutation_csv = "dest_idx,src_idx\n0,1\n1,0\n"
    logged: list[dict[str, object]] = []

    class _Spy:
        """Stand-in for ``wandb.run`` that records ``log`` payloads; no-ops SDK lifecycle calls.

        ``__getattr__`` absorbs wandb SDK cleanup methods (e.g. ``finish``,
        ``summary``) that Lightning triggers after predict — they are irrelevant to
        this test's contract.
        """

        def log(self, payload: dict[str, object]) -> None:
            """Record one ``wandb.run.log`` call's argument.

            :param payload: The dict passed to ``wandb.run.log``.
            """
            logged.append(payload)

        def __getattr__(self, _name: str) -> object:
            """Return a no-op callable for any undeclared wandb SDK method.

            :param _name: Unused; any undeclared attribute resolves to the no-op.
            :returns: A callable accepting any args and returning ``None``.
            """
            return lambda *_args, **_kwargs: None

    cfg = _compose_parametrized_fake_oracle_eval_cfg(
        tmp_path, request, dataset_variant, mode="predict"
    )
    monkeypatch.setattr(
        "synth_setter.cli.eval.subprocess.run",
        fake_postprocessing_subprocess(shuffle_permutation_csv=permutation_csv),
    )
    monkeypatch.setattr("synth_setter.cli.eval.vst_headless_wrapper", lambda: object())
    monkeypatch.setattr(
        "synth_setter.cli.eval.as_file",
        lambda _traversable: nullcontext(Path("/fake/headless-wrapper")),
    )
    monkeypatch.setattr(wandb, "run", _Spy())
    monkeypatch.setattr(wandb, "finish", lambda *_args, **_kwargs: None)

    HydraConfig().set_config(cfg)
    try:
        evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    table_payloads = [p for p in logged if "shuffle/permutation" in p]
    assert len(table_payloads) == 1
    table = cast(wandb.Table, table_payloads[0]["shuffle/permutation"])
    assert isinstance(table, wandb.Table)
    assert table.columns == ["dest_idx", "src_idx"]
    assert table.data == [[0, 1], [1, 0]]


@pytest.mark.fake_vst
@pytest.mark.parametrize("dataset_variant", _FAKE_ORACLE_DATASETS)
def test_evaluate_validate_mode_legacy_val_spelling_runs_oracle(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    dataset_variant: _FakeOracleDataset,
) -> None:
    """``mode=val`` (legacy spelling) routes to ``trainer.validate`` and logs zero MSE.

    The ``evaluate`` mode branch accepts both ``val`` and ``validate``; only
    ``validate`` is otherwise covered (the GPU train→validate test). This pins the
    backward-compatible ``val`` alias on the fast loop: the oracle returns params
    verbatim, so ``val/param_mse`` is exactly zero.

    :param tmp_path: Pinned as Hydra ``output_dir`` / ``log_dir``.
    :param request: Fetches the parametrized dataset fixture.
    :param dataset_variant: Dataset fixture and datamodule override under test.
    """
    cfg = _compose_parametrized_fake_oracle_eval_cfg(
        tmp_path, request, dataset_variant, mode="val"
    )

    HydraConfig().set_config(cfg)
    try:
        metric_dict, _ = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    param_mse = metric_dict["val/param_mse"]
    assert isinstance(param_mse, torch.Tensor)
    assert param_mse.item() == 0.0


def test_evaluate_unregistered_param_spec_name_raises_resolution_error(
    tmp_path: Path,
) -> None:
    """An unregistered ``datamodule.param_spec_name`` fails during model resolution.

    The model width resolver rejects an unknown spec before model construction or
    dataset access.

    :param tmp_path: Pinned as Hydra ``output_dir`` / ``log_dir``; the dataset root
        points at a nonexistent subdirectory that is never read.
    """
    cfg = _compose_fake_oracle_eval_cfg(tmp_path, tmp_path / "missing-datasets", mode="validate")
    with open_dict(cfg):
        cfg.datamodule.param_spec_name = "does_not_exist"

    HydraConfig().set_config(cfg)
    try:
        with pytest.raises(InterpolationResolutionError, match="does_not_exist"):
            evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()


@pytest.mark.fake_vst
def test_evaluate_unknown_mode_returns_only_callback_metrics(
    tmp_path: Path,
    fake_surge_smoke_datasets: Path,
) -> None:
    """An unrecognized ``mode`` runs no trainer stage and returns the empty callback metrics.

    ``evaluate`` has no ``else``/raise on its mode branch: an unknown spelling is a
    silent no-op that skips test/validate/predict, so ``trainer.callback_metrics``
    is empty and no ``audio/*`` postprocessing runs. Pins that contract so a typo'd
    mode fails visibly (empty metrics) rather than masquerading as a passing run.

    :param tmp_path: Pinned as Hydra ``output_dir`` / ``log_dir``.
    :param fake_surge_smoke_datasets: CPU-fast surge_4 dataset (no real VST).
    """
    cfg = _compose_fake_oracle_eval_cfg(tmp_path, fake_surge_smoke_datasets, mode="bogus-mode")

    HydraConfig().set_config(cfg)
    try:
        metric_dict, _ = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert metric_dict == {}


@pytest.mark.fake_vst
@pytest.mark.parametrize("dataset_variant", _FAKE_ORACLE_DATASETS)
def test_evaluate_predict_mode_includes_shuffled_audio_metrics_when_subprocess_writes_shuffled_csv(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    dataset_variant: _FakeOracleDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Merges ``shuffled_audio/*`` keys when the metrics subprocess also writes the shuffled CSV.

    The fake subprocess writes both ``aggregated_metrics.csv`` and
    ``aggregated_metrics_shuffled.csv``, exercising the ``evaluate()`` →
    ``_run_predict_postprocessing`` → ``_load_audio_metrics`` path that merges
    the shuffled probe output into the returned metric dict under the
    ``shuffled_audio/`` prefix. Pins that the new branch in ``_load_audio_metrics``
    is wired through the real ``evaluate()`` entrypoint (#489).

    :param tmp_path: Hydra ``output_dir``; output files are derived beneath it.
    :param request: Fetches the parametrized dataset fixture.
    :param dataset_variant: Dataset fixture and datamodule override under test.
    :param monkeypatch: Stubs render/metrics subprocesses; no real VST launches.
    """
    _SHUFFLED_CSV = ",mean,std\nmss,0.8,0.05\nwmfcc,0.4,0.03\nsot,0.3,0.02\nrms,0.7,0.01\n"

    cfg = _compose_parametrized_fake_oracle_eval_cfg(
        tmp_path, request, dataset_variant, mode="predict"
    )
    monkeypatch.setattr(
        "synth_setter.cli.eval.subprocess.run",
        fake_postprocessing_subprocess(shuffled_metrics_csv=_SHUFFLED_CSV),
    )
    monkeypatch.setattr("synth_setter.cli.eval.vst_headless_wrapper", lambda: object())
    monkeypatch.setattr(
        "synth_setter.cli.eval.as_file",
        lambda _traversable: nullcontext(Path("/fake/headless-wrapper")),
    )

    HydraConfig().set_config(cfg)
    try:
        metric_dict, _ = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert metric_dict["shuffled_audio/mss_mean"] == pytest.approx(0.8)
    assert metric_dict["shuffled_audio/rms_std"] == pytest.approx(0.01)
    for key in ("mss", "wmfcc", "sot", "rms"):
        for stat in ("mean", "std"):
            value = metric_dict[f"shuffled_audio/{key}_{stat}"]
            assert isinstance(value, float) and math.isfinite(value)


@pytest.mark.parametrize("render_group", ["surge_simple", "surge_xt"])
def test_eval_render_group_exposes_postprocessing_keys(render_group: str) -> None:
    """Composing ``render=<group>`` into eval exposes the three keys postprocessing reads.

    ``_run_predict_postprocessing`` reads ``cfg.render.param_spec_name`` /
    ``plugin_state_path`` / ``plugin_path`` to build the renderer argv. This composition
    test pins that both shipped render groups supply all three keys, so a future
    rename in a ``render/*.yaml`` surfaces here rather than mid-eval.

    :param render_group: Render config group composed into the eval cfg.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=["experiment=surge/fake_oracle", f"render={render_group}"],
        )
    try:
        assert cfg.render.param_spec_name
        assert cfg.render.plugin_state_path
        assert cfg.render.plugin_path
    finally:
        GlobalHydra.instance().clear()


@pytest.mark.requires_vst
@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("WANDB_API_KEY"),
    reason="real W&B round-trip needs WANDB_API_KEY (injected on trusted CI only)",
)
@pytest.mark.parametrize("experiment_name", ["surge/ffn_full"], indirect=True)
def test_evaluate_loads_wandb_resolved_checkpoint_and_runs_inference(
    tmp_path: Path,
    cfg_surge_xt: DictConfig,
    cfg_surge_xt_eval: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
    experiment_name: str,
) -> None:
    """Predict eval resolves ckpt_path via ``${wandb:...}`` from the live registry, loads it, and runs inference.

    The full wandb_checkpoint contract end to end: train a real checkpoint, publish it to
    ``tinaudio/synth-setter-citest``, then pin ``ckpt_path: ${wandb:...}`` (no CLI path, the form
    the ``surge/wandb_checkpoint/<id>`` overlay produces) and run ``evaluate()`` in predict mode.
    The resolver downloads the artifact, Lightning loads the weights, and predict-mode inference
    writes finite per-sample predictions — the contract a fake-stub test cannot prove.

    :param tmp_path: Shared output dir; also the workspace root the resolver caches under.
    :param cfg_surge_xt: Surge XT smoke training config — one step produces the checkpoint.
    :param cfg_surge_xt_eval: Matching predict-mode eval config; its ckpt_path is repinned here.
    :param monkeypatch: Pins ``SYNTH_SETTER_WORKSPACE`` so the download cache stays under tmp_path.
    :param experiment_name: Pinned to ``surge/ffn_full`` — the artifact id need only round-trip.
    """
    HydraConfig().set_config(cfg_surge_xt)
    train(cfg_surge_xt)
    ckpt = Path(cfg_surge_xt_eval.ckpt_path)
    assert ckpt.is_file(), "train step did not write the checkpoint"

    # Body runs inside the ``with`` so the resolver downloads before the artifact/run teardown.
    with publish_checkpoint_artifact(
        ckpt, "model-citest-ffn_full-eval", tmp_path / "wandb"
    ) as ref:
        # Contain the resolver's download cache under tmp_path so each run fetches fresh (a warm
        # self-hosted runner must not reuse a stale cached ckpt for the same :latest ref).
        monkeypatch.setenv("SYNTH_SETTER_WORKSPACE", str(tmp_path))
        monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
        operator_workspace.cache_clear()
        register_resolvers()
        with open_dict(cfg_surge_xt_eval):
            cfg_surge_xt_eval.ckpt_path = "${wandb:" + ref + "}"

        HydraConfig().set_config(cfg_surge_xt_eval)
        evaluate(cfg_surge_xt_eval)

    assert (tmp_path / ".cache" / "checkpoints").is_dir(), "resolver did not download the artifact"
    predictions_dir = tmp_path / "predictions"
    assert predictions_dir.is_dir()
    preds = sorted(predictions_dir.glob("pred-*.pt"))
    assert preds, "predict mode wrote no predictions"
    for pred_file in preds:
        tensor = torch.load(pred_file, weights_only=True)
        assert torch.isfinite(tensor).all(), f"{pred_file.name} contains NaN/Inf"


@pytest.mark.fake_vst
def test_evaluate_validate_mode_lance_datamodule_runs_oracle(
    tmp_path: Path,
    fake_surge_smoke_datasets: Path,
) -> None:
    """``datamodule=surge_lance`` drives ``evaluate`` end-to-end over Lance splits.

    The oracle returns params verbatim, so ``val/param_mse`` is exactly zero,
    with every batch read from Lance.

    :param tmp_path: Pinned as Hydra ``output_dir`` / ``log_dir``.
    :param fake_surge_smoke_datasets: Natively-generated Lance smoke dataset.
    """
    cfg = _compose_fake_oracle_eval_cfg(
        tmp_path, fake_surge_smoke_datasets, mode="validate", datamodule="surge_lance"
    )

    HydraConfig().set_config(cfg)
    try:
        metric_dict, object_dict = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert_log_per_param_mse_wired(object_dict["trainer"], "surge_4")

    param_mse = metric_dict["val/param_mse"]
    assert isinstance(param_mse, torch.Tensor)
    assert param_mse.item() == 0.0


def test_evaluate_test_mode_partial_lance_root_returns_metric(
    cfg_train_lance: DictConfig,
) -> None:
    """Real ``evaluate`` consumes ``test.lance`` when train and val are absent.

    :param cfg_train_lance: Tiny production-composed Lance configuration.
    """
    dataset_root = Path(cfg_train_lance.datamodule.dataset_root)
    shutil.rmtree(dataset_root / "train.lance")
    shutil.rmtree(dataset_root / "val.lance")
    with open_dict(cfg_train_lance):
        cfg_train_lance.mode = "test"
        cfg_train_lance.ckpt_path = None

    HydraConfig().set_config(cfg_train_lance)
    try:
        metric_dict, object_dict = evaluate(cfg_train_lance)
    finally:
        GlobalHydra.instance().clear()

    assert math.isfinite(metric_dict["test/param_mse"].item())
    assert Path(object_dict["datamodule"].dataset_root) == dataset_root


def test_evaluate_builds_vst_datamodule_with_ram_bounded_num_workers() -> None:
    """The datamodule eval instantiates carries the RAM-bounded worker default.

    ``num_workers`` is applied per dataloader, so a run holding both a test and a
    predict loader doubles the live worker count. Lance workers are ~1.4 GB each,
    and the previous default of 11 put a 32 GB host past its RAM plus swap
    (#1916).

    Instantiates the datamodule the way ``evaluate`` does rather than asserting
    the composed dict, so the default is checked where it is consumed. Composed
    explicitly rather than via ``cfg_eval``: that fixture pins ``num_workers``
    itself, so nothing else here would catch the default drifting back up.
    """
    GlobalHydra.instance().clear()
    try:
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            cfg = compose(
                config_name="eval.yaml",
                return_hydra_config=True,
                overrides=[
                    "datamodule=surge_simple",
                    "model=ffn",
                    "trainer=cpu",
                    "ckpt_path=.",
                ],
            )
        HydraConfig().set_config(cfg)
        datamodule = instantiate(cfg.datamodule)
    finally:
        GlobalHydra.instance().clear()
    assert datamodule.num_workers == 4
