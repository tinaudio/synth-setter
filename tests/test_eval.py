"""Tests for the ``synth-setter-eval`` CLI entrypoint.

Every test composes a Hydra ``cfg`` and drives the in-process ``evaluate(cfg)``
entrypoint. Helper-level unit tests live in the sibling ``test_eval_*`` modules:
postprocessing argv in ``test_eval_postprocessing``, metric IO in
``test_eval_metrics``, and R2 upload / CLI e2e in ``test_eval_upload``.
``tests/_meta/test_entrypoint_test_modules.py`` enforces that no private
``synth_setter.cli`` helper is imported here.
"""

import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Literal, NamedTuple
from unittest.mock import MagicMock, patch

import lance
import numpy as np
import pytest
import torch
import wandb
from click.testing import CliRunner
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, open_dict
from omegaconf.errors import InterpolationResolutionError, MissingMandatoryValue
from pedalboard.io import AudioFile

from synth_setter.cli.eval import evaluate
from synth_setter.cli.migrate_checkpoint import main
from synth_setter.cli.train import train
from synth_setter.data.pyfdn_param_spec import PYFDN_N8_MONO_PARAM_SPEC
from synth_setter.data.vst import plugin_state_paths
from synth_setter.data.vst.shapes import AUDIO_FIELD
from synth_setter.models.components.embed_pool import EmbeddingPool
from synth_setter.models.components.pretrained_encoder import (
    ClapAudioEncoder,
    PretrainedConditioningEncoder,
)
from synth_setter.models.components.same_encoder import SameAudioEncoder
from synth_setter.models.components.vector_projection import VectorProjection
from synth_setter.models.slap_module import SLAPModule
from synth_setter.models.vst_ff_module import VSTFeedForwardModule
from synth_setter.pipeline.data.matpac_plus import MATPAC_PLUS_FRONTEND
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig
from synth_setter.pipeline.spec_io import write_spec_to_path
from synth_setter.utils.utils import register_resolvers
from synth_setter.workspace import operator_workspace
from tests.conftest import (
    _render_smoke_train_subprocess,
    assert_clap_preserves_resampler_output,
    assert_log_per_param_mse_wired,
    augment_lance_splits_with_embedding,
    augment_lance_splits_with_embeddings,
    augment_lance_splits_with_same,
    augment_lance_splits_with_ssondo,
    build_surge_xt_embedding_train_cfg,
    flatten_lance_embedding_column,
)
from tests.helpers.eval_fakes import (
    COMPUTE_AUDIO_METRICS_FRAGMENT,
    FAKE_METRICS_CSV,
    fake_postprocessing_subprocess,
)
from tests.helpers.generic_launcher import run_generic_launcher_command
from tests.helpers.lance_fixtures import write_blob_audio_corpus
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


@pytest.mark.slow
def test_generic_launcher_runs_workflow_default_eval_entrypoint(tmp_path: Path) -> None:
    """Run workflow-default evaluation through the launcher and headless wrapper.

    :param tmp_path: Output root for launcher and worker subprocesses.
    """
    repo_root = Path(__file__).parents[1]
    worker_command = shlex.join(
        [
            "exec",
            "src/synth_setter/scripts/run-linux-vst-headless.sh",
            "synth-setter-eval",
            "experiment=surge/ffn_simple",
            "trainer=cpu",
            "callbacks=none",
            "~logger",
            "ckpt_path=null",
            "datamodule=surge_lance",
            "+datamodule.fake=true",
            "datamodule.batch_size=1",
            "datamodule.num_workers=0",
            "model.net.d_model=16",
            "model.net.n_heads=1",
            "model.net.n_layers=1",
            "model.net.patch_size=128",
            "model.net.patch_stride=64",
            "model.compile=false",
            "mode=test",
            "+trainer.limit_test_batches=1",
            f"hydra.run.dir={tmp_path / 'eval'}",
        ]
    )

    result = run_generic_launcher_command(tmp_path, worker_command, repo_root)

    assert result.returncode == 0, result.stderr
    assert "test/param_mse" in result.stdout


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_evaluate_slap_checkpoint_reports_finite_loss(
    cfg_slap_train_lance: DictConfig,
) -> None:
    """Reload a GPU-trained SLAP checkpoint through the public eval path.

    :param cfg_slap_train_lance: Tiny paired Lance training configuration.
    """
    with open_dict(cfg_slap_train_lance):
        cfg_slap_train_lance.trainer.accelerator = "gpu"
        cfg_slap_train_lance.trainer.devices = 1
        cfg_slap_train_lance.trainer.precision = "32-true"
    HydraConfig().set_config(cfg_slap_train_lance)
    _, train_objects = train(cfg_slap_train_lance)
    checkpoint_path = train_objects["trainer"].checkpoint_callback.best_model_path

    GlobalHydra.instance().clear()
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                "datamodule=surge_lance",
                "model=slap",
                "callbacks=none",
                "trainer=gpu",
                "synth=surge_4",
            ],
        )
    with open_dict(cfg):
        cfg.paths = deepcopy(cfg_slap_train_lance.paths)
        cfg.paths.output_dir = str(Path(cfg.paths.output_dir) / "eval")
        cfg.datamodule = deepcopy(cfg_slap_train_lance.datamodule)
        cfg.model = deepcopy(cfg_slap_train_lance.model)
        cfg.logger = None
        cfg.ckpt_path = checkpoint_path
        cfg.mode = "test"
        cfg.trainer.limit_test_batches = 1
        cfg.trainer.enable_model_summary = False
    HydraConfig().set_config(cfg)

    metric_dict, object_dict = evaluate(cfg)
    GlobalHydra.instance().clear()

    assert isinstance(object_dict["model"], SLAPModule)
    assert torch.isfinite(metric_dict["loss/test/total_loss"])
    metrics_path = Path(cfg.paths.output_dir) / "metrics" / "metrics.json"
    assert math.isfinite(json.loads(metrics_path.read_text())["loss/test/total_loss"])


@pytest.mark.slow
def test_evaluate_pyfdn_householder_checkpoint_logs_param_mse(
    cfg_pyfdn_train: DictConfig,
) -> None:
    """Evaluate a real checkpoint through the fixed-Householder pyFDN config.

    :param cfg_pyfdn_train: One-step fixed-Householder pyFDN configuration.
    """
    HydraConfig().set_config(cfg_pyfdn_train)
    train(cfg_pyfdn_train)
    checkpoint = Path(cfg_pyfdn_train.paths.output_dir) / "checkpoints" / "last.ckpt"
    with open_dict(cfg_pyfdn_train):
        cfg_pyfdn_train.ckpt_path = str(checkpoint)
        cfg_pyfdn_train.mode = "test"
        cfg_pyfdn_train.trainer.limit_test_batches = 1

    HydraConfig().set_config(cfg_pyfdn_train)
    metrics, _ = evaluate(cfg_pyfdn_train)

    assert torch.isfinite(metrics["test/param_mse"])


def test_evaluate_without_checkpoint_override_raises_missing_mandatory_value() -> None:
    """The evaluation entrypoint rejects a config without checkpoint provenance."""
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=["experiment=surge/eval_flow_sketch_nsynth"],
        )
    with open_dict(cfg):
        cfg.ckpt_path = "???"
    HydraConfig().set_config(cfg)

    with pytest.raises(MissingMandatoryValue, match="ckpt_path"):
        evaluate(cfg)


def _compose_sketch_cfg_eval(
    cfg_train_sketch_lance: DictConfig,
    experiment: str = "surge/flow_sketch_prelim",
) -> DictConfig:
    """Compose the toy pooled-sketch evaluation configuration.

    :param cfg_train_sketch_lance: Fixture providing paths and pooled Lance splits.
    :param experiment: Experiment config group exercised by evaluation.
    :returns: Evaluation config with sketch guidance disabled initially.
    """
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                f"experiment={experiment}",
                "datamodule=surge_lance",
                "synth=surge_4",
                "conditioning=m2l",
                "trainer=cpu",
                "callbacks=none",
            ],
        )
    with open_dict(cfg):
        cfg.paths.root_dir = cfg_train_sketch_lance.paths.root_dir
        cfg.paths.output_dir = cfg_train_sketch_lance.paths.output_dir
        cfg.paths.log_dir = cfg_train_sketch_lance.paths.log_dir
        cfg.logger = None
        cfg.ckpt_path = None
        cfg.mode = "test"
        cfg.datamodule.dataset_root = cfg_train_sketch_lance.datamodule.dataset_root
        cfg.datamodule.download_dataset_root_uri = None
        cfg.datamodule.download_dataset_row_limit = None
        cfg.datamodule.batch_size = 2
        cfg.datamodule.num_workers = 0
        cfg.datamodule.persistent_workers = False
        cfg.datamodule.pin_memory = False
        cfg.model.compile = False
        cfg.model.validation_sample_steps = 2
        cfg.model.test_sample_steps = 2
        cfg.model.test_sketch_cfg_strength = 0.0
        cfg.model.vector_field.num_layers = 1
        cfg.model.vector_field.d_model = 32
        cfg.model.vector_field.d_ff = 32
        cfg.model.vector_field.projection.num_tokens = 8
        cfg.trainer.fast_dev_run = True
        cfg.trainer.precision = "32-true"
    return cfg


def _save_nonzero_sketch_checkpoint(cfg: DictConfig, checkpoint_path: Path) -> None:
    """Save a checkpoint whose sketch branch differs from its unconditional branch.

    :param cfg: Sketch-conditioned model configuration.
    :param checkpoint_path: Checkpoint destination.
    """
    model = instantiate(cfg.model)
    torch.manual_seed(11)
    with torch.no_grad():
        for projection in model.sketch_tokens.projections.values():
            projection.weight.normal_()
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.strategy.connect(model)
    trainer.save_checkpoint(checkpoint_path)


def test_evaluate_flow_sketch_cfg_ablation_validate_returns_finite_metric(
    cfg_train_sketch_lance: DictConfig,
) -> None:
    """Consume the ablation config through the real evaluation entrypoint.

    :param cfg_train_sketch_lance: Fixture providing pooled-sketch Lance splits.
    """
    cfg = _compose_sketch_cfg_eval(
        cfg_train_sketch_lance,
        experiment="surge/flow_sketch_cfg_ablation",
    )
    cfg.mode = "validate"

    HydraConfig().set_config(cfg)
    try:
        metric_dict, object_dict = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert torch.isfinite(metric_dict["val/param_mse"])
    assert object_dict["model"].sketch_tokens is not None
    assert object_dict["datamodule"].sketch_controls is not None
    assert cfg.consumed_train_config_id == "flow_sketch_prelim"
    assert cfg.evaluation.render_vst is True
    assert cfg.evaluation.compute_metrics is True


def test_evaluate_flow_sketch_prelim_routes_independent_sketch_cfg_strength(
    cfg_train_sketch_lance: DictConfig,
) -> None:
    """The eval entrypoint routes sketch CFG independently into test sampling.

    :param cfg_train_sketch_lance: Fixture providing pooled-sketch Lance splits.
    """
    cfg = _compose_sketch_cfg_eval(cfg_train_sketch_lance)
    checkpoint_path = Path(cfg.paths.output_dir) / "sketch-cfg.ckpt"
    _save_nonzero_sketch_checkpoint(cfg, checkpoint_path)
    cfg.ckpt_path = str(checkpoint_path)

    torch.manual_seed(17)
    HydraConfig().set_config(cfg)
    try:
        sketch_disabled_metrics, objects = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    cfg.model.test_sketch_cfg_strength = 8.0
    torch.manual_seed(17)
    HydraConfig().set_config(cfg)
    try:
        sketch_guided_metrics, _ = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    sketch_disabled = sketch_disabled_metrics["test/param_mse"]
    sketch_guided = sketch_guided_metrics["test/param_mse"]
    assert torch.isfinite(sketch_disabled)
    assert torch.isfinite(sketch_guided)
    assert sketch_disabled != sketch_guided
    assert objects["model"].sketch_tokens is not None
    assert objects["datamodule"].sketch_controls is not None
    assert objects["datamodule"].sketch_controls.num_frames == 32


def test_eval_faust_render_group_resolves_production_renderer_contract() -> None:
    """The eval operator config accepts the production brightOrgan render group."""
    try:
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            cfg = compose(
                config_name="eval.yaml",
                overrides=[
                    "datamodule=torchsynth",
                    "model=ffn",
                    "trainer=cpu",
                    "ckpt_path=.",
                    "synth=faust_bright_organ",
                    "render=faust",
                ],
            )
        render = RenderConfig.from_cfg_nodes(cfg.render, cfg.synth)
    finally:
        GlobalHydra.instance().clear()

    assert render.renderer_backend == "dawdreamer_faust"
    assert render.plugin_path == "faust"
    assert render.plugin_reload_cadence == "render"
    assert render.gui_toggle_cadence == "never"
    assert render.param_spec_name == "faust_bright_organ"


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
# 4 s of audio at a smoke-test rate: the spec draws each row's note window across a 4 s
# range, so a shorter buffer would start most notes past the end and render silence.
_TORCHSYNTH_AUDIO_OVERRIDES = (
    "datamodule.sample_rate=22050",
    "datamodule.signal_length=88200",
)
# Pitch and note timing now vary per row, so a single-row overfit no longer transfers to
# the val split; the checkpoint has to see enough rows to beat the untrained baseline.
_TORCHSYNTH_TRAIN_ROWS = 64
_TORCHSYNTH_BATCH_SIZE = 8
_TORCHSYNTH_EPOCHS = 10
_TORCHSYNTH_LR = 0.001


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
    # Mirrors the shipped jobs/predict scripts: non-VAE checkpoints select the
    # per-param MSE callback group alongside eval_surge; the VAE lane omits it.
    callbacks = (
        "callbacks=eval_surge"
        if case.experiment == "vae_full"
        else "callbacks=[eval_surge,log_per_param_mse]"
    )
    return [
        sys.executable,
        "-m",
        "synth_setter.cli.eval",
        f"experiment=surge/{case.experiment}",
        f"datamodule={case.datamodule}",
        callbacks,
        "mode=predict",
        "trainer=cpu",
        "logger=wandb",
        "logger.wandb.offline=true",
        f"ckpt_path={checkpoint}",
        f"datamodule.root={audio_root}",
        "datamodule.stats_file=null",
        "datamodule.batch_size=1",
        "datamodule.num_workers=0",
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
                f"experiment=surge/{case.experiment}",
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
    assert type(prediction) is torch.Tensor
    assert type(target_audio) is torch.Tensor
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

    # Drop WANDB_SERVICE: earlier in-process offline runs leave a service token in
    # the session env, and connecting to that dead socket aborts wandb.init (#2564).
    subprocess_env = {key: value for key, value in os.environ.items() if key != "WANDB_SERVICE"}
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
        env=subprocess_env,
    )

    assert result.returncode == 0, result.stderr
    _assert_audio_prediction_artifacts(output_dir)


@pytest.mark.requires_vst
@pytest.mark.slow
def test_audio_dataset_predict_no_params_renders_against_dataset_audio(tmp_path: Path) -> None:
    """Render a parameterless prediction against its staged dataset audio.

    :param tmp_path: Isolated input, checkpoint, and output directories.
    """
    case = _AUDIO_PREDICTION_CASES[0]
    audio_root = tmp_path / case.datamodule
    audio_root.mkdir()
    _write_audio_prediction_fixture(audio_root / case.filename)
    checkpoint = tmp_path / f"{case.experiment}.ckpt"
    _save_audio_prediction_checkpoint(case, checkpoint)
    output_dir = tmp_path / "no-params-output"
    args = _audio_prediction_cli_args(
        case,
        checkpoint=checkpoint,
        audio_root=audio_root,
        output_dir=output_dir,
    )
    args.extend(
        (
            "render=vst",
            "evaluation.render_vst=true",
            "evaluation.compute_metrics=false",
            "evaluation.no_params=true",
            "evaluation.rerender_target=false",
        )
    )
    subprocess_env = {key: value for key, value in os.environ.items() if key != "WANDB_SERVICE"}

    result = subprocess.run(  # noqa: S603 — argv contains only test-owned paths
        args,
        capture_output=True,
        text=True,
        timeout=300,
        env=subprocess_env,
    )

    assert result.returncode == 0, result.stderr
    prediction_dir = output_dir / "predictions"
    assert not (prediction_dir / "target-params-0.pt").exists()
    staged_target = torch.load(
        prediction_dir / "target-audio-0.pt", map_location="cpu", weights_only=True
    )[0]
    sample_dir = output_dir / "audio" / "sample_0"
    with AudioFile(str(sample_dir / "target.wav")) as target_file:
        rendered_target = torch.from_numpy(target_file.read(target_file.frames))
    assert (sample_dir / "pred.wav").is_file()
    torch.testing.assert_close(rendered_target, staged_target, atol=1e-4, rtol=0)


def _compose_torchsynth_train_cfg(tmp_path: Path) -> DictConfig:
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
                f"+trainer.max_epochs={_TORCHSYNTH_EPOCHS}",
                "+trainer.limit_val_batches=1",
                "trainer.val_check_interval=1.0",
                "trainer.check_val_every_n_epoch=1",
                "datamodule.resample_train_per_epoch=false",
                *_TORCHSYNTH_AUDIO_OVERRIDES,
                f"datamodule.train_val_test_sizes=[{_TORCHSYNTH_TRAIN_ROWS},1,1]",
                f"datamodule.batch_size={_TORCHSYNTH_BATCH_SIZE}",
                "datamodule.num_workers=0",
                "model.compile=false",
                f"model.optimizer.lr={_TORCHSYNTH_LR}",
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
        for baseline_batch in dataloader:
            baseline_audio = baseline_batch["audio"]
            baseline_params = baseline_batch["params"]
            # VSTFeedForwardModule has no forward(); predictions go through its net.
            squared_error = torch.nn.functional.mse_loss(
                baseline_model.net(baseline_audio), baseline_params, reduction="sum"
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
                *_TORCHSYNTH_AUDIO_OVERRIDES,
                "datamodule.train_val_test_sizes=[2,32,2]",
                f"datamodule.batch_size={_TORCHSYNTH_BATCH_SIZE}",
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
    train_cfg = _compose_torchsynth_train_cfg(tmp_path)
    initial_loss = _torchsynth_initial_loss(train_cfg, stage="fit", split="train")
    HydraConfig().set_config(train_cfg)
    try:
        train_metrics, train_objects = train(train_cfg)
    finally:
        GlobalHydra.instance().clear()

    train_loss = train_metrics["train/loss_epoch"].item()
    assert math.isfinite(train_loss)
    assert train_loss < initial_loss

    checkpoint = Path(train_objects["trainer"].checkpoint_callback.best_model_path)
    assert checkpoint.is_file()
    eval_cfg = _compose_torchsynth_eval_cfg(tmp_path, checkpoint)
    initial_val_loss = _torchsynth_initial_loss(eval_cfg, stage="validate", split="val")
    HydraConfig().set_config(eval_cfg)
    try:
        metric_dict, eval_objects = evaluate(eval_cfg)
    finally:
        GlobalHydra.instance().clear()

    val_loss = metric_dict["val/param_mse"]
    assert torch.isfinite(val_loss)
    assert val_loss < initial_val_loss * (1 - _TORCHSYNTH_MIN_RELATIVE_VAL_IMPROVEMENT)
    val_dataloader = eval_objects["datamodule"].val_dataloader()
    assert val_dataloader.num_workers == 0
    eval_batch = next(iter(val_dataloader))
    assert torch.isfinite(eval_batch["audio"]).all()


@pytest.mark.slow
def test_eval_torchsynth_flow_logs_grouped_per_param_metrics_by_default(
    cfg_torchsynth_flow_audio_train: DictConfig,
) -> None:
    """The eval entrypoint publishes grouped-swap errors for the active synth spec.

    :param cfg_torchsynth_flow_audio_train: Composed tiny production flow configuration.
    """
    cfg = cfg_torchsynth_flow_audio_train
    with open_dict(cfg):
        cfg.mode = "validate"
        cfg.ckpt_path = None
        cfg.logger = None
        cfg.trainer.limit_val_batches = 1

    HydraConfig().set_config(cfg)
    try:
        metric_dict, _ = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert torch.isfinite(metric_dict["per_param_mse_number_group_swap/adsr_1.attack"]).all()


@pytest.mark.slow
def test_eval_torchsynth_clap_online_validates_real_offline_backbone(
    cfg_torchsynth_clap_online_train: DictConfig,
) -> None:
    """The eval entrypoint validates raw audio through a real tiny CLAP model.

    :param cfg_torchsynth_clap_online_train: Offline tiny-CLAP production configuration.
    """
    cfg = cfg_torchsynth_clap_online_train
    with open_dict(cfg):
        cfg.mode = "validate"
        cfg.ckpt_path = None
        cfg.logger = None
        cfg.trainer.limit_val_batches = 1

    HydraConfig().set_config(cfg)
    try:
        metric_dict, object_dict = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert torch.isfinite(metric_dict["val/param_mse"])
    model = object_dict["model"]
    encoder = model.encoder
    assert isinstance(encoder, PretrainedConditioningEncoder)
    assert isinstance(encoder.backbone, ClapAudioEncoder)
    assert isinstance(encoder.head, VectorProjection)
    assert encoder.head.n_conditioning_outputs == len(model.vector_field.layers) == 2
    assert_clap_preserves_resampler_output(encoder.backbone, cfg.model.encoder.backbone.checkpoint)


@pytest.mark.slow
def test_eval_torchsynth_same_online_validates_real_offline_backbone(
    cfg_torchsynth_same_online_train: DictConfig,
) -> None:
    """The eval entrypoint validates raw audio through a real tiny SAME model.

    :param cfg_torchsynth_same_online_train: Offline tiny-SAME production configuration.
    """
    cfg = cfg_torchsynth_same_online_train
    with open_dict(cfg):
        cfg.mode = "validate"
        cfg.ckpt_path = None
        cfg.logger = None
        cfg.trainer.limit_val_batches = 1

    HydraConfig().set_config(cfg)
    try:
        metric_dict, object_dict = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert torch.isfinite(metric_dict["val/param_mse"])
    model = object_dict["model"]
    encoder = model.encoder
    assert isinstance(encoder, PretrainedConditioningEncoder)
    assert isinstance(encoder.backbone, SameAudioEncoder)
    assert isinstance(encoder.head, EmbeddingPool)
    assert encoder.head.n_conditioning_outputs == len(model.vector_field.layers) == 2


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
    the metric dict. The production path also carries the datamodule's ``mel``
    entry through the oracle, so a stale model-batch key fails this test.

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
                "synth=surge_4",
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


@pytest.mark.slow
def test_evaluate_test_mps_flow_config_runs_one_cpu_batch(tmp_path: Path) -> None:
    """The MPS flow experiment also runs through the shared eval entrypoint.

    :param tmp_path: Pinned as Hydra output and log directory.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=surge/test-mps-flow",
                "trainer=cpu",
                "mode=test",
                "model.test_sample_steps=1",
            ],
        )

    with open_dict(cfg):
        cfg.paths.root_dir = str(operator_workspace())
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.datamodule.fake = True
        cfg.datamodule.batch_size = 1
        cfg.datamodule.num_workers = 0
        cfg.datamodule.use_saved_mean_and_variance = False
        cfg.trainer.limit_test_batches = 1
        cfg.ckpt_path = None
        cfg.logger = None

    HydraConfig().set_config(cfg)
    try:
        metric_dict, _ = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert torch.isfinite(metric_dict["test/param_mse"])


# slow: dominates the inner loop (#2274 profile; flow_simple case is #2280).
@pytest.mark.slow
@pytest.mark.parametrize("experiment", sorted(_FLOW_LAD_EVAL_OVERRIDES))
def test_evaluate_flow_simple_test_mode_logs_param_mse_best_swap(
    tmp_path: Path, experiment: str
) -> None:
    """``mode=test`` through both flow configs logs ``test/param_mse_best_swap`` beside the MSE.

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
    param_mse_best_swap = metric_dict["test/param_mse_best_swap"]
    assert torch.isfinite(param_mse_best_swap)
    assert param_mse_best_swap.item() <= metric_dict["test/param_mse"].item() + 1e-6


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

    assert math.isfinite(test_metric_dict["test/param_mse"].item())
    assert (
        abs(train_metric_dict["test/param_mse"].item() - test_metric_dict["test/param_mse"].item())
        < 0.001
    )


def test_evaluate_loads_compiled_cpu_training_checkpoint(
    tmp_path: Path,
    cfg_train: DictConfig,
    cfg_eval: DictConfig,
) -> None:
    """Uncompiled CPU evaluation loads a checkpoint written by compiled training.

    :param tmp_path: Shared training and evaluation output directory.
    :param cfg_train: Tiny TorchSynth CPU training configuration.
    :param cfg_eval: Matching TorchSynth CPU evaluation configuration.
    """
    for cfg in (cfg_train, cfg_eval):
        with open_dict(cfg):
            cfg.datamodule.signal_length = 512
            cfg.model.net.channels = 2
            cfg.model.net.encoder_blocks = 1
            cfg.model.net.hidden_dim = 8
            cfg.model.net.norm = "ln"
            cfg.model.net.trunk_blocks = 1
    with open_dict(cfg_train):
        cfg_train.model.compile = True
        cfg_train.test = False
        cfg_train.trainer.limit_train_batches = 1
        cfg_train.trainer.limit_val_batches = 1
    with open_dict(cfg_eval):
        cfg_eval.trainer.limit_test_batches = 1

    HydraConfig().set_config(cfg_train)
    train(cfg_train)

    checkpoint_path = tmp_path / "checkpoints" / "last.ckpt"
    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(checkpoint_path)
    HydraConfig().set_config(cfg_eval)
    metrics, _ = evaluate(cfg_eval)

    assert math.isfinite(metrics["test/param_mse"].item())


def test_evaluate_legacy_wrapped_checkpoint_hints_migration_cli_which_recovers(
    tmp_path: Path,
    cfg_train: DictConfig,
    cfg_eval: DictConfig,
) -> None:
    """A legacy ``_orig_mod`` checkpoint fails with the migration command, which fixes it.

    :param tmp_path: Shared training and evaluation output directory.
    :param cfg_train: Tiny TorchSynth CPU training configuration.
    :param cfg_eval: Matching TorchSynth CPU evaluation configuration.
    """
    for cfg in (cfg_train, cfg_eval):
        with open_dict(cfg):
            cfg.datamodule.signal_length = 512
            cfg.model.net.channels = 2
            cfg.model.net.encoder_blocks = 1
            cfg.model.net.hidden_dim = 8
            cfg.model.net.norm = "ln"
            cfg.model.net.trunk_blocks = 1
    with open_dict(cfg_train):
        cfg_train.test = False
        cfg_train.trainer.limit_train_batches = 1
        cfg_train.trainer.limit_val_batches = 1
    with open_dict(cfg_eval):
        cfg_eval.trainer.limit_test_batches = 1

    HydraConfig().set_config(cfg_train)
    train(cfg_train)

    checkpoint_path = tmp_path / "checkpoints" / "last.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["state_dict"] = {
        (f"net._orig_mod.{key.removeprefix('net.')}" if key.startswith("net.") else key): value
        for key, value in checkpoint["state_dict"].items()
    }
    torch.save(checkpoint, checkpoint_path)

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(checkpoint_path)
    HydraConfig().set_config(cfg_eval)
    with pytest.raises(RuntimeError, match="synth-setter-migrate-checkpoint"):
        evaluate(cfg_eval)

    migrated_path = tmp_path / "checkpoints" / "last.migrated.ckpt"
    result = CliRunner().invoke(main, [str(checkpoint_path), str(migrated_path)])
    assert result.exit_code == 0, result.output
    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(migrated_path)
    metrics, _ = evaluate(cfg_eval)

    assert math.isfinite(metrics["test/param_mse"].item())


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

    assert math.isfinite(val_metric_dict["val/param_mse"].item())
    assert (
        abs(train_metric_dict["val/param_mse"].item() - val_metric_dict["val/param_mse"].item())
        < 0.001
    )


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
            overrides=["experiment=surge/fake_oracle", f"synth={param_spec_name}", f"mode={mode}"]
            + ([f"datamodule={datamodule}"] if datamodule else [])
            + (["render=pyfdn"] if param_spec_name == "pyfdn_n8_mono" else []),
        )
    with open_dict(cfg):
        cfg.paths.root_dir = str(operator_workspace())
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.datamodule.dataset_root = str(dataset_root)
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
        # Render group is null on fake_oracle; set it inline for fake VST datasets.
        if param_spec_name != "pyfdn_n8_mono":
            cfg.render = RenderConfig.model_validate(
                {
                    "synth": {
                        "name": param_spec_name,
                        "param_spec_name": param_spec_name,
                        "plugin_state_path": str(plugin_state_paths[param_spec_name]),
                        "plugin_path": "plugins/fake.vst3",
                        "synth_version": "1.3.4",
                    },
                    "sample_rate": 44100,
                    "channels": 2,
                    "velocity": 100,
                    "signal_duration_seconds": 4.0,
                    "min_loudness": -55.0,
                    "samples_per_render_batch": 1,
                    "samples_per_shard": 5,
                    "plugin_reload_cadence": "render",
                    "gui_toggle_cadence": "never",
                }
            ).model_dump(mode="json")
    return cfg


def test_evaluate_row_limited_file_uri_hydration_without_txids(
    cfg_train_lance: DictConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The eval entrypoint consumes latest-snapshot row-limited hydration.

    :param cfg_train_lance: Composed Lance config supplying the source dataset.
    :param tmp_path: Parent of the fresh local hydration destination.
    :param monkeypatch: Replaces only the separately tested rclone sidecar boundary.
    """
    source = Path(cfg_train_lance.datamodule.dataset_root)
    destination = tmp_path / "row-limited-data"

    def copy_stats(_source_uri: str, dest_path: Path, exclude: str | None = None) -> None:
        del _source_uri, exclude
        dest_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "stats.npz", dest_path / "stats.npz")

    monkeypatch.setattr(
        "synth_setter.data.vst_datamodule.r2_io.download_dir_no_overwrite",
        copy_stats,
    )
    ensure_r2 = MagicMock(side_effect=AssertionError("R2 preflight called"))
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", ensure_r2)
    cfg = _compose_fake_oracle_eval_cfg(
        tmp_path,
        destination,
        mode="test",
        param_spec_name=str(cfg_train_lance.datamodule.param_spec_name),
        datamodule="surge_lance",
    )
    with open_dict(cfg):
        cfg.datamodule.download_dataset_root_uri = source.as_uri()
        cfg.datamodule.download_dataset_row_limit = 2

    HydraConfig().set_config(cfg)
    metric_dict, object_dict = evaluate(cfg)

    ensure_r2.assert_not_called()
    assert torch.isfinite(metric_dict["test/param_mse"])
    datamodule = object_dict["datamodule"]
    assert datamodule.high_memory_materialization is False
    datamodule.setup("test")
    try:
        assert len(datamodule.test_dataset) == 2
    finally:
        datamodule.teardown("test")


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
    fake_run = fake_postprocessing_subprocess()

    def _write_current_and_unsupported_metrics(
        args: list[str],
        **kwargs: object,
    ) -> None:
        fake_run(args, **kwargs)
        if any(COMPUTE_AUDIO_METRICS_FRAGMENT in arg for arg in args):
            metrics_dir = Path(args[args.index("-m") + 3])
            (metrics_dir / "aggregated_metrics_shuffled.csv").write_text(
                ",mean,std\nmss,1.0,0.2\n"
            )
            (metrics_dir / "shuffle_permutation.csv").write_text("dest_idx,src_idx\n0,1\n1,0\n")

    monkeypatch.setattr(
        "synth_setter.cli.eval.subprocess.run",
        _write_current_and_unsupported_metrics,
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
    assert not any("shuffle" in key for key in metric_dict)


@pytest.mark.slow
def test_evaluate_predict_mode_pyfdn_renders_finite_audio_end_to_end(
    tmp_path: Path,
) -> None:
    """The eval entrypoint renders a pyFDN prediction through real post-processing.

    :param tmp_path: Hydra output root and two-row Lance predict dataset.
    """
    from tests.helpers.lance_fixtures import write_lance_shard

    dataset_root = tmp_path / "pyfdn-data"
    dataset_root.mkdir()
    params, notes = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(17))
    encoded = PYFDN_N8_MONO_PARAM_SPEC.encode(params, notes)
    write_lance_shard(
        dataset_root / "test.lance",
        {
            "audio": np.zeros((2, 1, 176_400), dtype=np.float32),
            "mel_spec": np.zeros((2, 1, 128, 401), dtype=np.float32),
            "param_array": np.repeat(encoded[None, :], 2, axis=0),
        },
    )
    cfg = _compose_fake_oracle_eval_cfg(
        tmp_path,
        dataset_root,
        mode="predict",
        param_spec_name="pyfdn_n8_mono",
        datamodule="pyfdn",
    )
    with open_dict(cfg):
        cfg.datamodule.predict_file = str(dataset_root / "test.lance")
        cfg.datamodule.use_saved_mean_and_variance = False
        cfg.evaluation.compute_metrics = False
        cfg.evaluation.rerender_target = True

    HydraConfig().set_config(cfg)
    try:
        evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    with AudioFile(str(tmp_path / "audio" / "sample_0" / "pred.wav")) as audio_file:
        rendered = audio_file.read(audio_file.frames)
    assert rendered.shape == (1, 176_400)
    assert np.isfinite(rendered).all()


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
    """An unregistered ``synth.param_spec_name`` fails during model resolution.

    The model width resolver rejects an unknown spec before model construction or
    dataset access.

    :param tmp_path: Pinned as Hydra ``output_dir`` / ``log_dir``; the dataset root
        points at a nonexistent subdirectory that is never read.
    """
    cfg = _compose_fake_oracle_eval_cfg(tmp_path, tmp_path / "missing-datasets", mode="validate")
    with open_dict(cfg):
        cfg.synth.param_spec_name = "does_not_exist"

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


@pytest.mark.parametrize("synth_group", ["surge_simple", "surge_xt"])
def test_eval_synth_group_exposes_postprocessing_keys(synth_group: str) -> None:
    """Composing ``synth=<group>`` into eval exposes the three keys postprocessing reads.

    ``_run_predict_postprocessing`` resolves identity via
    ``RenderConfig.from_cfg_nodes`` to build the renderer argv. This composition
    test pins that both shipped synth groups supply the whole identity, so a
    future rename in a ``synth/*.yaml`` surfaces here rather than mid-eval.

    :param synth_group: Synth config group composed into the eval cfg.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=surge/fake_oracle",
                f"synth={synth_group}",
                "render=vst",
            ],
        )
    try:
        assert cfg.synth.param_spec_name
        assert cfg.synth.plugin_state_path
        assert cfg.synth.plugin_path
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
    """Run the workflow's W&B checkpoint contract through real prediction.

    The full W&B checkpoint contract end to end: train a real checkpoint, publish it to
    ``tinaudio/synth-setter-citest``, then pass ``ckpt_path=${wandb:...}`` explicitly and run
    ``evaluate()`` in predict mode.
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
    assert object_dict["datamodule"].val_num_workers == 0

    param_mse = metric_dict["val/param_mse"]
    assert isinstance(param_mse, torch.Tensor)
    assert param_mse.item() == 0.0
    assert metric_dict["per_param_mse/a_amp_eg_attack"].item() == 0.0


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
                    "synth=surge_simple",
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


def _compose_fake_t5gemma_ffn_eval_cfg(
    output_dir: Path, param_spec_name: str, checkpoint_path: Path
) -> DictConfig:
    """Compose the eval-side counterpart of the feed-forward T5Gemma train cfg.

    :param output_dir: Pinned as Hydra ``output_dir`` / ``log_dir``.
    :param param_spec_name: Param spec driving model width and callback labels.
    :param checkpoint_path: Train-produced checkpoint loaded by ``evaluate``.
    :returns: Resolved ``mode=validate`` DictConfig over synthetic T5Gemma batches.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                "model=vst_ffn",
                f"synth={param_spec_name}",
                "conditioning=t5gemma",
                "datamodule=surge_lance",
                "trainer=cpu",
                "mode=validate",
                "model.compile=false",
                # eval's trainer has no max_steps for the scheduler interpolation.
                "model.scheduler=null",
                "model.net.d_model=16",
                "model.net.n_heads=1",
                "model.net.n_layers=1",
                "model.net.patch_size=128",
                "model.net.patch_stride=64",
            ],
        )
    with open_dict(cfg):
        cfg.paths.root_dir = str(operator_workspace())
        cfg.paths.output_dir = str(output_dir)
        cfg.paths.log_dir = str(output_dir)
        cfg.datamodule.fake = True
        cfg.datamodule.dataset_root = str(output_dir)
        cfg.datamodule.batch_size = 2
        cfg.datamodule.num_workers = 0
        cfg.trainer.limit_val_batches = 1
        cfg.ckpt_path = str(checkpoint_path)
        cfg.logger = None
    return cfg


def test_evaluate_validate_mode_feed_forward_cached_conditioning_ckpt_returns_finite_metric(
    tmp_path: Path,
    param_spec_name: str,
) -> None:
    """``model=vst_ffn conditioning=t5gemma`` composes in eval and validates its checkpoint.

    Feed-forward cached conditioning is reachable from the eval entrypoint, so the
    checkpoint a training run produces is loaded back through ``evaluate`` here rather
    than only inside ``tests/test_train.py``. Synthetic batches at the profile's shapes
    keep the regression on the CPU-fast lane.

    :param tmp_path: Pinned as Hydra ``output_dir`` / ``log_dir`` for both stages.
    :param param_spec_name: Param spec driving model width and callback labels.
    """
    cfg_train = build_surge_xt_embedding_train_cfg(
        tmp_path,
        tmp_path,
        param_spec_name=param_spec_name,
        conditioning="t5gemma",
        architecture="feed_forward",
        fake=True,
    )
    HydraConfig().set_config(cfg_train)
    try:
        train(cfg_train)
    finally:
        GlobalHydra.instance().clear()
    checkpoint_path = tmp_path / "checkpoints" / "last.ckpt"
    assert checkpoint_path.is_file()

    cfg_eval = _compose_fake_t5gemma_ffn_eval_cfg(
        tmp_path / "evaluation", param_spec_name, checkpoint_path
    )
    HydraConfig().set_config(cfg_eval)
    try:
        metric_dict, object_dict = evaluate(cfg_eval)
    finally:
        GlobalHydra.instance().clear()

    assert isinstance(object_dict["model"], VSTFeedForwardModule)
    assert math.isfinite(metric_dict["val/param_mse"].item())


_EMBEDDING_CONDITIONING_PROFILES = ("m2l", "clap")


def _assert_conditioning_train_validate_finite(
    tmp_path: Path, dataset_root: Path, param_spec_name: str, conditioning: str
) -> float:
    """Train one step then ``evaluate(validate)`` a conditioning profile, asserting finiteness.

    The shared train->checkpoint->validate flow behind both the clap/m2l and SAME
    real-e2e tests; callers differ only in how they augment ``dataset_root`` with the
    profile's Lance column. Trains ``experiment=surge/flow_simple`` under
    ``conditioning=<profile>`` to a checkpoint, then validates it and asserts a finite
    ``val/param_mse`` (the flow model logs param MSE, not ``val/loss``).

    :param tmp_path: The temporary output/log path shared by train and eval.
    :param dataset_root: Dataset root already augmented with the profile's column.
    :param param_spec_name: Param spec driving model width and callback labels.
    :param conditioning: Cached-conditioning profile group.
    :returns: Finite validation parameter MSE.
    """
    cfg_train = build_surge_xt_embedding_train_cfg(
        tmp_path, dataset_root, param_spec_name=param_spec_name, conditioning=conditioning
    )

    HydraConfig().set_config(cfg_train)
    _, train_objects = train(cfg_train)

    train_model = train_objects["model"]
    assert train_model.encoder.n_conditioning_outputs == len(train_model.vector_field.layers)
    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg_eval = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=surge/flow_simple",
                f"synth={param_spec_name}",
                f"conditioning={conditioning}",
                "trainer=cpu",
                "datamodule=surge_lance",
                "mode=validate",
                "model.compile=false",
                "model.validation_sample_steps=1",
            ],
        )
    with open_dict(cfg_eval):
        cfg_eval.paths.root_dir = str(operator_workspace())
        cfg_eval.paths.output_dir = str(tmp_path)
        cfg_eval.paths.log_dir = str(tmp_path)
        cfg_eval.datamodule.fake = False
        cfg_eval.datamodule.dataset_root = str(dataset_root)
        cfg_eval.datamodule.predict_file = str(dataset_root / "test.lance")
        cfg_eval.datamodule.batch_size = 1
        cfg_eval.datamodule.num_workers = 0
        cfg_eval.datamodule.use_saved_mean_and_variance = True
        cfg_eval.trainer.limit_val_batches = 1.0
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_eval.logger = None

    HydraConfig().set_config(cfg_eval)
    try:
        val_metric_dict, eval_objects = evaluate(cfg_eval)
    finally:
        GlobalHydra.instance().clear()

    eval_model = eval_objects["model"]
    assert eval_model.encoder.n_conditioning_outputs == len(eval_model.vector_field.layers)
    validation_mse = val_metric_dict["val/param_mse"].item()
    assert math.isfinite(validation_mse)
    return validation_mse


@pytest.mark.requires_vst
@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("conditioning", _EMBEDDING_CONDITIONING_PROFILES)
def test_train_eval_embedding_conditioning_real_e2e(
    tmp_path: Path,
    surge_xt_smoke_datasets: Path,
    param_spec_name: str,
    conditioning: str,
) -> None:
    """Train the flow model then validate its checkpoint over a real clap/m2l dataset.

    Renders a Surge XT dataset, appends the profile's embedding column via the real
    ``add_embeddings`` endpoint (real music2latent + CLAP encoders — no mocks), then
    runs the shared train->validate flow under ``conditioning=<profile>``.

    :param tmp_path: The temporary output/log path shared by train and eval.
    :param surge_xt_smoke_datasets: Real-VST Lance dataset root (``{train,val,test}.lance``).
    :param param_spec_name: Param spec driving model width and callback labels.
    :param conditioning: Embedding-conditioning profile under test (``m2l`` / ``clap``).
    """
    dataset_root = augment_lance_splits_with_embeddings(surge_xt_smoke_datasets)
    _assert_conditioning_train_validate_finite(
        tmp_path, dataset_root, param_spec_name, conditioning
    )


@pytest.mark.requires_vst
@pytest.mark.slow
@pytest.mark.network
def test_train_eval_meanaudio_conditioning_real_lance_returns_bounded_metric(
    tmp_path: Path,
    surge_xt_smoke_datasets: Path,
    param_spec_name: str,
) -> None:
    """Train and validate the MeanAudio profile through the public eval path.

    :param tmp_path: Shared train/eval output directory.
    :param surge_xt_smoke_datasets: Real-VST Lance dataset root.
    :param param_spec_name: Parameter specification driving model width.
    """
    validation_split = surge_xt_smoke_datasets / "val.lance"
    shutil.rmtree(validation_split)
    _render_smoke_train_subprocess(validation_split, param_spec_name, base_seed=1)
    train_audio = lance.dataset(surge_xt_smoke_datasets / "train.lance").to_table(
        columns=[AUDIO_FIELD]
    )
    validation_audio = lance.dataset(validation_split).to_table(columns=[AUDIO_FIELD])
    train_rows = {
        np.asarray(row.as_py(), dtype=np.float32).tobytes()
        for row in train_audio.column(AUDIO_FIELD)
    }
    validation_rows = {
        np.asarray(row.as_py(), dtype=np.float32).tobytes()
        for row in validation_audio.column(AUDIO_FIELD)
    }
    assert train_rows.isdisjoint(validation_rows)
    dataset_root = augment_lance_splits_with_embedding(surge_xt_smoke_datasets, "meanaudio_16k")

    validation_mse = _assert_conditioning_train_validate_finite(
        tmp_path,
        dataset_root,
        param_spec_name,
        "meanaudio_16k",
    )
    assert validation_mse < 3.0


@pytest.mark.requires_vst
@pytest.mark.slow
@pytest.mark.network
def test_train_eval_pupujepa_large_conditioning_real_lance_returns_finite_metric(
    tmp_path: Path,
    surge_xt_embedding_smoke_datasets: Path,
    param_spec_name: str,
) -> None:
    """Train and validate cached PupuJEPA Large sequences through both entrypoints.

    :param tmp_path: Shared train/eval output directory.
    :param surge_xt_embedding_smoke_datasets: Two-row real-VST Lance dataset.
    :param param_spec_name: Parameter specification driving model width.
    """
    validation_split = surge_xt_embedding_smoke_datasets / "val.lance"
    shutil.rmtree(validation_split)
    _render_smoke_train_subprocess(validation_split, param_spec_name, base_seed=1)
    train_audio = lance.dataset(surge_xt_embedding_smoke_datasets / "train.lance").to_table(
        columns=[AUDIO_FIELD]
    )
    validation_audio = lance.dataset(validation_split).to_table(columns=[AUDIO_FIELD])
    train_rows = {
        np.asarray(row.as_py(), dtype=np.float32).tobytes()
        for row in train_audio.column(AUDIO_FIELD)
    }
    validation_rows = {
        np.asarray(row.as_py(), dtype=np.float32).tobytes()
        for row in validation_audio.column(AUDIO_FIELD)
    }
    assert train_rows.isdisjoint(validation_rows)
    dataset_root = augment_lance_splits_with_embedding(
        surge_xt_embedding_smoke_datasets, "pupujepa_large"
    )

    validation_mse = _assert_conditioning_train_validate_finite(
        tmp_path,
        dataset_root,
        param_spec_name,
        "pupujepa_large",
    )
    assert validation_mse < 2.0


@pytest.mark.slow
@pytest.mark.network
def test_train_eval_pupujepa_large_online_conditioning_returns_finite_metric(
    tmp_path: Path,
    cfg_torchsynth_pupujepa_large_online_train: DictConfig,
) -> None:
    """Train and validate real-weight PupuJEPA Large through both entrypoints.

    :param tmp_path: Shared train/eval output directory.
    :param cfg_torchsynth_pupujepa_large_online_train: Two-row production-path config.
    """
    cfg_train = cfg_torchsynth_pupujepa_large_online_train
    HydraConfig().set_config(cfg_train)
    _, train_objects = train(cfg_train)
    datamodule = train_objects["datamodule"]
    datamodule.setup("fit")
    train_params = next(iter(datamodule.train_dataloader()))["params"]
    validation_params = next(iter(datamodule.val_dataloader()))["params"]
    train_rows = {tuple(row.tolist()) for row in train_params}
    validation_rows = {tuple(row.tolist()) for row in validation_params}
    assert train_rows.isdisjoint(validation_rows)
    checkpoint_path = tmp_path / "pupujepa-large-online.ckpt"
    train_objects["trainer"].save_checkpoint(checkpoint_path)

    cfg_eval = cfg_train.copy()
    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(checkpoint_path)
        cfg_eval.mode = "validate"
        cfg_eval.trainer.limit_val_batches = 1
    HydraConfig().set_config(cfg_eval)
    try:
        metric_dict, _ = evaluate(cfg_eval)
    finally:
        GlobalHydra.instance().clear()

    validation_mse = metric_dict["val/param_mse"].item()
    assert math.isfinite(validation_mse)
    assert validation_mse < 2.0


@pytest.mark.requires_vst
@pytest.mark.slow
@pytest.mark.integration_r2
@pytest.mark.r2
def test_train_eval_matpac_plus_conditioning_real_lance_returns_finite_metric(
    tmp_path: Path,
    surge_xt_smoke_datasets: Path,
    param_spec_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train and validate real MATPAC++ tensors through the generic pooler.

    :param tmp_path: Shared train/eval output directory.
    :param surge_xt_smoke_datasets: Real-VST Lance dataset root.
    :param param_spec_name: Parameter specification driving model width.
    :param monkeypatch: Fixture recording tensors consumed by the conditioning pooler.
    """
    pooled_inputs: list[torch.Tensor] = []
    original_forward = EmbeddingPool.forward

    def record_forward(pool: EmbeddingPool, embed: torch.Tensor) -> torch.Tensor:
        pooled_inputs.append(embed.detach().cpu())
        return original_forward(pool, embed)

    monkeypatch.setattr(EmbeddingPool, "forward", record_forward)
    validation_split = surge_xt_smoke_datasets / "val.lance"
    shutil.rmtree(validation_split)
    _render_smoke_train_subprocess(validation_split, param_spec_name, base_seed=1)
    train_audio = lance.dataset(surge_xt_smoke_datasets / "train.lance").to_table(
        columns=[AUDIO_FIELD]
    )
    validation_audio = lance.dataset(validation_split).to_table(columns=[AUDIO_FIELD])
    train_rows = {
        np.asarray(row.as_py(), dtype=np.float32).tobytes()
        for row in train_audio.column(AUDIO_FIELD)
    }
    validation_rows = {
        np.asarray(row.as_py(), dtype=np.float32).tobytes()
        for row in validation_audio.column(AUDIO_FIELD)
    }
    assert train_rows.isdisjoint(validation_rows)
    dataset_root = augment_lance_splits_with_embedding(surge_xt_smoke_datasets, "matpac_plus")
    flatten_lance_embedding_column(dataset_root, "matpac_plus")
    validation_mse = _assert_conditioning_train_validate_finite(
        tmp_path,
        dataset_root,
        param_spec_name,
        "matpac_plus",
    )
    assert validation_mse < 2.0
    assert pooled_inputs
    assert all(embed.shape[1] == MATPAC_PLUS_FRONTEND.embedding_dim for embed in pooled_inputs)
    assert any(torch.count_nonzero(embed).item() > 0 for embed in pooled_inputs)


@pytest.mark.requires_vst
@pytest.mark.slow
@pytest.mark.network
def test_evaluate_ssondo_conditioning_real_e2e(
    local_embedding_checkpoints: dict[str, str],
    tmp_path: Path,
    surge_xt_embedding_smoke_datasets: Path,
    param_spec_name: str,
) -> None:
    """Compose eval independently and validate a real S-SONDO checkpoint.

    :param local_embedding_checkpoints: Preflighted real model checkpoints.
    :param tmp_path: Shared train/eval output directory.
    :param surge_xt_embedding_smoke_datasets: Two-row real-VST Lance dataset.
    :param param_spec_name: Parameter specification driving model width.
    """
    dataset_root = augment_lance_splits_with_ssondo(
        surge_xt_embedding_smoke_datasets,
        local_embedding_checkpoints["ssondo"],
    )

    _assert_conditioning_train_validate_finite(
        tmp_path,
        dataset_root,
        param_spec_name,
        "ssondo",
    )


_SAME_CONDITIONING_PROFILES = ("same_s", "same_l")


@pytest.mark.requires_vst
@pytest.mark.slow
@pytest.mark.network
@pytest.mark.same_e2e
@pytest.mark.parametrize("conditioning", _SAME_CONDITIONING_PROFILES)
def test_train_eval_same_conditioning_real_e2e(
    tmp_path: Path,
    surge_xt_smoke_datasets: Path,
    param_spec_name: str,
    conditioning: str,
) -> None:
    """Train the flow model then validate its checkpoint over a real SAME dataset.

    The SAME sibling of :func:`test_train_eval_embedding_conditioning_real_e2e`:
    renders a Surge XT dataset, appends the ``same_s``/``same_l`` column through the
    production Stable Audio 3 encoder, trains ``experiment=surge/flow_simple`` one
    step to a checkpoint, then drives ``evaluate(mode=validate)`` and asserts a
    finite ``val/param_mse``.

    :param tmp_path: The temporary output/log path shared by train and eval.
    :param surge_xt_smoke_datasets: Real-VST Lance dataset root (``{train,val,test}.lance``).
    :param param_spec_name: Param spec driving model width and callback labels.
    :param conditioning: SAME conditioning profile under test (``same_s`` / ``same_l``).
    """
    dataset_root = augment_lance_splits_with_same(surge_xt_smoke_datasets, conditioning)
    _assert_conditioning_train_validate_finite(
        tmp_path, dataset_root, param_spec_name, conditioning
    )


_THIRD_PARTY_CORPUS_ROWS = 2
_THIRD_PARTY_SOURCE_SAMPLE_RATE = 16_000
_THIRD_PARTY_TEST_SEED = 3407
_SURGE_SIMPLE_PREDICTION_WIDTH = 92
# Matches jobs-style tiny widths: the checkpoint only has to load and sample once.
_THIRD_PARTY_MODEL_OVERRIDES = (
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
)


def _save_third_party_checkpoint(
    path: Path,
    experiment: str = "surge/flow_simple",
) -> None:
    """Save a real surge-simple flow checkpoint from a shipped Hydra config.

    :param path: Destination checkpoint path.
    :param experiment: Experiment whose model architecture is serialized.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            overrides=[
                f"experiment={experiment}",
                "trainer=cpu",
                *_THIRD_PARTY_MODEL_OVERRIDES,
            ],
        )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    seed_everything(_THIRD_PARTY_TEST_SEED, workers=True)
    trainer.strategy.connect(instantiate(cfg.model))
    trainer.save_checkpoint(path)


def test_third_party_checkpoint_creation_is_reproducible(tmp_path: Path) -> None:
    """Checkpoint weights do not depend on RNG consumed by earlier tests.

    :param tmp_path: Isolated checkpoint directory.
    """
    first_path = tmp_path / "first.ckpt"
    torch.manual_seed(1)
    _save_third_party_checkpoint(first_path)
    second_path = tmp_path / "second.ckpt"
    torch.manual_seed(2)
    _save_third_party_checkpoint(second_path)

    first = torch.load(first_path, map_location="cpu", weights_only=False)
    second = torch.load(second_path, map_location="cpu", weights_only=False)
    torch.testing.assert_close(first["state_dict"], second["state_dict"], rtol=0, atol=0)


def _run_third_party_eval(
    *,
    corpus: Path,
    checkpoint: Path | str,
    output_dir: Path,
    experiment: str = "surge/flow_simple",
    datamodule: str = "third_party/nsynth_test",
    extra_overrides: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    """Run the public eval CLI over a third-party corpus with no ground-truth patch.

    :param corpus: Blob-audio Lance corpus to serve.
    :param checkpoint: Checkpoint to load.
    :param output_dir: Eval output root.
    :param experiment: Experiment config exercised by the subprocess.
    :param datamodule: Third-party datamodule config exercised by the subprocess.
    :param extra_overrides: Scenario-specific Hydra overrides.
    :returns: The completed CLI process.
    """
    subprocess_env = {key: value for key, value in os.environ.items() if key != "WANDB_SERVICE"}
    return subprocess.run(  # noqa: S603 — argv contains only test-owned paths
        [
            sys.executable,
            "-m",
            "synth_setter.cli.eval",
            f"experiment={experiment}",
            f"datamodule={datamodule}",
            "render=vst",
            f"seed={_THIRD_PARTY_TEST_SEED}",
            f"datamodule.dataset_uri={corpus}",
            "datamodule.use_saved_mean_and_variance=false",
            "datamodule.mel_stats_uri=null",
            "datamodule.num_workers=0",
            "callbacks=eval_vst",
            "mode=predict",
            "trainer=cpu",
            f"ckpt_path={checkpoint}",
            "evaluation.no_params=true",
            "evaluation.rerender_target=false",
            *_THIRD_PARTY_MODEL_OVERRIDES,
            *extra_overrides,
            f"paths.output_dir={output_dir}",
            "hydra.job.chdir=false",
            "+trainer.enable_progress_bar=false",
            "+trainer.enable_model_summary=false",
        ],
        capture_output=True,
        text=True,
        timeout=900,
        env=subprocess_env,
    )


@pytest.mark.slow
def test_third_party_corpus_predict_entrypoint_writes_artifacts(tmp_path: Path) -> None:
    """The public eval CLI predicts a published corpus with no ground-truth patch.

    Drives the real entrypoint over blob-stored source audio: decode, resample,
    up-mix, and the mel front-end all happen in the dataloader, and
    ``no_params`` keeps target audio sourced from the corpus.

    :param tmp_path: Isolated corpus, checkpoint, and output directories.
    """
    corpus = tmp_path / "corpus.lance"
    write_blob_audio_corpus(
        corpus,
        [
            np.zeros(_THIRD_PARTY_SOURCE_SAMPLE_RATE, dtype=np.float32) + 0.1 * index
            for index in range(_THIRD_PARTY_CORPUS_ROWS)
        ],
        sample_rate=_THIRD_PARTY_SOURCE_SAMPLE_RATE,
    )
    checkpoint = tmp_path / "flow_simple.ckpt"
    _save_third_party_checkpoint(checkpoint)
    output_dir = tmp_path / "output"

    result = _run_third_party_eval(
        corpus=corpus,
        checkpoint=checkpoint,
        output_dir=output_dir,
        extra_overrides=("datamodule.batch_size=2",),
    )

    assert result.returncode == 0, result.stderr
    prediction = torch.load(
        output_dir / "predictions" / "pred-0.pt", map_location="cpu", weights_only=True
    )
    target_audio = torch.load(
        output_dir / "predictions" / "target-audio-0.pt", map_location="cpu", weights_only=True
    )
    assert prediction.shape == (_THIRD_PARTY_CORPUS_ROWS, _SURGE_SIMPLE_PREDICTION_WIDTH)
    assert torch.isfinite(prediction).all()
    assert not (output_dir / "predictions" / "target-params-0.pt").exists()
    assert target_audio.shape[1:] == (2, 4 * 44_100)
    # Mean amplitude verifies content, order, padding, and up-mixing.
    served = [float(row.abs().mean()) for row in target_audio]
    assert served == pytest.approx([0.0, 0.025], abs=1e-3)


@pytest.mark.slow
def test_nsynth_sketch_eval_entrypoint_writes_prediction(
    tmp_path: Path,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The NSynth preset runs an R2 checkpoint and live PESTO through prediction.

    :param tmp_path: Isolated corpus, checkpoint, and output directories.
    :param fake_r2_remote: Local filesystem backing the real rclone remote.
    :param monkeypatch: Configures storage credentials inherited by the subprocess.
    """
    sample_rate = _THIRD_PARTY_SOURCE_SAMPLE_RATE
    samples = np.arange(sample_rate, dtype=np.float32)
    tone = (0.5 * np.sin(2 * np.pi * 440.0 * samples / sample_rate)).astype(np.float32)
    corpus = tmp_path / "corpus.lance"
    write_blob_audio_corpus(corpus, [tone], sample_rate=sample_rate)
    checkpoint = tmp_path / "flow_sketch.ckpt"
    _save_third_party_checkpoint(checkpoint, experiment="surge/flow_sketch_prelim")
    remote_checkpoint = fake_r2_remote / "bucket" / "runs" / "flow_sketch.ckpt"
    remote_checkpoint.parent.mkdir(parents=True)
    shutil.copyfile(checkpoint, remote_checkpoint)
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_ENDPOINT_URL", "http://localhost:0")
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_RCLONE_TYPE", "local")
    output_dir = tmp_path / "output"

    result = _run_third_party_eval(
        corpus=corpus,
        checkpoint="r2://bucket/runs/flow_sketch.ckpt",
        output_dir=output_dir,
        experiment="surge/eval_flow_sketch_nsynth",
        datamodule="third_party/nsynth_sketch",
        extra_overrides=(
            f"ckpt_sha256={checkpoint_digest}",
            "datamodule.batch_size=1",
            "datamodule.mel_stats_sha256=null",
            "evaluation.compute_metrics=false",
            "evaluation.render_vst=false",
            "~logger",
        ),
    )

    assert result.returncode == 0, result.stderr
    prediction = torch.load(
        output_dir / "predictions" / "pred-0.pt", map_location="cpu", weights_only=True
    )
    assert prediction.shape == (1, _SURGE_SIMPLE_PREDICTION_WIDTH)
    assert torch.isfinite(prediction).all()


@pytest.mark.requires_vst
@pytest.mark.slow
def test_third_party_corpus_no_params_renders_against_dataset_audio(tmp_path: Path) -> None:
    """The ``no_params`` render branch scores predictions against the corpus's own audio.

    With no target params on disk, ``target.wav`` must come from staged dataset
    audio; a dropped or rejected ``--no-params`` flag fails this test.

    :param tmp_path: Isolated corpus, checkpoint, and output directories.
    """
    corpus = tmp_path / "corpus.lance"
    write_blob_audio_corpus(
        corpus,
        [np.full(_THIRD_PARTY_SOURCE_SAMPLE_RATE, 0.25, dtype=np.float32)],
        sample_rate=_THIRD_PARTY_SOURCE_SAMPLE_RATE,
    )
    checkpoint = tmp_path / "flow_simple.ckpt"
    _save_third_party_checkpoint(checkpoint)
    output_dir = tmp_path / "output"

    result = _run_third_party_eval(
        corpus=corpus,
        checkpoint=checkpoint,
        output_dir=output_dir,
        extra_overrides=("datamodule.batch_size=1", "evaluation.render_vst=true"),
    )

    assert result.returncode == 0, result.stderr
    sample = output_dir / "audio" / "sample_0"
    with AudioFile(str(sample / "target.wav")) as handle:
        target = handle.read(handle.frames)
    assert (sample / "pred.wav").is_file()
    # Staged corpus audio must be preserved; a re-render or zeroed target changes its level.
    assert float(np.abs(target).mean()) == pytest.approx(0.0625, abs=5e-3)
