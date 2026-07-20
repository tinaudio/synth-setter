"""Tests for the ``synth-setter-train`` CLI entrypoint.

Each test composes a Hydra ``cfg`` fixture and drives the in-process
``train(cfg)`` entrypoint (some chain ``evaluate``). Keep this module to
cfg-entrypoint tests; unit tests for helper functions belong in sibling
``test_*`` modules. ``tests/_meta/test_entrypoint_test_modules.py`` enforces
that no private ``synth_setter.cli`` helper is imported here.
"""

import logging
import os
import re
import shutil
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Literal
from unittest.mock import PropertyMock, patch
from uuid import UUID

import hydra
import numpy as np
import pandas as pd
import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import Trainer
from omegaconf import DictConfig, open_dict

from synth_setter.cli.eval import evaluate
from synth_setter.cli.train import train
from synth_setter.data.vst import param_specs
from synth_setter.models.components.cnn import LogMelEncoder
from synth_setter.models.components.pretrained_ast import PretrainedASTEncoder
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.pipeline.spec_io import write_spec_to_path
from synth_setter.utils import resolve_run_config_id
from synth_setter.utils.callbacks import ValidationAlignedModelCheckpoint
from synth_setter.utils.utils import register_resolvers
from synth_setter.workspace import operator_workspace
from tests._vst import PLUGIN_PATH
from tests.conftest import (
    _SURGE_FIXTURE_CHANNELS,
    _SURGE_FIXTURE_DURATION_SECONDS,
    _SURGE_FIXTURE_SAMPLE_RATE,
    FAKE_VST_VARIANTS,
    NUM_FIXTURE_SAMPLES,
    REAL_VST_VARIANTS,
    _build_surge_xt_smoke_cfg,
    _SurgeSmokeVariant,
    assert_finite_train_loss,
    assert_log_per_param_mse_wired,
    build_fake_flow_ast_pretrained_train_cfg,
    build_fake_train_cfg,
    train_loss_keys,
)
from tests.evaluation._oracle_helpers import ORACLE_AUDIO_METRIC_BOUNDS
from tests.helpers.eval_fakes import (
    FAKE_AGGREGATED_METRICS_CSV,
    fake_metrics_csv,
    fake_postprocessing_subprocess,
)
from tests.helpers.noise_capture import NoiseCaptureCallback
from tests.helpers.recording_wandb_logger import RecordingWandbLogger as _RecordingWandbLogger
from tests.helpers.run_if import RunIf
from tests.helpers.wandb_artifacts import publish_checkpoint_artifact

# Experiments cycled through the Surge XT VST smoke tests below. Single source of truth so
# the parametrize lists on the two ``test_train_*_surge_xt`` tests cannot drift apart.
_ORACLE_EXPERIMENT = "surge/fake_oracle"
_SURGE_SMOKE_EXPERIMENTS = (_ORACLE_EXPERIMENT, "surge/ffn_full")


def _assert_oracle_zero_train_loss(metric_dict: dict[str, torch.Tensor]) -> None:
    # The oracle constructs its loss as 0.0 * net(...).sum(); any nonzero value
    # means it stopped being an oracle.
    for key in train_loss_keys(metric_dict):
        loss_value = metric_dict[key].item()
        assert loss_value == 0.0, f"oracle {key} not exactly zero: {loss_value}"


_PREDICTION_PT_PREFIXES = ("pred", "target-audio", "target-params")
_FAKE_METRICS_CSV = fake_metrics_csv(NUM_FIXTURE_SAMPLES)


# TODO(#40): add @pytest.mark.ram gate for memory-intensive CPU tests test_train_fast_dev_run


def _smoke_eval_postprocessing_fake() -> Callable[[list[str]], None]:
    """Return fake eval postprocessing that materializes the fake-plugin smoke outputs.

    :returns: ``subprocess.run``-compatible callable.
    """
    return fake_postprocessing_subprocess(
        audio_metrics_csv=FAKE_AGGREGATED_METRICS_CSV,
        per_sample_metrics_csv=_FAKE_METRICS_CSV,
        render_sample_count=NUM_FIXTURE_SAMPLES,
    )


def _record_successful_r2_uploads(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[Path, str, bytes]]:
    """Record uploads only after the real rclone-backed copy succeeds.

    :param monkeypatch: Stubs the R2 auth probe and wraps the upload transport.
    :returns: Successful ``(local_path, uri, source_bytes_at_upload)`` copies.
    """
    real_upload = r2_io.upload_to_uri
    uploads: list[tuple[Path, str, bytes]] = []

    def _record(local_path: Path, uri: str) -> None:
        real_upload(local_path, uri)
        uploads.append((Path(local_path), uri, Path(local_path).read_bytes()))

    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(r2_io, "upload_to_uri", _record)
    return uploads


@pytest.mark.dataloader_multiprocess
@pytest.mark.xdist_group(name="dataloader-multiprocess")
def test_train_fast_dev_run_tiny_model_tiny_data(cfg_train: DictConfig) -> None:
    """Run 1 train, val, and test step on CPU with `fast_dev_run`.

    Dataset/batch size constraints come from the shared `cfg_train` fixture
    (`batch_size=1`, `train_val_test_sizes=[2, 2, 2]`). DataLoader workers
    exercise spawn integration while ``fast_dev_run`` caps each loop.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.datamodule.num_workers = 2
        cfg_train.trainer.fast_dev_run = True
    train(cfg_train)


@pytest.mark.parametrize(
    ("received_sigterm", "expected_exit_code"),
    [(True, 143), (False, 7)],
)
def test_train_fit_system_exit_uses_signal_status_only_for_sigterm(
    cfg_train: DictConfig,
    received_sigterm: bool,
    expected_exit_code: int,
) -> None:
    """Translate Lightning's handled SIGTERM without masking other exits.

    :param cfg_train: A DictConfig containing a valid training configuration.
    :param received_sigterm: Whether Lightning recorded a SIGTERM.
    :param expected_exit_code: Exit code expected from the training entrypoint.
    """
    HydraConfig().set_config(cfg_train)
    with (
        patch.object(Trainer, "fit", side_effect=SystemExit(7)),
        patch.object(
            Trainer,
            "received_sigterm",
            new_callable=PropertyMock,
            return_value=received_sigterm,
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        train(cfg_train)

    assert exc_info.value.code == expected_exit_code


def test_train_torchsynth_experiment_renders_audio_online(
    cfg_torchsynth_train: DictConfig,
) -> None:
    """Run the TorchSynth experiment without a materialized audio dataset.

    :param cfg_torchsynth_train: Composed CPU TorchSynth smoke configuration.
    """
    HydraConfig().set_config(cfg_torchsynth_train)
    metric_dict, object_dict = train(cfg_torchsynth_train)

    assert "train/loss" in metric_dict
    assert torch.isfinite(metric_dict["train/loss"])
    batch = next(iter(object_dict["datamodule"].train_dataloader()))
    audio, params, *_ = batch
    assert audio.shape == (1, cfg_torchsynth_train.datamodule.signal_length)
    assert audio.shape[-1] == 176_400
    assert params.shape == (1, cfg_torchsynth_train.datamodule.num_params)
    assert torch.isfinite(audio).all()
    assert isinstance(object_dict["model"].net.encoder, LogMelEncoder)


def test_train_torchsynth_resample_per_epoch_completes_multi_epoch_fit(
    cfg_torchsynth_train: DictConfig,
) -> None:
    """Train two epochs with per-epoch resampling through the real entrypoint.

    Pins that Lightning's fit loop accepts the fresh-index train sampler across
    epoch boundaries (one ``iter()`` per epoch on the same loader).

    :param cfg_torchsynth_train: Composed CPU TorchSynth smoke configuration.
    """
    HydraConfig().set_config(cfg_torchsynth_train)
    with open_dict(cfg_torchsynth_train):
        cfg_torchsynth_train.datamodule.resample_train_per_epoch = True
        cfg_torchsynth_train.trainer.fast_dev_run = False
        cfg_torchsynth_train.trainer.max_epochs = 2
    metric_dict, _ = train(cfg_torchsynth_train)

    assert "train/loss" in metric_dict
    assert torch.isfinite(metric_dict["train/loss"])


@pytest.mark.gpu
@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step on GPU.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
    train(cfg_train)


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_fast_dev_run_gpu_compile(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step on GPU with torch.compile enabled.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.model.compile = True
    train(cfg_train)


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(cfg_train: DictConfig) -> None:
    """Train 1 epoch on GPU with mixed-precision.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.precision = 16
    train(cfg_train)


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_train: DictConfig) -> None:
    """Train 1 epoch with validation loop twice per epoch.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.check_val_every_n_epoch = 1
        cfg_train.trainer.val_check_interval = 0.5
    train(cfg_train)


@pytest.mark.slow
def test_train_ddp_sim(cfg_train: DictConfig) -> None:
    """Simulate DDP (Distributed Data Parallel) on 2 CPU processes.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 2
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 2
        cfg_train.trainer.strategy = "ddp_spawn"
        # Workaround for #709: ddp_spawn rank processes start with torch's
        # default `file_descriptor` sharing strategy, and their forked
        # dataloader workers inherit it. On the GitHub-hosted
        # `ubuntu-latest-4core` runner that strategy fails with
        # `RuntimeError: unable to resize file ... Invalid argument (22)`
        # because anonymous shm-backed fds can't be ftruncate'd in the
        # runner sandbox. Setting num_workers=0 keeps dataloading inline in
        # each rank process, sidestepping cross-process tensor shm entirely.
        # This test exercises ddp_spawn coordination, not dataloader
        # parallelism, so dropping workers does not weaken coverage.
        cfg_train.datamodule.num_workers = 0
    train(cfg_train)


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_resume(tmp_path: Path, cfg_train: DictConfig) -> None:
    """Run 1 epoch, finish, and resume for another epoch.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    with open_dict(cfg_train):
        cfg_train.trainer.accelerator = "gpu"
    HydraConfig().set_config(cfg_train)
    _, object_dict_1 = train(cfg_train)
    step_after_first = object_dict_1["trainer"].global_step
    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_train.trainer.max_epochs = 2

    _, object_dict_2 = train(cfg_train)
    step_after_resume = object_dict_2["trainer"].global_step

    files = os.listdir(tmp_path / "checkpoints")
    assert "epoch_001.ckpt" in files
    assert "epoch_002.ckpt" not in files

    # The resume must actually train another epoch — `trainer.global_step` advancing
    # past the post-first-train value is the cheapest signal that the second `train()`
    # call did real work and didn't just load the checkpoint and exit. Replaces the
    # earlier `train/loss_epoch` decrease assertion, which broke when the metric_dict
    # keys changed.
    assert step_after_resume > step_after_first, (
        f"resume did not advance training: "
        f"global_step before={step_after_first}, after={step_after_resume}"
    )


@pytest.mark.parametrize("param_spec_name", ["surge_4", "surge_simple", "surge_xt"])
def test_cfg_surge_xt_global_wires_param_spec(param_spec_name: str) -> None:
    """Templated ``_build_surge_xt_smoke_cfg`` propagates the param spec for every supported spec.

    The datamodule selection drives both ``model.net.d_out`` and
    ``callbacks.log_per_param_mse.param_spec``.

    Calls the builder directly (not the ``cfg_surge_xt_global`` fixture) and pins
    ``accelerator="cpu"``: the cfg-shape contract is accelerator-independent and going
    through the fixture would drag in the parametrized ``accelerator`` hardware gate that
    hardfails on hosts without MPS/CUDA. ``experiment`` is pinned to the fixture default
    because param-spec propagation is itself experiment-independent — there's no need to
    cross-product with the experiment axis here.

    :param param_spec_name: Spec name driving the cfg builder.
    """
    cfg = _build_surge_xt_smoke_cfg(
        accelerator="cpu",
        param_spec_name=param_spec_name,
        experiment=_ORACLE_EXPERIMENT,
    )
    assert cfg.datamodule.param_spec_name == param_spec_name
    assert cfg.model.net.d_out == len(param_specs[param_spec_name])
    assert cfg.callbacks.log_per_param_mse.param_spec == param_spec_name


@pytest.mark.slow
def test_train_fake_mode_nondefault_spec_sizes_batches_from_registry(tmp_path: Path) -> None:
    """Fake-mode train through the entrypoint sizes batches from a non-default ``param_spec_name``.

    Drives the real ``train(cfg)`` entrypoint with ``datamodule.fake=true`` and the
    non-default ``surge_simple`` spec: no dataset on disk, so the run exercises the
    registry-derived fake width end-to-end. The width-agnostic ``surge/fake_oracle``
    experiment (oracle returns ``batch["params"]``) tolerates the registry-width batches,
    and the datamodule the entrypoint built carries that registry-derived width.

    :param tmp_path: Pinned as Hydra ``output_dir`` / ``log_dir``; no dataset is read.
    """
    expected_width = len(param_specs["surge_simple"])
    cfg = build_fake_train_cfg(tmp_path, param_spec_name="surge_simple")

    HydraConfig().set_config(cfg)
    _, object_dict = train(cfg)

    trainer = object_dict["trainer"]
    assert trainer.global_step >= 1, f"trainer did not advance: global_step={trainer.global_step}"

    assert_log_per_param_mse_wired(trainer, "surge_simple")

    datamodule = object_dict["datamodule"]
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    assert batch["params"].shape == (2, expected_width)
    datamodule.teardown("fit")


@pytest.mark.dataloader_multiprocess
@pytest.mark.xdist_group(name="dataloader-multiprocess")
@pytest.mark.slow
def test_train_prefetch_factor_override_advances_with_spawn_workers(tmp_path: Path) -> None:
    """An explicit ``datamodule.prefetch_factor`` reaches train's composed spawn loaders.

    Drives the real ``train(cfg)`` entrypoint in fake mode with spawn workers
    so the non-default prefetch depth governs live worker buffering.

    :param tmp_path: Pinned as Hydra ``output_dir`` / ``log_dir``; no dataset is read.
    """
    cfg = build_fake_train_cfg(tmp_path, param_spec_name="surge_simple")
    with open_dict(cfg):
        cfg.datamodule.num_workers = 2
        cfg.datamodule.prefetch_factor = 3

    HydraConfig().set_config(cfg)
    _, object_dict = train(cfg)

    trainer = object_dict["trainer"]
    assert trainer.global_step >= 1, f"trainer did not advance: global_step={trainer.global_step}"
    assert object_dict["datamodule"].prefetch_factor == 3


def test_train_legacy_vst_groups_wire_per_param_callback(tmp_path: Path) -> None:
    """Legacy model and callback aliases run through the train entrypoint.

    :param tmp_path: Pinned as Hydra ``output_dir`` / ``log_dir``; no dataset is read.
    """
    cfg = build_fake_train_cfg(
        tmp_path,
        param_spec_name="surge_simple",
        model_group="surge_fake_oracle",
        callbacks_group="default_surge",
    )

    HydraConfig().set_config(cfg)
    _, object_dict = train(cfg)

    assert_log_per_param_mse_wired(object_dict["trainer"], "surge_simple")


def test_train_val_audio_probe_spec_mismatch_fails_at_configure_time(tmp_path: Path) -> None:
    """The real train entrypoint dies at configure time on a probe/model spec mismatch.

    The guard kills a launch whose probe cannot decode the model's predictions
    before a single training step runs (#1990).

    :param tmp_path: Pinned as Hydra ``output_dir`` / ``log_dir``; no dataset is read.
    """
    cfg = build_fake_train_cfg(tmp_path, param_spec_name="surge_simple")
    with open_dict(cfg):
        cfg.training.val_audio_probe = True
        cfg.render = {
            "param_spec_name": "surge_xt",
            "plugin_state_path": "presets/surge-base.vstpreset",
        }

    HydraConfig().set_config(cfg)
    with pytest.raises(ValueError, match="param_spec_name"):
        train(cfg)


@pytest.mark.slow
@pytest.mark.parametrize("model_group", ["vst_flow", "vst_flowmlp"])
def test_train_surge_simple_flow_default_width_matches_fake_batch(
    tmp_path: Path, model_group: str
) -> None:
    """A real optimizer step succeeds at each flow group's registry-derived default width.

    The vector field is shrunk only for CPU speed — width wiring is untouched. The loss
    must come out finite: ``global_step`` advances even past a NaN loss.

    :param tmp_path: Hydra output and log directory; no dataset is read.
    :param model_group: Canonical flow model group under test.
    """
    cfg = build_fake_train_cfg(
        tmp_path,
        param_spec_name="surge_simple",
        model_group=model_group,
    )
    with open_dict(cfg):
        cfg.model.compile = False
        cfg.test = False
        cfg.model.vector_field.num_layers = 1
        cfg.model.vector_field.d_model = 32
        if model_group == "vst_flow":
            cfg.model.vector_field.d_ff = 32
            cfg.model.vector_field.projection.num_tokens = 8
        else:
            cfg.model.vector_field.d_enc = 32

    assert cfg.model.num_params == len(param_specs["surge_simple"])

    HydraConfig().set_config(cfg)
    metric_dict, object_dict = train(cfg)

    assert object_dict["trainer"].global_step >= 1
    assert_finite_train_loss(metric_dict)


@pytest.mark.slow
@pytest.mark.parametrize("experiment", ["surge/ffn_simple", "surge/flow_simple"])
@pytest.mark.parametrize("param_spec_name", ["surge_simple"], indirect=True)
def test_train_runpod_experiment_default_datamodule_advances(
    tmp_path: Path,
    fake_surge_smoke_datasets: Path,
    experiment: str,
) -> None:
    """Train one step with each RunPod experiment's own datamodule selection.

    :param tmp_path: Hydra output and log directory.
    :param fake_surge_smoke_datasets: Tiny loadable Lance train/validation/test splits.
    :param experiment: Shipped RunPod launch experiment to train.
    """
    cfg = _build_surge_xt_smoke_cfg(
        accelerator="cpu",
        param_spec_name="surge_simple",
        experiment=experiment,
        datamodule_group=None,
    )
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.datamodule.dataset_root = str(fake_surge_smoke_datasets)

    HydraConfig().set_config(cfg)
    metric_dict, object_dict = train(cfg)

    assert object_dict["trainer"].global_step >= 1
    assert_finite_train_loss(metric_dict)


def test_train_flow_simple_with_ast_pretrained_encoder_advances(tmp_path: Path) -> None:
    """Train one real flow step through the offline pretrained-AST config.

    The loss must come out finite: ``global_step`` advances even past a NaN loss.

    :param tmp_path: Hydra output and log directory; no dataset is read.
    """
    cfg = build_fake_flow_ast_pretrained_train_cfg(tmp_path)

    HydraConfig().set_config(cfg)
    metric_dict, object_dict = train(cfg)

    trainer = object_dict["trainer"]
    assert trainer.global_step >= 1, f"trainer did not advance: global_step={trainer.global_step}"
    assert_finite_train_loss(metric_dict)

    encoder = object_dict["model"].encoder
    assert isinstance(encoder, PretrainedASTEncoder)


@pytest.mark.requires_vst
@pytest.mark.slow
@pytest.mark.parametrize("experiment_name", _SURGE_SMOKE_EXPERIMENTS, indirect=True)
@pytest.mark.parametrize("surge_smoke_variant", REAL_VST_VARIANTS, indirect=True)
def test_train_surge_xt(cfg_surge_real_train: DictConfig, experiment_name: str) -> None:
    """Train both Surge XT smoke experiments through the map-style Lance datamodule.

    Asserts the trainer advanced and produced a finite ``train/loss`` — catches silent
    no-op trainers and NaN/Inf regressions that a bare ``train()`` call would not. The
    ``surge/fake_oracle`` leg additionally pins ``train/loss`` to exactly zero (the
    oracle constructs its loss as ``0.0 * net(mel_spec).sum()`` — any drift means the
    oracle stopped being an oracle); meaningful loss-progression coverage comes from
    the ``surge/ffn_full`` leg. Both train through the real Surge XT render.

    :param cfg_surge_real_train: Surge XT training config parametrized over experiment.
    :param experiment_name: Hydra experiment override the cfg was built from — drives
        the oracle-specific tight bound below.
    """
    HydraConfig().set_config(cfg_surge_real_train)
    metric_dict, object_dict = train(cfg_surge_real_train)

    trainer = object_dict["trainer"]
    assert trainer.global_step >= 1, f"trainer did not advance: global_step={trainer.global_step}"

    assert_finite_train_loss(metric_dict)

    if experiment_name == _ORACLE_EXPERIMENT:
        _assert_oracle_zero_train_loss(metric_dict)


@pytest.mark.requires_vst
@pytest.mark.slow
@pytest.mark.parametrize("experiment_name", _SURGE_SMOKE_EXPERIMENTS, indirect=True)
@pytest.mark.parametrize("surge_smoke_variant", REAL_VST_VARIANTS, indirect=True)
def test_train_eval_surge_xt(
    tmp_path: Path,
    cfg_surge_real_train: DictConfig,
    cfg_surge_real_eval: DictConfig,
    param_spec_name: str,
    experiment_name: str,
) -> None:
    """Train Surge XT briefly, then evaluate its checkpoint through Lance.

    :param tmp_path: The temporary logging path.
    :param cfg_surge_real_train: Surge XT smoke-test training config (Lance).
    :param cfg_surge_real_eval: Matching smoke-test eval config (ckpt_path set by this test).
    :param param_spec_name: Param spec the fixtures (and therefore the trained model) are
        wired for — passed to ``predict_vst_audio.py`` so the script's decode layout matches
        the predicted tensor's encoding (mismatched specs go off-the-end and crash with
        ``can only convert an array of size 1 to a Python scalar``).
    :param experiment_name: Hydra experiment override the cfg was built from — drives
        the oracle-specific tight audio-metric bounds at the end of the test.
    """
    from pedalboard.io import AudioFile

    NUM_AUDIO_METRICS = 4  # mss, wmfcc, sot, rms
    METRICS_FILE_EXPECTATIONS = {
        "aggregated_metrics.csv": {
            "rows": NUM_AUDIO_METRICS,
            "columns": {"mean", "std"},
        },
        "metrics.csv": {
            "rows": NUM_FIXTURE_SAMPLES,
            "columns": {"mss", "wmfcc", "sot", "rms"},
        },
    }

    HydraConfig().set_config(cfg_surge_real_train)
    train(cfg_surge_real_train)

    # `cfg_surge_real_eval.ckpt_path` is pre-pointed at this same `tmp_path` by the
    # fixture; assert the train step actually produced the file before eval reads it.
    assert Path(cfg_surge_real_eval.ckpt_path).exists()

    HydraConfig().set_config(cfg_surge_real_eval)
    evaluate(cfg_surge_real_eval)

    # `PredictionWriter` (in `src/synth_setter/utils/callbacks.py`) with `write_interval=batch` saves three
    # tensors per predict batch: `pred-{i}.pt`, `target-audio-{i}.pt`, `target-params-{i}.pt`.
    predictions_dir = tmp_path / "predictions"
    assert predictions_dir.is_dir()
    expected_names = sorted(
        f"{prefix}-{i}.pt"
        for prefix in ("pred", "target-audio", "target-params")
        for i in range(NUM_FIXTURE_SAMPLES)
    )
    assert sorted(p.name for p in predictions_dir.iterdir()) == expected_names

    for i in range(NUM_FIXTURE_SAMPLES):
        pred = torch.load(predictions_dir / f"pred-{i}.pt", weights_only=True)
        assert torch.isfinite(pred).all(), f"pred-{i}.pt contains NaN/Inf"

        # The oracle's ``predict_step`` returns ``batch["params"]`` verbatim, so the
        # saved prediction tensor must be bit-identical to the saved target params.
        # This is the strongest oracle invariant — pinning it here isolates regressions
        # in ``predict_step`` from the noisier downstream audio metrics (which absorb
        # Surge XT's per-voice render jitter and would mask a small deviation).
        if experiment_name == _ORACLE_EXPERIMENT:
            target_params = torch.load(
                predictions_dir / f"target-params-{i}.pt", weights_only=True
            )
            assert torch.equal(pred, target_params), f"oracle pred-{i}.pt != target-params-{i}.pt"

    audio_dir = tmp_path / "audio"
    sample_dirs = sorted(d for d in audio_dir.iterdir() if d.is_dir())
    assert [d.name for d in sample_dirs] == [f"sample_{i}" for i in range(NUM_FIXTURE_SAMPLES)]
    # ``target.wav`` is rendered from fixture-truth params and must be audible —
    # silence there would be a real bug. ``pred.wav`` from a 1-step-trained model
    # can legitimately land in a silent region of Surge XT's param space (MPS
    # non-determinism); ``compute_rms`` clamps its denominator so silent pred
    # yields ``cosine_sim = 0`` rather than NaN, and the finite-metric assertion
    # at the end of this test is the real end check.
    for sample_dir in sample_dirs:
        assert (sample_dir / "target.wav").is_file()
        assert (sample_dir / "pred.wav").is_file()
        assert (sample_dir / "spec.png").is_file()
        assert (sample_dir / "params.csv").is_file()

        with AudioFile(str(sample_dir / "target.wav")) as f:
            target_audio = f.read(f.frames)
        target_peak = float(np.abs(target_audio).max())
        assert target_peak > 1e-6, (
            f"{sample_dir.name}/target.wav is silent (peak={target_peak:.2e})"
        )

    metrics_dir = tmp_path / "metrics"
    for metrics_file, expected in METRICS_FILE_EXPECTATIONS.items():
        assert (metrics_dir / metrics_file).is_file(), f"{metrics_file} not found in {metrics_dir}"
        metrics_df = pd.read_csv(metrics_dir / metrics_file)
        assert len(metrics_df) == expected["rows"]
        assert expected["columns"].issubset(metrics_df.columns)
        numeric = metrics_df[sorted(expected["columns"])].to_numpy()
        assert np.isfinite(numeric).all(), f"{metrics_file} contains NaN/Inf:\n{metrics_df}"

    if experiment_name == _ORACLE_EXPERIMENT:
        # Surge XT injects per-voice render jitter (oscillator phase, noise seed)
        # even with bit-identical params, so the audio metrics don't collapse to
        # zero — bounds absorb that jitter while still failing on a real regression.
        per_sample = pd.read_csv(metrics_dir / "metrics.csv")
        bounds = ORACLE_AUDIO_METRIC_BOUNDS
        assert per_sample["mss"].max() < bounds.mss_max, (
            f"oracle mss too high: {per_sample['mss'].tolist()}"
        )
        assert per_sample["wmfcc"].max() < bounds.wmfcc_max, (
            f"oracle wmfcc too high: {per_sample['wmfcc'].tolist()}"
        )
        assert per_sample["sot"].max() < bounds.sot_max, (
            f"oracle sot too high: {per_sample['sot'].tolist()}"
        )
        assert per_sample["rms"].min() > bounds.rms_min, (
            f"oracle rms too low: {per_sample['rms'].tolist()}"
        )


@pytest.mark.requires_vst
@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("WANDB_API_KEY"),
    reason="real W&B round-trip needs WANDB_API_KEY (injected on trusted CI only)",
)
@pytest.mark.parametrize("experiment_name", ["surge/ffn_full"], indirect=True)
def test_train_resumes_from_wandb_resolved_checkpoint(
    tmp_path: Path,
    cfg_surge_xt: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
    experiment_name: str,
) -> None:
    """Training resumes from a ``ckpt_path`` pinned as ``${wandb:...}``, downloaded from the live registry.

    Exercises the exact train-side seam the wandb_checkpoint split protects: ``train.py`` reads
    ``cfg.get("ckpt_path")`` into ``trainer.fit(ckpt_path=...)``. A first one-step run produces a
    real Lightning checkpoint, published to ``tinaudio/synth-setter-citest``; a second run pins
    that artifact via ``${wandb:...}`` and must download + resume it, advancing ``global_step``.
    Proves the resolver works through the real W&B API on the train entrypoint, not just a fake.

    :param tmp_path: Shared output dir; also the workspace root the resolver caches under.
    :param cfg_surge_xt: Surge XT smoke-test training config (one step on the fixture dataset).
    :param monkeypatch: Pins ``SYNTH_SETTER_WORKSPACE`` so the download cache stays under tmp_path.
    :param experiment_name: Pinned to ``surge/ffn_full`` — the artifact id need only round-trip.
    """
    HydraConfig().set_config(cfg_surge_xt)
    _, first = train(cfg_surge_xt)
    step_after_first = first["trainer"].global_step
    ckpt = tmp_path / "checkpoints" / "last.ckpt"
    assert ckpt.is_file(), "first train step did not write last.ckpt"

    # Body runs inside the ``with`` so the resolver downloads before the artifact/run teardown.
    with publish_checkpoint_artifact(
        ckpt, "model-citest-ffn_full-resume", tmp_path / "wandb"
    ) as ref:
        # Contain the resolver's download cache under tmp_path so each run fetches fresh (a warm
        # self-hosted runner must not reuse a stale cached ckpt for the same :latest ref).
        monkeypatch.setenv("SYNTH_SETTER_WORKSPACE", str(tmp_path))
        monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
        operator_workspace.cache_clear()
        register_resolvers()
        with open_dict(cfg_surge_xt):
            cfg_surge_xt.ckpt_path = "${wandb:" + ref + "}"
            cfg_surge_xt.trainer.max_steps = step_after_first + 1

        HydraConfig().set_config(cfg_surge_xt)
        _, second = train(cfg_surge_xt)
        step_after_resume = second["trainer"].global_step

    assert (tmp_path / ".cache" / "checkpoints").is_dir(), "resolver did not download the artifact"
    assert step_after_resume > step_after_first, (
        f"resume did not advance training: before={step_after_first}, after={step_after_resume}"
    )


@pytest.mark.dataloader_multiprocess
@pytest.mark.xdist_group(name="dataloader-multiprocess")
def test_train_fast_dev_run_lance_datamodule(cfg_train_lance: DictConfig) -> None:
    """Run one spawned-worker train, val, and test step from Lance shards.

    Exercises config wiring, ``LanceVSTDataModule`` setup, and real Lance batch
    reads end-to-end through the in-process ``train(cfg)`` entrypoint with
    spawned workers; the Hydra composition path lives on the ``cfg_train_lance``
    fixture. Also pins the
    Dataset-API migration's two e2e-visible contracts on the live datamodule:
    splits open as directory datasets, and a column accepts unsorted fancy
    indices returning rows in the requested order.

    :param cfg_train_lance: Composed ``datamodule=surge_lance`` training config.
    """
    with open_dict(cfg_train_lance):
        cfg_train_lance.datamodule.num_workers = 1
    HydraConfig().set_config(cfg_train_lance)
    _, object_dict = train(cfg_train_lance)

    # Pin the Dataset-API migration e2e: the split the datamodule trained over
    # is a Lance dataset directory, not the legacy single ``.lance`` file.
    train_split = Path(object_dict["datamodule"].dataset_root) / "train.lance"
    assert train_split.is_dir()


def test_train_fit_mode_partial_lance_root_does_not_build_test_split(
    cfg_train_lance: DictConfig,
) -> None:
    """Real training completes without opening a test split during fit.

    :param cfg_train_lance: Composed ``datamodule=surge_lance`` training config.
    """
    dataset_root = Path(cfg_train_lance.datamodule.dataset_root)
    shutil.rmtree(dataset_root / "test.lance")
    with open_dict(cfg_train_lance):
        cfg_train_lance.test = False
        cfg_train_lance.datamodule.num_workers = 0
    HydraConfig().set_config(cfg_train_lance)

    _, object_dict = train(cfg_train_lance)

    with pytest.raises(RuntimeError, match="test split was not built"):
        object_dict["datamodule"].test_dataloader()


@pytest.mark.dataloader_multiprocess
@pytest.mark.xdist_group(name="dataloader-multiprocess")
def test_train_lance_records_dataset_lineage_from_local_spec(
    cfg_train_lance: DictConfig,
    dataset_spec_factory: Callable[..., DatasetSpec],
) -> None:
    """A real Lance training run records its local dataset artifact as a W&B input.

    :param cfg_train_lance: Composed Lance training configuration.
    :param dataset_spec_factory: Factory producing the frozen dataset provenance.
    """
    dataset_root = Path(cfg_train_lance.datamodule.dataset_root)
    write_spec_to_path(
        dataset_spec_factory(
            task_name="lineage-lance",
            train_val_test_sizes=[4, 4, 0],
            r2={"bucket": "intermediate-data"},
            render={"samples_per_shard": 4},
        ),
        dataset_root / "input_spec.json",
    )
    HydraConfig().set_config(cfg_train_lance)
    logger = _RecordingWandbLogger()
    with patch("synth_setter.cli.train.instantiate_loggers", return_value=[logger]):
        train(cfg_train_lance)

    assert logger.used_artifacts == ["data-lineage-lance:lineage-lance-20260520T000000000Z"]


def test_train_same_seed_reproduces_noise_stream(cfg_train_lance: DictConfig) -> None:
    """Two ``train(cfg)`` runs under one ``cfg.seed`` consume identical batch noise.

    Pins the operator-facing seeding contract: ``PrepareBatchCollate`` noise is
    governed by ``seed_everything(cfg.seed, workers=True)``. This isolates the
    in-process stream; spawned-worker stream behavior is covered in
    ``tests/data/test_lance_map_datamodule.py::TestLanceMapDataModuleModes``.

    :param cfg_train_lance: Composed ``datamodule=surge_lance`` training config.
    """
    HydraConfig().set_config(cfg_train_lance)
    with open_dict(cfg_train_lance):
        cfg_train_lance.seed = 1234
        cfg_train_lance.callbacks.noise_capture = {
            "_target_": "tests.helpers.noise_capture.NoiseCaptureCallback"
        }
    runs: list[list[torch.Tensor]] = []
    for _ in range(2):
        NoiseCaptureCallback.captured.clear()
        train(cfg_train_lance)
        assert NoiseCaptureCallback.captured, "callback captured no training batches"
        runs.append(list(NoiseCaptureCallback.captured))
    assert len(runs[0]) == len(runs[1])
    for first, second in zip(runs[0], runs[1], strict=True):
        # atol=rtol=0: the same cfg.seed must reproduce the noise draw bit-for-bit.
        torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


@pytest.mark.fake_vst
@pytest.mark.parametrize("experiment_name", _SURGE_SMOKE_EXPERIMENTS, indirect=True)
@pytest.mark.parametrize("surge_smoke_variant", FAKE_VST_VARIANTS, indirect=True)
def test_train_surge_fake(
    cfg_surge_fake_train: DictConfig,
    surge_smoke_variant: _SurgeSmokeVariant,
    experiment_name: str,
) -> None:
    """Run the Surge smoke training matrix over the fake-plugin Lance splits.

    :param cfg_surge_fake_train: CPU training config for the dataset-format arm under test.
    :param surge_smoke_variant: Lance smoke fixture configuration.
    :param experiment_name: Hydra experiment override the cfg was built from.
    """
    HydraConfig().set_config(cfg_surge_fake_train)
    metric_dict, object_dict = train(cfg_surge_fake_train)

    trainer = object_dict["trainer"]
    assert trainer.global_step >= 1, f"trainer did not advance: global_step={trainer.global_step}"

    train_split = (
        Path(object_dict["datamodule"].dataset_root) / f"train{surge_smoke_variant.split_ext}"
    )
    assert train_split.exists()

    assert_finite_train_loss(metric_dict)

    if experiment_name == _ORACLE_EXPERIMENT:
        _assert_oracle_zero_train_loss(metric_dict)


@pytest.mark.fake_vst
@pytest.mark.parametrize("experiment_name", _SURGE_SMOKE_EXPERIMENTS, indirect=True)
@pytest.mark.parametrize("surge_smoke_variant", FAKE_VST_VARIANTS, indirect=True)
def test_train_eval_surge_fake(
    tmp_path: Path,
    cfg_surge_fake_train: DictConfig,
    cfg_surge_fake_eval: DictConfig,
    surge_smoke_variant: _SurgeSmokeVariant,
    monkeypatch: pytest.MonkeyPatch,
    experiment_name: str,
) -> None:
    """Train on a fake-plugin arm, then verify prediction tensors from checkpoint eval.

    :param tmp_path: The temporary logging path.
    :param cfg_surge_fake_train: CPU training config for the dataset-format arm under test.
    :param cfg_surge_fake_eval: Matching eval config pinned to ``last.ckpt``.
    :param surge_smoke_variant: Lance smoke fixture configuration.
    :param monkeypatch: Stubs render/metrics subprocesses so no real VST host launches.
    :param experiment_name: Hydra experiment override the cfg was built from.
    """
    metric_dict, object_dict = _evaluate_surge_fake_checkpoint(
        cfg_surge_fake_train, cfg_surge_fake_eval, monkeypatch
    )

    _assert_surge_fake_eval_basics(metric_dict, object_dict, surge_smoke_variant)

    predictions_dir = tmp_path / "predictions"
    assert predictions_dir.is_dir()
    assert sorted(path.name for path in predictions_dir.iterdir()) == _prediction_file_names()

    for sample_idx in range(NUM_FIXTURE_SAMPLES):
        pred = torch.load(predictions_dir / f"pred-{sample_idx}.pt", weights_only=True)
        assert torch.isfinite(pred).all(), f"pred-{sample_idx}.pt contains NaN/Inf"

        if experiment_name == _ORACLE_EXPERIMENT:
            target_params = torch.load(
                predictions_dir / f"target-params-{sample_idx}.pt", weights_only=True
            )
            assert torch.equal(pred, target_params), (
                f"oracle pred-{sample_idx}.pt != target-params-{sample_idx}.pt"
            )


@pytest.mark.fake_vst
@pytest.mark.parametrize("experiment_name", _SURGE_SMOKE_EXPERIMENTS, indirect=True)
@pytest.mark.parametrize("surge_smoke_variant", FAKE_VST_VARIANTS, indirect=True)
def test_train_eval_surge_fake_writes_audio_and_metrics_outputs(
    tmp_path: Path,
    cfg_surge_fake_train: DictConfig,
    cfg_surge_fake_eval: DictConfig,
    surge_smoke_variant: _SurgeSmokeVariant,
    monkeypatch: pytest.MonkeyPatch,
    experiment_name: str,
) -> None:
    """Train on a fake-plugin arm, then verify fake render and metrics outputs.

    :param tmp_path: The temporary logging path.
    :param cfg_surge_fake_train: CPU training config for the dataset-format arm under test.
    :param cfg_surge_fake_eval: Matching eval config pinned to ``last.ckpt``.
    :param surge_smoke_variant: Lance smoke fixture configuration.
    :param monkeypatch: Stubs render/metrics subprocesses so no real VST host launches.
    :param experiment_name: Hydra experiment override; parametrizes the train/eval run.
    """
    metric_dict, object_dict = _evaluate_surge_fake_checkpoint(
        cfg_surge_fake_train, cfg_surge_fake_eval, monkeypatch
    )

    _assert_surge_fake_eval_basics(metric_dict, object_dict, surge_smoke_variant)

    audio_dir = tmp_path / "audio"
    sample_dirs = sorted(path for path in audio_dir.iterdir() if path.is_dir())
    assert [path.name for path in sample_dirs] == [
        f"sample_{sample_idx}" for sample_idx in range(NUM_FIXTURE_SAMPLES)
    ]
    for sample_dir in sample_dirs:
        assert (sample_dir / "target.wav").is_file()
        assert (sample_dir / "pred.wav").is_file()
        assert (sample_dir / "spec.png").is_file()
        assert (sample_dir / "params.csv").is_file()

    metrics_dir = tmp_path / "metrics"
    for metrics_file, expected_rows in {
        "aggregated_metrics.csv": 4,
        "metrics.csv": NUM_FIXTURE_SAMPLES,
    }.items():
        assert (metrics_dir / metrics_file).is_file(), f"{metrics_file} not found"
        metrics_df = pd.read_csv(metrics_dir / metrics_file)
        assert len(metrics_df) == expected_rows
        numeric = metrics_df.select_dtypes(include=[np.number]).to_numpy()
        assert np.isfinite(numeric).all(), f"{metrics_file} contains NaN/Inf:\n{metrics_df}"


def _evaluate_surge_fake_checkpoint(
    cfg_train: DictConfig,
    cfg_eval: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object]]:
    """Run the fake-plugin train-to-predict smoke path with faked postprocessing.

    :param cfg_train: CPU training config for the dataset-format arm under test.
    :param cfg_eval: Matching eval config pinned to ``last.ckpt``.
    :param monkeypatch: Stubs postprocessing subprocess dependencies.
    :returns: ``evaluate`` metric and object dictionaries.
    """
    HydraConfig().set_config(cfg_train)
    train(cfg_train)

    assert Path(cfg_eval.ckpt_path).exists()

    monkeypatch.setattr("synth_setter.cli.eval.subprocess.run", _smoke_eval_postprocessing_fake())
    monkeypatch.setattr("synth_setter.cli.eval.vst_headless_wrapper", lambda: object())
    monkeypatch.setattr(
        "synth_setter.cli.eval.as_file",
        lambda _traversable: nullcontext(Path("/fake/headless-wrapper")),
    )

    HydraConfig().set_config(cfg_eval)
    return evaluate(cfg_eval)


def _assert_surge_fake_eval_basics(
    metric_dict: dict[str, object],
    object_dict: dict[str, object],
    variant: _SurgeSmokeVariant,
) -> None:
    """Assert the shared fake-plugin predict-mode eval invariants.

    :param metric_dict: Metrics returned by ``evaluate``.
    :param object_dict: Objects returned by ``evaluate``.
    :param variant: Dataset-format arm selecting the predicted split's suffix.
    """
    assert metric_dict["audio/mss_mean"] == pytest.approx(0.5)
    assert metric_dict["audio/rms_std"] == pytest.approx(0.01)

    dataset_root = getattr(object_dict["datamodule"], "dataset_root")
    assert isinstance(dataset_root, str | os.PathLike)

    test_split = Path(dataset_root) / f"test{variant.split_ext}"
    assert test_split.exists()


def _prediction_file_names() -> list[str]:
    """Return the per-batch files ``PredictionWriter`` writes for the smoke split.

    :returns: Sorted expected prediction filenames.
    """
    return sorted(
        f"{prefix}-{sample_idx}.pt"
        for prefix in _PREDICTION_PT_PREFIXES
        for sample_idx in range(NUM_FIXTURE_SAMPLES)
    )


def test_train_default_checkpoint_callback_is_validation_aligned(cfg_train: DictConfig) -> None:
    """Guard the callback target against stale-metric checkpoint selection.

    :param cfg_train: Configuration whose default callback is instantiated.
    """
    checkpoint = hydra.utils.instantiate(cfg_train.callbacks.model_checkpoint)

    assert isinstance(checkpoint, ValidationAlignedModelCheckpoint)


type _CheckpointScenario = tuple[int, int | float, float, int, int]


@pytest.mark.parametrize("save_last", [True, "link"])
@pytest.mark.parametrize(
    "scenario",
    [
        (5, 3, 2.0, 3, 4),
        (6, 1.0, 2.0, 6, 6),
    ],
)
def test_train_best_checkpoint_contains_metric_producing_weights(
    cfg_train: DictConfig,
    save_last: bool | Literal["link"],
    scenario: _CheckpointScenario,
) -> None:
    """The train entrypoint keeps monitored weights aligned with validation.

    :param cfg_train: Tiny CPU training configuration.
    :param save_last: Recovery checkpoint mode under test.
    :param scenario: Validation and checkpoint cadence with expected selection.
    """
    (
        limit_train_batches,
        val_check_interval,
        expected_score,
        expected_best_step,
        expected_last_step,
    ) = scenario
    with open_dict(cfg_train):
        cfg_train.model = {
            "_target_": "tests.helpers.checkpoint_alignment.ValidationTrajectoryModule"
        }
        cfg_train.datamodule = {
            "_target_": "tests.helpers.checkpoint_alignment.ValidationTrajectoryDataModule"
        }
        cfg_train.callbacks = {
            "model_checkpoint": {
                "_target_": "synth_setter.utils.callbacks.ValidationAlignedModelCheckpoint",
                "dirpath": f"{cfg_train.paths.output_dir}/checkpoints",
                "filename": "step_{step}",
                "monitor": "val/score",
                "mode": "min",
                "save_top_k": 1,
                "save_last": save_last,
                "auto_insert_metric_name": False,
                "every_n_train_steps": 2,
            }
        }
        cfg_train.logger = None
        cfg_train.test = False
        cfg_train.trainer.limit_train_batches = limit_train_batches
        cfg_train.trainer.limit_val_batches = 1
        cfg_train.trainer.num_sanity_val_steps = 0
        cfg_train.trainer.val_check_interval = val_check_interval
        cfg_train.training.val_audio_probe = False
    HydraConfig().set_config(cfg_train)

    _, objects = train(cfg_train)

    checkpoint = objects["trainer"].checkpoint_callback
    assert isinstance(checkpoint, ValidationAlignedModelCheckpoint)
    best = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
    last = torch.load(checkpoint.last_model_path, map_location="cpu", weights_only=False)
    assert checkpoint.best_model_score == expected_score
    assert best["global_step"] == expected_best_step
    assert best["state_dict"]["trained_batches"] == expected_best_step
    assert last["global_step"] == expected_last_step
    assert last["state_dict"]["trained_batches"] == expected_last_step


@pytest.mark.slow
def test_train_mirrors_checkpoints_to_r2_mid_run_when_enabled(
    cfg_train: DictConfig, fake_r2_remote: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prove a periodic upload precedes the final flush and preserves checkpoint bytes.

    :param cfg_train: Tiny CPU training cfg (ksin/ffn, ``save_last``).
    :param fake_r2_remote: Tmp root backing ``r2:`` through the real rclone binary.
    :param monkeypatch: Stubs the R2 auth-ping and wraps the upload to record URIs.
    """
    uploads = _record_successful_r2_uploads(monkeypatch)
    run_id = "train-fixed-run-id"
    recovery_uuids = iter((UUID(int=1), UUID(int=2)))
    monkeypatch.setattr("synth_setter.cli.train.make_wandb_run_id", lambda _config_id: run_id)
    monkeypatch.setattr("synth_setter.cli.train.uuid4", lambda: next(recovery_uuids))
    with open_dict(cfg_train):
        cfg_train.test = False
        cfg_train.trainer.max_epochs = 2
        cfg_train.training.upload_checkpoints_during_training = True
    HydraConfig().set_config(cfg_train)
    train(cfg_train)

    first_uploads = list(uploads)
    assert len(first_uploads) >= 2
    config_id = resolve_run_config_id(cfg_train)
    first_uri = (
        f"r2://{cfg_train.r2.bucket}/checkpoints/{config_id}/{run_id}-{'0' * 31}1/last.ckpt"
    )
    assert {uri for _, uri, _ in first_uploads} == {first_uri}
    assert any(snapshot != first_uploads[-1][2] for _, _, snapshot in first_uploads[:-1])
    assert (fake_r2_remote / first_uri.removeprefix("r2://")).read_bytes() == first_uploads[-1][2]

    train(cfg_train)

    second_uploads = uploads[len(first_uploads) :]
    assert len(second_uploads) >= 2
    second_uri = (
        f"r2://{cfg_train.r2.bucket}/checkpoints/{config_id}/{run_id}-{'0' * 31}2/last.ckpt"
    )
    assert {uri for _, uri, _ in second_uploads} == {second_uri}
    assert second_uri != first_uri
    assert (fake_r2_remote / second_uri.removeprefix("r2://")).read_bytes() == second_uploads[-1][
        2
    ]


@pytest.mark.slow
def test_train_recovers_r2_checkpoint_after_fit_raises(
    cfg_train: DictConfig, fake_r2_remote: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash-time ``last.ckpt`` reaches R2 before ``train()`` re-raises.

    :param cfg_train: Tiny CPU training cfg with a real ``ModelCheckpoint``.
    :param fake_r2_remote: Tmp root backing ``r2:`` through the real rclone binary.
    :param monkeypatch: Stubs the R2 auth-ping and wraps uploads to record their URIs.
    """
    uploads = _record_successful_r2_uploads(monkeypatch)
    with open_dict(cfg_train):
        cfg_train.callbacks.crash_callback = {
            "_target_": "tests.helpers.crash_callback._RaiseOnTrainBatchEnd"
        }
        cfg_train.test = False
        cfg_train.training.upload_checkpoints_during_training = True
    HydraConfig().set_config(cfg_train)

    with pytest.raises(RuntimeError, match="simulated mid-fit crash"):
        train(cfg_train)

    assert uploads
    last_local, last_uri, uploaded_bytes = uploads[-1]
    assert last_local.name == "last.ckpt"
    mirrored = fake_r2_remote / last_uri.removeprefix("r2://")
    assert mirrored.read_bytes() == uploaded_bytes == last_local.read_bytes()
    recovered = last_local.with_name("recovered-last.ckpt")
    r2_io.download_to_path(last_uri, recovered)
    assert recovered.read_bytes() == last_local.read_bytes()
    saved_step = int(torch.load(recovered, map_location="cpu", weights_only=False)["global_step"])

    with open_dict(cfg_train):
        del cfg_train.callbacks.crash_callback
        cfg_train.ckpt_path = str(recovered)
        cfg_train.trainer.max_epochs = 2
        cfg_train.training.upload_checkpoints_during_training = False
    HydraConfig().set_config(cfg_train)
    _, resumed_objects = train(cfg_train)
    assert resumed_objects["trainer"].global_step > saved_step


@pytest.mark.requires_vst
@pytest.mark.slow
@pytest.mark.parametrize("experiment_name", [_ORACLE_EXPERIMENT], indirect=True)
@pytest.mark.parametrize("surge_smoke_variant", REAL_VST_VARIANTS[:1], indirect=True)
def test_train_surge_xt_val_audio_probe_renders_scores_and_uploads(
    cfg_surge_real_train: DictConfig,
    param_spec_name: str,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The val audio probe renders real audio, scores it, and archives it to R2.

    Drives the whole chain for real — Surge XT renders the oracle's predictions through
    the headless wrapper, ``compute_audio_metrics`` scores the wavs, and the real rclone
    binary uploads the snapshot to the fake remote. The oracle predicts ``batch["params"]``
    verbatim, so ``pred.wav`` and ``target.wav`` are renders of identical parameters and
    the metrics land at their perfect-match bounds — which is what makes the returned
    numbers assertable rather than merely present.

    The smoke cfg runs a single validation, so the probe launches but is never harvested
    by a second validation; this waits on the worker directly and asserts its real return
    value. The harvest-and-log half of the loop is covered by ``test_val_audio_probe.py``.

    :param cfg_surge_real_train: Surge XT smoke-test training config (h5 arm, oracle).
    :param param_spec_name: Spec the smoke fixture dataset was rendered with — the probe
        must decode and re-render with the same spec and its registry preset, or the
        prediction rows decode against the wrong parameter layout.
    :param fake_r2_remote: Backs ``r2:`` with the local filesystem; chdirs into tmp_path.
    :param monkeypatch: Neutralizes the R2 auth ping (the fake remote needs no creds).
    :param tmp_path: Doubles as the run's output dir and the fake R2 root.
    """
    import concurrent.futures

    from synth_setter.data.vst.param_spec_registry import plugin_state_paths
    from synth_setter.utils.callbacks import ValAudioProbe

    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *_args, **_kwargs: None)
    assert cfg_surge_real_train.training.val_audio_probe == "auto"

    # fake_r2_remote chdirs into tmp_path, so the render config's repo-relative
    # plugin/preset paths must be absolutized before they are handed to the renderer.
    workspace = operator_workspace()
    probe_samples = 2
    with open_dict(cfg_surge_real_train):
        cfg_surge_real_train.render = {
            "param_spec_name": param_spec_name,
            "plugin_path": str(
                Path(PLUGIN_PATH) if Path(PLUGIN_PATH).is_absolute() else workspace / PLUGIN_PATH
            ),
            "plugin_state_path": str(workspace / plugin_state_paths[param_spec_name]),
            "sample_rate": _SURGE_FIXTURE_SAMPLE_RATE,
            "channels": _SURGE_FIXTURE_CHANNELS,
            "velocity": 100,
            "signal_duration_seconds": _SURGE_FIXTURE_DURATION_SECONDS,
        }
        # Smoke builder leaves the datamodule spec at surge_xt; re-pin to the fixture
        # spec so the configure-time spec-match guard (#1990) passes.
        cfg_surge_real_train.datamodule.param_spec_name = param_spec_name
        cfg_surge_real_train.training.val_audio_probe_samples = probe_samples
        # max_steps=1 stops fit before the end-of-epoch val check; an integer interval
        # forces a real validation after step 1 (the sanity check never stages a probe).
        cfg_surge_real_train.trainer.val_check_interval = 1
        cfg_surge_real_train.trainer.num_sanity_val_steps = 0

    HydraConfig().set_config(cfg_surge_real_train)
    _, object_dict = train(cfg_surge_real_train)

    probes = [cb for cb in object_dict["trainer"].callbacks if isinstance(cb, ValAudioProbe)]
    assert len(probes) == 1, "val_audio_probe=auto did not wire exactly one ValAudioProbe"
    probe = probes[0]
    assert probe._future is not None, "validation ran but no probe was launched"
    concurrent.futures.wait([probe._future], timeout=600)
    metrics = probe._future.result()

    step_dirs = sorted((tmp_path / "val_audio_probe").glob("step-*"))
    assert len(step_dirs) == 1, f"expected one probe dir, got {[d.name for d in step_dirs]}"
    probe_dir = step_dirs[0]

    sample_dirs = sorted((probe_dir / "audio").glob("sample_*"))
    # Staging is capped by the first val batch — the smoke cfg trains at batch_size=1.
    expected_samples = min(probe_samples, cfg_surge_real_train.datamodule.batch_size)
    assert len(sample_dirs) == expected_samples
    for sample_dir in sample_dirs:
        for wav_name in ("pred.wav", "target.wav"):
            wav = sample_dir / wav_name
            assert wav.is_file(), f"{wav} was not rendered"
            assert wav.stat().st_size > 0, f"{wav} is empty"

    assert set(metrics) == {
        f"val_audio/{name}_{stat}"
        for name in ("mss", "wmfcc", "sot", "rms")
        for stat in ("mean", "std")
    }
    bounds = ORACLE_AUDIO_METRIC_BOUNDS
    assert metrics["val_audio/mss_mean"] < bounds.mss_max
    assert metrics["val_audio/wmfcc_mean"] < bounds.wmfcc_max
    assert metrics["val_audio/sot_mean"] < bounds.sot_max
    assert metrics["val_audio/rms_mean"] > bounds.rms_min

    uploaded = fake_r2_remote / cfg_surge_real_train.r2.bucket / "probes"
    landed = sorted(p.relative_to(uploaded).as_posix() for p in uploaded.rglob("*") if p.is_file())
    assert landed, f"probe snapshot never reached {uploaded}"
    # Launch namespace between config_id and step dirs keeps concurrent runs apart (#2230).
    launch_scoped = re.compile(r"^[^/]+/[^/]+-[0-9a-f]{32}/step-\d+/")
    assert all(launch_scoped.match(p) for p in landed), f"snapshot not launch-scoped: {landed}"
    assert any(p.endswith("pred.wav") for p in landed), f"no pred.wav in snapshot: {landed}"
    assert any(p.endswith("aggregated_metrics.csv") for p in landed), f"no metrics: {landed}"
    assert not [p for p in landed if p.endswith(".pt")], (
        f"raw prediction tensors must stay local, but reached R2: {landed}"
    )


@pytest.mark.slow
def test_train_resume_auto_continues_from_newest_sibling_run(
    cfg_train: DictConfig, tmp_path: Path
) -> None:
    """A second launch with ``training.resume=auto`` loads the first run's ``last.ckpt``.

    Both runs share ``max_epochs=1``: the resumed launch restores the completed
    epoch and trains zero additional steps, so its final weights must equal the
    first run's — proving the discovered checkpoint was actually loaded, not
    merely located.

    :param cfg_train: A DictConfig containing a valid training configuration.
    :param tmp_path: Parent of both sibling run output dirs.
    """
    HydraConfig().set_config(cfg_train)
    first_cfg = cfg_train.copy()
    with open_dict(first_cfg):
        first_cfg.paths.output_dir = str(tmp_path / "run-first")
        first_cfg.test = False
    _, first_objects = train(first_cfg)
    # Parameters only: the state dict also holds ``_d``, an *uninitialized*
    # device-tracker buffer whose bytes are garbage and never restored-to-equal.
    first_weights = {
        name: param.detach().clone() for name, param in first_objects["model"].named_parameters()
    }
    first_ckpt = tmp_path / "run-first" / "checkpoints" / "last.ckpt"
    assert first_ckpt.is_file()
    # In-process test runs write no wandb/.hydra state, so plant the identity
    # evidence a real launch leaves behind (config_id falls back to task_name).
    config_id = resolve_run_config_id(first_cfg)
    run_dir_name = f"run-20260716_000000-{config_id}-20260716T000000000Z"
    (tmp_path / "run-first" / "wandb" / run_dir_name).mkdir(parents=True)
    (tmp_path / "run-first" / ".hydra").mkdir()

    second_cfg = cfg_train.copy()
    with open_dict(second_cfg):
        second_cfg.paths.output_dir = str(tmp_path / "run-second")
        second_cfg.test = False
        second_cfg.training.resume = "auto"
        # A _target_-less wandb stub: instantiate_loggers skips it, but the
        # continuity wiring must still pin the recovered id and resume=allow.
        second_cfg.logger = {"wandb": {"id": None, "resume": None, "job_type": ""}}
    _, second_objects = train(second_cfg)

    assert second_objects["cfg"].ckpt_path == str(first_ckpt)
    second_logger_cfg = second_objects["cfg"].logger.wandb
    assert second_logger_cfg.id == f"{config_id}-20260716T000000000Z"
    assert second_logger_cfg.resume == "allow"
    second_weights = dict(second_objects["model"].named_parameters())
    assert set(second_weights) == set(first_weights)
    for name, first_param in first_weights.items():
        assert torch.equal(second_weights[name], first_param), f"{name} diverged"


@pytest.mark.slow
def test_train_resume_require_from_r2_continues_wandb_run(
    cfg_train: DictConfig,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``require`` loads an R2 mirror and continues its recovered W&B run.

    :param cfg_train: A DictConfig containing a valid training configuration.
    :param fake_r2_remote: Tmp root backing ``r2:`` through the real rclone binary.
    :param monkeypatch: Pytest fixture used to bypass the R2 credential probe.
    :param tmp_path: Parent of distinct source and recovery output families.
    """
    HydraConfig().set_config(cfg_train)
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda: None)
    source_cfg = cfg_train.copy()
    with open_dict(source_cfg):
        source_cfg.paths.output_dir = str(tmp_path / "source-runs" / "run-first")
        source_cfg.test = False
    _, source_objects = train(source_cfg)
    source_ckpt = tmp_path / "source-runs" / "run-first" / "checkpoints" / "last.ckpt"
    assert source_ckpt.is_file()
    config_id = resolve_run_config_id(source_cfg)
    run_id = f"{config_id}-20260716T000000000Z"
    mirror = (
        fake_r2_remote
        / "test-bucket"
        / "checkpoints"
        / config_id
        / f"{run_id}-{'a' * 32}"
        / "last.ckpt"
    )
    mirror.parent.mkdir(parents=True)
    mirror.write_bytes(source_ckpt.read_bytes())

    recovery_cfg = cfg_train.copy()
    with open_dict(recovery_cfg):
        recovery_cfg.paths.output_dir = str(tmp_path / "recovery-runs" / "run-second")
        recovery_cfg.test = False
        recovery_cfg.r2.bucket = "test-bucket"
        recovery_cfg.training.resume = "require"
        recovery_cfg.logger = {"wandb": {"id": None, "resume": None, "job_type": ""}}
    _, recovered_objects = train(recovery_cfg)

    recovered_cfg = recovered_objects["cfg"]
    assert recovered_cfg.ckpt_path == str(
        tmp_path / "recovery-runs" / "run-second" / "resume" / "last.ckpt"
    )
    assert recovered_cfg.logger.wandb.id == run_id
    assert recovered_cfg.logger.wandb.resume == "allow"
    source_weights = dict(source_objects["model"].named_parameters())
    recovered_weights = dict(recovered_objects["model"].named_parameters())
    assert set(recovered_weights) == set(source_weights)
    for name, source_param in source_weights.items():
        assert torch.equal(recovered_weights[name], source_param), f"{name} diverged"


def test_train_resume_require_without_checkpoint_raises(
    cfg_train: DictConfig, tmp_path: Path
) -> None:
    """``training.resume=require`` refuses to silently start fresh.

    :param cfg_train: A DictConfig containing a valid training configuration.
    :param tmp_path: Empty run-dir family (no sibling checkpoints).
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.paths.output_dir = str(tmp_path / "run-only")
        cfg_train.test = False
        cfg_train.training.resume = "require"

    with pytest.raises(
        RuntimeError, match=r"training.resume=require found no checkpoint for config_id 'train'"
    ):
        train(cfg_train)


def test_train_resume_rejects_explicit_empty_checkpoint_before_instantiation(
    cfg_train: DictConfig, tmp_path: Path
) -> None:
    """An empty manual checkpoint override fails before the training graph is built.

    :param cfg_train: A DictConfig containing a valid training configuration.
    :param tmp_path: Output directory that remains unpopulated on validation failure.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.paths.output_dir = str(tmp_path / "run-only")
        cfg_train.test = False
        cfg_train.training.resume = "auto"
        cfg_train.ckpt_path = ""

    with pytest.raises(ValueError, match="ckpt_path"):
        train(cfg_train)

    assert not (tmp_path / "run-only").exists()


def test_train_resume_require_reports_r2_degradation(
    cfg_train: DictConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``require`` reports why R2 discovery degraded instead of finding nothing.

    :param cfg_train: A DictConfig containing a valid training configuration.
    :param tmp_path: Empty run-dir family (no sibling checkpoints).
    :param monkeypatch: Pytest fixture used to make the R2 tier unavailable.
    """
    HydraConfig().set_config(cfg_train)

    def _unavailable() -> None:
        raise RuntimeError("no R2 credentials")

    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", _unavailable)
    with open_dict(cfg_train):
        cfg_train.paths.output_dir = str(tmp_path / "run-only")
        cfg_train.test = False
        cfg_train.training.resume = "require"
        cfg_train.r2.bucket = "test-bucket"

    with pytest.raises(RuntimeError, match="no R2 credentials"):
        train(cfg_train)


def test_train_resume_require_without_r2_reports_no_degradation(
    cfg_train: DictConfig, tmp_path: Path
) -> None:
    """``require`` distinguishes no configured R2 tier from a degraded one.

    :param cfg_train: A DictConfig containing a valid training configuration.
    :param tmp_path: Empty run-dir family (no sibling checkpoints).
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.paths.output_dir = str(tmp_path / "run-only")
        cfg_train.test = False
        cfg_train.training.resume = "require"
        cfg_train.r2.bucket = None

    with pytest.raises(RuntimeError, match="found no checkpoint") as exc_info:
        train(cfg_train)

    assert "degraded" not in str(exc_info.value)


def test_train_resume_auto_without_checkpoint_initializes_fresh_run(
    cfg_train: DictConfig, tmp_path: Path
) -> None:
    """``auto`` initializes normally when neither local nor R2 recovery exists.

    :param cfg_train: A DictConfig containing a valid training configuration.
    :param tmp_path: Empty run-dir family (no sibling checkpoints).
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.paths.output_dir = str(tmp_path / "run-only")
        cfg_train.train = False
        cfg_train.test = False
        cfg_train.training.resume = "auto"
        cfg_train.r2.bucket = None

    _, objects = train(cfg_train)

    assert objects["cfg"].ckpt_path is None


def test_train_resume_auto_local_checkpoint_continues_wandb_run(
    cfg_train: DictConfig, tmp_path: Path
) -> None:
    """``auto`` selects a verified local checkpoint and preserves W&B continuity.

    :param cfg_train: A DictConfig containing a valid training configuration.
    :param tmp_path: Parent of the prior and current Hydra output dirs.
    """
    HydraConfig().set_config(cfg_train)
    prior = tmp_path / "run-prior"
    checkpoint = prior / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (prior / ".hydra").mkdir()
    (prior / "wandb" / "run-20260716_000000-train-20260716T000000000Z").mkdir(parents=True)
    with open_dict(cfg_train):
        cfg_train.paths.output_dir = str(tmp_path / "run-current")
        cfg_train.train = False
        cfg_train.test = False
        cfg_train.training.resume = "auto"
        cfg_train.r2.bucket = None
        cfg_train.logger = {"wandb": {"id": None, "resume": None, "job_type": ""}}

    _, objects = train(cfg_train)

    recovered_cfg = objects["cfg"]
    assert recovered_cfg.ckpt_path == str(checkpoint)
    assert recovered_cfg.logger.wandb.id == "train-20260716T000000000Z"
    assert recovered_cfg.logger.wandb.resume == "allow"


def test_train_resume_auto_hydra_only_checkpoint_uses_fresh_wandb_run(
    cfg_train: DictConfig, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A Hydra-verified checkpoint without a run ID starts a fresh W&B run.

    :param cfg_train: A DictConfig containing a valid training configuration.
    :param tmp_path: Parent of the prior and current Hydra output dirs.
    :param caplog: Pytest log capture fixture enabling the resume diagnostic.
    """
    HydraConfig().set_config(cfg_train)
    prior = tmp_path / "run-prior"
    checkpoint = prior / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    hydra_dir = prior / ".hydra"
    hydra_dir.mkdir()
    (hydra_dir / "hydra.yaml").write_text(
        "hydra:\n  runtime:\n    choices:\n      experiment: null\n"
    )
    (hydra_dir / "config.yaml").write_text("task_name: train\n")
    with open_dict(cfg_train):
        cfg_train.paths.output_dir = str(tmp_path / "run-current")
        cfg_train.train = False
        cfg_train.test = False
        cfg_train.training.resume = "auto"
        cfg_train.r2.bucket = None
        cfg_train.logger = {"wandb": {"id": None, "resume": None, "job_type": ""}}

    with caplog.at_level(logging.INFO, logger="synth_setter.cli.train"):
        _, objects = train(cfg_train)

    recovered_cfg = objects["cfg"]
    assert recovered_cfg.ckpt_path == str(checkpoint)
    assert recovered_cfg.logger.wandb.id.startswith("train-")
    assert recovered_cfg.logger.wandb.resume is None


@pytest.mark.slow
def test_train_resume_auto_without_checkpoint_starts_fresh(
    cfg_train: DictConfig, tmp_path: Path
) -> None:
    """``training.resume=auto`` with nothing to resume trains from scratch.

    :param cfg_train: A DictConfig containing a valid training configuration.
    :param tmp_path: Empty run-dir family (no sibling checkpoints).
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.paths.output_dir = str(tmp_path / "run-only")
        cfg_train.test = False
        cfg_train.training.resume = "auto"

    metric_dict, object_dict = train(cfg_train)

    assert "train/loss" in metric_dict
    assert object_dict["cfg"].ckpt_path is None


@pytest.mark.slow
def test_train_resume_auto_hydra_evidence_sibling_resumes_with_fresh_run_id(
    cfg_train: DictConfig, tmp_path: Path
) -> None:
    """A wandb-less sibling proven by ``.hydra`` state resumes without id reuse.

    Covers the ``decision.wandb_run_id is None`` branch through the real
    entrypoint: ``ckpt_path`` is set, but no wandb continuity is pinned.

    :param cfg_train: A DictConfig containing a valid training configuration.
    :param tmp_path: Parent of both sibling run output dirs.
    """
    HydraConfig().set_config(cfg_train)
    first_cfg = cfg_train.copy()
    with open_dict(first_cfg):
        first_cfg.paths.output_dir = str(tmp_path / "run-first")
        first_cfg.test = False
    train(first_cfg)
    first_ckpt = tmp_path / "run-first" / "checkpoints" / "last.ckpt"
    assert first_ckpt.is_file()
    # Identity via recorded Hydra state only (no wandb dir): config_id falls
    # back to task_name for this experiment-less composition.
    hydra_dir = tmp_path / "run-first" / ".hydra"
    hydra_dir.mkdir()
    (hydra_dir / "hydra.yaml").write_text(
        "hydra:\n  runtime:\n    choices:\n      experiment: null\n"
    )
    (hydra_dir / "config.yaml").write_text(f"task_name: {resolve_run_config_id(first_cfg)}\n")

    second_cfg = cfg_train.copy()
    with open_dict(second_cfg):
        second_cfg.paths.output_dir = str(tmp_path / "run-second")
        second_cfg.test = False
        second_cfg.training.resume = "auto"
        second_cfg.logger = {"wandb": {"id": None, "resume": None, "job_type": ""}}
    _, second_objects = train(second_cfg)

    assert second_objects["cfg"].ckpt_path == str(first_ckpt)
    second_logger_cfg = second_objects["cfg"].logger.wandb
    # A fresh id is minted (no recovered one) and continuity is NOT pinned.
    assert second_logger_cfg.id is not None
    assert second_logger_cfg.resume is None
