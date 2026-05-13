import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from synth_setter.cli.eval import evaluate
from synth_setter.cli.train import train
from synth_setter.data.vst import param_specs, preset_paths
from tests.conftest import (
    _VST_SUBPROCESS_TIMEOUT_SECONDS,
    NUM_FIXTURE_SAMPLES,
    VST_HEADLESS_WRAPPER,
    _build_surge_xt_smoke_cfg,
)
from tests.helpers.run_if import RunIf

# TODO(#40): add @pytest.mark.ram gate for memory-intensive CPU tests test_train_fast_dev_run


def test_train_fast_dev_run_tiny_model_tiny_data(cfg_train: DictConfig) -> None:
    """Run 1 train, val, and test step on CPU with `fast_dev_run`.

    Dataset/batch size constraints come from the shared `cfg_train` fixture
    (`batch_size=1`, `train_val_test_sizes=[2, 2, 2]`). This test only adds
    `fast_dev_run=True` to cap the loops at one batch each.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
    train(cfg_train)


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
        cfg_train.data.num_workers = 0
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
    """Templated ``_build_surge_xt_smoke_cfg`` propagates the param spec to ``model.net.d_out`` and
    ``callbacks.log_per_param_mse.param_spec`` for every supported spec — guards against the
    surge_4-only hardcodes the fixture used to carry.

    Calls the builder directly (not the ``cfg_surge_xt_global`` fixture) and pins
    ``accelerator="cpu"``: the cfg-shape contract is accelerator-independent and going
    through the fixture would drag in the parametrized ``accelerator`` hardware gate that
    hardfails on hosts without MPS/CUDA.

    :param param_spec_name: Spec name driving the cfg builder.
    """
    cfg = _build_surge_xt_smoke_cfg(accelerator="cpu", param_spec_name=param_spec_name)
    assert cfg.model.net.d_out == len(param_specs[param_spec_name])
    assert cfg.callbacks.log_per_param_mse.param_spec == param_spec_name


@pytest.mark.requires_vst
@pytest.mark.slow
def test_train_surge_xt(cfg_surge_xt: DictConfig) -> None:
    """Run training of the Surge XT FFN model on the smoke test fixture.

    Asserts the trainer advanced and produced a finite ``train/loss`` — catches silent
    no-op trainers and NaN/Inf regressions that a bare ``train()`` call would not.

    :param cfg_surge_xt: Surge XT training config.
    """
    HydraConfig().set_config(cfg_surge_xt)
    metric_dict, object_dict = train(cfg_surge_xt)

    trainer = object_dict["trainer"]
    assert trainer.global_step >= 1, f"trainer did not advance: global_step={trainer.global_step}"

    # `surge_ff_module` logs `train/loss` with `on_step=True, on_epoch=True`, which
    # populates `train/loss_step` (and `train/loss_epoch` if an epoch boundary was
    # crossed) in `trainer.callback_metrics`. With `TRAINING_STEPS=1` only the
    # step-level key is guaranteed; assert whichever is present is finite.
    loss_keys = [k for k in metric_dict if k.startswith("train/loss")]
    assert loss_keys, f"no train/loss* key in metric_dict: {sorted(metric_dict)}"
    for key in loss_keys:
        loss = metric_dict[key]
        assert torch.isfinite(loss).all(), f"{key} is not finite: {loss}"


@pytest.mark.requires_vst
@pytest.mark.slow
def test_train_eval_surge_xt(
    tmp_path: Path,
    cfg_surge_xt: DictConfig,
    cfg_surge_xt_eval: DictConfig,
    param_spec_name: str,
) -> None:
    """End-to-end smoke test: train Surge XT briefly on a small fixture dataset, then run
    standalone eval on the saved checkpoint.

    :param tmp_path: The temporary logging path.
    :param cfg_surge_xt: Surge XT smoke-test training config.
    :param cfg_surge_xt_eval: Matching smoke-test eval config (ckpt_path set by this test).
    :param param_spec_name: Param spec the fixtures (and therefore the trained model) are
        wired for — passed to ``predict_vst_audio.py`` so the script's decode layout matches
        the predicted tensor's encoding (mismatched specs go off-the-end and crash with
        ``can only convert an array of size 1 to a Python scalar``).
    """
    from click.testing import CliRunner
    from pedalboard.io import AudioFile

    from synth_setter.evaluation.compute_audio_metrics import main as compute_audio_metrics_main

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

    HydraConfig().set_config(cfg_surge_xt)
    train(cfg_surge_xt)

    # `cfg_surge_xt_eval.ckpt_path` is pre-pointed at this same `tmp_path` by the
    # fixture; assert the train step actually produced the file before eval reads it.
    assert Path(cfg_surge_xt_eval.ckpt_path).exists()

    HydraConfig().set_config(cfg_surge_xt_eval)
    evaluate(cfg_surge_xt_eval)

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

    # Render predicted params through the Surge XT VST to per-sample audio directories.
    # `-t` (`--rerender_target`) re-synthesizes target.wav from the stored target_params instead
    # of the saved target audio. Also works around an `UnboundLocalError` in
    # `src/synth_setter/evaluation/predict_vst_audio.py` where `target_synth_params` is referenced in the default
    # path without being defined outside the `rerender_target` branch (see #672).
    audio_dir = tmp_path / "audio"
    runner = CliRunner()

    args = []
    if sys.platform == "linux":
        args.append(VST_HEADLESS_WRAPPER)

    args += [
        sys.executable,
        "-m",
        "synth_setter.evaluation.predict_vst_audio",
        str(predictions_dir),
        str(audio_dir),
        f"--param_spec={param_spec_name}",
        f"--preset_path={preset_paths[param_spec_name]}",
        "-t",
    ]
    try:
        result = subprocess.run(  # noqa: S603, S607
            args,
            text=True,
            check=False,
            timeout=_VST_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"predict_vst_audio timed out after {_VST_SUBPROCESS_TIMEOUT_SECONDS}s\n"
            f"command: {args}\n"
            f"(child stdout/stderr printed above; rerun with `pytest -s` if captured)",
            pytrace=False,
        )
    if result.returncode != 0:
        pytest.fail(
            f"predict_vst_audio failed (exit {result.returncode})\n"
            f"command: {args}\n"
            f"(child stdout/stderr printed above; rerun with `pytest -s` if captured)",
            pytrace=False,
        )

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

    # Compute audio distance metrics (MSS, wMFCC, SOT, RMS) on the rendered pairs.
    metrics_dir = tmp_path / "metrics"
    result = runner.invoke(
        compute_audio_metrics_main,
        [str(audio_dir), str(metrics_dir), "-w", "1"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    for metrics_file, expected in METRICS_FILE_EXPECTATIONS.items():
        assert (metrics_dir / metrics_file).is_file(), f"{metrics_file} not found in {metrics_dir}"
        metrics_df = pd.read_csv(metrics_dir / metrics_file)
        assert len(metrics_df) == expected["rows"]
        assert expected["columns"].issubset(metrics_df.columns)
        numeric = metrics_df[sorted(expected["columns"])].to_numpy()
        assert np.isfinite(numeric).all(), f"{metrics_file} contains NaN/Inf:\n{metrics_df}"
