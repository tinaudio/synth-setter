import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from click.testing import CliRunner
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from pedalboard.io import AudioFile

from scripts.compute_audio_metrics import main as compute_audio_metrics_main
from src.eval import evaluate
from src.train import train
from tests.helpers.run_if import RunIf

# TODO(#39): replace hardcoded accelerator overrides with --accelerator pytest flag
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
    _, _ = train(cfg_train)
    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_train.trainer.max_epochs = 2

    _, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "epoch_001.ckpt" in files
    assert "epoch_002.ckpt" not in files


@pytest.mark.slow
def test_train_surge_xt(cfg_surge_xt: DictConfig) -> None:
    """Run training of the Surge XT flow-matching model on the smoke test fixture.

    :param cfg_surge_xt: Surge XT training config.
    """
    HydraConfig().set_config(cfg_surge_xt)
    train(cfg_surge_xt)


@pytest.mark.slow
def test_train_eval_surge_xt(
    tmp_path: Path, cfg_surge_xt: DictConfig, cfg_surge_xt_eval: DictConfig
) -> None:
    """End-to-end smoke test: train Surge XT briefly on a small fixture dataset, then run
    standalone eval on the saved checkpoint.

    :param tmp_path: The temporary logging path.
    :param cfg_surge_xt: Surge XT smoke-test training config.
    :param cfg_surge_xt_eval: Matching smoke-test eval config (ckpt_path set by this test).
    """
    NUM_FIXTURE_SAMPLES = 5
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
    with open_dict(cfg_surge_xt_eval):
        # Eval on the same training set the smoke run used — this is a wiring smoke test,
        # not a generalization check, so train/predict overlap is intentional.
        cfg_surge_xt_eval.data.predict_file = "tests/fixtures/surge_xt/train.h5"
    evaluate(cfg_surge_xt_eval)

    # `PredictionWriter` (`src/utils/callbacks.py:332`) with `write_interval=batch` saves three
    # tensors per predict batch: `pred-{i}.pt`, `target-audio-{i}.pt`, `target-params-{i}.pt`.
    predictions_dir = tmp_path / "predictions"
    assert predictions_dir.is_dir()
    # Note that the fixture has 5 examples but ShiftedBatchSampler drops a batch per epoch,
    # so we get 4 predictions + targets from the first epoch and 1 from the second.
    assert sorted(p.name for p in predictions_dir.iterdir()) == [
        "pred-0.pt",
        "pred-1.pt",
        "pred-2.pt",
        "pred-3.pt",
        "pred-4.pt",
        "target-audio-0.pt",
        "target-audio-1.pt",
        "target-audio-2.pt",
        "target-audio-3.pt",
        "target-audio-4.pt",
        "target-params-0.pt",
        "target-params-1.pt",
        "target-params-2.pt",
        "target-params-3.pt",
        "target-params-4.pt",
    ]

    for i in range(NUM_FIXTURE_SAMPLES):
        pred = torch.load(predictions_dir / f"pred-{i}.pt", weights_only=True)
        assert torch.isfinite(pred).all(), f"pred-{i}.pt contains NaN/Inf"

    # Render predicted params through the Surge XT VST to per-sample audio directories.
    # `-t` (`--rerender_target`) re-synthesizes target.wav from the stored target_params instead
    # of the saved target audio. Also works around an `UnboundLocalError` in
    # `scripts/predict_vst_audio.py:220` where `target_synth_params` is referenced in the default
    # path without being defined outside the `rerender_target` branch.
    audio_dir = tmp_path / "audio"
    runner = CliRunner()

    # Bootstraps Xvfb + xsettingsd + dbus for VST3 plugin init; resolved relative
    # to the container WORKDIR (``/home/build/synth-setter``) baked in the image.
    # X11 wrapping lives at the audio-rendering boundary (this subprocess call),
    # not at the container entrypoint — the click CLI stays X11-agnostic so idle
    # and passthrough don't pay the Xvfb startup cost.
    VST_HEADLESS_WRAPPER = "scripts/run-linux-vst-headless.sh"
    args = []
    if sys.platform == "linux":
        args.append(VST_HEADLESS_WRAPPER)

    args += [
        sys.executable,
        "scripts/predict_vst_audio.py",
        str(predictions_dir),
        str(audio_dir),
        "-t",
    ]
    subprocess.check_call(args)  # noqa: S603, S607

    sample_dirs = sorted(d for d in audio_dir.iterdir() if d.is_dir())
    assert [d.name for d in sample_dirs] == [
        "sample_0",
        "sample_1",
        "sample_2",
        "sample_3",
        "sample_4",
    ]
    # ~-80 dBFS — below this, librosa RMS norms underflow and `compute_rms`
    # produces 0/0 → NaN (see `scripts/compute_audio_metrics.py:227`).
    SILENCE_PEAK_THRESHOLD = 1e-4
    for sample_dir in sample_dirs:
        assert (sample_dir / "target.wav").is_file()
        assert (sample_dir / "pred.wav").is_file()
        assert (sample_dir / "spec.png").is_file()
        assert (sample_dir / "params.csv").is_file()

        for wav_name in ("target.wav", "pred.wav"):
            with AudioFile(str(sample_dir / wav_name)) as f:
                audio = f.read(f.frames)
            peak = float(np.abs(audio).max())
            assert peak > SILENCE_PEAK_THRESHOLD, (
                f"{sample_dir.name}/{wav_name} is silent (peak={peak:.2e})"
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
