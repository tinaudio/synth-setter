import os
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from scripts.compute_audio_metrics import main as compute_audio_metrics_main
from scripts.predict_vst_audio import main as predict_vst_audio_main
from src.eval import evaluate
from src.train import train
from tests.helpers.run_if import RunIf

# TODO(#39): replace hardcoded accelerator overrides with --accelerator pytest flag
# TODO(#40): add @pytest.mark.ram gate for memory-intensive CPU tests test_train_fast_dev_run


def test_train_fast_dev_run_tiny_model_tiny_data(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step with small batch size, no compile.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        # Prevent CPU unittest OOM by shrinking model,
        # batch, training example, dataset size.
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.data.batch_size = 32
        cfg_train.model.net.channels = 4
        cfg_train.model.net.encoder_blocks = 1
        cfg_train.model.net.trunk_blocks = 1
        cfg_train.model.net.hidden_dim = 32
        cfg_train.data.signal_length = 64
        cfg_train.data.train_val_test_sizes = [4, 4, 4]
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
        cfg_train.data.batch_size = 32
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


# TODO: fix val_check_interval incompatibility with check_val_every_n_epoch=None (#47)
@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_train: DictConfig) -> None:
    """Train 1 epoch with validation loop twice per epoch.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "gpu"
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
        # Integer limits avoid the `fraction * batches < 1` error when the
        # fixture's fractions (0.01, 0.1) shrink too far under DDP sharding.
        cfg_train.trainer.limit_train_batches = 1
        cfg_train.trainer.limit_val_batches = 1
        cfg_train.trainer.limit_test_batches = 1
        # Shrink model, batch, and dataset to keep DDP-on-CPU fast.
        cfg_train.data.batch_size = 2
        cfg_train.data.signal_length = 64
        cfg_train.data.train_val_test_sizes = [4, 4, 4]
        cfg_train.model.net.channels = 4
        cfg_train.model.net.encoder_blocks = 1
        cfg_train.model.net.trunk_blocks = 1
        cfg_train.model.net.hidden_dim = 32
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
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.seed = 42
        cfg_train.trainer.deterministic = True

    HydraConfig().set_config(cfg_train)
    metric_dict_1, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_train.trainer.max_epochs = 2

    metric_dict_2, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "epoch_001.ckpt" in files
    assert "epoch_002.ckpt" not in files

    # `ksin_ff_module.training_step` logs `train/loss` with `on_step=True, on_epoch=True`, which
    # populates `train/loss_epoch` in `trainer.callback_metrics`. `validation_step` logs `val/loss`
    # with `on_epoch=True` only. Resuming for another epoch should drive both losses down, so we
    # expect strict decrease (note: reversed direction vs. the legacy `train/acc < ...` assertion).
    assert metric_dict_1["train/loss_epoch"] > metric_dict_2["train/loss_epoch"]
    assert metric_dict_1["val/loss"] > metric_dict_2["val/loss"]


@pytest.mark.gpu
@RunIf(min_gpus=1)
def test_train_surge_xt_one_step(cfg_surge_xt: DictConfig) -> None:
    """Run one training step of the Surge XT flow-matching model on the 5-sample fixture.

    :param cfg_surge_xt: One-step Surge XT training config.
    """
    HydraConfig().set_config(cfg_surge_xt)
    _, object_dict = train(cfg_surge_xt)
    assert object_dict["trainer"].global_step == 1


@pytest.mark.gpu
@RunIf(min_gpus=1)
def test_train_eval_surge_xt(
    tmp_path: Path, cfg_surge_xt: DictConfig, cfg_surge_xt_eval: DictConfig
) -> None:
    """Train Surge XT for one step, then run standalone eval on the saved checkpoint.

    :param tmp_path: The temporary logging path.
    :param cfg_surge_xt: One-step Surge XT training config.
    :param cfg_surge_xt_eval: Matching eval config (ckpt_path set by this test).
    """
    HydraConfig().set_config(cfg_surge_xt)
    train(cfg_surge_xt)

    ckpt_path = tmp_path / "checkpoints" / "last.ckpt"
    assert ckpt_path.exists()

    with open_dict(cfg_surge_xt_eval):
        cfg_surge_xt_eval.ckpt_path = str(ckpt_path)
        cfg_surge_xt_eval.mode = "predict"

    HydraConfig().set_config(cfg_surge_xt_eval)
    evaluate(cfg_surge_xt_eval)

    # `PredictionWriter` (`src/utils/callbacks.py:332`) with `write_interval=batch` saves three
    # tensors per predict batch: `pred-{i}.pt`, `target-audio-{i}.pt`, `target-params-{i}.pt`.
    # With `limit_predict_batches=1` we expect exactly one batch's worth of files.
    predictions_dir = tmp_path / "predictions"
    assert predictions_dir.is_dir()
    assert sorted(p.name for p in predictions_dir.iterdir()) == [
        "pred-0.pt",
        "target-audio-0.pt",
        "target-params-0.pt",
    ]

    # `predict_vst_audio.py` defaults to `plugins/Surge XT.vst3` relative to CWD (= repo root).
    # Mirror the main tree's symlink to the system VST bundle so the test runs without
    # committing a binary.
    repo_root = Path(cfg_surge_xt.paths.root_dir)
    plugin_link = repo_root / "plugins" / "Surge XT.vst3"
    plugin_link.parent.mkdir(parents=True, exist_ok=True)
    if not plugin_link.exists():
        plugin_link.symlink_to("/usr/lib/vst3/Surge XT.vst3")

    # Render predicted params through the Surge XT VST to per-sample audio directories.
    # `-t` (`--rerender_target`) re-synthesizes target.wav from the stored target_params instead
    # of the saved target audio. Also works around an `UnboundLocalError` in
    # `scripts/predict_vst_audio.py:220` where `target_synth_params` is referenced in the default
    # path without being defined outside the `rerender_target` branch.
    audio_dir = tmp_path / "audio"
    runner = CliRunner()
    result = runner.invoke(
        predict_vst_audio_main,
        [str(predictions_dir), str(audio_dir), "-t"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    # batch_size=2 * limit_predict_batches=1 → 2 rendered samples.
    sample_dirs = sorted(d for d in audio_dir.iterdir() if d.is_dir())
    assert [d.name for d in sample_dirs] == ["sample_0", "sample_1"]
    for sample_dir in sample_dirs:
        assert (sample_dir / "target.wav").is_file()
        assert (sample_dir / "pred.wav").is_file()
        assert (sample_dir / "spec.png").is_file()
        assert (sample_dir / "params.csv").is_file()

    # Compute audio distance metrics (MSS, wMFCC, SOT, RMS) on the rendered pairs.
    metrics_dir = tmp_path / "metrics"
    result = runner.invoke(
        compute_audio_metrics_main,
        [str(audio_dir), str(metrics_dir), "-w", "1"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    metrics_csv = metrics_dir / "metrics.csv"
    assert metrics_csv.is_file()
    assert (metrics_dir / "aggregated_metrics.csv").is_file()

    metrics_df = pd.read_csv(metrics_csv)
    assert len(metrics_df) == 2
    assert {"mss", "wmfcc", "sot", "rms"}.issubset(metrics_df.columns)
