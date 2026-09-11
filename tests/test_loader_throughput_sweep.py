"""Contract tests for the loader-throughput sweep and its enabling config.

Pins the ``batch_size_fn`` shim ``ThroughputMonitor`` needs, that the
``throughput`` callback group instantiates the two built-in monitors, and
that both sweep YAMLs keep the shape ``wandb agent`` executes.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
import yaml
from lightning.pytorch.callbacks import DeviceStatsMonitor, ThroughputMonitor
from omegaconf import OmegaConf

from synth_setter.utils.callbacks import batch_sample_count

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SWEEP_DIR = _REPO_ROOT / "sweeps"
_CALLBACK_CFG = _REPO_ROOT / "src/synth_setter/configs/callbacks/throughput.yaml"


def test_batch_sample_count_reads_leading_dim_of_dict_batch() -> None:
    """The sample count is the leading dim of the batch's first tensor."""
    batch = {"params": torch.zeros(1024, 40), "noise": torch.zeros(1024, 40)}
    assert batch_sample_count(batch) == 1024


def test_batch_sample_count_handles_a_bare_tensor_batch() -> None:
    """A tensor batch (no mapping) still yields its leading dimension."""
    assert batch_sample_count(torch.zeros(2048, 8)) == 2048


def test_throughput_callback_group_builds_the_two_builtin_monitors() -> None:
    """The config composes to a live ThroughputMonitor + DeviceStatsMonitor."""
    cfg = OmegaConf.load(_CALLBACK_CFG)
    callbacks = [hydra.utils.instantiate(node) for node in cfg.values()]
    kinds = {type(cb) for cb in callbacks}
    assert kinds == {ThroughputMonitor, DeviceStatsMonitor}


def _assert_common_sweep_shape(cfg: dict) -> None:
    assert cfg["program"] == "src/synth_setter/cli/train.py"
    assert cfg["method"] == "grid"
    assert cfg["metric"] == {"goal": "maximize", "name": "train/device/samples_per_sec"}
    cmd = cfg["command"]
    assert cmd[:2] == ["python", "src/synth_setter/cli/train.py"]
    assert "${args_no_hyphens}" in cmd
    assert "experiment=surge/flow_simple_440k" in cmd
    assert "callbacks=throughput" in cmd
    assert "trainer.max_steps=200" in cmd
    # A pre-hydrated local dataset_root with the download URI nulled keeps the
    # 509 GiB dataset from being re-fetched per trial.
    assert "datamodule.download_dataset_root_uri=null" in cmd
    assert any(t.startswith("datamodule.dataset_root=") for t in cmd)


def test_mapstyle_sweep_grids_the_worker_pool_axes() -> None:
    """Map-style sweep varies workers/batch/ot as the sole production loader path."""
    cfg = yaml.safe_load((_SWEEP_DIR / "loader_throughput_mapstyle.yaml").read_text())
    _assert_common_sweep_shape(cfg)
    assert set(cfg["parameters"]) == {
        "datamodule.num_workers",
        "datamodule.batch_size",
        "datamodule.ot",
    }
    assert cfg["parameters"]["datamodule.num_workers"]["values"] == [0, 12, 24]
