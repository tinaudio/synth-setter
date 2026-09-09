"""Entrypoint integration coverage for grouped parameter projection training."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

from synth_setter.cli.train import train
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.models.components.transformer import GroupedParameterProjection
from synth_setter.workspace import operator_workspace
from tests.helpers.lance_fixtures import write_lance_shard


@pytest.fixture(autouse=True)
def _clear_hydra_state() -> Iterator[None]:
    """Isolate Hydra's process-global composition state across test outcomes.

    :yields: Control to the test with a clean Hydra singleton.

    :ytype: None
    """
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def _write_tiny_lance_dataset(dataset_root: Path) -> None:
    """Write deterministic production-shaped Lance splits for one training step.

    :param dataset_root: Directory receiving train, validation, and test splits.
    """
    spec = param_specs["surge_4"]
    mel_shape = (2, 128, 401)
    dataset_root.mkdir()
    for seed, split in enumerate(("train", "val", "test")):
        rng = np.random.default_rng(seed)
        write_lance_shard(
            dataset_root / f"{split}.lance",
            {
                "audio": rng.uniform(-1.0, 1.0, (2, 2, 64)).astype(np.float16),
                "mel_spec": rng.standard_normal((2, *mel_shape)).astype(np.float32),
                "param_array": rng.random((2, spec.encoded_width)).astype(np.float32),
            },
        )
    np.savez(
        dataset_root / "stats.npz",
        mean=np.zeros(mel_shape, dtype=np.float32),
        std=np.ones(mel_shape, dtype=np.float32),
    )
    (dataset_root / "dataset.complete").touch()


def _grouped_train_config(tmp_path: Path) -> DictConfig:
    """Compose the shipped grouped projection and shrink only the smoke workload.

    :param tmp_path: Isolated dataset, checkpoint, and log root.
    :returns: Hydra config that runs one real flow-training step on CPU.
    """
    dataset_root = tmp_path / "lance-data"
    _write_tiny_lance_dataset(dataset_root)
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=surge/flow_simple",
                "datamodule=surge_lance",
                "synth=surge_4",
                "model/projection=grouped",
                "trainer=cpu",
            ],
        )
    with open_dict(cfg):
        cfg.seed = 3273
        cfg.paths.root_dir = str(operator_workspace())
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.logger = None
        cfg.test = False
        cfg.training.val_audio_probe = False
        cfg.datamodule.dataset_root = str(dataset_root)
        cfg.datamodule.download_dataset_root_uri = None
        cfg.datamodule.batch_size = 2
        cfg.datamodule.num_workers = 0
        cfg.datamodule.pin_memory = False
        cfg.datamodule.ot = False
        cfg.model.compile = False
        cfg.model.scheduler = None
        cfg.model.encoder.d_model = 16
        cfg.model.encoder.n_heads = 1
        cfg.model.encoder.n_layers = 1
        cfg.model.encoder.n_conditioning_outputs = 1
        cfg.model.vector_field.d_model = 16
        cfg.model.vector_field.d_ff = 16
        cfg.model.vector_field.num_heads = 1
        cfg.model.vector_field.num_layers = 1
        cfg.model.validation_sample_steps = 1
        cfg.callbacks.model_checkpoint.monitor = None
        cfg.callbacks.model_checkpoint.save_last = True
        if "lr_monitor" in cfg.callbacks:
            del cfg.callbacks.lr_monitor
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.precision = "32-true"
        cfg.trainer.max_steps = 1
        cfg.trainer.min_steps = 1
        cfg.trainer.limit_train_batches = 1
        cfg.trainer.limit_val_batches = 1
        cfg.trainer.num_sanity_val_steps = 0
        cfg.trainer.enable_model_summary = False
        cfg.trainer.deterministic = True
    return cfg


def test_train_grouped_projection_hydra_path_writes_loadable_checkpoint(tmp_path: Path) -> None:
    """Train the grouped Hydra selection through the real flow entrypoint and Lance loader.

    :param tmp_path: Isolated dataset, checkpoint, and log root.
    """
    cfg = _grouped_train_config(tmp_path)
    HydraConfig().set_config(cfg)

    metric_dict, object_dict = train(cfg)

    projection = object_dict["model"].vector_field.projection
    assert isinstance(projection, GroupedParameterProjection)
    assert projection.num_tokens == sum(1 for _ in param_specs["surge_4"].encoded_slices())
    assert object_dict["trainer"].global_step == 1
    assert torch.isfinite(metric_dict["train/loss"])
    checkpoint = tmp_path / "checkpoints" / "last.ckpt"
    assert checkpoint.is_file()

    restored = instantiate(cfg.model)
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored.load_state_dict(checkpoint_state["state_dict"], strict=True)
    trained = object_dict["model"].cpu().eval()
    restored.eval()
    generator = torch.Generator().manual_seed(3273)
    params = torch.randn(2, param_specs["surge_4"].encoded_width, generator=generator)
    time = torch.rand(2, 1, generator=generator)
    conditioning = torch.randn(2, 16, generator=generator)
    with torch.no_grad():
        trained_output = trained.vector_field(params, time, conditioning)
        restored_output = restored.vector_field(params, time, conditioning)
    torch.testing.assert_close(restored_output, trained_output, atol=0.0, rtol=0.0)
