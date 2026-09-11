"""Shared fixtures for grouped-projection entrypoint tests."""

from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.workspace import operator_workspace
from tests.helpers.lance_fixtures import write_lance_shard

GROUPED_PROJECTION_DATASET_ROWS = 2
GROUPED_PROJECTION_SEED = 3273


def write_grouped_projection_lance_dataset(dataset_root: Path) -> None:
    """Write deterministic production-shaped Lance splits for entrypoint tests.

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
                "audio": rng.uniform(-1.0, 1.0, (GROUPED_PROJECTION_DATASET_ROWS, 2, 64)).astype(
                    np.float16
                ),
                "mel_spec": rng.standard_normal(
                    (GROUPED_PROJECTION_DATASET_ROWS, *mel_shape)
                ).astype(np.float32),
                "param_array": rng.random(
                    (GROUPED_PROJECTION_DATASET_ROWS, spec.encoded_width)
                ).astype(np.float32),
            },
        )
    np.savez(
        dataset_root / "stats.npz",
        mean=np.zeros(mel_shape, dtype=np.float32),
        std=np.ones(mel_shape, dtype=np.float32),
    )
    (dataset_root / "dataset.complete").touch()


def build_grouped_projection_config(
    tmp_path: Path, *, config_name: str, projection_name: str = "grouped"
) -> DictConfig:
    """Compose a tiny grouped-projection train or evaluation configuration.

    :param tmp_path: Isolated dataset, checkpoint, and log root.
    :param config_name: Hydra entrypoint config, either ``train.yaml`` or ``eval.yaml``.
    :param projection_name: Shipped projection selector.
    :returns: CPU config using the real grouped model and Lance datamodule.
    :raises ValueError: If ``config_name`` is not a supported entrypoint config.
    """
    if config_name not in {"train.yaml", "eval.yaml"}:
        raise ValueError(f"unsupported grouped projection config: {config_name}")

    dataset_root = tmp_path / "lance-data"
    if not dataset_root.exists():
        write_grouped_projection_lance_dataset(dataset_root)
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name=config_name,
            return_hydra_config=True,
            overrides=[
                "experiment=surge/flow_simple",
                "datamodule=surge_lance",
                "synth=surge_4",
                f"model/projection={projection_name}",
                "trainer=cpu",
            ],
        )
    with open_dict(cfg):
        cfg.seed = GROUPED_PROJECTION_SEED
        cfg.paths.root_dir = str(operator_workspace())
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.logger = None
        cfg.datamodule.dataset_root = str(dataset_root)
        cfg.datamodule.download_dataset_root_uri = None
        cfg.datamodule.batch_size = GROUPED_PROJECTION_DATASET_ROWS
        cfg.datamodule.num_workers = 0
        cfg.datamodule.persistent_workers = False
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
        cfg.model.test_sample_steps = 1
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.precision = "32-true"
        cfg.trainer.enable_model_summary = False
        cfg.trainer.deterministic = True
        if config_name == "train.yaml":
            cfg.test = False
            cfg.training.val_audio_probe = False
            cfg.callbacks.model_checkpoint.monitor = None
            cfg.callbacks.model_checkpoint.save_last = True
            if "lr_monitor" in cfg.callbacks:
                del cfg.callbacks.lr_monitor
            cfg.trainer.max_steps = 1
            cfg.trainer.min_steps = 1
            cfg.trainer.limit_train_batches = 1
            cfg.trainer.limit_val_batches = 1
            cfg.trainer.num_sanity_val_steps = 0
        else:
            cfg.mode = "predict"
            cfg.trainer.enable_progress_bar = False
            language_initializer = cfg.callbacks.get("parameter_language")
            cfg.callbacks = {
                "prediction_writer": {
                    "_target_": "synth_setter.utils.callbacks.PredictionWriter",
                    "output_dir": "${paths.output_dir}/predictions",
                    "write_interval": "batch",
                }
            }
            if language_initializer is not None:
                cfg.callbacks["parameter_language"] = language_initializer
            cfg.datamodule.predict_file = str(dataset_root / "test.lance")
            cfg.evaluation.render_vst = False
            cfg.evaluation.compute_metrics = False
    return cfg
