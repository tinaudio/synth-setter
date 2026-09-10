"""Run paired, real-audio parameter-language smoke ablations through production training."""

import argparse
import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_module
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from synth_setter.cli.eval import evaluate
from synth_setter.cli.train import train
from synth_setter.data.vst.param_spec import CategoricalParameter, DiscreteLiteralParameter
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.data.vst.shapes import PARAM_ARRAY_FIELD
from synth_setter.model_cache import checkpoint_tree_sha256
from synth_setter.pipeline.data.lance_shard import iter_lance_column_rows
from synth_setter.pipeline.data.param_language import (
    EMBEDDING_REVISION,
    encode_param_language,
    load_param_language,
    matryoshka_vectors,
    save_param_language,
)
from synth_setter.pipeline.data.stats import finalize, fold_lance_shard_into_welford
from synth_setter.pipeline.schemas.spec import _get_git_sha
from tests.conftest import PLUGIN_PATH, _render_smoke_train_subprocess, _surge_smoke_render_config

VARIANTS = ("grouped", "learned", "random", "language128", "language768")


def build_config(root: Path, consumer: str, variant: str, *, seed: int, steps: int) -> DictConfig:
    """Compose matched small models without changing the production learning objectives.

    :param root: Experiment root containing the shared real dataset.
    :param consumer: Flow or SLAP.
    :param variant: Grouped baseline or residual metadata treatment.
    :param seed: Paired initialization and training seed.
    :param steps: Optimizer-step budget.
    :returns: Fully specified training configuration.
    """
    experiment = "surge/flow_simple" if consumer == "flow" else "surge/slap_language"
    overrides = [f"experiment={experiment}", "synth=surge_simple", "trainer=gpu"]
    if consumer == "flow":
        overrides.append("model/projection=language")
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="train", overrides=overrides, return_hydra_config=True)
    output = root / "runs" / f"{consumer}-{variant}-{seed}"
    with open_dict(cfg):
        cfg.seed = seed
        cfg.paths.root_dir = str(Path.cwd())
        cfg.paths.output_dir = str(output)
        cfg.paths.log_dir = str(output)
        cfg.train = True
        cfg.test = False
        cfg.training.resume = None
        cfg.training.val_audio_probe = False
        cfg.logger = {
            "csv": {
                "_target_": "lightning.pytorch.loggers.CSVLogger",
                "save_dir": str(output),
                "name": "metrics",
            }
        }
        cfg.datamodule.dataset_root = str(root / "dataset")
        cfg.datamodule.download_dataset_root_uri = None
        cfg.datamodule.num_workers = 0
        cfg.datamodule.persistent_workers = False
        cfg.datamodule.pin_memory = False
        cfg.datamodule.batch_size = 8
        cfg.datamodule.ot = False
        cfg.model.compile = False
        cfg.model.scheduler = None
        cfg.model.optimizer.lr = 3e-4
        cfg.trainer.accelerator = "gpu"
        cfg.trainer.devices = 1
        cfg.trainer.precision = "32-true"
        cfg.trainer.min_steps = steps
        cfg.trainer.max_steps = steps
        cfg.trainer.max_epochs = -1
        cfg.trainer.num_sanity_val_steps = 0
        cfg.trainer.check_val_every_n_epoch = None
        cfg.trainer.val_check_interval = steps
        cfg.trainer.limit_train_batches = 1.0
        cfg.trainer.limit_val_batches = 1.0
        cfg.trainer.limit_test_batches = 1.0
        cfg.trainer.log_every_n_steps = 1
        cfg.trainer.enable_progress_bar = False
        cfg.trainer.enable_model_summary = False
        cfg.trainer.deterministic = True
        initializer = cfg.callbacks.parameter_language
        cfg.callbacks = {
            "parameter_language": initializer,
            "model_checkpoint": {
                "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                "dirpath": str(output / "checkpoints"),
                "save_last": True,
                "save_top_k": 1,
                "monitor": None,
                "every_n_train_steps": steps,
                "save_on_train_epoch_end": False,
            },
        }
        projection = {
            "_target_": "synth_setter.models.components.language_projection.LanguageParameterProjection",
            "d_model": 32,
            "param_spec_name": "${synth.param_spec_name}",
            "synth_name": "${synth.name}",
            "embedding_dim": 768 if variant == "language768" else 128,
            "embedding_path": str(
                root
                / "dataset"
                / ("language768.npz" if variant == "language768" else "language128.npz")
            ),
            "embedding_source": variant if variant in {"random", "learned"} else "language",
            "embedding_seed": seed,
        }
        if variant == "grouped":
            projection = {
                "_target_": "synth_setter.models.components.transformer.GroupedParameterProjection",
                "d_model": 32,
                "param_spec_name": "${synth.param_spec_name}",
            }
        if consumer == "flow":
            cfg.model.vector_field.projection = projection
            cfg.model.encoder.d_model = 32
            cfg.model.encoder.n_heads = 2
            cfg.model.encoder.n_layers = 1
            cfg.model.encoder.n_conditioning_outputs = 1
            cfg.model.encoder.patch_stride = 10
            cfg.model.vector_field.d_model = 32
            cfg.model.vector_field.d_ff = 64
            cfg.model.vector_field.num_heads = 2
            cfg.model.vector_field.num_layers = 1
            cfg.model.validation_sample_steps = 4
            cfg.model.test_sample_steps = 4
        else:
            cfg.model.param_encoder.encoder.projection = projection
            cfg.model.param_encoder.encoder.d_model = 32
            cfg.model.param_encoder.encoder.d_out = 32
            cfg.model.param_encoder.encoder.n_heads = 2
            cfg.model.param_encoder.encoder.n_layers = 1
            audio_ast = cfg.model.audio_encoder.encoder._args_[0]
            audio_ast.d_model = 32
            audio_ast.n_heads = 2
            audio_ast.n_layers = 1
            audio_ast.patch_stride = 10
            for arm in (cfg.model.audio_encoder, cfg.model.param_encoder):
                for section in (arm.projector, arm.transform):
                    for layer in section._args_:
                        for key in ("in_features", "out_features", "num_features"):
                            if key in layer:
                                layer[key] = 128 if layer[key] == 4096 else 32
    return cfg


def prepare_dataset(root: Path) -> None:
    """Render disjoint seed streams and cache real static language vectors once.

    :param root: Experiment root receiving local production-format artifacts.
    :raises RuntimeError: Cached split sizes differ or parameter rows overlap.
    """
    dataset = root / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    for split, count, seed in (("train", 32, 4100), ("val", 8, 4200), ("test", 8, 4300)):
        path = dataset / f"{split}.lance"
        if not path.exists():
            _render_smoke_train_subprocess(path, "surge_simple", num_samples=count, base_seed=seed)
    state = fold_lance_shard_into_welford((0, 0, 0), dataset / "train.lance")
    mean, std = finalize(state, mask_degenerate=True)
    np.savez(dataset / "stats.npz", mean=mean, std=std)
    if not (dataset / "language768.npz").exists():
        full = encode_param_language("surge_simple", "surge_simple", device="cuda")
        save_param_language(dataset / "language768.npz", full, "surge_simple", "surge_simple")
    full, _ = load_param_language(dataset / "language768.npz", "surge_simple", "surge_simple")
    save_param_language(
        dataset / "language128.npz", matryoshka_vectors(full, 128), "surge_simple", "surge_simple"
    )
    rows = [
        np.stack(list(iter_lance_column_rows(dataset / f"{split}.lance", PARAM_ARRAY_FIELD)))
        for split in ("train", "val", "test")
    ]
    if [len(split) for split in rows] != [32, 8, 8]:
        raise RuntimeError("cached dataset split sizes do not match this experiment")
    combined = np.concatenate(rows)
    if len(np.unique(combined, axis=0)) != len(combined):
        raise RuntimeError("duplicate parameter rows across research splits")
    (root / "dataset.sha256").write_text(checkpoint_tree_sha256(dataset) + "\n")


def summarize_parameters(
    predicted: torch.Tensor, target: torch.Tensor, param_spec_name: str
) -> dict[str, float]:
    """Report field MSE and categorical accuracy only for one-hot encoded fields.

    :param predicted: Model-coordinate predictions for the held-out examples.
    :param target: Corresponding model-coordinate targets.
    :param param_spec_name: Registry layout of both matrices.
    :returns: Per-field errors and applicable one-hot accuracies.
    """
    metrics = {}
    accuracies = []
    for field, span in param_specs[param_spec_name].encoded_slices():
        metrics[f"field_mse/{field.name}"] = float(
            (predicted[:, span] - target[:, span]).square().mean()
        )
        if (
            isinstance(field, (CategoricalParameter, DiscreteLiteralParameter))
            and field.encoding == "onehot"
        ):
            accuracy = float(
                (predicted[:, span].argmax(-1) == target[:, span].argmax(-1)).float().mean()
            )
            metrics[f"onehot/{field.name}"] = accuracy
            accuracies.append(accuracy)
    if accuracies:
        metrics["onehot/mean_accuracy"] = float(np.mean(accuracies))
    return metrics


def run_experiment(root: Path, consumer: str, variant: str, *, seed: int, steps: int) -> None:
    """Train, reload, evaluate, and persist one treatment's real observations.

    :param root: Shared dataset and per-run output root.
    :param consumer: Production learning objective.
    :param variant: Metadata treatment.
    :param seed: Paired random seed.
    :param steps: Optimizer-step budget.
    :raises RuntimeError: Training or evaluation violates the smoke contract.
    """
    cfg = build_config(root, consumer, variant, seed=seed, steps=steps)
    output = Path(cfg.paths.output_dir)
    if (output / "results.json").exists():
        raise RuntimeError("completed run already exists; choose a new experiment root")
    dataset_sha256 = checkpoint_tree_sha256(root / "dataset")
    if dataset_sha256 != (root / "dataset.sha256").read_text().strip():
        raise RuntimeError("dataset changed after preparation")
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "config.yaml")
    HydraConfig().set_config(cfg)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    _, objects = train(cfg)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated()
    trainer = objects["trainer"]
    if trainer.global_step != steps:
        raise RuntimeError(f"expected {steps} optimizer steps, got {trainer.global_step}")
    checkpoint = trainer.checkpoint_callback.last_model_path
    restored_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    if variant != "grouped":
        projection_cfg = (
            restored_cfg.vector_field.projection
            if consumer == "flow"
            else restored_cfg.param_encoder.encoder.projection
        )
        projection_cfg.embedding_path = str(output / "absent-source.npz")
    restored = hydra.utils.instantiate(restored_cfg)
    torch.manual_seed(seed + 100000)
    results = trainer.test(
        restored, datamodule=objects["datamodule"], ckpt_path=checkpoint, weights_only=False
    )
    metrics = {key: float(value) for row in results for key, value in row.items()}
    if not metrics or not all(np.isfinite(value) for value in metrics.values()):
        raise RuntimeError("checkpoint evaluation returned absent or nonfinite metrics")
    if consumer == "flow":
        with open_dict(cfg):
            cfg.mode = "predict"
            cfg.ckpt_path = checkpoint
            cfg.datamodule.predict_file = str(root / "dataset" / "test.lance")
            cfg.datamodule.batch_size = 2
            cfg.trainer.limit_predict_batches = 1
            cfg.callbacks = {
                "parameter_language": {
                    "_target_": "synth_setter.utils.parameter_language.ParameterLanguageInitializer"
                },
                "prediction_writer": {
                    "_target_": "synth_setter.utils.callbacks.PredictionWriter",
                    "output_dir": str(output / "predictions"),
                    "write_interval": "batch",
                },
            }
            render = _surge_smoke_render_config("surge_simple", PLUGIN_PATH)
            cfg.synth = render.pop("synth")
            cfg.render = render
            eval_defaults = (
                Path(__file__).resolve().parents[2] / "src/synth_setter/configs/eval.yaml"
            )
            cfg.evaluation = OmegaConf.load(eval_defaults).evaluation
            cfg.evaluation.render_vst = True
            cfg.evaluation.compute_metrics = True
        torch.manual_seed(seed + 100000)
        evaluate(cfg)
        audio_metrics = pd.read_csv(output / "metrics" / "metrics.csv", index_col=0).select_dtypes(
            include="number"
        )
        if not np.isfinite(audio_metrics.to_numpy()).all():
            raise RuntimeError("rendered-audio metrics are nonfinite")
        metrics.update(
            {f"audio/{key}": float(value) for key, value in audio_metrics.mean().items()}
        )
        spec = param_specs["surge_simple"]
        predicted = torch.load(output / "predictions" / "pred-0.pt", weights_only=True).reshape(
            -1, spec.encoded_width
        )
        target = torch.load(
            output / "predictions" / "target-params-0.pt", weights_only=True
        ).reshape_as(predicted)
        metrics.update(summarize_parameters(predicted, target, "surge_simple"))
        metrics["audio/evaluated_rows"] = float(len(predicted))
    history_files = sorted((output / "metrics").glob("version_*/metrics.csv"))
    history = pd.read_csv(history_files[0])
    loss_columns = [
        key for key in history if "train" in key and "loss" in key and not key.endswith("epoch")
    ]
    losses = {
        key: {
            "first10": float(np.mean(history[key].dropna().to_numpy()[:10])),
            "last10": float(np.mean(history[key].dropna().to_numpy()[-10:])),
        }
        for key in loss_columns
    }
    git_sha = _get_git_sha()
    if len(git_sha) != 40:
        raise RuntimeError("git revision unavailable for provenance")
    summary = {
        "consumer": consumer,
        "variant": variant,
        "seed": seed,
        "steps": steps,
        "git_sha": git_sha,
        "embedding_revision": EMBEDDING_REVISION,
        "dataset_sha256": dataset_sha256,
        "train_seconds": seconds,
        "steps_per_second": steps / seconds,
        "peak_allocated_bytes": peak_memory,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in objects["model"].parameters()
            if parameter.requires_grad
        ),
        "checkpoint": checkpoint,
        "losses": losses,
        "metrics": metrics,
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    sys.stdout.write(json.dumps(summary, allow_nan=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    """Run one selected treatment or prepare its shared real dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--consumer", choices=("flow", "slap"))
    parser.add_argument("--variant", choices=VARIANTS, default="language128")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if not 100 <= args.steps <= 1000:
        parser.error("--steps must be between 100 and 1000")
    root = args.root.resolve()
    if args.consumer is None:
        prepare_dataset(root)
    else:
        run_experiment(root, args.consumer, args.variant, seed=args.seed, steps=args.steps)


if __name__ == "__main__":
    main()
