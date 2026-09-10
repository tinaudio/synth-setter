# Grouped parameter projection smoke ablation

Issue [#3273](https://github.com/tinaudio/synth-setter/issues/3273) introduces a
field-aligned projection for flow models. Each logical `ParamSpec` field becomes
one token, including a whole one-hot or array field. The existing learned dense
assignment remains the default.

## Usage and checkpoint compatibility

The default needs no override:

```bash
synth-setter-train experiment=surge/flow_full
```

Select grouped projection with the Hydra model sub-group:

```bash
synth-setter-train experiment=surge/flow_full model/projection=grouped
```

The group is selected as `model/projection`, while its package is
`model.vector_field.projection`. The grouped constructor receives
`param_spec_name: ${synth.param_spec_name}` and `d_model`; its token count is the
number of logical fields in that registered spec. The learned projection keeps
its configured token count (128 by default).

A checkpoint must be restored with the same projection selection used to train
it. Grouped and learned projections have different state-dict keys and tensor
layouts, so strict cross-loading fails by design. For example:

```bash
synth-setter-eval \
  experiment=surge/flow_full \
  model/projection=grouped \
  ckpt_path=/path/to/grouped-last.ckpt
```

Do not evaluate a grouped checkpoint under the default learned projection, or a
learned checkpoint under `model/projection=grouped`.

## Question and scope

This run asks only whether both projections can complete the same small local
production flow path and provides rough resource and metric observations. It is
a **tiny smoke ablation, not a model-quality conclusion**: one seed, eight rows,
20 optimizer steps, eight ODE sample steps, and one timing observation per arm
are inadequate for ranking architectures.

Both arms used the same:

- eight real Surge XT renders, cloned into train/validation/test Lance splits;
- full `surge_xt` parameter spec (300 encoded columns, 164 logical fields, 32
  one-hot fields);
- seed 3273, batch size 2, Adam settings from `vst_flow`, and 20 optimizer steps;
- one-layer 32-wide AST encoder and one-layer 32-wide flow field;
- eight-step production flow sampling, checkpoint reload, prediction writer,
  Surge XT re-render, and audio-metric pipeline;
- eager execution (`model.compile=false`) and float32 on one local GPU.

Only `model/projection=learnt` versus `model/projection=grouped` changed. Eager
mode was pinned for both arms to isolate projection architecture. CUDA
`torch.compile` support is covered separately by implementation tests and is
not measured here.

Environment: git base `07c9f2c17ba39f237f9a58cb88d2920a77fdf12a`, PyTorch
2.12.0+cu130, NVIDIA GeForce RTX 5060 Ti, driver 595.71.05. No paid compute,
Docker, or remote service was used.

## Results

Lower is better for loss, MSE, MSS, wMFCC, and SOT. Higher is better for RMS
cosine similarity, categorical accuracy, and throughput.

| Metric                                        |           Learned |           Grouped |
| --------------------------------------------- | ----------------: | ----------------: |
| Final train loss                              |           1.69585 |           1.85598 |
| Parameter MSE                                 |           1.64548 |           1.92961 |
| Number-group optimal-assignment MSE           |           1.36429 |           1.59635 |
| Number-group assignment gap                   |           0.28118 |           0.33325 |
| Oscillator-only parameter MSE                 |           1.63709 |           1.78374 |
| Oscillator-only optimal-assignment MSE        |           1.33456 |           1.43208 |
| Oscillator-only assignment gap                |           0.30253 |           0.35167 |
| One-hot field argmax accuracy (256 decisions) |           0.26172 |           0.28906 |
| MSS, mean ± sample std                        | 30.6432 ± 20.0909 |  29.1528 ± 9.6172 |
| wMFCC, mean ± sample std                      |  19.4501 ± 6.7236 |  22.7703 ± 9.0212 |
| SOT, mean ± sample std                        | 0.43606 ± 0.14772 | 0.42745 ± 0.15181 |
| RMS cosine similarity, mean ± sample std      | 0.21838 ± 0.20092 | 0.31802 ± 0.21672 |
| Train wall time, seconds                      |            2.9649 |            6.9354 |
| Optimizer steps/second                        |            6.7455 |            2.8838 |
| Peak CUDA allocated, MiB                      |           70.7466 |           73.0410 |
| Total/trainable parameters                    |           124,352 |            89,388 |
| Parameter tokens                              |               128 |               164 |
| Checkpoint bytes                              |         1,569,051 |         2,091,457 |

The grouped model had 28.1% fewer model parameters but 28.1% more parameter
tokens. In this single sequential-process observation it used 3.2% more peak
allocated CUDA memory and had 57.2% lower measured step throughput. Timing covers
the complete `train(cfg)` call, including setup, validation, and checkpoint I/O;
it is not warmed-up optimizer-kernel throughput. The learned arm runs first,
so process warmup and allocator history also confound these resource numbers. The grouped checkpoint
was larger despite fewer model parameters because the Lightning checkpoint also
contains optimizer and training state with architecture-dependent serialization.

Parameter-space and audio metrics were mixed: learned had lower plain and
optimal-assignment parameter MSE, while grouped had slightly higher aggregate one-hot
accuracy, lower MSS/SOT, and higher RMS similarity but higher wMFCC. Surge XT
injects render variation, and eight clips give wide audio-metric standard
deviations. None of these differences establishes quality.

The number-group metric optimally assigns fields whose names differ only by a
number. The oscillator-only rows restrict its per-coordinate output to `a_osc_*`
fields. The assignment gap is plain MSE minus assigned MSE; a larger gap means more
error can be removed by reassigning values among numbered peers. Grouped showed
a larger oscillator gap in this smoke run, not reduced oscillator placement
ambiguity. A quality claim would require multiple seeds, a representative
held-out set, matched training to convergence, warm timing repetitions, and
confidence intervals.

### Per-categorical-field accuracy

Each fraction below is argmax accuracy over the same eight prediction rows.
These are endpoint diagnostics only; neither arm adds softmax or categorical
cross-entropy to its coordinate-wise velocity loss.

| One-hot field                 | Learned accuracy | Grouped accuracy |
| ----------------------------- | ---------------: | ---------------: |
| `a_amp_eg_envelope_mode`      |            0.500 |            0.500 |
| `a_filter_eg_envelope_mode`   |            0.625 |            0.625 |
| `a_filter_configuration`      |            0.000 |            0.000 |
| `a_filter_1_type`             |            0.000 |            0.000 |
| `a_filter_2_type`             |            0.250 |            0.250 |
| `a_waveshaper_type`           |            0.250 |            0.250 |
| `a_osc_1_mute`                |            0.250 |            0.250 |
| `a_osc_1_octave`              |            0.250 |            0.375 |
| `a_osc_1_route`               |            0.375 |            0.375 |
| `a_osc_1_unison_voices`       |            0.375 |            0.375 |
| `a_osc_2_mute`                |            0.500 |            0.500 |
| `a_osc_2_octave`              |            0.000 |            0.250 |
| `a_osc_2_route`               |            0.500 |            0.500 |
| `a_osc_2_unison_voices`       |            0.000 |            0.000 |
| `a_osc_3_mute`                |            0.750 |            0.625 |
| `a_osc_3_octave`              |            0.250 |            0.250 |
| `a_osc_3_route`               |            0.125 |            0.000 |
| `a_osc_3_unison_voices`       |            0.500 |            0.625 |
| `a_fm_routing`                |            0.375 |            0.375 |
| `a_lfo_1_type`                |            0.250 |            0.250 |
| `a_lfo_2_type`                |            0.000 |            0.125 |
| `a_lfo_3_type`                |            0.125 |            0.125 |
| `a_lfo_4_type`                |            0.000 |            0.000 |
| `a_lfo_5_type`                |            0.250 |            0.375 |
| `a_noise_mute`                |            0.250 |            0.375 |
| `a_noise_route`               |            0.625 |            0.500 |
| `a_ring_modulation_1x2_mute`  |            0.250 |            0.250 |
| `a_ring_modulation_1x2_route` |            0.000 |            0.250 |
| `a_ring_modulation_2x3_mute`  |            0.375 |            0.500 |
| `a_ring_modulation_2x3_route` |            0.250 |            0.250 |
| `fx_a2_delay_time_left`       |            0.000 |            0.000 |
| `fx_a2_delay_time_right`      |            0.125 |            0.125 |

## Reproduction

From the repository worktree, save the script below as
`.cache/grouped_projection_ablation.py`, then run:

```bash
uv run python .cache/grouped_projection_ablation.py \
  > .cache/grouped_projection_ablation.log 2>&1
```

It writes raw JSON to
`.cache/grouped-projection-ablation/results.json` and audio/prediction artifacts
under each arm's output directory. Delete
`.cache/grouped-projection-ablation/dataset-source` to force regeneration of the
real Surge XT source renders.

```python
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from synth_setter.cli.eval import evaluate
from synth_setter.cli.train import train
from synth_setter.data.vst.param_spec import CategoricalParameter, DiscreteLiteralParameter
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.metrics import number_group_optimal_assignment_per_param_mse
from tests._vst import PLUGIN_PATH
from tests.conftest import (
    _build_surge_smoke_lance_datasets,
    _configure_surge_xt_eval_cfg,
    _render_smoke_train_subprocess,
)

OUTPUT = ROOT / ".cache" / "grouped-projection-ablation"
DATA_ROWS = 8
SEED = 3273
STEPS = 20
SAMPLE_STEPS = 8
SPEC_NAME = "surge_xt"


def build_dataset() -> Path:
    dataset_parent = OUTPUT / "dataset-source"
    dataset_root = dataset_parent / "data" / "smoke-lance"
    required = [
        dataset_root / name
        for name in ("train.lance", "val.lance", "test.lance", "stats.npz")
    ]
    if all(path.exists() for path in required):
        return dataset_root
    if dataset_parent.exists():
        shutil.rmtree(dataset_parent)
    return _build_surge_smoke_lance_datasets(
        dataset_parent,
        SPEC_NAME,
        lambda path, name: _render_smoke_train_subprocess(
            path, name, num_samples=DATA_ROWS, base_seed=SEED
        ),
        num_samples=DATA_ROWS,
    )


def configure_train(kind: str, dataset_root: Path, run_dir: Path):
    from hydra import compose, initialize_config_module

    GlobalHydra.instance().clear()
    with initialize_config_module(
        version_base="1.3", config_module="synth_setter.configs"
    ):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=surge/flow_full",
                "datamodule=surge_lance",
                "synth=surge_xt",
                f"model/projection={kind}",
                "callbacks=[default_vst,eval_vst,log_per_param_mse]",
            ],
        )
    with open_dict(cfg):
        cfg.seed = SEED
        cfg.paths.root_dir = str(ROOT)
        cfg.paths.output_dir = str(run_dir)
        cfg.paths.log_dir = str(run_dir)
        cfg.logger = None
        cfg.test = False
        cfg.training.val_audio_probe = False
        cfg.datamodule.dataset_root = str(dataset_root)
        cfg.datamodule.predict_file = str(dataset_root / "test.lance")
        cfg.datamodule.download_dataset_root_uri = None
        cfg.datamodule.batch_size = 2
        cfg.datamodule.num_workers = 0
        cfg.datamodule.pin_memory = False
        cfg.datamodule.ot = False
        cfg.model.compile = False
        cfg.model.scheduler = None
        cfg.model.encoder.d_model = 32
        cfg.model.encoder.n_heads = 2
        cfg.model.encoder.n_layers = 1
        cfg.model.encoder.n_conditioning_outputs = 2
        cfg.model.vector_field.d_model = 32
        cfg.model.vector_field.d_ff = 64
        cfg.model.vector_field.num_heads = 2
        cfg.model.vector_field.num_layers = 1
        cfg.model.validation_sample_steps = SAMPLE_STEPS
        cfg.model.test_sample_steps = SAMPLE_STEPS
        cfg.callbacks.model_checkpoint.monitor = None
        cfg.callbacks.model_checkpoint.save_last = True
        if "lr_monitor" in cfg.callbacks:
            del cfg.callbacks.lr_monitor
        cfg.trainer.accelerator = "gpu"
        cfg.trainer.devices = 1
        cfg.trainer.precision = "32-true"
        cfg.trainer.max_steps = STEPS
        cfg.trainer.min_steps = STEPS
        cfg.trainer.limit_val_batches = 1
        cfg.trainer.num_sanity_val_steps = 0
        cfg.trainer.enable_model_summary = False
        cfg.trainer.deterministic = True
        cfg.trainer.check_val_every_n_epoch = None
        cfg.trainer.val_check_interval = STEPS
        cfg.trainer.log_every_n_steps = STEPS
    return cfg


def onehot_accuracy(
    predicted: torch.Tensor, target: torch.Tensor
) -> tuple[float, int, dict[str, float]]:
    correct = 0
    fields = 0
    per_field = {}
    for parameter, span in param_specs[SPEC_NAME].encoded_slices():
        is_onehot = isinstance(
            parameter, (CategoricalParameter, DiscreteLiteralParameter)
        ) and parameter.encoding == "onehot"
        if is_onehot:
            matches = (
                predicted[:, span].argmax(dim=1)
                == target[:, span].argmax(dim=1)
            )
            correct += int(matches.sum())
            fields += predicted.shape[0]
            per_field[parameter.name] = float(matches.float().mean())
    return correct / fields, fields, per_field


def load_predictions(run_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    pred_files = sorted((run_dir / "predictions").glob("pred-*.pt"))
    target_files = sorted((run_dir / "predictions").glob("target-params-*.pt"))
    return (
        torch.cat(
            [torch.load(path, weights_only=True).reshape(1, -1) for path in pred_files]
        ),
        torch.cat(
            [
                torch.load(path, weights_only=True).reshape(1, -1)
                for path in target_files
            ]
        ),
    )


def run_arm(kind: str, dataset_root: Path) -> dict[str, object]:
    run_dir = OUTPUT / kind
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    cfg = configure_train(kind, dataset_root, run_dir)
    HydraConfig().set_config(cfg)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    metric_dict, objects = train(cfg)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    peak_bytes = torch.cuda.max_memory_allocated()

    model = objects["model"]
    checkpoint = run_dir / "checkpoints" / "last.ckpt"
    eval_cfg = cfg.copy()
    _configure_surge_xt_eval_cfg(
        eval_cfg,
        tmp_path=run_dir,
        dataset_root=dataset_root,
        predict_file=dataset_root / "test.lance",
        param_spec_name=SPEC_NAME,
        plugin_path=PLUGIN_PATH,
        rerender_target=True,
    )
    with open_dict(eval_cfg):
        eval_cfg.model.test_sample_steps = SAMPLE_STEPS
        eval_cfg.trainer.accelerator = "gpu"
        eval_cfg.trainer.precision = "32-true"
    HydraConfig().set_config(eval_cfg)
    evaluate(eval_cfg)

    predicted, target = load_predictions(run_dir)
    per_param_mse = (predicted - target).square().mean(dim=0)
    grouped_per_param_mse = number_group_optimal_assignment_per_param_mse(
        predicted, target, param_specs[SPEC_NAME]
    )
    mse = float(per_param_mse.mean())
    assignment_mse = float(grouped_per_param_mse.mean())
    oscillator_indices = torch.tensor(
        [
            index
            for parameter, span in param_specs[SPEC_NAME].encoded_slices()
            if parameter.name.startswith("a_osc_")
            for index in range(span.start, span.stop)
        ]
    )
    oscillator_mse = float(per_param_mse[oscillator_indices].mean())
    oscillator_assignment_mse = float(grouped_per_param_mse[oscillator_indices].mean())
    categorical_accuracy, categorical_decisions, per_field_accuracy = onehot_accuracy(
        predicted, target
    )
    audio = pd.read_csv(run_dir / "metrics" / "aggregated_metrics.csv", index_col=0)
    return {
        "projection": kind,
        "seed": SEED,
        "dataset_rows": DATA_ROWS,
        "optimizer_steps": objects["trainer"].global_step,
        "sample_steps": SAMPLE_STEPS,
        "train_loss": float(metric_dict["train/loss"]),
        "parameter_mse": mse,
        "number_group_optimal_assignment_mse": assignment_mse,
        "number_group_assignment_gap": mse - assignment_mse,
        "oscillator_parameter_mse": oscillator_mse,
        "oscillator_number_group_optimal_assignment_mse": oscillator_assignment_mse,
        "oscillator_assignment_gap": oscillator_mse - oscillator_assignment_mse,
        "onehot_field_argmax_accuracy": categorical_accuracy,
        "onehot_field_decisions": categorical_decisions,
        "onehot_accuracy_by_field": per_field_accuracy,
        "train_seconds": train_seconds,
        "optimizer_steps_per_second": STEPS / train_seconds,
        "peak_cuda_allocated_mib": peak_bytes / 2**20,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "parameter_token_count": model.vector_field.projection.num_tokens,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "audio_metrics": {
            metric: {
                column: float(audio.loc[metric, column])
                for column in ("mean", "std")
            }
            for metric in ("mss", "wmfcc", "sot", "rms")
        },
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset_root = build_dataset()
    results = [run_arm(kind, dataset_root) for kind in ("learnt", "grouped")]
    output_path = OUTPUT / "results.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
```
