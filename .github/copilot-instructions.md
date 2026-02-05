# Copilot Instructions

## Project intent
- Inverts audio synthesizer parameters via approximately equivariant flow matching; core paper code for ISMIR 2025 submission.
- PyTorch Lightning + Hydra project; `rootutils` uses `.project-root` to put `src/` on `PYTHONPATH` and populate `PROJECT_ROOT` for path configs.

## Key architecture
- Entrypoints: training in [src/train.py](../src/train.py), evaluation in [src/eval.py](../src/eval.py); both wrap work in `@task_wrapper` (saves output dir, closes wandb) and use Hydra configs under [configs/](../configs).
- Models live in [src/models](../src/models): flow-matching LightningModules ([src/models/ksin_flow_matching_module.py](../src/models/ksin_flow_matching_module.py), [src/models/surge_flow_matching_module.py](../src/models/surge_flow_matching_module.py)), plus FFN and VAE baselines. Vector fields/encoders are in [src/models/components](../src/models/components) (Transformer/DiT variants, residual MLPs, CNN encoders, VAE/RealNVP baseline).
- Data: synthetic sinusoid task in [src/data/ksin_datamodule.py](../src/data/ksin_datamodule.py); Surge XT HDF5 loader in [src/data/surge_datamodule.py](../src/data/surge_datamodule.py) with OT alignment and optional mel/audio/music2latent conditioning. OT helpers live in [src/data/ot.py](../src/data/ot.py).
- Metrics for spectra/parameter permutation handling in [src/metrics.py](../src/metrics.py) (log-spectral distance, chamfer, linear assignment).

## Configuration conventions (Hydra)
- Default train config: [configs/train.yaml](../configs/train.yaml); eval config: [configs/eval.yaml](../configs/eval.yaml). Overrides follow Hydra CLI syntax (e.g., `python src/train.py model=surge_flow data=surge callbacks=eval_surge trainer=gpu`).
- Paths: [configs/paths/default.yaml](../configs/paths/default.yaml) reads `PROJECT_ROOT`; outputs/logs land under `${paths.log_dir}` per [configs/hydra/default.yaml](../configs/hydra/default.yaml).
- Trainer presets in [configs/trainer/](../configs/trainer) (cpu/gpu/mps/ddp). Models and datamodules are swapped via [configs/model/](../configs/model) and [configs/data/](../configs/data). Callbacks/loggers configured in [configs/callbacks/](../configs/callbacks) and [configs/logger/](../configs/logger).
- `extras` block toggles warnings, rich config print, enforced tags, and matmul precision (see `extras()` in [src/utils/utils.py](../src/utils/utils.py)).

## Data specifics
- K-Sin datamodule samples sine mixtures on-the-fly; params are frequency/amp pairs, optional OT or symmetry-breaking via config flags.
- Surge XT datamodule expects HDF5 shards with `audio`, `mel_spec`, `music2latent`, `param_array`; stats file `stats.npz` must sit beside each shard for normalization. Config defaults set `dataset_root` to an absolute path—override for local data. `use_saved_mean_and_variance` and `rescale_params` gate normalization and mapping to [-1, 1]. `ot: true` enables Hungarian matching of noise/params/mels.

## Training/eval workflow
- Training: `python src/train.py` (or `make train`). Common overrides: `data=surge_mini`, `model=surge_flow`, `trainer=gpu`, `logger=wandb`, `callbacks=default_surge`, `seed=123`. `cfg.train`/`cfg.test` booleans control phases; `ckpt_path` resumes.
- Eval/predict: `python src/eval.py ckpt_path=/path/to.ckpt mode=test data=surge_mini model=surge_flow trainer=gpu logger=null`.
- Sampling cfg uses classifier-free guidance mix (`cfg_strength`) via helper `call_with_cfg`/`rk4_with_cfg` in modules.

## Optimization patterns
- Flow matching modules sample time `t`, build probability paths (rectified/CFM/FM), and minimize squared error to target vector fields; optional penalties from vector field projection are logged as `train/penalty`.
- K-Sin supports OT/EOT/Kabsch/Procrustes coupling and optional freeze phase for symmetry discovery (`freeze_for_first_n_steps`). Sampling uses RK4 with time warping for inference.
- Surge module conditions on mel or music2latent tensors; uses rectified path only and optional warmup scheduler.

## Testing & quality
- Fast tests: `make test` (runs `pytest -k "not slow"`); full: `make test-full` or `pytest`. CI runs `pytest -v` across Python 3.8–3.10 on linux/mac/windows and uploads coverage (`pytest --cov src`).
- Formatting/linting via `pre-commit run -a` (see [.pre-commit-config.yaml](../.pre-commit-config.yaml)); CI also runs pre-commit on main/pr.

## Misc
- Metrics retrieval uses `get_metric_value` with `optimized_metric` config; watch gradients requires WandbLogger. Hydra output dirs contain logs/ckpts; cleanup via `make clean` or `make clean-logs`.

Please suggest edits if any sections are unclear or missing project-specific tips.
