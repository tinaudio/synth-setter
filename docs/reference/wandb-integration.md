# W&B Integration Reference

> **Code version**: `8af7575` (2026-03-24, `main`)
> **PyTorch**: see `requirements.txt` · **Lightning**: see `requirements.txt`
> **Tracking**: #252, #263

______________________________________________________________________

## Overview

W&B runs are created via Lightning's `WandbLogger` — there are no direct
`wandb.init()` calls in training or eval code. The logger is instantiated via
Hydra config and passed to the `Trainer`. Most metric logging goes through
Lightning's `self.log()` / `self.log_dict()` API; a handful of visualization
callbacks additionally call `wandb.log()` directly for image uploads.

______________________________________________________________________

## 1. Initialization

| Concern           | How it works                                                     | File                              |
| ----------------- | ---------------------------------------------------------------- | --------------------------------- |
| W&B run creation  | `WandbLogger` instantiated by Hydra                              | `configs/logger/wandb.yaml`       |
| Entity / project  | Hardcoded: `entity: "benhayes"`, `project: "synth-permutations"` | `configs/logger/wandb.yaml:10,13` |
| Run ID            | `null` (W&B auto-generates)                                      | `configs/logger/wandb.yaml:8`     |
| Checkpoint upload | `log_model: true`                                                | `configs/logger/wandb.yaml:11`    |
| Code saving       | `wandb.Settings(code_dir=".")`                                   | `configs/logger/wandb.yaml:17-19` |
| Run teardown      | `wandb.finish()` in `task_wrapper` finally block                 | `src/utils/utils.py:102-107`      |

**No direct `wandb.init()` or `wandb.config.update()` calls exist anywhere in the codebase.**

______________________________________________________________________

## 2. What Gets Logged

### 2a. Hyperparameters (logged once at run start)

`log_hyperparameters()` in `src/utils/logging_utils.py` sends a single dict
to all loggers via `logger.log_hyperparams()`:

| Key                          | Source                                       |
| ---------------------------- | -------------------------------------------- |
| `model`                      | Full model config subtree                    |
| `model/params/total`         | `sum(p.numel() for p in model.parameters())` |
| `model/params/trainable`     | Trainable param count                        |
| `model/params/non_trainable` | Frozen param count                           |
| `data`                       | Full data config subtree                     |
| `trainer`                    | Full trainer config subtree                  |
| `callbacks`                  | Callback config                              |
| `extras`                     | Extras config                                |
| `task_name`                  | From config                                  |
| `tags`                       | From config                                  |
| `ckpt_path`                  | From config                                  |
| `seed`                       | From config                                  |

### 2b. Training Metrics (per step / per epoch)

Logged via `self.log()` in each LightningModule:

| Module                    | Metric                                                            | Step | Epoch |
| ------------------------- | ----------------------------------------------------------------- | ---- | ----- |
| `SurgeFlowMatchingModule` | `train/loss`                                                      | yes  | yes   |
|                           | `train/penalty`                                                   | yes  | yes   |
|                           | `val/param_mse`                                                   | —    | yes   |
|                           | `test/param_mse`                                                  | —    | yes   |
|                           | `vector_field/*_norm`                                             | yes  | —     |
|                           | `encoder/*_norm`                                                  | yes  | —     |
| `KSinFlowMatchingModule`  | `train/loss`                                                      | yes  | yes   |
|                           | `train/penalty`                                                   | yes  | yes   |
|                           | `val/lsd`, `val/chamfer`                                          | —    | yes   |
|                           | `test/param_mse`, `test/lsd`, `test/chamfer`, `test/lad`          | —    | yes   |
|                           | `vector_field/*_norm`, `encoder/*_norm`                           | yes  | yes   |
| `SurgeFlowVAEModule`      | `train/loss`, `train/param_mean`, `train/param_std`, `train/beta` | yes  | yes   |
|                           | `val/*` losses                                                    | —    | yes   |
|                           | `test/*` losses                                                   | —    | yes   |
| `SurgeFeedForwardModule`  | `train/loss`                                                      | yes  | yes   |
|                           | `val/param_mse`, `test/param_mse`                                 | —    | yes   |
| `KSinFeedForwardModule`   | `train/loss`                                                      | yes  | yes   |
|                           | `val/lsd`, `val/chamfer`, `val/loss`                              | —    | yes   |
|                           | `test/*` metrics                                                  | —    | yes   |
| `MNISTLitModule`          | `train/loss`, `train/acc`                                         | —    | yes   |
|                           | `val/loss`, `val/acc`, `val/acc_best`                             | —    | yes   |
|                           | `test/loss`, `test/acc`                                           | —    | yes   |

### 2c. Callbacks — Visualization (direct `wandb.log()`)

| Callback                           | Logged key                       | Trigger                                         | File                             |
| ---------------------------------- | -------------------------------- | ----------------------------------------------- | -------------------------------- |
| `PlotLossPerTimestep`              | `plot` (image)                   | `on_validation_epoch_end`                       | `src/utils/callbacks.py:77`      |
| `PlotPositionalEncodingSimilarity` | `pos_enc_similarity` (image)     | `on_validation_epoch_end`                       | `src/utils/callbacks.py:135`     |
| `PlotLearntProjection`             | `assignment`, `value` (images)   | `on_validation_epoch_end` or every N steps      | `src/utils/callbacks.py:259`     |
| `LogPerParamMSE`                   | `per_param_mse/{name}` per param | `on_validation_epoch_end` (via `self.log_dict`) | `src/utils/callbacks.py:376-378` |

### 2d. Callbacks — Non-W&B

| Callback              | What it does                                          | Config                                     |
| --------------------- | ----------------------------------------------------- | ------------------------------------------ |
| `ModelCheckpoint`     | Saves `.ckpt` locally (uploaded by `log_model: true`) | `configs/callbacks/model_checkpoint.yaml`  |
| `LearningRateMonitor` | Logs LR to Lightning logger                           | `configs/callbacks/lr_monitor.yaml`        |
| `RichProgressBar`     | Terminal display only                                 | `configs/callbacks/rich_progress_bar.yaml` |
| `ModelSummary`        | Prints param summary to console                       | `configs/callbacks/model_summary.yaml`     |
| `PredictionWriter`    | Saves predictions to `.pt` files locally              | `src/utils/callbacks.py:307-342`           |

### 2e. Gradient Watching

If `cfg.watch_gradients` is set, `watch_gradients()` calls
`WandbLogger.watch(model, log="gradients")` — logs gradient histograms per
layer according to the WandbLogger / W&B logging defaults.

Source: `src/utils/utils.py:138-149`, called from `src/train.py:89-91`.

### 2f. Auto-Captured by W&B (not in our code)

| What           | How                                            |
| -------------- | ---------------------------------------------- |
| `sys.argv`     | W&B auto-captures command-line args            |
| System metrics | GPU utilization, memory, CPU, disk (W&B agent) |
| Git SHA + diff | W&B auto-captures if in a git repo             |
| Python deps    | `requirements.txt` / `pip freeze` snapshot     |

______________________________________________________________________

## 3. Artifacts

| Artifact          | Source                                | When                                                            |
| ----------------- | ------------------------------------- | --------------------------------------------------------------- |
| Model checkpoints | `ModelCheckpoint` + `log_model: true` | Every 5000 steps (with `default_surge` callbacks) + best + last |
| Source code       | `wandb.Settings(code_dir=".")`        | Run start                                                       |

______________________________________________________________________

## 4. Entry Points

| Entry point    | W&B usage                                                             | File           |
| -------------- | --------------------------------------------------------------------- | -------------- |
| `src/train.py` | Full: logger init → hparams → train metrics → test metrics → teardown | `src/train.py` |
| `src/eval.py`  | Full: logger init → hparams → test/val/predict metrics → teardown     | `src/eval.py`  |

Both use `@task_wrapper` which ensures `wandb.finish()` runs even on exception.

______________________________________________________________________

## 5. Known Gaps

| #   | Gap                                                                                                                               | Impact                                                                                              | Tracking |
| --- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | -------- |
| 1   | **Entity/project hardcoded** to `benhayes`/`synth-permutations`                                                                   | Runs log to wrong account                                                                           | #133     |
| 2   | **No `wandb.config` for env vars** — W&B captures `sys.argv` but not env vars like `TRAINING_ARGS`                                | Config passed via env vars is silently missing from W&B                                             | #252     |
| 3   | **No `github_sha` in `wandb.config`** — design docs require it but it's not implemented                                           | Can't reliably link a run to the exact code that produced it (W&B auto-capture is best-effort)      | —        |
| 4   | **No GitHub issue integration** — train job doesn't post run ID back to GitHub                                                    | Manual lookup to match runs to issues                                                               | #263     |
| 5   | **`log_model: true` vs `"all"`** — config uses `true` (uploads best + last only), design doc specifies `"all"` (every checkpoint) | Intermediate checkpoints not uploaded to W&B                                                        | —        |
| 6   | **Visualization callbacks use `wandb.log()` directly** — bypasses Lightning logger abstraction                                    | Breaks if logger is swapped; step alignment relies on `trainer.global_step`                         | —        |
| 7   | **`torch.compile` crashes test-stage `setup()`** — eval after training fails                                                      | Post-training test metrics never logged to W&B                                                      | #248     |
| 8   | **No structured run ID convention** — `id: null` means W&B generates random IDs                                                   | Can't reconstruct run lineage from ID alone; design doc specifies `{config_id}-{timestamp}` pattern | —        |
