# W&B Integration Reference

> **Code version**: `0b55a9e` (2026-04-20, `feat/wandb-optional-by-default`)
> **PyTorch**: see `requirements.txt` · **Lightning**: see `requirements.txt`
> **Tracking**: #252, #263

______________________________________________________________________

## Overview

W&B runs are created via Lightning's `WandbLogger` — there are no direct
`wandb.init()` calls in training or eval code. The logger is instantiated via
Hydra config and passed to the `Trainer`. Metric logging goes through
Lightning's `self.log()` / `self.log_dict()` API; visualization callbacks
route matplotlib figures through a small logger-dispatch helper
(`_log_figure` in `src/utils/callbacks.py`) that calls
`WandbLogger.log_image` or `TensorBoardLogger.experiment.add_figure`
depending on which loggers are attached.

______________________________________________________________________

## 1. Initialization

| Concern           | How it works                                                                                             | File                               |
| ----------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| W&B run creation  | `WandbLogger` instantiated by Hydra — opt-in via `logger=wandb` (the default `many_loggers` is CSV + TB) | `configs/logger/wandb.yaml`        |
| Entity / project  | Env-var driven: `entity: ${oc.env:WANDB_ENTITY,null}`, `project: "${oc.env:WANDB_PROJECT,synth-setter}"` | `configs/logger/wandb.yaml:10,13`  |
| Default compose   | `many_loggers` composes `csv + tensorboard` (W&B excluded by default)                                    | `configs/logger/many_loggers.yaml` |
| Run ID            | `null` (W&B auto-generates)                                                                              | `configs/logger/wandb.yaml:8`      |
| Checkpoint upload | `log_model: "all"`                                                                                       | `configs/logger/wandb.yaml:11`     |
| Code saving       | `wandb.Settings(code_dir=".")`                                                                           | `configs/logger/wandb.yaml:17-19`  |
| Run teardown      | `wandb.finish()` in `task_wrapper` finally block                                                         | `src/utils/utils.py:102-107`       |

**No direct `wandb.init()` calls exist in runtime code.** One `wandb.config.update()` call exists: `log_wandb_provenance()` in `src/utils/logging_utils.py:91` writes provenance metadata (see [2g](#2g-provenance-metadata-logged-once-at-run-start)).

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

| Module                    | Metric                                                   | Step | Epoch |
| ------------------------- | -------------------------------------------------------- | ---- | ----- |
| `SurgeFlowMatchingModule` | `train/loss`                                             | yes  | yes   |
|                           | `train/penalty`                                          | yes  | yes   |
|                           | `val/param_mse`                                          | —    | yes   |
|                           | `test/param_mse`                                         | —    | yes   |
|                           | `vector_field/*_norm`                                    | yes  | —     |
|                           | `encoder/*_norm`                                         | yes  | —     |
| `KSinFlowMatchingModule`  | `train/loss`                                             | yes  | yes   |
|                           | `train/penalty`                                          | yes  | yes   |
|                           | `val/lsd`, `val/chamfer`                                 | —    | yes   |
|                           | `test/param_mse`, `test/lsd`, `test/chamfer`, `test/lad` | —    | yes   |
|                           | `vector_field/*_norm`, `encoder/*_norm`                  | yes  | yes   |
| `SurgeFlowVAEModule`      | `train/loss`, `train/param_mean`, `train/param_std`      | yes  | yes   |
|                           | `train/{reconstruction,latent,param}_loss`               | yes  | yes   |
|                           | `train/beta`                                             | yes  | —     |
|                           | `val/{reconstruction,latent,param}_loss`                 | —    | yes   |
|                           | `val/param_mean`, `val/param_std`                        | —    | yes   |
|                           | `test/{reconstruction,latent,param}_loss`                | —    | yes   |
|                           | `net/*` gradient norms                                   | yes  | —     |
| `SurgeFeedForwardModule`  | `train/loss`                                             | yes  | yes   |
|                           | `val/param_mse`, `test/param_mse`                        | —    | yes   |
| `KSinFeedForwardModule`   | `train/loss`                                             | yes  | yes   |
|                           | `val/lsd`, `val/chamfer`, `val/loss`                     | —    | yes   |
|                           | `test/*` metrics                                         | —    | yes   |
| `MNISTLitModule`          | `train/loss`, `train/acc`                                | —    | yes   |
|                           | `val/loss`, `val/acc`, `val/acc_best`                    | —    | yes   |
|                           | `test/loss`, `test/acc`                                  | —    | yes   |

### 2c. Callbacks — Visualization (via Lightning logger dispatch)

Image-producing callbacks route figures through `_log_figure` in
`src/utils/callbacks.py`, which dispatches to `WandbLogger.log_image` and/or
`TensorBoardLogger.experiment.add_figure` depending on the attached loggers.
Under the default `many_loggers` composition (CSV + TB), plots land in
TensorBoard; with `logger=wandb` they go to W&B; with both attached they go
to both.

| Callback                           | Logged key                       | Trigger                                         | Symbol                                                               |
| ---------------------------------- | -------------------------------- | ----------------------------------------------- | -------------------------------------------------------------------- |
| `PlotLossPerTimestep`              | `plot` (image)                   | `on_validation_epoch_end`                       | `src/utils/callbacks.py::PlotLossPerTimestep._log_plot`              |
| `PlotPositionalEncodingSimilarity` | `pos_enc_similarity` (image)     | `on_validation_epoch_end`                       | `src/utils/callbacks.py::PlotPositionalEncodingSimilarity._log_plot` |
| `PlotLearntProjection`             | `assignment`, `value` (images)   | `on_validation_epoch_end` or every N steps      | `src/utils/callbacks.py::PlotLearntProjection._log_plots`            |
| `LogPerParamMSE`                   | `per_param_mse/{name}` per param | `on_validation_epoch_end` (via `self.log_dict`) | `src/utils/callbacks.py::LogPerParamMSE`                             |

### 2d. Callbacks — Non-W&B

| Callback              | What it does                                           | Config                                     |
| --------------------- | ------------------------------------------------------ | ------------------------------------------ |
| `ModelCheckpoint`     | Saves `.ckpt` locally (uploaded by `log_model: "all"`) | `configs/callbacks/model_checkpoint.yaml`  |
| `LearningRateMonitor` | Logs LR to Lightning logger                            | `configs/callbacks/lr_monitor.yaml`        |
| `RichProgressBar`     | Terminal display only                                  | `configs/callbacks/rich_progress_bar.yaml` |
| `ModelSummary`        | Prints param summary to console                        | `configs/callbacks/model_summary.yaml`     |
| `PredictionWriter`    | Saves predictions to `.pt` files locally               | `src/utils/callbacks.py::PredictionWriter` |

### 2e. Gradient Watching

If `cfg.watch_gradients` is set, `watch_gradients()` calls
`WandbLogger.watch(model, log="gradients")` — logs gradient histograms per
layer according to the WandbLogger / W&B logging defaults.

Source: `src/utils/utils.py:138-149`, called from `src/train.py:91-93`.

### 2g. Provenance metadata (logged once at run start)

`log_wandb_provenance()` (`src/utils/logging_utils.py:64-98`) is called in both
`src/train.py:89` and `src/eval.py:82`, after `log_hyperparameters()`.

| Key          | Source               | Example                                |
| ------------ | -------------------- | -------------------------------------- |
| `github_sha` | `git rev-parse HEAD` | `3e60c47c6131...`                      |
| `image_tag`  | `IMAGE_TAG` env var  | `dev-snapshot-abc123`                  |
| `command`    | `" ".join(sys.argv)` | `"src/train.py experiment=surge/flow"` |

Written via `wandb.config.update(..., allow_val_change=True)`.

### 2h. Auto-Captured by W&B (not in our code)

| What           | How                                            |
| -------------- | ---------------------------------------------- |
| `sys.argv`     | W&B auto-captures command-line args            |
| System metrics | GPU utilization, memory, CPU, disk (W&B agent) |
| Git SHA + diff | W&B auto-captures if in a git repo             |
| Python deps    | `requirements.txt` / `pip freeze` snapshot     |

______________________________________________________________________

## 3. Artifacts

| Artifact          | Source                                 | When                                                                                      |
| ----------------- | -------------------------------------- | ----------------------------------------------------------------------------------------- |
| Model checkpoints | `ModelCheckpoint` + `log_model: "all"` | Every 5000 steps (with `default_surge` callbacks) + best + last, all uploaded immediately |
| Source code       | `wandb.Settings(code_dir=".")`         | Run start                                                                                 |

______________________________________________________________________

## 4. Entry Points

| Entry point    | W&B usage                                                                                       | File           |
| -------------- | ----------------------------------------------------------------------------------------------- | -------------- |
| `src/train.py` | Full: logger init → hparams → provenance → train metrics → test metrics → teardown              | `src/train.py` |
| `src/eval.py`  | Full: logger init → hparams → provenance → test/val metrics (+ optional predictions) → teardown | `src/eval.py`  |

Both use `@task_wrapper` which ensures `wandb.finish()` runs even on exception.

______________________________________________________________________

## 5. Known Gaps

| #   | Gap                                                                                                                                                                                                                                                                           | Impact                                                                                              | Tracking |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | -------- |
| 1   | ~~**Entity/project hardcoded** to `benhayes`/`synth-permutations`~~ **RESOLVED** — env-var driven (`WANDB_ENTITY`/`WANDB_PROJECT`); entity defaults to `null` (user's W&B default entity), project defaults to `synth-setter`                                                 | ~~Runs log to wrong account~~                                                                       | #133     |
| 2   | **No `wandb.config` for env vars** — W&B captures `sys.argv` but not env vars like `TRAINING_ARGS`. **Partially resolved:** `IMAGE_TAG` is now captured by `log_wandb_provenance()`.                                                                                          | Config passed via env vars is silently missing from W&B (except `IMAGE_TAG`)                        | #252     |
| 3   | ~~**No `github_sha` in `wandb.config`**~~ **RESOLVED** — `log_wandb_provenance()` now logs `github_sha` via `git rev-parse HEAD`                                                                                                                                              | ~~Can't reliably link a run to the exact code that produced it~~                                    | —        |
| 4   | **No GitHub issue integration** — train job doesn't post run ID back to GitHub                                                                                                                                                                                                | Manual lookup to match runs to issues                                                               | #263     |
| 5   | ~~**`log_model: true` vs `"all"`**~~ **RESOLVED** — changed to `log_model: "all"` for crash resilience (every checkpoint uploaded immediately)                                                                                                                                | —                                                                                                   | —        |
| 6   | ~~**Visualization callbacks use `wandb.log()` directly** — bypasses Lightning logger abstraction~~ **RESOLVED** — callbacks now dispatch through `_log_figure` to whichever Lightning loggers are attached (WandbLogger and/or TensorBoardLogger); CSV-only setups are silent | ~~Breaks if logger is swapped; step alignment relies on `trainer.global_step`~~                     | #614     |
| 7   | **`torch.compile` crashes test-stage `setup()`** — eval after training fails                                                                                                                                                                                                  | Post-training test metrics never logged to W&B                                                      | #248     |
| 8   | **No structured run ID convention** — `id: null` means W&B generates random IDs                                                                                                                                                                                               | Can't reconstruct run lineage from ID alone; design doc specifies `{config_id}-{timestamp}` pattern | —        |
