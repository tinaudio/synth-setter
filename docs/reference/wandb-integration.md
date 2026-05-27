# W&B Integration Reference

> **Code version**: `1970388` (2026-04-25, `feat/wandb-default-logger`)
> **PyTorch**: see `pyproject.toml` (`[project.optional-dependencies].torch`) · **Lightning**: see `pyproject.toml` (`[project.optional-dependencies].torch`)
> **Tracking**: #252, #263

______________________________________________________________________

## Overview

W&B runs are created via Lightning's `WandbLogger` — there are no direct
`wandb.init()` calls in training or eval code. The logger is instantiated via
Hydra config and passed to the `Trainer`. Metric logging goes through
Lightning's `self.log()` / `self.log_dict()` API; visualization callbacks
route matplotlib figures through a small logger-dispatch helper
(`_log_figure` in `src/synth_setter/utils/callbacks.py`) that calls
`WandbLogger.log_image` or `TensorBoardLogger.experiment.add_figure`
depending on which loggers are attached.

______________________________________________________________________

## 1. Initialization

| Concern           | How it works                                                                                             | File                                                |
| ----------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| W&B run creation  | `WandbLogger` instantiated by Hydra — included in the default `many_loggers` compose (W&B + CSV + TB)    | `src/synth_setter/configs/logger/wandb.yaml`        |
| Entity / project  | Env-var driven: `entity: ${oc.env:WANDB_ENTITY,null}`, `project: "${oc.env:WANDB_PROJECT,synth-setter}"` | `src/synth_setter/configs/logger/wandb.yaml:10,13`  |
| Default compose   | `many_loggers` composes `csv + tensorboard + wandb` (W&B enabled by default)                             | `src/synth_setter/configs/logger/many_loggers.yaml` |
| Run ID            | `null` (W&B auto-generates)                                                                              | `src/synth_setter/configs/logger/wandb.yaml:8`      |
| Checkpoint upload | `log_model: "all"`                                                                                       | `src/synth_setter/configs/logger/wandb.yaml:11`     |
| Code saving       | `wandb.Settings(code_dir=".")`                                                                           | `src/synth_setter/configs/logger/wandb.yaml:17-19`  |
| Run teardown      | `wandb.finish()` in `task_wrapper` finally block                                                         | `src/synth_setter/utils/utils.py:101-106`           |

**No direct `wandb.init()` calls exist in runtime code.** One `wandb.config.update()` call exists: `log_wandb_provenance()` in `src/synth_setter/utils/logging_utils.py:91` writes provenance metadata (see [2g](#2g-provenance-metadata-logged-once-at-run-start)).

______________________________________________________________________

## 2. What Gets Logged

### 2a. Hyperparameters (logged once at run start)

`log_hyperparameters()` in `src/synth_setter/utils/logging_utils.py` sends a single dict
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

### 2c. Callbacks — Visualization (via Lightning logger dispatch)

Image-producing callbacks route figures through `_log_figure` in
`src/synth_setter/utils/callbacks.py`, which dispatches to `WandbLogger.log_image` and/or
`TensorBoardLogger.experiment.add_figure` depending on the attached loggers.
Under the default `many_loggers` composition (W&B + CSV + TB), plots land in
both W&B and TensorBoard; with `logger=tensorboard` they go to TensorBoard
only; with `logger=wandb` they go to W&B only.

| Callback                           | Logged key                       | Trigger                                         | Symbol                                                                            |
| ---------------------------------- | -------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------- |
| `PlotLossPerTimestep`              | `plot` (image)                   | `on_validation_epoch_end`                       | `src/synth_setter/utils/callbacks.py::PlotLossPerTimestep._log_plot`              |
| `PlotPositionalEncodingSimilarity` | `pos_enc_similarity` (image)     | `on_validation_epoch_end`                       | `src/synth_setter/utils/callbacks.py::PlotPositionalEncodingSimilarity._log_plot` |
| `PlotLearntProjection`             | `assignment`, `value` (images)   | `on_validation_epoch_end` or every N steps      | `src/synth_setter/utils/callbacks.py::PlotLearntProjection._log_plots`            |
| `LogPerParamMSE`                   | `per_param_mse/{name}` per param | `on_validation_epoch_end` (via `self.log_dict`) | `src/synth_setter/utils/callbacks.py::LogPerParamMSE`                             |

### 2d. Callbacks — Non-W&B

| Callback              | What it does                                           | Config                                                      |
| --------------------- | ------------------------------------------------------ | ----------------------------------------------------------- |
| `ModelCheckpoint`     | Saves `.ckpt` locally (uploaded by `log_model: "all"`) | `src/synth_setter/configs/callbacks/model_checkpoint.yaml`  |
| `LearningRateMonitor` | Logs LR to Lightning logger                            | `src/synth_setter/configs/callbacks/lr_monitor.yaml`        |
| `RichProgressBar`     | Terminal display only                                  | `src/synth_setter/configs/callbacks/rich_progress_bar.yaml` |
| `ModelSummary`        | Prints param summary to console                        | `src/synth_setter/configs/callbacks/model_summary.yaml`     |
| `PredictionWriter`    | Saves predictions to `.pt` files locally               | `src/synth_setter/utils/callbacks.py::PredictionWriter`     |

### 2e. Gradient Watching

If `cfg.watch_gradients` is set, `watch_gradients()` calls
`WandbLogger.watch(model, log="gradients")` — logs gradient histograms per
layer according to the WandbLogger / W&B logging defaults.

Source: `src/synth_setter/utils/utils.py:137-148`, called from `src/synth_setter/cli/train.py:91-93`.

### 2g. Provenance metadata (logged once at run start)

`log_wandb_provenance()` (`src/synth_setter/utils/logging_utils.py:64-98`) is called in both
`src/synth_setter/cli/train.py:89` and `src/synth_setter/cli/eval.py:82`, after `log_hyperparameters()`.

| Key          | Source               | Example                                                 |
| ------------ | -------------------- | ------------------------------------------------------- |
| `github_sha` | `git rev-parse HEAD` | `3e60c47c6131...`                                       |
| `image_tag`  | `IMAGE_TAG` env var  | `dev-snapshot-abc123`                                   |
| `command`    | `" ".join(sys.argv)` | `"src/synth_setter/cli/train.py experiment=surge/flow"` |

Written via `wandb.config.update(..., allow_val_change=True)`.

### 2h. Auto-Captured by W&B (not in our code)

| What           | How                                            |
| -------------- | ---------------------------------------------- |
| `sys.argv`     | W&B auto-captures command-line args            |
| System metrics | GPU utilization, memory, CPU, disk (W&B agent) |
| Git SHA + diff | W&B auto-captures if in a git repo             |
| Python deps    | `pyproject.toml` / `pip freeze` snapshot       |

### 2i. Eval-mode Audio Metrics (predict mode only)

When `synth-setter-eval mode=predict evaluation.compute_metrics=true` runs and a W&B run is active, `_log_audio_metrics_to_wandb` (`src/synth_setter/cli/eval.py`) forwards the values from `aggregated_metrics.csv` directly to `wandb.run.log`, so they land in the same run as `test/param_mse`:

| Key                | What                                                        |
| ------------------ | ----------------------------------------------------------- |
| `audio/mss_mean`   | Multi-scale log-mel spectrogram distance, mean over samples |
| `audio/mss_std`    | Same, standard deviation                                    |
| `audio/wmfcc_mean` | DTW-aligned MFCC distance, mean                             |
| `audio/wmfcc_std`  | Same, standard deviation                                    |
| `audio/sot_mean`   | Spectral optimal-transport distance, mean                   |
| `audio/sot_std`    | Same, standard deviation                                    |
| `audio/rms_mean`   | RMS envelope cosine similarity, mean                        |
| `audio/rms_std`    | Same, standard deviation                                    |

The same dict is also merged into the dict returned by `evaluate()` alongside Lightning's `trainer.callback_metrics`. See [eval-pipeline.md §5.1](../design/eval-pipeline.md) for the surrounding subprocess chain.

______________________________________________________________________

## 3. Artifacts

| Artifact                 | Source                                                             | When                                                                                                |
| ------------------------ | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| Model checkpoints        | `ModelCheckpoint` + `log_model: "all"`                             | Every 5000 steps (with `default_surge` callbacks) + best + last, all uploaded immediately           |
| Source code              | `wandb.Settings(code_dir=".")`                                     | Run start                                                                                           |
| `<task_name>-input-spec` | `_log_spec_artifact` in `src/synth_setter/cli/generate_dataset.py` | Dataset-generation run start; artifact type `dataset-spec`, payload = `DatasetSpec.model_dump_json` |

______________________________________________________________________

## 4. Entry Points

| Entry point                                | W&B usage                                                                                                                                                                            | File                                       |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------ |
| `src/synth_setter/cli/train.py`            | Full: logger init → hparams → provenance → train metrics → test metrics → teardown                                                                                                   | `src/synth_setter/cli/train.py`            |
| `src/synth_setter/cli/eval.py`             | Full: logger init → hparams → provenance → test/val metrics (+ optional predictions) → predict-mode `audio/<metric>_{mean,std}` keys from `_log_audio_metrics_to_wandb` → teardown   | `src/synth_setter/cli/eval.py`             |
| `src/synth_setter/cli/generate_dataset.py` | Dataset-generation: logger init pinned to `spec.run_id` → spec hparams → `<task_name>-input-spec` artifact → per-shard metrics → run summary → `finalize(status)` + `wandb.finish()` | `src/synth_setter/cli/generate_dataset.py` |

Both training and eval use `@task_wrapper` which ensures `wandb.finish()` runs even on exception.
`generate_dataset` brackets `generate(...)` in its own `try/finally` that calls `_close_loggers` — see §5 for the metric / run-id contract.

______________________________________________________________________

## 5. Dataset Generation Runs

`src/synth_setter/cli/generate_dataset.py` instantiates a `WandbLogger` via Hydra
(`configs/dataset.yaml` includes `- logger: wandb` in its defaults list) and pins
`logger.wandb.id` to `spec.run_id` — derived deterministically by
`make_dataset_wandb_run_id` (`src/synth_setter/pipeline/schemas/prefix.py`) — so
the W&B run ID matches the R2 prefix under `data/<task_name>/<run_id>/`. This is
the single binding point: re-running with the same `spec` resumes the same W&B run.

### 5a. Hyperparameters and artifact (logged once at run start)

| Key / artifact           | Source                                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------ |
| All `DatasetSpec` fields | `_log_hyperparams(loggers, spec)` → `spec.model_dump(mode="json")`                                           |
| `<task_name>-input-spec` | `_log_spec_artifact(loggers, spec)` — artifact type `dataset-spec`, payload `spec.model_dump_json(indent=2)` |

### 5b. Per-shard metrics (one history row per shard, `step=shard_id`)

| Key                    | What                                                                           |
| ---------------------- | ------------------------------------------------------------------------------ |
| `shard/bytes`          | Local shard file size in bytes (stable; shards retained at `work_dir`)         |
| `shard/render_seconds` | Wall-clock seconds from subprocess invoke through upload-end; `0.0` on R2-skip |

Emitted by `_log_shard_metrics` from `_render_one_owned_shard` in both the
serial and parallel dispatchers.

### 5c. Run summary (one terminal row)

| Key                             | What                                              |
| ------------------------------- | ------------------------------------------------- |
| `shards/rendered`               | Shards this rank actually rendered                |
| `shards/skipped`                | Shards short-circuited by the R2-skip probe       |
| `shards/total`                  | `len(my_range)` — owned shard count for this rank |
| `generation/elapsed_seconds`    | Wall-clock dispatcher duration (mirrors #1304)    |
| `generation/samples`            | `rendered * spec.render.samples_per_shard`        |
| `generation/samples_per_second` | `samples / elapsed_s` (0.0 when `elapsed_s == 0`) |

Emitted by `_log_summary` after the dispatcher returns. Final `finalize(status)` records
the run as `"success"` even on partial-success (some shards rendered, others failed);
crash-on-raise reports `"failed"`.

### 5d. Linked issues

| Issue                                                         | Topic                                                                                                                   |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| [#1318](https://github.com/tinaudio/synth-setter/issues/1318) | v2: per-sample loudness telemetry relay (stdout protocol, worker → launcher) — deferred non-goal from the design doc Q5 |

______________________________________________________________________

## 6. Known Gaps

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
