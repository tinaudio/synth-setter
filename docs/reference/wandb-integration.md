# W&B Integration Reference

> **Code version**: `1970388` (2026-04-25, `feat/wandb-default-logger`)
> **PyTorch**: see `pyproject.toml` (`[dependency-groups].torch`) · **Lightning**: see `pyproject.toml` (`[dependency-groups].torch`)
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

| Concern           | How it works                                                                                                                                                                                                                                                                                      | File                                                            |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| W&B run creation  | `WandbLogger` instantiated by Hydra — included in the default `many_loggers` compose (W&B + CSV + TB)                                                                                                                                                                                             | `src/synth_setter/configs/logger/wandb.yaml`                    |
| Entity / project  | Env-var driven: `entity: ${oc.env:WANDB_ENTITY,null}`, `project: "${oc.env:WANDB_PROJECT,synth-setter}"`                                                                                                                                                                                          | `src/synth_setter/configs/logger/wandb.yaml:10,15`              |
| Default compose   | `many_loggers` composes `csv + tensorboard + wandb` (W&B enabled by default)                                                                                                                                                                                                                      | `src/synth_setter/configs/logger/many_loggers.yaml`             |
| Run ID            | `null` (W&B auto-generates)                                                                                                                                                                                                                                                                       | `src/synth_setter/configs/logger/wandb.yaml:8`                  |
| Checkpoint upload | `log_model: False` — no checkpoint files to W&B; the best ckpt is uploaded to R2 at train end and referenced by the `model-{config_id}` artifact                                                                                                                                                  | `src/synth_setter/configs/logger/wandb.yaml:13`                 |
| Code saving       | `wandb.Settings(code_dir=".")`                                                                                                                                                                                                                                                                    | `src/synth_setter/configs/logger/wandb.yaml` § `wandb.settings` |
| Console capture   | `wandb.Settings(console="wrap", console_multipart=True)` — `redirect` captures into a local `output.log` that wandb 0.26.x never uploads (#1465); `wrap` reaches the server but sees only this process's Python-level writes. Subprocess tee + multipart semantics: see the note below this table | `src/synth_setter/configs/logger/wandb.yaml` § `wandb.settings` |
| Run teardown      | `wandb.finish()` in `task_wrapper` finally block                                                                                                                                                                                                                                                  | `src/synth_setter/utils/utils.py` § `task_wrapper`              |

**Subprocess console capture.** `generate_dataset` tees the children it spawns — the renderer, the per-shard rclone upload, and the inline oracle eval — through `sys.stderr` via `check_call_streamed` (`src/synth_setter/pipeline/subprocess_stream.py`, exit-keyed so a pipe-holding descendant can't stall it). Other rclone call sites (spec upload, `finalize_from_spec`'s `r2_io` transfers) still write to the inherited fd and bypass capture. `console_multipart=True` gives each resumed session (generate → finalize → oracle eval) its own `logs/output_*.log` instead of overwriting one `output.log`.

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
| `ModelCheckpoint`     | Saves `.ckpt` locally (best ckpt later uploaded to R2) | `src/synth_setter/configs/callbacks/model_checkpoint.yaml`  |
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

`log_wandb_provenance()` (`src/synth_setter/utils/logging_utils.py:64-98`) is called on all three
entrypoints: `src/synth_setter/cli/train.py` and `src/synth_setter/cli/eval.py` after
`log_hyperparameters()`, and `src/synth_setter/cli/generate_dataset.py` inside `generate()` right
after `_log_hyperparams()` (covering both the local `main` and worker `from_hydra` paths).

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

| Key                        | What                                                                                                                        |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `audio/mss_mean`           | Multi-scale log-mel spectrogram distance, mean over samples                                                                 |
| `audio/mss_std`            | Same, standard deviation                                                                                                    |
| `audio/wmfcc_mean`         | DTW-aligned MFCC distance, mean                                                                                             |
| `audio/wmfcc_std`          | Same, standard deviation                                                                                                    |
| `audio/sot_mean`           | Spectral optimal-transport distance, mean                                                                                   |
| `audio/sot_std`            | Same, standard deviation                                                                                                    |
| `audio/rms_mean`           | RMS envelope cosine similarity, mean                                                                                        |
| `audio/rms_std`            | Same, standard deviation                                                                                                    |
| `audio/per_sample_metrics` | Per-sample metrics from `metrics.csv` as a `wandb.Table`; columns match `compute_audio_metrics` output (one row per sample) |

When the auto-shuffle probe ran (uniform params, ≥ 2 sample dirs), a parallel set of `shuffled_audio/<metric>_{mean,std}` keys is also logged from `aggregated_metrics_shuffled.csv`. `_log_metrics_csv_to_wandb` (`src/synth_setter/cli/eval.py`) is a no-op when `metrics.csv` is absent or `wandb.run` is unset; wandb errors are swallowed so a logging failure never aborts the run.

The aggregated scalar metrics dict is also merged into the dict returned by `evaluate()` alongside Lightning's `trainer.callback_metrics`; the `audio/per_sample_metrics` Table is W&B-only and is not included in that dict. See [eval-pipeline.md §5.1](../design/eval-pipeline.md) for the surrounding subprocess chain.

______________________________________________________________________

## 3. Artifacts

| Artifact                 | Source                                                                                           | When                                                                                                                                                                                                                                                                                            |
| ------------------------ | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model checkpoints        | `ModelCheckpoint` (best ckpt → R2; `log_model: False`)                                           | Best + last + every-5000-step `.ckpt` written locally; only the best is uploaded to R2 at train end (no checkpoint files go to W&B)                                                                                                                                                             |
| Source code              | `wandb.Settings(code_dir=".")`                                                                   | Run start                                                                                                                                                                                                                                                                                       |
| `<task_name>-input-spec` | `_log_spec_artifact` in `src/synth_setter/cli/generate_dataset.py`                               | Dataset-generation run start; artifact type `dataset-spec`, payload = `DatasetSpec.model_dump_json`                                                                                                                                                                                             |
| `data-{task_name}`       | `build_dataset_artifact` / `_log_dataset_artifact` in `src/synth_setter/cli/finalize_dataset.py` | Finalize, after the R2 outputs land; type `dataset`, `s3://` R2 references (`checksum=False`), metadata `shard_count` / `n_samples` / `git_sha`                                                                                                                                                 |
| `model-{config_id}`      | `build_model_artifact` / `_log_model_artifact` in `src/synth_setter/cli/train.py`                | Train end, after fit/test (global-zero); type `model`, metadata `git_sha`; the best ckpt is uploaded to `r2://{r2.bucket}/checkpoints/{config_id}/model.ckpt` and referenced as an `s3://` URI (`checksum=False`); degrades to lineage-only when R2 is unreachable or no ckpt was written (#92) |
| `eval-{config_id}`       | `build_eval_results_artifact` / `_log_eval_results_artifact` in `src/synth_setter/cli/eval.py`   | After the eval output dir is mirrored to R2 (global-zero only); type `eval-results`, `s3://` R2 reference (`checksum=False`), metadata = scalar summary metrics + `git_sha`                                                                                                                     |

______________________________________________________________________

## 4. Entry Points

| Entry point                                | W&B usage                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | File                                       |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| `src/synth_setter/cli/train.py`            | Full: logger init → hparams → provenance → dataset `use_artifact` lineage (`use_input_artifacts`, opt-in `consumed_dataset_config_id`) → train metrics → test metrics → `model-{config_id}` `model` artifact → teardown                                                                                                                                                                                                                                                                                         | `src/synth_setter/cli/train.py`            |
| `src/synth_setter/cli/eval.py`             | Full: logger init → hparams → provenance → model+dataset `use_artifact` lineage (`use_input_artifacts`, opt-in `consumed_train_config_id`/`consumed_dataset_config_id`) → test/val metrics (+ optional predictions) → predict-mode `audio/<metric>_{mean,std}` keys from `_log_audio_metrics_to_wandb` + `audio/per_sample_metrics` Table from `_log_metrics_csv_to_wandb` → (global-zero, when `upload_output_dir_uri` is set) `eval-{config_id}` `eval-results` artifact with `s3://` R2 reference → teardown | `src/synth_setter/cli/eval.py`             |
| `src/synth_setter/cli/generate_dataset.py` | Dataset-generation: logger init pinned to `spec.run_id` → spec hparams → provenance → `<task_name>-input-spec` artifact → per-shard metrics → run summary → `finalize(status)` + `wandb.finish()`                                                                                                                                                                                                                                                                                                               | `src/synth_setter/cli/generate_dataset.py` |
| `src/synth_setter/cli/finalize_dataset.py` | Dataset-finalize: resumes the data-generation run (`id=spec.run_id`, `job_type=data-generation`, `resume=allow`) → logs the `data-{config_id}` `dataset` artifact with `s3://` R2 references → `close_loggers`. Best-effort: a finalize without `WANDB_API_KEY` / logger group degrades to a no-op                                                                                                                                                                                                              | `src/synth_setter/cli/finalize_dataset.py` |

Both training and eval use `@task_wrapper` which ensures `wandb.finish()` runs even on exception.
`generate_dataset` brackets `generate(...)` in its own `try/finally` that calls `close_loggers` (now in `synth_setter.utils.instantiators`, shared with finalize) — see §5 for the metric / run-id contract.

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

Emitted by `_log_summary` after the dispatcher returns. The dispatcher is fail-fast — there
is no partial-success path. Either every owned shard's contract is fulfilled (rendered or
short-circuited by the R2-skip probe) and `finalize(status)` records `"success"`, or any
shard's exception propagates up `generate()`'s `try/except`, `status` flips to `"failed"`,
the summary is not emitted, and `close_loggers` calls `finalize("failed")` + `wandb.finish()`
in the `finally`.

### 5d. Linked issues

| Issue                                                         | Topic                                                                                                                   |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| [#1318](https://github.com/tinaudio/synth-setter/issues/1318) | v2: per-sample loudness telemetry relay (stdout protocol, worker → launcher) — deferred non-goal from the design doc Q5 |

### 5e. Sweeps

Operator-authored sweep configs under `sweeps/` drive `synth-setter-generate-dataset`
via `wandb sweep` + `wandb agent`. Each YAML pins `program:` at
`src/synth_setter/cli/generate_dataset.py` and uses `${args_no_hyphens}` so sweep
parameters reach the launcher as Hydra overrides (e.g. `render.plugin_reload_cadence=once`).
`entity:` / `project:` are pinned in the YAML — wandb's sweep CLI does not honor
`WANDB_ENTITY` / `WANDB_PROJECT` at sweep-creation time (those only steer
`wandb.init` inside each trial).

To launch:

```bash
wandb sweep sweeps/generate_dataset_cadence.yaml   # prints sweep_id
wandb agent <entity>/<project>/<sweep_id>          # runs trials
```

Each trial subprocess opens its own wandb run with `id = spec.run_id`; the
`WANDB_SWEEP_ID` env (set by the agent) attaches the run to the sweep grid.

### 5f. Inline oracle eval (`oracle_eval_inline=true`)

When `oracle_eval_inline=true`, the local-run path shells out to
`synth_setter.cli.eval` **once per split (train, val, test)** after `generate(...)` has closed its run.
`_run_oracle_eval_subprocess` (`src/synth_setter/cli/generate_dataset.py`)
re-opens the same run via `logger.wandb.id=<spec.run_id> +logger.wandb.resume=must`, runs `mode=predict` with `render=surge_simple` to
re-render the predicted params, and deposits audio-similarity metrics onto the
generate run — a `wandb sync` then merges both phases under one run id. Because
all three splits resume the **same** run, the metric keys are namespaced per
split so they don't overwrite each other's run summary: `test` keeps the bare
`audio/*` key, while `train`/`val` are logged under `train/audio/*` and
`val/audio/*` (via `+evaluation.metric_prefix=<split>/`). The prefix applies to
every metric key, so the `shuffled_audio/*` keys (§2) become `train/shuffled_audio/*`
etc. too. The eval subprocess
inherits `WANDB_MODE` from the launcher, so its offline/online posture follows
the parent's. Console logs survive the split-by-split resumes via
`console_multipart=True`: each session uploads its own `logs/output_*.log`
rather than overwriting a single server-side `output.log`.

______________________________________________________________________

## 6. Known Gaps

| #   | Gap                                                                                                                                                                                                                                                                           | Impact                                                                                              | Tracking |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | -------- |
| 1   | ~~**Entity/project hardcoded** to `benhayes`/`synth-permutations`~~ **RESOLVED** — env-var driven (`WANDB_ENTITY`/`WANDB_PROJECT`); entity defaults to `null` (user's W&B default entity), project defaults to `synth-setter`                                                 | ~~Runs log to wrong account~~                                                                       | #133     |
| 2   | **No `wandb.config` for env vars** — W&B captures `sys.argv` but not env vars like `TRAINING_ARGS`. **Partially resolved:** `IMAGE_TAG` is now captured by `log_wandb_provenance()`.                                                                                          | Config passed via env vars is silently missing from W&B (except `IMAGE_TAG`)                        | #252     |
| 3   | ~~**No `github_sha` in `wandb.config`**~~ **RESOLVED** — `log_wandb_provenance()` now logs `github_sha` via `git rev-parse HEAD`                                                                                                                                              | ~~Can't reliably link a run to the exact code that produced it~~                                    | —        |
| 4   | **No GitHub issue integration** — train job doesn't post run ID back to GitHub                                                                                                                                                                                                | Manual lookup to match runs to issues                                                               | #263     |
| 5   | ~~**`log_model` checkpoint upload to W&B**~~ **RESOLVED** — set to `log_model: False`; no checkpoint files go to W&B (5 GB budget). The best ckpt is uploaded to R2 and referenced by the `model-{config_id}` artifact                                                        | —                                                                                                   | #92      |
| 6   | ~~**Visualization callbacks use `wandb.log()` directly** — bypasses Lightning logger abstraction~~ **RESOLVED** — callbacks now dispatch through `_log_figure` to whichever Lightning loggers are attached (WandbLogger and/or TensorBoardLogger); CSV-only setups are silent | ~~Breaks if logger is swapped; step alignment relies on `trainer.global_step`~~                     | #614     |
| 7   | **`torch.compile` crashes test-stage `setup()`** — eval after training fails                                                                                                                                                                                                  | Post-training test metrics never logged to W&B                                                      | #248     |
| 8   | **No structured run ID convention** — `id: null` means W&B generates random IDs                                                                                                                                                                                               | Can't reconstruct run lineage from ID alone; design doc specifies `{config_id}-{timestamp}` pattern | —        |
