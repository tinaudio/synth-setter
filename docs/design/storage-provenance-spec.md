# Storage & Provenance Spec

> Authoritative source of truth for R2 paths, W&B artifacts, and GitHub Actions workflows.
> Individual design docs must not define their own storage or provenance conventions — point here instead.

| Field        | Value      |
| ------------ | ---------- |
| Status       | Draft      |
| Last Updated | 2026-03-20 |
| Tracking     | #122       |

______________________________________________________________________

## 1. IDs

| ID                     | Construction                                                      | Source                                                           | Example                           |
| ---------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------- | --------------------------------- |
| `dataset_config_id`    | Config filename (stem)                                            | `src/synth_setter/configs/experiment/generate_dataset/{id}.yaml` | `diva-v1`                         |
| `dataset_wandb_run_id` | Configurable, default `{dataset_config_id}-{YYYYMMDDTHHMMSSsssZ}` | `wandb.init(id=...)`                                             | `diva-v1-20260312T143022500Z`     |
| `train_config_id`      | Config filename (stem)                                            | `src/synth_setter/configs/experiment/.../{id}.yaml`              | `flow-simple`                     |
| `train_wandb_run_id`   | Configurable, default `{train_config_id}-{YYYYMMDDTHHMMSSsssZ}`   | `wandb.init(id=...)`                                             | `flow-simple-20260315T091500250Z` |
| `eval_config_id`       | Eval dataset config filename (stem)                               | `src/synth_setter/configs/experiment/{id}.yaml`                  | `nsynth-v1`                       |
| `eval_wandb_run_id`    | Configurable, default `{eval_config_id}-{YYYYMMDDTHHMMSSsssZ}`    | `wandb.init(id=...)`                                             | `nsynth-v1-20260320T160000750Z`   |

- `*_config_id` = filename of the YAML config, without extension
- `*_wandb_run_id` = the W&B run ID, set via `wandb.init(id=...)`. Default convention is `{*_config_id}-{timestamp}`, but the path format is agnostic to how the ID is generated.
- Default timestamp format: `YYYYMMDDTHHMMSSsssZ` (millisecond precision, UTC, filesystem-safe). The `sss` suffix is a zero-padded 3-digit millisecond field; it disambiguates run IDs minted within the same wall-clock second by concurrent launchers.
- W&B run ID limit: 64 characters. Keep config filenames short.

______________________________________________________________________

## 2. R2 Bucket Layout

```
intermediate-data/
├── data/{dataset_config_id}/{dataset_wandb_run_id}/
├── train/{dataset_config_id}/{dataset_wandb_run_id}/{train_config_id}/{train_wandb_run_id}/
└── eval/{dataset_config_id}/{dataset_wandb_run_id}/{train_config_id}/{train_wandb_run_id}/{eval_config_id}/{eval_wandb_run_id}/
```

The three prefixes (`data/`, `train/`, `eval/`) are the canonical per-run dataset footprint.

______________________________________________________________________

## 3. R2 Contents Per Workflow

### 3a. Data Generation

> **Implementation status:** The layout below is the target architecture. The current MVP uses a flat structure: spec and all shards upload directly to `data/{config_id}/{run_id}/`. The `metadata/workers/` staging prefix and the `finalize` promotion step are **future state** — see [#406](https://github.com/tinaudio/synth-setter/issues/406).

```
data/{dataset_config_id}/{dataset_wandb_run_id}/
├── shards/                  # Future state — no shards/ subdir exists yet; current workers upload shard files directly under the run prefix root. #406
│   ├── shard-000000.h5
│   └── ...
├── metadata/                # Future state — current `input_spec.json` lives flat at the run prefix root. #385
│   ├── config.yaml          # Frozen pipeline config (provenance copy)
│   ├── input_spec.json      # Frozen input specification (authoritative; currently at `<run_prefix>input_spec.json` — `r2.prefix` already ends in `/`)
│   ├── dataset.json         # Self-describing dataset card
│   ├── dataset.complete     # Completion marker
│   └── workers/             # Future state — worker staging area; current workers write shards directly to `data/{config_id}/{run_id}/`. #406
│       ├── shards/{shard_id}/{worker_id}-{attempt_uuid}.*
│       └── attempts/{worker_id}-{attempt_uuid}/report.json
├── train.h5, val.h5, test.h5  # Split virtual datasets
└── stats.npz                   # Normalization statistics
```

- Workers may only write under `metadata/workers/` *(future state — current workers write directly to `data/{config_id}/{run_id}/`; see [#406](https://github.com/tinaudio/synth-setter/issues/406))*
- `shards/` is written only by finalize *(future state — current workers write directly into the run prefix; finalize stage does not yet exist, see [#406](https://github.com/tinaudio/synth-setter/issues/406))*
- All `rclone` operations use `--checksum`
- Datasets are immutable once `dataset.complete` exists. New versions require a new `dataset_wandb_run_id`. *(future state — completion-marker handling lands with finalize, [#406](https://github.com/tinaudio/synth-setter/issues/406))*

#### Materialized spec: two destinations, two purposes

The `DatasetSpec` JSON is written from two distinct call sites, with two distinct purposes:

| Writer                                                                                           | Location                                                                                                                                                                                                                                                                                                                                                                                    | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Consumers                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Runner (`src/synth_setter/cli/generate_dataset.py`'s `main()`, via `spec_io.write_spec_locally`) | local: `<workspace>/data/<task_name>/<run_id>/metadata/input_spec.json` (`main()` passes `_OPERATOR_WORKSPACE` via `synth_setter.workspace.operator_workspace()` — env override → checkout marker → CWD; the per-run shard `work_dir` is separately `cfg.paths.output_dir`, populated by `@hydra.main` from `${hydra:runtime.output_dir}`, and is not the operator-side spec-mirror anchor) | **Operator-side artifact**: an on-disk copy of the frozen spec available before any dispatch happens. Used for offline inspection, reproducing a run from a saved spec, and as a stable file path the workflow YAML can read when recovering the canonical R2 URI for a run whose `run_id` was sampled rather than pinned. Anticipates §3a's target `metadata/` layout while the R2 destination remains flat per [#385](https://github.com/tinaudio/synth-setter/issues/385). | The operator / CI workflow (e.g. `nightly-parallel-datagen.yml`'s `Resolve spec_uri from local materialized spec` step, which reads the JSON via `python3` because the nightly inline `generate(...)` path samples `run_id` rather than pinning it).                                                                                                                                                                                                                                                                                                   |
| Runner (`src/synth_setter/cli/generate_dataset.py`'s `main()`, via `spec_io.upload_spec`)        | `{spec.r2.prefix}input_spec.json` (via `spec.r2.input_spec_uri()`)                                                                                                                                                                                                                                                                                                                          | **Provenance**: the canonical, authoritative frozen spec, archived next to the shards it parameterized. Same object §3a's table lists as "Input spec" (target path `metadata/input_spec.json` — flat under `r2.prefix` today per [#385](https://github.com/tinaudio/synth-setter/issues/385)). Written exactly once per `main()` invocation on the launcher host; workers do not re-upload.                                                                                   | The `validate-spec` / `validate-shard` CI workflows read this copy via the `spec_uri` output of `generate-dataset-shards.yaml`. The `dispatch_via_skypilot` step injects the same URI into each worker's env as `WORKER_SPEC_URI` (informational; the worker today rebuilds the spec from the Hydra overrides the launcher pins into its `run:` command). The future `finalize` stage and `status` command ([#406](https://github.com/tinaudio/synth-setter/issues/406), [#72](https://github.com/tinaudio/synth-setter/issues/72)) will also read it. |

Pre-flight: `main()` calls `r2_io.ensure_r2_env_loaded` (dotenv merge + `rclone lsd r2:` auth ping) before either of its two writes, so a bad-creds run aborts before producing any artifacts. Subsequent rclone calls (shard uploads, skip-existing probes) inherit the populated `os.environ`.

### 3b. Training

```
train/{dataset_config_id}/{dataset_wandb_run_id}/{train_config_id}/{train_wandb_run_id}/
├── checkpoints/
│   ├── last.ckpt
│   └── best.ckpt
└── config.yaml               # Frozen experiment config
```

- Set `log_model="all"` in the W&B Lightning Logger to persist every saved checkpoint immediately as a W&B artifact (crash-resilient).

### 3c. Evaluation

```
eval/{dataset_config_id}/{dataset_wandb_run_id}/{train_config_id}/{train_wandb_run_id}/{eval_config_id}/{eval_wandb_run_id}/
├── predictions/
│   ├── pred-{batch_idx}.pt
│   ├── target-audio-{batch_idx}.pt
│   └── target-params-{batch_idx}.pt
├── audio/
│   └── {sample_N}/
│       ├── pred.wav
│       ├── target.wav
│       ├── spec.png
│       └── params.csv
└── metrics/
    ├── metrics.csv
    └── aggregated_metrics.csv
```

______________________________________________________________________

## 4. W&B Artifact Types

| Type           | Name Pattern               | Logged By                         | Example name        |
| -------------- | -------------------------- | --------------------------------- | ------------------- |
| `dataset`      | `data-{dataset_config_id}` | `pipeline.cli finalize` (planned) | `data-diva-v1`      |
| `model`        | `model-{train_config_id}`  | `src/synth_setter/cli/train.py`   | `model-flow-simple` |
| `eval-results` | `eval-{eval_config_id}`    | eval script                       | `eval-nsynth-v1`    |

> **Note:** `pipeline.cli finalize` is the target CLI (Phase 5). In Docker, the finalize step runs as `MODE=finalize-shards` (scoped, validated on experiment branch — [#408](https://github.com/tinaudio/synth-setter/issues/408)). Current entrypoint: `pipeline.entrypoints.generate_dataset`.

- W&B auto-versions artifacts (`:v0`, `:v1`, `:v2`). Each new run of the same config produces the next version.
- The `*_wandb_run_id` is stored in `artifact.metadata`, not the artifact name

**Alias strategy:**

| Alias         | Set by           | When                                                                 |
| ------------- | ---------------- | -------------------------------------------------------------------- |
| `:latest`     | W&B (automatic)  | Every `log_artifact` call                                            |
| `:best`       | Training script  | When val metric improves (`run.log_artifact(art, aliases=["best"])`) |
| `:production` | Promote workflow | When model is promoted to GitHub Release                             |

______________________________________________________________________

## 5. W&B Lineage DAG

```
dataset config
  → [data-generation run] → dataset artifact
                               ├→ [training run] → model artifact
                               │                      │
eval dataset artifact ─────────┴→ [evaluation run] ←───┘
                                        │
                                   eval-results artifact
                                        │
                                  [promote workflow] → GitHub Release
```

- Every run must call `run.use_artifact()` for inputs and `run.log_artifact()` for outputs
- `run.use_artifact()` (not `api.artifact()`) — only the former creates lineage links
- Every run must include `github_sha` in `wandb.config`

______________________________________________________________________

## 6. W&B Metadata Convention

| Location            | What Goes Here                                   | Examples                                           |
| ------------------- | ------------------------------------------------ | -------------------------------------------------- |
| `wandb.config`      | Hyperparams — things you SET before the run      | `lr`, `epochs`, `batch_size`, `github_sha`         |
| `wandb.summary`     | Final metrics — things you MEASURE after the run | `mse`, `spectral_convergence`, `param_accuracy`    |
| `artifact.metadata` | Properties of the artifact itself                | `n_samples`, `mel_shape`, `shard_count`, `git_sha` |

- Dataset properties belong on the dataset artifact, not on training runs that consume it

______________________________________________________________________

## 7. `job_type` Values

| `job_type`        | Stage         | Script                            |
| ----------------- | ------------- | --------------------------------- |
| `data-generation` | Data pipeline | `pipeline.cli finalize` (planned) |
| `training`        | Training      | `src/synth_setter/cli/train.py`   |
| `evaluation`      | Evaluation    | eval script                       |

> **Note:** `pipeline.cli finalize` is the target CLI (Phase 5). In Docker, the finalize step runs as `MODE=finalize-shards` (scoped, validated on experiment branch — [#408](https://github.com/tinaudio/synth-setter/issues/408)). Current entrypoint: `pipeline.entrypoints.generate_dataset`.

- Set on every `wandb.init(job_type=...)` call

______________________________________________________________________

## 8. GitHub Actions Workflows

| Workflow        | File                           | Trigger                              | Runner                          | Secrets             | Key Inputs                                                       |
| --------------- | ------------------------------ | ------------------------------------ | ------------------------------- | ------------------- | ---------------------------------------------------------------- |
| Tests           | `test.yml`                     | push, PR, dispatch                   | `ubuntu-latest`, `macos-latest` | —                   | —                                                                |
| GPU Tests       | `test-gpu.yml`                 | schedule, dispatch                   | `gpu-x64`                       | —                   | —                                                                |
| CPU Slow Tests  | `cpu-slow.yml`                 | push (main), dispatch                | `ubuntu-latest-4core`           | —                   | —                                                                |
| Data Generation | `generate-dataset-shards.yaml` | `workflow_call`, `workflow_dispatch` | `ubuntu-latest`                 | R2, RunPod, OCI     | see `workflow_call.inputs` in `generate-dataset-shards.yaml`     |
| Data Validation | `validate-dataset-shards.yaml` | `workflow_call`, `workflow_dispatch` | `ubuntu-latest`                 | R2                  | `image_tag`, `spec_uri`                                          |
| Training        | TBD                            | `workflow_dispatch`                  | TBD                             | R2, W&B, RunPod     | experiment, overrides                                            |
| Evaluation      | TBD                            | `workflow_dispatch`                  | TBD                             | R2, W&B             | `train_wandb_run_id`, `eval_config_id`                           |
| Model Promotion | `promote.yml` (planned)        | `workflow_dispatch`                  | `ubuntu-latest`                 | W&B, `GITHUB_TOKEN` | `train_wandb_run_id`, `eval_wandb_run_id`, `registry`, `dry_run` |

- All workflows that create W&B runs must export `GITHUB_SHA` into the run environment.
- Evaluation requires `train_wandb_run_id` (to find the model artifact) and `eval_config_id` (which dataset to evaluate on).
- Promotion requires both `train_wandb_run_id` and `eval_wandb_run_id`. It pulls the model artifact from the training run and eval metrics from the eval run.

**GitHub Release body schema** (produced by promote workflow):

```
## Eval Card
| Field       | Value                                        |
|-------------|----------------------------------------------|
| W&B Train   | link to training run                         |
| W&B Eval    | link to evaluation run                       |
| Date        | UTC timestamp                                |
| Git SHA     | commit that produced the training run        |

### Training Metrics
| Metric               | Value  |
|----------------------|--------|
| (from training run's wandb.summary, excluding _ prefixed)  |

### Eval Metrics
| Metric               | Value  |
|----------------------|--------|
| (from eval run's wandb.summary, excluding _ prefixed)      |

### Training Config
(full training config as JSON)

### Dataset
| Type    | Artifact (version) |
|---------|--------------------|
| (each input artifact from training run's used_artifacts())  |

### Eval Dataset
| Type    | Artifact (version) |
|---------|--------------------|
| (each input artifact from eval run's used_artifacts())      |
```

- Tag format: `model-v{N}` (incrementing integer)
- Asset: model file (`.pt` / `.onnx`) attached to the release
- Promote also sets `:production` alias on the model artifact in W&B
- See [promotion-pipeline-reference.md](../reference/promotion-pipeline-reference.md) for implementation.

______________________________________________________________________

## 9. Secrets

| Secret                               | Used By                             | Source                  |
| ------------------------------------ | ----------------------------------- | ----------------------- |
| `WANDB_API_KEY`                      | data-gen, training, eval, promotion | wandb.ai/settings       |
| `GITHUB_TOKEN`                       | promotion                           | Automatic in GHA        |
| `RUNPOD_API_KEY`                     | data-gen, training                  | runpod.io               |
| `RCLONE_CONFIG_R2_ACCESS_KEY_ID`     | data-gen, training, eval            | Cloudflare R2 dashboard |
| `RCLONE_CONFIG_R2_SECRET_ACCESS_KEY` | data-gen, training, eval            | Cloudflare R2 dashboard |
| `RCLONE_CONFIG_R2_ENDPOINT`          | data-gen, training, eval            | Cloudflare R2 dashboard |

- Secrets must only be available to workflows that require them.
- Training needs R2 read (download dataset shards) and write (upload checkpoints).

______________________________________________________________________

## 10. W&B Identity

| Field   | Value          |
| ------- | -------------- |
| Entity  | `tinaudio`     |
| Project | `synth-setter` |

- Set via env vars: `WANDB_ENTITY`, `WANDB_PROJECT`
- Configs use: `entity: ${oc.env:WANDB_ENTITY,tinaudio}`, `project: ${oc.env:WANDB_PROJECT,synth-setter}`
- Legacy runs under `benhayes`/`synth-permutations` remain read-only. New runs must use `tinaudio`/`synth-setter`.

______________________________________________________________________

## 11. Artifact → Storage Mapping

- W&B artifacts reference R2 objects via `artifact.add_reference("s3://intermediate-data/...")` (R2 is S3-compatible)
- Requires `AWS_ENDPOINT_URL` (or `WANDB_S3_ENDPOINT_URL`) set to the R2 endpoint in any environment that calls `add_reference` or downloads reference artifacts. Without this, W&B will attempt to resolve against AWS S3.
- Artifacts do not duplicate large data files — they contain metadata, manifests, and statistics
- Bulk data lives in R2; W&B provides the index and lineage graph

______________________________________________________________________

## 12. Invariants

1. `*_wandb_run_id` uniquely identifies an immutable run output.
2. Every run must log and consume W&B artifacts for its inputs and outputs.
3. Every run must record `github_sha` in `wandb.config`.
4. All configs are frozen into the artifact or R2 storage path at run time.
5. R2 paths are append-only after completion markers exist.
6. Runs must not consume data from R2 paths that lack their completion marker.

______________________________________________________________________

## 13. References

- [promotion-pipeline-reference.md](../reference/promotion-pipeline-reference.md) — W&B → GitHub Release workflow, promote script, GHA workflow
- artifact-provenance-reference.md — TBD (#122): W&B artifact patterns, lineage DAG examples, API reference
