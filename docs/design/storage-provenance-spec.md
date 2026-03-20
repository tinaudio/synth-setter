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

| ID                  | Construction                             | Source                             | Example                        |
| ------------------- | ---------------------------------------- | ---------------------------------- | ------------------------------ |
| `dataset_config_id` | Config filename (stem)                   | `configs/dataset/{id}.yaml`        | `diva-v1`                      |
| `dataset_run_id`    | `{dataset_config_id}-{YYYYMMDDTHHMMSSZ}` | `wandb.init(id=...)`               | `diva-v1-20260312T143022Z`     |
| `train_config_id`   | Config filename (stem)                   | `configs/experiment/.../{id}.yaml` | `flow-simple`                  |
| `train_run_id`      | `{train_config_id}-{YYYYMMDDTHHMMSSZ}`   | `wandb.init(id=...)`               | `flow-simple-20260315T091500Z` |
| `eval_config_id`    | Eval dataset config filename (stem)      | `configs/dataset/{id}.yaml`        | `nsynth-v1`                    |
| `eval_run_id`       | `{eval_config_id}-{YYYYMMDDTHHMMSSZ}`    | `wandb.init(id=...)`               | `nsynth-v1-20260320T160000Z`   |

- `*_config_id` = filename of the YAML config, without extension
- `*_run_id` = `{*_config_id}-{timestamp}`, set as the W&B run ID via `wandb.init(id=...)`
- Timestamp format: `YYYYMMDDTHHMMSSZ` (seconds, UTC, filesystem-safe)
- W&B run ID limit: 64 characters. Keep config filenames short.
- Config filenames must be globally unique across all config directories.

______________________________________________________________________

## 2. R2 Bucket Layout

```
synth-data/
├── data/{dataset_config_id}/{dataset_run_id}/
├── train/{dataset_config_id}/{train_config_id}/{train_run_id}/
└── eval/{dataset_config_id}/{train_config_id}/{train_run_id}/{eval_config_id}/{eval_run_id}/
```

______________________________________________________________________

## 3. R2 Contents Per Workflow

### 3a. Data Generation

```
data/{dataset_config_id}/{dataset_run_id}/
├── shards/
│   ├── shard-000000.h5
│   └── ...
├── metadata/
│   ├── config.yaml          # Frozen pipeline config (provenance copy)
│   ├── input_spec.json      # Frozen input specification (authoritative)
│   ├── dataset.json         # Self-describing dataset card
│   ├── dataset.complete     # Completion marker
│   └── workers/             # Worker staging area
│       ├── shards/{shard_id}/{worker_id}-{attempt_uuid}.*
│       └── attempts/{worker_id}-{attempt_uuid}/report.json
├── train.h5, val.h5, test.h5  # Split virtual datasets
└── stats.npz                   # Normalization statistics
```

- Workers may only write under `metadata/workers/`
- `shards/` is written only by finalize
- All `rclone` operations use `--checksum`
- Datasets are immutable once `dataset.complete` exists. New versions require a new `dataset_run_id`.

### 3b. Training

```
train/{dataset_config_id}/{train_config_id}/{train_run_id}/
├── checkpoints/
│   ├── last.ckpt
│   └── best.ckpt
└── config.yaml               # Frozen experiment config
```

### 3c. Evaluation

```
eval/{dataset_config_id}/{train_config_id}/{train_run_id}/{eval_config_id}/{eval_run_id}/
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

| Type           | Name Pattern          | Logged By               | Example name  |
| -------------- | --------------------- | ----------------------- | ------------- |
| `dataset`      | `{dataset_config_id}` | `pipeline.cli finalize` | `diva-v1`     |
| `model`        | `{train_config_id}`   | `src/train.py`          | `flow-simple` |
| `eval-results` | `{eval_config_id}`    | eval script             | `nsynth-v1`   |

- W&B auto-versions artifacts (`:v0`, `:v1`, `:v2`). Each new run of the same config produces the next version.
- Use aliases for human-readable pointers: `:latest`, `:best`, `:production`
- The `run_id` is stored in `artifact.metadata`, not the artifact name

______________________________________________________________________

## 5. W&B Lineage DAG

```
dataset config
  → [data-generation run] → dataset artifact
    → [training run] → model artifact
      → [evaluation run] → eval-results artifact
        → [promote workflow] → GitHub Release
```

- Every run must call `run.use_artifact()` for inputs and `run.log_artifact()` for outputs
- `run.use_artifact()` (not `api.artifact()`) — only the former creates lineage links
- Every run must include `github_sha` in `wandb.config`

______________________________________________________________________

## 6. W&B Metadata Convention

| Location            | What Goes Here                                   | Examples                                                |
| ------------------- | ------------------------------------------------ | ------------------------------------------------------- |
| `wandb.config`      | Hyperparams — things you SET before the run      | `lr`, `epochs`, `batch_size`, `github_sha`              |
| `wandb.summary`     | Final metrics — things you MEASURE after the run | `mse`, `spectral_convergence`, `param_accuracy`         |
| `artifact.metadata` | Properties of the artifact itself                | `n_samples`, `mel_shape`, `shard_count`, `code_version` |

- Dataset properties belong on the dataset artifact, not on training runs that consume it

______________________________________________________________________

## 7. `job_type` Values

| `job_type`        | Stage         | Script                  |
| ----------------- | ------------- | ----------------------- |
| `data-generation` | Data pipeline | `pipeline.cli finalize` |
| `training`        | Training      | `src/train.py`          |
| `evaluation`      | Evaluation    | eval script             |

- Set on every `wandb.init(job_type=...)` call

______________________________________________________________________

## 8. GitHub Actions Workflows

| Workflow        | File                 | Trigger              | Runner                          | Secrets             | Key Inputs                      |
| --------------- | -------------------- | -------------------- | ------------------------------- | ------------------- | ------------------------------- |
| Tests           | `test.yml`           | push, PR             | `ubuntu-latest`, `macos-latest` | —                   | —                               |
| Full Tests      | `test-expensive.yml` | push(main), dispatch | `gpu-x64`                       | —                   | —                               |
| Data Generation | TBD                  | `workflow_dispatch`  | TBD                             | R2, W&B, RunPod     | config, n_workers               |
| Training        | TBD                  | `workflow_dispatch`  | TBD                             | R2, W&B, RunPod     | experiment, overrides           |
| Evaluation      | TBD                  | `workflow_dispatch`  | TBD                             | R2, W&B             | experiment                      |
| Model Promotion | `promote.yml`        | `workflow_dispatch`  | `ubuntu-latest`                 | W&B, `GITHUB_TOKEN` | `run_id`, `registry`, `dry_run` |

- All workflows that create W&B runs must export `GITHUB_SHA` into the run environment.

______________________________________________________________________

## 9. Secrets

| Secret                               | Used By                             | Source                  |
| ------------------------------------ | ----------------------------------- | ----------------------- |
| `WANDB_API_KEY`                      | data-gen, training, eval, promotion | wandb.ai/settings       |
| `GITHUB_TOKEN`                       | promotion                           | Automatic in GHA        |
| `RUNPOD_API_KEY`                     | data-gen, training                  | runpod.io               |
| `RCLONE_CONFIG_R2_ACCESS_KEY_ID`     | data-gen, eval                      | Cloudflare R2 dashboard |
| `RCLONE_CONFIG_R2_SECRET_ACCESS_KEY` | data-gen, eval                      | Cloudflare R2 dashboard |
| `RCLONE_CONFIG_R2_ENDPOINT`          | data-gen, eval                      | Cloudflare R2 dashboard |

- Secrets must only be available to workflows that require them (e.g., training does not need R2 write access).

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

- W&B artifacts reference R2 objects via `artifact.add_reference("r2://...")`
- Artifacts do not duplicate large data files — they contain metadata, manifests, and statistics
- Bulk data lives in R2; W&B provides the index and lineage graph

______________________________________________________________________

## 12. Invariants

1. `*_run_id` uniquely identifies an immutable run output.
2. Every run must log and consume W&B artifacts for its inputs and outputs.
3. Every run must record `github_sha` in `wandb.config`.
4. All configs are frozen into the artifact or R2 storage path at run time.
5. R2 paths are append-only after completion markers exist.

______________________________________________________________________

## 13. References

- [promotion-pipeline-reference.md](../reference/promotion-pipeline-reference.md) — W&B → GitHub Release workflow, promote script, GHA workflow
- artifact-provenance-reference.md — TBD (#122): W&B artifact patterns, lineage DAG examples, API reference
