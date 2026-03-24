# Storage & Provenance Spec

> Authoritative source of truth for R2 paths, W&B artifacts, and GitHub Actions workflows.
> Individual design docs must not define their own storage or provenance conventions — point here instead.

| Field        | Value      |
| ------------ | ---------- |
| Status       | Draft      |
| Last Updated | 2026-03-24 |
| Tracking     | #122       |

______________________________________________________________________

## 1. IDs

| ID                     | Construction                                                   | Source                             | Example                        |
| ---------------------- | -------------------------------------------------------------- | ---------------------------------- | ------------------------------ |
| `image_config_id`      | Config filename (stem)                                         | `configs/image/{id}.yaml`          | `gpu-torch24`                  |
| `image_tag`            | `{image_config_id}-{git_sha_short}`                            | Docker build                       | `gpu-torch24-a1b2c3d`          |
| `dataset_config_id`    | Config filename (stem)                                         | `configs/dataset/{id}.yaml`        | `diva-v1`                      |
| `dataset_wandb_run_id` | Configurable, default `{dataset_config_id}-{YYYYMMDDTHHMMSSZ}` | `wandb.init(id=...)`               | `diva-v1-20260312T143022Z`     |
| `train_config_id`      | Config filename (stem)                                         | `configs/experiment/.../{id}.yaml` | `flow-simple`                  |
| `train_wandb_run_id`   | Configurable, default `{train_config_id}-{YYYYMMDDTHHMMSSZ}`   | `wandb.init(id=...)`               | `flow-simple-20260315T091500Z` |
| `eval_config_id`       | Eval dataset config filename (stem)                            | `configs/dataset/{id}.yaml`        | `nsynth-v1`                    |
| `eval_wandb_run_id`    | Configurable, default `{eval_config_id}-{YYYYMMDDTHHMMSSZ}`    | `wandb.init(id=...)`               | `nsynth-v1-20260320T160000Z`   |

- `image_config_id` = filename of the image config YAML, without extension
- `image_tag` = Docker image tag. The short SHA (7 chars) ties the image to the exact commit that built it.
- `*_config_id` = filename of the YAML config, without extension
- `*_wandb_run_id` = the W&B run ID, set via `wandb.init(id=...)`. Default convention is `{*_config_id}-{timestamp}`, but the path format is agnostic to how the ID is generated.
- Default timestamp format: `YYYYMMDDTHHMMSSZ` (seconds, UTC, filesystem-safe)
- W&B run ID limit: 64 characters. Keep config filenames short.

______________________________________________________________________

## 2. Image Config

### 2a. Config File

Each image variant is defined by a YAML file in `configs/image/`. The config captures everything needed to reproduce the image that isn't already in the Dockerfile.

```yaml
# configs/image/gpu-torch24.yaml
base_image: nvidia/cuda:12.4.1-devel-ubuntu22.04
python_version: "3.11"
torch_version: "2.4.0"
cuda_version: "12.4"
extra_pip:
  - lightning>=2.3
  - wandb
  - hydra-core
  - rclone  # if not installed via apt
notes: "Primary training image for Surge XT flow matching"
```

The Dockerfile reads from this config (or the values are passed as build args). The config is committed to git, so the `github_sha` in the image tag provides full traceability.

### 2b. Build Convention

```bash
docker build \
  --build-arg IMAGE_CONFIG=gpu-torch24 \
  --build-arg GIT_SHA=$(git rev-parse --short HEAD) \
  -t tinaudio/perm:gpu-torch24-a1b2c3d \
  .
```

The Dockerfile should bake `IMAGE_TAG` into the image as an environment variable so running containers can self-report:

```dockerfile
ARG IMAGE_CONFIG
ARG GIT_SHA
ENV IMAGE_TAG=${IMAGE_CONFIG}-${GIT_SHA}
```

### 2c. Registry

| Field    | Value                                  |
| -------- | -------------------------------------- |
| Registry | Docker Hub                             |
| Repo     | `tinaudio/perm`                        |
| Tag      | `{image_config_id}-{git_sha_short}`    |
| Dev tag  | `dev-snapshot` (mutable, latest build) |

- Production images use the full `{image_config_id}-{git_sha_short}` tag and are immutable.
- `dev-snapshot` is a mutable convenience tag for iteration. Do not use for reproducible runs.

______________________________________________________________________

## 3. R2 Bucket Layout

```
synth-data/
├── data/{dataset_config_id}/{dataset_wandb_run_id}/
├── train/{dataset_config_id}/{dataset_wandb_run_id}/{train_config_id}/{train_wandb_run_id}/
└── eval/{dataset_config_id}/{dataset_wandb_run_id}/{train_config_id}/{train_wandb_run_id}/{eval_config_id}/{eval_wandb_run_id}/
```

______________________________________________________________________

## 4. R2 Contents Per Workflow

### 4a. Data Generation

```
data/{dataset_config_id}/{dataset_wandb_run_id}/
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
- Datasets are immutable once `dataset.complete` exists. New versions require a new `dataset_wandb_run_id`.

### 4b. Training

```
train/{dataset_config_id}/{dataset_wandb_run_id}/{train_config_id}/{train_wandb_run_id}/
├── checkpoints/
│   ├── last.ckpt
│   └── best.ckpt
└── config.yaml               # Frozen experiment config
```

### 4c. Evaluation

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

## 5. W&B Artifact Types

| Type           | Name Pattern               | Logged By               | Example name        |
| -------------- | -------------------------- | ----------------------- | ------------------- |
| `dataset`      | `data-{dataset_config_id}` | `pipeline.cli finalize` | `data-diva-v1`      |
| `model`        | `model-{train_config_id}`  | `src/train.py`          | `model-flow-simple` |
| `eval-results` | `eval-{eval_config_id}`    | eval script             | `eval-nsynth-v1`    |

- W&B auto-versions artifacts (`:v0`, `:v1`, `:v2`). Each new run of the same config produces the next version.
- The `*_wandb_run_id` is stored in `artifact.metadata`, not the artifact name

**Alias strategy:**

| Alias         | Set by           | When                                                                 |
| ------------- | ---------------- | -------------------------------------------------------------------- |
| `:latest`     | W&B (automatic)  | Every `log_artifact` call                                            |
| `:best`       | Training script  | When val metric improves (`run.log_artifact(art, aliases=["best"])`) |
| `:production` | Promote workflow | When model is promoted to GitHub Release                             |

______________________________________________________________________

## 6. W&B Lineage DAG

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
- Every run must include `github_sha` and `image_tag` in `wandb.config`

______________________________________________________________________

## 7. W&B Metadata Convention

| Location            | What Goes Here                                   | Examples                                                           |
| ------------------- | ------------------------------------------------ | ------------------------------------------------------------------ |
| `wandb.config`      | Hyperparams — things you SET before the run      | `lr`, `epochs`, `batch_size`, `github_sha`, `image_tag`, `command` |
| `wandb.summary`     | Final metrics — things you MEASURE after the run | `mse`, `spectral_convergence`, `param_accuracy`                    |
| `artifact.metadata` | Properties of the artifact itself                | `n_samples`, `mel_shape`, `shard_count`, `code_version`            |

- Dataset properties belong on the dataset artifact, not on training runs that consume it

**Required `wandb.config` fields for all runs:**

| Field        | Source                               | Example                                 |
| ------------ | ------------------------------------ | --------------------------------------- |
| `github_sha` | Git HEAD inside container            | `a1b2c3d4e5f6...`                       |
| `image_tag`  | `IMAGE_TAG` env var baked into image | `gpu-torch24-a1b2c3d`                   |
| `command`    | `" ".join(sys.argv)`                 | `train.py experiment=surge/flow_simple` |

```python
import subprocess, sys, os, wandb

sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
wandb.config.update({
    "github_sha": sha,
    "image_tag": os.environ.get("IMAGE_TAG", "unknown"),
    "command": " ".join(sys.argv),
})
```

______________________________________________________________________

## 8. `job_type` Values

| `job_type`        | Stage         | Script                  |
| ----------------- | ------------- | ----------------------- |
| `data-generation` | Data pipeline | `pipeline.cli finalize` |
| `training`        | Training      | `src/train.py`          |
| `evaluation`      | Evaluation    | eval script             |

- Set on every `wandb.init(job_type=...)` call

______________________________________________________________________

## 9. GitHub Actions Workflows

| Workflow        | File                 | Trigger                                        | Runner                          | Secrets             | Key Inputs                                                       |
| --------------- | -------------------- | ---------------------------------------------- | ------------------------------- | ------------------- | ---------------------------------------------------------------- |
| Tests           | `test.yml`           | push, PR                                       | `ubuntu-latest`, `macos-latest` | —                   | —                                                                |
| Full Tests      | `test-expensive.yml` | push(main), dispatch                           | `gpu-x64`                       | —                   | —                                                                |
| Image Build     | `build-image.yml`    | push (Dockerfile/configs/image/\*\*), dispatch | `ubuntu-latest`                 | Docker Hub PAT      | `image_config_id`                                                |
| Data Generation | TBD                  | `workflow_dispatch`                            | TBD                             | R2, W&B, RunPod     | config, n_workers                                                |
| Training        | TBD                  | `workflow_dispatch`                            | TBD                             | R2, W&B, RunPod     | experiment, overrides                                            |
| Evaluation      | TBD                  | `workflow_dispatch`                            | TBD                             | R2, W&B             | `train_wandb_run_id`, `eval_config_id`                           |
| Model Promotion | `promote.yml`        | `workflow_dispatch`                            | `ubuntu-latest`                 | W&B, `GITHUB_TOKEN` | `train_wandb_run_id`, `eval_wandb_run_id`, `registry`, `dry_run` |

- All workflows that create W&B runs must export `GITHUB_SHA` and `IMAGE_TAG` into the run environment.
- Image Build workflow tags with `{image_config_id}-{sha_short}` and pushes to Docker Hub. Also updates `dev-snapshot` on non-release builds.
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
| Image       | Docker image tag used for training           |

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

## 10. Secrets

| Secret                               | Used By                             | Source                  |
| ------------------------------------ | ----------------------------------- | ----------------------- |
| `WANDB_API_KEY`                      | data-gen, training, eval, promotion | wandb.ai/settings       |
| `GITHUB_TOKEN`                       | promotion                           | Automatic in GHA        |
| `RUNPOD_API_KEY`                     | data-gen, training                  | runpod.io               |
| `DOCKERHUB_USERNAME`                 | image build                         | hub.docker.com          |
| `DOCKERHUB_TOKEN`                    | image build                         | hub.docker.com          |
| `RCLONE_CONFIG_R2_ACCESS_KEY_ID`     | data-gen, training, eval            | Cloudflare R2 dashboard |
| `RCLONE_CONFIG_R2_SECRET_ACCESS_KEY` | data-gen, training, eval            | Cloudflare R2 dashboard |
| `RCLONE_CONFIG_R2_ENDPOINT`          | data-gen, training, eval            | Cloudflare R2 dashboard |

- Secrets must only be available to workflows that require them.
- Training needs R2 read (download dataset shards) and write (upload checkpoints).

______________________________________________________________________

## 11. W&B Identity

| Field   | Value          |
| ------- | -------------- |
| Entity  | `tinaudio`     |
| Project | `synth-setter` |

- Set via env vars: `WANDB_ENTITY`, `WANDB_PROJECT`
- Configs use: `entity: ${oc.env:WANDB_ENTITY,tinaudio}`, `project: ${oc.env:WANDB_PROJECT,synth-setter}`
- Legacy runs under `benhayes`/`synth-permutations` remain read-only. New runs must use `tinaudio`/`synth-setter`.

______________________________________________________________________

## 12. Artifact → Storage Mapping

- W&B artifacts reference R2 objects via `artifact.add_reference("s3://synth-data/...")` (R2 is S3-compatible)
- Requires `AWS_ENDPOINT_URL` (or `WANDB_S3_ENDPOINT_URL`) set to the R2 endpoint in any environment that calls `add_reference` or downloads reference artifacts. Without this, W&B will attempt to resolve against AWS S3.
- Artifacts do not duplicate large data files — they contain metadata, manifests, and statistics
- Bulk data lives in R2; W&B provides the index and lineage graph

______________________________________________________________________

## 13. Invariants

1. `*_wandb_run_id` uniquely identifies an immutable run output.
2. Every run must log and consume W&B artifacts for its inputs and outputs.
3. Every run must record `github_sha` in `wandb.config`.
4. Every run must record `image_tag` in `wandb.config`.
5. All configs are frozen into the artifact or R2 storage path at run time.
6. R2 paths are append-only after completion markers exist.
7. Runs must not consume data from R2 paths that lack their completion marker.
8. Production images are immutable. A new build from the same config at a different commit produces a new tag.

______________________________________________________________________

## 14. References

- [promotion-pipeline-reference.md](../reference/promotion-pipeline-reference.md) — W&B → GitHub Release workflow, promote script, GHA workflow
- artifact-provenance-reference.md — TBD (#122): W&B artifact patterns, lineage DAG examples, API reference
