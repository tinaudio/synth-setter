# Configuration Reference

> **Code version**: `f97fc7e` (2026-03-29, `main`)
> **Tracking**: #383, #107

______________________________________________________________________

## 1. Configuration Layers

| Layer                 | Tool                                           | Validation                                                   | Stored In                          | Example                                      |
| --------------------- | ---------------------------------------------- | ------------------------------------------------------------ | ---------------------------------- | -------------------------------------------- |
| Experiment config     | Hydra YAML composition                         | Deferred — class constructors at `hydra.utils.instantiate()` | git (`configs/experiment/`)        | `configs/experiment/surge/flow_simple.yaml`  |
| Pipeline input config | Pydantic `BaseModel(strict=True)`              | Parse-time — `load_dataset_config()`                         | git (`configs/dataset/`)           | `configs/dataset/surge-simple-480k-10k.yaml` |
| Frozen runtime spec   | Pydantic `BaseModel(strict=True, frozen=True)` | Materialization — `materialize_spec()`                       | R2 (`{r2_prefix}/input_spec.json`) | `DatasetPipelineSpec`                        |
| Cloud infrastructure  | SkyPilot Task YAML                             | Launcher script (not Hydra)                                  | git (`configs/compute/`)           | `configs/compute/runpod-template.yaml`       |
| Secrets / credentials | Environment variables                          | Runtime                                                      | `.env` (local), CI secrets         | `WANDB_API_KEY`                              |

### Why These Boundaries

- **Pydantic strict** at trust boundaries — where data enters from external sources (user config YAML, JSON from R2, worker reports). Catches type errors, missing fields, and invalid values at parse time.
- **Hydra DictConfig** for training — composable experiment configs validated by class constructors at instantiation. Hydra handles defaults, overrides, and interpolation natively.
- **Plain YAML for cloud infrastructure** — consumed by a launcher script that calls provider APIs before the training job starts. Different program, different time, no Hydra composition needed.
- **No training input spec** — training is a single long-running job with no distributed coordination. The data pipeline's spec exists for reconciliation across hundreds of parallel workers; training has no equivalent need. Provenance is captured by W&B run metadata + frozen `config.yaml` in R2.

Reference: `data-pipeline.md` §14.4, `training-pipeline.md` §7.1

______________________________________________________________________

## 2. Config Architecture Per Stage

### 2.1 Data Generation

```
Config YAML → load_dataset_config() → DatasetConfig (Pydantic, validated)
  → materialize_spec() → DatasetPipelineSpec (frozen, immutable)
    → uploaded to R2 as {r2_prefix}/input_spec.json
```

- Config is mutable, human-authored YAML in `configs/dataset/`
- Spec is immutable, machine-generated JSON capturing runtime state (git SHA, renderer version, per-shard seeds)
- Spec is the reproducibility unit and reconciliation target
- **Config drift protection (planned):** the design doc specifies that re-passing `--config` for a `run_id` that already has a spec should error — but this is not yet enforced. The current implementation always generates a new `run_id` and writes a fresh spec. Tracked in [#386](https://github.com/tinaudio/synth-setter/issues/386).
- **Path note:** `storage-provenance-spec.md` §3a documents the target path as `metadata/input_spec.json`, but the current implementation uploads to `{r2_prefix}/input_spec.json` (no `metadata/` subdirectory). Tracked in [#385](https://github.com/tinaudio/synth-setter/issues/385).

Reference: `data-pipeline.md` §14.5

### 2.2 Training

```
train.yaml + defaults (experiment, data, model, trainer, callbacks, logger)
  → Hydra composes DictConfig
    → hydra.utils.instantiate() → LightningModule, DataModule, Trainer
      → trainer.fit(model, datamodule)
```

- No intermediate spec — Hydra instantiates directly to Python objects
- Provenance: W&B config (hyperparams, `github_sha`) + frozen `config.yaml` in R2
- Resume: Lightning native `ckpt_path=` with W&B artifact download
- Single-job model — no reconciliation, no distributed coordination

Reference: `training-pipeline.md` §4–5

### 2.3 Evaluation

```
eval.yaml + experiment config (pins model + data + checkpoint)
  → Hydra composes DictConfig → predict → render → metrics
```

- Experiment config pins everything: model checkpoint (W&B artifact ref), data config, eval settings
- No eval spec — configs are the source of truth
- Full provenance in R2 path: `eval/{dataset_config_id}/{dataset_wandb_run_id}/{train_config_id}/{train_wandb_run_id}/{eval_config_id}/{eval_wandb_run_id}/`

Reference: `eval-pipeline.md` §4–5

### 2.4 Cloud Infrastructure

```
configs/compute/{provider}-template.yaml (SkyPilot Task YAML)
  → launcher script (pipeline.entrypoints.skypilot_launch_smoke)
    reads YAML, materializes spec, mounts spec into worker
    → SkyPilot provisions pod (RunPod, Vast.ai planned, …)
      → pod runs: python -m pipeline.entrypoints.generate_dataset --spec ...
```

- Separate from Hydra — different consumer (SkyPilot's `Task.from_yaml`), different time (before worker starts)
- Launcher takes the task template + a `DatasetConfig`, materializes a
  `DatasetPipelineSpec`, uploads it to R2 (under `skypilot-launcher-specs/<cluster>.json`),
  and forwards the `r2://` URI to the worker via `task.update_envs(WORKER_SPEC_URI=...)`.
  R2 is used instead of `task.update_file_mounts` because the SkyPilot RunPod backend
  rejects programmatic file_mounts with a pubkey-overflow error (see [#749](https://github.com/tinaudio/synth-setter/issues/749)).
- Invoked via: `python -m pipeline.entrypoints.skypilot_launch_smoke --config <yaml> --template <yaml>`

Reference: `training-pipeline.md` Appendix D

______________________________________________________________________

## 3. Cloud Provider Comparison

| Concern             | RunPod                                   | Vast.ai                                                                      |
| ------------------- | ---------------------------------------- | ---------------------------------------------------------------------------- |
| API model           | Create pod with explicit GPU type        | Search offers with query filters → rent best match                           |
| GPU selection       | `gpu_type_id` from catalog               | Query: `gpu_name=RTX_4090 num_gpus>=1 gpu_ram>=24`                           |
| Pricing model       | Fixed $/hr per GPU type                  | Market-based: on-demand or bid (interruptible)                               |
| Spot / preemptible  | Community cloud (cheaper, less reliable) | `--type=bid` with custom bid price                                           |
| Persistent storage  | Network volumes (`networkVolumeId`)      | Volumes (create new or attach existing by ID)                                |
| Docker image        | `image_name` parameter                   | `image` parameter                                                            |
| Environment vars    | `env` dict (key-value pairs)             | Docker-flag format: `"-e KEY=VALUE"`                                         |
| Startup command     | `docker_args` string                     | `onstart` script (SSH mode) or `args` array (args mode)                      |
| SSH access          | Always available                         | Runtype-dependent (`ssh_direct`, `jupyter_direct`, `args`)                   |
| Port exposure       | Default open                             | Configurable: direct ports or proxy                                          |
| Auto-terminate      | Open question (#107 §11)                 | Supported via `destroy instance` API call                                    |
| GPU filtering       | Choose from fixed catalog                | Rich query language (40+ fields: FLOPS, bandwidth, reliability, geolocation) |
| Reliability scoring | N/A                                      | `reliability` field (0–1 score per machine)                                  |
| Datacenter option   | All datacenter                           | `datacenter=True` filter (vs community hosts)                                |
| Region selection    | Limited datacenter selection             | `geolocation in [US,CA,DE]` filter                                           |
| Benchmark data      | N/A                                      | `dlperf` (DL perf score), `dlperf_per_dphtotal` (perf/$)                     |

### Config Shape

The RunPod template exists today (data-pipeline smoke). Vast.ai template not yet implemented.

**RunPod** (`configs/compute/runpod-template.yaml`) — landed. Abridged
shape (see the file for the full template, including the inline `os._exit(0)`
workaround for [#735](https://github.com/tinaudio/synth-setter/issues/735)):

The launcher injects `image_id` per-launch via `--worker-image-tag` (default `dev-snapshot`) for non-OCI backends, so the template no longer carries the literal:

```yaml
resources:
  cloud: runpod
  accelerators: RTXA4000:1
  use_spot: false
  disk_size: 50

envs:
  RCLONE_CONFIG_R2_TYPE: ""           # the 5 RCLONE_CONFIG_R2_* keys + WANDB_API_KEY
  RCLONE_CONFIG_R2_PROVIDER: ""       # are injected at launch time from
  RCLONE_CONFIG_R2_ACCESS_KEY_ID: ""  # .env or process env (see _WORKER_ENV_KEYS).
  RCLONE_CONFIG_R2_SECRET_ACCESS_KEY: ""
  RCLONE_CONFIG_R2_ENDPOINT: ""
  WANDB_API_KEY: ""
  WORKER_SPEC_URI: ""                 # set by the launcher to r2://<bucket>/skypilot-launcher-specs/<cluster>.json

run: |
  cd /home/build/synth-setter && python -c '
  from pipeline.entrypoints.generate_dataset import load_spec_from_uri, run
  import os
  run(load_spec_from_uri(os.environ["WORKER_SPEC_URI"]))
  os._exit(0)
  '
```

The launcher uploads the materialized spec to R2 (under
`r2://{bucket}/skypilot-launcher-specs/<cluster>.json`) rather than using
`task.update_file_mounts(...)` — see [#749](https://github.com/tinaudio/synth-setter/issues/749).

**Vast.ai** (`configs/compute/vast-template.yaml`) — planned, not implemented:

```yaml
provider: vast
search_query: "gpu_name=RTX_4090 num_gpus>=1 gpu_ram>=24 reliability>=0.99 datacenter=True"
disk: 64
runtype: "args"
image: "tinaudio/synth-setter-train:latest"
```

______________________________________________________________________

## 4. Config Boundary Rules

| Boundary                             | Tool                       | Why                                                                                       |
| ------------------------------------ | -------------------------- | ----------------------------------------------------------------------------------------- |
| External input (config YAML)         | Pydantic `strict=True`     | Untrusted human input — catch type errors, missing fields, invalid values at parse time   |
| Serialization (spec, reports, cards) | Pydantic `strict=True`     | JSON crossing process boundaries (R2 ↔ CLI ↔ workers) — enforce schema on every read      |
| Training experiment config           | Hydra DictConfig           | Composable defaults + overrides; validation deferred to class constructors                |
| HDF5 shard data (NumPy arrays)       | Custom validation function | Pydantic can't validate `ndarray`; custom shape/dtype/value checks required               |
| Internal data transform              | `dataclass(frozen=True)`   | Already validated — typed container prevents field mixups; no runtime validation overhead |
| Cloud infrastructure                 | Plain YAML                 | Different consumer (launcher script), no composition needed, no `_target_` instantiation  |
| Secrets / credentials                | Environment variables      | Never committed to git; loaded from `.env` (local) or CI secrets                          |

Reference: `data-pipeline.md` §14.4

______________________________________________________________________

## 5. Known Gaps

Gaps are configuration inputs that design docs specify or that standard practice recommends, but that don't exist in the codebase yet.

### 5.1 Identity & Provenance

| Input                          | Type   | What's Needed                                                               | Reference                   |
| ------------------------------ | ------ | --------------------------------------------------------------------------- | --------------------------- |
| `train_wandb_run_id`           | string | `{train_config_id}-{YYYYMMDDTHHMMSSZ}` — structured, reconstructible run ID | storage-provenance-spec §1  |
| `dataset_config_id` linkage    | string | Explicit link from training config to consumed dataset                      | storage-provenance-spec §2  |
| `dataset_wandb_run_id` linkage | string | Explicit link to specific dataset run version                               | storage-provenance-spec §2  |
| `job_type`                     | string | Must be `"training"` in W&B config — currently empty                        | storage-provenance-spec §7  |
| `github_sha` in `wandb.config` | string | Logged via `log_wandb_provenance()` but not in Hydra config                 | wandb-integration.md gap #3 |

### 5.2 W&B / Artifact Lineage

| Input                        | Type   | What's Needed                                                    | Reference                   |
| ---------------------------- | ------ | ---------------------------------------------------------------- | --------------------------- |
| `logger.wandb.log_model`     | string | `"all"` — uploads every checkpoint immediately (crash-resilient) | training-pipeline.md §6.2   |
| `logger.wandb.id`            | string | `{train_config_id}-{YYYYMMDDTHHMMSSZ}` instead of null/random    | wandb-integration.md gap #8 |
| `logger.wandb.job_type`      | string | `"training"` instead of empty                                    | storage-provenance-spec §7  |
| `logger.wandb.resume`        | string | `"allow"` for W&B resume support                                 | training-pipeline.md §5.3   |
| Dataset `run.use_artifact()` | code   | Lineage link to consumed dataset artifact                        | storage-provenance-spec §5  |
| Model `run.log_artifact()`   | code   | Lineage link for produced model artifact                         | storage-provenance-spec §5  |

### 5.3 Data Portability

| Input               | Type   | What's Needed                                                                    | Reference                 |
| ------------------- | ------ | -------------------------------------------------------------------------------- | ------------------------- |
| `data.dataset_root` | string | Remove hardcoded `/data/scratch/...` paths; use env-based or `${paths.data_dir}` | training-pipeline.md §1   |
| `data.r2_path`      | string | Optional R2 URI for remote dataset sync before training                          | training-pipeline.md §6.1 |
| `data.stats_file`   | string | Remove hardcoded paths in nsynth/fsd configs                                     | training-pipeline.md §1   |

### 5.4 Hardware & Compute

| Input                             | Type   | What's Needed                                                       | Reference                   |
| --------------------------------- | ------ | ------------------------------------------------------------------- | --------------------------- |
| `trainer.precision`               | string | Not set in most configs — should be explicit (e.g., `"bf16-mixed"`) | Lightning Trainer option    |
| `trainer.accumulate_grad_batches` | int    | Effective batch size scaling                                        | Lightning Trainer option    |
| `trainer.benchmark`               | bool   | cuDNN benchmark mode for performance                                | Lightning Trainer option    |
| `data.pin_memory`                 | bool   | Not set in active configs — should be standard for GPU training     | Lightning DataLoader option |
| `data.persistent_workers`         | bool   | Keep workers alive between epochs — performance optimization        | Lightning DataLoader option |
| `data.prefetch_factor`            | int    | DataLoader prefetch                                                 | Lightning DataLoader option |
| `model.optimizer.fused`           | bool   | CUDA fused Adam — significant speedup on GPU                        | torch.optim.Adam option     |

### 5.5 Cloud Infrastructure

| Input               | Type               | What's Needed                                                                                                                         | Reference                                             |
| ------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| RunPod config       | SkyPilot Task YAML | Landed for the data pipeline smoke at `configs/compute/runpod-template.yaml`; training launcher still uses the legacy RunPod-API path | data-pipeline.md §14, training-pipeline.md Appendix D |
| Vast.ai config      | SkyPilot Task YAML | Planned — `configs/compute/vast-template.yaml` not yet authored                                                                       | new provider                                          |
| `configs/compute/`  | directory          | SkyPilot Task templates for the data pipeline launcher (RunPod landed; Vast.ai planned)                                               | —                                                     |
| `make train`        | Makefile target    | Training shorthand with EXPERIMENT arg                                                                                                | training-pipeline.md §2                               |
| `make docker-train` | Makefile target    | Docker training shorthand                                                                                                             | training-pipeline.md §2                               |
| `make runpod-train` | Makefile target    | RunPod launcher shorthand                                                                                                             | training-pipeline.md §2                               |
| `make resume`       | Makefile target    | Resume from W&B artifact with EXPERIMENT + RUN_ID                                                                                     | training-pipeline.md §2                               |

### 5.6 Other

| Input                        | Type       | What's Needed                                                               | Reference                          |
| ---------------------------- | ---------- | --------------------------------------------------------------------------- | ---------------------------------- |
| Training Docker image        | Dockerfile | Separate from data pipeline image (needs CUDA+torch, not VST+rclone)        | training-pipeline.md §7.4          |
| Training R2 paths            | code       | Dataset R2 path, training R2 path, frozen config upload                     | storage-provenance-spec §3b        |
| `RUNPOD_API_KEY`             | env var    | RunPod launcher needs API access                                            | storage-provenance-spec §9         |
| `AWS_ENDPOINT_URL`           | env var    | Required for W&B to resolve R2 artifact references                          | storage-provenance-spec §11        |
| `torch.compile` backend/mode | string     | Currently just a bool toggle — no backend, mode, fullgraph, dynamic options | Lightning/PyTorch option           |
| `PYTHONHASHSEED`             | env var    | Fixed hash seed for reproducibility                                         | standard ML practice               |
| `CUBLAS_WORKSPACE_CONFIG`    | env var    | `:4096:8` for deterministic cuBLAS                                          | required when `deterministic=True` |

______________________________________________________________________

## 6. Cross-References

This document does not duplicate content from authoritative sources. Consult them directly:

| Topic                                 | Authoritative Source         | What It Covers                                                 |
| ------------------------------------- | ---------------------------- | -------------------------------------------------------------- |
| IDs, R2 paths, W&B artifacts, secrets | `storage-provenance-spec.md` | Naming conventions, bucket layout, artifact types, lineage DAG |
| Current W&B integration state         | `wandb-integration.md`       | What gets logged, initialization, 8 known gaps                 |
| Training architecture                 | `training-pipeline.md`       | Phase plan, design decisions, stage definitions                |
| Data pipeline config → spec flow      | `data-pipeline.md` §14       | Schema design, materialization, validation boundaries          |
| Eval pipeline architecture            | `eval-pipeline.md`           | Three-stage pipeline, R2 integration, config pinning           |
| Docker image contract                 | `docker-spec.md`             | Entrypoint modes, env vars, image targets                      |
