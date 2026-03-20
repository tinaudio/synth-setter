# Design Doc: Training Pipeline & Durable Checkpointing

> **Status**: Draft
> **Author**: ktinubu@
> **Last Updated**: 2026-03-20
> **Tracking**: #107
> **Storage conventions**: [storage-provenance-spec.md](storage-provenance-spec.md)
> **Issue tracking**: [github-taxonomy.md](github-taxonomy.md)

______________________________________________________________________

### Index

| §   | Section                                                                       | What it covers                                                                 |
| --- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| 1   | [Context & Motivation](#1-context--motivation)                                | Problem statement, current state, why training needs different ops             |
| 2   | [Typical Workflow](#2-typical-workflow)                                       | End-to-end local, Docker, and RunPod examples                                  |
| 3   | [Goals, Non-Goals & Design Principles](#3-goals-non-goals--design-principles) | Requirements, principles, anti-goals, success metrics                          |
| 4   | [System Overview](#4-system-overview)                                         | Single-job architecture, checkpoint lifecycle, environment matrix              |
| 5   | [Stage Definitions](#5-stage-definitions)                                     | Train, checkpoint persistence, resume, promotion handoff                       |
| 6   | [R2 & W&B Integration](#6-r2--wb-integration)                                 | Dataset access, checkpoint upload, resume from R2, lineage, release handoff    |
| 7   | [Design Decisions](#7-design-decisions)                                       | Single-job model, durable checkpoints, launcher shape, Docker split, W&B roles |
| 8   | [Phase Plan](#8-phase-plan)                                                   | Epic → Phase → Task hierarchy, issue mapping, file lists, test strategy        |
| 9   | [Dependency Overview](#9-dependency-overview)                                 | Issue dependencies, parallel execution windows, critical path                  |
| 10  | [Alternatives Considered](#10-alternatives-considered)                        | Rejected approaches and why                                                    |
| 11  | [Open Questions & Risks](#11-open-questions--risks)                           | Known gaps and trade-offs                                                      |
| 12  | [Out of Scope](#12-out-of-scope)                                              | Future work — not referenced elsewhere                                         |
| A–D | [Appendices](#appendix-a-glossary)                                            | Glossary, file inventory, checkpoint policy, implementation recipes            |

______________________________________________________________________

## 1. Context & Motivation

Topline goal: Run training portably on local machines, Docker, and RunPod, with durable checkpoints in cloud storage and enough provenance to resume, compare, and promote models safely.

**synth-setter** already has a strong training entry point: Hydra config composition, PyTorch Lightning, W&B logging, CSV logging, and `ModelCheckpoint` saving every 5000 steps plus best and last. What it lacks is **portable durability**. Today, if a long-running cloud pod dies, local checkpoints on that pod die with it.

Training is operationally different from the data pipeline:

| Concern          | Data pipeline                             | Training                                   |
| ---------------- | ----------------------------------------- | ------------------------------------------ |
| Job shape        | Many short parallel workers               | One long-running job                       |
| Duration         | Minutes per worker                        | Hours to days                              |
| Coordination     | Reconciliation (desired vs actual shards) | None — single job                          |
| Failure recovery | Re-run missing shards                     | Resume from latest durable checkpoint      |
| Monitoring       | Storage-based completion                  | W&B metrics + pod liveness                 |
| Output           | Dataset shards → R2                       | Checkpoints → W&B and/or R2, metrics → W&B |

**Key insight:** the data pipeline's reconciliation backend does not apply to training. Training on RunPod is just "launch one pod with `python src/train.py ...`". No shard partitioning, no worker graph, no storage-based coordination loop.

### Current Strengths

| Already works today                          | Notes                                              |
| -------------------------------------------- | -------------------------------------------------- |
| `python src/train.py` with Hydra composition | Mature entry point                                 |
| W&B logger                                   | Tracks training metrics and can upload checkpoints |
| `ModelCheckpoint`                            | Saves every 5000 steps + best + last               |
| CSV logger                                   | Local fallback                                     |
| Lightning resume                             | `ckpt_path=` already supported                     |
| `rootutils` / `PROJECT_ROOT`                 | Paths already resolve cleanly                      |

### Current Gaps

| Gap                                     | Impact                               |
| --------------------------------------- | ------------------------------------ |
| Hardcoded dataset paths in configs      | Fixed by shared config cleanup work  |
| No durable cloud checkpoint persistence | Pod death loses progress             |
| No RunPod training launcher             | Manual cloud startup                 |
| No training-focused Docker image        | Hard to reproduce cloud/local parity |
| Hardcoded W&B identity                  | Wrong defaults for new ownership     |

______________________________________________________________________

## 2. Typical Workflow

### Local development (target state)

```yaml
# configs/experiment/surge/flow_simple.yaml (proposed)
defaults:
  - override /data: surge_simple
  - override /model: surge_flow
  - override /callbacks: default

experiment_name: flow_simple
trainer:
  max_epochs: 100

callbacks:
  model_checkpoint:
    every_n_train_steps: 5000
    save_top_k: 1
    save_last: true
```

```bash
# 1. Set up credentials (one-time)
cp .env.example .env
# Edit .env: WANDB_API_KEY, R2 credentials
# Secrets are documented in storage-provenance-spec.md §9

# 2. Train locally
make train EXPERIMENT=surge/flow_simple

# 3. Resume from a local checkpoint
python src/train.py experiment=surge/flow_simple ckpt_path=logs/train/.../checkpoints/last.ckpt
```

### RunPod training (target state)

```bash
# Launch a single long-running training pod
make runpod-train EXPERIMENT=surge/flow_simple

# If the pod dies after uploading last.ckpt to durable storage:
# Use make resume to construct the full R2 path from the run ID:
make resume EXPERIMENT=surge/flow_simple RUN_ID=<train_wandb_run_id>

# Or specify the full path directly:
python src/train.py \
  experiment=surge/flow_simple \
  ckpt_path=r2:synth-data/train/{dataset_config_id}/{dataset_wandb_run_id}/{train_config_id}/{train_wandb_run_id}/checkpoints/last.ckpt
```

### Docker

```bash
# Train inside a reproducible image
make docker-train EXPERIMENT=surge/flow_simple
```

### Promotion handoff

```bash
# Training produces a model artifact in W&B.
# Promotion stays a separate workflow and follows promotion-pipeline-reference.md.
gh workflow run promote.yml -f run_id=<train_wandb_run_id>
```

______________________________________________________________________

## 3. Goals, Non-Goals & Design Principles

### Goals

- **Run anywhere.** Training must work on local Linux/macOS, Docker, and RunPod.
- **Durable progress.** Checkpoints must survive pod death.
- **Fast recovery.** Resume from the last durable checkpoint with full optimizer / scheduler state.
- **Clear provenance.** Dataset version, train config, git SHA, and checkpoint lineage must be inspectable.
- **Minimal orchestration.** One job, one launcher, one resume path.

### Design Principles

- **Single-job mental model.** Training is not a distributed workflow engine.
- **Checkpoint durability is the recovery mechanism.** No reconciliation layer.
- **Reuse Lightning semantics.** Resume behavior should stay native to Lightning.
- **Storage conventions are shared.** Training uses the same storage / provenance rules as data and eval.
- **W&B for metrics and lineage, R2 for bulk durability.**

### What This System Deliberately Avoids

- Training-specific orchestration framework
- Queueing / polling backend for long-running jobs
- Storage-based coordination like the data pipeline
- Multi-node design before it is needed
- Automatic promotion after training

### Success Metrics

| Metric            | Target                                                       | How to Measure                                          |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------- |
| Resume durability | Pod death loses at most one checkpoint interval              | Crash at step N, resume from latest uploaded checkpoint |
| Portability       | Same experiment runs locally, in Docker, and on RunPod       | Smoke tests + manual parity run                         |
| Provenance        | Every run records dataset, config, SHA, and artifact lineage | Inspect W&B run + storage path                          |
| Local smoke test  | Tiny fixture reaches checkpoint and exits cleanly            | CI                                                      |
| Crash recovery UX | One documented command to resume                             | Runbook                                                 |

### Non-Goals

- Multi-node distributed training
- Hyperparameter sweep orchestration
- Automated model promotion
- Replacing Lightning checkpoint logic
- Replacing W&B with a custom registry

______________________________________________________________________

## 4. System Overview

Training is a **single long-running job** with two durable output channels:

1. **Metrics and lineage → W&B**
2. **Checkpoint files → local disk and optionally R2**

```
                  ┌──────────────────────────────────────────────┐
                  │               R2 (synth-data)               │
                  │                                              │
                  │ data/{dataset_config_id}/{dataset_run_id}/   │
                  │ train/{dataset_config_id}/{dataset_run_id}/  │
                  │   {train_config_id}/{train_run_id}/          │
                  │     checkpoints/                             │
                  │     config.yaml                              │
                  └───────────────┬──────────────────────────────┘
                                  │
                          upload / resume
                                  │
                                  ▼
┌──────────────┐     ┌────────────────────────┐      ┌────────────────────┐
│ Dataset      │────►│       TRAIN JOB        │─────►│  Checkpoints       │
│ local or R2  │     │  python src/train.py   │      │  last.ckpt         │
└──────────────┘     │                        │      │  best.ckpt         │
                     │  Lightning + Hydra     │      └────────────────────┘
                     │  W&B + CSV logger      │
                     └──────────┬─────────────┘
                                │
                                ▼
                         W&B training run
                         model artifact
                         metrics / lineage
```

### Environment Matrix

| Environment | Dataset source  | Checkpoint durability | GPU          | Trigger             |
| ----------- | --------------- | --------------------- | ------------ | ------------------- |
| macOS dev   | Local or R2     | Local only            | MPS          | `make train`        |
| Linux dev   | Local or R2     | Local or R2           | CUDA         | `make train`        |
| Docker      | Local or R2     | Mounted volume or R2  | CUDA         | `make docker-train` |
| RunPod      | R2-backed       | R2 + W&B              | CUDA         | `make runpod-train` |
| CI          | Fixture dataset | Ephemeral             | CPU/GPU-lite | PR trigger          |

______________________________________________________________________

## 5. Stage Definitions

### 5.1 Train

| Property     | Value                                                                             |
| ------------ | --------------------------------------------------------------------------------- |
| **Command**  | `python src/train.py experiment={exp} [overrides...]`                             |
| **Input**    | Dataset, model config, optimizer / trainer config                                 |
| **Output**   | W&B metrics, CSV logs, local checkpoints                                          |
| **Compute**  | GPU                                                                               |
| **Contract** | Train until configured stopping condition; emit checkpoints on checkpoint cadence |

### 5.2 Durable Checkpoint Upload

| Property     | Value                                                             |
| ------------ | ----------------------------------------------------------------- |
| **Trigger**  | After local checkpoint save                                       |
| **Input**    | `last.ckpt`, `best.ckpt` (and optionally other saved checkpoints) |
| **Output**   | R2 checkpoint objects under canonical train path                  |
| **Compute**  | CPU / network I/O                                                 |
| **Contract** | Idempotent upload with `--checksum`; safe to re-run               |

### 5.3 Resume

| Property     | Value                                                                |
| ------------ | -------------------------------------------------------------------- |
| **Command**  | `python src/train.py ... ckpt_path={local_or_r2_path}`               |
| **Input**    | Local or R2 checkpoint                                               |
| **Output**   | Continued training with restored optimizer / scheduler / epoch state |
| **Compute**  | GPU                                                                  |
| **Contract** | Reuse Lightning native resume semantics                              |

### 5.4 Promotion Handoff

| Property     | Value                                                                                      |
| ------------ | ------------------------------------------------------------------------------------------ |
| **Producer** | Training run                                                                               |
| **Consumer** | Promotion workflow                                                                         |
| **Output**   | Model artifact + lineage + config metadata                                                 |
| **Contract** | Training does not promote; it only produces the artifact and metadata needed for promotion |

______________________________________________________________________

## 6. R2 & W&B Integration

> Authoritative storage and W&B conventions are defined in [storage-provenance-spec.md](storage-provenance-spec.md#4-wb-artifact-types). Repeated here for training context.

### 6.1 Dataset Access

Training uses the same dataset provenance convention as the storage spec:

```yaml
dataset_root: ${paths.data_dir}/{dataset_config_id}/{dataset_wandb_run_id}
# Optional:
# r2_path: r2:synth-data/data/{dataset_config_id}/{dataset_wandb_run_id}/
```

Behavior:

- Local-only by default
- If `r2_path` is specified, sync dataset before training
- No hidden default R2 fetch

### 6.2 Durable Training Outputs

The canonical R2 training path follows [storage-provenance-spec.md §2](storage-provenance-spec.md#2-r2-bucket-layout):

```text
train/{dataset_config_id}/{dataset_wandb_run_id}/{train_config_id}/{train_wandb_run_id}/
```

Expected contents:

```text
train/{dataset_config_id}/{dataset_wandb_run_id}/{train_config_id}/{train_wandb_run_id}/
├── checkpoints/
│   ├── last.ckpt
│   └── best.ckpt
└── config.yaml
```

### 6.3 Checkpoint Upload Policy

Two options exist:

- **W&B artifacts** for browsable model registry and lineage
- **R2 checkpoint upload** for durable raw files and fast resume

Selected approach: **keep both**

- W&B remains the model registry / lineage system
- R2 is the durable raw checkpoint store for resume and portability

### 6.4 Resume From R2

Resume must work with the same `ckpt_path=` interface users already know.

Resolution behavior:

- local path → use directly
- `r2:` path → download to local cache, then pass local path to Lightning
- resume semantics stay entirely inside Lightning

A `make resume` target constructs the full R2 path from experiment and run ID to avoid manual path assembly.

### 6.5 W&B Lineage

Every training run must:

- set `job_type="training"` (per [storage-provenance-spec.md §7](storage-provenance-spec.md#7-job_type-values))
- call `run.use_artifact()` for the dataset artifact
- call `run.log_artifact()` for the model artifact
- include `github_sha` in `wandb.config`

Model artifact naming follows [storage-provenance-spec.md §4](storage-provenance-spec.md#4-wb-artifact-types):

```text
model-{train_config_id}
```

Dataset artifact naming follows:

```text
data-{dataset_config_id}
```

### 6.6 Promotion Interface

Promotion remains separate and follows [promotion-pipeline-reference.md](../reference/promotion-pipeline-reference.md).

Training is responsible for producing:

- model artifact
- training config
- dataset lineage
- final metrics in `wandb.summary`
- `github_sha`

It is **not** responsible for creating GitHub Releases.

______________________________________________________________________

## 7. Design Decisions

### 7.1 No Reconciliation Layer

**Decision:** training does not reuse the data pipeline's reconciliation backend.

**Rationale:** there is only one long-running job. Progress is recovered via checkpoints, not by recomputing missing work units.

### 7.2 Durable Checkpoints as the Recovery Boundary

**Decision:** recovery boundary is the latest successfully uploaded checkpoint.

**Rationale:** simple mental model, native Lightning compatibility, and bounded loss on crash.

### 7.3 Single-Pod RunPod Launcher

**Decision:** RunPod support is a thin launcher script, not a backend abstraction.

**Rationale:** training does not need submission graphs, worker pools, or shard assignment.

### 7.4 Separate Training Docker Image

**Decision:** use a training-focused Docker image rather than forcing the data-generation image to cover training needs.

**Rationale:** training needs CUDA / torch / model deps; data generation needs VST / rclone / headless rendering. Overlap is limited.

### 7.5 W&B + R2 Split

**Decision:** W&B is the registry / lineage layer; R2 is the raw durable checkpoint store.

**Rationale:** W&B is better for browsing and lineage; R2 is better for raw checkpoint persistence and resume.

### 7.6 No Automatic Promotion

**Decision:** promotion stays a separate workflow.

**Rationale:** keeps training focused on producing artifacts, not release management.

______________________________________________________________________

## 8. Phase Plan

> This section follows the Epic → Phase → Task hierarchy defined in
> [github-taxonomy.md](github-taxonomy.md) §3. Phase/task issue numbers marked `TBD`
> will be created when implementation begins.

### Issue Mapping

| Issue | Type  | Description                               | Parent  |
| ----- | ----- | ----------------------------------------- | ------- |
| #107  | Epic  | Training pipeline & ops                   | —       |
| TBD   | Phase | Phase 1: Portable Training Foundation     | #107    |
| TBD   | Phase | Phase 2: Checkpoint Durability            | #107    |
| TBD   | Phase | Phase 3: RunPod Launcher                  | #107    |
| TBD   | Phase | Phase 4: Docker & CI                      | #107    |
| TBD   | Phase | Phase 5: Documentation                    | #107    |
| TBD   | Task  | Task 1.1: Config Cleanup for Training     | Phase 1 |
| TBD   | Task  | Task 1.2: W&B Config Cleanup              | Phase 1 |
| TBD   | Task  | Task 2.1: R2 Checkpoint Uploader Callback | Phase 2 |
| #92   | Task  | Task 2.2: Resume From R2 Checkpoint       | Phase 2 |
| TBD   | Task  | Task 3.1: RunPod Training Launcher        | Phase 3 |
| TBD   | Task  | Task 4.1: Training Docker Image           | Phase 4 |
| TBD   | Task  | Task 4.2: Training Smoke CI               | Phase 4 |
| TBD   | Task  | Task 5.1: Training Runbook                | Phase 5 |

### Per-Phase Metadata

| Phase | Label(s)                    | Milestone         |
| ----- | --------------------------- | ----------------- |
| 1     | `training`                  | `training v1.0.0` |
| 2     | `training`, `storage`       | `training v1.0.0` |
| 3     | `training`                  | `training v1.0.0` |
| 4     | `training`, `ci-automation` | `training v1.0.0` |
| 5     | `training`                  | `training v1.0.0` |

### Completion Tracking

Use the same linkage pattern as the eval and data pipeline docs:

```text
### Task 2.1: R2 Checkpoint Uploader (TBD) ✅ — Completed in PR #XXX
```

### Estimated Change Size

| Area                         | Actual change                        | Lines |
| ---------------------------- | ------------------------------------ | ----- |
| W&B config cleanup           | env-driven entity / project defaults | ~5    |
| R2 checkpoint callback       | new callback + tests                 | ~80   |
| R2 checkpoint resolver reuse | shared path resolution logic         | ~20   |
| RunPod launcher              | one script + tests                   | ~80   |
| Docker train image           | new Dockerfile + Make targets        | ~120  |
| Training smoke CI            | workflow + fixtures + tests          | ~150  |

> A separate implementation plan document (like `data-pipeline-implementation-plan.md`)
> with per-task file lists, key behaviors, and reference tests will be created when
> implementation begins.

______________________________________________________________________

## 9. Dependency Overview

> GitHub issue dependencies are the canonical DAG. This section summarizes the
> critical path only.

### Known Dependencies

| Training work item               | Depends on                       |
| -------------------------------- | -------------------------------- |
| Task 2.1: R2 checkpoint uploader | #90 (rclone wrapper)             |
| Task 2.2: Resume from R2         | #92                              |
| Task 1.1: Config cleanup         | #94                              |
| Task 3.1: RunPod launcher        | shared credential / rclone setup |
| Task 4.1: Docker train image     | independent                      |
| Task 1.2: W&B cleanup            | independent                      |
| Task 4.2: Training CI            | independent                      |

### Critical Path

`Task 1.1 (config cleanup) → Task 2.1 (checkpoint upload) → Task 2.2 (resume from R2) → Task 3.1 (RunPod training)`

______________________________________________________________________

## 10. Alternatives Considered

| Alternative                                   | Why rejected                                                        |
| --------------------------------------------- | ------------------------------------------------------------------- |
| Reuse data pipeline reconciliation backend    | Wrong job shape; adds complexity without value                      |
| W&B-only checkpoint durability                | Good registry UX, but raw checkpoint resume path is weaker / slower |
| R2-only checkpoint strategy                   | Simpler raw files, but loses W&B registry and lineage UX            |
| Automatic promotion after training            | Blurs training and release responsibilities                         |
| Single shared Docker image with data pipeline | Too many unrelated deps in one image                                |

______________________________________________________________________

## 11. Open Questions & Risks

| #   | Question / Risk                                             | Impact                   | Status           |
| --- | ----------------------------------------------------------- | ------------------------ | ---------------- |
| 1   | Should RunPod pods auto-terminate after training exits?     | Cloud cost / orphan pods | Open             |
| 2   | Should R2 mirror every checkpoint or only best + last?      | Storage growth           | Open             |
| 3   | Is single-GPU sufficient for next-generation models?        | Future scaling           | Accepted for now |
| 4   | How should stale checkpoints be garbage-collected in R2?    | Long-term storage cost   | Open             |
| 5   | Do we keep both W&B and R2 checkpoint copies for every run? | Cost vs ergonomics       | Leaning yes      |

______________________________________________________________________

## 12. Out of Scope

- Multi-node distributed training
- Sweep orchestration
- Automatic model promotion
- Custom scheduler / queueing system
- Replacing Lightning training loop semantics
- Replacing W&B as the experiment system

______________________________________________________________________

## Appendix A: Glossary

| Term                     | Definition                                                                                                                                                         |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`train_config_id`**    | Config filename stem for the training experiment. See [storage-provenance-spec.md §1](storage-provenance-spec.md#1-ids).                                           |
| **`train_wandb_run_id`** | W&B run ID for a specific training run. Format: `{train_config_id}-{YYYYMMDDTHHMMSSZ}`. See [storage-provenance-spec.md §1](storage-provenance-spec.md#1-ids).     |
| **durable checkpoint**   | A checkpoint persisted outside the training pod (to R2 and/or W&B).                                                                                                |
| **promotion**            | Converting a trained model artifact into a GitHub Release / production alias. See [promotion-pipeline-reference.md](../reference/promotion-pipeline-reference.md). |
| **RunPod launcher**      | Thin script that creates one training pod — not a backend abstraction.                                                                                             |

## Appendix B: Current File Inventory

| File                        | Current role                    | Gap                               |
| --------------------------- | ------------------------------- | --------------------------------- |
| `src/train.py`              | Main training entry point       | No durable checkpoint upload      |
| `configs/logger/wandb.yaml` | W&B config                      | Identity cleanup needed           |
| `configs/data/*.yaml`       | Dataset paths                   | Shared portability cleanup needed |
| `docker/*`                  | Existing container setup        | Training-specific image needed    |
| `scripts/runpod_*.py`       | Data-pipeline-focused launchers | No training launcher              |

## Appendix C: Checkpoint Policy

| Checkpoint              | Keep locally          | Upload to W&B            | Upload to R2                  |
| ----------------------- | --------------------- | ------------------------ | ----------------------------- |
| `last.ckpt`             | Yes                   | Yes                      | Yes                           |
| `best.ckpt`             | Yes                   | Yes                      | Yes                           |
| Intermediate step ckpts | Per checkpoint config | Yes if `log_model="all"` | Optional; policy-configurable |

## Appendix D: Implementation Recipes

### D.1 R2 Checkpoint Uploader Callback

```python
class R2CheckpointUploader(Callback):
    """Upload checkpoints to R2 after each save event.

    Piggybacks on ModelCheckpoint's existing save events. Uploads happen at the
    same cadence as local saves. --checksum prevents redundant uploads.
    """

    def __init__(self, r2_path: str):
        self.r2_path = r2_path

    def on_train_epoch_end(self, trainer, pl_module):
        ckpt = trainer.checkpoint_callback
        if ckpt is None:
            return
        for path in [ckpt.best_model_path, ckpt.last_model_path]:
            if path and Path(path).exists():
                rclone_copyto(path, f"{self.r2_path}/{Path(path).name}")
```

Config — **no default, must be explicitly specified**:

```yaml
r2_checkpoint:
  _target_: src.callbacks.r2_checkpoint.R2CheckpointUploader
  r2_path: ???  # required when included
```

Reuses: rclone wrapper from #90, `--checksum` always.

### D.2 RunPod Training Launcher

```python
# scripts/runpod_train.py
def launch_training(
    experiment: str,
    config_overrides: list[str],
    gpu_type: str = "NVIDIA RTX A5000",
    image: str = "tinaudio/synth-setter-train:latest",
):
    """Launch a single training pod on RunPod."""
    cmd = f"python src/train.py experiment={experiment} {' '.join(config_overrides)}"
    pod = runpod.create_pod(
        name=f"train-{experiment}-{timestamp}",
        image_name=image,
        gpu_type_id=gpu_type,
        docker_args=cmd,
        env={
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
            "RCLONE_CONFIG_R2_TYPE": "s3",
            # ... R2 credentials (see storage-provenance-spec.md §9)
        },
    )
    return pod
```

No reconciliation, no batch submission. One pod, one training run.

______________________________________________________________________
