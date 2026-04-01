# Design Doc: Training Pipeline

> **Status**: Draft
> **Author**: ktinubu@
> **Last Updated**: 2026-03-20
> **Tracking**: #107
> **Storage conventions**: [storage-provenance-spec.md](storage-provenance-spec.md)
> **Issue tracking**: [github-taxonomy.md](github-taxonomy.md)

______________________________________________________________________

### Index

| §   | Section                                                                       | What it covers                                                          |
| --- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 1   | [Context & Motivation](#1-context--motivation)                                | Problem statement, current state, why training needs different ops      |
| 2   | [Typical Workflow](#2-typical-workflow)                                       | End-to-end local, Docker, and RunPod examples                           |
| 3   | [Goals, Non-Goals & Design Principles](#3-goals-non-goals--design-principles) | Requirements, principles, anti-goals, success metrics                   |
| 4   | [System Overview](#4-system-overview)                                         | Single-job architecture, checkpoint lifecycle, environment matrix       |
| 5   | [Stage Definitions](#5-stage-definitions)                                     | Train, W&B checkpoint durability, resume, promotion handoff             |
| 6   | [W&B Integration](#6-wb-integration)                                          | Dataset access, checkpoint durability, resume, lineage, release handoff |
| 7   | [Design Decisions](#7-design-decisions)                                       | Single-job model, W&B checkpoints, launcher shape, Docker split         |
| 8   | [Phase Plan](#8-phase-plan)                                                   | Epic → Phase → Task hierarchy, issue mapping, file lists, test strategy |
| 9   | [Dependency Overview](#9-dependency-overview)                                 | Issue dependencies, parallel execution windows, critical path           |
| 10  | [Alternatives Considered](#10-alternatives-considered)                        | Rejected approaches and why                                             |
| 11  | [Open Questions & Risks](#11-open-questions--risks)                           | Known gaps and trade-offs                                               |
| 12  | [Out of Scope](#12-out-of-scope)                                              | Future work — not referenced elsewhere                                  |
| A–D | [Appendices](#appendix-a-glossary)                                            | Glossary, file inventory, checkpoint policy, RunPod launcher recipe     |

______________________________________________________________________

## 1. Context & Motivation

Topline goal: Run training portably on local machines, Docker, and RunPod, with durable checkpoints in cloud storage and enough provenance to resume, compare, and promote models safely.

**synth-setter** already has a strong training entry point: Hydra config composition, PyTorch Lightning, W&B logging, CSV logging, and `ModelCheckpoint` saving every 5000 steps plus best and last. What it lacks is **portable durability**. Today, if a long-running cloud pod dies, local checkpoints on that pod die with it.

Training is operationally different from the data pipeline:

| Concern          | Data pipeline                             | Training                              |
| ---------------- | ----------------------------------------- | ------------------------------------- |
| Job shape        | Many short parallel workers               | One long-running job                  |
| Duration         | Minutes per worker                        | Hours to days                         |
| Coordination     | Reconciliation (desired vs actual shards) | None — single job                     |
| Failure recovery | Re-run missing shards                     | Resume from latest durable checkpoint |
| Monitoring       | Storage-based completion                  | W&B metrics + pod liveness            |
| Output           | Dataset shards → R2                       | Checkpoints → W&B, metrics → W&B      |

**Key insight:** the data pipeline's reconciliation backend does not apply to training. Training on RunPod is just "launch one pod with `python src/train.py ...`". No shard partitioning, no worker graph, no storage-based coordination loop.

### Current Strengths

| Already works today                          | Notes                                                                                     |
| -------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `python src/train.py` with Hydra composition | Mature entry point                                                                        |
| W&B logger                                   | Tracks metrics; `log_model: "all"` uploads every checkpoint as a W&B artifact immediately |
| `ModelCheckpoint`                            | Saves every 5000 steps + best + last                                                      |
| CSV logger                                   | Local fallback                                                                            |
| Lightning resume                             | `ckpt_path=` already supported                                                            |
| `rootutils` / `PROJECT_ROOT`                 | Paths already resolve cleanly                                                             |

### Current Gaps

| Gap                                         | Impact                                                                                                                        |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Hardcoded dataset paths in configs          | Fixed by shared config cleanup work                                                                                           |
| ~~No durable cloud checkpoint persistence~~ | ~~Pod death loses progress~~ — **Resolved:** `log_model: "all"` uploads every checkpoint to W&B immediately (crash-resilient) |
| No RunPod training launcher                 | Manual cloud startup                                                                                                          |
| No training-focused Docker image            | Hard to reproduce cloud/local parity                                                                                          |
| ~~Hardcoded W&B identity~~                  | ~~Wrong defaults for new ownership~~ — **Resolved:** entity/project now env-var driven via `oc.env` resolver                  |

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

# If the pod dies, resume from the W&B model artifact:
make resume EXPERIMENT=surge/flow_simple RUN_ID=<train_wandb_run_id>

# Or specify a W&B artifact alias directly:
python src/train.py \
  experiment=surge/flow_simple \
  ckpt_path=wandb:model-surge-flow-simple:latest
```

In the target/experimental setup (scoped and validated on the `experiment` branch — [#409](https://github.com/tinaudio/synth-setter/issues/409)), cloud training is expected to run with `MODE=train`. This downloads the dataset from R2 via rclone, runs `src/train.py` with Hydra config, and uploads checkpoints to R2 at `{R2_PREFIX}/training/{wandb_run_id}/`. On main, checkpoint durability is W&B-only (see Section 6.2).

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
- **W&B for metrics, lineage, and checkpoint durability.**

### What This System Deliberately Avoids

- Training-specific orchestration framework
- Queueing / polling backend for long-running jobs
- Storage-based coordination like the data pipeline
- Multi-node design before it is needed
- Automatic promotion after training

### Success Metrics

| Metric            | Target                                                       | How to Measure                                   |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| Resume durability | Pod death loses at most one checkpoint interval              | Crash at step N, resume from latest W&B artifact |
| Portability       | Same experiment runs locally, in Docker, and on RunPod       | Smoke tests + manual parity run                  |
| Provenance        | Every run records dataset, config, SHA, and artifact lineage | Inspect W&B run + storage path                   |
| Local smoke test  | Tiny fixture reaches checkpoint and exits cleanly            | CI                                               |
| Crash recovery UX | One documented command to resume                             | Runbook                                          |

### Non-Goals

- Multi-node distributed training
- Hyperparameter sweep orchestration
- Automated model promotion
- Replacing Lightning checkpoint logic
- Replacing W&B with a custom registry

______________________________________________________________________

## 4. System Overview

Training is a **single long-running job**. All durable outputs flow through W&B:

1. **Metrics and lineage → W&B run**
2. **Checkpoint files → W&B model artifact** (via `log_model: "all"`)
3. **Local checkpoints → local disk** (Lightning default, ephemeral on cloud pods)

```
┌──────────────┐     ┌────────────────────────┐      ┌────────────────────┐
│ Dataset      │────►│       TRAIN JOB        │─────►│  Local checkpoints │
│ local or R2  │     │  python src/train.py   │      │  last.ckpt         │
└──────────────┘     │                        │      │  best.ckpt         │
                     │  Lightning + Hydra     │      └────────────────────┘
                     │  W&B + CSV logger      │
                     └──────────┬─────────────┘
                                │
                                ▼
                         W&B training run
                         model artifact (log_model: "all")
                         metrics / lineage
```

R2 checkpoint upload is a future optimization if W&B artifact download becomes a bottleneck for large models. See §10.

### Environment Matrix

| Environment | Dataset source  | Checkpoint durability | GPU          | Trigger             |
| ----------- | --------------- | --------------------- | ------------ | ------------------- |
| macOS dev   | Local or R2     | Local only            | MPS          | `make train`        |
| Linux dev   | Local or R2     | Local + W&B           | CUDA         | `make train`        |
| Docker      | Local or R2     | Local + W&B           | CUDA         | `make docker-train` |
| RunPod      | R2-backed       | W&B                   | CUDA         | `make runpod-train` |
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

### 5.2 W&B Checkpoint Durability

| Property     | Value                                                                                           |
| ------------ | ----------------------------------------------------------------------------------------------- |
| **Trigger**  | Automatic — `WandbLogger(log_model="all")` uploads every checkpoint immediately after each save |
| **Input**    | `last.ckpt`, `best.ckpt`                                                                        |
| **Output**   | W&B model artifact with checkpoint files and run lineage                                        |
| **Compute**  | Background upload (non-blocking)                                                                |
| **Contract** | Zero custom code — Lightning's `WandbLogger` handles upload natively                            |

### 5.3 Resume

| Property     | Value                                                                |
| ------------ | -------------------------------------------------------------------- |
| **Command**  | `python src/train.py ... ckpt_path={local_path_or_wandb_artifact}`   |
| **Input**    | Local checkpoint or W&B artifact reference                           |
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

## 6. W&B Integration

> Authoritative storage and W&B conventions are defined in [storage-provenance-spec.md](storage-provenance-spec.md#4-wb-artifact-types). Repeated here for training context.

### 6.1 Dataset Access

Training uses the same dataset provenance convention as the storage spec:

```yaml
dataset_root: ${paths.data_dir}/{dataset_config_id}/{dataset_wandb_run_id}
# Optional:
# r2_path: r2:intermediate-data/data/{dataset_config_id}/{dataset_wandb_run_id}/
```

Behavior:

- Local-only by default
- If `r2_path` is specified, sync dataset before training
- No hidden default R2 fetch

### 6.2 Checkpoint Durability via W&B

Lightning's `WandbLogger` with `log_model: "all"` uploads every checkpoint as a W&B artifact immediately after each save. No custom callback needed. Every 5000-step checkpoint, best, and last are all uploaded immediately — pod death loses at most one checkpoint interval.

```yaml
# configs/logger/wandb.yaml
wandb:
  _target_: lightning.pytorch.loggers.WandbLogger
  project: synth-setter
  log_model: "all"  # uploads every checkpoint immediately as model artifacts (crash-resilient)
```

This gives us:

- **Durability** — checkpoints survive pod death as W&B artifacts
- **Lineage** — each artifact is linked to the run, dataset, config, and git SHA
- **Resume** — download the artifact to resume from any machine
- **Registry** — browsable in the W&B model registry

### 6.3 Resume From W&B

Resume must work with the same `ckpt_path=` interface users already know.

Resolution behavior:

- local path → use directly
- `wandb:` artifact reference → download to local cache, then pass local path to Lightning
- resume semantics stay entirely inside Lightning

A `make resume` target resolves the W&B artifact from experiment and run ID to avoid manual path assembly.

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

### 7.2 W&B as the Checkpoint Durability Layer

**Decision:** checkpoint durability uses W&B artifacts via `log_model: "all"` (every checkpoint uploaded immediately). No custom R2 upload callback.

**Rationale:** zero custom code — Lightning's `WandbLogger` handles upload natively. Lineage is automatic. Resume downloads the artifact. R2 checkpoint upload is deferred until W&B download speed becomes a bottleneck for large models. `log_model: "all"` is used for crash resilience — every checkpoint is uploaded immediately, so pod death loses at most one checkpoint interval (5000 steps).

### 7.3 Single-Pod RunPod Launcher

**Decision:** RunPod support is a thin launcher script, not a backend abstraction.

**Rationale:** training does not need submission graphs, worker pools, or shard assignment.

### 7.4 Separate Training Docker Image

**Decision:** use a training-focused Docker image rather than forcing the data-generation image to cover training needs.

**Rationale:** training needs CUDA / torch / model deps; data generation needs VST / rclone / headless rendering. Overlap is limited.

### 7.5 W&B-Only Checkpoints (v1)

**Decision:** W&B handles both checkpoint durability and lineage. R2 checkpoint upload deferred.

**Rationale:** `log_model: "all"` provides durability (every checkpoint uploaded immediately), lineage, and resume with zero custom code. Adding an R2 mirror would require a custom callback (~100 lines), rclone in the training path, R2 credential plumbing in every environment, and three open design questions (which checkpoints to mirror, GC policy, dual-copy cost). Defer until needed.

### 7.6 No Automatic Promotion

**Decision:** promotion stays a separate workflow.

**Rationale:** keeps training focused on producing artifacts, not release management.

______________________________________________________________________

## 8. Phase Plan

> This section follows the Epic → Phase → Task hierarchy defined in
> [github-taxonomy.md](github-taxonomy.md) §3. Phase/task issue numbers marked `TBD`
> will be created when implementation begins.

### Issue Mapping

| Issue | Type  | Description                           | Parent  |
| ----- | ----- | ------------------------------------- | ------- |
| #107  | Epic  | Training pipeline & ops               | —       |
| TBD   | Phase | Phase 1: Portable Training Foundation | #107    |
| TBD   | Phase | Phase 2: W&B Checkpoint Durability    | #107    |
| TBD   | Phase | Phase 3: RunPod Launcher              | #107    |
| TBD   | Phase | Phase 4: Docker & CI                  | #107    |
| TBD   | Phase | Phase 5: Documentation                | #107    |
| TBD   | Task  | Task 1.1: Config Cleanup for Training | Phase 1 |
| TBD   | Task  | Task 1.2: W&B Config Cleanup          | Phase 1 |
| #92   | Task  | Task 2.2: Resume From W&B Artifact    | Phase 2 |
| TBD   | Task  | Task 3.1: RunPod Training Launcher    | Phase 3 |
| TBD   | Task  | Task 4.1: Training Docker Image       | Phase 4 |
| TBD   | Task  | Task 4.2: Training Smoke CI           | Phase 4 |
| TBD   | Task  | Task 5.1: Training Runbook            | Phase 5 |

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
### Task 2.2: Resume from W&B artifact (#92) ✅ — Completed in PR #XXX
```

### Estimated Change Size

| Area                         | Actual change                        | Lines |
| ---------------------------- | ------------------------------------ | ----- |
| W&B config cleanup           | env-driven entity / project defaults | ~5    |
| W&B artifact resume resolver | download artifact → local path       | ~40   |
| RunPod launcher              | one script + tests                   | ~80   |
| Docker train image           | new Dockerfile + Make targets        | ~120  |
| Training smoke CI            | workflow + fixtures + tests          | ~150  |

> A separate implementation plan document, [`training-pipeline-implementation-plan.md`](training-pipeline-implementation-plan.md),
> provides per-task file lists, key behaviors, and reference tests for this design.

______________________________________________________________________

## 9. Dependency Overview

> GitHub issue dependencies are the canonical DAG. This section summarizes the
> critical path only.

### Known Dependencies

| Training work item           | Depends on                       |
| ---------------------------- | -------------------------------- |
| Task 2.2: Resume from W&B    | #92                              |
| Task 1.1: Config cleanup     | #94                              |
| Task 3.1: RunPod launcher    | shared credential / rclone setup |
| Task 4.1: Docker train image | independent                      |
| Task 1.2: W&B cleanup        | independent                      |
| Task 4.2: Training CI        | independent                      |

### Critical Path

`Task 1.1 (config cleanup) → Task 2.2 (resume from W&B) → Task 3.1 (RunPod training)`

______________________________________________________________________

## 10. Alternatives Considered

| Alternative                                   | Why rejected / deferred                                                                                                                         |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Reuse data pipeline reconciliation backend    | Wrong job shape; adds complexity without value                                                                                                  |
| R2 + W&B dual checkpoint strategy             | Custom callback (~100 lines), rclone in training path, credential plumbing in every env. Deferred until W&B download speed becomes a bottleneck |
| R2-only checkpoint strategy                   | Simpler raw files, but loses W&B registry, lineage, and model browsing UX                                                                       |
| Automatic promotion after training            | Blurs training and release responsibilities                                                                                                     |
| Single shared Docker image with data pipeline | Too many unrelated deps in one image                                                                                                            |

______________________________________________________________________

## 11. Open Questions & Risks

| #   | Question / Risk                                              | Impact                   | Status                                   |
| --- | ------------------------------------------------------------ | ------------------------ | ---------------------------------------- |
| 1   | Should RunPod pods auto-terminate after training exits?      | Cloud cost / orphan pods | Open                                     |
| 2   | ~~Should R2 mirror every checkpoint or only best + last?~~   | ~~Storage growth~~       | Resolved — W&B-only for v1; no R2 mirror |
| 3   | Is single-GPU sufficient for next-generation models?         | Future scaling           | Accepted for now                         |
| 4   | ~~How should stale checkpoints be garbage-collected in R2?~~ | ~~Storage cost~~         | Resolved — no R2 checkpoints in v1       |
| 5   | ~~Do we keep both W&B and R2 checkpoint copies?~~            | ~~Cost~~                 | Resolved — W&B-only for v1               |

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

| Term                     | Definition                                                                                                                                                             |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`train_config_id`**    | Config filename stem for the training experiment. See [storage-provenance-spec.md §1](storage-provenance-spec.md#1-ids).                                               |
| **`train_wandb_run_id`** | W&B run ID for a specific training run. Default format: `{train_config_id}-{YYYYMMDDTHHMMSSZ}`. See [storage-provenance-spec.md §1](storage-provenance-spec.md#1-ids). |
| **durable checkpoint**   | A checkpoint persisted outside the training pod as a W&B model artifact.                                                                                               |
| **promotion**            | Converting a trained model artifact into a GitHub Release / production alias. See [promotion-pipeline-reference.md](../reference/promotion-pipeline-reference.md).     |
| **RunPod launcher**      | Thin script that creates one training pod — not a backend abstraction.                                                                                                 |

## Appendix B: Current File Inventory

| File                        | Current role                                                                                   | Gap                               |
| --------------------------- | ---------------------------------------------------------------------------------------------- | --------------------------------- |
| `src/train.py`              | Main training entry point                                                                      | —                                 |
| `configs/logger/wandb.yaml` | W&B config (`log_model: "all"` — uploads every checkpoint immediately, env-var entity/project) | —                                 |
| `configs/data/*.yaml`       | Dataset paths                                                                                  | Shared portability cleanup needed |
| `docker/*`                  | Existing container setup                                                                       | Training-specific image needed    |
| `scripts/runpod_*.py`       | Data-pipeline-focused launchers                                                                | No training launcher              |

## Appendix C: Checkpoint Policy

| Checkpoint              | Keep locally          | Upload to W&B            |
| ----------------------- | --------------------- | ------------------------ |
| `last.ckpt`             | Yes                   | Yes (`log_model: "all"`) |
| `best.ckpt`             | Yes                   | Yes (`log_model: "all"`) |
| Intermediate step ckpts | Per checkpoint config | Yes (`log_model: "all"`) |

> R2 checkpoint upload is deferred. If added later, only `best` + `last` would be mirrored.

## Appendix D: Implementation Recipes

### D.1 RunPod Training Launcher

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
