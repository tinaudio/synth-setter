# Design Doc: Training Pipeline

> **Status**: Draft
> **Author**: ktinubu@
> **Last Updated**: 2026-05-13
> **Tracking**: #107
> **Issue tracking**: [github-taxonomy.md](github-taxonomy.md)

> **Storage & provenance conventions**: [storage-provenance-spec.md](storage-provenance-spec.md) (authoritative)

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

**Key insight:** the data pipeline's reconciliation backend does not apply to training. Training on RunPod is just "launch one pod with `python -m synth_setter.cli.train ...`". No shard partitioning, no worker graph, no storage-based coordination loop.

### Current Strengths

| Already works today                                       | Notes                                                                                                                          |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `python -m synth_setter.cli.train` with Hydra composition | Mature entry point                                                                                                             |
| W&B logger                                                | Tracks metrics; `log_model: False` (no checkpoint files uploaded — best ckpt goes to R2 as a referenced artifact at train end) |
| `ModelCheckpoint`                                         | Saves every 5000 steps + best + last                                                                                           |
| CSV logger                                                | Local fallback                                                                                                                 |
| Lightning resume                                          | `ckpt_path=` already supported                                                                                                 |
| `operator_workspace()` / `PROJECT_ROOT`                   | Paths already resolve cleanly via `synth_setter.workspace`                                                                     |

### Current Gaps

| Gap                                         | Impact                                                                                                                                                                                                                                                                                                |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hardcoded dataset paths in configs          | Fixed by shared config cleanup work                                                                                                                                                                                                                                                                   |
| ~~No durable cloud checkpoint persistence~~ | ~~Pod death loses progress~~ — **Resolved:** at train end the best checkpoint is uploaded to R2 and the `model-{config_id}` artifact references it ([#92](https://github.com/tinaudio/synth-setter/issues/92)). Intermediate checkpoints are not synced, so pod death before train end loses progress |
| No RunPod training launcher                 | Manual cloud startup                                                                                                                                                                                                                                                                                  |
| No training-focused Docker image            | Hard to reproduce cloud/local parity                                                                                                                                                                                                                                                                  |
| ~~Hardcoded W&B identity~~                  | ~~Wrong defaults for new ownership~~ — **Resolved:** entity/project now env-var driven via `oc.env` resolver                                                                                                                                                                                          |

______________________________________________________________________

## 2. Typical Workflow

### Local development (target state)

```yaml
# src/synth_setter/configs/experiment/surge/flow_simple.yaml (proposed)
defaults:
  - override /datamodule: surge_simple
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
python -m synth_setter.cli.train experiment=surge/flow_simple ckpt_path=logs/train/.../checkpoints/last.ckpt
```

### RunPod training (target state)

```bash
# Launch a single long-running training pod
make runpod-train EXPERIMENT=surge/flow_simple

# If the pod dies, resume from the W&B model artifact:
make resume EXPERIMENT=surge/flow_simple RUN_ID=<train_wandb_run_id>

# Or specify a W&B artifact alias directly:
python -m synth_setter.cli.train \
  experiment=surge/flow_simple \
  ckpt_path=wandb:model-surge-flow-simple:latest
```

In the target/experimental setup (scoped and validated on the `experiment` branch — [#409](https://github.com/tinaudio/synth-setter/issues/409)), cloud training is expected to run with `MODE=train`. This downloads the dataset from R2 via rclone, runs `src/synth_setter/cli/train.py` with Hydra config, and uploads checkpoints to R2 at `r2:intermediate-data/train/{dataset_config_id}/{dataset_wandb_run_id}/{train_config_id}/{train_wandb_run_id}/` (per [storage-provenance-spec.md §2](storage-provenance-spec.md#2-r2-bucket-layout)). On main, checkpoint durability is W&B-only (see Section 6.2).

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

| Metric            | Target                                                       | How to Measure                                              |
| ----------------- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| Resume durability | Best checkpoint survives pod death in R2                     | Resume from the `model-{config_id}` artifact's R2 reference |
| Portability       | Same experiment runs locally, in Docker, and on RunPod       | Smoke tests + manual parity run                             |
| Provenance        | Every run records dataset, config, SHA, and artifact lineage | Inspect W&B run + storage path                              |
| Local smoke test  | Tiny fixture reaches checkpoint and exits cleanly            | CI                                                          |
| Crash recovery UX | One documented command to resume                             | Runbook                                                     |

### Non-Goals

- Multi-node distributed training
- Hyperparameter sweep orchestration
- Automated model promotion
- Replacing Lightning checkpoint logic
- Replacing W&B with a custom registry

______________________________________________________________________

## 4. System Overview

Training is a **single long-running job**. Durable outputs flow through W&B (metrics, lineage) and R2 (the best checkpoint):

1. **Metrics and lineage → W&B run**
2. **Best checkpoint → R2**, referenced by the `model-{config_id}` W&B artifact as an `s3://` URI (no file uploaded to W&B; `log_model: False`)
3. **Local checkpoints → local disk** (Lightning default, ephemeral on cloud pods)

```
┌──────────────┐     ┌────────────────────────┐      ┌────────────────────┐
│ Dataset      │────►│       TRAIN JOB        │─────►│  Local checkpoints │
│ local or R2  │     │  python -m synth_setter.cli.train   │      │  last.ckpt         │
└──────────────┘     │                        │      │  best.ckpt         │
                     │  Lightning + Hydra     │      └────────────────────┘
                     │  W&B + CSV logger      │
                     └──────────┬─────────────┘
                                │
                                ▼
                         W&B training run
                         model artifact (s3:// ref → R2)
                         metrics / lineage
```

The best checkpoint is uploaded to R2 at train end and referenced by the model artifact ([#92](https://github.com/tinaudio/synth-setter/issues/92)). See §10.

### Environment Matrix

| Environment | Dataset source  | Checkpoint durability | GPU          | Trigger             |
| ----------- | --------------- | --------------------- | ------------ | ------------------- |
| macOS dev   | Local or R2     | Local only            | MPS          | `make train`        |
| Linux dev   | Local or R2     | Local + R2 (best)     | CUDA         | `make train`        |
| Docker      | Local or R2     | Local + R2 (best)     | CUDA         | `make docker-train` |
| RunPod      | R2-backed       | R2 (best)             | CUDA         | `make runpod-train` |
| CI          | Fixture dataset | Ephemeral             | CPU/GPU-lite | PR trigger          |

______________________________________________________________________

## 5. Stage Definitions

### 5.1 Train

| Property     | Value                                                                             |
| ------------ | --------------------------------------------------------------------------------- |
| **Command**  | `python -m synth_setter.cli.train experiment={exp} [overrides...]`                |
| **Input**    | Dataset, model config, optimizer / trainer config                                 |
| **Output**   | W&B metrics, CSV logs, local checkpoints                                          |
| **Compute**  | GPU                                                                               |
| **Contract** | Train until configured stopping condition; emit checkpoints on checkpoint cadence |

### 5.2 R2 Checkpoint Durability

| Property     | Value                                                                                                                                     |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Trigger**  | At train end, on global-zero, best-effort — `_upload_best_checkpoint` uploads the best checkpoint to R2; the model artifact references it |
| **Input**    | `best.ckpt`                                                                                                                               |
| **Output**   | `r2://{r2.bucket}/checkpoints/{config_id}/model.ckpt` + a `model-{config_id}` artifact referencing it as an `s3://` URI                   |
| **Compute**  | One rclone upload at train end                                                                                                            |
| **Contract** | Degrades to a lineage-only artifact when R2 is unreachable or no checkpoint was written; training is never aborted                        |

### 5.3 Resume

| Property     | Value                                                                           |
| ------------ | ------------------------------------------------------------------------------- |
| **Command**  | `python -m synth_setter.cli.train ... ckpt_path={local_path_or_wandb_artifact}` |
| **Input**    | Local checkpoint or W&B artifact reference                                      |
| **Output**   | Continued training with restored optimizer / scheduler / epoch state            |
| **Compute**  | GPU                                                                             |
| **Contract** | Reuse Lightning native resume semantics                                         |

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

The R2 training-artifact subtree (under the `intermediate-data/` bucket root — see [storage-provenance-spec §2](storage-provenance-spec.md#2-r2-bucket-layout)) mirrors [§3b Training](storage-provenance-spec.md#3b-training):

```
train/{dataset_config_id}/{dataset_wandb_run_id}/{train_config_id}/{train_wandb_run_id}/
├── checkpoints/
│   ├── last.ckpt
│   └── best.ckpt
└── config.yaml               # Frozen experiment config
```

The best checkpoint is uploaded to R2 at train end and referenced by the `model-{config_id}` artifact (see §6.2, §7.5); intermediate checkpoints stay local. The per-run path above is the layout the experimental `MODE=train` flow ([#409](https://github.com/tinaudio/synth-setter/issues/409)) writes to; the train-end best-checkpoint upload uses the per-`config_id` path `checkpoints/{config_id}/model.ckpt`.

### 6.1 Dataset Access

Storage backend is selected by datamodule group: `datamodule=surge` reads `train/val/test.h5`
via `VSTDataModule`; `datamodule=surge_lance` reads single-file `train/val/test.lance` shards —
the format the data pipeline's finalize step emits — via `LanceVSTDataModule`
(`src/synth_setter/data/lance_datamodule.py`), a subclass that overrides the `dataset_cls` /
`shard_suffix` extension points. The shipped `src/synth_setter/configs/datamodule/surge*.yaml` default `dataset_root` to the per-run Hydra
output dir; a fixed dataset is pinned by overriding to the storage-spec provenance layout:

```yaml
dataset_root: ${paths.output_dir}/data # shipped default (per-run Hydra dir)
# Optional — pin a fixed dataset by provenance convention instead:
# dataset_root: ${paths.data_dir}/{dataset_config_id}/{dataset_wandb_run_id}
# download_dataset_root_uri: r2://intermediate-data/data/{dataset_config_id}/{dataset_wandb_run_id}/
```

Behavior:

- Local-only by default
- If `download_dataset_root_uri` is specified, no-clobber-copy the dataset before training
- With the Lance datamodule, `datamodule.stream_from_r2=true` reads splits natively over R2's S3 API instead of downloading them (only `stats.npz` is fetched locally)
- No hidden default R2 fetch

### 6.2 Checkpoint Durability via R2

`log_model: False` keeps checkpoint files out of W&B (5 GB total storage budget). At train end, on global-zero, `train.py` uploads the best checkpoint to R2 (`_upload_best_checkpoint`) at the auto-derived `r2://{r2.bucket}/checkpoints/{config_id}/model.ckpt` (`_derive_checkpoint_uri`), then the `model-{config_id}` artifact references that object as an `s3://` URI (`checksum=False`) — so W&B stores only a ~0-byte reference. `training.upload_checkpoints_uri` optionally overrides the target (null = auto-derive). Intermediate checkpoints are not synced.

```yaml
# src/synth_setter/configs/logger/wandb.yaml
wandb:
  _target_: lightning.pytorch.loggers.WandbLogger
  project: synth-setter
  log_model: False  # no checkpoint files to W&B; best ckpt goes to R2 as a referenced artifact
```

This gives us:

- **Durability** — the best checkpoint survives pod death in R2 (intermediate checkpoints are not persisted)
- **Lineage** — the artifact is linked to the run, dataset, config, and git SHA
- **Resume** — the `${wandb:...}` resolver rclone-downloads the referenced checkpoint from R2 to resume from any machine
- **Registry** — browsable in the W&B model registry

> **Known limitation:** the R2 object lives at a per-`config_id` path and is overwritten each run, so an older artifact version (`:vN` for N < latest) resolves to the current object. (Per-version immutability would fold the run id into the path — deferred, YAGNI.)

### 6.3 Resume From W&B

Resume must work with the same `ckpt_path=` interface users already know.

Resolution behavior:

- local path → use directly
- `wandb:` artifact reference → resolve the artifact's `s3://` reference, rclone-download from R2 to local cache, then pass the local path to Lightning (legacy file-upload artifacts fall back to native `download()`)
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

### 7.2 R2 as the Checkpoint Durability Layer

**Decision:** the best checkpoint is uploaded to R2 at train end and referenced by the W&B `model-{config_id}` artifact. `log_model: False` keeps checkpoint files out of W&B.

**Rationale:** W&B's 5 GB total storage cannot hold every checkpoint file. Storing only an `s3://` reference (~0 bytes in W&B) keeps lineage and resume working while the bytes live in R2. Only the best checkpoint is persisted, at train end — intermediate checkpoints are not synced, so this is durability for the final result, not crash resilience.

### 7.3 Single-Pod RunPod Launcher

**Decision:** RunPod support is a thin launcher script, not a backend abstraction.

**Rationale:** training does not need submission graphs, worker pools, or shard assignment.

### 7.4 Separate Training Docker Image

**Decision:** use a training-focused Docker image rather than forcing the data-generation image to cover training needs.

**Rationale:** training needs CUDA / torch / model deps; data generation needs VST / rclone / headless rendering. Overlap is limited.

### 7.5 R2-Referenced Checkpoints

**Decision:** the best checkpoint lives in R2; the W&B artifact holds only an `s3://` reference and the run lineage.

**Rationale:** uploading checkpoint files to W&B (`log_model: "all"`) would exhaust the 5 GB storage budget. Referencing R2 keeps lineage and resume intact at near-zero W&B storage cost. Only the best checkpoint is mirrored at train end — GC is implicit (the per-`config_id` path is overwritten each run), and dual-copy cost is one upload, not a per-interval stream.

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

| Alternative                                              | Why rejected / deferred                                                                                                                  |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Reuse data pipeline reconciliation backend               | Wrong job shape; adds complexity without value                                                                                           |
| Upload every checkpoint file to W&B (`log_model: "all"`) | Would exhaust the 5 GB total W&B storage budget. **Selected instead:** best checkpoint to R2, referenced by the W&B artifact             |
| R2-only checkpoint strategy (no W&B artifact)            | Simpler raw files, but loses W&B registry, lineage, and model browsing UX. The chosen design keeps the W&B artifact as a reference to R2 |
| Automatic promotion after training                       | Blurs training and release responsibilities                                                                                              |
| Single shared Docker image with data pipeline            | Too many unrelated deps in one image                                                                                                     |

______________________________________________________________________

## 11. Open Questions & Risks

| #   | Question / Risk                                              | Impact                   | Status                                                                       |
| --- | ------------------------------------------------------------ | ------------------------ | ---------------------------------------------------------------------------- |
| 1   | Should RunPod pods auto-terminate after training exits?      | Cloud cost / orphan pods | Open                                                                         |
| 2   | ~~Should R2 mirror every checkpoint or only best + last?~~   | ~~Storage growth~~       | Resolved — only the best checkpoint is mirrored to R2 at train end           |
| 3   | Is single-GPU sufficient for next-generation models?         | Future scaling           | Accepted for now                                                             |
| 4   | ~~How should stale checkpoints be garbage-collected in R2?~~ | ~~Storage cost~~         | Resolved — the per-`config_id` path is overwritten each run (no separate GC) |
| 5   | ~~Do we keep both W&B and R2 checkpoint copies?~~            | ~~Cost~~                 | Resolved — bytes live in R2; W&B holds only an `s3://` reference             |

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

| Term                     | Definition                                                                                                                                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`train_config_id`**    | Config filename stem for the training experiment. See [storage-provenance-spec.md §1](storage-provenance-spec.md#1-ids).                                                                          |
| **`train_wandb_run_id`** | W&B run ID for a specific training run. Default format: `{train_config_id}-{YYYYMMDDTHHMMSSsssZ}` (millisecond precision). See [storage-provenance-spec.md §1](storage-provenance-spec.md#1-ids). |
| **durable checkpoint**   | A checkpoint persisted outside the training pod as a W&B model artifact.                                                                                                                          |
| **promotion**            | Converting a trained model artifact into a GitHub Release / production alias. See [promotion-pipeline-reference.md](../reference/promotion-pipeline-reference.md).                                |
| **RunPod launcher**      | Thin script that creates one training pod — not a backend abstraction.                                                                                                                            |

## Appendix B: Current File Inventory

| File                                         | Current role                                                                         | Gap                               |
| -------------------------------------------- | ------------------------------------------------------------------------------------ | --------------------------------- |
| `src/synth_setter/cli/train.py`              | Main training entry point                                                            | —                                 |
| `src/synth_setter/configs/logger/wandb.yaml` | W&B config (`log_model: False` — no checkpoint files to W&B, env-var entity/project) | —                                 |
| `src/synth_setter/configs/datamodule/*.yaml` | Dataset paths                                                                        | Shared portability cleanup needed |
| `docker/*`                                   | Existing container setup                                                             | Training-specific image needed    |
| `scripts/runpod_*.py`                        | Data-pipeline-focused launchers                                                      | No training launcher              |

## Appendix C: Checkpoint Policy

| Checkpoint              | Keep locally          | Persisted to cloud                                                   |
| ----------------------- | --------------------- | -------------------------------------------------------------------- |
| `last.ckpt`             | Yes                   | No                                                                   |
| `best.ckpt`             | Yes                   | Yes — uploaded to R2, referenced by the `model-{config_id}` artifact |
| Intermediate step ckpts | Per checkpoint config | No                                                                   |

> No checkpoint files are uploaded to W&B (`log_model: False`). Only the best checkpoint is mirrored to R2, at train end.

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
    cmd = f"python -m synth_setter.cli.train experiment={experiment} {' '.join(config_overrides)}"
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
