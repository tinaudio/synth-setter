# Implementation Plan: Training Pipeline

> **Status**: INCOMPLETE — Implementation is not yet tracked against the phase criteria.
>
> **Canonical design:** [training-pipeline.md](training-pipeline.md)
> **Tracking:** #107
> **Issue tracking:** [github-taxonomy.md](github-taxonomy.md)
> **Storage conventions:** [storage-provenance-spec.md](storage-provenance-spec.md)
> **Last Updated:** 2026-03-31

______________________________________________________________________

### Index

| Section                                          | Content                       |
| ------------------------------------------------ | ----------------------------- |
| [Branch Strategy](#branch-strategy)              | Merge path and phase overview |
| [Phase 1](#phase-1-portable-training-foundation) | Config cleanup, smoke test    |
| [Phase 2](#phase-2-wb-checkpoint-durability)     | W&B resume                    |
| [Phase 3](#phase-3-runpod-launcher)              | RunPod launcher, resume       |
| [Phase 4](#phase-4-docker--ci)                   | Docker image, CI smoke test   |
| [Phase 5](#phase-5-documentation)                | Training runbook              |
| [Estimated Change Size](#estimated-change-size)  | Per-area line estimates       |

______________________________________________________________________

## Branch Strategy

```
main ──●──────────────●──────────────●──────────────●──────────────●──→
       │              │              │              │              │
       Phase 1        Phase 2        Phase 3        Phase 4        Phase 5
```

Each phase lands through one or more PRs. The unit of planning is the Phase issue.

| Phase   | Issue | Goal                         |
| ------- | ----- | ---------------------------- |
| Phase 1 | TBD   | Portable training foundation |
| Phase 2 | TBD   | W&B checkpoint durability    |
| Phase 3 | TBD   | RunPod launcher              |
| Phase 4 | TBD   | Docker + CI                  |
| Phase 5 | TBD   | Documentation                |

______________________________________________________________________

## Phase 1: Portable Training Foundation

**Issue:** TBD
**Epic:** #107

### Tasks

| Task     | Description              |
| -------- | ------------------------ |
| Task 1.1 | Training config cleanup  |
| Task 1.2 | W&B config cleanup       |
| Task 1.3 | Makefile training target |
| Task 1.4 | Training smoke test      |

### Files to modify

```
configs/data/*.yaml
configs/logger/wandb.yaml
Makefile
src/train.py
```

### Files to create

```
tests/test_train_smoke.py
```

### Completion criteria

- `make train EXPERIMENT=...` works on local machine
- no `/data/scratch` paths in training configs
- W&B entity/project configurable via env

______________________________________________________________________

## Phase 2: W&B Checkpoint Durability

**Issue:** TBD
**Epic:** #107

### Tasks

| Task     | Description              |
| -------- | ------------------------ |
| Task 2.2 | Resume from W&B artifact |

### Files to modify

```
configs/logger/wandb.yaml
src/train.py
Makefile
```

### Files to create

```
src/utils/wandb_checkpoint.py
tests/test_wandb_checkpoint.py
```

### Completion criteria

- `log_model: "all"` configured; every checkpoint appears as W&B model artifact
- resume works with `ckpt_path=wandb:model-{train_config_id}:latest`
- Lightning optimizer state restored correctly

______________________________________________________________________

## Phase 3: RunPod Launcher

**Issue:** TBD
**Epic:** #107

### Tasks

| Task     | Description              |
| -------- | ------------------------ |
| Task 3.1 | RunPod training launcher |
| Task 3.2 | Resume integration       |
| Task 3.3 | Pod termination handling |

### Files to create

```
scripts/runpod_train.py
tests/test_runpod_launcher.py
```

### Completion criteria

- `make runpod-train EXPERIMENT=...` launches training pod
- launcher supports checkpoint resume
- training exits cleanly when job completes

______________________________________________________________________

## Phase 4: Docker & CI

**Issue:** TBD
**Epic:** #107

### Tasks

| Task     | Description           |
| -------- | --------------------- |
| Task 4.1 | Training Docker image |
| Task 4.2 | CI smoke test         |

### Files to create

```
docker/train/Dockerfile
.github/workflows/train-smoke.yml
tests/test_train_e2e.py
```

### Completion criteria

- docker training run succeeds
- CI fixture reaches first checkpoint
- no secrets required for smoke test

______________________________________________________________________

## Phase 5: Documentation

**Issue:** TBD
**Epic:** #107

### Tasks

| Task     | Description      |
| -------- | ---------------- |
| Task 5.1 | Training runbook |
| Task 5.2 | Resume guide     |

### Files to create

```
docs/training-runbook.md
```

______________________________________________________________________

## Estimated Change Size

| Area                    | Change                    | Lines |
| ----------------------- | ------------------------- | ----- |
| Training config cleanup | remove cluster paths      | ~5    |
| W&B config cleanup      | env-driven entity/project | ~5    |
| Resume from W&B         | artifact resolver         | ~40   |
| RunPod launcher         | new launcher script       | ~80   |
| Docker train image      | Dockerfile + make targets | ~120  |
| CI smoke test           | workflow + fixtures       | ~150  |

______________________________________________________________________
