# Design Doc: Data Pipeline

> **Status**: Draft
> **Author**: ktinubu@
> **Last Updated**: 2026-03-15

______________________________________________________________________

### Index

| §   | Section                                                                                | What it covers                                                       |
| --- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| 1   | [Context & Motivation](#1-context--motivation)                                         | Problem statement, infrastructure layers                             |
| 2   | [Typical Workflow](#2-typical-workflow)                                                | End-to-end CLI example                                               |
| 3   | [Goals, Non-Goals & Design Principles](#3-goals-non-goals--design-principles)          | Requirements, principles, anti-goals, success metrics                |
| 4   | [System Overview](#4-system-overview)                                                  | Architecture summary, data/control plane, reconciliation correctness |
| 5   | [Stage Definitions](#5-stage-definitions)                                              | Generate and finalize stages                                         |
| 6   | [Data Flow & Architecture](#6-data-flow--architecture)                                 | Diagrams, R2 layout, artifact taxonomy                               |
| 7   | [Design Decisions](#7-design-decisions)                                                | Storage-as-truth, reconciliation, concurrency, output formats        |
| 8   | [Experiment Tracking](#8-experiment-tracking-weights--biases)                          | W&B metrics, artifacts, lineage                                      |
| 9   | [Alternatives Considered](#9-alternatives-considered)                                  | Comparison chart, detailed rejections                                |
| 10  | [Operations & Infrastructure](#10-operations--infrastructure)                          | Credentials, monitoring, cost model                                  |
| 11  | [Concurrency, Consistency & Failure Modes](#11-concurrency-consistency--failure-modes) | R2 consistency, edge cases, failure analysis                         |
| 12  | [Open Questions, Risks & Limitations](#12-open-questions-risks--limitations)           | Known gaps and trade-offs                                            |
| 13  | [Out of Scope](#13-out-of-scope)                                                       | Future work — not referenced elsewhere                               |
| 14  | [Implementation Details](#14-implementation-details)                                   | Schemas, CLI structure, config materialization                       |
| A–E | [Appendices](#appendix-a-glossary)                                                     | Glossary, tech stack, references, roadmap, implementation recipes    |

______________________________________________________________________

## 1. Context & Motivation

Topline goal: Get massive dataset generation working reliably enough, and know what went wrong when there is unexpected behavior.

**synth-setter** is a machine learning research project studying how neural networks can infer synthesizer parameters from audio. The core task: given a recording of a synthesizer, predict the knob settings (parameters) that produced it.

Training these models requires large-scale datasets: 500k–1M+ audio samples, each rendered through a real VST synthesizer plugin (Surge XT) with random parameter configurations. Each sample produces an audio waveform, mel spectrogram, and ground-truth parameter array, stored as an HDF5 shard. This rendering is CPU-bound — each sample requires a real-time audio render through the plugin — and takes hours to days on a single machine.

The distributed data pipeline solves this by splitting generation across N cloud workers on **[RunPod](https://www.runpod.io/)** (a GPU/CPU cloud marketplace offering cheap on-demand compute), each independently producing shards in parallel. Workers write shards to **[Cloudflare R2](https://developers.cloudflare.com/r2/)** (an S3-compatible object storage service with free egress), which serves as both the data store and the coordination layer. A separate finalize step downloads all shards, reshards them into train/val/test splits, computes normalization statistics, registers the dataset as a **[Weights & Biases](https://wandb.ai/)** (W&B) artifact, and uploads the final dataset.

The pipeline is designed for datasets that scale to multi-terabyte sizes while keeping costs minimal — cheap compute, free egress, no infrastructure to manage.

### Infrastructure Layers

| Layer         | Technology                                                   | Role                                                       |
| ------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| **Build**     | [Docker](https://docs.docker.com/build/buildkit/) (BuildKit) | Reproducible compute environments with baked dependencies  |
| **Storage**   | [Cloudflare R2](https://developers.cloudflare.com/r2/)       | Data storage, coordination state, free egress              |
| **Execution** | [RunPod](https://www.runpod.io/)                             | Cheap on-demand cloud workers for CPU/GPU workloads        |
| **Tracking**  | [Weights & Biases](https://wandb.ai/)                        | Lightweight experiment tracking, dataset artifact registry |

RunPod is used because it's the platform where GPUs are already available and convenient — spot-like pricing, simple pod API, no cluster management. It is not a deep architectural dependency; the reconciliation model means any provider that can run a Docker container and upload to R2 is sufficient.

## 2. Typical Workflow

```bash
# 1. Create a dataset config
cat configs/pipeline/surge_simple_480k.yaml
# → experiment_name: surge_simple, num_shards: 48, shard_size: 10000, ...

# 2. Launch generation — creates spec, launches workers, exits
python -m pipeline.cli generate --config configs/pipeline/surge_simple_480k.yaml --workers 10
# → Created run surge_simple-480k-10k-20260313-100000
# → Launched 10 workers for 48 shards
# → Exiting. Run 'status' to check progress.

# 3. Check progress (can run from any machine, any time)
python -m pipeline.cli status --run-id surge_simple-480k-10k-20260313-100000
# → Valid: 44/48  Missing: 2  Quarantined: 2

# 4. Re-run generation for missing shards only
python -m pipeline.cli generate --run-id surge_simple-480k-10k-20260313-100000
# → 4 shards missing, launching 1 worker

# 5. Finalize — download, reshard, compute stats, register in W&B
python -m pipeline.cli finalize --run-id surge_simple-480k-10k-20260313-100000
# → 48/48 valid. output_format: hdf5
# → Resharding → train.h5, val.h5, test.h5  (or .tar shards if wds)
# → Stats computed. Dataset registered in W&B.
# → dataset.complete written.
```

Make targets are thin aliases for convenience:

```bash
make generate ARGS="--config configs/pipeline/surge_simple_480k.yaml --workers 10"
make status ARGS="--run-id surge_simple-480k-10k-20260313-100000"
make finalize ARGS="--run-id surge_simple-480k-10k-20260313-100000"
```

## 3. Goals, Non-Goals & Design Principles

### Goals

- **Reproducible pipeline with full provenance.** The input spec freezes all generation parameters — per-shard seeds, shapes, renderer version. Re-running from the same spec on the same hardware and Docker image produces an identical dataset. This is a *controlled-conditions* guarantee, not an absolute one: VST plugin floating-point behavior may vary across CPU architectures, and Docker base image updates could change system libraries. The pipeline records enough provenance to detect and diagnose these differences (git commit, `is_repo_dirty: bool`, renderer version, per-shard content hashes), but does not claim bit-identical output across arbitrary hardware. Provenance matters because this is ML research: when a model behaves unexpectedly, you need to trace from the trained model back to the exact dataset, the exact code that generated it, and which worker attempt produced each shard. Per-shard provenance is tracked via lifecycle markers and content hashes in worker reports.
- **Minimal hand-holding.** Two commands, no babysitting. Launch generation, come back later, run finalize. No monitoring dashboards to watch, no coordination services to keep alive, no manual intervention between steps.
- **Debugability.** When something fails, the failure is easy to find and understand. Per-shard error tracking, structured debug logs that survive crashes, reconciliation reports that show exactly which shards are missing and why. No need to dig through cloud provider consoles.
- **Low cost.** Cheap GPUs, free egress, no infrastructure to manage. A full dataset generation run costs ~$2. Monthly compute at 1-2 runs/week is ~$8-16. R2 storage accumulates but free egress makes it far cheaper than S3 for frequently-downloaded datasets.
- **Crash resilience.** The pipeline must handle errors and crashes from data generation code we don't own — SIGSEGV from the VST plugin, OOM kills, Python crashes. Per-shard isolation means one crash doesn't take down the worker. Bash EXIT traps upload logs even when the process dies. Reconciliation detects missing shards regardless of how they were lost.
- **Safe and resumable.** Every command is safe to run at any time, in any order, any number of times — no data corruption possible, the worst case is wasted compute (a redundant write replaces identical content). On retry, only missing/invalid shards are regenerated. Shards are validated (structural integrity, shape, value bounds, row count) before merging; corrupt shards are quarantined, not silently included.
- **Auth validation before compute.** Verify all credentials (R2, RunPod) before launching any workers to avoid wasting money on misconfigured runs.
- **Local compute mode.** The full pipeline must run locally (Docker containers, local filesystem instead of R2) for development, unit tests, and integration tests.

### Design Principles

- **Storage is truth** — shard completeness is determined by file existence + validation, not metadata ([§7.1](#71-storage-as-the-source-of-truth))
- **Reconciliation over orchestration** — compare desired state (spec) against actual state (validated shards) to determine remaining work ([§7.4](#74-reconciliation-based-execution))
- **Deterministic work identity** — shard IDs are logical (`shard-000042`), not tied to infrastructure ([§7.3](#73-deterministic-shard-identities))
- **Stage isolation** — each stage is an independent, reconcilable transform with well-defined inputs and outputs ([§5](#5-stage-definitions))
- **Fail visibly** — errors are captured, structured, and surfaced, never swallowed ([§7.8](#78-error-handling--crash-resilience))
- **Validate at boundaries** — data is verified when entering and leaving each stage ([§7.5](#75-shard-validation))
- **Thin abstractions** — only abstract what's needed; two compute backends, not a speculative framework ([§7.9](#79-compute-abstraction))

### What This System Deliberately Avoids

- **Consensus protocols** — one writer per shard, no conflicts
- **Distributed transactions** — stages are independent
- **Service discovery** — workers don't communicate
- **Message queues** — reconciliation-based reporting is adequate at 1-2x/week
- **Automatic stage chaining** — explicit commands are clearer at 2 stages
- **Provider job supervision** — submit work and exit; storage determines completeness
- **Speculative provider abstractions** — only local + RunPod until a third is needed
- **Owning provider observability** — provider-side monitoring is the provider's responsibility; completeness is determined from storage

### Success Metrics

| Metric                | Target                                                   | How to Measure                                                              |
| --------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------- |
| End-to-end automation | Zero manual intervention from `generate` to `finalize`   | Run completes with only two user commands                                   |
| Generation throughput | 500k samples in under 2 hours (10 workers)               | Timestamps in spec (`created_at` → `finalized_at`)                          |
| Error visibility      | 100% of shard failures include actionable error messages | Worker reports contain per-shard error details; `make status` surfaces them |
| Resumability          | Retry re-generates only missing/invalid shards           | On retry, compare missing shard count vs total                              |
| Cost per run          | < $5 for a 500k-sample dataset                           | RunPod billing + R2 storage                                                 |
| Dataset scale         | Support multi-terabyte datasets                          | Validated with 1M+ sample runs                                              |

### Non-Goals

- **Real-time streaming pipeline.** Batch system, 1-2x/week.
- **General-purpose job orchestrator.** Purpose-built for this pipeline.
- **Multi-tenant support.** Single-user research pipeline.
- **Sub-second latency.** Worker startup and R2 operations are minute-scale.
- **GPU scheduling.** Generation is CPU-bound; GPU allocation is RunPod's concern.
- **Replacing training infrastructure.** This pipeline produces datasets. Training is a separate concern.
- **Production-grade infrastructure.** This is a research pipeline optimized for simplicity and debuggability, not a production system. We accept wasted compute (redundant writes, full restarts on finalize crash) in exchange for simpler code with fewer failure modes. Automation that minimizes wasted work (auto-retry, partial checkpoints, progress-aware scheduling) adds complexity not justified at 1-2 runs/week.

### Explicit Non-Requirements

- Generic DAG parser or dependency graph executor
- Automatic stage chaining engine
- Stage plugin registry or dynamic stage discovery
- Workflow definition language or config-driven stage ordering
- Stage lifecycle hooks (pre-stage, post-stage, on-failure)
- RunPod health monitoring or observability tooling — not justified for a 1-2x/week pipeline
- Provider-agnostic abstraction beyond the two backends actually tested

## 4. System Overview

The pipeline is a batch-oriented, fully parallel data generation system built on a **reconciliation model**: inspect storage, determine what work is missing, launch only that work.

A CLI running on the user's local machine reads a spec (desired state), lists existing validated shards in R2 (actual state), computes the difference, and launches N workers to produce the missing shards. Each worker independently renders audio samples through a VST plugin and writes HDF5 shards to R2. When all shards are present, a separate finalize command reshards into train/val/test splits, computes normalization statistics, registers the dataset in W&B, and writes a completion marker.

R2 serves as both the **data plane** and the **control plane**:

| Plane             | What flows through it  | Examples                                             |
| ----------------- | ---------------------- | ---------------------------------------------------- |
| **Data plane**    | Actual dataset content | HDF5 shards, virtual datasets, stats.npz             |
| **Control plane** | Coordination metadata  | Spec, worker reports, debug logs, `dataset.complete` |

Both planes use R2. There is no separate database, message queue, or coordination service. This means one piece of infrastructure to manage, one set of credentials, one failure mode to reason about. The trade-off: R2 has no atomic test-and-set, so mutual exclusion is not possible. This is acceptable because all operations are idempotent and produce deterministic outputs ([§7.7](#77-concurrency-semantics)).

### Reconciliation Correctness

What if reconciliation itself has a bug — e.g., it validates a corrupt shard as good?

- **Defense in depth:** Workers run four independent validation checks before upload; all must pass ([§7.5](#75-shard-validation)).
- **Tiered validation:** Workers do full 4-check validation. Finalize structural-checks staged shards before promoting. Each tier catches a different class of failure.
- **Training is the ultimate check:** A corrupt dataset will fail to train properly, providing end-to-end verification.
- **Manual spot-checking is feasible:** At 1-2 runs/week, eyeballing a few shards is practical and encouraged.

## 5. Stage Definitions

The pipeline has two stages. Each is an independent command with well-defined inputs and outputs.

| Stage        | Command                 | Input                                     | Output                                                                        | Compute                        |
| ------------ | ----------------------- | ----------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------ |
| **Generate** | `pipeline.cli generate` | Config YAML (first run) or spec (retries) | HDF5 shards in staging                                                        | CPU — VST audio rendering      |
| **Finalize** | `pipeline.cli finalize` | Validated staged shards in R2             | Format-dependent (see below), `stats.npz`, `dataset.json`, `dataset.complete` | CPU — download, reshard, stats |

Finalize output depends on `output_format` in the spec:

| `output_format` | Finalize outputs                                                                 | Training access pattern            |
| --------------- | -------------------------------------------------------------------------------- | ---------------------------------- |
| `hdf5`          | `train.h5`, `val.h5`, `test.h5` (HDF5 virtual datasets)                          | Local random access                |
| `wds`           | `train-{shard}.tar`, `val-{shard}.tar`, `test-{shard}.tar` (WebDataset archives) | Sequential streaming (local or R2) |

**Stage order is static and explicit.** Generate must complete before finalize. The user runs generate, checks the report, then runs finalize. There is no automatic chaining.

**Each stage follows a contract:**

- Reads from a well-defined storage input prefix
- Writes to a well-defined storage output prefix
- Can determine its own completeness by inspecting storage
- Can be re-run at any time without affecting other stages
- Transitions are explicit (user runs the next command)

## 6. Data Flow & Architecture

### Reconciliation Flow

```
┌───────────────────────────────┐
│  make generate RUN_ID=...     │
│  (CLI — local machine)        │
│                               │         ┌────────────────┐
│  1. Validate auth (R2+RunPod) │         │  Cloudflare R2  │
│  2. Read/create spec  ◄───┼────────►│                │
│  3. List staged shards    ◄───┼─────────┤  {run_id}/     │
│  4. Validate staged shards    │         │   metadata/    │
│  5. Compute missing set       │         │   workers/     │
│  6. Partition across N workers│         │                │
│  7. Submit N tasks            │         │                │
│  8. Exit                      │         │                │
└───────────────────────────────┘         │                │
                                          │                │
         ┌───────────────────┐            │                │
         │  Worker 1         │───────────►│  metadata/     │
         │  (RunPod worker)  │            │   workers/     │
         │  shards 0-47      │            │   shards/      │
         └───────────────────┘            │                │
         ┌───────────────────┐            │                │
         │  Worker N         │───────────►│  (staging)     │
         │  shards 432+      │            │                │
         └───────────────────┘            │                │
                                          │                │
┌───────────────────────────────┐         │                │
│  make finalize RUN_ID=...     │         │                │
│  (local or cloud)             │         │                │
│                               │         │  data/         │
│  1. Read spec          ◄───┼─────────┤   train.h5     │
│  2. Validate staged shards ◄──┼─────────┤   val.h5       │
│  3. Promote to data/shards ───┼─────────►   test.h5      │
│  4. Download canonical shards ┼─────────┤   stats.npz    │
│  5. Reshard → train/val/test  │         │   dataset.json │
│  6. Compute stats             │         │   dataset.complete│
│  7. Register in W&B      ────┼──┐      │                │
│  8. Upload finalized      ────┼──┼─────►│                │
│  9. Write dataset.complete ───┼──┘      │                │
└───────────────────────────────┘         └────────────────┘
```

### R2 File Structure

```
{run_id}/                              # e.g. surge_simple-480k-10k-20260312-143022
  data/
    shards/                              # Written ONLY by finalize (promoted from staging)
      shard-000000.h5                    # Canonical finalized shards
      shard-000001.h5
      ...
      shard-000479.h5
    # output_format: hdf5
    train.h5                             # Virtual dataset (written by finalize)
    val.h5
    test.h5
    # output_format: wds
    train-000000.tar                     # WebDataset archives (written by finalize)
    train-000001.tar
    ...
    val-000000.tar
    test-000000.tar
    stats.npz                            # Normalization statistics
  metadata/
    config.yaml                          # User recipe (provenance copy, not authoritative)
    input_spec.json                      # Frozen input specification (authoritative)
    dataset.json                         # Self-describing dataset card (written by finalize)
    dataset.complete                     # Completion marker (written by finalize)
    workers/                             # Everything workers produce goes here
      shards/                            # Per-shard staging area + lifecycle markers
        shard-000000/
          {worker_id}-{attempt_uuid}.h5          # Worker's validated shard output
          {worker_id}-{attempt_uuid}.valid       # Commit marker: staged shard committed
        shard-000042/
          {worker_id}-{attempt_uuid}.h5          # Second attempt's shard output
          {worker_id}-{attempt_uuid}.rendering   # First attempt: started but crashed
          {worker_id}-{attempt_uuid}.valid       # Second attempt: succeeded
          quarantine/
            {worker_id}-{attempt_uuid}.h5        # Corrupt version, preserved for debugging
      attempts/                           # Per-attempt worker artifacts
        {worker_id}-{attempt_uuid}/
          report.json                    # Worker summary — per-shard results, content_hash, timing
          debug.log                      # Debug log (JSONL), uploaded by EXIT trap
```

### Artifact Taxonomy

All structured files in the pipeline, in one place:

```
                     ┌──────────────┐
                     │  config.yaml  │ ─── User-authored recipe
                     │  (user input) │     Human-written YAML
                     └──────┬───────┘
                            │
                   pipeline.cli generate  ← creates on first run
                            │
                     ┌──────▼───────┐
                     │  input_spec.json    │ ─── Frozen input specification
                     │  (immutable)  │     Machine-generated, write-once
                     └──────┬───────┘
                            │
                   Workers (RunPod / local)  ← submitted by generate
                            │
              ┌─────────────┼─────────────┐
              │             │             │
       ┌──────▼──────┐  ┌──▼──────────┐  ┌▼──────────────┐
       │ {w}-{a}.h5   │  │ report.json  │  │ debug.log     │
       │ (staged      │  │ (worker      │  │ (worker       │
       │  shard)      │  │  summary)    │  │  debug log)   │
       └──────┬───────┘  └─────────────┘  └───────────────┘
              │
       ┌──────▼──────┐   All worker output → metadata/workers/
       │ .rendering   │
       │ .valid       │
       │ .invalid     │
       │ (lifecycle)  │
       └──────┬───────┘
              │
     pipeline.cli finalize  ← validates + promotes staged shards
              │
       ┌──────▼───────┐
       │shard-{id}.h5  │ ─── Promoted to data/shards/ (canonical)
       │(finalized)    │     Written ONLY by finalize
       └──────┬───────┘
              │
       ┌──────▼───────┐
       │ dataset.json  │ ─── Output record (dataset card)
       │ (output)      │     What was produced, how to use it
       └──────┬───────┘
              │
       ┌──────▼────────┐
       │dataset.complete│ ─── Completion marker
       │ (marker)       │     "Finalization is done"
       └───────────────┘
```

| Artifact          | Path                                                                  | Format     | Produced By                     | Consumed By                        |
| ----------------- | --------------------------------------------------------------------- | ---------- | ------------------------------- | ---------------------------------- |
| Config            | `metadata/config.yaml`                                                | YAML       | User                            | `generate`                         |
| Input spec        | `metadata/input_spec.json`                                            | JSON       | `generate` (first run)          | `generate`, `status`, `finalize`   |
| Staged shard      | `metadata/workers/shards/shard-{id}/{worker}-{attempt}.h5`            | HDF5       | Workers                         | `finalize` (promotes to canonical) |
| Canonical shard   | `data/shards/shard-{id}.h5`                                           | HDF5       | `finalize` (copy from staging)  | Training scripts                   |
| Lifecycle marker  | `metadata/workers/shards/shard-{id}/{worker}-{attempt}.{state}`       | Empty file | Workers / Finalize              | `status`, humans                   |
| Quarantined shard | `metadata/workers/shards/shard-{id}/quarantine/{worker}-{attempt}.h5` | HDF5       | Workers (on validation failure) | Humans (debugging)                 |
| Worker report     | `metadata/workers/attempts/{worker_id}-{attempt}/report.json`         | JSON       | Workers (at exit)               | `status`                           |
| Debug log         | `metadata/workers/attempts/{worker_id}-{attempt}/debug.log`           | JSONL      | Workers (continuous)            | Humans (`jq`)                      |
| Dataset card      | `metadata/dataset.json`                                               | JSON       | `finalize`                      | Training scripts, humans           |
| Completion marker | `metadata/dataset.complete`                                           | JSON       | `finalize` (last step)          | `finalize` (idempotency check)     |
| Stats             | `data/stats.npz`                                                      | NumPy      | `finalize`                      | Training scripts                   |

**Shard attempt lifecycle:** Each attempt produces a shard file and lifecycle markers in the shard's staging directory, all named `{worker_id}-{attempt_uuid}`:

| File / Marker                  | Written by                                                               | Meaning                                                                                                                                                                                                                                                                    |
| ------------------------------ | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `{worker}-{attempt}.h5`        | Worker (after local validation)                                          | The worker's validated shard output. Sits alongside its lifecycle markers. Multiple attempts for the same shard are visible by name.                                                                                                                                       |
| `{worker}-{attempt}.rendering` | Worker (at start)                                                        | Attempt started. Append-only — not deleted when `.valid` is written. Orphaned `.rendering` without a `.valid` indicates a crashed attempt.                                                                                                                                 |
| `{worker}-{attempt}.valid`     | Worker (last step of shard lifecycle)                                    | **Commit point for staged shard.** Written only after render, validation, upload, and bookkeeping are complete. `generate`/`status` uses this as the staging admission signal. Not sufficient for final dataset correctness — finalize structural-checks before promoting. |
| `{worker}-{attempt}.invalid`   | Worker (on validation failure) or Finalize (on structural check failure) | Shard failed validation. Corrupt shard uploaded to `quarantine/` for debugging.                                                                                                                                                                                            |
| `{worker}-{attempt}.promoted`  | Finalize                                                                 | Staged shard was structural-checked and promoted to `data/shards/shard-{id}.h5`. Content hash recorded in `dataset.json`.                                                                                                                                                  |

Listing a shard's staging directory shows the full history — shard files, lifecycle markers, and quarantined attempts — at a glance:

```
$ rclone ls r2:bucket/{run_id}/metadata/workers/shards/shard-000042/
         0  pod-abc123-a1b2c3d4.rendering   # first attempt started (crashed — no .valid)
  67108864  pod-def456-e5f6a7b8.h5          # second attempt's shard
         0  pod-def456-e5f6a7b8.rendering   # second attempt started
         0  pod-def456-e5f6a7b8.valid       # second attempt committed (staged-valid)
         0  pod-def456-e5f6a7b8.promoted    # promoted to data/shards/ by finalize
```

**Naming conventions:** Config is YAML (human-authored). Everything machine-generated is JSON. Debug logs are JSONL (one event per line). Data is HDF5. All worker-produced files live under `metadata/workers/` — shards and markers grouped by shard ID (`metadata/workers/shards/shard-{id}/`), worker artifacts grouped by attempt (`metadata/workers/attempts/{worker_id}-{attempt_uuid}/`). The `data/` prefix is written only by finalize. Lifecycle markers are empty files — presence is the state, no content to parse.

## 7. Design Decisions

### 7.1 Storage as the Source of Truth

The pipeline uses R2 as both the data layer and the coordination layer. Integrity is guaranteed by content hashes. Workers write shard files and markers to a **staging prefix** (`metadata/workers/shards/`). Finalize validates staged shards and **promotes** them to the **canonical prefix** (`data/shards/`). This separation ensures workers never write to the canonical data path, and finalized data is stable once promoted.

**Why R2 for coordination and not a database or queue:**

Object storage lacks atomic compare-and-set, locking, and transactions. A traditional coordination system (Redis, Postgres, SQS) would provide these. We use R2 anyway — this is a deliberate trade-off, not a convenience choice.

The pipeline doesn't need what coordination systems provide:

- **No atomic compare-and-set** — workers write to per-attempt filenames in a staging prefix, so concurrent writes never collide. Finalize is the only writer to canonical paths ([§7.7](#77-concurrency-semantics)).
- **No locking** — each shard is assigned to one worker per invocation. The assignment is a simple partition of the missing set. No lock acquisition needed.
- **No transactions** — stages are independent. There is no multi-shard operation that must succeed or fail atomically.
- **No queue** — work discovery is reconciliation (compare spec against storage). A queue would be a second source of truth for "what work remains" — and a less reliable one than the files themselves.

What a coordination system *would* cost:

- **Infrastructure to manage.** Redis/Postgres must be provisioned, monitored, backed up, and secured. For a pipeline that runs 1-2x/week, this is disproportionate.
- **A second failure mode.** If the coordination system is down, the pipeline can't run — even though R2 (where the actual data lives) is fine.
- **Split-brain risk.** Coordination system says "shard-042 is complete" but the file is missing from R2. Now you have two sources of truth that disagree. The current design has one source of truth: the files.

What R2 provides instead:

- Workers already upload shards to R2, so the coordination write path is free
- R2 state survives worker termination and cleanup
- S3-compatible — coordination layer is portable to any cloud
- Free egress — datasets are frequently downloaded for training
- Files are human-readable and inspectable (`rclone cat` + `jq`)
- [Strong read-after-write consistency](https://developers.cloudflare.com/r2/reference/consistency/) — no stale reads

The patterns that make R2-as-coordination safe despite no atomicity:

- **Deterministic shard IDs** — canonical paths are known from the spec, no claiming needed ([§7.3](#73-deterministic-shard-identities))
- **Idempotent operations** — every command is safe to re-run
- **Append-only metadata** — lifecycle markers and worker reports use unique filenames, never overwritten ([§7.2](#72-shard-lifecycle))
- **Reconciliation from storage** — the pipeline re-derives state from files on every invocation, never caches coordination state locally

**State model — pipeline state is determined from two prefixes:**

1. **Spec** — desired state: which shards should exist
2. **Staging prefix** (`metadata/workers/shards/`) — which shards workers have produced and validated. This is where `generate` and `status` look.
3. **Canonical prefix** (`data/shards/`) — which shards finalize has promoted. Written only by finalize. Once promoted, zombie worker uploads to staging cannot affect the finalized data.
4. **Worker metadata** (reports, logs) — debugging hints, never authoritative, never block completion

This separation eliminates an entire class of edge cases:

- Worker crashes before upload → no staged file → reconciliation detects it
- Worker report claims success but upload failed → no staged file → reconciliation detects it
- Staged shard exists but report is missing → shard passes validation → it's complete
- Zombie worker uploads after finalize → goes to staging, doesn't touch canonical `data/shards/`

### 7.2 Shard Lifecycle

> **Completeness rule:** A shard is **staged-valid** (ready for finalization) if both a `.h5` file and a `.valid` marker exist in `metadata/workers/shards/shard-{id}/`. The `.valid` marker is the **commit point** for a staged shard — it means the worker completed rendering, validation, upload, and bookkeeping successfully. It is authoritative for staging admission but not for final dataset correctness; finalize remains the gate before promotion ([§7.5](#75-shard-validation)). A shard is **finalized** if it has been promoted to `data/shards/shard-{id}.h5` by finalize.

The lifecycle has three commit points — each marks the completion of a distinct phase:

| Marker       | Commit point              | Written by |
| ------------ | ------------------------- | ---------- |
| `.rendering` | Attempt started           | Worker     |
| `.valid`     | Staged shard committed    | Worker     |
| `.promoted`  | Canonical shard committed | Finalize   |

```
missing → rendering → staged-valid → finalized (canonical)
               ↓
            invalid (quarantined)
```

| State            | How it's determined                                                                                                            | Where it lives                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| **missing**      | Shard is in the input spec but no `.valid` marker exists for any attempt                                                       | Implicit (absence). May have orphaned `.rendering` markers from crashed attempts |
| **rendering**    | `.rendering` marker exists but no `.valid` marker yet — attempt is in progress                                                 | `.rendering` marker in `metadata/workers/shards/shard-{id}/`                     |
| **staged-valid** | `.h5` + `.valid` exist. Worker completed full lifecycle (render, validate, upload, bookkeeping). Ready for finalize to promote | `metadata/workers/shards/shard-{id}/{worker_id}-{attempt}.h5` + `.valid` marker  |
| **invalid**      | Worker validation failed. Corrupt file uploaded to `quarantine/` for debugging                                                 | `.invalid` marker + `quarantine/{worker_id}-{attempt}.h5`                        |
| **finalized**    | Finalize structural-checked and promoted the shard to `data/shards/shard-{id}.h5`. Dataset is sealed                           | `data/shards/shard-{id}.h5` + `.promoted` marker + `dataset.complete`            |

**Transitions:**

- **missing → rendering:** Worker begins shard generation, writes `.rendering` marker.
- **rendering → staged-valid:** Worker validates locally (full 4-check), uploads `.h5` to staging, writes worker report, then writes `.valid` marker as the **last step**. The `.valid` marker is the commit point — it signals that the worker completed the full shard lifecycle (render, validate, upload, bookkeeping). The `.rendering` marker is not deleted — both remain visible, preserving the full timeline.
- **rendering → invalid:** Worker validates locally and the shard fails. Worker uploads the corrupt shard to `quarantine/` and writes `.invalid` marker, preserving the evidence for debugging. The shard is treated as missing on next `generate`.
- **rendering → missing:** Worker crashes before writing `.valid`. The `.rendering` marker is orphaned — observable evidence of the crashed attempt. Any `.h5` uploaded before the crash exists but without `.valid` is not considered staged-valid.
- **staged-valid → finalized:** Finalize structural-checks the staged shard, copies it to `data/shards/shard-{id}.h5`, writes `.promoted` marker, records content hash in `dataset.json`, and writes `dataset.complete` after all shards are promoted. Staged files remain in place (append-only — no deletion).

**Key properties:**

- **`.valid` is authoritative for staging admission, not for final correctness.** `generate`/`status` trusts `.valid` markers for cheap reconciliation. Finalize does a structural check before promoting — it is the gate for canonical data. Canonical truth lives in `data/shards/`, not in `.valid` markers.
- **Multiple attempts are visible.** A shard's staging directory might contain `pod-abc.rendering` (crashed, no `.valid`), `pod-def.h5` + `pod-def.valid` (committed), and `pod-def.promoted` (finalized) — the full history is one `rclone ls` away.
- **Workers and finalize write to separate prefixes.** Workers only write under `metadata/workers/`. Finalize only writes to `data/` (plus `.promoted` markers). A zombie worker uploading after finalize cannot overwrite canonical data.
- **The finalized state is per-shard and dataset-level.** Per-shard: `data/shards/shard-{id}.h5` + `.promoted` marker. Dataset-level: `dataset.complete` marker and content hashes in `dataset.json`.

### 7.3 Deterministic Shard Identities

Shard IDs are logical and deterministic: `shard-000000.h5` through `shard-000479.h5`. Defined at run creation, independent of which worker computes them.

- **Any worker can compute any shard** — retries simply recompute the same logical shard
- **Resumability is a set difference:** `spec_shards - validated_shards = work_remaining`
- **No naming collisions** — each worker attempt writes to a unique filename (`{worker_id}-{attempt_uuid}.h5`), and the canonical path (`data/shards/shard-{id}.h5`) is written only by finalize
- **Infrastructure details** (worker IDs) appear in staging filenames and metadata, not in canonical shard paths

**Work assignment:** The CLI partitions shards across N workers. Worker 1 gets shards 0-47, Worker 2 gets shards 48-95, etc. But the shard's identity is independent of which worker computes it. If Worker 1 fails and its shards are reassigned to Worker 3, output paths are unchanged.

**Shard write protocol:**

1. Write `.rendering` marker: `metadata/workers/shards/shard-{id}/{worker_id}-{attempt_uuid}.rendering`
2. Render shard to a local temp file
3. **Validate locally** — full 4-check validation (structural, shape, value, row count). This is the primary defense against corrupt data.
4. **If validation passes:**
   - Upload shard to staging: `metadata/workers/shards/shard-{id}/{worker_id}-{attempt_uuid}.h5`
   - Write worker report (content hash, timing, per-shard results): `metadata/workers/attempts/{worker_id}-{attempt_uuid}/report.json`
   - **Write `.valid` marker** (last step — the commit point): `metadata/workers/shards/shard-{id}/{worker_id}-{attempt_uuid}.valid`
5. **If validation fails:**
   - Upload shard to quarantine: `metadata/workers/shards/shard-{id}/quarantine/{worker_id}-{attempt_uuid}.h5`
   - Write `.invalid` marker: `metadata/workers/shards/shard-{id}/{worker_id}-{attempt_uuid}.invalid`
   - Write worker report with error details
   - Log the failure (which check failed, values found)

The `.valid` marker is written **only after** the worker has completed rendering, validation, upload, and bookkeeping. It is the commit point for a staged shard. Worker reports and debug logs are auxiliary metadata — they are not part of the shard admission protocol.

Workers never write to `data/shards/`. The canonical path `data/shards/shard-{id}.h5` is written only by finalize during promotion ([§7.6](#76-finalize-workflow)).

> **Invariant:** Only worker-validated shards reach the staging path, and only committed shards (with `.valid` markers) are visible to `generate`/`status`. Corrupt renders are uploaded directly to quarantine, preserving the evidence for debugging while keeping the staging area clean.

### 7.4 Reconciliation-Based Execution

Instead of tracking worker state or polling provider APIs, the pipeline determines what work remains by inspecting storage.

**`generate` reconciliation:**

1. Read spec from R2 (or create if first run)
2. List staged shards in `metadata/workers/shards/` — check for `.h5` + `.valid` marker per shard (no data loading, no re-validation — [§7.5](#75-shard-validation))
3. Compute `missing = spec_shards - staged_valid_shards`
4. If nothing missing → "generation complete", exit 0
5. Partition missing shards across N workers
6. Submit N tasks, exit

**`finalize` reconciliation:**

1. Read spec from R2
2. Check for `dataset.complete` — if present and all canonical outputs exist, exit 0 ("already finalized")
3. List staged shards — check for `.h5` + `.valid` marker per shard
4. If any shards missing → report which ones, exit 1
5. Structural-check each staged shard (valid HDF5, datasets present, shapes match — [§7.5](#75-shard-validation))
6. Promote validated staged shards to `data/shards/`, write `.promoted` marker per shard
7. Download canonical shards, reshard, compute stats, register in W&B, upload, write `dataset.complete`

**Key properties:**

- **Safe at any time.** Running `generate` when all shards exist is a no-op. Running `finalize` when 3 shards are missing reports the gap and exits.
- **Machine-independent.** Authoritative state lives in R2. If the laptop that launched the run dies, any machine can continue.
- **Phase separation.** Generation and finalization are independent steps. No idle worker waiting for shards. No implicit coordination.

**`make status` — reconciliation report:**

`make status` runs the same reconciliation logic as `generate` but only prints the result. It checks for `.h5` + `.valid` marker existence — no data loading or re-validation. It does not query RunPod, check worker health, or monitor live tasks. The output is fully determined by storage contents — running it from any machine, at any time, produces the same result.

```
$ python -m pipeline.cli status --run-id surge_simple-480k-10k-20260313-100000

Run: surge_simple-480k-10k-20260313-100000
Spec shards: 48
Staged (valid):   44
Missing:           2
Quarantined:       2
Finalized:         0
Worker reports:    9

Missing:
  shard-000005
  shard-000019

Quarantined:
  shard-000006  NaN values (worker-quarantined)
  shard-000023  row count mismatch (worker-quarantined)

Recent worker errors (from metadata):
  worker-abc123: shard-000006 failed: NaN in audio buffer
  worker-def456: shard-000019 upload timeout
```

Worker error details are overlaid from metadata files when present. The core output (staged/missing/quarantined/finalized counts) is derived from file existence and markers, not from data validation.

### 7.5 Shard Validation

Validation is **tiered** — each stage does the minimum work needed for its role, avoiding redundant re-validation of shards that workers already checked.

**Full validation (4 checks)** — run by workers before upload:

- **Structural**: Valid HDF5, expected datasets present (`audio`, `mel_spec`, `param_array`)
- **Shape**: Array dimensions match spec (sample rate, spectrogram bins, parameter count)
- **Value**: No NaN/Inf values, audio within [-1, 1], parameters within spec bounds
- **Row count**: Matches spec's expected shard size

**Existence check** — run by `generate`/`status` during reconciliation:

- Check for `.h5` + `.valid` marker in staging directory. No data loading.
- The `.valid` marker is authoritative for staging admission: it means a worker completed the full shard lifecycle and committed the result ([§7.2](#72-shard-lifecycle)). It is not sufficient for final dataset correctness — finalize remains the gate before promotion.
- The trust chain justifies this: workers do full 4-check validation before upload, `rclone --checksum` verifies transfer integrity, and R2 PUTs are atomic (the object either exists completely or not). Re-validating hundreds of shards to find a few missing ones is wasted work.

**Structural check** — run by finalize before promoting staged shards to `data/shards/`:

- Valid HDF5 file that opens with `h5py`, expected datasets present, shapes match spec.
- This catches the only realistic failure between worker validation and finalize: transfer corruption or bit rot. Value-level corruption (NaN, wrong bounds) and row count mismatches were already caught by workers — re-checking would require loading all data from every shard, which is redundant.
- If a staged shard fails the structural check, finalize writes `.invalid` for that attempt, reports the failure, and exits 1. The shard is treated as missing and regenerated on next `generate`.

| Stage               | Validation                                              | Cost                                   | Why                                                                 |
| ------------------- | ------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------- |
| **Worker**          | Full 4-check (structural, shape, value, row count)      | Expensive (loads all data)             | Primary defense — catches VST crashes, NaN, wrong shapes            |
| **Generate/status** | Existence (`.h5` + `.valid` marker)                     | Cheap (file listing)                   | Workers already validated; re-validation is redundant               |
| **Finalize**        | Structural (valid HDF5, datasets present, shapes match) | Moderate (opens file, no data loading) | Catches transfer corruption; last checkpoint before sealing dataset |

**Content hashes** (SHA-256 over the full HDF5 file) are recorded in worker reports for provenance and divergence detection. They are not used as acceptance criteria. If two workers produce different hashes for the same shard, the content hashes surface the divergence for investigation.

**Semantic corruption:** Validation catches structural and numerical issues but cannot detect all semantic corruption (e.g., audio that is valid float32 in [-1, 1] but sounds wrong due to a renderer bug). Training is the ultimate semantic check ([§4](#4-system-overview), "Reconciliation Correctness"). At 1-2 runs/week, manual spot-checking of a few samples is practical and encouraged.

**Quarantine:** Workers that fail local validation upload the corrupt shard directly to `metadata/workers/shards/shard-{id}/quarantine/{worker}-{attempt}.h5` with an `.invalid` marker, preserving the evidence for debugging. `generate`/`status` sees the shard as missing (no `.valid` marker) and assigns it on the next run.

### 7.6 Finalize Workflow

01. **Check for `dataset.complete`** — if present and all canonical outputs exist, print "already finalized" and exit 0
02. **Read spec** from R2
03. **Check completeness** — list staged shards, check for `.h5` + `.valid` marker per shard. If any missing, report which ones and exit 1
04. **Select and structural-check staged shards** — for each shard, if multiple staged attempts exist, select the one with the lexicographically smallest `{worker_id}-{attempt_uuid}` filename. Deterministic selection avoids dependence on clock accuracy or storage timestamp behavior. Open each selected shard with `h5py`, verify expected datasets present and shapes match spec. No data loading. If any fail, write `.invalid` marker, report the failure, and exit 1 (shard is treated as missing on next `generate`)
05. **Promote staged shards** — copy each selected shard from `metadata/workers/shards/shard-{id}/{worker}-{attempt}.h5` to `data/shards/shard-{id}.h5`. Write `.promoted` marker for each. Staged files are not deleted (append-only)
06. **Download canonical shards** from `data/shards/` to local storage
07. **Compute normalization statistics** (mean, std across training set)
08. **Produce training outputs** — format depends on `output_format` in the spec:
    - `hdf5`: Reshard into `train.h5`, `val.h5`, `test.h5` (HDF5 virtual datasets). Good for local single-GPU training.
    - `wds`: Transcode into `train-{shard}.tar`, `val-{shard}.tar`, `test-{shard}.tar` (WebDataset archives). Each `.tar` shard contains samples as `{sample_id}.audio.npy` + `{sample_id}.params.npy` + `{sample_id}.mel.npy`. Good for multi-GPU streaming from R2.
09. **Write `dataset.json`** — self-describing dataset card (includes content hashes, output format, shard manifest)
10. **Register dataset in W&B** — log as artifact with spec, card, and metrics (§8)
11. **Upload finalized dataset** to R2
12. **Write `dataset.complete`** — completion marker (last step)

**`dataset.complete` semantics:**

`dataset.complete` means **finalization is done**. It is not a mutex, not an in-progress marker, and does not provide mutual exclusion.

- Written as the very last step, after all outputs are uploaded and verified
- Contains: `run_id` and finalization timestamp
- If `dataset.complete` exists and all outputs validate → dataset is ready for training
- If `dataset.complete` exists but outputs are missing → stale marker from a crashed finalize, cleaned up on next run
- Two concurrent finalize processes both write `dataset.complete` — this is fine, they produce identical outputs ([§7.7](#77-concurrency-semantics))

**Why `dataset.complete` and not `dataset.lock`:** The file is a completion marker, not a lock. Calling it `.lock` implies mutex semantics that don't exist and can't exist (R2 has no atomic test-and-set). The name should communicate what it means: finalization is complete.

**Finalize idempotency:** Finalize reruns from scratch unless `dataset.complete` plus all finalized outputs are present and valid. No partial checkpoints — if finalize crashes after `train.h5` but before `stats.npz`, the next run starts over. This is simple and correct: finalize processes data already in R2, so reruns are cheap (minutes).

**Canonical data immutability:** After finalize writes `dataset.complete`, the contents of `data/shards/` and the finalized outputs are considered immutable. The pipeline does not enforce this at the storage level (R2 has no object locking), but no pipeline command modifies canonical data after finalization. Manual modification of `data/shards/` after finalize invalidates the dataset hash and provenance chain.

**Quarantine cleanup:** Quarantined shards accumulate across retries. After `dataset.complete` is written, finalize can optionally delete `quarantine/` contents for completed shards (`--keep-quarantine-days` controls retention). Default: keep all quarantined files.

**`dataset.json` — dataset card:**

The output artifact metadata — a self-describing card for the finalized dataset. It answers "what is this dataset and how do I use it?" without requiring access to the metadata directory.

The input spec defines what the run should produce. `dataset.json` is the *output* record (what was actually produced). The spec has hundreds of shard-level entries. `dataset.json` inlines only what someone needs to load and use the dataset, and references the spec by SHA-256.

**Inlined:** provenance (code version, git dirty, param spec, renderer version, output format), structure (splits, total samples), stats (normalization values), validation summary.
**Referenced:** full spec via `input_spec_sha256` and `input_spec_path`.
**Excluded:** worker reports and debug logs — these are process artifacts, not dataset metadata.

### 7.7 Concurrency Semantics

This is a single-user research pipeline running 1-2x/week. It is not designed for concurrent operation, but it is **safe** under concurrent operation. Nothing gets corrupted — you just waste compute.

**Why concurrent operations can't corrupt data:**

1. **Workers write to per-attempt filenames.** Each attempt uploads to `{worker_id}-{attempt_uuid}.h5` — unique per invocation. Two workers computing the same shard write to different files in the same staging directory. No overwrites, no races.
2. **Finalize is the only writer to canonical paths.** `data/shards/shard-{id}.h5` is written only by finalize, which picks a single validated staged shard to promote. Zombie workers uploading to staging after finalize has run cannot affect the canonical data.
3. **Deterministic outputs within the same execution environment.** Two workers computing the same shard (same seed, same config, same Docker image, same CPU architecture) produce identical content. Non-determinism across hardware is detectable via content hashes.

**Scenario: `generate` run on the same run_id multiple times in quick succession**

Both invocations read the staging prefix, both see the same missing shards, both launch workers for the same shards. Two workers both generate shard-042:

1. Worker A uploads `metadata/workers/shards/shard-000042/pod-abc-uuid1.h5`
2. Worker B uploads `metadata/workers/shards/shard-000042/pod-def-uuid2.h5`
3. Both files coexist — different filenames, no overwrite
4. **Result:** two valid staged shards, wasted compute. Finalize picks the lexicographically smallest filename to promote.

**Skip-if-valid optimization:** Workers check the staging directory for an existing valid shard before uploading. If one exists, the worker skips the upload and moves to the next shard. This is an optimization, not a correctness requirement — the staging model is safe even without it.

**Scenario: concurrent `finalize` on the same run_id**

Two finalize invocations both read the input spec, validate staged shards, promote to `data/shards/`, download, reshard, upload, and write `dataset.complete`. Both pick the same staged shards (or equivalent valid ones) and produce identical canonical outputs. `dataset.complete` does not provide mutex semantics — R2 has no atomic test-and-set. The marker's purpose is to let subsequent invocations skip finalization ("already finalized"), not to prevent concurrent finalization. **Result:** identical outputs, wasted compute.

**Scenario: accidentally-launched finalize while another finalize is running**

Same as above. Both produce identical outputs. The second finalize either:

- Sees `dataset.complete` from the first (if it finished) → exits with "already finalized"
- Doesn't see it yet → runs to completion independently, produces identical outputs

No data corruption either way.

**Scenario: `generate` while `finalize` is running**

Finalize takes a snapshot of staged shard state during its validation pass. If generate launches new workers that upload to the staging prefix during finalize, those uploads don't affect the canonical `data/shards/` prefix that finalize writes to. Neither case produces a corrupt dataset.

**Scenario: `finalize` while workers are still uploading**

Finalize checks completeness by validating the staging prefix first. If shards are missing, it reports "generation incomplete" and exits 1. No partial dataset is produced.

**Scenario: zombie worker uploads after finalize completes**

A worker from a previous `generate` invocation hangs for hours, then finally uploads its shard to the staging prefix. This upload lands in `metadata/workers/shards/`, not in `data/shards/`. The canonical finalized data is unaffected. The zombie's staged shard is visible but harmless — it's simply an additional attempt record. See [§11.2](#112-failure-modes--edge-cases) for detailed analysis.

**What this system does NOT protect against:**

- **Non-deterministic outputs across hardware.** If floating-point non-determinism across different CPU architectures produces different audio for the same seed, multiple staged shards for the same shard ID may differ. Finalize picks the lexicographically smallest attempt — the selection is deterministic, but content may vary across heterogeneous environments. Content hashes and `cpu_arch` in worker reports detect this divergence. The mitigation is to fix non-determinism in the renderer or constrain the execution environment.
- **Concurrent spec modification.** The spec is written once and never modified. If something modifies it after creation, correctness guarantees do not hold.

> **Scope of concurrency safety:** The safety arguments in this section assume deterministic rendering within the execution environment (same Docker image + CPU architecture). The pipeline does not enforce homogeneous worker hardware — it detects but does not prevent architectural divergence. Workers record `cpu_arch` and `os_info` in their reports; when content hashes diverge across attempts, these fields identify the source. `dataset.json` records all unique worker architectures encountered (`worker_architectures`); if multiple architectures appear, finalize logs a warning. For bit-reproducible runs, pin RunPod instance types to a consistent CPU architecture. The mitigation for divergence is to fix the renderer or constrain the environment, not to add locking.

### 7.8 Error Handling & Crash Resilience

The pipeline handles three categories of failure, including crashes from data generation code we don't own (the Surge XT VST plugin can SIGSEGV, OOM, or produce corrupt output).

**Layer 1 — Per-shard isolation in Python:**
Each shard generation is wrapped in try/except. A crash or error in one shard is logged and the worker moves to the next shard. The worker report accumulates per-shard results. This isolates VST plugin crashes, rendering errors, and validation failures.

**Layer 2 — Entrypoint crash trap:**
A bash EXIT trap uploads the debug log and a fallback error JSON to R2 if the Python process dies entirely (OOM kill, SIGSEGV, import error). The debug log captures everything up to the crash. Even if no worker report is written, the log survives.

**Layer 3 — Reconciliation fills gaps:**
Regardless of how a shard was lost (crash, timeout, upload failure, corrupt output), reconciliation detects it as missing. The next `generate` invocation launches workers for exactly the missing shards. No manual investigation of failure modes is required to resume.

**Error tracking artifacts:**
Each worker invocation produces three artifacts, all with unique filenames keyed by `{worker_id}-{attempt_uuid}`:

- **Staged shard + lifecycle markers** (`workers/shards/shard-{id}/{worker_id}-{attempt}.h5` / `.rendering` / `.valid`) — shard file and empty markers tracking attempt state. Orphaned `.rendering` without `.valid` = crashed attempt.
- **Worker report** (`workers/attempts/{worker_id}-{attempt}/report.json`, JSON) — derived summary with content hashes, written at end of execution, missing if worker crashed
- **Debug log** (`workers/attempts/{worker_id}-{attempt}/debug.log`, JSONL) — append-only narrative, uploaded by EXIT trap, survives crashes

All worker artifacts live under `metadata/workers/`. Unique filenames per attempt mean retries never overwrite previous artifacts. Missing worker metadata never blocks completion. A run is successful when all staged shard files exist and validate.

### 7.9 Compute Abstraction

The CLI is not a job supervisor. It submits work and exits. Completeness is determined solely by validated outputs in storage, never by polling a provider API.

The compute interface has one method: submit work.

```python
class ComputeBackend(Protocol):
    def submit(self, image: str, task_specs: list[TaskSpec]) -> list[SubmittedTask]: ...
```

**Two implementations:**

- **RunPodBackend**: Production. Wraps the `runpod` Python SDK. Maps tasks to workers (RunPod calls these "pods").
- **LocalBackend**: Development and testing. Launches Docker containers locally. Uses local filesystem as the "R2" equivalent — same directory structure, same spec format, same validation logic.

No `check_tasks` method exists. Provider APIs answer the wrong question ("is the worker running?") when the right question is "are the shards done?" Storage answers that definitively.

**Local mode fidelity:** Local mode mimics R2 exactly — same directory structure, same spec format, same shard naming, same validation function. Only the storage transport changes.

**RunPod instance tagging:** The CLI tags all RunPod instances with the `run_id` at launch. A `pipeline.cli cleanup --run-id <id>` command queries the RunPod API for any pods matching that `run_id` and terminates them — a safety net for orphaned pods if the CLI crashes after launching workers but before logging pod IDs locally.

### 7.10 Output Format: HDF5 vs WebDataset

The pipeline supports two output formats, selected via `output_format` in the config and frozen in the spec. The format determines what finalize produces and how training consumes the data. Generation is unaffected — workers always produce HDF5 shards regardless of output format.

**Why two formats:**

- **HDF5** (`output_format: hdf5`): Virtual datasets (`train.h5`, `val.h5`, `test.h5`) that reference promoted shards. Good for local single-GPU training where the full dataset is downloaded to the training machine. Random access, fast local I/O.
- **WebDataset** (`output_format: wds`): Sequential `.tar` archives (`train-{shard}.tar`, etc.) optimized for streaming. Each archive contains N samples as individual NumPy files. Good for multi-GPU training (B200s, many GPUs) where streaming from R2 avoids downloading the full dataset to every node.

**Why HDF5 is insufficient for multi-GPU training:**

HDF5 is random-access oriented. Multi-GPU DataLoaders need to stream shards sequentially without coordinating seeks across workers. Streaming HDF5 virtual datasets from R2 during training creates heavy seek traffic and GPU idle time. Downloading the full dataset to every training node wastes storage and time at scale. WebDataset solves both — each `.tar` shard is a sequential stream that can be read over HTTP with near-zero overhead.

**Why generation stays HDF5 regardless of output format:**

Workers generate HDF5 because it is the right format for atomic writes, random-access validation, and debugging. The staging/canonical split is unaffected by output format. WebDataset is a training distribution format, not a generation format. Finalize handles the transcoding — it already downloads, validates, and reshards, so adding a format conversion step is a natural extension.

**WebDataset shard structure:**

Each `.tar` shard contains samples as individual files:

```
train-000000.tar
├── 000000.audio.npy
├── 000000.params.npy
├── 000000.mel.npy
├── 000001.audio.npy
├── 000001.params.npy
├── 000001.mel.npy
└── ...
```

Shard count is tuned for GPU worker count — one shard per GPU worker per epoch is ideal; exact sizing depends on batch size and network bandwidth.

**Training integration:** The `webdataset` Python library provides streaming, shuffling, batching, and multi-worker support out of the box. R2 free egress makes streaming from object storage practical. Each GPU worker gets a disjoint subset of `.tar` shards — no coordination needed. Training code must use WebDataset's built-in shuffle (or `shardshuffle`) — finalize writes shards in deterministic order for reproducibility; shuffling is the training loader's responsibility.

## 8. Experiment Tracking (Weights & Biases)

W&B serves as a lightweight observability layer for the pipeline — a few key metrics and the dataset as a first-class artifact. It is not a monitoring dashboard or a log aggregator. W&B is an index and lineage tracker, not the authoritative dataset store. R2 holds the data; `dataset.json` holds the metadata; W&B points to both.

### What Goes in W&B

**Pipeline metrics** (logged by `finalize`):

| Metric                             | Type  | Description                                           |
| ---------------------------------- | ----- | ----------------------------------------------------- |
| `pipeline/shards_total`            | int   | Total shards in spec                                  |
| `pipeline/shards_valid`            | int   | Shards that passed validation                         |
| `pipeline/shards_quarantined`      | int   | Shards copied to quarantine                           |
| `pipeline/total_samples`           | int   | Total samples across all shards                       |
| `pipeline/generation_time_seconds` | float | Wall clock: spec created_at → last shard uploaded     |
| `pipeline/finalize_time_seconds`   | float | Wall clock: finalize start → dataset.complete written |
| `pipeline/errors_total`            | int   | Total errors across all worker reports                |

**Dataset artifact** (logged by `finalize`):

The finalized dataset is registered as a W&B Artifact of type `"dataset"`:

- **Files included:** `input_spec.json`, `dataset.json` (the card)
- **Metadata:** run_id, param_spec, code_version, is_repo_dirty, total_samples, split sizes
- **References:** R2 path to the actual HDF5 data (not uploaded to W&B — too large)

This creates a dataset entry in the W&B artifact registry that can be referenced by training runs, establishing **artifact lineage**: code version → dataset artifact → training run → model checkpoint. Training runs close the lineage loop by declaring the dataset as an input: `artifact = run.use_artifact(f"dataset-{run_id}:latest")`. See [Appendix E.3](#e3-wb-integration) for the full implementation.

## 9. Alternatives Considered

### Comparison

| Alternative                | Cheap Compute | Free Egress | Low Ops Burden | Resumable | No Infra to Own | Verdict                        |
| -------------------------- | :-----------: | :---------: | :------------: | :-------: | :-------------: | ------------------------------ |
| **R2 + RunPod** (selected) |       ✓       |      ✓      |       ✓        |     ✓     |        ✓        | Selected — cheapest, simplest  |
| Kubernetes Jobs            |       ✗       |      ✗      |       ✗        |     ✓     |        ✗        | Too much ops for 1-2x/week     |
| AWS Batch                  |       ✗       |      ✗      |       ✓        |     ✓     |        ✓        | Egress costs kill the budget   |
| Modal                      |       ✗       |      ?      |       ✓        |     ✓     |        ✓        | Revisit when pricing is proven |
| Hadoop / Spark             |       ✗       |      ✗      |       ✗        |     ✓     |        ✗        | Wrong tool — no reduce step    |
| Ray                        |       ✗       |      ✗      |       ✗        |     ✓     |        ✗        | Overkill for fan-out           |
| Airflow / Prefect          |       —       |      —      |       ✓        |     ✓     |        ✓        | Overhead for 2 stages          |
| Single command             |       ✓       |      ✓      |       ✓        |     ✗     |        ✓        | Can't resume or debug mid-run  |

### 9.1 Single Command That Does Everything

**Rejected.** The most obvious alternative: `make dataset CONFIG=...` that runs generate, polls for completion, and finalizes in one blocking command.

Why it doesn't work:

- Generation takes hours. A blocking command ties up a terminal for hours, and a laptop sleep or SSH disconnect kills the run.
- No ability to debug between steps. If 3 shards fail, you want to inspect why before retrying — not have the system silently retry or fail the entire run.
- Phase separation is the point. The reconciliation model means you can launch from machine A, check from machine B, finalize from machine C. A single command loses this.
- Resumability requires separate invocations. "Pick up where you left off" means re-running the same command and having it skip completed work — which is exactly what separate `generate` / `status` / `finalize` commands do.

### 9.2 Simultaneous Launch with Finalize-as-Waiter

**Rejected (was the original design).** Co-launch generation workers and a finalize worker simultaneously. The finalize worker polls R2, waiting for all shards before merging. Worker status files are the authoritative record.

- **Status files as authority is fragile.** A worker could report success but fail to upload. The finalize worker trusts the status and either merges incomplete data or hangs forever.
- **The finalize worker as a waiter wastes money.** A worker sitting idle for 30-60 minutes polling R2 costs compute time for no work.
- **Infrastructure-derived shard names break resumability.** Naming shards `shard-{pod_id}-{attempt_id}-{seq}` means retries produce different filenames.
- **No reconciliation means no partial retry.** If 3 of 10 workers fail, the only option was to rerun everything.

### 9.3 ComputeBackend with Task Lifecycle Management

**Rejected (was the original interface).** The first draft of `ComputeBackend` had `submit()` and `check_tasks()` — the latter polling provider APIs for task status.

- **Provider task state is unreliable.** RunPod can report "running" for a worker that OOM-killed 10 minutes ago.
- **It duplicates reconciliation.** Storage already answers "is this shard done?" ([§7.1](#71-storage-as-the-source-of-truth)). Adding a second, weaker signal creates two sources of truth.
- **It couples the protocol to provider lifecycles.** Pods are persistent and inspectable; serverless functions are ephemeral. A `check_tasks` method pretends providers are more similar than they are.
- **Scope creep risk.** Once you can check status, the next step is retry logic, timeout heuristics, and notifications — a scheduler, not a data pipeline.

### 9.4 Hadoop / MapReduce

**Rejected.** Hadoop is designed for processing existing large datasets with shuffle, sort, and reduce phases over HDFS. This pipeline's workload is fully parallel data *generation* — no inter-worker communication, no data dependencies, no reduce step. Hadoop's infrastructure (HDFS cluster, YARN resource manager, JVM-based framework) would be unused overhead — the pipeline would only use it as a pod launcher.

### 9.5 `make status` as Provider-Status Command

**Rejected (was the original monitoring approach).** The first draft described `make status` as showing live pod status from RunPod's API.

- **Provider APIs answer the wrong question.** "Is the worker running?" ≠ "are the shards done?" Storage answers the right question ([§7.1](#71-storage-as-the-source-of-truth)).
- **Not portable.** Polling RunPod workers is RunPod-specific.
- **Scope creep risk.** Leads to status enums, timeout heuristics, and provider-specific error parsing.

### 9.6 CLI Flags as Run Configuration

**Rejected (was the original approach).** Run configuration via Make variables: `make generate PARAM_SPEC=... SHARD_SIZE=... NUM_SHARDS=...`.

- **Not versionable.** Exact flags exist only in shell history.
- **Drift silently.** Same command with slightly different flag = different dataset, no detection.
- **Mix concerns.** Dataset spec and operational config in the same flat namespace.

The design uses typed YAML config files. The config describes what to produce; the input spec freezes it. Operational concerns (worker count, backend) are CLI arguments.

### 9.7 Minor Alternatives

| Alternative                            | Verdict  | One-line reason                                                              |
| -------------------------------------- | -------- | ---------------------------------------------------------------------------- |
| Apache Spark                           | Rejected | JVM dependency, no reduce step needed, fully parallel workload               |
| Ray                                    | Rejected | Cluster management overhead, overkill for independent tasks                  |
| Dataclasses + manual JSON              | Rejected | No validation on deserialization — Pydantic strict mode is better            |
| OmegaConf for pipeline config          | Rejected | Interpolation/merge features not needed — PyYAML + Pydantic sufficient       |
| Worker report as only debugging record | Rejected | Crashes erase end-of-execution artifacts — debug logs with EXIT trap survive |
| Duplicating spec into dataset.json     | Rejected | Two sources of truth — reference by SHA-256 instead                          |
| Make as primary CLI                    | Rejected | No typed arguments, no --help — Click CLI with Make as thin alias            |
| Hand-rolled retry loops                | Rejected | Proliferate and diverge — centralized retry policy                           |
| Generic stage orchestration framework  | Rejected | Two stages don't justify a framework                                         |

## 10. Operations & Infrastructure

### 10.1 Security & Credentials

| Credential       | Used By                 | Storage                   | Scope                      |
| ---------------- | ----------------------- | ------------------------- | -------------------------- |
| `RUNPOD_API_KEY` | CLI (worker submission) | Docker secrets (BuildKit) | Worker CRUD, most powerful |
| R2 credentials   | All workers             | Docker secrets (BuildKit) | Object storage read/write  |
| `WANDB_API_KEY`  | Finalize, training      | Docker secrets (BuildKit) | Experiment logging         |

Credentials are baked into Docker images via BuildKit `--secret` — not visible in `docker history`. Auth validation runs before any worker launches. Push images only to private registries.

### 10.2 Monitoring & Observability

**During a run:**

- `make status` runs the storage reconciliation report — deterministic, stateless, runnable from any machine
- No long-running monitoring process required
- Provider-side worker health monitoring is out of scope — not justified for 1-2x/week usage

**After a run:**

- `dataset.json` contains the complete output record
- W&B shows pipeline metrics and dataset artifact lineage
- Debug logs in R2 (`metadata/workers/attempts/{worker_id}-{attempt}/debug.log`) provide full JSONL debug log streams, queryable with `jq`
- Quarantined shards preserved for debugging

### 10.3 Cost Model

| Resource          | Cost                      | Notes                                            |
| ----------------- | ------------------------- | ------------------------------------------------ |
| RunPod CPU worker | ~$0.10-0.20/hr            | Cheap on-demand compute, no cluster to manage    |
| Finalize (local)  | $0                        | Runs on user's machine                           |
| R2 storage        | Free egress, $0.015/GB/mo | Major advantage over S3 for frequent downloads   |
| W&B               | Free tier                 | Sufficient for experiment tracking at this scale |

**Typical run cost** (10 workers, 50 shards each, ~1hr):

- Generation: 10 × $0.15/hr × 1hr = ~$1.50
- R2: ~$0.50/mo for a 480k-sample dataset
- **Total per run: ~$2**

**Why these providers:**

- **RunPod:** Cheapest GPUs/CPUs available, no minimum commitment, simple pod API, large model downloads and multi-GB image pulls work reliably
- **R2:** Free egress is the killer feature. Datasets are downloaded frequently for training — S3 egress costs would dwarf compute costs. Reliable, S3-compatible.
- **W&B:** Free tier covers our needs. Dataset artifact tracking without building custom tooling.

### 10.4 Requirements at Scale

The pipeline must support datasets scaling to multi-terabyte sizes while keeping costs minimal:

- Cheap compute (RunPod spot-like pricing)
- Free egress (R2)
- Reliable providers with minimal restarts
- Handle large model downloads and multi-GB Docker image pulls
- No infrastructure to own or manage beyond R2 bucket

## 11. Concurrency, Consistency & Failure Modes

This section covers dense correctness analysis — R2 storage semantics, concurrency edge cases, and failure modes. Separated from the high-level design ([§7](#7-design-decisions)) to keep that section focused on architecture.

### 11.1 R2 Consistency Model

The pipeline's correctness depends on R2's consistency guarantees. [R2 provides strong read-after-write consistency](https://developers.cloudflare.com/r2/reference/consistency/):

- **PUT then GET:** A GET immediately after a PUT returns the new object. Workers upload a shard, and reconciliation immediately sees it.
- **PUT then LIST:** A LIST immediately after a PUT includes the new key. Reconciliation listing shard prefixes sees recently-uploaded shards.
- **DELETE then GET:** A GET immediately after a DELETE returns 404. Quarantine-then-regenerate works correctly.

This is stronger than S3's original eventual consistency model (which was [upgraded to strong consistency in 2020](https://aws.amazon.com/blogs/aws/amazon-s3-update-strong-read-after-write-consistency/)). R2 has always been strongly consistent.

**What R2 does NOT provide:**

- **Conditional writes.** No `If-None-Match` or `If-Match` headers via rclone. Writes are unconditional (last-writer-wins). See [§7.7](#77-concurrency-semantics) for why this is acceptable.
- **Atomic multi-object writes.** Writing `shard-042.h5` and `worker-{id}.json` are two separate PUTs. They can't be made atomic. See [§11.2](#112-failure-modes--edge-cases).
- **Read-your-writes across regions.** R2 is globally distributed; the pipeline assumes single-region usage (all operations from one R2 endpoint).

### 11.2 Failure Modes & Edge Cases

Non-obvious failure modes, edge cases, and blind spots. Each includes the scenario, consequence, and mitigation.

**Corrupt shard in staging:**
A worker's VST plugin crashes mid-render, producing a corrupt file. The worker uploads it to the staging prefix with a unique per-attempt filename. **Consequence:** A corrupt .h5 file exists alongside valid ones in the shard's staging directory. **Mitigation:** Workers validate shards locally *before* uploading ([§7.5](#75-shard-validation)). If local validation fails, the upload is skipped and the failure is logged. Only validated shards reach staging. Even if a corrupt shard slips through, reconciliation re-validates staged files and quarantines failures. The staging model means a corrupt upload never overwrites a valid one — each attempt has a unique filename.

**Non-atomic cross-file writes:**
A worker uploads `shard-042.h5` successfully but crashes before writing `worker-{id}.json`. Or vice versa. These are separate R2 PUTs — they cannot be made atomic. **Consequence:** Worker report may be out of sync with actual shard state. **Mitigation:** `generate`/`status` checks file existence and `.valid` markers, not worker reports ([§7.1](#71-storage-as-the-source-of-truth)). Per-attempt UUIDs make mismatches observable.

**Partial shard upload:**
`rclone` crashes mid-upload. R2 may have a partial or corrupt object at the staging path (though this is rare — R2 PUTs are atomic for single objects, and multipart uploads don't appear until finalized). **Consequence:** A corrupt `.h5` may exist in staging without a `.valid` marker (worker crashed before writing it). Or with a `.valid` marker if the crash happened during a subsequent upload. **Mitigation:** If no `.valid` marker, `generate` treats the shard as missing. If `.valid` exists, finalize's structural check catches the corruption before promotion. The partial upload does not affect any other attempt's file (unique filenames).

**Silent data corruption (bit rot / transfer corruption):**
Local disk corruption between render and upload, or network corruption during transfer, produces a shard in R2 that differs from what the worker intended. **Consequence:** Corrupt shard passes filename checks but contains wrong data. **Mitigation:** All rclone operations use `--checksum`, which verifies content hashes after transfer. If a checksum mismatch is detected, the worker must delete the local shard, re-generate, and re-upload. Storage-layer bit rot within R2 is handled by R2 internally (server-side object checksums).

**Slow worker overtaken by retry:**
Worker A is assigned shard-042. Worker A is slow. User runs `generate` again, reconciliation sees shard-042 as missing (no valid staged shard), assigns to Worker B. Worker B completes first, uploads `pod-B-uuid2.h5` with `.valid` marker. Worker A completes later, uploads `pod-A-uuid1.h5` with `.valid` marker. **Consequence:** Two valid staged shards exist for shard-042. Finalize picks one to promote to `data/shards/shard-000042.h5`. If both workers ran on the same hardware, the shards are identical. If on different hardware, the shards may differ at the floating-point level — finalize picks arbitrarily but the result is valid. **Mitigation:** Content hashes in worker reports detect divergence. Hard timeout on workers prevents long-running zombies.

**Zombie worker uploads after finalize:**
Worker A hangs for 12 hours, then completes and uploads shard-042 to the staging prefix. Meanwhile, Worker B already uploaded shard-042, finalize promoted it to `data/shards/shard-000042.h5`, and the dataset is in use for training. **Consequence:** Worker A's upload lands in `metadata/workers/shards/shard-000042/pod-A-uuid1.h5` — a new file in the staging directory. The canonical `data/shards/shard-000042.h5` is unaffected. The finalized dataset hash is stable. **Mitigation:** The staging/canonical separation ensures zombie uploads cannot corrupt finalized data. Hard timeout on workers and RunPod auto-stop prevent zombies in the first place. Re-running finalize would re-validate staging and re-promote, but would pick the same (or equivalent) shard — the canonical output is stable.

**Spec deleted after generation starts:**
Workers receive shard assignments via environment variables at launch. If the spec is deleted from R2 after launch, workers continue fine but subsequent `status`/`generate` fail. **Consequence:** Orphaned run. **Mitigation:** Spec is immutable, should never be deleted.

**`dataset.complete` exists but outputs are corrupt:**
Finalize wrote `dataset.complete` but outputs were later corrupted. **Mitigation:** Finalize validates outputs when `dataset.complete` exists. If missing/corrupt, deletes stale marker and reruns.

**Stale `dataset.complete` after re-generation:**
User runs `generate` after finalization (e.g., to replace a quarantined shard). `dataset.complete` from old finalize still exists. **Mitigation:** Finalize re-validates outputs against current spec. Mismatches trigger stale marker deletion and rerun.

## 12. Open Questions, Risks & Limitations

### Known Limitations

1. **Single-machine finalize bottleneck.** Finalize downloads all shards to one machine. A 10k-sample shard is typically 50-100MB depending on audio length and spectrogram resolution. At 480 shards (~30GB), this takes minutes on a laptop. Current architecture scales to ~100-200GB datasets on a laptop. Beyond that, finalize should run on a cloud worker (same Docker image, same `ComputeBackend` protocol — add a `--finalize-on-cloud` flag that reuses the worker entrypoint). Incremental statistics (reservoir sampling or one-pass mean/std sketches) would eliminate the need to download all shards for stats computation.

2. **No incremental finalization.** Crashes during finalize restart from scratch. Acceptable because finalize processes existing data and is fast to retry.

3. **Reproducibility is controlled-conditions, not absolute.** The pipeline guarantees that the same spec + same Docker image + same hardware = identical dataset. But VST plugin floating-point behavior can vary across CPU architectures (x86 vs ARM, SSE vs AVX), and Docker base image updates change system libraries. Content hashes in worker reports detect when this happens, but the pipeline does not enforce cross-hardware bit-identity. If multiple workers produce different output for the same shard on different hardware, finalize picks the lexicographically smallest attempt — the selection is deterministic but the content may vary across heterogeneous environments.

4. **R2 listing at scale.** Reconciliation lists staged shard objects (1000/page). At 480 shards: 1 API call. At 48,000: 48 calls — still fast. A future optimization could write a `metadata/shards.manifest` file listing all promoted shard IDs after finalize, allowing subsequent operations to read a single file instead of listing the prefix. Not needed at current scale.

5. **No partial dataset usage.** Training can't start until finalize completes. Acceptable for batch workflows.

6. **Spec immutability.** Can't add shards to an existing run — must create a new run. By design (prevents drift), but means config mistakes cost a new run_id.

7. **No cost controls.** No budget cap or automatic shutdown. `generate --dry-run` prints shard assignments without launching workers; `finalize --dry-run` shows what would be downloaded/resharded without doing it.

8. **No real-time progress.** Only `make status` shows shard counts. No live progress bar or ETA.

9. **Multi-run dataset composition is manual.** Combining datasets from multiple runs: copy shards into a new R2 prefix, run `finalize`, register in W&B via the web UI. No automated tooling.

### Risks

| Risk                              | Likelihood | Impact                                      | Mitigation                                                               |
| --------------------------------- | ---------- | ------------------------------------------- | ------------------------------------------------------------------------ |
| RunPod availability               | Medium     | Workers queue or fail                       | ComputeBackend enables fallback                                          |
| VST plugin crashes (SIGSEGV, OOM) | Medium     | Individual shards fail                      | Per-shard isolation + reconciliation fills gaps                          |
| R2 eventual consistency           | Low        | Stale listing                               | R2 is strongly consistent for PUTs; polling interval handles edge cases  |
| HDF5 virtual dataset portability  | Low        | Breaks on different h5py versions           | Pin h5py in Docker                                                       |
| Cost overrun from stuck workers   | Low        | Workers run indefinitely                    | Hard timeout in entrypoint; RunPod auto-stop                             |
| Non-deterministic rendering       | Low        | Concurrent writes produce different content | Fix renderer; specs record renderer version                              |
| Silent data corruption            | Low        | Corrupt shard accepted as valid             | `rclone --checksum` on all transfers; overhead negligible vs render time |

## 13. Out of Scope

The following are of interest for next steps but are out of scope for this document. They should not influence the design of the two-stage pipeline described above.

### Future Processing Stages

Additional stages could follow the same contract (§5) without modifying existing stages:

| Stage              | Input        | Output                 | Compute |
| ------------------ | ------------ | ---------------------- | ------- |
| **augment-reverb** | raw shards   | augmented shards       | CPU     |
| **add-captions**   | audio shards | shards + text column   | GPU     |
| **add-embeddings** | audio shards | shards + latent column | GPU     |
| **render-presets** | preset bank  | audio shards           | CPU     |

Stage order would remain static and explicit — user runs commands in sequence. If the number of stages grows to 4-6 and manual commands become unwieldy, adopt Prefect rather than building a homegrown orchestrator.

### Data Format Abstraction

The pipeline supports HDF5 and WebDataset output formats ([§7.10](#710-output-format-hdf5-vs-webdataset)). A general `ShardWriter`/`ShardReader` protocol could allow adding further formats (Parquet, Lance) in the future. This should be added when a third format is concretely needed, not speculatively.

### Content-Addressable Outputs

Shards named by input hash (`shard-{sha256(config+seed)}.h5`) would enable cross-run deduplication and integrity verification. Not planned — deterministic logical naming is simpler and sufficient.

### Automatic Stage Chaining

A lightweight trigger (stage A completion → stage B start) could be added when more than 2 stages exist. At 2 stages, explicit commands are clearer.

### Preset Rendering

A `render-presets` stage that uses a curated preset bank instead of random parameters. Questions around preset bank storage, versioning, and shard format compatibility are deferred.

### Credential Management for Open Source

When the repo goes public, contributors configure credentials via `.env` (already used in the repo, with `.env.example` as template). Docker builds read from `.env` via BuildKit secrets. Full contributor onboarding flow is future work.

## 14. Implementation Details

This section covers how the design is realized — specific libraries, configuration, and code patterns. These details support the design decisions above but are not essential to understanding the architecture.

### 14.1 Input Spec Schema

Schema for the frozen input specification described in [§7.1](#71-storage-as-the-source-of-truth) and [§6 artifact taxonomy](#artifact-taxonomy).

```python
class ShardSpec(BaseModel):
    model_config = ConfigDict(strict=True)

    shard_id: int
    filename: str           # "shard-000042.h5"
    seed: int
    row_start: int
    row_count: int
    expected_datasets: list[str]  # ["audio", "mel_spec", "param_array"]
    audio_shape: tuple[int, int]
    param_shape: tuple[int, int]

class PipelineSpec(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)

    run_id: str
    created_at: str         # ISO 8601
    code_version: str       # git commit hash
    is_repo_dirty: bool         # True if working tree had uncommitted changes
    param_spec: str
    renderer_version: str   # Auto-extracted from plugin bundle at materialization
    output_format: str      # "hdf5" or "wds" — determines finalize output
    sample_rate: int
    shard_size: int
    num_shards: int
    splits: dict[str, int]  # {"train": 44, "val": 2, "test": 2} (shard counts, not sample counts)
    shards: list[ShardSpec]
```

All structured data uses Pydantic `BaseModel` in strict mode. Strict mode catches silent type coercion at serialization boundaries. `frozen=True` makes specs immutable at the type level.

**Seed derivation:** Per-shard seeds are computed deterministically during spec materialization: `seed = base_seed + shard_id`, where `base_seed` is derived from the run config. This means the same config always produces the same spec (and therefore the same seeds). Reproducibility comes from re-running with the same frozen spec — the spec is the reproducibility unit, not the config.

**Why JSON for specs and reports:** Machine-generated, stored in R2, read back by the CLI. JSON is the simplest correct format — Pydantic has native JSON methods (`.model_dump_json()` / `.model_validate_json()`), it's human-readable (`rclone cat` + `jq`), and handles nested structures natively. Config files use YAML because they're human-authored.

### 14.2 Dataset Card Schema

```python
class DatasetCard(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)

    schema_version: int
    run_id: str
    finalized_at: str       # ISO 8601

    # Provenance
    code_version: str
    is_repo_dirty: bool
    param_spec: str
    renderer_version: str
    output_format: str      # "hdf5" or "wds"
    sample_rate: int

    # Structure
    total_samples: int
    splits: dict[str, int]  # {"train": 440000, "val": 20000, "test": 20000}
    stats: dict[str, float]

    # Integrity
    validation_summary: ValidationSummary
    worker_architectures: list[str]  # e.g. ["x86_64"] or ["x86_64", "aarch64"] if heterogeneous

    # Reference to full spec
    input_spec_sha256: str
    input_spec_path: str      # "metadata/input_spec.json"
```

### 14.3 Worker Report Schema

Schema for worker reports described in [§7.8](#78-error-handling--crash-resilience).

```python
class ShardResult(BaseModel):
    model_config = ConfigDict(strict=True)
    shard_id: int
    filename: str
    rows: int
    success: bool
    content_hash: str | None = None  # SHA-256 of the .h5 file (None if failed)
    render_time_sec: float
    error: str | None = None

class WorkerReport(BaseModel):
    model_config = ConfigDict(strict=True)
    worker_id: str          # Infrastructure ID (for debugging)
    attempt_uuid: str       # Unique per invocation, used in staging filenames
    assigned_shards: list[int]
    results: list[ShardResult]
    errors: list[str]
    started_at: str         # ISO 8601
    completed_at: str

    # Environment (for debugging non-determinism across hardware)
    cpu_arch: str            # e.g. "x86_64", "aarch64"
    os_info: str             # e.g. "Linux 5.15.0-generic"
```

### 14.4 Sample Type

A typed container for individual training samples during finalize's transcode step (HDF5 → WebDataset). This is a `dataclass`, not a Pydantic model — the data is already validated NumPy arrays at this point, so Pydantic's serialization validation is unnecessary overhead (see validation boundaries below).

```python
@dataclass(frozen=True, slots=True)
class Sample:
    sample_id: int
    audio: np.ndarray       # shape: (channels, samples)
    mel_spec: np.ndarray    # shape: (mels, frames)
    params: np.ndarray      # shape: (num_params,)
```

The `Sample` type ensures the transcode loop reads and writes the correct fields — a bug that drops `mel_spec` or swaps `audio` and `params` is caught by type hints rather than silently producing a broken `.tar` archive. `frozen=True` prevents accidental mutation during transcoding.

**Validation boundaries — when to use what:**

The pipeline uses different validation tools depending on where data crosses a trust boundary:

| Boundary                        | Tool                | Why                                                                                     |
| ------------------------------- | ------------------- | --------------------------------------------------------------------------------------- |
| External input (config YAML)    | Pydantic (strict)   | Untrusted human input — catch type errors, missing fields, invalid values at parse time |
| Serialization (spec, reports)   | Pydantic (strict)   | JSON crossing process boundaries (R2 ↔ CLI ↔ workers) — enforce schema on every read    |
| Shard data (HDF5 arrays)        | Validation function | NumPy arrays inside HDF5 — Pydantic can't validate `ndarray`; custom checks required    |
| Internal transform (HDF5 → WDS) | `dataclass` (above) | Data already validated — typed container prevents field mixups during transcoding       |

Pydantic is for trust boundaries — where data enters the system from an external source (user config, JSON from R2, worker reports from other processes). Dataclasses are for internal contracts — where data has already been validated and you just need a typed container to prevent programming errors. No runtime validation overhead on 480k samples.

### 14.5 Config Materialization

A run starts from a typed YAML config file:

```yaml
# configs/pipeline/surge_simple_480k.yaml
experiment_name: surge_simple
param_spec: surge_simple
plugin_path: plugins/Surge XT.vst3
output_format: hdf5       # "hdf5" (local training) or "wds" (multi-GPU streaming)
sample_rate: 16000
shard_size: 10000
num_shards: 48
splits:
  train: 44
  val: 2
  test: 2
```

On first `generate`:

1. Load YAML, validate against Pydantic `RunConfig` (strict mode)
2. Extract `renderer_version` from the plugin bundle (`CFBundleShortVersionString` from `Info.plist` on macOS, `Version` from `moduleinfo.json` on Linux)
3. Derive `run_id`: `{experiment_name}-{train_size}-{shard_size}-{YYYYMMDD-HHMMSS}`
4. Materialize `PipelineSpec` — expand config into shard-level spec (seeds, shapes, row ranges)
5. Upload spec + source config to R2
6. Proceed with reconciliation

**Dirty repo handling:** When `is_repo_dirty` is true, `generate` automatically creates a `git diff` and uploads it to `metadata/run_diff.patch`. This allows reconstructing the exact code state even when changes weren't committed — common during rapid ML research iteration.

**Config drift protection:** If `--config` is passed for a `run_id` that already has a spec, the command errors. The spec is authoritative after creation.

### 14.6 Run ID Format

```
{experiment_name}-{total_train_samples}-{shard_size}-{YYYYMMDD-HHMMSS}
Example: surge_simple-480k-10k-20260312-143022
```

### 14.7 CLI & Directory Structure

```
pipeline/
  cli.py               # Click entry point: generate, status, finalize

  stages/              # Each stage is a self-contained module
    generate.py         # Generate stage logic
    finalize.py         # Finalize stage logic

  backends/            # Compute provider implementations
    base.py             # ComputeBackend Protocol definition
    runpod.py           # RunPodBackend (production)
    local.py            # LocalBackend (development/testing)

  storage.py            # R2 operations (list, upload, download, quarantine)
  reconcile.py          # Read spec, validate shards, compute missing set
  schemas.py            # Pydantic models (PipelineSpec, WorkerReport, DatasetCard)
  validation.py         # Shard validation (structural, shape, value, row count)
  retry.py              # Centralized tenacity retry policy
  logging_config.py     # structlog configuration
```

Pipeline configs live in `configs/pipeline/` to distinguish them from training configs in `configs/data/` and `configs/trainer/`:

```
configs/
  pipeline/            # Dataset generation recipes (YAML)
    surge_simple_480k.yaml
    surge_xt_1m.yaml
  data/                # Training data module configs (Hydra)
    surge_simple.yaml
    surge.yaml
  trainer/             # Training configs (Hydra)
    ddp.yaml
```

## Appendix A: Glossary

| Term                       | Definition                                                                                                                                                                                                                                                                                                         |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **R2**                     | [Cloudflare R2](https://developers.cloudflare.com/r2/), an S3-compatible object storage service. Key feature: free egress (no cost to download data). Used for shard storage and pipeline coordination. [Consistency model](https://developers.cloudflare.com/r2/reference/consistency/): strong read-after-write. |
| **RunPod**                 | [RunPod](https://www.runpod.io/), a cloud compute marketplace offering on-demand GPU and CPU instances ("pods"). Used for running data generation workers. Pods are ephemeral — they run a Docker container and terminate.                                                                                         |
| **Worker**                 | A cloud compute instance that generates shards. On RunPod, a worker is a "pod" — a single Docker container with assigned shard work. The design uses "worker" to stay infrastructure-agnostic.                                                                                                                     |
| **Shard**                  | An HDF5 file containing a batch of training samples (audio, mel spectrograms, parameter arrays). Typically 1k-10k samples per shard. Named by logical index (`shard-000042.h5`).                                                                                                                                   |
| **W&B (Weights & Biases)** | [Weights & Biases](https://wandb.ai/), an experiment tracking platform. Used here as a lightweight observability layer: pipeline metrics, dataset artifact registry, and lineage tracking from dataset → training run.                                                                                             |
| **Virtual dataset**        | HDF5 feature that creates a logical view over multiple files without copying data. Used by finalize to compose train/val/test splits from individual shards.                                                                                                                                                       |
| **Input spec**             | JSON file (`input_spec.json`) defining the frozen input specification for a run — shard specs, seeds, shapes, splits, renderer version. Written once on first `generate`, never modified.                                                                                                                          |
| **run_id**                 | Unique identifier for a pipeline execution. Format: `{experiment}-{size}-{shard_size}-{timestamp}`. Example: `surge_simple-480k-10k-20260312-143022`.                                                                                                                                                              |
| **Shard ID**               | Logical index for a shard (`shard-000042`). Deterministic, defined at run creation, independent of which worker computes it.                                                                                                                                                                                       |
| **worker_id**              | Infrastructure identifier (e.g., RunPod's `RUNPOD_POD_ID`). Appears only in metadata, not in shard paths.                                                                                                                                                                                                          |
| **Reconciliation**         | Comparing desired state (spec) against actual state (validated shards in R2) to determine what work remains.                                                                                                                                                                                                       |
| **dataset.complete**       | Marker file written by finalize as the very last step. Means "finalization is done" — not a mutex or lock. Contains run_id and timestamp.                                                                                                                                                                          |
| **Debug log**              | JSONL file (`metadata/workers/attempts/{worker_id}-{attempt}/debug.log`) of structured events from a worker. Append-only, uploaded by EXIT trap, survives crashes.                                                                                                                                                 |
| **Worker report**          | JSON summary (`metadata/workers/attempts/{worker_id}-{attempt}/report.json`) of a worker's results, including content hashes for provenance. Written at exit, missing if worker crashed.                                                                                                                           |
| **Lifecycle marker**       | Empty file in `metadata/workers/shards/shard-{id}/` named `{worker_id}-{attempt}.{state}`. Three commit points: `.rendering` (attempt started), `.valid` (staged shard committed), `.promoted` (canonical shard committed). Plus `.invalid` (validation failed). Presence is the state — no content to parse.      |
| **Quarantined shard**      | A corrupt shard uploaded by the worker to `metadata/workers/shards/shard-{id}/quarantine/` on validation failure. Preserves the evidence for debugging alongside lifecycle markers.                                                                                                                                |
| **Dataset card**           | JSON file (`dataset.json`) describing the finalized dataset: provenance, structure, stats. References the spec by SHA-256.                                                                                                                                                                                         |
| **param_spec**             | Configuration selecting which synthesizer parameters to vary. Determines prediction task dimensionality. Examples: `surge_simple` (92 params), `surge_xt` (189 params).                                                                                                                                            |
| **VST**                    | Virtual Studio Technology — plugin format for audio synthesizers. Surge XT is the VST used for rendering.                                                                                                                                                                                                          |
| **Mel spectrogram**        | Frequency-domain audio representation used as neural network input. 128 mels, ~100 frames/sec.                                                                                                                                                                                                                     |
| **Fully parallel**         | Workload where tasks are completely independent — no communication or shared state between workers.                                                                                                                                                                                                                |
| **rclone**                 | CLI tool for syncing files to cloud storage. Used as the R2 upload/download mechanism.                                                                                                                                                                                                                             |
| **WebDataset**             | [WebDataset](https://github.com/webdataset/webdataset), a PyTorch-compatible format for streaming training data. Stores samples in sequential `.tar` archives optimized for HTTP/S3 streaming. Used as the `wds` output format for multi-GPU training.                                                             |

## Appendix B: Tech Stack

| Component       | Technology                                                        | Role                                             |
| --------------- | ----------------------------------------------------------------- | ------------------------------------------------ |
| Build           | Docker (BuildKit)                                                 | Reproducible compute environments                |
| Storage         | Cloudflare R2                                                     | Data + coordination, free egress                 |
| Execution       | RunPod                                                            | Cheap on-demand cloud workers                    |
| Tracking        | Weights & Biases                                                  | Pipeline metrics, dataset artifact registry      |
| Data format     | [HDF5](https://www.h5py.org/) (h5py + hdf5plugin)                 | Shard generation + local training format         |
| Training format | [WebDataset](https://github.com/webdataset/webdataset)            | Streaming `.tar` shards for multi-GPU training   |
| CLI             | [Click](https://click.palletsprojects.com/)                       | Typed arguments, validation, `--help`            |
| Validation      | [Pydantic](https://docs.pydantic.dev/) (strict mode)              | PipelineSpec, report, and config validation      |
| Logging         | [structlog](https://www.structlog.org/)                           | Structured JSON debug logging                    |
| Retry           | [tenacity](https://tenacity.readthedocs.io/)                      | Centralized retry policy                         |
| Upload/download | [rclone](https://rclone.org/)                                     | R2 file transfer; all transfers use `--checksum` |
| Containers      | [Docker](https://docs.docker.com/build/buildkit/) (BuildKit)      | Reproducible environments                        |
| Audio           | [Surge XT](https://surge-synthesizer.github.io/) (headless, Xvfb) | VST synthesis                                    |

## Appendix C: References

- [Industrial Empathy — Design Docs at Google](https://www.industrialempathy.com/posts/design-docs-at-google/) — section structure, review process
- [Eugene Yan — ML Design Docs](https://eugeneyan.com/writing/ml-design-docs/) — ML-specific methodology sections

## Appendix D: Implementation Roadmap

Full staged implementation plan: [greedy-tickling-harbor.md](greedy-tickling-harbor.md)

| Stage | Scope                                    | Dependencies |
| ----- | ---------------------------------------- | ------------ |
| 0     | Mutation testing + presubmit             | None         |
| 1     | Constants module + naming overhaul       | Stage 0      |
| 2     | Auth checks                              | Stage 1      |
| 3     | Structured logging + worker reports      | Stage 1      |
| 4     | R2 operations + shard validation         | Stage 1      |
| 5     | Reconciliation-based generate + finalize | Stages 1-4   |
| 6     | End-to-end integration test              | Stage 5      |
| 7     | Credential management (Docker)           | Stage 5      |
| 8     | ComputeBackend protocol + LocalBackend   | Stage 5      |

## Appendix E: Implementation Recipes

Library configuration snippets referenced from the main design. These are illustrative — the authoritative implementations live in the codebase.

### E.1 Structured Logging

Workers use `structlog` with JSON rendering for append-only debug log streams ([§7.8](#78-error-handling--crash-resilience)):

```python
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
)

log = structlog.get_logger().bind(run_id=run_id, worker_id=worker_id)
log.info("shard_started", shard_id=42)
log.info("shard_validated", shard_id=42, rows=10000)
log.error("shard_failed", shard_id=43, error="NaN in audio buffer")
```

Structured output goes to stdout (live debugging via `docker logs`). The entrypoint tees stdout to a local file, uploaded to R2 by the bash EXIT trap — survives crashes.

### E.2 Retry Policy

All transient storage operations use `tenacity` with a centralized policy:

```python
# pipeline/retry.py
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

storage_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=16),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    reraise=True,
)
```

One definition, applied everywhere via decorator. Permanent failures (auth, wrong bucket) reraise immediately.

### E.3 W&B Integration

Implementation of experiment tracking ([§8](#8-experiment-tracking-weights--biases)):

```python
# In finalize, after resharding and stats:
import wandb

run = wandb.init(project="surge-data-pipeline", job_type="data-pipeline")

# Log pipeline metrics
run.log({
    "pipeline/shards_total": spec.num_shards,
    "pipeline/shards_valid": validation_summary.valid,
    "pipeline/shards_quarantined": validation_summary.quarantined,
    "pipeline/total_samples": total_samples,
    "pipeline/errors_total": total_errors,
})

# Register dataset as artifact
artifact = wandb.Artifact(
    name=f"dataset-{spec.run_id}",
    type="dataset",
    metadata={
        "run_id": spec.run_id,
        "param_spec": spec.param_spec,
        "code_version": spec.code_version,
        "total_samples": total_samples,
        "splits": card.splits,
    },
)
artifact.add_file(input_spec_path)        # input_spec.json
artifact.add_file(card_path)        # dataset.json
artifact.add_reference(f"r2://{bucket}/{run_id}/data/")  # pointer to R2 data
run.log_artifact(artifact)
run.finish()
```

Training runs declare the dataset as an input, closing the lineage loop:

```python
artifact = run.use_artifact(f"dataset-{run_id}:latest")
```

______________________________________________________________________
