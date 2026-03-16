# Design Doc: Data Pipeline

> **Status**: Draft
> **Author**: ktinubu@
> **Last Updated**: 2026-03-15

______________________________________________________________________

### Index

| §   | Section                                                                                | What it covers                                                |
| --- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 1   | [Context & Motivation](#1-context--motivation)                                         | Problem statement, infrastructure layers                      |
| 2   | [Typical Workflow](#2-typical-workflow)                                                | End-to-end CLI example                                        |
| 3   | [Goals, Non-Goals & Design Principles](#3-goals-non-goals--design-principles)          | Requirements, principles, anti-goals, success metrics         |
| 4   | [System Overview](#4-system-overview)                                                  | Architecture summary, data/control plane, watchmen            |
| 5   | [Stage Definitions](#5-stage-definitions)                                              | Generate and finalize stages                                  |
| 6   | [Data Flow & Architecture](#6-data-flow--architecture)                                 | Diagrams, R2 layout, artifact taxonomy                        |
| 7   | [Design Decisions](#7-design-decisions)                                                | Storage-as-truth, reconciliation, concurrency, error handling |
| 8   | [Experiment Tracking](#8-experiment-tracking-weights--biases)                          | W&B metrics, artifacts, lineage                               |
| 9   | [Alternatives Considered](#9-alternatives-considered)                                  | Comparison chart, detailed rejections                         |
| 10  | [Operations & Infrastructure](#10-operations--infrastructure)                          | Credentials, monitoring, cost model                           |
| 11  | [Concurrency, Consistency & Failure Modes](#11-concurrency-consistency--failure-modes) | R2 consistency, edge cases, failure analysis                  |
| 12  | [Open Questions, Risks & Limitations](#12-open-questions-risks--limitations)           | Known gaps and trade-offs                                     |
| 13  | [Out of Scope](#13-out-of-scope)                                                       | Future work — not referenced elsewhere                        |
| 14  | [Implementation Details](#14-implementation-details)                                   | Schemas, CLI structure, libraries, W&B integration            |
| 15  | [Discarded Choices](#15-discarded-choices)                                             | Brief rejections of minor alternatives                        |
| A–D | [Appendices](#appendix-a-glossary)                                                     | Glossary, tech stack, references, roadmap                     |

______________________________________________________________________

## 1. Context & Motivation

Topline goal: Get massive dataset generation working reliably enough, and know what went wrong when there is unexpected behavior.

**synth-permutations** is a machine learning research project studying how neural networks can infer synthesizer parameters from audio. The core task: given a recording of a synthesizer, predict the knob settings (parameters) that produced it.

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
# → 48/48 valid. Resharding → train.h5, val.h5, test.h5
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

- **Storage is truth** — correctness is determined by inspecting actual shard files in R2, not by trusting worker-reported metadata. Metadata files are debugging hints, not the source of truth ([§7.1](#71-storage-as-the-source-of-truth))
- **Reconciliation over orchestration** — the pipeline determines what work remains by comparing desired state (spec) against actual state (validated shards). We are not in the business of orchestrating. Any command is safe at any time ([§7.4](#74-reconciliation-based-execution))
- **Deterministic work identity** — shard IDs are logical and deterministic (`shard-000042`), not tied to infrastructure. Any worker can compute any shard ([§7.3](#73-deterministic-shard-identities))
- **Stage isolation** — each processing stage is an independent, reconcilable transform with well-defined inputs and outputs ([§5](#5-stage-definitions))
- **Fail visibly** — errors are captured, structured, and surfaced, never swallowed ([§7.8](#78-error-handling--crash-resilience))
- **Validate at boundaries** — data is verified when entering and leaving each stage ([§7.5](#75-shard-validation))
- **Thin abstractions** — only abstract what's needed. Two compute backends (local + RunPod), not a speculative provider framework ([§7.9](#79-compute-abstraction))

### What This System Deliberately Avoids

- **Consensus protocols** — one writer per shard, no conflicts
- **Distributed transactions** — stages are independent
- **Service discovery** — workers don't communicate
- **Message queues** — reconciliation-based reporting is adequate at 1-2x/week
- **Automatic stage chaining** — explicit commands are clearer at 2 stages
- **Provider job supervision** — submit work and exit; storage determines completeness
- **Speculative provider abstractions** — only local + RunPod until a third is needed
- **Owning provider observability** — RunPod pod health monitoring is RunPod's problem, not ours. We determine completeness from storage, not provider APIs

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

### Do Not Build

- Generic DAG parser or dependency graph executor
- Automatic stage chaining engine
- Stage plugin registry or dynamic stage discovery
- Workflow definition language or config-driven stage ordering
- Stage lifecycle hooks (pre-stage, post-stage, on-failure)
- RunPod health monitoring or observability tooling — engineering effort is not justified for a 1-2x/week pipeline
- Provider-agnostic abstraction beyond the two backends actually tested

## 4. System Overview

The pipeline is a batch-oriented, fully parallel data generation system built on a **reconciliation model**: inspect storage, determine what work is missing, launch only that work.

A CLI running on the user's local machine reads a spec (desired state), lists existing validated shards in R2 (actual state), computes the difference, and launches N workers to produce the missing shards. Each worker independently renders audio samples through a VST plugin and writes HDF5 shards to R2. When all shards are present, a separate finalize command reshards into train/val/test splits, computes normalization statistics, registers the dataset in W&B, and writes a completion marker.

R2 serves as both the **data plane** and the **control plane**:

| Plane             | What flows through it  | Examples                                             |
| ----------------- | ---------------------- | ---------------------------------------------------- |
| **Data plane**    | Actual dataset content | HDF5 shards, virtual datasets, stats.npz             |
| **Control plane** | Coordination metadata  | Spec, worker reports, debug logs, `dataset.complete` |

Both planes use R2. There is no separate database, message queue, or coordination service. This means one piece of infrastructure to manage, one set of credentials, one failure mode to reason about. The trade-off: R2 has no atomic test-and-set, so mutual exclusion is not possible. This is acceptable because all operations are idempotent and produce deterministic outputs (§7.6).

### Who Watches the Watchmen

The reconciliation report (`make status`) is the system's single watchman. It reads the input spec, lists and validates every shard, and prints a deterministic summary. It's stateless, runs from any machine, and its logic is a simple set difference: `spec_shards - validated_shards = missing`.

But what if reconciliation itself has a bug — e.g., it validates a corrupt shard as good?

- **Defense in depth:** Validation runs four independent checks (structural, shape, value, row count). All four must pass.
- **Finalize re-validates:** Finalize runs its own validation pass before merging, independent of any earlier reconciliation.
- **Training is the ultimate check:** A corrupt dataset will fail to train properly. This provides end-to-end verification.
- **Manual spot-checking is feasible:** At 1-2 runs/week, eyeballing a few shards is practical and encouraged.
- **The spec is immutable:** The spec is written once and never modified. The `run_id` in `dataset.complete` links back to the exact spec used.

## 5. Stage Definitions

The pipeline has two stages. Each is an independent command with well-defined inputs and outputs.

| Stage        | Command                 | Input                                     | Output                                                                           | Compute                        |
| ------------ | ----------------------- | ----------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------ |
| **Generate** | `pipeline.cli generate` | Config YAML (first run) or spec (retries) | HDF5 shards in `{run_id}/data/shards/`                                           | CPU — VST audio rendering      |
| **Finalize** | `pipeline.cli finalize` | Validated shards in R2                    | `train.h5`, `val.h5`, `test.h5`, `stats.npz`, `dataset.json`, `dataset.complete` | CPU — download, reshard, stats |

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
│  (local machine)              │
│                               │         ┌────────────────┐
│  1. Validate auth (R2+RunPod) │         │  Cloudflare R2  │
│  2. Read/create spec  ◄───┼────────►│                │
│  3. List existing shards  ◄───┼─────────┤  {run_id}/     │
│  4. Validate existing shards  │         │   data/shards/ │
│  5. Compute missing set       │         │   metadata/    │
│  6. Partition across N workers│         │                │
│  7. Submit N tasks            │         │                │
│  8. Exit                      │         │                │
└───────────────────────────────┘         │                │
                                          │                │
         ┌───────────────────┐            │                │
         │  Worker 1         │───────────►│  shard-000000  │
         │  (RunPod worker)  │            │  shard-000001  │
         │  shards 0-47      │            │  ...           │
         └───────────────────┘            │                │
         ┌───────────────────┐            │                │
         │  Worker N         │───────────►│  shard-000432  │
         │  shards 432+      │            │  ...           │
         └───────────────────┘            │                │
                                          │                │
┌───────────────────────────────┐         │                │
│  make finalize RUN_ID=...     │         │                │
│  (local or cloud)             │         │                │
│                               │         │  data/         │
│  1. Read spec         ◄───┼─────────┤   train.h5     │
│  2. Validate all shards   ◄───┼─────────┤   val.h5       │
│  3. Download shards       ◄───┼─────────┤   test.h5      │
│  4. Reshard → train/val/test  │         │   stats.npz    │
│  5. Compute stats             │         │   dataset.json │
│  6. Register in W&B      ────┼──┐      │   dataset.complete│
│  7. Upload finalized      ────┼──┼─────►│                │
│  8. Write dataset.complete ───┼──┘      │                │
└───────────────────────────────┘         └────────────────┘
```

### R2 File Structure

```
{run_id}/                              # e.g. surge_simple-480k-10k-20260312-143022
  data/
    shards/
      shard-000000.h5                    # Deterministic logical shard IDs
      shard-000001.h5
      ...
      shard-000479.h5
    metadata/
      config.yaml                        # User recipe (provenance copy, not authoritative)
      input_spec.json                    # Frozen input specification (authoritative)
      dataset.json                       # Self-describing dataset card (written by finalize)
      dataset.complete                   # Completion marker (written by finalize)
      shards/                            # Per-shard metadata, grouped by shard ID
        shard-000000/
          {worker_id}-{attempt_uuid}.valid       # Lifecycle marker: validated + uploaded
        shard-000042/
          {worker_id}-{attempt_uuid}.rendering   # First attempt: started but crashed
          {worker_id}-{attempt_uuid}.valid       # Second attempt: succeeded
          quarantine/
            shard-000042.h5                      # Corrupt version, preserved for debugging
      worker_attempts/                    # Per-attempt worker artifacts
        {worker_id}-{attempt_uuid}/
          report.json                    # Worker summary — per-shard results, content_hash, timing
          debug.log                      # Debug log (JSONL), uploaded by EXIT trap
    train.h5                             # Virtual dataset (written by finalize)
    val.h5
    test.h5
    stats.npz                            # Normalization statistics
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
       │ shard-*.h5   │  │ report.json  │  │ debug.log     │
       │ (data)       │  │ (worker      │  │ (worker       │
       │ HDF5 shards  │  │  summary)    │  │  debug log)   │
       └──────┬───────┘  └─────────────┘  └───────────────┘
              │                │
       ┌──────▼──────┐  ┌─────▼─────────┐
       │ .rendering   │  │ .valid         │
       │ .valid       │  │ .invalid       │
       │ .invalid     │  │ (lifecycle     │
       │ (lifecycle)  │  │  markers)      │
       └──────┬───────┘  └───────────────┘
              │
     pipeline.cli finalize  ← consumes shards
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

| Artifact          | Path                                                         | Format     | Produced By              | Consumed By                      |
| ----------------- | ------------------------------------------------------------ | ---------- | ------------------------ | -------------------------------- |
| Config            | `metadata/config.yaml`                                       | YAML       | User                     | `generate`                       |
| Input spec        | `metadata/input_spec.json`                                   | JSON       | `generate` (first run)   | `generate`, `status`, `finalize` |
| Shard             | `shards/shard-{id}.h5`                                       | HDF5       | Workers                  | `finalize`                       |
| Lifecycle marker  | `metadata/shards/shard-{id}/{worker}-{attempt}.{state}`      | Empty file | Workers / Reconciliation | `status`, humans                 |
| Quarantined shard | `metadata/shards/shard-{id}/quarantine/shard-{id}.h5`        | HDF5       | Reconciliation           | Humans (debugging)               |
| Worker report     | `metadata/worker_attempts/{worker_id}-{attempt}/report.json` | JSON       | Workers (at exit)        | `status`                         |
| Debug log         | `metadata/worker_attempts/{worker_id}-{attempt}/debug.log`   | JSONL      | Workers (continuous)     | Humans (`jq`)                    |
| Dataset card      | `metadata/dataset.json`                                      | JSON       | `finalize`               | Training scripts, humans         |
| Completion marker | `metadata/dataset.complete`                                  | JSON       | `finalize` (last step)   | `finalize` (idempotency check)   |
| Stats             | `stats.npz`                                                  | NumPy      | `finalize`               | Training scripts                 |

**Shard attempt lifecycle:** Each attempt is tracked by an empty marker file in the shard's metadata directory, named `{worker_id}-{attempt_uuid}.{state}`:

| Marker       | Written by                               | Meaning                                                                                                                       |
| ------------ | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `.rendering` | Worker (at start)                        | Rendering in progress. Replaced by `.valid` on success. Orphaned `.rendering` without a `.valid` indicates a crashed attempt. |
| `.valid`     | Worker (after local validation + upload) | Shard validated locally and uploaded to canonical path. Content hash recorded in the worker report.                           |
| `.invalid`   | Reconciliation                           | Existing shard failed validation. Shard moved to `quarantine/`. Reason logged in reconciliation output.                       |

Listing a shard's metadata directory shows the full history at a glance:

```
$ rclone ls r2:bucket/{run_id}/data/metadata/shards/shard-000042/
  0  pod-abc123-a1b2c3d4.rendering     # first attempt crashed
  0  pod-def456-e5f6a7b8.valid         # second attempt succeeded
```

**Naming conventions:** Config is YAML (human-authored). Everything machine-generated is JSON. Debug logs are JSONL (one event per line). Data is HDF5. Metadata is grouped by shard ID (`metadata/shards/shard-{id}/`) so you can inspect all attempts and quarantined files for a single shard in one listing. Worker artifacts are grouped under `worker_attempts/{worker_id}-{attempt_uuid}/`. Lifecycle markers are empty files — presence is the state, no content to parse.

## 7. Design Decisions

### 7.1 Storage as the Source of Truth

The pipeline uses R2 as both the data layer and the coordination layer. Integrity is guaranteed by content hashes. **A shard is complete if and only if the canonical shard file exists in R2 and passes validation.** Worker metadata is supplementary — useful for debugging, never authoritative.

**Why R2:**

- Workers already upload shards to R2, so the write path exists
- R2 state survives worker termination and cleanup
- S3-compatible — coordination layer is portable to any cloud
- Free egress — datasets are frequently downloaded for training
- No additional infrastructure to manage
- Files are human-readable and inspectable (`rclone cat` + `jq`)

**State model — three things determine pipeline state:**

1. **Spec** — desired state: which shards should exist
2. **Shard files** — actual state: which shards exist and pass validation
3. **Metadata files** — hints: worker reports, timing, error messages (never authoritative, never block completion)

This eliminates an entire class of edge cases:

- Worker crashes before writing status → shard file is missing → reconciliation detects it
- Status reports success but upload failed → shard file is missing → reconciliation detects it
- Shard exists but status was never written → shard passes validation → it's complete
- Worker metadata is missing but all shards validate → run is complete

### 7.2 Shard Lifecycle

Every shard in the pipeline moves through a well-defined set of states. These states are derived from storage — marker files, shard files, and the dataset card — not from worker self-reporting.

```
missing → rendering → valid → finalized
               ↓
            invalid (quarantined)
```

| State         | How it's determined                                                                                                                  | Persisted as                                                                            |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| **missing**   | Shard is in the input spec but no valid shard file or `.valid` marker exists in R2                                                   | Implicit (absence of shard + marker)                                                    |
| **rendering** | `.rendering` marker exists for this shard. Worker has started but not yet uploaded a validated shard.                                | `metadata/shards/shard-{id}/{worker_id}-{attempt}.rendering`                            |
| **valid**     | `.valid` marker exists. Shard was validated locally by the worker and uploaded to the canonical path.                                | `metadata/shards/shard-{id}/{worker_id}-{attempt}.valid`                                |
| **invalid**   | `.invalid` marker exists. Reconciliation found the shard in R2 but it failed validation. The corrupt file is moved to `quarantine/`. | `metadata/shards/shard-{id}/{worker_id}-{attempt}.invalid` + `quarantine/shard-{id}.h5` |
| **finalized** | Shard was consumed by `finalize`, content hash recorded in `dataset.json`. The dataset is sealed.                                    | Content hash in `dataset.json`, `dataset.complete` marker exists                        |

**Transitions:**

- **missing → rendering:** Worker begins shard generation, writes `.rendering` marker.
- **rendering → valid:** Worker validates locally, uploads shard, replaces `.rendering` with `.valid`.
- **rendering → missing:** Worker crashes. `.rendering` is orphaned — no `.valid` follows. Reconciliation treats the shard as missing. The orphaned `.rendering` marker is observable evidence of the crashed attempt.
- **valid → invalid:** Reconciliation re-validates an existing shard and finds it corrupt (e.g., renderer non-determinism, bit rot detected by checksum mismatch). Shard moved to quarantine, `.invalid` written. Shard is now effectively missing and will be regenerated.
- **valid → finalized:** `finalize` consumes the shard, records its content hash in `dataset.json`, and writes `dataset.complete`.

**Key properties:**

- **States are derived from storage, not reported by workers.** A `.rendering` marker with no `.valid` means the attempt failed, regardless of what the worker report says.
- **Multiple attempts are visible.** A shard directory might contain `pod-abc.rendering` (crashed) and `pod-def.valid` (succeeded) — the full history is one `rclone ls` away.
- **The finalized state is dataset-level, not shard-level.** Individual shards don't know they've been finalized. The `dataset.json` content hash and `dataset.complete` marker are the authority.

### 7.3 Deterministic Shard Identities

Shard IDs are logical and deterministic: `shard-000000.h5` through `shard-000479.h5`. Defined at run creation, independent of which worker computes them.

- **Any worker can compute any shard** — retries simply recompute the same logical shard
- **Resumability is a set difference:** `spec_shards - validated_shards = work_remaining`
- **No naming collisions** — the canonical path is always `shard-{logical_index}.h5`
- **Infrastructure details** (worker IDs) appear only in metadata files, not in shard paths

**Work assignment:** The CLI partitions shards across N workers. Worker 1 gets shards 0-47, Worker 2 gets shards 48-95, etc. But the shard's identity is independent of which worker computes it. If Worker 1 fails and its shards are reassigned to Worker 3, output paths are unchanged.

**Shard write protocol:**

1. Write `.rendering` marker: `metadata/shards/shard-{id}/{worker_id}-{attempt_uuid}.rendering`
2. Render shard to a local temp file
3. **Validate locally** (structural, shape, value, row count) — this is critical: only validated shards are uploaded. A corrupt render (VST crash, NaN values, truncated output) is caught here and never reaches R2. See [§11.2](#112-failure-modes--edge-cases) for why this matters.
4. Upload to canonical path: `{run_id}/data/shards/shard-{id}.h5`
5. Replace `.rendering` with `.valid` marker: `metadata/shards/shard-{id}/{worker_id}-{attempt_uuid}.valid`
6. Write worker report (including content hash) to `metadata/workers/{worker_id}-{attempt_uuid}/report.json`

> **Invariant:** A shard counts as complete only when the canonical object exists in storage and passes validation. Workers must validate locally before upload to prevent corrupt writes from overwriting valid data ([§7.7](#77-concurrency-semantics)).

### 7.4 Reconciliation-Based Execution

Instead of tracking worker state or polling provider APIs, the pipeline determines what work remains by inspecting storage.

**`generate` reconciliation:**

1. Read spec from R2 (or create if first run)
2. List existing shard files in R2
3. Validate each existing shard (structure, shape, value, row count)
4. Compute `missing = spec_shards - validated_shards`
5. If nothing missing → "generation complete", exit 0
6. Partition missing shards across N workers
7. Submit N tasks, exit

**`finalize` reconciliation:**

1. Read spec from R2
2. Check for `dataset.complete` — if present and outputs validate, exit 0 ("already finalized")
3. List and validate all shards
4. If any shards missing → report which ones, exit 1
5. Download, reshard, compute stats, register in W&B, upload, write `dataset.complete`

**Key properties:**

- **Safe at any time.** Running `generate` when all shards exist is a no-op. Running `finalize` when 3 shards are missing reports the gap and exits.
- **Machine-independent.** Authoritative state lives in R2. If the laptop that launched the run dies, any machine can continue.
- **Phase separation.** Generation and finalization are independent steps. No idle worker waiting for shards. No implicit coordination.

**`make status` — reconciliation report:**

`make status` runs the same reconciliation logic as `generate` but only prints the result. It does not query RunPod, check worker health, or monitor live tasks. The output is fully determined by storage contents — running it from any machine, at any time, produces the same result.

```
$ python -m pipeline.cli status --run-id surge_simple-480k-10k-20260313-100000

Run: surge_simple-480k-10k-20260313-100000
Spec shards: 48
Valid shards:     44
Missing shards:   2
Quarantined:      2
Worker reports:   9

Missing:
  shard-000005
  shard-000019

Quarantined:
  shard-000006  NaN values
  shard-000023  row count mismatch

Recent worker errors (from metadata):
  worker-abc123: shard-000006 failed: NaN in audio buffer
  worker-def456: shard-000019 upload timeout
```

Worker error details are overlaid from metadata files when present, but the core output (valid/missing/quarantined counts) is derived entirely from shard files.

### 7.5 Shard Validation

Every shard is validated before finalize merges it. Validation checks:

- **Structural**: Valid HDF5, expected datasets present (`audio`, `mel_spec`, `param_array`)
- **Shape**: Array dimensions match spec spec (sample rate, spectrogram bins, parameter count)
- **Value**: No NaN/Inf values, audio within [-1, 1], parameters within spec bounds
- **Row count**: Matches spec's expected shard size

Corrupt shards are quarantined (moved to `metadata/shards/shard-{id}/quarantine/`) and reported. Keeping quarantined files alongside the shard's receipts means `rclone ls` of a shard's metadata directory shows the full history. Reconciliation treats quarantined shards as missing — they'll be regenerated on next `generate`.

### 7.6 Finalize Workflow

01. **Check for `dataset.complete`** — if present and all outputs validate, print "already finalized" and exit 0
02. **Read spec** from R2
03. **List and validate** all shard files
04. **Check completeness** — if any shards missing/quarantined, report which ones and exit 1
05. **Download validated shards** to local storage
06. **Reshard** into train/val/test virtual HDF5 datasets
07. **Compute normalization statistics** (mean, std across training set)
08. **Write `dataset.json`** — self-describing dataset card
09. **Register dataset in W&B** — log as artifact with spec, card, and metrics (§8)
10. **Upload finalized dataset** to R2
11. **Write `dataset.complete`** — completion marker (last step)

**`dataset.complete` semantics:**

`dataset.complete` means **finalization is done**. It is not a mutex, not an in-progress marker, and does not provide mutual exclusion.

- Written as the very last step, after all outputs are uploaded and verified
- Contains: `run_id` and finalization timestamp
- If `dataset.complete` exists and all outputs validate → dataset is ready for training
- If `dataset.complete` exists but outputs are missing → stale marker from a crashed finalize, cleaned up on next run
- Two concurrent finalize processes both write `dataset.complete` — this is fine, they produce identical outputs (§7.6)

**Why `dataset.complete` and not `dataset.lock`:** The file is a completion marker, not a lock. Calling it `.lock` implies mutex semantics that don't exist and can't exist (R2 has no atomic test-and-set). The name should communicate what it means: finalization is complete.

**Finalize idempotency:** Finalize reruns from scratch unless `dataset.complete` plus all finalized outputs are present and valid. No partial checkpoints — if finalize crashes after `train.h5` but before `stats.npz`, the next run starts over. This is simple and correct: finalize processes data already in R2, so reruns are cheap (minutes).

**`dataset.json` — dataset card:**

The output artifact metadata — a self-describing card for the finalized dataset. It answers "what is this dataset and how do I use it?" without requiring access to the metadata directory.

The input spec defines what the run should produce. `dataset.json` is the *output* record (what was actually produced). The spec has hundreds of shard-level entries. `dataset.json` inlines only what someone needs to load and use the dataset, and references the spec by SHA-256.

**Inlined:** provenance (code version, git dirty, param spec, renderer), structure (splits, total samples), stats (normalization values), validation summary.
**Referenced:** full spec via `input_spec_sha256` and `input_spec_path`.
**Excluded:** worker reports and debug logs — these are process artifacts, not dataset metadata.

### 7.7 Concurrency Semantics

This is a single-user research pipeline running 1-2x/week. It is not designed for concurrent operation, but it is **safe** under concurrent operation. Nothing gets corrupted — you just waste compute.

**Why concurrent operations can't corrupt data:**

1. **Deterministic outputs.** Two workers computing the same shard (same seed, same config) produce identical content. No randomness or timestamp-dependent state in the output.
2. **R2 last-writer-wins.** R2 (like S3) has PUT-overwrites-PUT semantics. When both writers produce identical content, the overwrite is a no-op in practice.

**Scenario: `generate` run on the same run_id multiple times in quick succession**

Both invocations read R2, both see the same missing shards, both launch workers for the same shards. Two workers race to write `shard-000042.h5`:

1. Both workers render with the same seed → identical HDF5 content
2. Both upload to the same canonical path in R2
3. R2 accepts both PUTs unconditionally — no conditional write, no ETag check, no if-none-match
4. `rclone copy --checksum` may skip the upload if the content already matches (rclone checks hash before uploading)
5. **Result:** identical shard in R2, wasted compute

**R2 does not reject duplicate writes.** There is no mechanism in the R2/S3 API (as used via rclone) to reject a write to an existing key. Writes always succeed. The design accepts this because deterministic outputs make overwrites harmless.

**What semantics do we want?** Skip-if-valid. The worker should check if the shard already exists and validates before uploading. This is implemented at the worker level (check canonical path before upload), not at the storage level (R2 has no conditional PUT via rclone). Even if the check races, the fallback (overwrite with identical content) is safe.

**Scenario: concurrent `finalize` on the same run_id**

Two finalize invocations both read the input spec, validate shards, download, reshard, upload, and write `dataset.complete`. Both produce identical outputs from the same validated shards. `dataset.complete` does not provide mutex semantics — R2 has no atomic test-and-set. The marker's purpose is to let subsequent invocations skip finalization ("already finalized"), not to prevent concurrent finalization. **Result:** identical outputs, wasted compute.

**Scenario: accidentally-launched finalize while another finalize is running**

Same as above. Both produce identical outputs. The second finalize either:

- Sees `dataset.complete` from the first (if it finished) → exits with "already finalized"
- Doesn't see it yet → runs to completion independently, produces identical outputs

No data corruption either way.

**Scenario: `generate` while `finalize` is running**

Finalize takes a snapshot of shard state during its validation pass. If generate launches new workers that upload shards during finalize's download phase, finalize either already has a consistent snapshot or sees additional valid shards (which only helps). Neither case produces a corrupt dataset.

**Scenario: `finalize` while workers are still uploading**

Finalize checks completeness first. If shards are missing, it reports "generation incomplete" and exits 1. No partial dataset is produced.

**Caveat: bad write can overwrite good write.** If `generate` is run twice in quick succession and the second invocation's worker produces a corrupt shard (e.g., VST crash mid-render) for a shard that was valid from the first invocation, R2 accepts the corrupt overwrite. **Mitigation:** Workers must validate shard integrity locally *before* uploading to R2. If local validation fails, the worker skips the upload and logs the failure. Only validated shards reach R2. This is the critical safety property: the upload path is `render → validate locally → upload only if valid`.

**What this system does NOT protect against:**

- **Non-deterministic outputs.** If floating-point non-determinism across different hardware produces different audio for the same seed, last-writer-wins means the final shard is arbitrary. The mitigation is to fix non-determinism in the renderer.
- **Concurrent spec modification.** The spec is written once and never modified. If something modifies it after creation, all bets are off.

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

- **Lifecycle markers** (`shards/shard-{id}/{worker_id}-{attempt}.rendering` / `.valid`) — empty files tracking attempt state. Orphaned `.rendering` without `.valid` = crashed attempt.
- **Worker report** (`worker_attempts/{worker_id}-{attempt}/report.json`, JSON) — derived summary with content hashes, written at end of execution, missing if worker crashed
- **Debug log** (`worker_attempts/{worker_id}-{attempt}/debug.log`, JSONL) — append-only narrative, uploaded by EXIT trap, survives crashes

Unique filenames per attempt mean retries never overwrite previous artifacts. Missing worker metadata never blocks completion. A run is successful when all shard files exist and validate.

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

## 8. Experiment Tracking (Weights & Biases)

W&B serves as a lightweight observability layer for the pipeline — a few key metrics and the dataset as a first-class artifact. It is not a monitoring dashboard or a log aggregator.

### What Goes in W&B

**Pipeline metrics** (logged by `finalize`):

| Metric                             | Type  | Description                                           |
| ---------------------------------- | ----- | ----------------------------------------------------- |
| `pipeline/shards_total`            | int   | Total shards in spec                                  |
| `pipeline/shards_valid`            | int   | Shards that passed validation                         |
| `pipeline/shards_quarantined`      | int   | Shards moved to quarantine                            |
| `pipeline/total_samples`           | int   | Total samples across all shards                       |
| `pipeline/generation_time_seconds` | float | Wall clock: spec created_at → last shard uploaded     |
| `pipeline/finalize_time_seconds`   | float | Wall clock: finalize start → dataset.complete written |
| `pipeline/errors_total`            | int   | Total errors across all worker reports                |

**Dataset artifact** (logged by `finalize`):

The finalized dataset is registered as a W&B Artifact of type `"dataset"`:

- **Files included:** `input_spec.json`, `dataset.json` (the card)
- **Metadata:** run_id, param_spec, code_version, is_repo_dirty, total_samples, split sizes
- **References:** R2 path to the actual HDF5 data (not uploaded to W&B — too large)

This creates a dataset entry in the W&B artifact registry that can be referenced by training runs, establishing **artifact lineage**: code version → dataset artifact → training run → model checkpoint. See [§14.11](#1411-wb-integration) for the implementation.

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
- **It duplicates reconciliation.** Storage already answers "is this shard done?" Adding a second, weaker signal creates two sources of truth.
- **It couples the protocol to provider lifecycles.** Pods are persistent and inspectable; serverless functions are ephemeral. A `check_tasks` method pretends providers are more similar than they are.
- **It tempts you into building a job supervisor.** Once you can check status, you'll want to retry, timeout, notify — and now you're building a scheduler instead of a data pipeline. We are not in the business of managing provider observability.

### 9.4 Hadoop / MapReduce

**Rejected.** Hadoop is designed for processing existing large datasets with shuffle, sort, and reduce phases over HDFS. This pipeline's workload is fully parallel data *generation* — no inter-worker communication, no data dependencies, no reduce step. Hadoop's infrastructure (HDFS cluster, YARN resource manager, JVM-based framework) is massive overkill. The pipeline would use Hadoop as an expensive pod launcher, ignoring everything that makes Hadoop Hadoop.

### 9.5 `make status` as Provider-Status Command

**Rejected (was the original monitoring approach).** The first draft described polling as showing live pod status from RunPod's API.

- **Provider APIs answer the wrong question.** "Is the worker running?" ≠ "are the shards done?" A worker can be running and producing nothing.
- **It's not portable.** Polling RunPod workers is RunPod-specific.
- **It tempts you into building monitoring tooling** — status enums, timeout heuristics, provider-specific error parsing — when the actual information needed is "which shards are missing?"

### 9.6 CLI Flags as Run Configuration

**Rejected (was the original approach).** Run configuration via Make variables: `make generate PARAM_SPEC=... SHARD_SIZE=... NUM_SHARDS=...`.

- **Not versionable.** Exact flags exist only in shell history.
- **Drift silently.** Same command with slightly different flag = different dataset, no detection.
- **Mix concerns.** Dataset spec and operational config in the same flat namespace.

The design uses typed YAML config files. The config describes what to produce; the input spec freezes it. Operational concerns (worker count, backend) are CLI arguments.

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
- We do not build or own RunPod worker health monitoring — the engineering effort is not justified for 1-2x/week usage, and provider observability is the provider's responsibility

**After a run:**

- `dataset.json` contains the complete output record
- W&B shows pipeline metrics and dataset artifact lineage
- Debug logs in R2 (`metadata/worker_attempts/{worker_id}-{attempt}/debug.log`) provide full JSONL debug log streams, queryable with `jq`
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

- **Conditional writes.** No `If-None-Match` or `If-Match` headers via rclone. Writes are unconditional (last-writer-wins). See [§7.6](#76-concurrency-semantics) for why this is acceptable.
- **Atomic multi-object writes.** Writing `shard-042.h5` and `worker-{id}.json` are two separate PUTs. They can't be made atomic. See [§11.2](#112-failure-modes--edge-cases).
- **Read-your-writes across regions.** R2 is globally distributed; the pipeline assumes single-region usage (all operations from one R2 endpoint).

### 11.2 Failure Modes & Edge Cases

Non-obvious failure modes, edge cases, and blind spots. Each includes the scenario, consequence, and mitigation.

**Corrupt shard overwrites valid shard:**
A second `generate` invocation assigns shard-042 to a new worker. The worker's VST plugin crashes mid-render, producing a corrupt file. The worker uploads it to R2, overwriting the valid shard from the first invocation. **Consequence:** Reconciliation may accept the corrupt shard if the corruption is subtle (e.g., wrong values but valid structure). **Mitigation:** Workers validate shards locally before uploading ([§7.4](#74-shard-validation)). If local validation fails, the upload is skipped. This is the primary defense.

**Non-atomic cross-file writes:**
A worker uploads `shard-042.h5` successfully but crashes before writing `worker-{id}.json`. Or: the worker writes `worker-{id}.json` (claiming success for shard-042) but crashes before the shard upload completes. These two artifacts are separate R2 PUTs — they cannot be made atomic. **Consequence:** Worker report may be out of sync with actual shard state. A report may claim a shard succeeded when the file doesn't exist, or a shard may exist without a corresponding report. **Mitigation:** Reconciliation never trusts worker reports for correctness ([§7.1](#71-storage-as-the-source-of-truth)). It validates actual shard files. Worker reports are supplementary debugging hints. Each worker invocation includes an attempt UUID, so mismatches between report and shards are at least observable.

**Partial shard upload:**
`rclone` crashes mid-upload. R2 may have a partial or corrupt object at the canonical path. **Consequence:** Reconciliation finds the file but it fails structural validation (not valid HDF5). **Mitigation:** The shard is quarantined and treated as missing. Next `generate` fills the gap.

**Silent data corruption (bit rot / transfer corruption):**
Local disk corruption between render and upload, or network corruption during transfer, produces a shard in R2 that differs from what the worker intended. **Consequence:** Corrupt shard passes filename checks but contains wrong data. **Mitigation:** All rclone operations use `--checksum`, which verifies content hashes after transfer. If a checksum mismatch is detected, the worker must delete the local shard, re-generate, and re-upload. Storage-layer bit rot within R2 is handled by R2 internally (server-side object checksums).

**Slow worker overtaken by retry:**
Worker A is assigned shard-042. Worker A is slow. User runs `generate` again, reconciliation sees shard-042 as missing, assigns to Worker B. Worker B completes first (valid). Worker A completes later (also valid, same content — deterministic). Worker A's upload overwrites Worker B's. **Consequence:** None if deterministic. If non-deterministic across hardware, last-writer-wins produces arbitrary result. **Mitigation:** Fix renderer non-determinism.

**Spec deleted after generation starts:**
Workers receive shard assignments via environment variables at launch. If the spec is deleted from R2 after launch, workers continue fine but subsequent `status`/`generate` fail. **Consequence:** Orphaned run. **Mitigation:** Spec is immutable, should never be deleted.

**`dataset.complete` exists but outputs are corrupt:**
Finalize wrote `dataset.complete` but outputs were later corrupted. **Mitigation:** Finalize validates outputs when `dataset.complete` exists. If missing/corrupt, deletes stale marker and reruns.

**Stale `dataset.complete` after re-generation:**
User runs `generate` after finalization (e.g., to replace a quarantined shard). `dataset.complete` from old finalize still exists. **Mitigation:** Finalize re-validates outputs against current spec. Mismatches trigger stale marker deletion and rerun.

## 12. Open Questions, Risks & Limitations

### Known Limitations

1. **Single-machine finalize bottleneck.** Finalize downloads all shards to one machine. At 480 shards × 10k samples (~30GB), this takes minutes. At 10M+ samples, finalize would need a cloud worker with large local storage, or a two-pass statistics approach.

2. **No incremental finalization.** Crashes during finalize restart from scratch. Acceptable because finalize processes existing data and is fast to retry.

3. **Reproducibility is controlled-conditions, not absolute.** The pipeline guarantees that the same spec + same Docker image + same hardware = identical dataset. But VST plugin floating-point behavior can vary across CPU architectures (x86 vs ARM, SSE vs AVX), and Docker base image updates change system libraries. Content hashes in worker reports detect when this happens, but the pipeline does not enforce cross-hardware bit-identity. Concurrency safety also depends on deterministic rendering — if two workers produce different output for the same seed, last-writer-wins makes the result arbitrary.

4. **R2 listing at scale.** Reconciliation lists all shard objects (1000/page). At 480 shards: 1 API call. At 48,000: 48 calls — still fast, but a completion-marker approach would scale better.

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

The pipeline currently uses HDF5 exclusively. A `ShardWriter`/`ShardReader` protocol could allow swapping to WebDataset (streaming), Parquet (columnar), or Lance (versioned). This should be added when a second format is concretely needed, not speculatively.

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
    renderer: str
    sample_rate: int
    shard_size: int
    num_shards: int
    splits: dict[str, int]  # {"train": 44, "val": 2, "test": 2}
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
    renderer: str
    sample_rate: int

    # Structure
    total_samples: int
    splits: dict[str, int]  # {"train": 440000, "val": 20000, "test": 20000}
    stats: dict[str, float]

    # Integrity
    validation_summary: ValidationSummary

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
    error: str | None = None

class WorkerReport(BaseModel):
    model_config = ConfigDict(strict=True)
    worker_id: str          # Infrastructure ID (for debugging)
    assigned_shards: list[int]
    results: list[ShardResult]
    errors: list[str]
    started_at: str         # ISO 8601
    completed_at: str
```

### 14.4 Config Materialization

A run starts from a typed YAML config file:

```yaml
# configs/pipeline/surge_simple_480k.yaml
experiment_name: surge_simple
param_spec: surge_simple
renderer: surge-xt-1.3.1
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
2. Derive `run_id`: `{experiment_name}-{train_size}-{shard_size}-{YYYYMMDD-HHMMSS}`
3. Materialize `PipelineSpec` — expand config into shard-level spec (seeds, shapes, row ranges)
4. Upload spec + source config to R2
5. Proceed with reconciliation

**Config drift protection:** If `--config` is passed for a `run_id` that already has a spec, the command errors. The spec is authoritative after creation.

### 14.5 Run ID Format

```
{experiment_name}-{total_train_samples}-{shard_size}-{YYYYMMDD-HHMMSS}
Example: surge_simple-480k-10k-20260312-143022
```

### 14.6 Structured Logging

Implementation of debug logging described in [§7.8](#78-error-handling--crash-resilience).

Workers use `structlog` with JSON rendering for append-only debug log streams:

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

Structured output goes to stdout (live debugging via `docker logs`). The entrypoint tees stdout to a local file, uploaded to R2 as `worker-{id}-debug.log` by the bash EXIT trap — survives crashes.

### 14.7 Retry Policy

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

### 14.8 CLI & Directory Structure

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

### 14.9 LocalBackend Storage Invariant

Local mode mimics R2 exactly: same directory structure, same spec format, same shard naming, same validation. Only the storage transport changes. This makes local mode a high-fidelity test of the production pipeline.

### 14.10 Overwrite and Retry Policy

- **Only one worker should be assigned a shard at a time.** The CLI partitions the missing set.
- **A shard may only be overwritten if missing or failing validation.** Workers check the canonical path before uploading; if a valid shard exists, they skip it.
- **If duplicate writes occur** (two workers racing), both produce identical content (deterministic), so the overwrite is harmless.
- **Finalize never merges ambiguous duplicates.** One canonical path per shard, no deduplication logic.

### 14.11 W&B Integration

Implementation of experiment tracking described in [§8](#8-experiment-tracking-weights--biases).

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

## 15. Discarded Choices

Alternatives considered but not impactful enough for detailed analysis.

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
| **Debug log**              | JSONL file (`metadata/worker_attempts/{worker_id}-{attempt}/debug.log`) of structured events from a worker. Append-only, uploaded by EXIT trap, survives crashes.                                                                                                                                                  |
| **Worker report**          | JSON summary (`metadata/worker_attempts/{worker_id}-{attempt}/report.json`) of a worker's results, including content hashes for provenance. Written at exit, missing if worker crashed.                                                                                                                            |
| **Lifecycle marker**       | Empty file in `metadata/shards/shard-{id}/` named `{worker_id}-{attempt}.{state}`. States: `.rendering` (in progress), `.valid` (validated + uploaded), `.invalid` (failed reconciliation). Presence is the state — no content to parse.                                                                           |
| **Quarantined shard**      | A corrupt shard moved to `metadata/shards/shard-{id}/quarantine/` for debugging. Kept alongside lifecycle markers so `rclone ls` of a shard's metadata directory shows the full history.                                                                                                                           |
| **Dataset card**           | JSON file (`dataset.json`) describing the finalized dataset: provenance, structure, stats. References the spec by SHA-256.                                                                                                                                                                                         |
| **param_spec**             | Configuration selecting which synthesizer parameters to vary. Determines prediction task dimensionality. Examples: `surge_simple` (92 params), `surge_xt` (189 params).                                                                                                                                            |
| **VST**                    | Virtual Studio Technology — plugin format for audio synthesizers. Surge XT is the VST used for rendering.                                                                                                                                                                                                          |
| **Mel spectrogram**        | Frequency-domain audio representation used as neural network input. 128 mels, ~100 frames/sec.                                                                                                                                                                                                                     |
| **Fully parallel**         | Workload where tasks are completely independent — no communication or shared state between workers.                                                                                                                                                                                                                |
| **rclone**                 | CLI tool for syncing files to cloud storage. Used as the R2 upload/download mechanism.                                                                                                                                                                                                                             |

## Appendix B: Tech Stack

| Component       | Technology                                                        | Role                                             |
| --------------- | ----------------------------------------------------------------- | ------------------------------------------------ |
| Build           | Docker (BuildKit)                                                 | Reproducible compute environments                |
| Storage         | Cloudflare R2                                                     | Data + coordination, free egress                 |
| Execution       | RunPod                                                            | Cheap on-demand cloud workers                    |
| Tracking        | Weights & Biases                                                  | Pipeline metrics, dataset artifact registry      |
| Data format     | [HDF5](https://www.h5py.org/) (h5py + hdf5plugin)                 | Shard storage with compression                   |
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

______________________________________________________________________
