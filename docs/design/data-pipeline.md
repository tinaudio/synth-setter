# Design Doc: Data Pipeline

> **Status**: Draft
> **Author**: ktinubu@
> **Last Updated**: 2026-07-16
> **Tracking**: #74
> **Storage conventions**: [storage-provenance-spec.md](storage-provenance-spec.md)

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

**synth-setter** is a collection of tools for synthesizer inversion, sound matching and preset exploration.

Training models for these tasks requires large-scale datasets: 500k–15M audio samples, each rendered through a real VST synthesizer plugin (Surge XT) with random parameter configurations. Each sample produces an audio waveform, mel spectrogram, and ground-truth parameter array, stored as a Lance dataset shard. This rendering is CPU-bound — each sample requires a real-time audio render through the plugin — and takes hours to days on a single machine.

### Prior Work

The core generation and training infrastructure was built by benhayes@: the VST rendering engine (`generate_vst_dataset.py`), a comprehensive parameter specification system covering Surge XT's full parameter space (`param_spec.py`, `surge_xt_param_spec.py` — ~1300 lines of sampling, encoding, and semantic representation), plugin loading with audio and mel extraction (`core.py`), and the PyTorch Lightning DataModule (`vst_datamodule.py`). Beyond the generation code, benhayes@ built an extensive Hydra configuration system — 40+ composable experiment configs across multiple datasets (Surge, k-osc, k-sin, FM, FSD, NSynth), multi-logger support (W&B, TensorBoard, MLflow), Optuna integration for hyperparameter search, and SGE job scripts for QMUL's HPC cluster with proper resource management, array jobs, and W&B checkpoint retrieval. This is a well-structured research codebase with strong configuration practices, and it remains the foundation the distributed pipeline builds on.

On top of this, ktinubu@ added sequential orchestration (`run_dataset_pipeline.py`), cloud storage integration via rclone (`uploader.py`), a containerized execution environment with Docker, a first parallelization attempt with per-instance shards (`generate_shards.py`), RunPod scaling (`runpod_launch.py`), and post-generation finalization (`finalize_shards.py`).

At scales up to hundreds of thousands of samples, this pipeline works well. It resumes at the shard level, tracks basic provenance, and produces correct datasets.

At research scale (500k–15M samples), the single-machine approach breaks down. Generation takes days to weeks. The entire dataset must fit on local disk for both generation and training — at 15M samples this is potentially terabytes, a hard blocker on dataset size with no streaming or remote-access path in the repo. There is no crash resilience: a single failure loses the entire run. There is no per-shard validation, so corrupt data enters training silently. There is no way to regenerate only the failed shards without restarting everything.

### Distributed Pipeline

> **Implementation status:** Single-machine sequential multi-shard generation is
> implemented today (`src/synth_setter/cli/generate_dataset.py` loops over
> `spec.shards`, skipping shards already present in R2 — worker-side
> resumability MVP per #750; the launcher-side reconciliation engine described
> in §7.4 / §7.7 is not yet built). When `render.parallel=True`, owned shards
> dispatch concurrently via a thread pool sized to half the worker's
> affinity-aware CPU count; transient renderer subprocess failures are retried
> up to `render.max_retries` times (default 0 = strict fail-fast).
> `use_shard_queue=true` instead claims shard IDs dynamically from the run's
> Lance shard-claims table (§7.1); claims mode renders one claim at a time and
> ignores `render.parallel`. The
> distributed/parallel pipeline described below — CLI, backends, reconciliation,
> and finalize stages — is the design target and not yet built.

The distributed data pipeline solves this by splitting generation across N cloud workers on **[RunPod](https://www.runpod.io/)** (a GPU/CPU cloud marketplace offering cheap on-demand compute), each independently producing shards in parallel. Workers write shards to **[Cloudflare R2](https://developers.cloudflare.com/r2/)** (an S3-compatible object storage service with free egress), which serves as both the data store and the coordination layer. A separate finalize step commits the winning shard attempts into train/val/test split datasets (for Lance, a metadata-only fragment commit — no shard download or row rewrite), reduces normalization statistics, and registers the dataset as a **[Weights & Biases](https://wandb.ai/)** (W&B) artifact.

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
# 1. Pick an experiment config. The filename stem (e.g. `surge-simple-480k-10k`)
#    is the dataset_config_id / task_name; the Hydra config-group path you pass
#    as `experiment=` is `generate_dataset/<stem>`.
#    Hydra composes the final DatasetSpec from src/synth_setter/configs/dataset.yaml + this overlay.
cat src/synth_setter/configs/experiment/generate_dataset/surge-simple-480k-10k.yaml
# → task_name: surge-simple-480k-10k, defaults: [/datamodule: surge_simple, /render: surge_simple, ...], ...

# 2. Run multi-shard generation on a single worker (default sequential loop;
#    `render.parallel=true` opts into thread-pool parallel dispatch).
python -m synth_setter.cli.generate_dataset experiment=generate_dataset/surge-simple-480k-10k
# → Loops over spec.shards, skipping shards already present in R2 (worker-side resumability MVP, #750).
# **Planned CLI** — the distributed pipeline CLI (`python -m synth_setter.pipeline generate/status/finalize`)
# is not yet implemented; `generate_dataset` is the current MVP, deprecated when
# `generate-shards` lands on main (#411).

# --- Target state (distributed pipeline, not yet implemented) ---
# python -m synth_setter.pipeline generate --experiment generate_dataset/surge-simple-480k-10k --workers 10
# → Created run surge-simple-480k-10k-20260313T100000123Z
# → Launched 10 workers for 48 shards
# → Exiting. Run 'status' to check progress.
#
# python -m synth_setter.pipeline status --run-id surge-simple-480k-10k-20260313T100000123Z
# → Valid: 44/48  Missing: 2  Quarantined: 2
#
# python -m synth_setter.pipeline generate --run-id surge-simple-480k-10k-20260313T100000123Z
# → 4 shards missing, launching 1 worker
#
# python -m synth_setter.pipeline finalize --run-id surge-simple-480k-10k-20260313T100000123Z
# → 48/48 valid. output_format: lance
# → Committing winner fragments → train.lance/, val.lance/, test.lance/
# → Stats reduced from shard sidecars. Dataset registered in W&B as data-surge-simple-480k-10k.
# → dataset.complete written (relocates under metadata/ with #385).
```

Make targets are thin aliases for convenience:

> **Not yet implemented.** These `make` targets are planned but do not exist in the Makefile yet ([#72](https://github.com/tinaudio/synth-setter/issues/72)).

```bash
make generate ARGS="--experiment generate_dataset/surge-simple-480k-10k --workers 10"
make status ARGS="--run-id surge-simple-480k-10k-20260313T100000123Z"
make finalize ARGS="--run-id surge-simple-480k-10k-20260313T100000123Z"
```

## 3. Goals, Non-Goals & Design Principles

### Goals

- **Reproducible pipeline with full provenance.** The input spec freezes all generation parameters: seeds, shapes, splits, and renderer version. Re-running from the same spec on the same hardware and Docker image produces an identical dataset. This is a *controlled-conditions* guarantee, not an absolute one: VST plugin floating-point behavior may vary across CPU architectures, and Docker base image updates could change system libraries. The pipeline records enough provenance to detect and diagnose these differences (git commit, `is_repo_dirty: bool`, renderer version, per-shard content hashes), but does not claim bit-identical output across arbitrary hardware. Provenance matters because this is ML research: when a model behaves unexpectedly, you need to trace from the trained model back to the exact dataset, the exact code that generated it, and which worker attempt produced each shard. Per-shard provenance is tracked via lifecycle markers and content hashes in worker reports. See [Deterministic Dataset Seeding](deterministic-seeding.md) for the seed contract.
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

A CLI running on the user's local machine reads a spec (desired state), lists existing validated shards in R2 (actual state), computes the difference, and launches N workers to produce the missing shards. Each worker independently renders audio samples through a VST plugin and writes shard attempts to R2 — uncommitted Lance fragments for the primary format. When all shards are present, a separate finalize command commits the winning fragments into train/val/test split datasets, reduces normalization statistics, registers the dataset in W&B, and writes a completion marker.

R2 serves as both the **data plane** and the **control plane**:

| Plane             | What flows through it  | Examples                                                      |
| ----------------- | ---------------------- | ------------------------------------------------------------- |
| **Data plane**    | Actual dataset content | Lance fragment data + split datasets, stats.npz               |
| **Control plane** | Coordination metadata  | Spec, worker reports, debug logs, `metadata/dataset.complete` |

Both planes use R2. There is no separate database, message queue, or coordination service. This means one piece of infrastructure to manage, one set of credentials, one failure mode to reason about. The trade-off: the pipeline does not rely on conditional writes for data-plane mutual exclusion — all operations are idempotent and produce deterministic outputs; the opt-in shard-claims table ([§7.1](#71-storage-as-the-source-of-truth)) is the sole CAS consumer ([§7.7](#77-concurrency-semantics)).

### Reconciliation Correctness

What if reconciliation itself has a bug — e.g., it validates a corrupt shard as good?

- **Defense in depth:** Workers run validation checks before upload; all must pass ([§7.5](#75-shard-validation)).
- **Tiered validation:** Full validation (4-check design target: structural, shape, value, row count — [#103](https://github.com/tinaudio/synth-setter/issues/103)). Current implementation: 3-check (valid Lance fragment, expected columns, row count). Shape and value checks are planned. Finalize structural-checks staged shards before committing. Each tier catches a different class of failure.
- **Training is the ultimate check:** A corrupt dataset will fail to train properly, providing end-to-end verification.
- **Manual spot-checking is feasible:** At 1-2 runs/week, eyeballing a few shards is practical and encouraged.

## 5. Stage Definitions

> **Implementation status.** Workers stage per-attempt fragment sidecars, Welford stats, and `.rendering`/`.valid` markers under `metadata/workers/shards/shard-{id}/` (`pipeline/data/lance_staging.py`), and finalize selects winners and commits their fragments into the split manifests (`pipeline/data/lance_finalize.py`). Not yet implemented: `.invalid`/quarantine and `.promoted` markers, and worker reports.

The pipeline has two stages. Each is an independent command with well-defined inputs and outputs.

| Stage        | Command                 | Input                                     | Output                                                                                                | Compute                       |
| ------------ | ----------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------- |
| **Generate** | `pipeline.cli generate` | Config YAML (first run) or spec (retries) | Lance shard attempts in R2                                                                            | CPU — VST audio rendering     |
| **Finalize** | `pipeline.cli finalize` | Validated shard attempts in R2            | `train.lance/`, `val.lance/`, `test.lance/`, `stats.npz`, `dataset.json`, `metadata/dataset.complete` | CPU — validate, commit, stats |

Finalize commits the winning worker fragments into the split datasets:

| `output_format` | Finalize outputs                                                            | Training access pattern            |
| --------------- | --------------------------------------------------------------------------- | ---------------------------------- |
| `lance`         | `train.lance/`, `val.lance/`, `test.lance/` committed from worker fragments | Columnar random access / streaming |

Each worker container runs with `MODE=generate-shards` — the entrypoint mode IS the worker. Scoped and validated on the `experiment` branch; pending port to main ([#407](https://github.com/tinaudio/synth-setter/issues/407)).

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
┌────────────────────────────────┐
│  make generate RUN_ID=...      │
│  (CLI — local machine)         │
│                                │        ┌──────────────────┐
│  1. Validate auth (R2+RunPod)  │        │  Cloudflare R2   │
│  2. Read/create spec     ◄─────┼───────►│                  │
│  3. List staged shards   ◄─────┼────────┤  data/{cfg}/{id}/ │
│  4. Validate staged shards     │        │   metadata/       │
│  5. Compute missing set        │        │   workers/        │
│  6. Partition across N workers │        │                  │
│  7. Submit N tasks             │        │                  │
│  8. Exit                       │        │                  │
└────────────────────────────────┘        │                  │
                                          │                  │
         ┌───────────────────┐            │                  │
         │  Worker 1         │───────────►│  metadata/       │
         │  (RunPod worker)  │            │   workers/       │
         │  shards 0-47      │            │   shards/        │
         └───────────────────┘            │                  │
         ┌───────────────────┐            │                  │
         │  Worker N         │───────────►│  (staging)       │
         │  shards 432+      │            │                  │
         └───────────────────┘            │                  │
                                          │                  │
┌────────────────────────────────┐        │                  │
│  make finalize RUN_ID=...      │        │                  │
│  (local or cloud)              │        │                  │
│                                │        │  data/           │
│  1. Read spec            ◄─────┼────────┤   train.*        │
│  2. Validate attempts    ◄─────┼────────┤   val.*          │
│  3. Select winners       ◄─────┼────────┤   stats.npz      │
│  4. Commit fragments     ──────┼───────►│   metadata/       │
│  5. Reduce stats              │         │   dataset.json   │
│  6. Register in W&B           │         │                  │
│  7. Write completion marker │         │   metadata/dataset.complete │
└────────────────────────────────┘        └──────────────────┘
```

### R2 File Structure

> **Implementation status — Workers stage attempts under `metadata/workers/shards/` and write fragment data under the split dataset `data/` directories (#1776).**

The canonical R2 bucket layout — root path, top-level prefixes, and per-workflow contents — is defined in [storage-provenance-spec.md §2](storage-provenance-spec.md#2-r2-bucket-layout) and [§3a](storage-provenance-spec.md#3a-data-generation). The data pipeline writes under `data/{dataset_config_id}/{dataset_wandb_run_id}/`: workers stage per-attempt artifacts under `metadata/workers/` and may write uncommitted fragment data under split dataset `data/` directories, but only `finalize` writes Lance manifests, transactions, `stats.npz`, and `metadata/dataset.{json,complete}`. Datasets are immutable once `metadata/dataset.complete` exists; new versions require a new `dataset_wandb_run_id`.

Pipeline-specific staging conventions (per-attempt shard filenames, lifecycle markers, quarantine layout) are additive detail — see [Artifact Taxonomy](#artifact-taxonomy) below.

### Artifact Taxonomy

> This section covers **pipeline file artifacts** — the structured files the data pipeline produces in R2, their producers and consumers, and the shard attempt lifecycle. The canonical R2 paths are defined in [storage-provenance-spec.md §3a](storage-provenance-spec.md#3a-data-generation); the **W&B artifact taxonomy** (dataset/model/eval-results artifact types) is defined in [storage-provenance-spec.md §4](storage-provenance-spec.md#4-wb-artifact-types).

All structured files in the pipeline, in one place:

```
                     ┌────────────────┐
                     │  config.yaml   │ ─── User-authored recipe
                     │  (user input)  │     Human-written YAML
                     └───────┬────────┘
                             │
                   pipeline.cli generate  ← creates on first run
                             │
                     ┌───────▼──────────┐
                     │ input_spec.json  │ ─── Frozen input specification
                     │ (immutable)      │     Machine-generated, write-once
                     └───────┬──────────┘
                             │
                   Workers (RunPod / local)  ← submitted by generate
                             │
              ┌──────────────┼──────────────────────┐
              │              │                      │
       ┌──────▼────────┐  ┌──▼──────────┐  ┌────────▼───────┐
       │ {w}-{a}.*     │  │ report.json │  │ debug.log      │
       │ (attempt      │  │ (worker     │  │ (worker        │
       │  payload)     │  │  summary)   │  │  debug log)    │
       └──────┬────────┘  └─────────────┘  └────────────────┘
              │
       ┌──────▼────────┐  All worker output → metadata/workers/
       │ .rendering    │
       │ .valid        │
       │ .invalid      │
       │ (lifecycle)   │
       └──────┬────────┘

     pipeline.cli finalize  ← validates + commits attempts
              │
       ┌──────▼──────────┐
       │ {split}.lance   │ ─── Winner fragments committed per split
       │ manifests       │     Written ONLY by finalize
       │ (finalized)     │
       └──────┬──────────┘
              │
       ┌──────▼────────┐
       │ dataset.json  │ ─── Output record (dataset card)
       │ (output)      │     What was produced, how to use it
       └──────┬────────┘
              │
       ┌──────▼───────────────────┐
       │ metadata/dataset.complete │ ─── Completion marker
       │ (marker)                  │     "Finalization is done"
       └───────────────────────────┘
```

| Artifact               | Path                                                                    | Format     | Produced By                     | Consumed By                      |
| ---------------------- | ----------------------------------------------------------------------- | ---------- | ------------------------------- | -------------------------------- |
| Config                 | `metadata/config.yaml`                                                  | YAML       | User                            | `generate`                       |
| Input spec             | `metadata/input_spec.json`                                              | JSON       | `generate` (first run)          | `generate`, `status`, `finalize` |
| Shard-claims table     | `metadata/shard-claims.lance/`                                          | Lance      | `generate` (operator populate)  | Workers (claim/complete via CAS) |
| Lance fragment sidecar | `metadata/workers/shards/shard-{id}/{worker}-{attempt}.fragment.json`   | JSON       | Workers                         | `finalize` (commits fragment)    |
| Lance shard stats      | `metadata/workers/shards/shard-{id}/{worker}-{attempt}.shard-stats.npz` | NumPy      | Workers                         | `finalize` (stats reduction)     |
| Lance fragment data    | `train.lance/data/*`, `val.lance/data/*`, `test.lance/data/*`           | Lance      | Workers                         | `finalize` (manifest commit)     |
| Lifecycle marker       | `metadata/workers/shards/shard-{id}/{worker}-{attempt}.{state}`         | Empty file | Workers / Finalize              | `status`, humans                 |
| Quarantined shard      | `metadata/workers/shards/shard-{id}/quarantine/{worker}-{attempt}.*`    | Lance      | Workers (on validation failure) | Humans (debugging)               |
| Worker report          | `metadata/workers/attempts/{worker_id}-{attempt}/report.json`           | JSON       | Workers (at exit)               | `status`                         |
| Debug log              | `metadata/workers/attempts/{worker_id}-{attempt}/debug.log`             | JSONL      | Workers (continuous)            | Humans (`jq`)                    |
| Dataset card           | `metadata/dataset.json`                                                 | JSON       | `finalize`                      | Training scripts, humans         |
| Completion marker      | `metadata/dataset.complete`                                             | JSON       | `finalize` (last step)          | `finalize` (idempotency check)   |
| Stats                  | `stats.npz`                                                             | NumPy      | `finalize`                      | Training scripts                 |

> The `metadata/dataset.{json,complete}` placements are the [#385](https://github.com/tinaudio/synth-setter/issues/385) future state; today finalize writes both at the run prefix root (`R2Location.dataset_card_uri()` / `dataset_complete_marker_uri()`).

**Shard attempt lifecycle:** Each attempt produces a shard file and lifecycle markers in the shard's staging directory, all named `{worker_id}-{attempt_uuid}`:

| File / Marker                        | Written by                                                               | Meaning                                                                                                                                                                                                                                                                      |
| ------------------------------------ | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `{worker}-{attempt}.fragment.json`   | Worker (after Lance fragment creation)                                   | Strict Pydantic sidecar for a Lance attempt: a schema version plus Lance's serialized fragment metadata string. Logical identity is derived from the path, filename, and spec, not stored ([§14.4](#144-lance-fragment-sidecar-schema)).                                     |
| `{worker}-{attempt}.shard-stats.npz` | Worker (after Lance stats computation)                                   | Per-shard Welford state (`count`, `mean`, `m2`) for the Lance attempt. Finalize reduces only the winning attempts into dataset-level `stats.npz`.                                                                                                                            |
| `{worker}-{attempt}.rendering`       | Worker (at start)                                                        | Attempt started. Append-only — not deleted when `.valid` is written. Orphaned `.rendering` without a `.valid` indicates a crashed attempt.                                                                                                                                   |
| `{worker}-{attempt}.valid`           | Worker (last step of shard lifecycle)                                    | **Commit point for a staged attempt.** Written only after render, validation, upload, and bookkeeping are complete. `generate`/`status` uses this as the staging admission signal. Not sufficient for final dataset correctness — finalize validates again before promotion. |
| `{worker}-{attempt}.invalid`         | Worker (on validation failure) or Finalize (on structural check failure) | Shard failed validation. Corrupt shard uploaded to `quarantine/` for debugging.                                                                                                                                                                                              |
| `{worker}-{attempt}.promoted`        | Finalize                                                                 | Staged attempt was structural-checked and included in the finalized output. Lance attempts are committed into split dataset manifests.                                                                                                                                       |

Listing a shard's staging directory shows the full history — shard files, lifecycle markers, and quarantined attempts — at a glance:

```
$ rclone ls r2:bucket/{run_id}/metadata/workers/shards/shard-000042/
         0  pod-abc123-a1b2c3d4.rendering        # first attempt started (crashed — no .valid)
       412  pod-def456-e5f6a7b8.fragment.json    # second attempt's fragment sidecar
      2048  pod-def456-e5f6a7b8.shard-stats.npz  # second attempt's Welford stats
         0  pod-def456-e5f6a7b8.rendering        # second attempt started
         0  pod-def456-e5f6a7b8.valid            # second attempt committed (staged-valid)
         0  pod-def456-e5f6a7b8.promoted         # selected by finalize
```

**Naming conventions:** Config is YAML (human-authored). Machine-generated control-plane records are JSON validated with Pydantic. Debug logs are JSONL (one event per line). Numeric stats are NumPy `.npz` files to avoid JSON precision loss. Worker metadata lives under `metadata/workers/` — shard-attempt records and markers are grouped by shard ID (`metadata/workers/shards/shard-{id}/`), worker artifacts are grouped by attempt (`metadata/workers/attempts/{worker_id}-{attempt_uuid}/`). For Lance, workers may also write uncommitted fragment data under split dataset `data/` directories; finalize remains the only writer of committed dataset manifests and completion markers. Lifecycle markers are empty files — presence is the state, no content to parse.

## 7. Design Decisions

### 7.1 Storage as the Source of Truth

> **Implementation status — implemented (#1776): staging prefix, `.rendering`/`.valid` markers, and finalize's fragment commit.**

The pipeline uses R2 as both the data layer and the coordination layer. Integrity is guaranteed by validation markers, structural checks, and content hashes. Workers write shard-attempt metadata and markers to a **staging prefix** (`metadata/workers/shards/`) and write uncommitted fragment data directly under the split dataset directories; finalize commits the selected fragment metadata into the dataset manifests. Finalized data is stable once `metadata/dataset.complete` is written.

**Why R2 for coordination and not a database or queue:**

The design avoids depending on compare-and-set, locking, and transactions (R2 does offer conditional puts; only the opt-in shard-claims table uses them, through Lance's commit protocol). A traditional coordination system (Redis, Postgres, SQS) would provide these natively. We use R2 anyway — this is a deliberate trade-off, not a convenience choice.

The pipeline doesn't need what coordination systems provide:

- **No compare-and-set for canonical membership** — workers write to per-attempt metadata filenames, and Lance fragment object names are owned by Lance. Concurrent attempts for the same logical shard never decide canonical membership themselves; finalize selects the winners ([§7.7](#77-concurrency-semantics)).
- **No locking** — each shard is assigned to one worker per invocation. The assignment is a simple partition of the missing set. No lock acquisition needed.
- **No transactions** — stages are independent. There is no multi-shard operation that must succeed or fail atomically.
- **No queue for completion truth** — work discovery is reconciliation (compare spec against storage). A queue deciding "what work remains" would be a second source of truth — and a less reliable one than the files themselves. Optional *distribution* is different: `use_shard_queue=true` spreads shard IDs across machines dynamically via a tiny [Lance](https://lancedb.github.io/lance/) claims table under the run prefix (`metadata/shard-claims.lance`, one row per shard) whose conditional `UPDATE` rides Lance's conditional-put commit protocol — R2 does support compare-and-set puts. A single lease expiry written at claim time replaces heartbeating (a crashed worker's claim is reclaimable once the lease lapses) and a `claim_gen` column fences stale completers. The table only routes claims; the per-shard R2 probe stays the sole completion authority, so a stale or duplicated claim costs at most a skip-probe, never a wrong dataset.

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
3. **Canonical outputs** (the split datasets) — which shards finalize has committed. Once finalized, zombie worker uploads cannot affect the selected output set.
4. **Worker metadata** (reports, logs) — debugging hints, never authoritative, never block completion

This separation eliminates an entire class of edge cases:

- Worker crashes before upload → no staged file → reconciliation detects it
- Worker report claims success but upload failed → no staged file → reconciliation detects it
- Staged shard exists but report is missing → shard passes validation → it's complete
- Zombie worker uploads after finalize → goes to staging, doesn't touch the committed split datasets

### 7.2 Shard Lifecycle

> **Completeness rule:** A shard is **staged-valid** (ready for finalization) when a complete attempt and its `.valid` marker exist in `metadata/workers/shards/shard-{id}/` — that means `{attempt}.fragment.json` + `{attempt}.shard-stats.npz` + `{attempt}.valid`, with fragment data referenced by Lance's serialized metadata. The `.valid` marker is the **commit point** for a staged attempt — it means the worker completed rendering, validation, upload, and bookkeeping successfully. It is authoritative for staging admission but not for final dataset correctness; finalize remains the gate before commit ([§7.5](#75-shard-validation)). A shard is **finalized** when finalize includes the selected attempt in the output dataset and writes `metadata/dataset.complete`.

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

| State            | How it's determined                                                                                                                     | Where it lives                                                                   |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **missing**      | Shard is in the input spec but no `.valid` marker exists for any attempt                                                                | Implicit (absence). May have orphaned `.rendering` markers from crashed attempts |
| **rendering**    | `.rendering` marker exists but no `.valid` marker yet — attempt is in progress                                                          | `.rendering` marker in `metadata/workers/shards/shard-{id}/`                     |
| **staged-valid** | Attempt payload + `.valid` exist. Worker completed full lifecycle (render, validate, upload, bookkeeping). Ready for finalize to commit | `metadata/workers/shards/shard-{id}/{worker_id}-{attempt}.*` + `.valid` marker   |
| **invalid**      | Worker validation failed. Corrupt file uploaded to `quarantine/` for debugging                                                          | `.invalid` marker + `quarantine/{worker_id}-{attempt}.*`                         |
| **finalized**    | Finalize structural-checked and committed the selected attempt. Dataset is sealed                                                       | Committed split dataset + `.promoted` marker + `metadata/dataset.complete`       |

**Transitions:**

- **missing → rendering:** Worker begins shard generation, writes `.rendering` marker.
- **rendering → staged-valid:** Worker validates locally (full validation — 4-check design target: structural, shape, value, row count — [#103](https://github.com/tinaudio/synth-setter/issues/103); current implementation: 3-check), uploads the fragment sidecar and stats to staging, writes worker report, then writes `.valid` marker as the **last step**. The `.valid` marker is the commit point — it signals that the worker completed the full shard lifecycle (render, validate, upload, bookkeeping). The `.rendering` marker is not deleted — both remain visible, preserving the full timeline.
- **rendering → invalid:** Worker validates locally and the shard fails. Worker uploads the corrupt shard to `quarantine/` and writes `.invalid` marker, preserving the evidence for debugging. The shard is treated as missing on next `generate`.
- **rendering → missing:** Worker crashes before writing `.valid`. The `.rendering` marker is orphaned — observable evidence of the crashed attempt. Any fragment data uploaded before the crash exists but without `.valid` is not considered staged-valid.
- **staged-valid → finalized:** Finalize selects one valid attempt per logical shard, structural-checks the selected attempt, and commits it to the final output. It deserializes the selected `fragment.json`, lets Lance own fragment identity, and commits those fragments into the split dataset manifests. Finalize writes `.promoted` markers and `metadata/dataset.complete` after all outputs are complete. Staged files remain in place (append-only — no deletion).

**Key properties:**

- **`.valid` is authoritative for staging admission, not for final correctness.** `generate`/`status` trusts `.valid` markers for cheap reconciliation. Finalize does a structural check before committing — it is the gate for canonical data. Canonical truth lives in the committed split datasets, not in `.valid` markers.
- **Multiple attempts are visible.** A shard's staging directory might contain `pod-abc.rendering` (crashed, no `.valid`), `pod-def.fragment.json` + `pod-def.shard-stats.npz` + `pod-def.valid` (committed), and `pod-def.promoted` (finalized) — the full history is one `rclone ls` away.
- **Workers and finalize have separate authority.** Workers own per-attempt metadata and uncommitted payloads. Finalize owns canonical membership: Lance manifests, dataset stats, and completion markers. A zombie worker uploading after finalize cannot change the committed dataset.
- **The finalized state is per-shard and dataset-level.** Per-shard: selected final output membership + `.promoted` marker. Dataset-level: `metadata/dataset.complete` marker and selected-attempt manifest in `dataset.json`.

### 7.3 Deterministic Shard Identities

Shard IDs are logical and deterministic: `shard-000000` through `shard-000479`, with a Lance fragment sidecar set. Defined at run creation, independent of which worker computes them.

- **Any worker can compute any shard** — retries simply recompute the same logical shard
- **Resumability is a set difference:** `spec_shards - validated_shards = work_remaining`
- **No naming collisions** — each worker attempt writes unique sidecar filenames (`{worker_id}-{attempt_uuid}.*`), and canonical membership is decided only by finalize
- **Infrastructure details** (worker IDs) appear in staging filenames and metadata, not in canonical shard paths

**Work assignment:** The CLI partitions shards across N workers. Worker 1 gets shards 0-47, Worker 2 gets shards 48-95, etc. With `use_shard_queue=true`, assignment is instead a dynamic claim from the run's shard-claims table. Either way the shard's identity is independent of which worker computes it: if Worker 1 fails and its shards are reassigned to Worker 3, output paths are unchanged.

**Shard write protocol:**

1. Write `.rendering` marker: `metadata/workers/shards/shard-{id}/{worker_id}-{attempt_uuid}.rendering`
2. Render shard to a local temp file
3. **Validate locally** — structural checks plus row count. The design target is 4-check validation adding shape and value checks ([#103](https://github.com/tinaudio/synth-setter/issues/103)). This is the primary defense against corrupt data.
4. **If validation passes:**
   - Write the Lance attempt payload: write uncommitted fragment data through Lance, upload `metadata/workers/shards/shard-{id}/{worker_id}-{attempt_uuid}.fragment.json`, and upload `metadata/workers/shards/shard-{id}/{worker_id}-{attempt_uuid}.shard-stats.npz`
   - Write worker report (content hash, timing, per-shard results): `metadata/workers/attempts/{worker_id}-{attempt_uuid}/report.json`
   - **Write `.valid` marker** (last step — the commit point): `metadata/workers/shards/shard-{id}/{worker_id}-{attempt_uuid}.valid`
5. **If validation fails:**
   - Upload the corrupt fragment to quarantine: `metadata/workers/shards/shard-{id}/quarantine/{worker_id}-{attempt_uuid}.*`
   - Write `.invalid` marker: `metadata/workers/shards/shard-{id}/{worker_id}-{attempt_uuid}.invalid`
   - Write worker report with error details
   - Log the failure (which check failed, values found)

The `.valid` marker is written **only after** the worker has completed rendering, validation, upload, and bookkeeping. It is the commit point for a staged shard. Worker reports and debug logs are auxiliary metadata — they are not part of the shard admission protocol.

Workers never decide canonical membership. Workers may write uncommitted fragment data under split dataset `data/` directories, but only finalize writes Lance manifests and transactions ([§7.6](#76-finalize-workflow)).

> **Invariant:** Only worker-validated shards reach the staging path, and only committed shards (with `.valid` markers) are visible to `generate`/`status`. Corrupt renders are uploaded directly to quarantine, preserving the evidence for debugging while keeping the staging area clean.

### 7.4 Reconciliation-Based Execution

Instead of tracking worker state or polling provider APIs, the pipeline determines what work remains by inspecting storage.

**`generate` reconciliation:**

1. Read spec from R2 (or create if first run)
2. List staged attempts in `metadata/workers/shards/` — check for the fragment sidecar + stats + `.valid` marker per shard (no data loading, no re-validation — [§7.5](#75-shard-validation))
3. Compute `missing = spec_shards - staged_valid_shards`
4. If nothing missing → "generation complete", exit 0
5. Partition missing shards across N workers
6. Submit N tasks, exit

**`finalize` reconciliation:**

1. Read spec from R2
2. Check for `metadata/dataset.complete` — if present and all canonical outputs exist, exit 0 ("already finalized")
3. List staged attempts — check for the fragment sidecar + stats + `.valid` marker per shard
4. If any shards missing → report which ones, exit 1
5. Select exactly one winning attempt per shard: earliest `.valid` storage `LastModified`, tie-broken by full marker key
6. Structural-check each selected attempt ([§7.5](#75-shard-validation))
7. Commit Lance fragments, reduce stats, register in W&B, upload final metadata, write `metadata/dataset.complete`

**Key properties:**

- **Safe at any time.** Running `generate` when all shards exist is a no-op. Running `finalize` when 3 shards are missing reports the gap and exits.
- **Machine-independent.** Authoritative state lives in R2. If the laptop that launched the run dies, any machine can continue.
- **Phase separation.** Generation and finalization are independent steps. No idle worker waiting for shards. No implicit coordination.

**`make status` — reconciliation report:**

`make status` runs the same reconciliation logic as `generate` but only prints the result. It checks for the fragment sidecar + stats + `.valid` marker existence — no data loading or re-validation. It does not query RunPod, check worker health, or monitor live tasks. The output is fully determined by storage contents — running it from any machine, at any time, produces the same result.

```
$ python -m synth_setter.pipeline status --run-id surge-simple-480k-10k-20260313T100000123Z

Run: surge-simple-480k-10k-20260313T100000123Z
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

**Worker validation (3 checks; 4-check design target)** — run by workers before upload:

Current implementation:

- **Structural**: Opens as a Lance fragment, expected columns present (`audio`, `mel_spec`, `param_array`)
- **Row count**: Matches spec's expected shard size
- **Schema**: the written fragment's physical schema matches the spec-derived schema — see `lance_fragment` in `pipeline/data/lance_shard.py`; Lance's append-mode writer otherwise silently inherits an existing committed dataset's schema ([#2084](https://github.com/tinaudio/synth-setter/issues/2084))

Design target ([#103](https://github.com/tinaudio/synth-setter/issues/103)) adds:

- **Shape**: Array dimensions match spec (sample rate, spectrogram bins, parameter count)
- **Value**: No NaN/Inf values, audio within [-1, 1], parameters within spec bounds

**Existence check** — run by `generate`/`status` during reconciliation:

- Check for the fragment sidecar + stats + `.valid` marker in the staging directory — `.fragment.json` and `.shard-stats.npz`. No data loading.
- The `.valid` marker is authoritative for staging admission: it means a worker completed the full shard lifecycle and committed the result ([§7.2](#72-shard-lifecycle)). It is not sufficient for final dataset correctness — finalize remains the gate before commit.
- The trust chain justifies this: workers do 3-check validation (4-check design target — [#103](https://github.com/tinaudio/synth-setter/issues/103)) before upload, `rclone --checksum` verifies transfer integrity, and R2 PUTs are atomic (the object either exists completely or not). Re-validating hundreds of shards to find a few missing ones is wasted work.

**Structural check** — run by finalize before committing selected attempts:

- Valid Lance sidecar that parses as a strict Pydantic model and contains Lance fragment metadata that round-trips through Lance's `FragmentMetadata.from_json(...)`. Finalize derives the shard from the staging path and the split from the spec, then cross-checks the fragment's row count against the sibling `.shard-stats.npz` `count` — the sidecar carries no shard/split/row fields to trust.
- Valid Lance shard stats file with `count`, `mean`, and `m2` arrays of the expected shapes and dtypes. Finalize reduces only the selected winners' `.shard-stats.npz` files into dataset-level `stats.npz`.
- This catches the only realistic failure between worker validation and finalize: transfer corruption or bit rot. Value-level corruption (NaN, wrong bounds) and row count mismatches were already caught by workers — re-checking would require loading all data from every shard, which is redundant.
- If a staged shard fails the structural check, finalize writes `.invalid` for that attempt, reports the failure, and exits 1. The shard is treated as missing and regenerated on next `generate`.

| Stage               | Validation                                                                                                                                                           | Cost                                   | Why                                                                           |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------- |
| **Worker**          | 3-check: valid Lance fragment, expected columns, row count. 4-check design target adds shape/value/NaN ([#103](https://github.com/tinaudio/synth-setter/issues/103)) | Moderate (opens file, checks metadata) | Primary defense — catches corrupt fragments, missing columns, wrong row count |
| **Generate/status** | Existence (fragment sidecar + stats + `.valid` marker)                                                                                                               | Cheap (file listing)                   | Workers already validated; re-validation is redundant                         |
| **Finalize**        | Structural (valid Lance sidecars/fragments/stats)                                                                                                                    | Moderate (metadata checks)             | Catches transfer corruption; last checkpoint before sealing dataset           |

**Content hashes** (SHA-256 over the fragment data) are recorded in worker reports for provenance and divergence detection. They are not used as acceptance criteria. If two workers produce different hashes for the same shard, the content hashes surface the divergence for investigation.

**Semantic corruption:** Validation catches structural and numerical issues but cannot detect all semantic corruption (e.g., audio that is valid float32 in [-1, 1] but sounds wrong due to a renderer bug). Training is the ultimate semantic check ([§4](#4-system-overview), "Reconciliation Correctness"). At 1-2 runs/week, manual spot-checking of a few samples is practical and encouraged.

**Quarantine:** Workers that fail local validation upload the corrupt fragment directly to `metadata/workers/shards/shard-{id}/quarantine/{worker}-{attempt}.*` with an `.invalid` marker, preserving the evidence for debugging. `generate`/`status` sees the shard as missing (no `.valid` marker) and assigns it on the next run.

### 7.6 Finalize Workflow

> **Implementation status — `synth-setter-finalize-dataset` implements this workflow (#1776): completeness check over staged attempts, winner selection, structural checks, per-split fragment commit, stats reduction, the `dataset.json` audit record, W&B registration, and the completion marker. Still open: `.promoted` markers and the full §14.2 `DatasetCard` schema (`dataset.json` carries the selected attempts only).**

01. **Check for `metadata/dataset.complete`** — if present and all canonical outputs exist, print "already finalized" and exit 0
02. **Read spec** from R2
03. **Check completeness** — list staged attempts, check for the fragment sidecar, stats, and `.valid` marker per shard. If any missing, report which ones and exit 1
04. **Select winning attempts** — for each shard, if multiple staged attempts exist, select the one whose `.valid` marker has the earliest storage `LastModified`, tie-broken by full marker key. `LastModified` is assigned server-side by R2 — a single authority, not a worker wall-clock — so it is safe to trust across workers, and it makes the winner *stable*: any attempt that lands later has a strictly greater timestamp and can never displace an already-selected winner. This monotonicity is what lets finalize re-run safely ([§11.2](#112-failure-modes--edge-cases))
05. **Structural-check selected attempts** — Finalize parses the Pydantic `fragment.json`, deserializes Lance's fragment metadata with Lance, cross-checks the fragment's row count against the sibling `shard-stats.npz` `count`, and confirms the fragment's data path falls under the split the spec assigns to this shard.
06. **Produce training outputs** — commit the selected fragments into `train.lance/`, `val.lance/`, and `test.lance/` with Lance's dataset commit APIs. The winners' uncommitted fragment data already sits under each split's `data/` directory (workers write it there — a fragment is only readable from the dataset whose `data/` dir physically holds its file, so finalize never copies or streams rows). Each split is one `LanceOperation.Overwrite` transaction over the full winner set: it lands as a single atomic manifest version (all winners visible together or none), and because it *replaces* rather than appends, a re-run rebuilds the identical manifest instead of double-committing rows. Idempotence comes from Overwrite-replace, not from `read_version` — Lance auto-rebases a stale-`read_version` append rather than rejecting it, so `read_version` is not a double-commit guard here. The whole model — fragment sidecar round-trip, atomic winner commit, duplicate-attempt dedup, idempotent re-commit, and the co-location requirement — is pinned by [`test_lance_fragment_finalize_poc.py`](../../tests/pipeline/data/test_lance_fragment_finalize_poc.py).
07. **Reduce normalization statistics** — reduce the selected training attempts' `.shard-stats.npz` files into dataset-level `stats.npz`.
08. **Write `dataset.json`** — self-describing dataset card (includes output format, selected attempts, and shard manifest)
09. **Register dataset in W&B** — log as artifact with spec, card, and metrics (§8)
10. **Upload finalized dataset** to R2
11. **Write `metadata/dataset.complete`** — completion marker (last step)

The finalize step runs with `MODE=finalize-shards`. Scoped and validated on the `experiment` branch ([#408](https://github.com/tinaudio/synth-setter/issues/408)).

**`metadata/dataset.complete` semantics:**

`metadata/dataset.complete` means **finalization is done**. It is not a mutex, not an in-progress marker, and does not provide mutual exclusion.

- Written as the very last step, after all outputs are uploaded and verified
- Contains: `run_id` and finalization timestamp
- If `metadata/dataset.complete` exists and all outputs validate → dataset is ready for training
- If `metadata/dataset.complete` exists but outputs are missing → stale marker from a crashed finalize, cleaned up on next run
- Two concurrent finalize processes both write `metadata/dataset.complete` — this is fine, they produce identical outputs ([§7.7](#77-concurrency-semantics))

**Why `metadata/dataset.complete` and not `dataset.lock`:** The file is a completion marker, not a lock. Calling it `.lock` implies mutex semantics the design deliberately does not implement — the marker records completion, it does not arbitrate it. The name should communicate what it means: finalization is complete.

**Finalize idempotency:** Finalize reruns from scratch unless `metadata/dataset.complete` plus all finalized outputs are present and valid. No partial checkpoints — if finalize crashes after `train.lance/` but before `stats.npz`, the next run starts over. This is simple and correct: finalize processes data already in R2, so reruns are cheap (minutes).

**Canonical data immutability:** After finalize writes `metadata/dataset.complete`, the contents of `data/shards/` and the finalized outputs are considered immutable. The pipeline does not enforce this at the storage level (R2 has no object locking), but no pipeline command modifies canonical data after finalization. Manual modification of `data/shards/` after finalize invalidates the dataset hash and provenance chain.

**Quarantine cleanup:** Quarantined shards accumulate across retries. After `metadata/dataset.complete` is written, finalize can optionally delete `quarantine/` contents for completed shards (`--keep-quarantine-days` controls retention). Default: keep all quarantined files.

**`dataset.json` — dataset card:**

The output artifact metadata — a self-describing card for the finalized dataset. It answers "what is this dataset and how do I use it?" without requiring access to the metadata directory.

The input spec defines what the run should produce. `dataset.json` is the *output* record (what was actually produced). The spec has hundreds of shard-level entries. `dataset.json` inlines only what someone needs to load and use the dataset, and references the spec by SHA-256.

**Inlined:** provenance (code version, git dirty, param spec, renderer version, output format), structure (splits, total samples), stats (normalization values), validation summary.
**Referenced:** full spec via `input_spec_sha256` and `input_spec_path`.
**Excluded:** worker reports and debug logs — these are process artifacts, not dataset metadata.

### 7.7 Concurrency Semantics

This is a single-user research pipeline running 1-2x/week. Workers may run concurrently, and finalization may be retried or invoked concurrently **after generation is quiescent**. The standard workflows enforce that barrier (`finalize` needs the completed `generate` job). Running generation against the same run prefix while finalization is publishing canonical outputs is unsupported because R2 does not provide the compare-and-set needed to freeze a winner snapshot.

**Why concurrent operations can't corrupt data:**

1. **Workers write to per-attempt filenames.** Each attempt uploads metadata to `{worker_id}-{attempt_uuid}.*` — unique per invocation. Two workers computing the same shard write to different sidecars in the same staging directory. No metadata overwrites, no races.
2. **Finalize is the only writer of canonical membership.** Lance manifests and transactions are written only by finalize. Zombie workers uploading after finalize has run cannot affect the committed data.
3. **Deterministic outputs within the same execution environment.** Two workers computing the same shard (same seed, same config, same Docker image, same CPU architecture) produce identical content. Non-determinism across hardware is detectable via content hashes.

**Scenario: `generate` run on the same run_id multiple times in quick succession**

Both invocations read the staging prefix, both see the same missing shards, both launch workers for the same shards. Two workers both generate shard-042:

1. Worker A uploads `metadata/workers/shards/shard-000042/pod-abc-uuid1.*`
2. Worker B uploads `metadata/workers/shards/shard-000042/pod-def-uuid2.*`
3. Both attempts coexist — different filenames, no overwrite
4. **Result:** two valid staged attempts, wasted compute. Finalize picks the earliest completed `.valid` marker, tie-broken by marker key.

**Skip-if-valid optimization:** Workers check the staging directory for an existing valid shard before uploading. If one exists, the worker skips the upload and moves to the next shard. This is an optimization, not a correctness requirement — the staging model is safe even without it.

**Scenario: concurrent `finalize` on the same quiescent run_id**

Two finalize invocations both read the input spec, validate the same stable attempt set, select winners, produce final outputs, and write `metadata/dataset.complete`. Both use the same deterministic winner rule and produce identical canonical outputs. `metadata/dataset.complete` does not provide mutex semantics — by design it records completion rather than arbitrating it. The marker's purpose is to let subsequent invocations skip finalization ("already finalized"), not to prevent concurrent finalization. **Result:** identical outputs, wasted compute.

Three properties make this idempotence hold for a *sequential* re-run after a crash, not just concurrent invocations. Winner selection is monotonic (earliest `.valid` `LastModified`), each split is a replace-semantics commit over the full winner set (never an append), and `metadata/dataset.complete` is written last. A finalize that dies after committing `train.lance` but before the marker re-runs to the identical result: the winners are unchanged and the commit rebuilds the manifest rather than appending rows. A straggler attempt that lands after a completed finalize cannot change the outcome either — its later `LastModified` loses to the existing winner.

**Scenario: accidentally-launched finalize while another finalize is running**

Same as above. Both produce identical outputs. The second finalize either:

- Sees `metadata/dataset.complete` from the first (if it finished) → exits with "already finalized"
- Doesn't see it yet → runs to completion independently, produces identical outputs

No data corruption either way.

**Scenario: `generate` while `finalize` is running**

Unsupported for the same run prefix. A newly completed attempt can change a shard's deterministic winner between concurrent finalizers, so split manifests, statistics, and the audit card would not share one immutable selection. The workflow dependency barrier prevents this overlap; operators repairing a run manually must also wait for all workers to stop before invoking finalize.

**Scenario: `finalize` while workers are still uploading**

Finalize checks completeness by validating the staging prefix first. If shards are missing, it reports "generation incomplete" and exits 1. No partial dataset is produced.

**Scenario: zombie worker uploads after finalize completes**

A worker from a previous `generate` invocation hangs for hours, then finally uploads its shard to the staging prefix. This upload lands in `metadata/workers/shards/`, not in `data/shards/`. The canonical finalized data is unaffected. The zombie's staged shard is visible but harmless — it's simply an additional attempt record. See [§11.2](#112-failure-modes--edge-cases) for detailed analysis.

**What this system does NOT protect against:**

- **Non-deterministic outputs across hardware.** If floating-point non-determinism across different CPU architectures produces different audio for the same seed, multiple staged attempts for the same shard ID may differ. Finalize picks the earliest completed valid attempt, tie-broken by marker key — the selection is deterministic for a fixed R2 state, but content may vary across heterogeneous environments. Content hashes and `cpu_arch` in worker reports detect this divergence. The mitigation is to fix non-determinism in the renderer or constrain the execution environment.
- **Concurrent spec modification.** The spec is written once and never modified. If something modifies it after creation, correctness guarantees do not hold.

> **Scope of concurrency safety:** The safety arguments in this section assume deterministic rendering within the execution environment (same Docker image + CPU architecture). The pipeline does not enforce homogeneous worker hardware — it detects but does not prevent architectural divergence. Workers record `cpu_arch` and `os_info` in their reports; when content hashes diverge across attempts, these fields identify the source. `dataset.json` records all unique worker architectures encountered (`worker_architectures`); if multiple architectures appear, finalize logs a warning. For bit-reproducible runs, pin RunPod instance types to a consistent CPU architecture. The mitigation for divergence is to fix the renderer or constrain the environment, not to add locking.

### 7.8 Error Handling & Crash Resilience

The pipeline handles three categories of failure, including crashes from data generation code we don't own (the Surge XT VST plugin can SIGSEGV, OOM, or produce corrupt output).

**Layer 1 — Per-shard process isolation:**
Each shard is rendered in a separate OS process using a spawn context (`multiprocessing.get_context("spawn").Process(...)`). This provides crash isolation at the OS level: a SIGSEGV or OOM kill in the VST plugin terminates only that child process — the parent worker catches the non-zero exit code, marks the shard as invalid, quarantines it, and moves to the next shard. The worker report accumulates per-shard results including exit codes for crashed shards.

Per-shard process isolation is necessary because try/except only catches Python exceptions — it cannot intercept SIGSEGV, OOM kills, or other OS-level crashes from the VST plugin (native C++ code). Without process boundaries, a single shard crash would kill the entire worker and all its in-progress shards.

See [§7.8.1](#781-per-shard-process-isolation) for the design decision, trade-offs, and alternatives considered.

**Layer 2 — Entrypoint crash trap:**
A bash EXIT trap uploads the debug log and a fallback error JSON to R2 if the Python worker process itself dies entirely (import error, uncaught exception that escapes Layer 1). The debug log captures everything up to the crash. Even if no worker report is written, the log survives. Note: Layer 1 process isolation handles most crash scenarios (SIGSEGV, OOM per shard). Layer 2 catches failures of the worker process itself.

**Limitation:** EXIT traps do not fire on SIGKILL (`kill -9`), which is how the Linux OOM-killer terminates processes. If the OOM-killer targets the worker process itself (the parent that spawns per-shard children), no logs are uploaded — the bash EXIT trap never runs. Per-shard child crashes (Layer 1) are unaffected since the parent catches their exit codes normally. Mitigation: (1) the entrypoint uses SIGTERM with a grace period before escalating to SIGKILL for timeouts, (2) Layer 3 reconciliation detects the missing shards regardless, and (3) RunPod pod logs provide a fallback audit trail for OOM events.

**Layer 3 — Reconciliation fills gaps:**
Regardless of how a shard was lost (crash, timeout, upload failure, corrupt output), reconciliation detects it as missing. The next `generate` invocation launches workers for exactly the missing shards. No manual investigation of failure modes is required to resume.

**Error tracking artifacts:**
Each worker invocation produces three artifacts, all with unique filenames keyed by `{worker_id}-{attempt_uuid}`:

- **Staged attempt payload + lifecycle markers** (`workers/shards/shard-{id}/{worker_id}-{attempt}.*` / `.rendering` / `.valid`) — the Lance attempt payload (`fragment.json` + `shard-stats.npz`) and empty markers tracking attempt state. Orphaned `.rendering` without `.valid` = crashed attempt.
- **Worker report** (`workers/attempts/{worker_id}-{attempt}/report.json`, JSON) — derived summary with content hashes, written at end of execution, missing if worker crashed
- **Debug log** (`workers/attempts/{worker_id}-{attempt}/debug.log`, JSONL) — append-only narrative, uploaded by EXIT trap, survives crashes

All worker artifacts live under `metadata/workers/`. Unique filenames per attempt mean retries never overwrite previous artifacts. Missing worker metadata never blocks completion. A run is successful when all staged shard files exist and validate.

#### 7.8.1 Per-Shard Process Isolation

**Problem:** The VST plugin (Surge XT) is native C++ code loaded into the Python process. It can SIGSEGV, OOM, or corrupt global state — failures that Python's try/except cannot catch. Without process boundaries, a single shard crash kills the entire worker and all its in-progress shards.

**Decision:** Each shard renders in a separate OS process via a spawn context (`ctx = multiprocessing.get_context("spawn"); ctx.Process(...)`).

```python
import multiprocessing

_spawn_ctx = multiprocessing.get_context("spawn")

def _render_shard(shard_spec, shard_path):
    """Runs in a child process — SIGSEGV here won't kill the parent.

    Under spawn, the child is a fresh Python interpreter with no inherited state.
    The import happens here, in the child, so the VST plugin loads cleanly.
    """
    from synth_setter.data.vst.writers import make_lance_dataset
    make_lance_dataset(shard_path, shard_spec)

# In the parent worker:
p = _spawn_ctx.Process(
    target=_render_shard,
    args=(shard_spec, local_path),
)
p.start()
p.join(timeout=SHARD_TIMEOUT)

if p.exitcode == 0:
    # validate, upload, write .valid
    ...
elif p.exitcode is None:
    # timed out
    p.kill()
    p.join()
else:
    # crashed: SIGSEGV = -11, OOM kill = -9
    # write .invalid, quarantine
    ...
```

**Why `spawn` and not `fork`:** The `fork` start method copies the parent's memory via copy-on-write. If the parent has loaded a VST plugin, the child inherits that plugin's global mutable state (internal buffers, audio engine state). VST plugins are not designed for this — shared mutable state across forked processes leads to undefined behavior. `spawn` starts a fresh Python interpreter with no inherited state: each child loads the plugin from scratch, with clean globals and its own memory space. This eliminates shared-state corruption between concurrent shard renders.

**Why not direct function call (`make_lance_dataset()` in-process):** A direct call is simpler and avoids per-shard process overhead, but a SIGSEGV in the plugin kills the entire worker process. Per-shard try/except cannot catch OS-level signals. The bash EXIT trap (Layer 2) would upload logs, and reconciliation (Layer 3) would detect the missing shards — but all in-progress shards on that worker are lost, not just the crashing one. For a 48-shard worker, losing all shards to one bad shard is unacceptable.

**Why not `subprocess.run` calling `generate_vst_dataset.py`:** Same crash isolation as `multiprocessing.Process`, but makes tests couple to CLI argument construction. With `multiprocessing.Process`, the child imports `make_lance_dataset` directly and receives only simple data (`shard_spec`, `shard_path`) as args. For tests, `LocalBackend` runs in-process (no spawn) so test fixtures can inject a fake generate function directly.

**Seeding design:** See
[Deterministic Dataset Seeding](deterministic-seeding.md) for the canonical
seed flow, guarantees, current status, and tracked follow-up work.

**Trade-off summary:**

| Approach                    | Crash isolation             | Seed passing   | Per-shard timeout   | Plugin state                       | Testability                        |
| --------------------------- | --------------------------- | -------------- | ------------------- | ---------------------------------- | ---------------------------------- |
| Direct function call        | None (SIGSEGV kills worker) | Python arg     | Manual timer        | Shared (unsafe if mutable)         | Inject fake `generate_fn`          |
| `multiprocessing` fork      | OS process boundary         | Python arg     | `join(timeout)`     | Inherited via COW (unsafe for VST) | Inject fake `generate_fn`          |
| **`multiprocessing` spawn** | **OS process boundary**     | **Python arg** | **`join(timeout)`** | **Fresh load (clean)**             | **LocalBackend in-process inject** |
| `subprocess.run`            | OS process boundary         | CLI args       | `timeout=` kwarg    | Fresh load (clean)                 | Must mock subprocess               |

**Cost:** Per-shard process startup (~0.5-1s) plus fresh plugin load (~1-3s for Surge XT). Shard renders take minutes, so this is negligible — roughly 1-3% overhead.

**Edge case — filesystem contention:** `spawn` eliminates shared memory state, but concurrent child processes could contend on shared filesystem resources if the VST plugin writes to global paths (lock files, temp directories, `~/.local` config). The `Xvfb` wrapper (`run-linux-vst-headless.sh`) should use per-process display numbers (`:N` where N is derived from the child's PID or shard ID) to avoid X11 socket contention. If Surge XT writes to other shared paths during headless rendering, those paths should be isolated per-child via environment variables or temp directories.

### 7.9 Compute Abstraction

The CLI is not a job supervisor. It submits work and exits. Completeness is determined solely by validated outputs in storage, never by polling a provider API.

The compute interface has one method: submit work.

```python
class ComputeBackend(Protocol):
    def submit(self, image: str, task_specs: list[TaskSpec]) -> list[SubmittedTask]: ...
```

**Two implementations:**

- **RunPodBackend**: Production. Wraps the `runpod` Python SDK. Maps tasks to workers (RunPod calls these "pods").
- **LocalBackend**: Development and testing. Runs the worker loop in-process (no Docker, no spawn) with a local filesystem as the "R2" equivalent — same directory structure, same spec format, same validation logic. Accepts an optional `generate_fn` callable for test injection. Docker container fidelity is validated separately via `test_local_docker.sh`.

No `check_tasks` method exists. Provider APIs answer the wrong question ("is the worker running?") when the right question is "are the shards done?" Storage answers that definitively.

**Local mode fidelity:** Local mode mimics R2 exactly — same directory structure, same spec format, same shard naming, same validation function. Only the storage transport changes.

**RunPod instance tagging:** The CLI tags all RunPod instances with the `run_id` at launch. A `pipeline.cli cleanup --run-id <id>` command queries the RunPod API for any pods matching that `run_id` and terminates them — a safety net for orphaned pods if the CLI crashes after launching workers but before logging pod IDs locally.

### 7.10 Output Format: Lance

The pipeline's output format is **Lance**. The renderer CLI dispatches on the shard's filename suffix (`.lance` → `make_lance_dataset`) via `OutputFormat.from_extension`.

Lance **dataset directories** (`train.lance/`, `val.lance/`, `test.lance/`) are committed from worker-produced fragments. Workers use Lance to write uncommitted fragment data and persist a strict Pydantic `fragment.json` sidecar whose `fragment_json` field is the exact Lance `FragmentMetadata.to_json()` payload. Lance owns Lance fragment IDs and physical data references; the pipeline derives logical identity (`shard_id`, `split`, `worker_id`, `attempt_uuid`) from the staging path, filename, and spec rather than storing it in the sidecar ([§14.4](#144-lance-fragment-sidecar-schema)). Rows are Arrow fixed-shape-tensor columns (float16 `audio`, float32 `mel_spec` / `param_array`) with the `ShardMetadata` JSON embedded in the schema metadata and the on-disk format pinned to `data_storage_version="2.2"`.

**Why Lance:** the columnar layout gives per-column projection (train on `mel_spec` + `param_array` without decoding `audio`), the dataset streams natively from object storage for both random-access and sequential loaders, and fragment-based finalize commits winning fragment metadata instead of rewriting rows — so finalize decodes zero audio rows and never becomes a single-machine bottleneck ([§12](#12-open-questions-risks--limitations)). One format serves both the local single-GPU random-access case and the multi-GPU streaming case.

**Lance shard attempts are not resumable:** a crashed worker re-renders the whole shard attempt. The staging/finalize split is unaffected.

## 8. Experiment Tracking (Weights & Biases)

> Authoritative W&B conventions (artifact naming, metadata placement, lineage DAG, `job_type` values) are defined in [storage-provenance-spec.md §4–§7](storage-provenance-spec.md#4-wb-artifact-types). Repeated here for data-generation context.

W&B serves as a lightweight observability layer for the pipeline — a few key metrics and the dataset as a first-class artifact. It is not a monitoring dashboard or a log aggregator. W&B is an index and lineage tracker, not the authoritative dataset store. R2 holds the data; `dataset.json` holds the metadata; W&B points to both.

The finalize stage initializes W&B with `wandb.init(project="synth-setter", job_type="data-generation")`.

### Metadata Placement

| Where             | What goes there                                                        | Why                                                  |
| ----------------- | ---------------------------------------------------------------------- | ---------------------------------------------------- |
| `wandb.summary`   | Pipeline metrics (see table below)                                     | Final values, not time-series — summary is correct   |
| Artifact metadata | `dataset_config_id`, `dataset_wandb_run_id`, `shard_count`, provenance | Travels with the artifact through the lineage DAG    |
| `dataset.json`    | Full dataset card (structure, stats, validation)                       | Self-describing record in R2, referenced by artifact |

**Pipeline metrics** (written to `wandb.summary` by `finalize`):

| Metric                             | Type  | Description                                                    |
| ---------------------------------- | ----- | -------------------------------------------------------------- |
| `pipeline/shards_total`            | int   | Total shards in spec                                           |
| `pipeline/shards_valid`            | int   | Shards that passed validation                                  |
| `pipeline/shards_quarantined`      | int   | Shards copied to quarantine                                    |
| `pipeline/total_samples`           | int   | Total samples across all shards                                |
| `pipeline/generation_time_seconds` | float | Wall clock: spec created_at → last shard uploaded              |
| `pipeline/finalize_time_seconds`   | float | Wall clock: finalize start → metadata/dataset.complete written |
| `pipeline/errors_total`            | int   | Total errors across all worker reports                         |

**Dataset artifact** (logged by `finalize`):

The finalized dataset is registered as a W&B Artifact named `data-{dataset_config_id}` of type `"dataset"`:

- **Files included:** `input_spec.json`, `dataset.json` (the card)
- **Metadata:** `dataset_config_id`, `dataset_wandb_run_id`, `shard_count`, `param_spec`, `code_version`, `is_repo_dirty`, `total_samples`, split sizes
- **References:** R2 path to the actual Lance data (not uploaded to W&B — too large)
- **Versioning:** W&B auto-versions artifacts: `data-{dataset_config_id}:v0`, `:v1`, etc. `:latest` always points to the most recent finalize.

This creates a dataset entry in the W&B artifact registry that can be referenced by training runs, establishing **artifact lineage**: code version → dataset artifact → training run → model checkpoint. Training runs close the lineage loop by declaring the dataset as an input: `artifact = run.use_artifact(f"data-{dataset_config_id}:latest")`. See [Appendix E.3](#e3-wb-integration) for the full implementation.

After finalize, datasets enter the training → evaluation → promotion pipeline. See [promotion-pipeline-reference.md](../reference/promotion-pipeline-reference.md).

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

**Rejected.** Co-launch generation workers and a finalize worker simultaneously. The finalize worker polls R2, waiting for all shards before merging. Worker status files are the authoritative record.

- **Status files as authority is fragile.** A worker can report success but fail to upload. The finalize worker trusts the status and either merges incomplete data or hangs forever.
- **The finalize worker as a waiter wastes money.** A worker sitting idle for 30-60 minutes polling R2 costs compute time for no work.
- **Infrastructure-derived shard names break resumability.** Naming shards `shard-{pod_id}-{attempt_id}-{seq}` means retries produce different filenames.
- **No reconciliation means no partial retry.** If 3 of 10 workers fail, the only option is to rerun everything.

### 9.3 ComputeBackend with Task Lifecycle Management

**Rejected.** Give `ComputeBackend` a `submit()` plus `check_tasks()` pair, where `check_tasks()` polls provider APIs for task status.

- **Provider task state is unreliable.** RunPod can report "running" for a worker that OOM-killed 10 minutes ago.
- **It duplicates reconciliation.** Storage already answers "is this shard done?" ([§7.1](#71-storage-as-the-source-of-truth)). Adding a second, weaker signal creates two sources of truth.
- **It couples the protocol to provider lifecycles.** Pods are persistent and inspectable; serverless functions are ephemeral. A `check_tasks` method pretends providers are more similar than they are.
- **Scope creep risk.** Once you can check status, the next step is retry logic, timeout heuristics, and notifications — a scheduler, not a data pipeline.

### 9.4 Hadoop / MapReduce

**Rejected.** Hadoop is designed for processing existing large datasets with shuffle, sort, and reduce phases over HDFS. This pipeline's workload is fully parallel data *generation* — no inter-worker communication, no data dependencies, no reduce step. Hadoop's infrastructure (HDFS cluster, YARN resource manager, JVM-based framework) would be unused overhead — the pipeline would only use it as a pod launcher.

### 9.5 `make status` as Provider-Status Command

**Rejected.** Have `make status` show live pod status from RunPod's API.

- **Provider APIs answer the wrong question.** "Is the worker running?" ≠ "are the shards done?" Storage answers the right question ([§7.1](#71-storage-as-the-source-of-truth)).
- **Not portable.** Polling RunPod workers is RunPod-specific.
- **Scope creep risk.** Leads to status enums, timeout heuristics, and provider-specific error parsing.

### 9.6 CLI Flags as Run Configuration

**Rejected.** Run configuration via Make variables: `make generate PARAM_SPEC=... SHARD_SIZE=... NUM_SHARDS=...`.

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
- **Atomic multi-object writes.** Writing a shard's fragment sidecar and `worker-{id}.json` are two separate PUTs. They can't be made atomic. See [§11.2](#112-failure-modes--edge-cases).
- **Read-your-writes across regions.** R2 is globally distributed; the pipeline assumes single-region usage (all operations from one R2 endpoint).

### 11.2 Failure Modes & Edge Cases

Non-obvious failure modes, edge cases, and blind spots. Each includes the scenario, consequence, and mitigation.

**Corrupt shard in staging:**
A worker's VST plugin crashes mid-render, producing a corrupt file. The worker uploads it to the staging prefix with a unique per-attempt filename. **Consequence:** A corrupt fragment exists alongside valid ones in the shard's staging directory. **Mitigation:** Workers validate shards locally *before* uploading ([§7.5](#75-shard-validation)). If local validation fails, the upload is skipped and the failure is logged. Only validated shards reach staging. Even if a corrupt shard slips through, reconciliation re-validates staged files and quarantines failures. The staging model means a corrupt upload never overwrites a valid one — each attempt has a unique filename.

**Non-atomic cross-file writes:**
A worker uploads a shard's fragment sidecar successfully but crashes before writing `worker-{id}.json`. Or vice versa. These are separate R2 PUTs — they cannot be made atomic. **Consequence:** Worker report may be out of sync with actual shard state. **Mitigation:** `generate`/`status` checks file existence and `.valid` markers, not worker reports ([§7.1](#71-storage-as-the-source-of-truth)). Per-attempt UUIDs make mismatches observable.

**Partial shard upload:**
`rclone` crashes mid-upload. R2 may have a partial or corrupt object at the staging path (though this is rare — R2 PUTs are atomic for single objects, and multipart uploads don't appear until finalized). **Consequence:** A corrupt fragment may exist in staging without a `.valid` marker (worker crashed before writing it). Or with a `.valid` marker if the crash happened during a subsequent upload. **Mitigation:** If no `.valid` marker, `generate` treats the shard as missing. If `.valid` exists, finalize's structural check catches the corruption before commit. The partial upload does not affect any other attempt's file (unique filenames).

**Silent data corruption (bit rot / transfer corruption):**
Local disk corruption between render and upload, or network corruption during transfer, produces a shard in R2 that differs from what the worker intended. **Consequence:** Corrupt shard passes filename checks but contains wrong data. **Mitigation:** All rclone operations use `--checksum`, which verifies content hashes after transfer. If a checksum mismatch is detected, the worker must delete the local shard, re-generate, and re-upload. Storage-layer bit rot within R2 is handled by R2 internally (server-side object checksums).

**Slow worker overtaken by retry:**
Worker A is assigned shard-042. Worker A is slow. User runs `generate` again, reconciliation sees shard-042 as missing (no valid staged attempt), assigns it to Worker B. Worker B completes first and writes its `.valid` marker. Worker A completes later and writes a second `.valid` marker. **Consequence:** Two valid staged attempts exist for shard-042. Finalize picks the earliest completed `.valid` marker, tie-broken by marker key. If both workers ran on the same hardware, the attempts are identical. If on different hardware, the attempts may differ at the floating-point level — finalize still picks one deterministic winner for the observed R2 state. **Mitigation:** Content hashes in worker reports detect divergence. Hard timeout on workers prevents long-running zombies.

**Zombie worker uploads after finalize:**
Worker A hangs for 12 hours, then completes and uploads shard-042 attempt metadata. Meanwhile, Worker B already uploaded shard-042, finalize selected it, and the dataset is in use for training. **Consequence:** Worker A's upload lands in `metadata/workers/shards/shard-000042/pod-A-uuid1.*` — a new attempt in the staging directory. The committed output is unaffected. The finalized dataset hash is stable. **Mitigation:** Finalize-owned membership ensures zombie uploads cannot corrupt finalized data. Hard timeout on workers and RunPod auto-stop prevent zombies in the first place. Re-running finalize would re-validate staging and pick the same winner unless the original completion marker is missing or stale.

**Spec deleted after generation starts:**
Workers receive shard assignments via environment variables at launch. If the spec is deleted from R2 after launch, workers continue fine but subsequent `status`/`generate` fail. **Consequence:** Orphaned run. **Mitigation:** Spec is immutable, should never be deleted.

**`metadata/dataset.complete` exists but outputs are corrupt:**
Finalize wrote `metadata/dataset.complete` but outputs were later corrupted. **Mitigation:** Finalize validates outputs when `metadata/dataset.complete` exists. If missing/corrupt, deletes stale marker and reruns.

**Stale `metadata/dataset.complete` after re-generation:**
User runs `generate` after finalization (e.g., to replace a quarantined shard). `metadata/dataset.complete` from old finalize still exists. **Mitigation:** Finalize re-validates outputs against current spec. Mismatches trigger stale marker deletion and rerun.

## 12. Open Questions, Risks & Limitations

### Known Limitations

1. **No incremental finalization.** Crashes during finalize restart from scratch. Acceptable because finalize processes existing data and is fast to retry.

2. **Reproducibility is controlled-conditions, not absolute.** The pipeline guarantees that the same spec + same Docker image + same hardware = identical dataset. But VST plugin floating-point behavior can vary across CPU architectures (x86 vs ARM, SSE vs AVX), and Docker base image updates change system libraries. Content hashes in worker reports detect when this happens, but the pipeline does not enforce cross-hardware bit-identity. If multiple workers produce different output for the same shard on different hardware, finalize picks the earliest completed valid attempt — the selection is deterministic for a fixed R2 state but the content may vary across heterogeneous environments.

3. **R2 listing at scale.** Reconciliation lists staged shard objects (1000/page). At 480 shards: 1 API call. At 48,000: 48 calls — still fast. A future optimization could write a `metadata/shards.manifest` file listing all promoted shard IDs after finalize, allowing subsequent operations to read a single file instead of listing the prefix. Not needed at current scale.

4. **No partial dataset usage.** Training can't start until finalize completes. Acceptable for batch workflows.

5. **Spec immutability.** Can't add shards to an existing run — must create a new run. By design (prevents drift), but means config mistakes cost a new run_id.

6. **No cost controls.** No budget cap or automatic shutdown. `generate --dry-run` prints shard assignments without launching workers; `finalize --dry-run` shows what would be committed without doing it.

7. **No real-time progress.** Only `make status` shows shard counts. No live progress bar or ETA.

8. **Multi-run dataset composition is manual.** Combining datasets from multiple runs: copy shards into a new R2 prefix, run `finalize`, register in W&B via the web UI. No automated tooling.

### Risks

| Risk                              | Likelihood | Impact                                      | Mitigation                                                               |
| --------------------------------- | ---------- | ------------------------------------------- | ------------------------------------------------------------------------ |
| RunPod availability               | Medium     | Workers queue or fail                       | ComputeBackend enables fallback                                          |
| VST plugin crashes (SIGSEGV, OOM) | Medium     | Individual shards fail                      | Per-shard isolation + reconciliation fills gaps                          |
| R2 eventual consistency           | Low        | Stale listing                               | R2 is strongly consistent for PUTs; polling interval handles edge cases  |
| Cost overrun from stuck workers   | Low        | Workers run indefinitely                    | Hard timeout in entrypoint; RunPod auto-stop                             |
| Non-deterministic rendering       | Low        | Concurrent writes produce different content | Fix renderer; specs record renderer version                              |
| Silent data corruption            | Low        | Corrupt shard accepted as valid             | `rclone --checksum` on all transfers; overhead negligible vs render time |

## 13. Out of Scope

The following are of interest for next steps but are out of scope for this document. They should not influence the design of the two-stage pipeline described above.

### Future Processing Stages

Additional stages could follow the same contract (§5) without modifying existing stages:

| Stage              | Input        | Output               | Compute |
| ------------------ | ------------ | -------------------- | ------- |
| **augment-reverb** | raw shards   | augmented shards     | CPU     |
| **add-captions**   | audio shards | shards + text column | GPU     |
| **render-presets** | preset bank  | audio shards         | CPU     |

`add-embeddings` is now implemented as the `synth-setter-add-embeddings` Hydra
endpoint (`synth-setter-add-embeddings lance_uri=DATASET.lance`, config
`configs/add_embeddings.yaml` validated into `AddEmbeddingsConfig`): it augments
a finalized Lance dataset in place with a `clap` (LAION-CLAP)
`FixedSizeList<float32, 512>` vector column — optionally IVF_PQ-indexed for
`nearest=` vector search — and an `m2l` (music2latent) fixed-shape-tensor
latent column, both derived from the audio column. `same_s`/`same_l` SAME
latent columns are also selectable via `embeddings=` (multi-GB encoders, each
loaded and written in its own sequential pass); the selectable set is
`EMBEDDING_REGISTRY`'s keys in `add_embeddings.py`. An optional
`resume_cache=<path>` caches per-batch encoder outputs so an interrupted run can
resume without re-encoding already-processed rows (see `add_embeddings.py`).

`synth-setter-add-preview-columns` (`pipeline/data/add_preview_columns.py`) follows the same contract: it takes Lance audio shards and adds an `audio_mp3` preview column plus an `audio_uuid` UUIDv5 fingerprint column (CPU), without modifying existing stages.

Training hydration that reads only a subset of a finalized dataset's columns/rows can materialize a transaction-uuid-pinned local copy via `materialize_lance_subset` (`pipeline/data/lance_materialize.py`) instead of transferring the whole dataset directory; a sidecar manifest gates cache reuse by request hash.

Stage order would remain static and explicit — user runs commands in sequence. If the number of stages grows to 4-6 and manual commands become unwieldy, adopt Prefect rather than building a homegrown orchestrator.

### Data Format Abstraction

The pipeline uses Lance as its output format ([§7.10](#710-output-format-lance)). `OutputFormat` stays a plain enum rather than a general `ShardWriter` protocol; extract a writer protocol only if a further format makes the dispatch unwieldy, not speculatively. Training-side reads go through the Lance backend (`src/synth_setter/data/lance_datamodule.py`).

### Content-Addressable Outputs

Shards named by input hash (`shard-{sha256(config+seed)}.lance`) would enable cross-run deduplication and integrity verification. Not planned — deterministic logical naming is simpler and sufficient.

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

See `src/synth_setter/pipeline/schemas/spec.py` for the authoritative definition. The model is `DatasetSpec` (unifies the previous `DatasetConfig` + `DatasetPipelineSpec` split; the constructed Pydantic instance **is** the artifact on R2 — `model.model_dump_json()` is the JSON).

```python
class ShardSpec(BaseModel):
    """Per-shard identity and pre-computed derived values."""
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    shard_id: int
    filename: str
    seed: int
    sample_offset: int = 0

class RenderConfig(BaseModel):
    """Renderer-specific configuration nested as ``DatasetSpec.render``."""
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    plugin_path: str  # "torchsynth" selects the in-process backend; must agree with renderer_backend (RenderConfig._validate_torchsynth_backend)
    plugin_state_path: str
    param_spec_name: str
    renderer_version: str
    sample_rate: int
    channels: int
    velocity: int
    signal_duration_seconds: float
    min_loudness: float
    samples_per_render_batch: int = 32
    samples_per_shard: int
    sample_offset: int = 0      # split-local index of this shard's first row
    attempts_per_sample: int = 100
    max_retries: int = 0        # per-shard retry budget for transient renderer failures
    parallel: bool = False      # dispatch shard renders concurrently (ThreadPoolExecutor)
    plugin_reload_cadence: Literal["once", "render"] = "once"  # per-shard load (#1999)
    # Platform-aware default via Field(default_factory=...): "never" on Darwin
    # (show_editor SIGTRAPs after ~3-4 calls, #714), "render" elsewhere
    # (preserves historical per-render warm-up). An explicit
    # gui_toggle_cadence="render" is still rejected on Darwin by a
    # model_validator; "always_on" requires plugin_reload_cadence="once".
    # Source of truth: _GuiToggleCadence / RenderConfig in pipeline/schemas/spec.py.
    gui_toggle_cadence: Literal["never", "once", "render", "always_on"] = Field(
        default_factory=_default_gui_toggle_cadence
    )
    # "shard" reuses one patch for every sample in the shard (a #489 per-patch
    # variance probe; a partial shard re-renders from row 0 rather than resuming).
    # Source of truth: _ParamSampleCadence / RenderConfig in pipeline/schemas/spec.py.
    param_sample_cadence: Literal["sample", "shard"] = "sample"

class DatasetSpec(BaseModel):
    """Unified dataset specification — input config + materialized runtime in one model."""
    # Strict everywhere; JSON round-trip coercions (list→tuple, str→datetime) happen via
    # explicit per-field validators, not by relaxing strict mode at the trust boundary.
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    # Layout
    task_name: str
    output_format: Literal["lance"]
    train_val_test_sizes: tuple[int, int, int]
    train_val_test_seeds: tuple[int, int, int] | None = None
    base_seed: int
    # Sub-models
    render: RenderConfig
    # R2 storage (nested R2Location: ``bucket`` / ``prefix_root`` / ``prefix`` —
    # see ``src/synth_setter/pipeline/schemas/r2_location.py``).
    r2: R2Location = Field(default_factory=_default_r2_location)

    # Runtime fields. All five auto-fill via ``default_factory`` when missing on
    # input; ``run_id`` / ``r2.prefix`` use the data-aware factories (``_default_run_id``,
    # and ``_fill_default_r2_prefix`` invoked from the ``mode='before'`` model
    # validator) that derive from already-validated ``task_name`` + ``created_at``.
    # JSON-loaded values pass through unchanged (workers reuse materialization-time values).
    git_sha: str = Field(default_factory=lambda: _get_git_sha())
    is_repo_dirty: bool = Field(default_factory=lambda: _is_repo_dirty())
    created_at: datetime = Field(default_factory=lambda: _utc_now())
    run_id: str = Field(default_factory=_default_run_id)

    # Computed: @computed_field + @cached_property — emitted by model_dump and
    # stripped on input (see _strip_computed_field_keys) so JSON round-trip works.
    @computed_field
    @cached_property
    def shards(self) -> tuple[ShardSpec, ...]: ...
    # num_shards / num_params follow the same @computed_field / @cached_property pattern.
```

All three models (`DatasetSpec`, `RenderConfig`, `ShardSpec`) use Pydantic strict mode at the trust boundary. JSON-mode coercions (`list→tuple` for `train_val_test_sizes` / `train_val_test_seeds`, `str→datetime` for `created_at`) are handled by explicit per-field validators on `DatasetSpec`; `extra="forbid"` plus those validators keep the boundary tight without relaxing strict. `frozen=True` makes specs immutable at the type level.

**Seed derivation:** `DatasetSpec.train_val_test_seeds` supplies independent
split masters. Each `ShardSpec` pairs its split master with a split-local
`sample_offset`; row-level retries derive RNGs from the master, absolute
split-local row, and attempt. `None` preserves legacy `base_seed + shard_id`
semantics for old specs. See
[Deterministic Dataset Seeding](deterministic-seeding.md) for the canonical
design.

**Why JSON for specs and reports:** Machine-generated, stored in R2, read back by the CLI. JSON is the simplest correct format — Pydantic has native JSON methods (`.model_dump_json()` / `.model_validate_json()`), it's human-readable (`rclone cat` + `jq`), and handles nested structures natively. Config files use YAML because they're human-authored.

### 14.2 Dataset Card Schema

```python
class DatasetCard(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)

    schema_version: int
    run_id: str
    finalized_at: str       # ISO 8601

    # Provenance
    git_sha: str
    is_repo_dirty: bool
    param_spec: str
    renderer_version: str
    output_format: str      # "lance"
    sample_rate: int

    # Structure
    total_samples: int
    splits: list[int]  # sample counts per split (length 3: train, val, test)
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
    content_hash: str | None = None  # SHA-256 of the fragment data (None if failed)
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

### 14.4 Lance Fragment Sidecar Schema

Lance worker attempts store their reconciliation contract in
`metadata/workers/shards/shard-{id}/{worker_id}-{attempt_uuid}.fragment.json`.
This JSON is validated with Pydantic at the R2 trust boundary. The sidecar
carries only what is not recoverable elsewhere: a schema version and Lance's own
serialized fragment metadata, which stays an opaque Lance-owned string that
finalize parses with Lance rather than treating as an untyped nested dictionary.
Lance's `FragmentMetadata.to_json()` returns a dict, so `fragment_json` holds
its `json.dumps(...)` string, which `FragmentMetadata.from_json(...)` re-parses
(verified round-trip: [`test_lance_fragment_finalize_poc.py`](../../tests/pipeline/data/test_lance_fragment_finalize_poc.py)).

```python
class LanceFragmentSidecar(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    schema_version: Literal[1]
    fragment_json: str
```

Logical identity is derived, not stored, so there is no field for it to
contradict: `worker_id` and `attempt_uuid` come from the filename, `shard_id`
from the staging path, `split` from the spec's deterministic split assignment,
and the row count from Lance's fragment metadata (cross-checked against `count`
in the sibling `.shard-stats.npz`). Storing any of these in the sidecar would
create a second source of truth free to drift from the first.

Per-shard Lance statistics are stored beside the sidecar as
`{worker_id}-{attempt_uuid}.shard-stats.npz`, not JSON, to avoid serialization
precision loss. The `.npz` contains Welford state:

| Array   | Type      | Meaning                                   |
| ------- | --------- | ----------------------------------------- |
| `count` | `int64`   | Number of rows represented                |
| `mean`  | `float64` | Running mean for normalization statistics |
| `m2`    | `float64` | Running sum of squared deviations         |

Finalize reduces only the selected winners' `.shard-stats.npz` files into
dataset-level `stats.npz`.

### 14.5 Validation Boundaries

**Validation boundaries — when to use what:**

The pipeline uses different validation tools depending on where data crosses a trust boundary:

| Boundary                      | Tool                | Why                                                                                          |
| ----------------------------- | ------------------- | -------------------------------------------------------------------------------------------- |
| External input (config YAML)  | Pydantic (strict)   | Untrusted human input — catch type errors, missing fields, invalid values at parse time      |
| Serialization (spec, reports) | Pydantic (strict)   | JSON crossing process boundaries (R2 ↔ CLI ↔ workers) — enforce schema on every read         |
| Shard data (array payloads)   | Validation function | NumPy/Arrow arrays inside shards — Pydantic can't validate `ndarray`; custom checks required |

Pydantic is for trust boundaries — where data enters the system from an external source (user config, JSON from R2, worker reports from other processes). Dataclasses are for internal contracts — where data has already been validated and you just need a typed container to prevent programming errors. No runtime validation overhead on 480k samples.

### 14.6 Config Materialization

A run starts from a Hydra experiment YAML composed against `src/synth_setter/configs/dataset.yaml`:

```yaml
# src/synth_setter/configs/experiment/generate_dataset/surge-simple-480k-10k.yaml (filename stem = dataset_config_id)
# @package _global_

defaults:
  - override /datamodule: surge_simple
  - override /render: surge_simple
  - _self_

task_name: surge-simple-480k-10k

train_val_test_sizes: [440000, 20000, 20000]

render:
  sample_rate: 44100
  min_loudness: -55.0
```

`src/synth_setter/configs/dataset.yaml` is the `@hydra.main` entry. Its `defaults` list pulls in `datamodule:` (param spec / channels / velocity / loudness floor), `render:` (renderer + plugin / preset / sample rate / batch sizes), `r2:` (bucket + prefix root), `paths:`, `hydra:`, and the named `experiment:`. Required slots are marked `???` and filled by the chosen experiment.

On first `generate` (`python -m synth_setter.cli.generate_dataset experiment=<id>`):

1. Hydra composes the experiment against `src/synth_setter/configs/dataset.yaml`, yielding an `OmegaConf` `DictConfig`.
2. `spec_from_cfg(cfg)` (a thin wrapper over `DatasetSpec.from_hydra_cfg`) masks the cfg to `DatasetSpec`'s own fields, resolves, and constructs a Pydantic `DatasetSpec` (`strict=True`, `frozen=True`) — the same model used for the on-R2 artifact.
3. Runtime fields (`run_id`, `r2`, `created_at`, `git_sha`, `is_repo_dirty`) auto-fill via `default_factory` when absent. `run_id` is `{task_name}-{YYYYMMDDTHHMMSSsssZ}` (millisecond precision); `r2.prefix` is `data/{task_name}/{run_id}/`. `renderer_version` is set by the configured renderer's pin; the worker re-derives via `extract_renderer_version` and refuses to render on mismatch.
4. Computed fields (`shards`, `num_shards`, `num_params`) derive deterministically from layout + render fields.
5. Upload the JSON-serialized `DatasetSpec` to R2 (`<r2.prefix>/input_spec.json`).
6. Proceed with reconciliation.

**Dirty repo handling (planned):** `is_repo_dirty` is captured in the spec, but the design's auto-upload of `git diff` to `metadata/run_diff.patch` is not yet implemented in `generate_dataset` — captured here as the intended behavior so a dirty repo's exact code state can be reconstructed during rapid ML research iteration.

**Config drift protection (planned, [#386](https://github.com/tinaudio/synth-setter/issues/386)):** The intended behavior is that composing `experiment=<id>` against a `run_id` that already has a spec on R2 errors and treats the existing spec as authoritative. Today, `generate_dataset` always derives a fresh `run_id` from `task_name + created_at` and writes a new spec — so this check is documented intent, not enforced behavior.

### 14.6 Run ID Format

> ID conventions follow [storage-provenance-spec.md §1](storage-provenance-spec.md#1-ids).

| Pipeline concept          | Storage spec concept                               | Example                                                                 |
| ------------------------- | -------------------------------------------------- | ----------------------------------------------------------------------- |
| Config filename (no ext)  | `dataset_config_id`                                | `surge-simple-480k-10k`                                                 |
| Config ID + ISO timestamp | `dataset_wandb_run_id`                             | `surge-simple-480k-10k-20260312T143022500Z`                             |
| R2 root path              | `data/{dataset_config_id}/{dataset_wandb_run_id}/` | `data/surge-simple-480k-10k/surge-simple-480k-10k-20260312T143022500Z/` |

Config filenames live in `src/synth_setter/configs/experiment/generate_dataset/`. Production training configs follow the pattern `{name}-{total_samples}-{shard_size}.yaml` (e.g. `surge-simple-480k-10k.yaml`, where 480k is the train+val+test total); Lance experiment configs instead encode the split sizes as `{name}-lance-{train}-{val}-{test}.yaml` (e.g. `surge-xt-lance-10k-2k-1k.yaml`); CI smoke and partitioner-exercise configs use shorter, role-descriptive names (e.g. `smoke-shard.yaml`, `10-1k-shards.yaml`). The filename without extension is the `dataset_config_id` — choose names that read clearly in R2 paths and W&B run IDs.

### 14.7 CLI & Directory Structure

```
src/
  generate_dataset.py   # Sequential multi-shard dataset-generation entrypoint (Hydra; MVP). Deprecated when generate-shards lands (#411).

  pipeline/             # Distributed-pipeline package
    __init__.py

    schemas/            # Pydantic models (implemented)
      __init__.py
      spec.py           # DatasetSpec (unified config + runtime; built by its own from_hydra_cfg classmethod, called via spec_from_cfg in cli/generate_dataset.py), RenderConfig, ShardSpec; OutputFormat str-enum carrying format↔suffix dispatch (.extension property + .from_extension reverse lookup)
      shard_metadata.py # ShardMetadata — per-shard provenance embedded in the Lance schema metadata (leaf module, no project imports)
      prefix.py         # DatasetConfigId, DatasetRunId, R2Prefix, assert_r2_prefix_matches helpers
      image_config.py   # Docker image configuration

    ci/                 # CI validation scripts (implemented)
      materialize_spec.py # Compose a DatasetSpec from a Hydra experiment and write it to disk as JSON
      validate_spec.py  # Spec structural validation (required fields, git_sha format, etc.)
      validate_shard.py # Shard validation (Lance, full per-dataset shapes via synth_setter.data.vst.shapes — not just row count); iterates spec.shards via R2
      load_image_config.py # Resolve Docker image configuration for the launcher

    constants.py        # Well-known filenames (INPUT_SPEC_FILENAME)
    r2_io.py            # rclone-backed R2 helpers (URI handling, download, upload, size probe)
    skypilot_launch.py  # Click CLI thin passthrough: shells out to an inner generator command, discovers + parses `input_spec.json` via `find_input_specs`, reads its R2 URI off `spec.r2.input_spec_uri()`, then hands off to `dispatch_via_skypilot`, which validates cfg + creds in a Phase 1 pass before invoking `sky.jobs.launch` (Phase 2). CI workflows derive the canonical spec URI deterministically via `synth-setter-spec-uri --from-experiment ... --run-id-override ...` from the same Hydra cfg the launcher composes.

  # --- Planned (not yet implemented) ---
  # cli.py              # Click entry point: generate, status, finalize
  # stages/             # Each stage is a self-contained module
  #   generate.py       # Generate stage logic
  #   finalize.py       # Finalize stage logic
  # backends/           # Compute provider implementations
  #   base.py           # ComputeBackend Protocol definition
  #   runpod.py         # RunPodBackend (production)
  #   local.py          # LocalBackend (development/testing)
  # storage.py          # R2 operations (list, upload, download, quarantine)
  # reconcile.py        # Read spec, validate shards, compute missing set
  # schemas/
  #   report.py         # WorkerReport, ShardResult
  #   card.py           # DatasetCard, ValidationSummary
  # validation.py       # Full shard validation
  # retry.py            # Centralized tenacity retry policy
  # logging_config.py   # structlog configuration
```

Pipeline configs live under `src/synth_setter/configs/` as Hydra groups composed by `src/synth_setter/configs/dataset.yaml` (filename stem of `src/synth_setter/configs/experiment/generate_dataset/<id>.yaml` = `dataset_config_id`):

```
src/synth_setter/configs/
  dataset.yaml         # @hydra.main entry point; defaults list (datamodule/render/r2/paths/hydra/experiment)
  experiment/
    generate_dataset/  # Dataset generation recipes; filename stem = dataset_config_id
      surge-simple-480k-10k.yaml
      smoke-shard.yaml
      ci-materialize-test.yaml
      10-1k-shards.yaml
  datamodule/          # Param spec / channels / velocity / loudness floor (shared with training)
    surge_simple.yaml
    surge.yaml
  render/              # Renderer + plugin / preset / sample rate / batch sizes
    surge_simple.yaml
    surge_xt.yaml
  r2/                  # R2 bucket + prefix root
    default.yaml
  trainer/             # Training configs (Hydra)
    ddp.yaml

  # --- Planned (Hydra-composed dataset layout; lands when PR-3 migrates the
  #     launcher to @hydra.main and removes load_dataset_spec_yaml) ---
  # dataset.yaml         # Top-level @hydra.main composition target
  # experiment/          # Per-experiment defaults files; each composes dataset.yaml + groups
  #   generate_dataset/
  #     surge-simple-480k-10k.yaml
  # render/              # Renderer-specific configs (param_spec_name, renderer_version, samples_per_shard, …)
  #   surge_xt.yaml
  # r2/                  # R2 bucket + prefix_root
  #   default.yaml
```

## Appendix A: Glossary

| Term                          | Definition                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **R2**                        | [Cloudflare R2](https://developers.cloudflare.com/r2/), an S3-compatible object storage service. Key feature: free egress (no cost to download data). Used for shard storage and pipeline coordination. [Consistency model](https://developers.cloudflare.com/r2/reference/consistency/): strong read-after-write.                                                                                                                                                                                                                                                                                                                                                               |
| **RunPod**                    | [RunPod](https://www.runpod.io/), a cloud compute marketplace offering on-demand GPU and CPU instances ("pods"). Used for running data generation workers. Pods are ephemeral — they run a Docker container and terminate.                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **Worker**                    | A cloud compute instance that generates shards. On RunPod, a worker is a "pod" — a single Docker container with assigned shard work. The design uses "worker" to stay infrastructure-agnostic.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **Shard**                     | A batch of training samples (audio, mel spectrograms, parameter arrays) stored as a Lance dataset directory (§7.10). Typically 1k-10k samples per shard. Named by logical index (the `shard-000042.lance/` directory).                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **W&B (Weights & Biases)**    | [Weights & Biases](https://wandb.ai/), an experiment tracking platform. Used here as a lightweight observability layer: pipeline metrics, dataset artifact registry, and lineage tracking from dataset → training run.                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **Input spec**                | JSON file (`input_spec.json`) defining the frozen input specification for a run — shard specs, seeds, shapes, splits, renderer version. Written once on first `generate`, never modified.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **dataset_config_id**         | Stable identifier for a dataset configuration, derived from the config filename (without extension). Production training configs follow `{name}-{total_samples}-{shard_size}` (example: `surge-simple-480k-10k`, where 480k is the train+val+test total); Lance experiment configs instead encode the split sizes as `{name}-lance-{train}-{val}-{test}` (example: `surge-xt-lance-10k-2k-1k`); CI smoke and partitioner-exercise configs use role-descriptive names. The legacy flat YAML's `shard_size` becomes `render.samples_per_shard` on the resulting `DatasetSpec` via `load_dataset_spec_yaml`. See [storage-provenance-spec.md §1](storage-provenance-spec.md#1-ids). |
| **dataset_wandb_run_id**      | Unique identifier for a pipeline execution. Format: `{dataset_config_id}-{YYYYMMDDTHHMMSSsssZ}` (millisecond precision). Example: `surge-simple-480k-10k-20260312T143022500Z`. See [storage-provenance-spec.md §1](storage-provenance-spec.md#1-ids).                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **Shard ID**                  | Logical index for a shard (`shard-000042`). Deterministic, defined at run creation, independent of which worker computes it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **worker_id**                 | Infrastructure identifier (e.g., RunPod's `RUNPOD_POD_ID`). Appears only in metadata, not in shard paths.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Reconciliation**            | Comparing desired state (spec) against actual state (validated shards in R2) to determine what work remains.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **metadata/dataset.complete** | Marker file written by finalize as the very last step. Means "finalization is done" — not a mutex or lock. Contains run_id and timestamp.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Debug log**                 | JSONL file (`metadata/workers/attempts/{worker_id}-{attempt}/debug.log`) of structured events from a worker. Append-only, uploaded by EXIT trap, survives crashes.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Worker report**             | JSON summary (`metadata/workers/attempts/{worker_id}-{attempt}/report.json`) of a worker's results, including content hashes for provenance. Written at exit, missing if worker crashed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **Lifecycle marker**          | Empty file in `metadata/workers/shards/shard-{id}/` named `{worker_id}-{attempt}.{state}`. Three commit points: `.rendering` (attempt started), `.valid` (staged attempt committed), `.promoted` (attempt included in finalized output). Plus `.invalid` (validation failed). Presence is the state — no content to parse.                                                                                                                                                                                                                                                                                                                                                       |
| **Quarantined shard**         | A corrupt shard uploaded by the worker to `metadata/workers/shards/shard-{id}/quarantine/` on validation failure. Preserves the evidence for debugging alongside lifecycle markers.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **Dataset card**              | JSON file (`dataset.json`) describing the finalized dataset: provenance, structure, stats. References the spec by SHA-256.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **param_spec**                | Configuration selecting which synthesizer parameters to vary. Determines prediction task dimensionality. Registered specs live in `param_specs` in [`src/synth_setter/data/vst/__init__.py`](../../src/synth_setter/data/vst/__init__.py); see also the [glossary entry](../glossary.md).                                                                                                                                                                                                                                                                                                                                                                                        |
| **VST**                       | Virtual Studio Technology — plugin format for audio synthesizers. Surge XT is the VST used for rendering.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Mel spectrogram**           | Frequency-domain audio representation used as neural network input. 128 mels, ~100 frames/sec.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **Fully parallel**            | Workload where tasks are completely independent — no communication or shared state between workers.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **rclone**                    | CLI tool for syncing files to cloud storage. Used as the R2 upload/download mechanism.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **Lance**                     | [Lance](https://github.com/lance-format/lance), a columnar data format for ML datasets. Stores samples as Arrow fixed-shape-tensor columns in versioned dataset directories with per-column projection (§7.10). The pipeline's sole output format.                                                                                                                                                                                                                                                                                                                                                                                                                               |

## Appendix B: Tech Stack

| Component       | Technology                                                                                                                                                                  | Role                                             |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| Build           | Docker (BuildKit)                                                                                                                                                           | Reproducible compute environments                |
| Storage         | Cloudflare R2                                                                                                                                                               | Data + coordination, free egress                 |
| Execution       | RunPod                                                                                                                                                                      | Cheap on-demand cloud workers                    |
| Tracking        | Weights & Biases                                                                                                                                                            | Pipeline metrics, dataset artifact registry      |
| Data format     | [Lance](https://github.com/lance-format/lance)                                                                                                                              | Shard generation, finalize, and training format  |
| CLI             | [Click](https://click.palletsprojects.com/)                                                                                                                                 | Typed arguments, validation, `--help`            |
| Validation      | [Pydantic](https://docs.pydantic.dev/) (frozen models; `strict=True` on `DatasetSpec`, `RenderConfig`, and `ShardSpec`; JSON round-trip coercions via per-field validators) | DatasetSpec, report, and config validation       |
| Logging         | [structlog](https://www.structlog.org/)                                                                                                                                     | Structured JSON debug logging                    |
| Retry           | [tenacity](https://tenacity.readthedocs.io/)                                                                                                                                | Centralized retry policy                         |
| Upload/download | [rclone](https://rclone.org/)                                                                                                                                               | R2 file transfer; all transfers use `--checksum` |
| Containers      | [Docker](https://docs.docker.com/build/buildkit/) (BuildKit)                                                                                                                | Reproducible environments                        |
| Audio           | [Surge XT](https://surge-synthesizer.github.io/) (headless, Xvfb)                                                                                                           | VST synthesis                                    |

## Appendix C: References

- [Industrial Empathy — Design Docs at Google](https://www.industrialempathy.com/posts/design-docs-at-google/) — section structure, review process
- [Eugene Yan — ML Design Docs](https://eugeneyan.com/writing/ml-design-docs/) — ML-specific methodology sections

## Appendix D: Implementation Roadmap

Full implementation plan: [data-pipeline-implementation-plan.md](data-pipeline-implementation-plan.md) · Epic: [#74](https://github.com/tinaudio/synth-setter/issues/74)

| Phase | Scope                                        | Tasks   | GitHub issue |
| ----- | -------------------------------------------- | ------- | ------------ |
| 1     | Foundation — deps, shared code, CI           | 1.1–1.5 | #68          |
| 2     | Pipeline Core — schemas, storage, validation | 2.1–2.3 | #69          |
| 3     | Docker — Dockerfile, entrypoint, headless    | 3.1     | #70          |
| 4     | Pipeline Engine — reconciliation, compute    | 4.1–4.2 | #71          |
| 5     | Pipeline CLI — generate, status, finalize    | 5.1–5.3 | #72          |
| 6     | Production — RunPod backend, E2E             | 6.1     | #73          |

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
# In finalize, after committing fragments and stats:
import wandb

run = wandb.init(
    project="synth-setter",
    job_type="data-generation",
    id=spec.run_id,
)

# Log pipeline metrics to summary (final values, not time-series)
run.summary["pipeline/shards_total"] = spec.num_shards
run.summary["pipeline/shards_valid"] = validation_summary.valid
run.summary["pipeline/shards_quarantined"] = validation_summary.quarantined
run.summary["pipeline/total_samples"] = total_samples
run.summary["pipeline/errors_total"] = total_errors

# Register dataset as artifact
artifact = wandb.Artifact(
    name=f"data-{spec.task_name}",  # name follows storage-provenance-spec.md §4
    type="dataset",
    metadata={
        "dataset_config_id": spec.task_name,
        "dataset_wandb_run_id": spec.run_id,
        "shard_count": spec.num_shards,
        "param_spec": spec.render.param_spec_name,
        "code_version": spec.git_sha,
        "total_samples": total_samples,
        "splits": card.splits,
    },
)
artifact.add_file(input_spec_path)        # input_spec.json
artifact.add_file(card_path)        # dataset.json
artifact.add_reference(
    f"s3://{spec.r2.bucket}/{spec.r2.prefix}"  # prefix is data/{dataset_config_id}/{dataset_wandb_run_id}/
)  # R2 is S3-compatible; W&B resolves via S3 API (uses AWS_ENDPOINT_URL or WANDB_S3_ENDPOINT_URL; see storage-provenance-spec.md §11)
run.log_artifact(artifact)
run.finish()
```

Training runs declare the dataset as an input, closing the lineage loop:

```python
artifact = run.use_artifact(f"data-{dataset_config_id}:latest")
```

______________________________________________________________________
