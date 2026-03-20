# Design Doc: Evaluation Pipeline & R2 Integration

> **Status**: Draft
> **Author**: ktinubu@
> **Last Updated**: 2026-03-19
> **Tracking**: #98 (eval epic), #99 (R2 epic)

______________________________________________________________________

### Index

| §   | Section                                                                       | What it covers                                                        |
| --- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 1   | [Context & Motivation](#1-context--motivation)                                | Problem statement, current state, why this matters                    |
| 2   | [Typical Workflow](#2-typical-workflow)                                       | End-to-end CLI example — local and Docker                             |
| 3   | [Goals, Non-Goals & Design Principles](#3-goals-non-goals--design-principles) | Requirements, principles, anti-goals, success metrics                 |
| 4   | [System Overview](#4-system-overview)                                         | Three-stage architecture, data flow, environment matrix               |
| 5   | [Stage Definitions](#5-stage-definitions)                                     | Predict, render, metrics — inputs, outputs, contracts                 |
| 6   | [R2 Integration](#6-r2-integration)                                           | Dataset download, checkpoint sync, artifact upload                    |
| 7   | [Design Decisions](#7-design-decisions)                                       | Secrets vs paths, rclone wrapper, headless rendering, ckpt resolution |
| 8   | [Dependency Graph & Parallelism](#8-dependency-graph--parallelism)            | Issue dependencies, parallel execution windows, critical path         |
| 9   | [Alternatives Considered](#9-alternatives-considered)                         | Rejected approaches and why                                           |
| 10  | [Open Questions & Risks](#10-open-questions--risks)                           | Known gaps and trade-offs                                             |
| 11  | [Out of Scope](#11-out-of-scope)                                              | Future work — not referenced elsewhere                                |
| 12  | [Implementation Plan](#12-implementation-plan)                                | Phase breakdown, PR groupings, file lists, test strategy              |
| A–C | [Appendices](#appendix-a-glossary)                                            | Glossary, current file inventory, metric definitions                  |

______________________________________________________________________

## 1. Context & Motivation

Topline goal: Run the full evaluation pipeline — predict, render, metrics — on any developer machine or CI runner, with datasets and checkpoints fetched from R2 on demand.

**synth-setter** trains models that predict synthesizer parameters from audio. Evaluating these models is a three-stage pipeline:

1. **Predict** — load a trained checkpoint, run inference on a test dataset, output predicted parameter tensors
2. **Render** — feed predicted parameters into the VST plugin (Surge XT), render audio waveforms for both predictions and ground-truth targets
3. **Metrics** — compare predicted and target audio using spectral, envelope, and transport-based distance metrics

This pipeline works end-to-end today but is tightly coupled to a university HPC cluster:

| Coupling             | Where                                                | Impact                                           |
| -------------------- | ---------------------------------------------------- | ------------------------------------------------ |
| Hardcoded paths      | `configs/data/surge*.yaml` → `/data/scratch/acw585/` | Cannot run on any other machine                  |
| SGE directives       | `jobs/predict/*.sh` → `#$ -l gpu=1`                  | 19 near-identical scripts, one per model variant |
| Module system        | `module load gcc`, `module load hdf5-parallel`       | Not available outside HPC                        |
| Conda env            | `mamba activate perm`                                | Specific to cluster user's env                   |
| Apptainer container  | `apptainer exec --nv ...`                            | Not available on Mac/Linux dev machines          |
| Checkpoint retrieval | `scripts/get-ckpt-from-wandb.sh` (W&B download)      | Fragile, no R2 option                            |
| Data locality        | Datasets assumed at fixed cluster paths              | No remote download capability                    |

Separately, the data pipeline (#74) already uses R2 as the source of truth for generated datasets. Extending R2 to the eval workflow — auto-downloading datasets, syncing checkpoints, uploading eval artifacts — closes the loop so the full workflow (generate → train → eval) can run from any machine with an internet connection.

### Infrastructure Layers

| Layer         | Technology                                                        | Role                                      |
| ------------- | ----------------------------------------------------------------- | ----------------------------------------- |
| **Rendering** | [Surge XT](https://surge-synthesizer.github.io/) via pedalboard   | Audio synthesis from predicted parameters |
| **Display**   | Xvfb (Linux headless) / native (macOS)                            | VST plugins require a display server      |
| **Storage**   | [Cloudflare R2](https://developers.cloudflare.com/r2/) via rclone | Datasets, checkpoints, eval artifacts     |
| **Tracking**  | [Weights & Biases](https://wandb.ai/)                             | Experiment tracking, metric dashboards    |
| **Config**    | [Hydra](https://hydra.cc/) + OmegaConf                            | Config composition, env var interpolation |

## 2. Typical Workflow

### Local development (target state)

```bash
# 1. Set up credentials (one-time) — .env is for secrets only
cp .env.example .env
# Edit .env: R2 credentials, WANDB_API_KEY

# 2. Run prediction — dataset path and checkpoint are Hydra args, not env vars
make predict EXPERIMENT=surge/flow_simple CKPT=r2:synth-data/checkpoints/flow-simple/best.ckpt
# → Checkpoint downloaded to .cache/checkpoints/flow-simple/best.ckpt
# → Predictions written to logs/eval/flow_simple/{run}-{timestamp}/predictions/

# 3. Render audio — auto-detects display, launches Xvfb if headless
make render PRED_DIR=logs/eval/flow_simple/{run}-{timestamp}/predictions/ OUTPUT_DIR=logs/eval/flow_simple/{run}-{timestamp}/audio/
# → Audio written to logs/eval/flow_simple/{run}-{timestamp}/audio/sample_{0..N}/

# 4. Compute metrics
make metrics AUDIO_DIR=logs/eval/flow_simple/{run}-{timestamp}/audio/ OUTPUT_DIR=logs/eval/flow_simple/{run}-{timestamp}/metrics/
# → logs/eval/flow_simple/{run}-{timestamp}/metrics/metrics.csv
# → logs/eval/flow_simple/{run}-{timestamp}/metrics/aggregated_metrics.csv

# 5. (Optional) Upload artifacts to R2
make upload-eval RUN_DIR=logs/eval/flow_simple/{run}-{timestamp}/
# → r2:synth-data/eval/flow_simple/{predictions,audio,metrics}/
```

### Full pipeline (CI or Docker)

```bash
# Docker — everything in one container, headless rendering included
make docker-eval EXPERIMENT=surge/flow_simple CKPT=r2:synth-data/checkpoints/flow-simple/best.ckpt
# → Runs predict → render → metrics inside container
# → Copies metrics.csv to host
```

### SGE cluster (deprecated — no engineering effort)

The 19 SGE scripts in `jobs/predict/` stay as-is. They are not ported, consolidated,
or maintained. If they still work on the cluster, great. If they break, use the
portable `make` targets instead. No new code references SGE.

## 3. Goals, Non-Goals & Design Principles

### Goals

- **Run anywhere.** The evaluation pipeline must work on local macOS dev machines, local Linux machines, Docker containers, and CI runners. Environment differences are handled by config, not by code forks.
- **R2 as the artifact backbone.** Datasets, checkpoints, and eval outputs are stored in R2. Any machine with credentials can pull what it needs and push what it produces. No more "the data is on the cluster."
- **Zero manual data wrangling.** If a dataset or checkpoint isn't local, the pipeline fetches it from R2 automatically. If eval outputs should be archived, the pipeline uploads them. No `rclone sync` commands in READMEs.
- **Idempotent and resumable.** Every `make` target is safe to re-run. `rclone --checksum` ensures no redundant transfers. Rendering only processes missing audio. Metrics only recompute when inputs change.
- **SGE is deprecated.** The 19 SGE scripts stay as-is — no engineering effort to maintain or consolidate them. They may still work on the cluster but are not tested or supported going forward.
- **Debuggable.** When a metric looks wrong, you can trace from the aggregated CSV → per-sample CSV → rendered audio → predicted parameters → checkpoint → training run → dataset. Every link in this chain is a file you can inspect.

### Design Principles

- **Secrets in `.env`, paths in Hydra** — `.env` holds only credentials (R2, W&B). All paths use Hydra defaults with CLI overrides ([§7.1](#71-secrets-in-env-paths-in-hydra))
- **Auto-detect, don't configure** — headless rendering detects the display server automatically ([§7.3](#73-headless-rendering))
- **Storage before compute** — verify datasets/checkpoints exist before running inference ([§7.4](#74-storage-before-compute))
- **Experiment configs pin models** — each model variant has its own experiment config with a pinned checkpoint ([§7.5](#75-checkpoint-resolution))
- **`--checksum` always** — all rclone operations use checksum verification (project rule from CLAUDE.md)

### What This System Deliberately Avoids

- **Automatic stage chaining** — predict, render, metrics are explicit `make` targets. At 1-2 evals/week, chaining adds complexity without value.
- **Eval-specific orchestrator** — Makefile targets are sufficient. No Airflow, no Prefect, no custom DAG engine.
- **Streaming metrics** — metrics are computed in batch after all audio is rendered, not incrementally.
- **GPU scheduling** — prediction uses whatever GPU is available; scheduling is the cluster's job.
- **Multi-model comparison framework** — comparing models is done by running the pipeline twice and diffing CSVs.

### Success Metrics

| Metric                     | Target                                                       | How to Measure                                       |
| -------------------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| Local eval from cold start | Fresh clone → metrics CSV in < 15 min (small fixture)        | Time from `git clone` to `metrics.csv` on dev laptop |
| Environment coverage       | Works on macOS, Linux, Docker, CI                            | CI matrix                                            |
| Data fetch reliability     | `r2:` paths resolve and download without manual intervention | E2E test with R2 fixture                             |
| Zero hardcoded paths       | No `/data/scratch/` in committed configs                     | `grep -r '/data/scratch' configs/`                   |

### Non-Goals

- **Training pipeline changes.** This doc covers eval and R2 integration only. Training orchestration is a separate concern.
- **Real-time eval.** Batch eval, triggered manually or by CI.
- **Custom metric development.** Existing metrics (MSS, wMFCC, SOT, RMS) are fixed. Adding new metrics is future work.
- **Multi-user eval infrastructure.** Single-user research pipeline.
- **Replacing W&B.** W&B remains the experiment tracker. R2 complements it for large artifacts.

## 4. System Overview

The evaluation pipeline is a three-stage batch pipeline. Each stage is an independent command with well-defined inputs and outputs. R2 serves as the backing store for datasets, checkpoints, and (optionally) eval artifacts.

```
                    ┌──────────────────────────────────────────────┐
                    │           R2 (synth-data bucket)             │
                    │                                              │
                    │  data/                checkpoints/           │
                    │    surge-simple/         flow-simple/        │
                    │    surge-full/           vae-simple/         │
                    │                                              │
                    │  eval/                                       │
                    │    {experiment}/{run_id}/                    │
                    │      predictions/ audio/ metrics/            │
                    └──────┬─────────────────┬─────────────────────┘
                           │                 │
                    download if needed  upload if configured
                           │                 │
                           ▼                 ▲
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  checkpoint  │───►│   PREDICT    │───►│    RENDER    │───►│   METRICS    │
│  + dataset   │    │              │    │              │    │              │
│  (local or   │    │ src/eval.py  │    │ renderscript │    │ compute_     │
│   R2)        │    │ mode=predict │    │ .sh          │    │ audio_       │
└─────────────┘    │              │    │              │    │ metrics.py   │
                    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
                           │                   │                   │
                           ▼                   ▼                   ▼
                    pred-*.pt            sample_N/           metrics.csv
                    target-audio-*.pt    ├─ pred.wav         aggregated_
                    target-params-*.pt   ├─ target.wav       metrics.csv
                                         ├─ spec.png
                                         └─ params.csv
```

### Environment Matrix

| Environment    | Dataset source | Display      | VST plugin          | GPU  | Trigger            |
| -------------- | -------------- | ------------ | ------------------- | ---- | ------------------ |
| macOS dev      | Local or R2    | Native       | `plugins/Surge XT`  | MPS  | `make predict`     |
| Linux dev      | Local or R2    | Xvfb (auto)  | `plugins/Surge XT`  | CUDA | `make predict`     |
| Docker         | R2             | Xvfb (baked) | Baked in image      | CUDA | `make docker-eval` |
| GitHub Actions | R2 (fixture)   | Xvfb         | Headless stub or CI | None | PR trigger         |

## 5. Stage Definitions

### 5.1 Predict

| Property    | Value                                                                                                 |
| ----------- | ----------------------------------------------------------------------------------------------------- |
| **Command** | `python src/eval.py mode=predict experiment={exp} data={data} ckpt_path={ckpt}`                       |
| **Input**   | Trained checkpoint (`.ckpt`), test dataset (HDF5 shard or virtual dataset)                            |
| **Output**  | `pred-{batch_idx}.pt`, `target-audio-{batch_idx}.pt`, `target-params-{batch_idx}.pt`                  |
| **Compute** | GPU — model forward pass                                                                              |
| **Config**  | Hydra composition: `configs/eval.yaml` + `configs/data/{data}.yaml` + `configs/experiment/{exp}.yaml` |

The predict stage loads a trained model checkpoint via PyTorch Lightning's `Trainer.predict()`, runs inference on the test split, and writes predicted parameter tensors to disk using a `PredictionWriter` callback.

**Key behaviors:**

- Dataset path resolved from `data.dataset_root` (default: `${paths.data_dir}/surge-simple`, CLI override for cluster)
- If `data.r2_path` is explicitly set, `SurgeDataModule.prepare_data()` syncs from R2 before loading
- Checkpoint path supports `r2:` prefix — auto-downloads to local cache before loading
- Output directory: `{paths.log_dir}/eval/{experiment_name}/{run_id}/predictions/`

### 5.2 Render

| Property     | Value                                                                                                    |
| ------------ | -------------------------------------------------------------------------------------------------------- |
| **Command**  | `python scripts/predict_vst_audio.py {pred_dir} {output_dir} --plugin_path {vst} --preset_path {preset}` |
| **Input**    | Predicted parameter tensors (`.pt` files from predict stage)                                             |
| **Output**   | `sample_{N}/pred.wav`, `sample_{N}/target.wav`, `sample_{N}/spec.png`, `sample_{N}/params.csv`           |
| **Compute**  | CPU — VST audio rendering via pedalboard                                                                 |
| **Requires** | Display server (Xvfb on headless Linux, native on macOS)                                                 |

The render stage loads each predicted parameter tensor, decodes it using the `ParamSpec`, and renders audio through the Surge XT VST plugin via pedalboard. It also renders the ground-truth target audio for comparison.

**Key behaviors:**

- `renderscript.sh` wraps `predict_vst_audio.py` with display server management
- On macOS: uses native display, no wrapper needed — `make render` calls the Python script directly
- On headless Linux: launches Xvfb, sets `DISPLAY`, runs script, kills Xvfb
- Plugin path default: `plugins/Surge XT.vst3` (overridable via `--plugin_path`)
- Preset path default: `presets/surge-base.vstpreset` (overridable via `--preset_path`)
- Parameters are denormalized from `[-1, 1]` → `[0, 1]` before decoding

### 5.3 Metrics

| Property    | Value                                                                                     |
| ----------- | ----------------------------------------------------------------------------------------- |
| **Command** | `python scripts/compute_audio_metrics.py {audio_dir} {output_dir}`                        |
| **Input**   | Directory of `sample_{N}/` subdirectories, each containing `pred.wav` and `target.wav`    |
| **Output**  | `metrics.csv` (per-sample), `aggregated_metrics.csv` (mean/std across samples)            |
| **Compute** | CPU — spectral analysis, DTW, optimal transport (parallelized with `ProcessPoolExecutor`) |

Four metrics are computed for each (predicted, target) audio pair:

| Metric    | Full Name                  | Method                                       | Range     |
| --------- | -------------------------- | -------------------------------------------- | --------- |
| **MSS**   | Multi-Scale Spectrogram    | L1 on mel spectrograms at 3 time scales      | \[0, ∞) ↓ |
| **wMFCC** | Weighted MFCC              | DTW cost between MFCC sequences              | \[0, ∞) ↓ |
| **SOT**   | Spectral Optimal Transport | Wasserstein distance on normalized STFT bins | \[0, ∞) ↓ |
| **RMS**   | RMS Amplitude Envelope     | Cosine similarity of RMS envelopes           | [-1, 1] ↑ |

**Key behaviors:**

- Uses `ProcessPoolExecutor` for parallel metric computation across samples
- Audio loaded via `pedalboard.io.AudioFile` at native sample rate
- MSS uses three windows: 10ms, 25ms, 100ms (hops: 5ms, 10ms, 50ms)
- Output CSV: per-sample metrics indexed by directory name, aggregated means/stds

## 6. R2 Integration

### 6.1 R2 Layout

```
r2:synth-data/
├── data/                         # Datasets (existing, from data pipeline)
│   └── {dataset_name}/
│       ├── shard-*.h5
│       ├── train.h5, val.h5, test.h5
│       └── stats.npz
├── checkpoints/                  # NEW: model checkpoints
│   └── {experiment}/
│       └── {run_id}/
│           ├── best.ckpt
│           └── last.ckpt
└── eval/                         # NEW: eval artifacts
    └── {experiment}/
        └── {run_id}/
            ├── predictions/      # .pt files
            ├── audio/            # sample_N/{pred,target}.wav
            └── metrics/          # metrics.csv, aggregated_metrics.csv
```

### 6.2 rclone Wrapper

All R2 operations go through a shared utility function. This avoids scattered `subprocess.run(["rclone", ...])` calls and enforces the `--checksum` rule.

```python
def rclone_sync(
    src: str,
    dst: str,
    *,
    flags: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Sync src to dst via rclone. Always uses --checksum. Raises on non-zero exit."""
```

R2 credentials are the **only** values that belong in `.env` — they are secrets that must never be committed:

```bash
# .env — secrets only, nothing else
RCLONE_CONFIG_R2_TYPE=s3
RCLONE_CONFIG_R2_ACCESS_KEY_ID=...
RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=...
RCLONE_CONFIG_R2_ENDPOINT=https://{account_id}.r2.cloudflarestorage.com
WANDB_API_KEY=...
```

Paths, dataset roots, checkpoint locations, and log directories are **not** in `.env` — they are Hydra config values with sensible defaults and CLI overrides.

### 6.3 Dataset Download

When `data.r2_path` is explicitly provided (via CLI override or experiment config), `SurgeDataModule.prepare_data()` syncs the dataset to `data.dataset_root` before the data loaders are created.

```yaml
# configs/data/surge_simple.yaml — no r2_path, no env vars for paths
_target_: src.data.surge_datamodule.SurgeDataModule
dataset_root: ${paths.data_dir}/surge-simple        # matches existing paths convention
# r2_path: deliberately absent — must be specified explicitly when needed
batch_size: 128
num_workers: 11
```

To use R2, pass it explicitly:

```bash
# CLI override — explicit, visible, no hidden state
python src/eval.py data.r2_path=r2:synth-data/data/surge-simple/ ...

# Or in an experiment config that opts in
# configs/experiment/surge/flow_simple.yaml
data:
  r2_path: r2:synth-data/data/surge-simple/
```

Behavior:

- If `r2_path` is absent (default) → no-op (local-only mode, no R2 dependency)
- If `dataset_root` already has the data (checksum match) → no-op
- Otherwise → `rclone_sync(r2_path, dataset_root)`
- **No default value for `r2_path`** — you opt in explicitly, never accidentally

### 6.4 Checkpoint Sync

See [§7.5](#75-checkpoint-resolution) for full `ckpt_path` resolution behavior.

**Download** (eval): `ckpt_path=r2:synth-data/checkpoints/flow-simple/best.ckpt`
resolves via rclone to `.cache/checkpoints/`, cached with checksum.

**Upload** (training): An `R2CheckpointUploader` callback piggybacks on Lightning's
`ModelCheckpoint` save events. Every time a checkpoint is saved locally (per existing
`every_n_train_steps: 5000`), the callback uploads it to R2:

```yaml
# configs/callbacks/default_surge.yaml — add to callback list when R2 upload desired
r2_checkpoint:
  _target_: src.callbacks.r2_checkpoint.R2CheckpointUploader
  r2_path: ???  # required, no default — e.g. r2:synth-data/checkpoints/flow-simple/run-123/
```

- **No default `r2_path`** — must be explicitly specified when adding the callback
- Uploads best + last on each save event
- `--checksum` prevents redundant uploads
- Worst-case loss on crash: `every_n_train_steps` interval (5000 steps)

### 6.5 Eval Artifact Upload

After metrics, optionally upload all eval outputs to R2:

```bash
make upload-eval RUN_DIR=logs/eval/flow_simple/{run}-{timestamp}/
# Equivalent to:
# rclone sync logs/eval/flow_simple/{run}-{timestamp}/ r2:synth-data/eval/flow_simple/{run}/ --checksum
```

Not automatic — explicit `make` target. Toggle via Hydra config or CLI flag.

## 7. Design Decisions

### 7.1 Secrets in `.env`, Paths in Hydra

**Decision:** `.env` holds only credentials (R2, W&B). All paths use plain Hydra defaults with CLI overrides — no `${oc.env:}` interpolation for paths.

```yaml
# Before — hardcoded cluster path
dataset_root: /data/scratch/acw585/surge-simple/

# After — uses existing paths convention, resolves to {PROJECT_ROOT}/data/surge-simple
dataset_root: ${paths.data_dir}/surge-simple
```

```bash
# On the cluster, override via CLI — explicit and visible
python src/eval.py data.dataset_root=/data/scratch/acw585/surge-simple/ ...

# R2 path — must be specified explicitly, no default
python src/eval.py data.r2_path=r2:synth-data/data/surge-simple/ ...
```

**Rationale:** `.env` files are invisible state — you can't read a config and know what it does without also reading `.env`. Hydra already has a CLI override mechanism. Using `${oc.env:DATA_ROOT}` in configs adds a second override layer that can conflict with the first. Keeping paths as plain Hydra values means:

- Configs are self-describing — read the YAML, know what happens
- CLI overrides are visible in the command line and in Hydra's `overrides.yaml` log
- No "what's in my `.env` again?" debugging
- `.env` has a single purpose: secrets that must never be committed

The only env vars in configs are `PROJECT_ROOT` (set automatically by `rootutils`) and credentials (`RCLONE_CONFIG_R2_*`, `WANDB_API_KEY`).

### 7.2 rclone Over boto3/S3 SDK

**Decision:** Use rclone (subprocess) for all R2 operations, not the AWS S3 SDK.

**Rationale:**

- The data pipeline already uses rclone — one tool, one set of docs, one failure mode
- `--checksum` is a first-class rclone flag (CLAUDE.md rule)
- rclone handles R2's S3 compatibility quirks transparently
- No additional Python dependency
- Trade-off: subprocess call is harder to unit test than a Python SDK. Mitigated by wrapping in a single function that can be mocked.

### 7.3 Headless Rendering

**Decision:** Auto-detect display availability rather than requiring configuration.

```bash
# renderscript.sh (simplified logic)
cleanup() { [[ -n "${XVFB_PID:-}" ]] && kill "$XVFB_PID" 2>/dev/null; }
trap cleanup EXIT

if [[ "$OSTYPE" == darwin* ]]; then
    # macOS — native display always available
    python scripts/predict_vst_audio.py "$@"
elif [[ -z "$DISPLAY" ]]; then
    # Headless Linux — launch Xvfb
    Xvfb :99 &
    XVFB_PID=$!
    export DISPLAY=:99
    python scripts/predict_vst_audio.py "$@"
else
    # Linux with display
    python scripts/predict_vst_audio.py "$@"
fi
```

**Rationale:** Requiring users to know whether they're headless and set `DISPLAY` manually is error-prone. Auto-detection handles all environments (macOS dev, Linux dev, Docker, CI) with zero configuration.

### 7.4 Storage Before Compute

**Decision:** Verify dataset and checkpoint availability before running any GPU inference.

**Rationale:** A missing dataset or corrupt checkpoint discovered mid-inference wastes GPU time. `prepare_data()` runs before `setup()` in Lightning's lifecycle — the natural place for this check. For R2 downloads, we validate the rclone exit code and file existence before proceeding.

### 7.5 Checkpoint Resolution

#### Current behavior

`ckpt_path` works differently in eval vs training:

| Context                     | Config value      | Behavior                                                                                                 |
| --------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------- |
| **Eval** (`eval.yaml`)      | `ckpt_path: ???`  | Required — Hydra errors if not provided. Forces explicit CLI arg.                                        |
| **Training** (`train.yaml`) | `ckpt_path: null` | Optional — `null` means start fresh. If provided, Lightning resumes (optimizer state, epoch, scheduler). |

Today, the 19 SGE scripts resolve checkpoints via `get-ckpt-from-wandb.sh`, which searches
`logs/train/` for a W&B run ID and finds the corresponding `last.ckpt` on the local filesystem.
This only works on the machine where training happened.

Each script hardcodes a specific W&B run ID — the checkpoint is **stable per model variant**,
not changing every run:

```bash
# jobs/predict/flow-simple.sh
source jobs/predict/get-ckpt-from-wandb.sh x118ylu9   # always this run ID
```

#### Proposed design

Three resolution patterns, each appropriate for a different use case:

| Pattern                | Where specified                             | Use case                                                                   | Example                                                      |
| ---------------------- | ------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------ |
| CLI arg                | Command line                                | Ad-hoc eval of a new/local checkpoint                                      | `python src/eval.py ckpt_path=./my-ckpt.ckpt`                |
| Experiment config      | `configs/experiment/surge/flow_simple.yaml` | Reproducible eval of a known model — checkpoint is pinned, portable via R2 | `ckpt_path: r2:synth-data/checkpoints/flow-simple/best.ckpt` |
| `null` (training only) | `configs/train.yaml`                        | Start training fresh                                                       | Already works                                                |

**Resolution order** (Hydra's standard override precedence):

1. CLI override → highest priority
2. Experiment config → pinned per model variant
3. Base config (`eval.yaml: ???`) → forces one of the above

**`r2:` prefix handling:**

- If `ckpt_path` starts with `r2:` → download to `.cache/checkpoints/{path}` via rclone
- If cached copy exists and checksum matches → no-op
- Replace `ckpt_path` with the resolved local path before passing to Lightning

**What this replaces:**

- `get-ckpt-from-wandb.sh` — replaced by `r2:` prefix in experiment configs
- Per-script W&B run IDs — replaced by pinned R2 paths in experiment YAML
- 19 SGE scripts — deprecated, not maintained

#### Proposed design outcomes

| Config value                                                 | What happens                                                      | Portable? | Reproducible?                 |
| ------------------------------------------------------------ | ----------------------------------------------------------------- | --------- | ----------------------------- |
| `ckpt_path: ???` (base eval.yaml)                            | Hydra errors — forces user to specify                             | —         | —                             |
| `ckpt_path: ./local/best.ckpt` (CLI)                         | Uses local file directly                                          | No        | No (path is machine-specific) |
| `ckpt_path: r2:synth-data/.../best.ckpt` (experiment config) | Downloads from R2, caches locally, passes local path to Lightning | Yes       | Yes (R2 path is stable)       |
| `ckpt_path: r2:synth-data/.../best.ckpt` (CLI override)      | Same as above, but ad-hoc                                         | Yes       | No (not pinned in config)     |
| `ckpt_path: null` (train.yaml)                               | Start training from scratch                                       | Yes       | Yes                           |
| `ckpt_path: r2:synth-data/.../last.ckpt` (training resume)   | Downloads last checkpoint, resumes optimizer/epoch state          | Yes       | Yes                           |

**Decision:** `ckpt_path` is not in `.env` (not a secret, not machine infrastructure). It is either a required CLI arg (ad-hoc) or pinned in an experiment config (reproducible). The `r2:` prefix makes pinned values portable across machines.

### 7.6 Makefile as CLI Interface

**Decision:** All eval operations are `make` targets — consistent with the existing `make test`, `make format` pattern.

| Target             | Maps to                                       |
| ------------------ | --------------------------------------------- |
| `make predict`     | `python src/eval.py mode=predict ...`         |
| `make render`      | `./renderscript.sh` or direct Python on macOS |
| `make metrics`     | `python scripts/compute_audio_metrics.py ...` |
| `make docker-eval` | `docker run ... make predict render metrics`  |
| `make upload-eval` | `rclone sync ... --checksum`                  |

**Rationale:** Make targets are discoverable (`make help`), composable, and already the project convention. They hide environment-specific complexity (display detection, R2 paths) behind a consistent interface.

### 7.7 Current vs Proposed: Full Comparison

This section consolidates every configuration and environment behavior change in one place.

#### Current behavior (as-is)

| Concern                   | Current mechanism                                                   | Where defined                              | Portable? | Problem                                           |
| ------------------------- | ------------------------------------------------------------------- | ------------------------------------------ | --------- | ------------------------------------------------- |
| **Dataset path**          | Hardcoded `/data/scratch/acw585/surge-simple/`                      | `configs/data/surge_simple.yaml`           | No        | Only works on university cluster                  |
| **Checkpoint resolution** | `get-ckpt-from-wandb.sh` searches local `logs/train/` by W&B run ID | `jobs/predict/*.sh` (19 scripts)           | No        | Requires training logs on same machine            |
| **Checkpoint path**       | `ckpt_path: ???` in eval, resolved by shell script to local path    | `configs/eval.yaml` + shell                | No        | Local filesystem dependency                       |
| **R2 dataset access**     | Not supported                                                       | —                                          | —         | Must manually copy data to machine                |
| **R2 checkpoint access**  | Not supported                                                       | —                                          | —         | Must train on same machine or use W&B `log_model` |
| **R2 checkpoint upload**  | W&B `log_model: true` uploads to W&B artifacts                      | `configs/logger/wandb.yaml`                | Partially | Slow for large files, no rclone/R2 path           |
| **Credentials**           | No `.env` pattern for R2                                            | —                                          | —         | No standardized credential management             |
| **Display handling**      | `renderscript.sh` assumes Linux + Xvfb                              | `renderscript.sh`                          | No        | Fails on macOS (no Xvfb needed), no auto-detect   |
| **Log directory**         | `${paths.root_dir}/logs/` via `PROJECT_ROOT`                        | `configs/paths/default.yaml`               | Yes       | Already works                                     |
| **Predict output**        | `${paths.output_dir}/predictions`                                   | `configs/callbacks/prediction_writer.yaml` | Yes       | Already works                                     |
| **W&B entity**            | Hardcoded `entity: "benhayes"`                                      | `configs/logger/wandb.yaml`                | No        | Wrong for other users                             |
| **SGE scripts**           | 19 near-identical scripts, one per model                            | `jobs/predict/*.sh`                        | No        | Copy-paste errors, cluster-only                   |
| **Eval CLI**              | Raw `python src/eval.py ...` with many args                         | Shell scripts                              | No        | No `make` targets, hard to discover               |

#### Proposed behavior (to-be)

| Concern                      | Proposed mechanism                                                          | Where defined                               | Portable? | Change from current                         |
| ---------------------------- | --------------------------------------------------------------------------- | ------------------------------------------- | --------- | ------------------------------------------- |
| **Dataset path**             | `dataset_root: ${paths.data_dir}/surge-simple` (paths convention)           | `configs/data/surge_simple.yaml`            | Yes       | Hardcoded → paths convention                |
| **Dataset path override**    | CLI: `data.dataset_root=/cluster/path/`                                     | Command line                                | Yes       | Implicit → explicit                         |
| **Checkpoint resolution**    | `ckpt_path: ???` (base), pinned in experiment configs                       | `configs/eval.yaml` + `configs/experiment/` | Yes       | Shell script → Hydra config                 |
| **Checkpoint: ad-hoc**       | CLI: `ckpt_path=./local/best.ckpt`                                          | Command line                                | No        | Same as today but without shell wrapper     |
| **Checkpoint: reproducible** | `ckpt_path: r2:synth-data/.../best.ckpt` in experiment config               | `configs/experiment/surge/flow_simple.yaml` | Yes       | **New** — portable, pinned                  |
| **R2 dataset access**        | `data.r2_path=r2:synth-data/...` triggers auto-download in `prepare_data()` | CLI or experiment config (no default)       | Yes       | **New** — explicit opt-in                   |
| **R2 checkpoint download**   | `r2:` prefix → rclone download to `.cache/checkpoints/`                     | `resolve_ckpt_path()` in `src/utils/`       | Yes       | **New** — replaces `get-ckpt-from-wandb.sh` |
| **R2 checkpoint upload**     | `R2CheckpointUploader` callback, fires on every `ModelCheckpoint` save      | `configs/callbacks/` (no default, explicit) | Yes       | **New** — crash-resilient periodic upload   |
| **Credentials**              | `.env` for R2 + W&B secrets only                                            | `.env` / `.env.example`                     | Yes       | **New** — secrets only, no paths            |
| **Display handling**         | Auto-detect: macOS native / Linux Xvfb / Docker baked                       | `renderscript.sh`                           | Yes       | Linux-only → cross-platform                 |
| **Log directory**            | `${paths.root_dir}/logs/` (unchanged)                                       | `configs/paths/default.yaml`                | Yes       | No change                                   |
| **Predict output**           | `${paths.output_dir}/predictions` (unchanged)                               | `configs/callbacks/prediction_writer.yaml`  | Yes       | No change                                   |
| **W&B entity**               | Configurable via env or CLI                                                 | `configs/logger/wandb.yaml`                 | Yes       | Hardcoded → configurable                    |
| **SGE scripts**              | Deprecated — left as-is, not maintained                                     | `jobs/predict/*.sh`                         | No        | Active → deprecated                         |
| **Eval CLI**                 | `make predict`, `make render`, `make metrics`                               | `Makefile`                                  | Yes       | **New** — discoverable, consistent          |

#### What changes, what stays

| Category       | Items that change                                                                                                                                                          | Items that stay |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| **Removed**    | Hardcoded cluster paths, `get-ckpt-from-wandb.sh` reliance, SGE as supported platform                                                                                      |                 |
| **Deprecated** | 19 SGE scripts (left in repo, no maintenance), W&B-only checkpoint download                                                                                                |                 |
| **New**        | `r2:` prefix resolution, `R2CheckpointUploader` callback, `r2_path` opt-in, `make` targets, cross-platform display, `.env` for secrets                                     |                 |
| **Modified**   | `dataset_root` (hardcoded → relative default), `renderscript.sh` (Linux-only → auto-detect), W&B entity (hardcoded → configurable)                                         |                 |
| **Unchanged**  | `ckpt_path: ???` in eval.yaml, `ckpt_path: null` in train.yaml, `log_dir`, `output_dir`, prediction writer, W&B metric logging, CSV logger, `ModelCheckpoint` save cadence |                 |

#### Diff analysis

**1. `.env` scope (§7.1)**

|                      | Current              | Proposed                                                             |
| -------------------- | -------------------- | -------------------------------------------------------------------- |
| **What's in `.env`** | Nothing standardized | R2 credentials + `WANDB_API_KEY`                                     |
| **Paths**            | Hardcoded in YAML    | Hydra defaults + CLI overrides                                       |
| **Risk eliminated**  | —                    | Invisible state: can't read YAML + `.env` and know what happens      |
| **Trade-off**        | —                    | Cluster users must pass CLI overrides instead of setting one env var |

**2. Checkpoint resolution (§7.5)**

|                            | Current                                     | Proposed                                                              |
| -------------------------- | ------------------------------------------- | --------------------------------------------------------------------- |
| **Eval checkpoint**        | Shell script finds local file by W&B run ID | Pinned `r2:` path in experiment config or CLI arg                     |
| **Training checkpoint**    | `ckpt_path: null` (start fresh)             | Same — no change                                                      |
| **Training resume**        | `ckpt_path=/local/path/last.ckpt`           | `ckpt_path=r2:synth-data/.../last.ckpt` (portable)                    |
| **Upload during training** | W&B `log_model: true` only                  | W&B + `R2CheckpointUploader` callback (explicit opt-in)               |
| **Risk eliminated**        | —                                           | "Checkpoint is on the cluster" — R2 makes it available everywhere     |
| **Trade-off**              | —                                           | Requires R2 credentials; first download is slow for large checkpoints |

**3. Dataset access (§6.3)**

|                     | Current                    | Proposed                                                                 |
| ------------------- | -------------------------- | ------------------------------------------------------------------------ |
| **Local data**      | Hardcoded path, must exist | `${paths.data_dir}/surge-simple` (existing convention), override via CLI |
| **Remote data**     | Not supported              | `r2_path` opt-in triggers auto-download                                  |
| **Risk eliminated** | —                          | "Data is on the cluster" — R2 makes it available everywhere              |
| **Trade-off**       | —                          | First download of a 100GB dataset takes time; cached after that          |

**4. Display handling (§7.3)**

|                     | Current       | Proposed                                 |
| ------------------- | ------------- | ---------------------------------------- |
| **macOS**           | Not supported | Native display, no Xvfb                  |
| **Headless Linux**  | Xvfb assumed  | Xvfb auto-launched if `$DISPLAY` unset   |
| **Docker**          | Not supported | Xvfb baked into image                    |
| **Risk eliminated** | —             | "Works on Linux only" → works everywhere |
| **Trade-off**       | —             | None — strictly better                   |

**5. SGE deprecation**

|                  | Current                            | Proposed                                                                                         |
| ---------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------ |
| **SGE scripts**  | 19 scripts, actively used          | Left as-is, not maintained                                                                       |
| **Cluster eval** | `qsub jobs/predict/flow-simple.sh` | `make predict EXPERIMENT=surge/flow_simple CKPT=r2:...` (SSH to cluster, run make target)        |
| **Risk**         | —                                  | If SGE scripts break, no fix is coming. Acceptable — cluster is not the primary dev environment. |

## 8. Dependency Graph & Parallelism

### Issue Dependencies

```
                    EVAL PIPELINE                              R2 INTEGRATION
                    ─────────────                              ──────────────

                ┌─── #86 Render ─────┐
                │    (P1, no blocker) │
                │                    │
 #94 Paths ──→ #85 Predict ─────────┼──→ #88 Docker ──→ #89 E2E CI
 (P1)       (P1)                    │    (P2)            (P2)
                │                    │
                ├─── #87 Metrics ────┤                     #90 rclone ──→ #91 R2 Dataset
                │    (P1, no blocker)│                     (P1)       │    (P1)
                │         │          │                                │
                │         ├──→ #96 W&B (P3)                           ├──→ #92 R2 Checkpoint
                │         │                                          │    (P1)
                │         └──→ #93 R2 Artifacts ◄────────────────────┘
                │              (P2)
                │
                └──→ #97 Runbook (P2)

```

### Blocking Matrix

| Issue | Title               | Blocked by    | Blocks                  |
| ----- | ------------------- | ------------- | ----------------------- |
| #94   | Config cleanup      | —             | #85, #91                |
| #85   | Portable predict    | #94           | #88, #89, #97           |
| #86   | Portable render     | —             | #88, #89, #97           |
| #87   | Portable metrics    | —             | #88, #89, #93, #96, #97 |
| #90   | rclone wrapper      | —             | #91, #92, #93           |
| #88   | Docker eval         | #85, #86, #87 | #89                     |
| #89   | E2E CI              | #85–88        | —                       |
| #91   | R2 dataset download | #90, #94      | —                       |
| #92   | R2 checkpoint sync  | #90           | —                       |
| #93   | R2 artifact upload  | #90, #87      | —                       |
| #96   | W&B metrics         | #87           | —                       |
| #97   | Eval runbook        | #85, #86, #87 | —                       |

### Parallel Execution Windows

**4 issues can start immediately (no blockers):** #94, #86, #87, #90

**Critical path:** `#94 → #85 → #88 → #89` (4 steps, longest chain)

**Two independent tracks** converge at:

- #91 (needs both #90 rclone wrapper + #94 clean configs)
- #93 (needs both #90 rclone wrapper + #87 metrics stage)

### Timeline

```
Mar 31 ─────────── Apr 07 ─────────── Apr 14 ── Apr 15
│                  │                  │          │
├── #94 Paths ─────┤                  │          │
├── #90 rclone ────┤                  │          │
│   ├── #85 Predict ───┤              │          │
│   ├── #91 R2 Dataset ┤              │          │
│   │   ├── #86 Render ────┤          │          │
│   │   ├── #92 R2 Ckpt ──┤          │          │
│   │   │   ├── #87 Metrics ───┤      │          │
│   │   │   │   ├── #88 Docker ──┤    │          │
│   │   │   │   ├── #93 R2 Art. ─┤    │          │
│   │   │   │   │   ├── #96 W&B ────┤  │          │
│   │   │   │   │   ├── #89 E2E CI ─┤  │          │
│   │   │   │   │   │   ├── #97 Runbook ┤         │
│                                              milestone
```

## 9. Alternatives Considered

| Alternative                              | Why rejected                                                                                             |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **DVC for data versioning**              | Adds a dependency for a problem rclone already solves. DVC's git integration is overkill for 2 datasets. |
| **boto3/S3 SDK instead of rclone**       | Data pipeline already standardized on rclone. Consistency > marginal testability gain.                   |
| **Snakemake/Nextflow for eval pipeline** | Massive dependency for a 3-stage linear pipeline. `make` is sufficient.                                  |
| **Automatic stage chaining**             | At 1-2 evals/week, the cognitive overhead of "what ran automatically?" exceeds the convenience.          |
| **Per-model Docker images**              | One eval image with model as a parameter. Multiple images are unnecessary build complexity.              |
| **W&B Artifacts for checkpoints**        | W&B artifacts are slow for large files and add API dependency. rclone + R2 is faster and simpler.        |
| **Config-driven display detection**      | Auto-detection is strictly better — no env var to forget, no "works on my machine."                      |

## 10. Open Questions & Risks

| #   | Question / Risk                                                                                 | Impact                                                | Status                   |
| --- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ------------------------ |
| 1   | **VST plugin licensing on CI runners** — can we legally run Surge XT in GitHub Actions?         | E2E CI may need a stub or fixture-based approach      | Open                     |
| 2   | **macOS pedalboard + Surge XT compatibility** — does the VST3 plugin load on Apple Silicon?     | Blocks macOS render stage                             | Needs testing            |
| 3   | **Large checkpoint download times** — best.ckpt may be 500MB+; first-run UX on slow connections | Mitigated by caching, but first run is slow           | Accepted                 |
| 4   | **Metrics reproducibility across platforms** — float differences in spectral computations       | May cause CI flakiness with tight tolerances          | Use relative tolerances  |
| 5   | **Xvfb availability in Docker base image** — may need to install in Dockerfile                  | Low risk, well-documented                             | Resolved by Docker stage |
| 6   | **rclone version skew** — different rclone versions on dev machines vs CI                       | Pin rclone version in Dockerfile and `.tool-versions` | Open                     |

## 11. Out of Scope

- **Automated hyperparameter sweeps** — eval runs are manually triggered
- **Multi-GPU distributed eval** — single-GPU is sufficient at current dataset sizes
- **Audio listening tests / perceptual evaluation** — future work, requires different tooling
- **Real-time inference server** — batch eval only
- **Custom metric development** — existing 4 metrics are fixed for v1.0.0
- **Training pipeline changes** — this doc covers eval and R2 only
- **Data pipeline modifications** — covered by [data pipeline design doc](data-pipeline.md) and #74

## 12. Implementation Plan

### Branch Strategy

```
main ──●──────────●────────────●──────────●──────────●──────────●──→
       │          │            │          │          │          │
       PR#1      PR#2         PR#3       PR#4       PR#5       PR#6
```

| PR                         | Issues        | Contents                                          | CI gate                                  |
| -------------------------- | ------------- | ------------------------------------------------- | ---------------------------------------- |
| **#1: Foundation**         | #94           | Config cleanup, sensible Hydra defaults           | `ruff check`, no hardcoded paths         |
| **#2: Portable Stages**    | #85, #86, #87 | Predict, render, metrics + Makefile targets       | `make predict/render/metrics` on fixture |
| **#3: R2 Core**            | #90, #91, #92 | rclone wrapper, dataset download, checkpoint sync | Unit tests with mock rclone              |
| **#4: Docker + Artifacts** | #88, #93      | Docker eval, R2 artifact upload                   | Docker build + `make docker-eval`        |
| **#5: CI + Observability** | #89, #96      | E2E CI, W&B metrics                               | E2E test passes in Actions               |
| **#6: Documentation**      | #97           | Eval runbook                                      | Docs build, link check                   |

**Branch:** `dev/eval-pipeline` off `main`
**Priorities:** TDD first, small commits, always-green CI.

### PR #1: Foundation (Phase 1)

#### Phase 1: Remove Hardcoded Paths (#94)

**Goal:** Replace all cluster-specific paths in committed configs with sensible Hydra defaults.

**Files to modify:**

- `configs/data/surge_simple.yaml` — `dataset_root` → `${paths.data_dir}/surge-simple` (matches `mnist.yaml` convention)
- `configs/data/surge_mini.yaml` — same pattern
- `configs/data/surge_simple_onehot.yaml` — same (if exists)
- `.env.example` — R2 credentials and `WANDB_API_KEY` only (no path vars)

**Tests:**

- `test_no_hardcoded_paths_in_configs` — grep committed YAML for `/data/scratch`
- `test_configs_have_sensible_defaults` — load each data config, verify `dataset_root` is a relative path

**Note:** The 19 SGE scripts in `jobs/predict/` are left as-is (deprecated, not consolidated).

### PR #2: Portable Stages (Phases 3–5)

#### Phase 3: Portable Predict (#85)

**Goal:** `make predict` works on local machines with portable Hydra config.

**Files to modify:**

- `src/eval.py` — ensure `mode=predict` works without cluster deps
- `Makefile` — add `predict` target

**Files to create:**

- `tests/test_eval_predict.py` — fixture-based predict test (`@pytest.mark.slow`)

**Key behaviors:**

- `dataset_root` has a sensible Hydra default; override via CLI when needed
- `ckpt_path` resolved per [§7.5](#75-checkpoint-resolution) — CLI arg or experiment config, supports `r2:` prefix
- `paths.log_dir` keeps the existing default (`${paths.root_dir}/logs/`)
- Fails fast with clear error if dataset not found

#### Phase 4: Portable Render (#86)

**Goal:** `make render` works on macOS (native display) and Linux (Xvfb auto-detect).

**Files to modify:**

- `renderscript.sh` — add macOS detection, Xvfb auto-launch
- `scripts/predict_vst_audio.py` — make plugin/preset paths configurable via env
- `Makefile` — add `render` target

**Files to create:**

- `tests/test_render.py` — test with fixture `.pt` files (`@pytest.mark.slow`)

**Key behaviors:**

- macOS: skip Xvfb, call Python directly
- Headless Linux: launch Xvfb on `:99`, export `DISPLAY`, clean up on exit
- Plugin/preset paths default to `plugins/` and `presets/` (overridable)

#### Phase 5: Portable Metrics (#87)

**Goal:** `make metrics` works with pinned dependencies and clean output schema.

**Files to modify:**

- `scripts/compute_audio_metrics.py` — remove dead JTFS code, pin dep versions
- `Makefile` — add `metrics` target

**Files to create:**

- `tests/test_metrics.py` — test with fixture `.wav` files, validate CSV schema

**Key behaviors:**

- Output CSV: per-sample metrics indexed by directory name, aggregated means/stds
- `ProcessPoolExecutor` parallelism preserved
- Dead code (commented JTFS, unused f0) removed

### PR #3: R2 Core (Phases 6–8)

#### Phase 6: rclone Wrapper (#90)

**Goal:** Shared `rclone_sync()` utility with `--checksum` enforcement.

**Files to create:**

- `src/data/rclone.py` — `rclone_sync()`, `rclone_ls()`, `rclone_copyto()`
- `tests/test_rclone.py` — mock subprocess, verify flags

**Key behaviors:**

- All operations include `--checksum`
- R2 config from env vars (`RCLONE_CONFIG_R2_*`)
- Raises `subprocess.CalledProcessError` on failure
- Dry-run mode for testing (`--dry-run` flag passthrough)

#### Phase 7: R2 Dataset Download (#91)

**Goal:** When `data.r2_path` is explicitly specified, `prepare_data()` syncs from R2.

**Files to modify:**

- `src/data/surge_datamodule.py` — add optional `r2_path` field, call `rclone_sync` in `prepare_data()`
- Data configs unchanged — `r2_path` is absent by default, specified via CLI or experiment config

**Files to create:**

- `tests/test_r2_dataset_download.py` — mock rclone, verify sync logic

**Key behaviors:**

- No-op if `r2_path` not specified (default — local-only mode)
- No-op if local data matches (checksum)
- Sync runs in `prepare_data()` (before `setup()`)
- Logs download progress via structlog
- **No default value** — R2 download is always an explicit opt-in

#### Phase 8: R2 Checkpoint Sync (#92)

**Goal:** `ckpt_path=r2:...` auto-downloads; training auto-uploads best checkpoint.

**Files to modify:**

- `src/eval.py` — intercept `r2:` prefix, resolve to local cache
- `src/train.py` (or callback) — optional R2 upload after training

**Files to create:**

- `src/utils/ckpt_resolver.py` — `resolve_ckpt_path()` function
- `tests/test_ckpt_resolver.py` — mock rclone, verify cache logic

**Key behaviors:**

- Cache dir: `.cache/checkpoints/` (gitignored)
- Checksum validation prevents redundant downloads
- Upload is opt-in via `R2CheckpointUploader` callback with `r2_path` config

### PR #4: Docker + Artifacts (Phases 9–10)

#### Phase 9: Docker Eval Environment (#88)

**Goal:** `make docker-eval` runs the full pipeline in a container.

**Files to create:**

- `docker/eval/Dockerfile` — multi-stage: base → deps → VST plugin → Xvfb
- `docker/eval/docker-compose.yaml` — env var passthrough, volume mounts
- `Makefile` — add `docker-eval`, `docker-eval-build` targets

**Key behaviors:**

- Xvfb baked into image (always headless in Docker)
- Surge XT plugin installed in image
- `.env` file mounted for credentials only (R2, W&B)
- Output directory mounted as volume
- Paths passed as `docker run` args, not env vars

#### Phase 10: R2 Eval Artifact Upload (#93)

**Goal:** `make upload-eval` pushes predictions + audio + metrics to R2.

**Files to modify:**

- `Makefile` — add `upload-eval` target

**Files to create:**

- `scripts/upload_eval_artifacts.py` — rclone sync wrapper for eval outputs
- `tests/test_upload_eval.py` — mock rclone, verify R2 paths

### PR #5: CI + Observability (Phases 11–12)

#### Phase 11: E2E Eval CI (#89)

**Goal:** GitHub Actions workflow runs predict → render → metrics on a small fixture.

**Files to create:**

- `.github/workflows/eval-ci.yml` — matrix: Ubuntu (Xvfb)
- `tests/test_eval_e2e.py` — fixture-based integration test (`@pytest.mark.slow`)
- `tests/fixtures/eval/` — small checkpoint + dataset fixture

**Key behaviors:**

- Runs on PR (if `src/eval.py`, `scripts/`, or `configs/` changed)
- Uses fixture dataset (not R2) to avoid credential dependency
- Validates: predictions exist, audio renders, metrics CSV has expected schema

#### Phase 12: W&B Metrics Logging (#96)

**Goal:** Optionally log metrics to W&B for cross-run comparison.

**Files to modify:**

- `scripts/compute_audio_metrics.py` — add `--wandb-run` flag

**Files to create:**

- `tests/test_metrics_wandb.py` — mock wandb, verify log calls

### PR #6: Documentation (Phase 13)

#### Phase 13: Eval Runbook (#97)

**Goal:** Document how to run the full eval pipeline locally and in Docker.

**Files to create:**

- `docs/eval-runbook.md` — setup, credentials, make targets, Docker, troubleshooting

______________________________________________________________________

## Appendix A: Glossary

| Term           | Definition                                                                      |
| -------------- | ------------------------------------------------------------------------------- |
| **Predict**    | Run model inference on test data, outputting predicted synth parameter tensors  |
| **Render**     | Feed parameters into VST plugin to produce audio waveforms                      |
| **Metrics**    | Compute distance metrics between predicted and target audio                     |
| **ParamSpec**  | Mapping between model output indices and synthesizer parameters                 |
| **Xvfb**       | X Virtual Framebuffer — provides a virtual display for headless Linux rendering |
| **pedalboard** | Spotify's Python library for loading and running VST plugins                    |
| **rclone**     | CLI tool for syncing files to/from cloud storage (S3, R2, GCS, etc.)            |
| **OmegaConf**  | Hydra's config library — supports interpolation and CLI overrides               |
| **SGE**        | Sun Grid Engine — HPC job scheduler used on university clusters                 |

## Appendix B: Current File Inventory

### Eval Scripts

| File                               | Lines    | Purpose                                     | Cluster coupling                               |
| ---------------------------------- | -------- | ------------------------------------------- | ---------------------------------------------- |
| `src/eval.py`                      | 121      | Hydra entry point for predict/test/validate | Hardcoded paths in data configs                |
| `scripts/predict_vst_audio.py`     | 232      | VST rendering from predicted parameters     | Plugin path defaults                           |
| `renderscript.sh`                  | 59       | Xvfb wrapper for headless rendering         | Assumes Linux, no macOS support                |
| `scripts/compute_audio_metrics.py` | 323      | Parallel metric computation                 | None (already portable)                        |
| `jobs/predict/*.sh`                | 19 files | SGE job scripts, one per model (deprecated) | SGE directives, hardcoded paths, `module load` |

### Data Configs

| File                             | Hardcoded path                       |
| -------------------------------- | ------------------------------------ |
| `configs/data/surge_simple.yaml` | `/data/scratch/acw585/surge-simple/` |
| `configs/data/surge_mini.yaml`   | `/data/scratch/acw585/surge-mini/`   |

### Audio Dir Manifests

| File                            | Purpose                                              |
| ------------------------------- | ---------------------------------------------------- |
| `scripts/audio_dirs/simple.txt` | List of absolute paths to rendered audio directories |
| `scripts/audio_dirs/full.txt`   | Same, for full dataset                               |
| `scripts/audio_dirs/nsynth.txt` | Same, for NSynth dataset                             |

## Appendix C: Metric Computation Details

### MSS (Multi-Scale Spectrogram)

Three mel-scale windows capture different temporal characteristics:

| Scale  | Window | Hop  | n_mels | What it captures          |
| ------ | ------ | ---- | ------ | ------------------------- |
| Fine   | 10ms   | 5ms  | 64     | Transient attacks, clicks |
| Mid    | 25ms   | 10ms | 128    | Timbral detail            |
| Coarse | 100ms  | 50ms | 128    | Spectral envelope shape   |

L1 distance between mel spectrograms at each scale, averaged.

### wMFCC (Weighted MFCC)

1. Extract 13 MFCCs from both signals
2. Compute DTW alignment cost between MFCC sequences
3. Weight by frame energy (loud frames matter more)

### SOT (Spectral Optimal Transport)

1. Compute STFT of both signals
2. Normalize magnitude spectra to probability distributions per frame
3. Compute Wasserstein-1 distance between distributions
4. Average across frames

### RMS (Amplitude Envelope)

1. Compute RMS energy in overlapping windows
2. Cosine similarity between RMS envelopes
3. Range: [-1, 1] where 1 = identical envelope shape
