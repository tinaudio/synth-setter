# Design Doc: Evaluation Pipeline & R2 Integration

> **Status**: Draft
> **Author**: ktinubu@
> **Last Updated**: 2026-03-20
> **Tracking**:
>
> - #98 (Epic: evaluation pipeline)
> - #99 (Epic: storage / R2 integration)
>
> **Milestone**: `evaluation v1.0.0`
> **Domain Labels**: `evaluation`, `storage`

______________________________________________________________________

### Index

| §   | Section                                                                       | What it covers                                                                                     |
| --- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 1   | [Context & Motivation](#1-context--motivation)                                | Problem statement, current state, why this matters                                                 |
| 2   | [Typical Workflow](#2-typical-workflow)                                       | End-to-end CLI example — local and Docker                                                          |
| 3   | [Goals, Non-Goals & Design Principles](#3-goals-non-goals--design-principles) | Requirements, principles, anti-goals, success metrics                                              |
| 4   | [System Overview](#4-system-overview)                                         | Three-stage architecture, data flow, environment matrix                                            |
| 5   | [Stage Definitions](#5-stage-definitions)                                     | Predict, render, metrics — inputs, outputs, contracts                                              |
| 6   | [R2 Integration](#6-r2-integration)                                           | Dataset download, checkpoint download, artifact upload, W&B lineage                                |
| 7   | [Design Decisions](#7-design-decisions)                                       | Headless rendering, checkpoint resolution, Makefile, storage split, current vs proposed comparison |
| 8   | [Phase Plan](#8-phase-plan)                                                   | Epic → Phase → Task hierarchy, issue mapping, file lists, test strategy                            |
| 9   | [Dependency Overview](#9-dependency-overview)                                 | Issue dependencies, parallel execution windows, critical path                                      |
| 10  | [Alternatives Considered](#10-alternatives-considered)                        | Rejected approaches and why                                                                        |
| 11  | [Open Questions & Risks](#11-open-questions--risks)                           | Known gaps and trade-offs                                                                          |
| 12  | [Out of Scope](#12-out-of-scope)                                              | Future work — not referenced elsewhere                                                             |
| A–C | [Appendices](#appendix-a-glossary)                                            | Glossary, current file inventory, metric definitions                                               |

______________________________________________________________________

## 1. Context & Motivation

Topline goal: Run the full evaluation pipeline — predict, render, metrics — on any developer machine or CI runner, with datasets fetched from R2 and checkpoints from W&B on demand.

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

Separately, the data pipeline (#74) already uses R2 as the source of truth for generated datasets. Extending R2 to the eval workflow — auto-downloading datasets and uploading eval artifacts — and using W&B for checkpoint storage closes the loop so the full workflow (generate → train → eval) can run from any machine with an internet connection.

### Infrastructure Layers

| Layer         | Technology                                                        | Role                                      |
| ------------- | ----------------------------------------------------------------- | ----------------------------------------- |
| **Rendering** | [Surge XT](https://surge-synthesizer.github.io/) via pedalboard   | Audio synthesis from predicted parameters |
| **Display**   | Xvfb (Linux headless) / native (macOS)                            | VST plugins require a display server      |
| **Storage**   | [Cloudflare R2](https://developers.cloudflare.com/r2/) via rclone | Datasets, eval artifacts                  |
| **Tracking**  | [Weights & Biases](https://wandb.ai/)                             | Experiment tracking, metric dashboards    |
| **Config**    | [Hydra](https://hydra.cc/) + OmegaConf                            | Config composition, env var interpolation |

## 2. Typical Workflow

### Local development (target state)

The experiment config pins everything needed to reproduce an eval — model, data, and checkpoint:

```yaml
# configs/experiment/surge/flow_simple.yaml (proposed)
defaults:
  - override /data: surge_simple
  - override /model: surge_flow
  - override /callbacks: eval_surge

experiment_name: flow_simple
ckpt_path: ${wandb:synth-permutations/model-x118ylu9:latest}
model:
  test_cfg_strength: 2.0
  test_sample_steps: 100
```

```bash
# 1. Set up credentials (one-time) — .env is for secrets only
cp .env.example .env
# Edit .env: R2 credentials, WANDB_API_KEY

# 2. Run full eval — predict → render → metrics in one command
make eval EXPERIMENT=surge/flow_simple
# → Checkpoint auto-downloaded from W&B via ${wandb:...} resolver (cached after)
# → Predictions, audio, and metrics written to logs/eval/surge_simple/x118ylu9/surge_simple/

# Or run stages individually:
make predict EXPERIMENT=surge/flow_simple
make render PRED_DIR=logs/eval/surge_simple/x118ylu9/surge_simple/predictions/ OUTPUT_DIR=logs/eval/surge_simple/x118ylu9/surge_simple/audio/
make metrics AUDIO_DIR=logs/eval/surge_simple/x118ylu9/surge_simple/audio/ OUTPUT_DIR=logs/eval/surge_simple/x118ylu9/surge_simple/metrics/

# 3. (Optional) Upload artifacts to R2
make upload-eval
# → rclone sync logs/eval/surge_simple/x118ylu9/surge_simple/ r2:synth-data/eval/surge_simple/x118ylu9/surge_simple/ --checksum
```

### Full pipeline (CI or Docker)

```bash
# Docker — everything in one container, headless rendering included
make docker-eval EXPERIMENT=surge/flow_simple
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
- **Debuggable.** When a metric looks wrong, you can trace from the aggregated CSV → per-sample CSV → rendered audio → predicted parameters → checkpoint → training run → dataset. Every link in this chain is a file you can inspect.
- **Idempotent and resumable.** Every `make` target is safe to re-run. `rclone --checksum` ensures no redundant transfers. Rendering only processes missing audio. Metrics only recompute when inputs change.

> **Note:** The 19 SGE scripts in `jobs/predict/` stay as-is — deprecated, no engineering effort to maintain them.

### Design Principles

- **Experiment configs pin models** — each model variant has its own experiment config with a pinned checkpoint ([§7.2](#72-checkpoint-resolution))
- **`--checksum` always** — all rclone operations use checksum verification (project rule from CLAUDE.md)

### What This System Deliberately Avoids

- **Automatic stage chaining** — predict, render, metrics are explicit `make` targets. At 1-2 evals/week, chaining adds complexity without value.
- **Eval-specific orchestrator** — Makefile targets are sufficient. No Airflow, no Prefect, no custom DAG engine.

### Success Metrics

| Metric                     | Target                                                       | How to Measure                                       |
| -------------------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| Local eval from cold start | Fresh clone → metrics CSV in < 15 min (small fixture)        | Time from `git clone` to `metrics.csv` on dev laptop |
| Environment coverage       | Works on macOS, Linux, Docker, CI                            | CI matrix                                            |
| Data fetch reliability     | `r2:` paths resolve and download without manual intervention | E2E test with R2 fixture                             |
| Zero hardcoded paths       | No `/data/scratch/` in committed configs                     | `grep -r '/data/scratch' configs/`                   |

### Non-Goals

- **Training pipeline changes.** This doc covers eval and R2 integration only. Training orchestration is a separate concern.
- **Custom metric development.** Existing metrics (MSS, wMFCC, SOT, RMS) are fixed. Adding new metrics is future work.

## 4. System Overview

The evaluation pipeline is a three-stage batch pipeline. Each stage is an independent command with well-defined inputs and outputs. R2 serves as the backing store for datasets and eval artifacts; checkpoints are stored in W&B.

```
                    ┌──────────────────────────────────────────────┐
                    │           R2 (synth-data bucket)             │
                    │                                              │
                    │  data/                                       │
                    │    surge-simple/                              │
                    │    surge-full/                                │
                    │                                              │
                    │  eval/                                       │
                    │    {train_data}/{run_id}/{eval_data}/        │
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
| GitHub Actions | Repo fixture   | Xvfb         | Headless stub or CI | None | PR trigger         |

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
- Checkpoint path supports `${wandb:...}` resolver — auto-downloads from W&B artifacts to local cache
- Output directory: `${paths.output_dir}/predictions` (see `configs/callbacks/prediction_writer.yaml`)

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

### 6.1 Dataset Download

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

### 6.2 Checkpoint Storage (W&B Artifacts)

Checkpoints are stored in **W&B artifacts**, not R2. This is a deliberate decision — see [§10](#10-alternatives-considered) for the full R2-vs-W&B analysis.

**Upload** (training): `log_model="all"` in `configs/logger/wandb.yaml` uploads every checkpoint
saved by `ModelCheckpoint` (currently every 5000 steps + best + last). Zero new code — already
configured, just change `log_model: true` → `log_model: "all"`.

**Download** (eval): Checkpoints are resolved lazily via a custom OmegaConf resolver. The
experiment config pins a W&B artifact reference using resolver syntax:

```yaml
# configs/experiment/surge/flow_simple.yaml
ckpt_path: ${wandb:synth-permutations/model-x118ylu9:latest}
```

The resolver is registered in `src/utils/utils.py` alongside existing `mul` and `div` resolvers
(already called via `register_resolvers()` at startup):

```python
def _wandb_resolver(artifact_ref: str) -> str:
    """Resolve a W&B artifact reference to a local path.

    Usage in config: ${wandb:synth-permutations/model-x118ylu9:latest}
    """
    cache_dir = Path(os.environ["PROJECT_ROOT"]) / ".cache" / "checkpoints"
    safe_name = artifact_ref.replace("/", "_").replace(":", "_")
    cached = cache_dir / safe_name
    if cached.exists():
        return str(cached / "model.ckpt")
    cached.mkdir(parents=True, exist_ok=True)
    import wandb
    artifact = wandb.Api().artifact(artifact_ref, type="model")
    artifact.download(root=str(cached))
    return str(cached / "model.ckpt")

if not OmegaConf.has_resolver("wandb"):
    OmegaConf.register_new_resolver("wandb", _wandb_resolver)
```

- Resolution is lazy — triggers on first access of `cfg.ckpt_path`
- If cached copy exists → no-op (no W&B API call)
- Zero changes to `src/eval.py` or `src/train.py`

See [§7.2](#72-checkpoint-resolution) for the full resolution behavior.

### 6.3 Eval Artifact Upload

After metrics, optionally upload all eval outputs to R2:

```bash
make upload-eval
# rclone sync logs/eval/surge_simple/x118ylu9/surge_simple/ r2:synth-data/eval/surge_simple/x118ylu9/surge_simple/ --checksum
```

Not automatic — explicit `make` target. Toggle via Hydra config or CLI flag.

**Browsing eval results in R2:**

```bash
# All models trained on surge_simple
rclone ls r2:synth-data/eval/surge_simple/

# All evals of training run x118ylu9
rclone ls r2:synth-data/eval/surge_simple/x118ylu9/

# Cross-dataset eval: model trained on surge_simple, evaluated on nsynth
rclone ls r2:synth-data/eval/surge_simple/x118ylu9/nsynth/
```

### 6.4 W&B Eval Lineage

W&B tracks the full provenance chain via artifact lineage. Each eval creates a lightweight
W&B run that declares its inputs (model checkpoint + dataset) and logs summary metrics:

```python
# Created automatically by the eval pipeline
eval_run = wandb.init(
    project="synth-permutations",
    job_type="eval",
    config={
        "training_run_id": "x118ylu9",
        "train_data": "surge_simple",
        "eval_data": "surge_simple",
    },
)

# Declare input artifacts — W&B builds the lineage graph
model_artifact = eval_run.use_artifact("model-x118ylu9:latest")
dataset_artifact = eval_run.use_artifact("dataset-surge_simple-480k:latest")

# Log summary metrics
eval_run.log({"mss": 0.42, "wmfcc": 0.31, "sot": 0.18, "rms": 0.94})

# Reference R2 location for bulk artifacts
eval_artifact = wandb.Artifact(
    "eval-surge_simple-x118ylu9-surge_simple", type="eval"
)
eval_artifact.add_reference(
    "r2://synth-data/eval/surge_simple/x118ylu9/surge_simple/"
)
eval_run.log_artifact(eval_artifact)
eval_run.finish()
```

This creates a lineage graph in W&B:

```
dataset-surge_simple-480k:v2 ──→ training run x118ylu9 ──→ model-x118ylu9:latest
                                                                    │
dataset-surge_simple-480k:v2 ──→ eval run (job_type=eval) ◄────────┘
                                        │
                                        └──→ eval-surge_simple-x118ylu9-surge_simple (R2 ref)
```

## 7. Design Decisions

### 7.1 Headless Rendering

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

### 7.2 Checkpoint Resolution

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

| Pattern                | Where specified                             | Use case                                                               | Example                                                        |
| ---------------------- | ------------------------------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------- |
| CLI arg                | Command line                                | Ad-hoc eval of a new/local checkpoint                                  | `python src/eval.py ckpt_path=./my-ckpt.ckpt`                  |
| Experiment config      | `configs/experiment/surge/flow_simple.yaml` | Reproducible eval of a known model — checkpoint pinned as W&B artifact | `ckpt_path: ${wandb:synth-permutations/model-x118ylu9:latest}` |
| `null` (training only) | `configs/train.yaml`                        | Start training fresh                                                   | Already works                                                  |

**Resolution order** (Hydra's standard override precedence):

1. CLI override → highest priority
2. Experiment config → pinned per model variant
3. Base config (`eval.yaml: ???`) → forces one of the above

**Resolver behavior:**

OmegaConf resolves `${wandb:...}` lazily — the W&B API is only called when `cfg.ckpt_path` is
first accessed. The resolver (registered in `src/utils/utils.py` via `register_resolvers()`)
checks `$PROJECT_ROOT/.cache/checkpoints/` for a cached copy, downloads via `wandb.Api().artifact().download()`
if not found, and returns the local path. No changes to `src/eval.py` or `src/train.py` — Hydra
hands Lightning a resolved local path transparently.

**What this replaces:**

- `get-ckpt-from-wandb.sh` — replaced by `${wandb:...}` resolver in experiment configs (same data source, cleaner interface)
- Per-script W&B run IDs — replaced by pinned W&B artifact references in experiment YAML
- 19 SGE scripts — deprecated, not maintained

#### Proposed design outcomes

| Config value                                                                       | What happens                                                                       | Portable? | Reproducible?                 |
| ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | --------- | ----------------------------- |
| `ckpt_path: ???` (base eval.yaml)                                                  | Hydra errors — forces user to specify                                              | —         | —                             |
| `ckpt_path: ./local/best.ckpt` (CLI)                                               | Uses local file directly                                                           | No        | No (path is machine-specific) |
| `ckpt_path: ${wandb:synth-permutations/model-x118ylu9:latest}` (experiment config) | OmegaConf resolves lazily → downloads from W&B, caches locally, returns local path | Yes       | Yes (artifact ref is stable)  |
| `ckpt_path: ${wandb:synth-permutations/model-x118ylu9:latest}` (CLI override)      | Same as above, but ad-hoc                                                          | Yes       | No (not pinned in config)     |
| `ckpt_path: null` (train.yaml)                                                     | Start training from scratch                                                        | Yes       | Yes                           |
| `ckpt_path: ${wandb:synth-permutations/model-x118ylu9:latest}` (training resume)   | Resolves lazily → downloads latest checkpoint, resumes optimizer/epoch state       | Yes       | Yes                           |

**Decision:** `ckpt_path` is not in `.env` (not a secret, not machine infrastructure). It is either a required CLI arg (ad-hoc) or pinned in an experiment config (reproducible). The `${wandb:...}` OmegaConf resolver makes pinned values portable across machines — resolution is lazy and cached. Checkpoints are stored in W&B (Teams plan, $50/mo) — see [§10](#10-alternatives-considered) for the full cost/benefit analysis vs R2.

### 7.3 Makefile as CLI Interface

**Decision:** All eval operations are `make` targets — consistent with the existing `make test`, `make format` pattern.

| Target             | Maps to                                       |
| ------------------ | --------------------------------------------- |
| `make eval`        | `make predict render metrics` (full pipeline) |
| `make predict`     | `python src/eval.py mode=predict ...`         |
| `make render`      | `./renderscript.sh` or direct Python on macOS |
| `make metrics`     | `python scripts/compute_audio_metrics.py ...` |
| `make docker-eval` | `docker run ... make eval`                    |
| `make upload-eval` | `rclone sync ... --checksum`                  |

**Rationale:** Make targets are discoverable (`make help`), composable, and already the project convention. They hide environment-specific complexity (display detection, R2 paths) behind a consistent interface.

### 7.4 Storage Responsibility Split

Each system handles what it's best at:

| System                                 | What it stores                                                                                                                 | Why                                                       |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------- |
| **W&B**                                | Training metrics, checkpoints (`log_model="all"`), eval summary metrics, artifact lineage                                      | UI for browsing/comparing, lineage graphs, model registry |
| **R2**                                 | Datasets (generated shards, train/val/test splits), eval bulk artifacts (predictions, audio, spectrograms, per-sample metrics) | Too large for W&B, cheaper per GB, fast rclone egress     |
| **Hydra config** (`config.yaml` in R2) | Full frozen config at eval time — every parameter, override, and version                                                       | Exact reproducibility without querying W&B                |

**Provenance is recorded in three places:**

- **R2 path** → human-readable: `eval/{train_data}/{run_id}/{eval_data}/` tells you what happened at a glance
- **W&B lineage** → programmatic: `use_artifact()` connects dataset → model → eval with exact versions
- **Hydra config** → complete: every parameter frozen, reproducible without any external system

### 7.5 Current vs Proposed: Full Comparison

This section consolidates every configuration and environment behavior change in one place.

#### Current behavior (as-is)

| Concern                   | Current mechanism                                                   | Where defined                              | Portable? | Problem                                            |
| ------------------------- | ------------------------------------------------------------------- | ------------------------------------------ | --------- | -------------------------------------------------- |
| **Dataset path**          | Hardcoded `/data/scratch/acw585/surge-simple/`                      | `configs/data/surge_simple.yaml`           | No        | Only works on university cluster                   |
| **Checkpoint resolution** | `get-ckpt-from-wandb.sh` searches local `logs/train/` by W&B run ID | `jobs/predict/*.sh` (19 scripts)           | No        | Requires training logs on same machine             |
| **Checkpoint path**       | `ckpt_path: ???` in eval, resolved by shell script to local path    | `configs/eval.yaml` + shell                | No        | Local filesystem dependency                        |
| **R2 dataset access**     | Not supported                                                       | —                                          | —         | Must manually copy data to machine                 |
| **Checkpoint access**     | `get-ckpt-from-wandb.sh` (local filesystem search by W&B run ID)    | `jobs/predict/*.sh`                        | No        | Only works on the machine where training happened  |
| **Checkpoint upload**     | W&B `log_model: true` — uploads best checkpoint only                | `configs/logger/wandb.yaml`                | Partially | No periodic upload, crash loses intermediate ckpts |
| **Credentials**           | No `.env` pattern for R2                                            | —                                          | —         | No standardized credential management              |
| **Display handling**      | `renderscript.sh` assumes Linux + Xvfb                              | `renderscript.sh`                          | No        | Fails on macOS (no Xvfb needed), no auto-detect    |
| **Log directory**         | `${paths.root_dir}/logs/` via `PROJECT_ROOT`                        | `configs/paths/default.yaml`               | Yes       | Already works                                      |
| **Predict output**        | `${paths.output_dir}/predictions`                                   | `configs/callbacks/prediction_writer.yaml` | Yes       | Already works                                      |
| **W&B entity**            | Hardcoded `entity: "benhayes"`                                      | `configs/logger/wandb.yaml`                | No        | Wrong for other users                              |
| **SGE scripts**           | 19 near-identical scripts, one per model                            | `jobs/predict/*.sh`                        | No        | Copy-paste errors, cluster-only                    |
| **Eval CLI**              | Raw `python src/eval.py ...` with many args                         | Shell scripts                              | No        | No `make` targets, hard to discover                |

#### Proposed behavior (to-be)

| Concern                      | Proposed mechanism                                                                                    | Where defined                                 | Portable? | Change from current                         |
| ---------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------- | --------- | ------------------------------------------- |
| **Dataset path**             | `dataset_root: ${paths.data_dir}/surge-simple` (paths convention)                                     | `configs/data/surge_simple.yaml`              | Yes       | Hardcoded → paths convention                |
| **Dataset path override**    | CLI: `data.dataset_root=/cluster/path/`                                                               | Command line                                  | Yes       | Implicit → explicit                         |
| **Checkpoint resolution**    | `ckpt_path: ???` (base), pinned in experiment configs                                                 | `configs/eval.yaml` + `configs/experiment/`   | Yes       | Shell script → Hydra config                 |
| **Checkpoint: ad-hoc**       | CLI: `ckpt_path=./local/best.ckpt`                                                                    | Command line                                  | No        | Same as today but without shell wrapper     |
| **Checkpoint: reproducible** | `ckpt_path: ${wandb:synth-permutations/model-x118ylu9:latest}` in experiment config                   | `configs/experiment/surge/flow_simple.yaml`   | Yes       | **New** — portable, pinned                  |
| **R2 dataset access**        | `data.r2_path=r2:synth-data/...` triggers auto-download in `prepare_data()`                           | CLI or experiment config (no default)         | Yes       | **New** — explicit opt-in                   |
| **Checkpoint download**      | `${wandb:...}` OmegaConf resolver → lazy W&B artifact download to `$PROJECT_ROOT/.cache/checkpoints/` | `src/utils/utils.py` (`register_resolvers()`) | Yes       | **New** — replaces `get-ckpt-from-wandb.sh` |
| **Checkpoint upload**        | W&B `log_model="all"` — uploads every saved checkpoint automatically                                  | `configs/logger/wandb.yaml`                   | Yes       | Config change only — `true` → `"all"`       |
| **Credentials**              | `.env` for R2 + W&B secrets only                                                                      | `.env` / `.env.example`                       | Yes       | **New** — secrets only, no paths            |
| **Display handling**         | Auto-detect: macOS native / Linux Xvfb / Docker baked                                                 | `renderscript.sh`                             | Yes       | Linux-only → cross-platform                 |
| **Log directory**            | `${paths.root_dir}/logs/` (unchanged)                                                                 | `configs/paths/default.yaml`                  | Yes       | No change                                   |
| **Predict output**           | `${paths.output_dir}/predictions` (unchanged)                                                         | `configs/callbacks/prediction_writer.yaml`    | Yes       | No change                                   |
| **W&B entity**               | Configurable via env or CLI                                                                           | `configs/logger/wandb.yaml`                   | Yes       | Hardcoded → configurable                    |
| **SGE scripts**              | Deprecated — left as-is, not maintained                                                               | `jobs/predict/*.sh`                           | No        | Active → deprecated                         |
| **R2 eval artifact upload**  | `make upload-eval` → `r2:synth-data/eval/{train_data}/{run_id}/{eval_data}/`                          | `Makefile`                                    | Yes       | **New** — path encodes full provenance      |
| **W&B eval lineage**         | `use_artifact()` connects dataset → model → eval; R2 reference artifact for bulk files                | `scripts/compute_audio_metrics.py`            | Yes       | **New** — programmatic provenance chain     |
| **Eval CLI**                 | `make predict`, `make render`, `make metrics`                                                         | `Makefile`                                    | Yes       | **New** — discoverable, consistent          |

#### What changes, what stays

| Category       | Items that change                                                                                                                                                           | Items that stay |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| **Removed**    | Hardcoded cluster paths, `get-ckpt-from-wandb.sh` shell script, SGE as supported platform                                                                                   |                 |
| **Deprecated** | 19 SGE scripts (left in repo, no maintenance)                                                                                                                               |                 |
| **New**        | `${wandb:...}` OmegaConf resolver, `data.r2_path` opt-in, `make` targets, cross-platform display, `.env` for secrets, W&B Teams plan, W&B eval lineage, R2 provenance paths |                 |
| **Modified**   | `dataset_root` (hardcoded → paths convention), `renderscript.sh` (Linux-only → auto-detect), W&B entity (hardcoded → configurable), `log_model` (`true` → `"all"`)          |                 |
| **Unchanged**  | `ckpt_path: ???` in eval.yaml, `ckpt_path: null` in train.yaml, `log_dir`, `output_dir`, prediction writer, W&B metric logging, CSV logger, `ModelCheckpoint` save cadence  |                 |

#### Diff analysis

**1. `.env` scope**

|                      | Current              | Proposed                                                             |
| -------------------- | -------------------- | -------------------------------------------------------------------- |
| **What's in `.env`** | Nothing standardized | R2 credentials + `WANDB_API_KEY`                                     |
| **Paths**            | Hardcoded in YAML    | Hydra defaults + CLI overrides                                       |
| **Risk eliminated**  | —                    | Invisible state: can't read YAML + `.env` and know what happens      |
| **Trade-off**        | —                    | Cluster users must pass CLI overrides instead of setting one env var |

**2. Checkpoint resolution (§7.2)**

|                            | Current                                     | Proposed                                                                                        |
| -------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Eval checkpoint**        | Shell script finds local file by W&B run ID | Pinned `${wandb:...}` resolver in experiment config or CLI arg                                  |
| **Training checkpoint**    | `ckpt_path: null` (start fresh)             | Same — no change                                                                                |
| **Training resume**        | `ckpt_path=/local/path/last.ckpt`           | `ckpt_path=${wandb:synth-permutations/model-x118ylu9:latest}` (portable)                        |
| **Upload during training** | W&B `log_model: true` (best only)           | W&B `log_model="all"` (every saved checkpoint — crash resilient)                                |
| **Risk eliminated**        | —                                           | "Checkpoint is on the cluster" — W&B artifacts available everywhere                             |
| **Trade-off**              | —                                           | W&B Teams at $50/mo; storage burns faster with `"all"` (see [§10](#10-alternatives-considered)) |

**3. Dataset access (§6.3)**

|                     | Current                    | Proposed                                                                 |
| ------------------- | -------------------------- | ------------------------------------------------------------------------ |
| **Local data**      | Hardcoded path, must exist | `${paths.data_dir}/surge-simple` (existing convention), override via CLI |
| **Remote data**     | Not supported              | `r2_path` opt-in triggers auto-download                                  |
| **Risk eliminated** | —                          | "Data is on the cluster" — R2 makes it available everywhere              |
| **Trade-off**       | —                          | First download of a 100GB dataset takes time; cached after that          |

**4. Display handling (§7.1)**

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

## 8. Phase Plan

> This section follows the Epic → Phase → Task hierarchy defined in
> [`github-taxonomy.md`](github-taxonomy.md) §3.
> Issues follow the standard lifecycle ([`github-taxonomy.md`](github-taxonomy.md) §11): Todo → In Progress → Done.
> Project fields required for all tasks: Priority, Start Date, Target Date.
> Per-task priority tracked via the project Priority field ([`github-taxonomy.md`](github-taxonomy.md) §5).
>
> For issue tracking structure see [`github-taxonomy.md`](github-taxonomy.md).

### Issue Mapping

| Issue | Type  | Description                    | Parent |
| ----- | ----- | ------------------------------ | ------ |
| #98   | Epic  | Evaluation pipeline            | —      |
| #99   | Epic  | Storage / R2 integration       | —      |
| #137  | Phase | Phase 1: Portable eval         | #98    |
| #138  | Phase | Phase 2: Storage integration   | #99    |
| #139  | Phase | Phase 3: W&B integration       | #98    |
| #140  | Phase | Phase 4: CI & containerization | #98    |
| #141  | Phase | Phase 5: Documentation         | #98    |
| #94   | Task  | Config cleanup                 | #137   |
| #85   | Task  | Portable predict               | #137   |
| #86   | Task  | Portable render                | #137   |
| #87   | Task  | Portable metrics               | #137   |
| #90   | Task  | rclone wrapper                 | #138   |
| #91   | Task  | R2 dataset download            | #138   |
| #93   | Task  | R2 artifact upload             | #138   |
| #92   | Task  | R2 checkpoint sync             | #138   |
| #128  | Task  | W&B checkpoint resolver        | #139   |
| #96   | Task  | W&B metrics logging            | #139   |
| #88   | Task  | Docker eval environment        | #140   |
| #89   | Task  | E2E eval CI                    | #140   |
| #97   | Task  | Eval runbook                   | #141   |

### Per-Phase Metadata

| Phase | Issue | Label(s)                      | Milestone           | Epic |
| ----- | ----- | ----------------------------- | ------------------- | ---- |
| 1     | #137  | `evaluation`                  | `evaluation v1.0.0` | #98  |
| 2     | #138  | `evaluation`, `storage`       | `storage v1.0.0`    | #99  |
| 3     | #139  | `evaluation`                  | `evaluation v1.0.0` | #98  |
| 4     | #140  | `evaluation`, `ci-automation` | `evaluation v1.0.0` | #98  |
| 5     | #141  | `evaluation`                  | `evaluation v1.0.0` | #98  |

### Completion Tracking

When tasks are completed, update this section using the taxonomy's design doc linkage pattern:

```
### Task 1.1: Config Cleanup (#94) ✅ — Completed in PR #XXX
```

### Branch Strategy

```
main ──●──────────────●──────────────●──────────────●──────────────●──→
       │              │              │              │              │
       Phase 1        Phase 2        Phase 3        Phase 4        Phase 5
```

### Estimated Change Size

| Task | Area                     | Already works                       | Actual change                      | Lines |
| ---- | ------------------------ | ----------------------------------- | ---------------------------------- | ----- |
| 1.1  | Data configs             | mnist.yaml uses `${paths.data_dir}` | Change 2 surge configs to match    | ~2    |
| 1.2  | Predict                  | src/eval.py works, `ckpt_path: ???` | Add Makefile target                | ~10   |
| 1.3  | Rendering                | Xvfb auto-launch in renderscript.sh | Add macOS `$OSTYPE` conditional    | ~5    |
| 1.4  | Metrics                  | All 4 metrics working, portable     | Add Makefile target                | ~5    |
| 2.1  | rclone wrapper           | —                                   | New utility `src/data/rclone.py`   | ~40   |
| 2.2  | `prepare_data()` R2 sync | —                                   | Override in SurgeDataModule        | ~15   |
| 3.1  | W&B resolver             | `register_resolvers()` exists       | Add wandb resolver (~15 lines)     | ~15   |
| 3.1  | W&B config               | `log_model: true`                   | Change to `"all"`                  | 1     |
| —    | Makefile targets         | help, test, format, train           | Add predict, render, metrics, etc. | ~40   |
| —    | Tests                    | conftest.py fixtures exist          | New test files + fixtures          | ~400  |

**Branch:** `dev/eval-pipeline` off `main`
**Priorities:** TDD first, small commits, always-green CI.

______________________________________________________________________

### Phase 1: Portable Evaluation Pipeline (#137, Epic #98)

> **Label:** `evaluation` · **Milestone:** `evaluation v1.0.0`

#### Task 1.1: Config Cleanup (#94)

**Goal:** Replace all cluster-specific paths in committed configs with sensible Hydra defaults.

**Files to modify:**

- `configs/data/surge_simple.yaml` — `dataset_root` → `${paths.data_dir}/surge-simple` (matches `mnist.yaml` convention)
- `configs/data/surge_mini.yaml` — same pattern
- `configs/data/surge_simple_onehot.yaml` — same (if exists)
- `.env.example` — R2 credentials and `WANDB_API_KEY` only (no path vars)

**Tests:**

- `test_no_hardcoded_paths_in_configs` — grep committed YAML for `/data/scratch`
- `test_configs_have_sensible_defaults` — load each data config, verify `dataset_root` is a relative path

**Note:** The 19 SGE scripts in `jobs/predict/` are left as-is (deprecated, not consolidated — see [§7.5](#75-current-vs-proposed-full-comparison) SGE deprecation).

#### Task 1.2: Portable Predict (#85)

**Goal:** `make predict` works on local machines with portable Hydra config.

**Files to modify:**

- `src/eval.py` — ensure `mode=predict` works without cluster deps
- `Makefile` — add `predict` target

**Files to create:**

- `tests/test_eval_predict.py` — fixture-based predict test (`@pytest.mark.slow`)

**Key behaviors:**

- `dataset_root` has a sensible Hydra default; override via CLI when needed
- `ckpt_path` resolved per [§7.2](#72-checkpoint-resolution) — CLI arg or experiment config, supports `${wandb:...}` resolver
- `paths.log_dir` keeps the existing default (`${paths.root_dir}/logs/`)
- Fails fast with clear error if dataset not found

#### Task 1.3: Portable Render (#86)

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

#### Task 1.4: Portable Metrics (#87)

**Goal:** `make metrics` works with pinned dependencies and clean output schema.

**Files to modify:**

- `scripts/compute_audio_metrics.py` — pin dep versions
- `Makefile` — add `metrics` target

**Files to create:**

- `tests/test_metrics.py` — test with fixture `.wav` files, validate CSV schema

**Key behaviors:**

- Output CSV: per-sample metrics indexed by directory name, aggregated means/stds
- `ProcessPoolExecutor` parallelism preserved
- JTFS and f0 code stays as-is — track reactivation as a separate issue (out of scope)

______________________________________________________________________

### Phase 2: Storage Integration (#138, Epic #99)

> **Labels:** `evaluation`, `storage` · **Milestone:** `storage v1.0.0`

#### Task 2.1: rclone Wrapper (#90)

**Goal:** Shared `rclone_sync()` utility with `--checksum` enforcement.

**Files to create:**

- `src/data/rclone.py` — `rclone_sync()`, `rclone_ls()`, `rclone_copyto()`
- `tests/test_rclone.py` — mock subprocess, verify flags

**Key behaviors:**

- All operations include `--checksum`
- R2 config from env vars (`RCLONE_CONFIG_R2_*`)
- Raises `subprocess.CalledProcessError` on failure
- Dry-run mode for testing (`--dry-run` flag passthrough)

#### Task 2.2: R2 Dataset Download (#91)

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

#### Task 2.3: R2 Eval Artifact Upload (#93)

**Goal:** `make upload-eval` pushes predictions + audio + metrics to R2.

**Files to modify:**

- `Makefile` — add `upload-eval` target

**Files to create:**

- `scripts/upload_eval_artifacts.py` — rclone sync wrapper for eval outputs
- `tests/test_upload_eval.py` — mock rclone, verify R2 paths

#### Task 2.4: R2 Checkpoint Sync (#92)

**Goal:** Sync checkpoints to R2 as a secondary backup alongside W&B.

**Depends on:** #90 (rclone wrapper)

______________________________________________________________________

### Phase 3: W&B Integration (#139, Epic #98)

> **Label:** `evaluation` · **Milestone:** `evaluation v1.0.0`

#### Task 3.1: W&B Checkpoint Config + Resolver (#128)

**Goal:** Enable crash-resilient checkpoint upload via W&B and lazy resolution via OmegaConf.

**Files to modify:**

- `configs/logger/wandb.yaml` — change `log_model: true` → `log_model: "all"`
- `src/utils/utils.py` — add `_wandb_resolver` to `register_resolvers()` (~15 lines)

**Files to create:**

- `tests/test_wandb_resolver.py` — mock W&B API, verify download + cache logic

**Key behaviors:**

- `log_model="all"` uploads every saved checkpoint (every 5000 steps + best + last)
- `${wandb:...}` OmegaConf resolver handles artifact download + cache
- Cache dir: `$PROJECT_ROOT/.cache/checkpoints/` (gitignored)
- Zero new modules — resolver lives in existing `register_resolvers()`

#### Task 3.2: W&B Metrics Logging (#96)

**Goal:** Optionally log metrics to W&B for cross-run comparison.

**Files to modify:**

- `scripts/compute_audio_metrics.py` — add `--wandb-run` flag

**Files to create:**

- `tests/test_metrics_wandb.py` — mock wandb, verify log calls

______________________________________________________________________

### Phase 4: CI & Containerization (#140, Epic #98)

> **Labels:** `evaluation`, `ci-automation` · **Milestone:** `evaluation v1.0.0`

#### Task 4.1: Docker Eval Environment (#88)

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

#### Task 4.2: E2E Eval CI (#89)

**Goal:** GitHub Actions workflow runs predict → render → metrics on a small fixture.

**Files to create:**

- `.github/workflows/eval-ci.yml` — matrix: Ubuntu (Xvfb)
- `tests/test_eval_e2e.py` — fixture-based integration test (`@pytest.mark.slow`)
- `tests/fixtures/eval/` — checked-in test fixtures:
  - `tiny.ckpt` (~1 MB) — checkpoint trained for 2 epochs on a handful of samples
  - `fixture-shard.h5` (~5 MB) — small HDF5 shard with 10-50 samples (enough for one predict batch)
  - `audio/sample_0/{pred,target}.wav` (~1 MB) — pre-rendered audio for metrics-only testing without VST plugin

**Key behaviors:**

- Runs on PR (if `src/eval.py`, `scripts/`, or `configs/` changed)
- Uses checked-in fixtures — no R2 credentials, no network dependency, no secrets in CI
- Validates: predictions exist, audio renders, metrics CSV has expected schema
- If fixtures grow past ~10 MB, migrate to Git LFS

______________________________________________________________________

### Phase 5: Documentation (#141, Epic #98)

> **Label:** `evaluation` · **Milestone:** `evaluation v1.0.0`

#### Task 5.1: Eval Runbook (#97)

**Goal:** Document how to run the full eval pipeline locally and in Docker.

**Files to create:**

- `docs/eval-runbook.md` — setup, credentials, make targets, Docker, troubleshooting

______________________________________________________________________

### Standalone Tasks (not in phase hierarchy)

| Issue | Title           | Type | Labels                      | Milestone           | Notes                                       |
| ----- | --------------- | ---- | --------------------------- | ------------------- | ------------------------------------------- |
| #95   | Consolidate SGE | Task | `evaluation`, `code-health` | `evaluation v1.0.0` | SGE deprecation is a design decision (§7.5) |

______________________________________________________________________

## 9. Dependency Overview

> Dependencies are implemented using GitHub's native "Blocked by / Blocking"
> relationships (issue sidebar → Relationships). The canonical dependency DAG
> lives in GitHub; this section documents the critical path and parallel
> execution windows for planning.

### Blocking Matrix

| Issue | Title               | Blocked by    | Blocking                |
| ----- | ------------------- | ------------- | ----------------------- |
| #94   | Config cleanup      | —             | #85, #91                |
| #85   | Portable predict    | #94           | #88, #89, #97           |
| #86   | Portable render     | —             | #88, #89, #97           |
| #87   | Portable metrics    | —             | #88, #89, #93, #96, #97 |
| #90   | rclone wrapper      | —             | #91, #93                |
| #88   | Docker eval         | #85, #86, #87 | #89                     |
| #89   | E2E CI              | #85–88        | —                       |
| #91   | R2 dataset download | #90, #94      | —                       |
| #93   | R2 artifact upload  | #90, #87      | —                       |
| #92   | R2 checkpoint sync  | #90           | —                       |
| #95   | Consolidate SGE     | —             | —                       |
| #96   | W&B metrics logging | #87           | —                       |
| #97   | Eval runbook        | #85, #86, #87 | —                       |
| #128  | W&B resolver        | #94           | #88                     |

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
├── Phase 1: Portable Eval ──────────┤           │
│   (#94, #85, #86, #87)             │           │
│                  ├── Phase 2: Storage ─────────┤│
│                  │   (#90, #91, #93)            ││
│                  ├── Phase 3: W&B ─────────────┤│
│                  │   (#128, #96)                ││
│                  │              ├── Phase 4: CI + Docker ──┤
│                  │              │   (#88, #89)              │
│                  │              │              ├── Phase 5 ─┤
│                  │              │              │   (#97)    │
│                                                      milestone
```

## 10. Alternatives Considered

### Quick rejections

| Alternative                              | Why rejected                                                                                             |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **DVC for data versioning**              | Adds a dependency for a problem rclone already solves. DVC's git integration is overkill for 2 datasets. |
| **boto3/S3 SDK instead of rclone**       | Data pipeline already standardized on rclone. Consistency > marginal testability gain.                   |
| **Snakemake/Nextflow for eval pipeline** | Massive dependency for a 3-stage linear pipeline. `make` is sufficient.                                  |
| **Automatic stage chaining**             | At 1-2 evals/week, the cognitive overhead of "what ran automatically?" exceeds the convenience.          |
| **Per-model Docker images**              | One eval image with model as a parameter. Multiple images are unnecessary build complexity.              |
| **Config-driven display detection**      | Auto-detection is strictly better — no env var to forget, no "works on my machine."                      |

### Checkpoint storage: W&B artifacts vs R2 (detailed analysis)

This was the most significant design decision in this doc. We evaluated three approaches
for checkpoint storage and chose W&B Teams.

#### The options

| Approach                                | Upload mechanism                                          | Download mechanism                                                      | New code                  |
| --------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------- |
| **A: W&B `log_model=true`** (current)   | Lightning auto-uploads best checkpoint                    | `get-ckpt-from-wandb.sh` (local filesystem search)                      | None                      |
| **B: R2 via custom callback**           | `R2CheckpointUploader` fires every `ModelCheckpoint` save | `r2:` prefix → rclone download to cache                                 | ~200 lines + tests        |
| **C: W&B `log_model="all"`** (selected) | Lightning auto-uploads every saved checkpoint             | `${wandb:...}` OmegaConf resolver → `wandb.Api().artifact().download()` | ~15 lines (resolver only) |

#### Cost comparison

| Concern            | W&B Free        | W&B Teams ($50/mo)   | R2 only            | W&B Teams + R2 (selected)          |
| ------------------ | --------------- | -------------------- | ------------------ | ---------------------------------- |
| Tracking hours     | 250 total       | Unlimited            | N/A                | Unlimited                          |
| Checkpoint storage | 100 GB shared   | 100 GB + $0.03/GB    | ~$0.015/GB         | Checkpoints in W&B, datasets in R2 |
| Dataset storage    | 100 GB shared   | 100 GB + $0.03/GB    | ~$0.015/GB         | R2 ($0.015/GB)                     |
| Egress             | Free (slow API) | Free (slow API)      | Free (fast rclone) | W&B for ckpts, rclone for data     |
| Annual cost (est.) | $0              | $600                 | ~$24               | ~$624                              |
| UI lockout risk    | Yes (>100 GB)   | No (overage billing) | N/A                | No                                 |

#### W&B free tier limitations

The free tier was insufficient for this project:

| Constraint                                       | Free tier                                                  | Impact                                                               |
| ------------------------------------------------ | ---------------------------------------------------------- | -------------------------------------------------------------------- |
| **250 tracking hours** (cumulative, not monthly) | At 12 hrs/run, ~20 runs total before W&B stops working     | Hard cap on total training tracked by W&B                            |
| **100 GB storage** (artifacts + files combined)  | One dataset (~50 GB) eats half the limit                   | Cannot store datasets; checkpoints fill up after ~200 runs           |
| **UI lockout on exceed**                         | Exceeding 100 GB locks you out of all projects, all charts | Entire W&B instance becomes unusable, not just the offending project |
| **Tracking hours are wall-clock**                | `WANDB_MODE=offline` defers but doesn't avoid the cost     | Syncing a 12-hour offline run later still burns 12 hours             |

Offline mode was considered as a workaround: train offline, sync selectively. But synced runs
still burn tracking hours based on original duration, so it only defers the cap — doesn't avoid it.

#### Why R2 for checkpoints was rejected

R2 checkpoints (Option B) would have cost ~$24/year vs $600/year for W&B Teams. The analysis:

| Concern              | R2 checkpoints                         | W&B checkpoints (selected)                       | Winner                           |
| -------------------- | -------------------------------------- | ------------------------------------------------ | -------------------------------- |
| **Upload**           | Custom callback (~200 lines)           | Config change: `log_model: "all"`                | W&B — zero new code              |
| **Download**         | `rclone copyto` (fast, free egress)    | `wandb.Api().artifact().download()` (slower)     | R2 — faster for large files      |
| **Crash resilience** | Callback uploads every 5000 steps      | `log_model="all"` uploads every saved checkpoint | Tie — same cadence               |
| **Browse/compare**   | `rclone ls` — no UI                    | W&B model registry, lineage graphs, side-by-side | W&B — significantly better       |
| **Cost**             | ~$2/mo                                 | $50/mo                                           | R2 — 25x cheaper                 |
| **New code**         | ~200 lines (callback, resolver, cache) | ~15 lines (OmegaConf resolver only)              | W&B — less to build and maintain |
| **Vendor lock-in**   | None — just files in S3                | W&B API dependency                               | R2 — more portable               |

**Decision: W&B Teams.** The $50/mo buys unlimited tracking hours (the real constraint),
a checkpoint UI that's genuinely useful for research (model registry, lineage, comparison),
and avoids ~200 lines of custom checkpoint infrastructure. The download speed trade-off is
acceptable — checkpoint downloads happen once per eval run, not in a hot loop.

R2 remains the right choice for **datasets** (too large for W&B storage, 2x cheaper per GB)
and **eval artifacts** (audio files, prediction tensors — no W&B UI benefit).

#### What each storage backend is responsible for

| Backend             | What it stores                                                                                        | Why                                                                                  |
| ------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **W&B**             | Checkpoints, training metrics, run configs, model registry                                            | UI for browsing/comparing, unlimited hours on Teams, already integrated              |
| **R2**              | Datasets (generated shards, train/val/test splits), eval artifacts (audio, predictions, metrics CSVs) | Too large for W&B, cheaper per GB, fast rclone egress, data pipeline already uses R2 |
| **Local** (`logs/`) | Hydra output dirs, TensorBoard logs, CSV metrics, checkpoints (before W&B upload)                     | Working directory, ephemeral                                                         |

## 11. Open Questions & Risks

| #   | Question / Risk                                                                             | Impact                                              | Status                   |
| --- | ------------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------ |
| 1   | **VST plugin licensing on CI runners** — can we legally run Surge XT in GitHub Actions?     | E2E CI may need a stub or fixture-based approach    | Open                     |
| 2   | **macOS pedalboard + Surge XT compatibility** — does the VST3 plugin load on Apple Silicon? | Blocks macOS render stage                           | Needs testing            |
| 3   | **W&B artifact download speed** — best.ckpt may be 500MB+; W&B API is slower than rclone    | Mitigated by caching; accepted trade-off for W&B UI | Accepted                 |
| 4   | **Metrics reproducibility across platforms** — float differences in spectral computations   | May cause CI flakiness with tight tolerances        | Use relative tolerances  |
| 5   | **Xvfb availability in Docker base image** — may need to install in Dockerfile              | Low risk, well-documented                           | Resolved by Docker stage |
| 6   | **rclone version skew** — different rclone versions on dev machines vs CI                   | Pin rclone version in Dockerfile and CI workflow    | Open                     |

## 12. Out of Scope

- **Automated hyperparameter sweeps** — eval runs are manually triggered
- **Multi-GPU distributed eval** — single-GPU is sufficient at current dataset sizes
- **Audio listening tests / perceptual evaluation** — future work, requires different tooling
- **Real-time inference server** — batch eval only
- **Custom metric development** — existing 4 metrics are fixed for v1.0.0
- **Training pipeline changes** — this doc covers eval and R2 only
- **Data pipeline modifications** — covered by [data pipeline design doc](data-pipeline.md) and #74

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
