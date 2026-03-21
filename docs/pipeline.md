# Dataset Pipeline: Architecture and Operations Guide

A newbie-friendly guide to how the dataset generation and training pipeline fits together.

______________________________________________________________________

## Table of Contents

1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [File Inventory](#file-inventory)
4. [Common Surface: run vs ci targets](#common-surface-run-vs-ci-targets)
5. [Plug-and-Play Commands](#plug-and-play-commands)
6. [Metadata and Traceability](#metadata-and-traceability)

______________________________________________________________________

## Overview

The pipeline has two main phases:

| Phase        | What it does                                                                                                                                                                               |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Generate** | Runs a VST synthesizer headlessly to produce audio samples, computes mel spectrograms and normalization stats, writes HDF5 files and `metadata.json`, uploads everything to Cloudflare R2. |
| **Train**    | Downloads the HDF5 dataset from R2, passes it to PyTorch Lightning via `SurgeDataModule`, trains the flow-matching model.                                                                  |

Both phases run inside Docker containers so that Surge XT, CUDA, and Python dependencies are fully encapsulated.

______________________________________________________________________

## Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════╗
║                   synth-setter pipeline                    ║
╚══════════════════════════════════════════════════════════════════╝

━━━ BUILD ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Makefile
    ├── docker-build-dev-live     ──► tinaudio/perm:dev
    │   (mount local source at runtime via USE_LOCAL_WORKSPACE=1)
    └── docker-build-dev-snapshot ──► tinaudio/perm:<base>-dev-snapshot-<sha>
        (self-contained, baked source, for vast.ai / CI)

  docker/ubuntu22_04/Dockerfile (multi-stage)
    ├── arch-vars              arch-specific config (amd64 / arm64)
    ├── builder-base           gcc-12, ninja, flex
    ├── builder-install-surge-from-source   ┐ pick one via
    ├── builder-install-surge-from-prebuilt ┘ BUILD_MODE=source|prebuilt
    ├── synth-setter-src  GitHub tarball download
    ├── python-base            Python 3.10 + venv
    ├── wheels                 pip wheel cache (torch + app deps)
    ├── builder-install-synth-setter-deps   Python + X11 + rclone
    ├── r2-config-base         bakes rclone.conf with R2 credentials ⚠️
    ├── dev-live               ← make docker-build-dev-live
    ├── dev-snapshot           ← make docker-build-dev-snapshot
    ├── prod                   (fully baked production image)
    └── test                   VST load check + pytest -k "not slow"

━━━ TRAIN (MODE=train) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  docker run -e MODE=train -e R2_PREFIX=runs/surge_simple/<sha> ...
    │
    └── scripts/docker_entrypoint.sh
          │
          ├── rclone copy --checksum --progress
          │     r2:<bucket>/<R2_PREFIX> → <OUTPUT_DIR>/
          │
          └── python src/train.py
                data=<surge_simple|surge>           ← Hydra config
                data.dataset_root=<OUTPUT_DIR>      ← override HPC path
                <TRAIN_ARGS>
                  └── SurgeDataModule
                        train.h5, val.h5, test.h5, stats.npz

━━━ DATA FLOW ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Surge XT VST
    → generate_vst_dataset.py (headless X11)
    → {train,val,test}.h5  +  stats.npz
    → metadata.json
    → RcloneUploader (rclone --checksum)
    → R2: runs/<param_spec>/<git_sha>/
    → rclone copy --checksum (download)
    → SurgeDataModule (Hydra: data=surge_simple or data=surge)
    → LightningDataModule → src/train.py
```

______________________________________________________________________

## File Inventory

| File                                   | Role                                                      | Called by                         |
| -------------------------------------- | --------------------------------------------------------- | --------------------------------- |
| `Makefile`                             | Build + run targets                                       | Developer, CI                     |
| `docker/ubuntu22_04/Dockerfile`        | Multi-stage image build                                   | `make docker-build-*`             |
| `scripts/docker_entrypoint.sh`         | Container entry point; dispatches on `MODE`               | `docker run`                      |
| `scripts/run_dataset_pipeline.py`      | Orchestrates generation pipeline (testable, VST-agnostic) | standalone CLI                    |
| `src/data/vst/generate_vst_dataset.py` | Renders audio + spectrograms via VST plugin               | `run_dataset_pipeline.py`         |
| `scripts/run-linux-vst-headless.sh`    | Headless X11/xvfb wrapper for VST                         | `run_dataset_pipeline.py`         |
| `scripts/get_dataset_stats.py`         | Computes mean/std normalization stats                     | `run_dataset_pipeline.py`         |
| `src/data/uploader.py`                 | `RcloneUploader` / `LocalFakeUploader`                    | `run_dataset_pipeline.py`, tests  |
| `src/data/surge_datamodule.py`         | `SurgeDataModule` + `SurgeXTDataset`                      | `src/train.py`                    |
| `configs/data/surge_simple.yaml`       | Hydra data config for 92-param spec                       | `src/train.py` via Hydra          |
| `configs/data/surge.yaml`              | Hydra data config for 189-param spec                      | `src/train.py` via Hydra          |
| `tests/test_run_dataset_pipeline.py`   | Unit tests (mocked subprocess, no VST/R2)                 | pytest, CI, Dockerfile test stage |
| `.github/workflows/data-pipeline.yml`  | CI: unit tests + 3 Docker smoke test jobs                 | GitHub Actions                    |

______________________________________________________________________

## Common Surface: run vs ci targets

A frequent source of confusion is the difference between the `docker-run-*` and `docker-ci-*` Makefile
targets. They use **identical environment variables** — the only difference is the `docker run` flags:

| Target                 | TTY   | GPU          | IDLE_AFTER  | Use when                            |
| ---------------------- | ----- | ------------ | ----------- | ----------------------------------- |
| `docker-run-gpu-train` | `-it` | `--gpus all` | forwarded   | vast.ai or local interactive run    |
| `docker-run-cpu-train` | none  | none         | hardcoded 0 | CI runners, non-interactive scripts |

All four targets also support `$(DOCKER_RUN_FLAGS)`, which is populated when you pass
`USE_LOCAL_WORKSPACE=1`. This mounts your local source tree and overrides the entrypoint so
you can test pipeline changes without rebuilding the image:

```bash
# Mount local source for any run target
make docker-run-generate USE_LOCAL_WORKSPACE=1 TRAIN_SAMPLES=5 VAL_SAMPLES=2 TEST_SAMPLES=2
make docker-run-dev USE_LOCAL_WORKSPACE=1
```

The `docker_entrypoint.sh` script is the **single source of truth** for `MODE` logic — all four targets
above invoke the same entrypoint via the same env vars.

The `.github/workflows/data-pipeline.yml` CI jobs call these Makefile targets (or equivalent raw
`docker run` for steps that need a volume mount for output inspection).

______________________________________________________________________

## Plug-and-Play Commands

Replace all `<...>` placeholders with your actual values before running.

### Prerequisites

You need:

- Docker with BuildKit (Docker Desktop ≥ 23 has this by default)
- A GitHub PAT with `repo` read access → `GIT_PAT`
- Cloudflare R2 API credentials → `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT`, `R2_BUCKET`

```bash
# Export credentials once per shell session (or add to ~/.zshrc / .env)
export GIT_PAT=<github-pat>
export R2_ACCESS_KEY_ID=<r2-access-key>
export R2_SECRET_ACCESS_KEY=<r2-secret-key>
export R2_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
export R2_BUCKET=<bucket-name>
```

### 1. Build the dev image

```bash
# Build once; rebuild only when deps or Surge version change.
make docker-build-dev-live \
  GIT_PAT=$GIT_PAT \
  R2_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID \
  R2_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY \
  R2_ENDPOINT=$R2_ENDPOINT \
  R2_BUCKET=$R2_BUCKET
```

### 2. Local development (mount your source)

```bash
# Your local code is live-mounted inside the container.
# Edit files on the host; changes appear immediately in the container.
make docker-run-dev USE_LOCAL_WORKSPACE=1
```

### 3. Train from an R2 dataset (or local)

```bash
# Download dataset and run full training
make docker-run-gpu-train R2_PREFIX=runs/surge_simple/<commit-sha>

# Override the experiment config
make docker-run-gpu-train \
  R2_PREFIX=runs/surge_simple/<commit-sha> \
  TRAIN_ARGS="experiment=surge/flow_simple trainer.max_epochs=50"

# 1-step plumbing check (no GPU needed)
make docker-run-cpu-train \
  R2_PREFIX=runs/surge_simple/<commit-sha> \
  TRAIN_ARGS="experiment=surge/flow_simple trainer.max_steps=1"
```

### 4. Build a self-contained snapshot image (for vast.ai / RunPod)

```bash
# Bakes source at <commit-sha> — run anywhere without mounting anything.
make docker-build-dev-snapshot \
  GIT_REF=<commit-sha> \
  GIT_PAT=$GIT_PAT \
  R2_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID \
  R2_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY \
  R2_ENDPOINT=$R2_ENDPOINT \
  R2_BUCKET=$R2_BUCKET

# Run the snapshot image (no volume mount needed)
make docker-run-gpu-train \
  IMAGE_TAG=<base-image-tag>-dev-snapshot-<commit-sha> \
  R2_PREFIX=runs/surge_simple/<commit-sha>
```

### 5. Local CI dry-run (no TTY, no GPU — mirrors what CI does)

```bash
# Simulate the CI train job locally
make docker-run-cpu-train \
  R2_PREFIX=runs/surge_simple/<commit-sha> \
  TRAIN_ARGS="experiment=surge/flow_simple trainer.max_steps=1"
```

### 6. Run unit tests

```bash
# Fast pipeline tests (no VST, no R2, no Docker)
pytest tests/test_run_dataset_pipeline.py -v

# Full not-slow suite
pytest -k "not slow" -v
```

______________________________________________________________________

## Metadata and Traceability

Every generation run writes `metadata.json` alongside the HDF5 files. Example:

```json
{
  "generated_at": "2026-02-22T12:00:00+00:00",
  "git_sha": "a1b2c3d",
  "git_ref_source": "baked",
  "git_dirty": false,
  "param_spec": "surge_simple",
  "param_spec_num_params": 92,
  "splits": {"train": 50000, "val": 5000, "test": 5000},
  "r2_prefix": "runs/surge_simple/a1b2c3d",
  "generation": {
    "plugin_path": "/usr/lib/vst3/Surge XT.vst3",
    "preset_path": "presets/surge-base.vstpreset",
    "sample_rate": 44100.0,
    "channels": 2,
    "velocity": 100,
    "signal_duration_seconds": 4.0,
    "min_loudness": -55.0,
    "sample_batch_size": 32
  }
}
```

### Git provenance fields

These fields let dataset consumers distinguish clean production datasets from dev experiments:

| Field            | Values      | Meaning                                                                                                               |
| ---------------- | ----------- | --------------------------------------------------------------------------------------------------------------------- |
| `git_sha`        | `"a1b2c3d"` | Commit SHA of the code that generated this dataset.                                                                   |
| `git_ref_source` | `"baked"`   | Source was downloaded as a tarball at image build time (prod or dev-snapshot). `git_sha` is authoritative.            |
|                  | `"local"`   | Source was mounted or git-cloned at runtime (dev-live image). `git_sha` reflects HEAD at run time. Check `git_dirty`. |
|                  | `"unknown"` | Provenance could not be determined.                                                                                   |
| `git_dirty`      | `false`     | Working tree was clean at generation time. Dataset is reproducible from `git_sha`.                                    |
|                  | `true`      | Working tree had uncommitted changes. Dataset may not be exactly reproducible from `git_sha` alone.                   |
|                  | `null`      | Could not be determined.                                                                                              |

**Rule of thumb:** A dataset is production-quality when `git_ref_source == "baked"` and
`git_dirty == false`. A dataset with `git_ref_source == "local"` and `git_dirty == true` is a
dev/experiment artifact.

### Schema versioning

> **TODO:** `dataset_schema_version` will be added to `metadata.json` once it is wired end-to-end
> through `generate_vst_dataset.py` (writes HDF5 root attribute) and `surge_datamodule.py`
> (validates on load). The constant is already defined in `src/data/dataset_version.py`.
> Increment it whenever the HDF5 structure changes in a backwards-incompatible way.

### CI dataset lifecycle

The `docker-smoke-test-train` CI job generates a tiny dataset (N=5/2/2), uploads it to
`r2:<bucket>/ci-smoke/<run_id>/`, downloads and trains for 1 step, then cleans up the prefix with
`rclone purge`. This means R2 is only used transiently during CI — no long-lived CI datasets accumulate.
