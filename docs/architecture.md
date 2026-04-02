# Architecture Overview

High-level system overview for synth-setter. For detailed design, see the
individual design docs linked throughout.

## What This Project Does

synth-setter is a collection of tools for **synthesizer inversion** (predicting
synthesizer parameters from audio), **sound matching**, and **preset
exploration**. The system generates large-scale audio datasets by rendering
random synthesizer configurations through a VST plugin (Surge XT), trains neural
networks on these datasets, and evaluates how well the models recover the
original parameters.

## System Diagram

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                          synth-setter pipeline                         │
 │                                                                        │
 │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
 │  │ GENERATE │───>│ FINALIZE │───>│  TRAIN   │───>│    EVALUATE      │  │
 │  │          │    │          │    │          │    │                  │  │
 │  │ Render   │    │ Reshard  │    │ Flow     │    │ Predict → Render │  │
 │  │ audio via│    │ into     │    │ matching │    │ → Metrics        │  │
 │  │ Surge XT │    │ splits   │    │ model    │    │                  │  │
 │  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────────┬─────────┘  │
 │       │               │               │                   │            │
 │       ▼               ▼               ▼                   ▼            │
 │    HDF5 shards   train/val/test   Checkpoints       Metrics CSV       │
 │    → R2          → R2             → W&B             Rendered audio     │
 └─────────────────────────────────────────────────────────────────────────┘

 Infrastructure:
   Storage:   Cloudflare R2 (data, coordination)
   Compute:   RunPod (on-demand workers)
   Tracking:  Weights & Biases (metrics, artifacts, lineage)
   Config:    Hydra (composable YAML configs)
   Training:  PyTorch Lightning
```

## Data Flow

1. **Configure** -- Define a dataset in `configs/dataset/*.yaml` (synth, sample
   count, shard size, parameter spec).

2. **Generate** -- Workers render audio samples through Surge XT, producing HDF5
   shards uploaded to R2. Each shard contains audio waveforms, mel spectrograms,
   and ground-truth parameter arrays. Workers are fully parallel with no shared
   state.
   Design: [data-pipeline.md](design/data-pipeline.md)

3. **Finalize** -- Downloads validated shards, reshards into train/val/test
   splits (HDF5 virtual datasets or WebDataset `.tar`), computes normalization
   statistics, registers the dataset as a W&B artifact, and writes
   `dataset.complete`.
   Design: [data-pipeline.md](design/data-pipeline.md)

4. **Train** -- A single long-running job trains a model (flow matching,
   feed-forward, or FlowVAE) on the generated dataset. Checkpoints are durably
   stored in W&B via `log_model: "all"`. Hydra composes experiment configs from
   data, model, trainer, and callback configs.
   Design: [training-pipeline.md](design/training-pipeline.md)

5. **Evaluate** -- Three stages: **predict** (model inference on test data),
   **render** (synthesize audio from predicted parameters via Surge XT), and
   **metrics** (spectral and transport-based distance metrics). Results upload to
   R2.
   Design: [eval-pipeline.md](design/eval-pipeline.md)

## Directory Structure

```
synth-setter/
├── src/                    # ML code
│   ├── train.py            #   Training entry point (Hydra)
│   ├── eval.py             #   Evaluation entry point (Hydra)
│   ├── metrics.py          #   Metric definitions
│   ├── data/               #   DataModules (Surge, K-Sin, K-Osc, etc.)
│   ├── models/             #   LightningModules (flow matching, FF, FlowVAE)
│   │   └── components/     #   Model building blocks (VAE, networks)
│   └── utils/              #   Logging, config helpers
│
├── pipeline/               # Distributed data pipeline
│   ├── schemas/            #   Pydantic models (config, spec, prefix, image_config)
│   ├── entrypoints/        #   Pipeline entry points (generate_dataset)
│   ├── ci/                 #   CI validation scripts (materialize_spec, validate_shard/spec)
│   └── constants.py        #   Shared constants (R2 bucket, spec filename)
│
├── configs/                # Hydra YAML configs
│   ├── dataset/            #   Pipeline dataset configs (synth, shard count, sample count)
│   ├── data/               #   DataModule configs (paths, splits, batch size)
│   ├── model/              #   Model architecture configs
│   ├── trainer/            #   Lightning Trainer configs
│   ├── experiment/         #   Experiment configs (compose data + model + trainer)
│   ├── callbacks/          #   Callback configs (checkpointing, early stopping)
│   ├── logger/             #   Logger configs (W&B, CSV, TensorBoard)
│   └── train.yaml          #   Root training config
│
├── scripts/                # Standalone utility scripts
├── tests/                  # Test suite (mirrors src/ and pipeline/ structure)
├── docs/                   # Documentation
│   └── design/             #   Design documents
└── docker/                 # Dockerfiles and entrypoints
```

## Key Design Decisions

**R2 as source of truth.** Pipeline state is determined by file existence and
validation in R2, not by metadata databases or coordination services. One piece
of infrastructure, one set of credentials, one failure mode. See
[data-pipeline.md](design/data-pipeline.md) section 7.1.

**Reconciliation over orchestration.** Instead of a job scheduler tracking task
state, the pipeline compares the desired state (input spec) against actual state
(validated shards in R2) to determine remaining work. Any command can be re-run
safely at any time. See [data-pipeline.md](design/data-pipeline.md) section 7.4.

**Deterministic shard identities.** Shard IDs are logical (`shard-000042`),
defined at run creation, independent of which worker or infrastructure computes
them. This makes reconciliation straightforward and results reproducible. See
[data-pipeline.md](design/data-pipeline.md) section 7.3.

**Worker isolation.** Workers are fully parallel with no shared state. Each
worker independently renders its assigned shards and uploads to R2. One worker
crashing does not affect others. See
[data-pipeline.md](design/data-pipeline.md) section 7.7.

**W&B for checkpoint durability.** Training checkpoints are uploaded to W&B
immediately via `log_model: "all"`, surviving pod death without a custom
persistence layer. See
[training-pipeline.md](design/training-pipeline.md) section 6.

**Storage conventions are shared.** All pipelines (data, training, eval) follow
the same R2 path structure and ID conventions defined in
[storage-provenance-spec.md](design/storage-provenance-spec.md).

## Design Documents

| Document                                                        | Covers                                                            |
| --------------------------------------------------------------- | ----------------------------------------------------------------- |
| [data-pipeline.md](design/data-pipeline.md)                     | Distributed dataset generation, finalization, reconciliation      |
| [training-pipeline.md](design/training-pipeline.md)             | Training orchestration, checkpoint durability, resume             |
| [eval-pipeline.md](design/eval-pipeline.md)                     | Evaluation pipeline (predict, render, metrics) and R2 integration |
| [storage-provenance-spec.md](design/storage-provenance-spec.md) | Authoritative R2 paths, W&B artifacts, ID conventions             |
