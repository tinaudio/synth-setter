# Architecture Overview

High-level system overview for synth-setter. For detailed design, see the
individual design docs linked throughout.

## What This Project Does

synth-setter is a collection of tools for **synthesizer inversion** (predicting
synthesizer parameters from audio), **sound matching**, and **preset
exploration**. The system generates large-scale audio datasets by rendering
random synthesizer configurations through a VST3 synth, trains neural networks
on these datasets, and evaluates how well the models recover the original
parameters.

The pipeline is **synth-agnostic**: rendering, storage, features, distributed
workers, and the models are all driven by a `ParamSpec` (parameter schema) and a
`RenderConfig` (plugin path, preset, spec name) looked up from a registry by
name. Surge XT is the default; OB-Xf is registered as a second synth, and any
VST3 plugin can be onboarded with **no edits to core pipeline, storage, or model
code**. See
[Adding a new synth](guides/adding-a-new-synth.md).

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
 │  │VST3 synth│    │ splits   │    │ model    │    │                  │  │
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

1. **Configure** -- Define a dataset in `src/synth_setter/configs/experiment/generate_dataset/*.yaml` (synth, sample
   count, shard size, parameter spec). The synth is selected by a `render`
   group override (e.g. `render=surge_xt` or `render=obxf`); each render config
   names the registered `param_spec_name`, preset, and plugin path. Hydra
   composes the experiment against
   `src/synth_setter/configs/dataset.yaml` and `spec_from_cfg(cfg)` (in
   `src/synth_setter/cli/generate_dataset.py`) builds the unified `DatasetSpec`.

2. **Generate** -- Workers render audio samples through the configured VST3
   synth, producing HDF5
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
   feed-forward, or FlowVAE) on the generated dataset. At train end the best
   checkpoint is uploaded to R2 and referenced by the `model-{config_id}` W&B
   artifact (`log_model: False`, so no checkpoint files go to W&B). Hydra composes
   experiment configs from datamodule, model, trainer, and callback configs.
   VST datasets load from HDF5 shards (`datamodule=surge`) or
   [Lance](https://github.com/lance-format/lance) shards (`datamodule=surge_lance`);
   both serve training and evaluation. The datamodule class is
   param-count-agnostic, though the `surge*` configs pin `param_spec_name`, so
   training a non-Surge dataset overrides `datamodule.param_spec_name=<name>`.
   Design: [training-pipeline.md](design/training-pipeline.md)

5. **Evaluate** -- Three stages: **predict** (model inference on test data),
   **render** (synthesize audio from predicted parameters via the same VST3
   synth that generated the dataset), and
   **metrics** (spectral and transport-based distance metrics). Results upload to
   R2.
   Design: [eval-pipeline.md](design/eval-pipeline.md)

## Directory Structure

```
synth-setter/
├── src/synth_setter/       # PEP src-layout package (#784)
│   ├── cli/                #   @hydra.main / click entrypoints (published as synth-setter-* console scripts)
│   │   ├── train.py        #     Training entrypoint
│   │   ├── eval.py         #     Evaluation entrypoint
│   │   └── generate_dataset.py  # Dataset-generation entrypoint
│   ├── metrics.py          #   Metric definitions
│   ├── data/               #   DataModules (Surge, K-Sin, K-Osc, etc.)
│   ├── models/             #   LightningModules (flow matching, FF, FlowVAE)
│   │   └── components/     #     Model building blocks (VAE, networks)
│   ├── utils/              #   Logging, config helpers
│   ├── pipeline/           #   Distributed data pipeline
│   │   ├── schemas/        #     Pydantic models (DatasetSpec, RenderConfig, prefix, image_config)
│   │   ├── ci/             #     CI validation scripts (materialize_spec, validate_shard, validate_spec)
│   │   ├── data/           #     Dataset-shaping utilities (reshard, rewrite_to_latest, stats, r2_report, ...)
│   │   ├── skypilot_launch.py  # SkyPilot launcher CLI
│   │   └── constants.py    #     Shared constants (`INPUT_SPEC_FILENAME`)
│   ├── evaluation/         #   predict_vst_audio, compute_audio_metrics, shuffle_pred_audio (library code called by cli/eval.py)
│   ├── tools/              #   `python -m` utilities (surge_xt_interactive, plot_param2tok, ...)
│   └── configs/            #   Hydra YAML configs (and SkyPilot Task templates under compute/) — #1236
│       ├── train.yaml      #     Root training config
│       ├── dataset.yaml    #     Root dataset-generation config (entrypoint mirrors train.yaml / eval.yaml)
│       ├── experiment/     #     Experiment configs — training (compose datamodule + model + trainer) and datagen (composes dataset.yaml)
│       ├── compute/        #     SkyPilot Task YAMLs for the data pipeline launcher (RunPod landed; Vast.ai planned)
│       ├── render/         #     Renderer configs (RenderConfig sub-model)
│       ├── datamodule/     #     DataModule configs (paths, splits, batch size)
│       ├── model/          #     Model architecture configs
│       ├── trainer/        #     Lightning Trainer configs
│       ├── callbacks/      #     Callback configs (checkpointing, early stopping)
│       └── logger/         #     Logger configs (W&B, CSV, TensorBoard)
│
├── scripts/                # SkyPilot/CI shell tooling (skypilot/, ci/) — bare root is empty by design
├── tests/                  # Test suite (mirrors src/synth_setter/ structure)
├── docs/                   # Documentation
│   └── design/             #   Design documents
└── docker/                 # Dockerfiles and image-build helpers
```

## Key Design Decisions

**Synth-agnostic core, registry as the contract.** A synth is fully described
by three registered artifacts — a `ParamSpec` (`param_specs[name]`), a baseline
preset (`preset_paths[name]`), and a `RenderConfig` (`configs/render/<name>.yaml`)
— keyed by name in `src/synth_setter/data/vst/param_spec_registry.py`. The
rendering, HDF5/Lance storage, mel features, distributed workers, and models all
read width and behavior from the resolved spec, never from a synth literal.
Onboarding a new VST3 synth is additive: scaffold a spec with
`synth-setter-introspect-plugin`, hand-tune it, register it, and write a render
config — no core edits. See
[Adding a new synth](guides/adding-a-new-synth.md).

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

**R2 for checkpoint durability.** `log_model: False` keeps checkpoint files out
of W&B (5 GB total budget); at train end the best checkpoint is uploaded to R2
and the `model-{config_id}` W&B artifact references it as an `s3://` URI. See
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
| [guides/adding-a-new-synth.md](guides/adding-a-new-synth.md)    | Onboard a new VST3 synth: introspect, tune, register, generate    |
