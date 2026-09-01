# Architecture Overview

High-level system overview for synth-setter. For detailed design, see the
individual design docs linked throughout.

## What This Project Does

synth-setter is a collection of tools for **synthesizer inversion** (predicting
synthesizer parameters from audio), **sound matching**, and **preset
exploration**. The system generates large-scale audio datasets by rendering
random synthesizer configurations through a configured audio renderer, trains neural networks
on these datasets, and evaluates how well the models recover the original
parameters.

The pipeline is **synth-agnostic**: rendering, storage, features, distributed
workers, and the models are all driven by a `ParamSpec` (parameter schema) and a
`RenderConfig` (backend and synth identity) looked up from a registry by name.
Surge XT is the default and can render through Pedalboard, DawDreamer, or the
pinned in-process SurgePy engine; an opt-in pyFDN profile effects rendered audio
before validation and feature extraction. OB-Xf is registered as a second VST3
synth, and Faust identities compile checked-in source through DawDreamer. SurgePy
recreates the native synth for every row and accepts only
`plugin_reload_cadence: render`. VST3 plugins can be
onboarded with **no edits to core pipeline, storage, or model code**. See
[Adding a new synth](guides/adding-a-new-synth.md).

## System Diagram

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                          synth-setter pipeline                         │
 │                                                                        │
 │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
 │  │ GENERATE │───>│ FINALIZE │───>│  TRAIN   │───>│    EVALUATE      │  │
 │  │          │    │          │    │          │    │                  │  │
 │  │ Render   │    │ Compose  │    │ Flow     │    │ Predict → Render │  │
 │  │ audio via│    │ into     │    │ matching │    │ → Metrics        │  │
 │  │ renderer │    │ splits   │    │ model    │    │                  │  │
 │  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────────┬─────────┘  │
 │       │               │               │                   │            │
 │       ▼               ▼               ▼                   ▼            │
 │    Lance shards  train/val/test   Checkpoints       Metrics CSV       │
 │    → R2          → R2             → W&B             Rendered audio     │
 └─────────────────────────────────────────────────────────────────────────┘

 Infrastructure:
   Storage:   Cloudflare R2 (data, coordination)
   Compute:   SkyPilot (RunPod, Vast.ai, local Kubernetes)
   Tracking:  Weights & Biases (metrics, artifacts, lineage)
   Config:    Hydra (composable YAML configs)
   Training:  PyTorch Lightning
```

## Data Flow

1. **Configure** -- Define a dataset in `src/synth_setter/configs/experiment/generate_dataset/*.yaml` (synth, sample
   count, shard size, parameter spec). The synth is selected by the root
   `synth` group (`synth=surge_xt`, `synth=obxf`, ...), which carries the
   registered parameter spec, preset, plugin path, and version; the paired
   `render` group selects a renderer profile. Base profiles configure a backend;
   derived profiles may add post-render processing. `render=vst` is dry, while
   `render=vst_pyfdn` adds the fixed live pyFDN effect. Hydra
   composes the experiment against
   `src/synth_setter/configs/dataset.yaml` and `spec_from_cfg(cfg)` (in
   `src/synth_setter/cli/generate_dataset.py`) builds the unified `DatasetSpec`.

2. **Generate** -- Workers render audio samples through the configured synth
   backend, producing Lance
   shards uploaded to R2. Each shard contains audio waveforms, mel spectrograms,
   and ground-truth parameter arrays. Workers are fully parallel with no shared
   state.
   Design: [data-pipeline.md](design/data-pipeline.md)

3. **Finalize** -- Downloads validated shards, commits their Lance fragments
   into train/val/test split datasets, computes normalization
   statistics, registers the dataset as a W&B artifact, and writes
   `dataset.complete`.
   Design: [data-pipeline.md](design/data-pipeline.md)

4. **Train** -- A single long-running job trains a model (flow matching,
   feed-forward, or FlowVAE) on the generated dataset. At train end the best
   checkpoint is uploaded to R2 and referenced by the `model-{config_id}` W&B
   artifact (`log_model: False`, so no checkpoint files go to W&B). Hydra composes
   experiment configs from datamodule, model, trainer, and callback configs.
   VST datasets load from
   [Lance](https://github.com/lance-format/lance) shards (`datamodule=surge_lance`)
   through sample-indexed native `lance.torch` map datasets. The sequential native
   loader remains available for streaming workflows — see
   [training-pipeline.md §6.1](design/training-pipeline.md#61-dataset-access). The datamodule class is
   param-count-agnostic; synth identity is selected once at the config root via
   the `synth` group (`synth=<name>`), which VST datamodules, models, and
   callbacks all resolve through `${synth.param_spec_name}`.
   Design: [training-pipeline.md](design/training-pipeline.md)

5. **Evaluate** -- Three stages: **predict** (model inference on test data),
   **render** (synthesize audio from predicted parameters via the same renderer
   backend that generated the dataset), and
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
│   │   ├── generate_dataset.py  # Dataset-generation entrypoint
│   │   └── ...
│   ├── metrics.py          #   Metric definitions
│   ├── data/               #   DataModules (TorchSynth, VST, Lance, audio folders)
│   ├── models/             #   LightningModules (flow matching, FF, FlowVAE)
│   │   └── components/     #     Model building blocks (VAE, networks)
│   ├── utils/              #   Logging, config helpers
│   ├── pipeline/           #   Distributed data pipeline
│   │   ├── schemas/        #     Pydantic models (DatasetSpec, RenderConfig, prefix, image_config)
│   │   ├── ci/             #     CI validation scripts (materialize_spec, validate_shard, validate_spec)
│   │   ├── data/           #     Dataset-shaping utilities (lance_staging, lance_finalize, stats, ...)
│   │   ├── skypilot_launch.py  # SkyPilot launcher CLI
│   │   └── constants.py    #     Shared constants (`INPUT_SPEC_FILENAME`)
│   ├── evaluation/         #   Render/metrics library code (predict_vst_audio, compute_audio_metrics, shuffle_pred_audio, audio_probe) shared by cli/eval.py and the training val-audio probe
│   ├── tools/              #   `python -m` utilities (vst_interactive, plot_param2tok, ...)
│   └── configs/            #   Hydra YAML configs (and SkyPilot Task templates under compute/) — #1236
│       ├── train.yaml      #     Root training config
│       ├── dataset.yaml    #     Root dataset-generation config (entrypoint mirrors train.yaml / eval.yaml)
│       ├── experiment/     #     Experiment configs — training (compose datamodule + model + trainer) and datagen (composes dataset.yaml)
│       ├── compute/        #     SkyPilot compute options (RunPod, Vast.ai, local Kubernetes)
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

**Synth-agnostic core, registry as the contract.** A synth's identity — which
`ParamSpec`, which plugin, which baseline preset — is authored once in
`SYNTHS` (`src/synth_setter/synth_spec.py`); `plugin_state_paths` and the
root identity group `src/synth_setter/configs/synth/<name>.yaml` are
projections of it, pinned against the table by `tests/test_synth_spec.py` and
`tests/schemas/test_synth_config.py`. Render configs in
`src/synth_setter/configs/render/<name>.yaml` declare backend-specific
settings only and are paired with a `synth=<name>` selection. The `ParamSpec` objects themselves live in
`src/synth_setter/data/vst/param_spec_registry.py`. The rendering, Lance
storage, mel features, distributed workers, and models all read width and
behavior from the resolved spec, never from a synth literal. Faust entries use
an empty state path and resolve checked-in source by the same identity.
`studiorack.json`, `studiorack.lock.json`, `plugin_manager.py`,
`plugin_integrity.py`, `plugin_runtime.py`, and `synth-setter-plugins` manage exact VST3 packages,
artifact identities, and content-sealed bundles beneath stable `plugins/*.vst3`
identity paths. Onboarding a
new VST3 synth is additive: install its package, scaffold and hand-tune a spec,
then pair it with the dry `render=vst` profile or the effected
`render=vst_pyfdn` profile. See
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
