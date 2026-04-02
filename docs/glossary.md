# Glossary

Project terminology for synth-setter. Grouped by domain.

## Core Concepts

| Term                | Definition                                                                                                                                                                                                |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Synth inversion** | Predicting synthesizer parameters from an audio recording. Also called **parameter estimation**. The core ML task this project addresses.                                                                 |
| **Sound matching**  | Finding synthesizer settings that reproduce a target sound. Synth inversion is the neural approach to this problem.                                                                                       |
| **Preset**          | A saved configuration of synthesizer parameters that produces a specific sound. The project generates random presets for training data.                                                                   |
| **Parameter space** | The set of all controllable parameters of a synthesizer. For Surge XT, this ranges from 92 parameters (`surge_simple`) to 189 (`surge_xt`) depending on the param spec.                                   |
| **param_spec**      | Configuration selecting which synthesizer parameters to vary during data generation. Determines the dimensionality of the prediction task. Examples: `surge_simple` (92 params), `surge_xt` (189 params). |

## Model Architecture

| Term                   | Definition                                                                                                                                                                                                              |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Flow matching**      | A generative modeling technique used for parameter prediction. Learns a vector field that transports noise to the target parameter distribution. Implemented in `SurgeFlowMatchingModule` and `KSinFlowMatchingModule`. |
| **Optimal transport**  | A mathematical framework for comparing probability distributions. Used in flow matching training and in the SOT evaluation metric.                                                                                      |
| **FlowVAE**            | A variational autoencoder variant combined with flow-based generation. Implemented in `FlowVAE` / `SurgeFlowVAEModule`.                                                                                                 |
| **Feed-forward model** | A direct regression model for parameter prediction (no iterative sampling). Implemented in `SurgeFFModule` and `KSinFFModule`.                                                                                          |

## Audio & Synthesis

| Term                                | Definition                                                                                                                                                      |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **VST (Virtual Studio Technology)** | Plugin format for audio synthesizers and effects. Surge XT is loaded as a VST plugin via Spotify's [pedalboard](https://github.com/spotify/pedalboard) library. |
| **Surge XT**                        | The open-source VST synthesizer used for audio rendering. See [surge-synthesizer.github.io](https://surge-synthesizer.github.io/).                              |
| **Mel spectrogram**                 | Frequency-domain audio representation used as neural network input. 128 mel bands, ~100 frames/sec.                                                             |
| **pedalboard**                      | Spotify's Python library for loading and running VST plugins programmatically.                                                                                  |
| **Xvfb**                            | X Virtual Framebuffer. Provides a virtual display server on headless Linux machines, required because VST plugins expect a display.                             |

## Data Pipeline

| Term                  | Definition                                                                                                                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Shard**             | An HDF5 file containing a batch of training samples (audio, mel spectrograms, parameter arrays). Typically 1k--10k samples per shard. Named by logical index (e.g., `shard-000042.h5`).          |
| **Shard ID**          | Logical index for a shard (`shard-000042`). Deterministic, defined at run creation, independent of which worker computes it.                                                                     |
| **Input spec**        | JSON file (`input_spec.json`) freezing all generation parameters for a run: per-shard seeds, shapes, splits, renderer version. Written once, never modified.                                     |
| **Finalize**          | The pipeline stage that downloads all validated shards, reshards into train/val/test splits, computes normalization statistics, registers the dataset in W&B, and writes `dataset.complete`.     |
| **Reconciliation**    | Comparing desired state (the spec) against actual state (validated shards in R2) to determine what work remains. The pipeline's coordination mechanism -- no message queues or databases needed. |
| **Worker**            | A cloud compute instance that generates shards. On RunPod, this is a single Docker container ("pod") with assigned shard work.                                                                   |
| **dataset.complete**  | Marker file written by finalize as the very last step. Signals that finalization is done. Contains run_id and timestamp. Once present, the dataset is immutable.                                 |
| **Lifecycle marker**  | Empty file in worker metadata tracking shard state: `.rendering` (started), `.valid` (committed), `.promoted` (canonical), `.invalid` (failed). Presence is the state -- no content to parse.    |
| **Quarantined shard** | A corrupt shard uploaded to a quarantine path on validation failure. Preserves evidence for debugging.                                                                                           |
| **Virtual dataset**   | HDF5 feature that creates a logical view over multiple files without copying data. Used by finalize to compose train/val/test splits from individual shards.                                     |
| **WebDataset**        | [WebDataset](https://github.com/webdataset/webdataset) format for streaming training data. Sequential `.tar` archives optimized for HTTP/S3 streaming, used for multi-GPU training.              |
| **Dataset card**      | JSON file (`dataset.json`) describing the finalized dataset: provenance, structure, stats. References the spec by SHA-256.                                                                       |

## Evaluation Pipeline

| Term        | Definition                                                                                                                                  |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **Predict** | Evaluation stage 1: load a trained checkpoint, run inference on test data, output predicted parameter tensors.                              |
| **Render**  | Evaluation stage 2: feed predicted parameters into the VST plugin to produce audio waveforms for both predictions and ground-truth targets. |
| **Metrics** | Evaluation stage 3: compute distance metrics (MSS, wMFCC, SOT, RMS) between predicted and target audio.                                     |
| **MSS**     | Multi-Scale Spectrogram distance. Captures temporal characteristics at three mel-scale windows (fine, mid, coarse).                         |
| **SOT**     | Spectral Optimal Transport. Wasserstein distance on normalized STFT bins.                                                                   |

## IDs & Provenance

| Term                     | Definition                                                                                                                                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **dataset_config_id**    | Stable identifier for a dataset configuration, derived from the config YAML filename stem. Example: `surge-simple-480k-10k`. See [storage-provenance-spec.md](design/storage-provenance-spec.md). |
| **dataset_wandb_run_id** | Unique identifier for a pipeline execution. Format: `{dataset_config_id}-{YYYYMMDDTHHMMSSZ}`. Example: `surge-simple-480k-10k-20260312T143022Z`.                                                  |
| **train_config_id**      | Config filename stem for a training experiment. Example: `flow-simple`.                                                                                                                           |
| **train_wandb_run_id**   | W&B run ID for a specific training run. Format: `{train_config_id}-{YYYYMMDDTHHMMSSZ}`.                                                                                                           |
| **worker_id**            | Infrastructure identifier (e.g., RunPod's `RUNPOD_POD_ID`). Appears only in metadata, not in shard paths.                                                                                         |

## Infrastructure & Tools

| Term                       | Definition                                                                                                                                                                |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **R2**                     | [Cloudflare R2](https://developers.cloudflare.com/r2/), S3-compatible object storage with free egress. Used for shard storage, pipeline coordination, and eval artifacts. |
| **RunPod**                 | [RunPod](https://www.runpod.io/), cloud compute marketplace for on-demand GPU/CPU instances. Used for data generation workers and training.                               |
| **W&B (Weights & Biases)** | [Weights & Biases](https://wandb.ai/), experiment tracking platform. Used for pipeline metrics, dataset artifact registry, checkpoint durability, and lineage tracking.   |
| **Hydra**                  | [Hydra](https://hydra.cc/), configuration framework for Python. Composes YAML configs with CLI overrides. All training, eval, and dataset configs use Hydra.              |
| **OmegaConf**              | Hydra's underlying config library. Supports interpolation (`${...}`) and environment variable resolvers (`${oc.env:VAR}`).                                                |
| **Lightning**              | [PyTorch Lightning](https://lightning.ai/), training framework. Handles training loops, checkpointing, logging, and multi-GPU support.                                    |
| **rclone**                 | CLI tool for syncing files to cloud storage. All R2 transfers use `--checksum` (project rule).                                                                            |
| **structlog**              | Structured logging library used in pipeline code. JSON-rendered, append-only debug log streams.                                                                           |
