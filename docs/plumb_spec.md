# Synth-Setter: Specification

## Configuration and Identity

The system derives dataset configuration identifiers from the configuration YAML filename stem.

The system derives training configuration identifiers from the experiment configuration YAML filename stem.

The system derives evaluation configuration identifiers from the evaluation configuration YAML filename stem.

The system enforces a 64-character limit on W&B run identifiers.

The system validates all external inputs at trust boundaries with strict mode and no implicit type coercion: configuration files, R2 JSON payloads, and worker reports.

The system rejects configuration files that contain unknown fields or invalid types.

Internal data containers are immutable after construction.

The training pipeline uses composable configuration via Hydra for experiment composition.

The system records the resolved training configuration for each run.

The system loads secrets exclusively from environment variables, never from committed files.

The system resolves W&B entity and project from environment variables with configurable defaults.

## Input Specification

The generate stage creates the input spec once as a frozen JSON document and uploads it to R2.

The pipeline rejects any attempt to generate against an existing run ID with a different configuration.

The pipeline reads the input spec exclusively from R2 after initial creation, never from local configuration files.

The input spec is immutable after creation — no pipeline stage modifies it.

The pipeline assigns run IDs in the format `{config_id}-{YYYYMMDDTHHMMSSZ}` with no random components.

The pipeline extracts the VST plugin version from bundle metadata and stores it in the input spec.

The input spec records per-shard seeds, shapes, renderer version, source commit SHA, and repository dirty status.

The same input configuration produces the same input spec.

## Audio Dataset Generation

The pipeline renders audio samples through a VST synthesizer plugin in headless mode on Linux.

The pipeline produces an HDF5 shard per worker containing audio waveforms, mel spectrograms, and ground-truth parameter arrays.

Each shard contains a fixed number of samples defined by the dataset configuration.

The pipeline assigns logical shard IDs in the format `shard-NNNNNN` that are independent of infrastructure.

The pipeline derives shard seeds deterministically as `base_seed + shard_id`.

Infrastructure identifiers appear only in staging filenames and worker metadata, never in canonical shard identity.

The pipeline partitions missing shards across N workers for parallel generation.

The worker writes each attempt to a unique filename using the worker ID and an attempt UUID to prevent collisions.

The worker writes a rendering lifecycle marker when a shard attempt begins.

The worker writes a valid lifecycle marker only after render, local validation, upload, and bookkeeping are all complete.

The worker writes an invalid lifecycle marker and uploads the corrupt shard to a quarantine path when local validation fails.

The pipeline never deletes lifecycle markers, preserving the full attempt history as append-only metadata.

## Shard Validation

The pipeline validates shards at four tiers: structural integrity, tensor shape, value range, and row count.

The worker performs full four-tier validation before uploading a shard.

The worker records a content hash of each validated shard in its report for provenance.

The generate and status commands determine shard completeness by checking for the shard file plus its valid marker without re-validating the data.

The finalize stage validates each staged shard by verifying expected datasets exist and shapes match the input spec before promotion.

The finalize stage quarantines shards that fail structural checks and treats them as missing on the next generation invocation.

## Worker Coordination

R2 file existence and validation markers determine pipeline state, not metadata, caches, or provider APIs.

The worker writes exclusively under the worker metadata path, never under canonical data paths.

The worker emits structured JSON logs to a dedicated local file.

The worker uploads its log file to R2 via a shell exit trap upon process termination.

All remote storage transfers use checksum verification.

The pipeline treats a checksum mismatch as an immediate failure and triggers re-transfer.

## Reconciliation and Resumability

The pipeline reconciles state from R2 files on every invocation.

The pipeline determines remaining work by comparing desired state from the input spec against validated shards in R2.

The system uses R2 as both the data plane and the control plane with no external database, queue, or coordination service.

The generate command exits with success and no side effects when all shards are already staged and valid.

The finalize command exits with success and no side effects when the completion marker exists and all canonical outputs are valid.

The status command produces a reconciliation report showing counts of valid, missing, quarantined, and rendering shards derived entirely from storage state.

The pipeline re-runs any command idempotently without data corruption or state inconsistency.

The pipeline detects duplicate shard identifiers during reconciliation.

The pipeline validates storage and compute credentials before launching any workers.

## Dataset Finalization

The finalize stage writes exclusively to the canonical data directory.

The worker cannot write to the canonical data directory.

The finalize stage selects one staged shard per logical shard ID using deterministic lexicographic ordering of attempt filenames.

The finalize stage promotes each validated staged shard by copying it to the canonical path and writing a promoted marker.

The finalize stage computes normalization statistics across the training split.

The finalize stage produces HDF5 virtual datasets for training, validation, and test splits when the output format is HDF5.

The finalize stage produces WebDataset tar archives per split when the output format is WebDataset.

The finalize stage writes a self-describing dataset card containing provenance, structure, statistics, and a reference to the input spec.

The finalize stage writes the completion marker as the very last step after all outputs are uploaded and verified.

The finalize stage registers the finalized dataset as a versioned W&B artifact.

The finalize stage reruns from scratch if the completion marker is absent, with no partial checkpoint recovery.

The completion marker gates all downstream consumption of a dataset.

No pipeline stage or training job reads from R2 paths that lack a completion marker.

Canonical data is immutable after the completion marker is written.

A new dataset version requires a new unique run ID — the pipeline never overwrites an existing dataset.

## Model Training

The training pipeline loads a dataset artifact produced by the dataset generation pipeline.

The training pipeline uploads every saved checkpoint as a W&B artifact immediately after saving.

The training pipeline saves checkpoints at configurable step intervals plus best-metric and last checkpoints.

The training pipeline resumes from either a local checkpoint path or a W&B artifact reference.

The training pipeline logs hyperparameters, training metrics, and validation metrics to W&B.

The training pipeline ensures W&B run teardown completes even on exception.

The training pipeline produces identical results across local, container, and cloud environments. (AMBIGUOUS)

Committed training configurations contain no hardcoded dataset paths.

## Model Evaluation

The evaluation pipeline executes in three sequential stages: prediction, rendering, and metric computation.

The evaluation pipeline resolves model checkpoints from W&B artifacts to a local cache on first access via a configuration resolver.

The evaluation pipeline supports headless rendering on Linux via auto-detected virtual framebuffers.

The evaluation pipeline uses native display rendering on macOS.

The evaluation pipeline renders both predicted and ground-truth audio for comparison.

The render stage denormalizes predicted parameters before decoding through the parameter specification.

The evaluation pipeline computes four audio distance metrics per sample: multi-scale spectrogram, weighted MFCC, spectral optimal transport, and RMS envelope similarity.

The evaluation pipeline writes per-sample metrics and aggregated statistics to structured output files.

The evaluation pipeline downloads datasets from R2 when a remote path is configured and skips download when absent.

The evaluation pipeline stores detailed metric results as bulk artifacts in R2 and logs summary metrics to W&B.

## Model Promotion

The promotion pipeline creates a GitHub release from a W&B run with an auto-incrementing versioned tag.

The promotion pipeline attaches the model artifact as a release asset.

The promotion pipeline generates a release body containing an evaluation card with metrics, configuration, and dataset artifact references.

The promotion pipeline optionally links the model artifact to the W&B model registry with release and latest aliases.

The promotion pipeline supports a dry-run mode that prints the release body without creating any release.

The promotion pipeline runs as an automated workflow triggered with a required run ID input.

## Artifact Provenance

The pipeline declares artifact inputs and outputs for every W&B run to establish lineage.

The pipeline establishes artifact lineage only through run-scoped APIs, never through global client APIs.

Every W&B run records the source commit SHA, container image tag, and full execution command in its configuration.

Every W&B run sets a job type of data-generation, training, or evaluation.

The pipeline creates all W&B runs through the Lightning logger integration.

The pipeline stores hyperparameters in run configuration, final metrics in run summary, and artifact properties in artifact metadata.

W&B artifacts receive a latest alias on every upload and a best alias when a validation metric improves.

The promotion workflow assigns a production alias to promoted model artifacts.

## Storage Layout

The R2 bucket stores dataset artifacts under a path keyed by dataset configuration ID and dataset run ID.

The R2 bucket stores training artifacts under a path keyed by dataset and training configuration IDs and their run IDs.

The R2 bucket stores evaluation artifacts under a path keyed by dataset, training, and evaluation configuration IDs and their run IDs.

The pipeline references R2 objects from W&B artifacts via S3-compatible reference URIs.

R2 paths are append-only after completion markers exist.

All dataset and log paths use configuration-resolved interpolation rather than absolute local paths.

## Container Environment

The container build uses multi-stage builds with baked-in dependencies for reproducibility.

The container entrypoint requires an execution mode environment variable and errors if it is unset or unknown.

The container supports an idle mode that keeps the container running for interactive debugging.

The container supports a passthrough mode that executes provided arguments directly.

The container supports a generation mode that runs single-shard dataset generation under a headless display server.

The dev-snapshot build target clones source code at a specified git reference and installs all dependencies.

The container build injects credentials via build-time secrets that never appear in image layers or runtime environment variables.

The container falls back to runtime environment variables for credentials when build-time secrets are not provided.

The system allows any compute provider capable of running the container image to execute generation workers.

## Concurrency and Crash Resilience

The pipeline isolates failures per shard so that one crash does not affect other shards or the overall pipeline.

The pipeline applies a centralized retry policy with exponential backoff for transient storage operations and reraises permanent failures immediately.

Concurrent generation invocations for the same run ID produce wasted compute but no data corruption.

Concurrent finalize invocations produce identical outputs due to deterministic shard selection and promotion.

The pipeline produces deterministic shard content given the same seed, configuration, container image, and CPU architecture.

## CLI Interface

The CLI starts dataset generation using a dataset configuration identifier.

The CLI starts dataset finalization for a generated dataset.

The CLI starts model training using an experiment configuration.

The CLI starts evaluation using a trained model checkpoint and dataset.

The CLI reports errors when required configuration inputs are missing.
