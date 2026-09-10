# CLI Command Cookbook

Use these recipes from the repository root after activating the project environment. Console
script names are defined in [`pyproject.toml`](../../pyproject.toml). Replace the uppercase
placeholder values before running a command.

## Dataset URIs

A finalized dataset has one **root URI** and up to three **split Lance URIs**:

```text
r2://BUCKET/data/TASK_NAME/RUN_ID/              # dataset root
r2://BUCKET/data/TASK_NAME/RUN_ID/train.lance   # one split
```

Generation resumes from the root's `input_spec.json`. Finalization and training take the root
URI; embedding augmentation takes one split `.lance` URI. The root layout comes from the
[`r2` config](../../src/synth_setter/configs/r2/default.yaml) and the selected
[dataset experiment](../../src/synth_setter/configs/experiment/generate_dataset/).

## Generate a dataset

Select a checked-in dataset experiment and run it locally:

```bash
DATASET_CONFIG='generate_dataset/CONFIG_NAME'
synth-setter-generate-dataset "experiment=${DATASET_CONFIG}"
```

For example, `generate_dataset/smoke-shard` is the small VST smoke config. The authoritative
inputs are [`dataset.yaml`](../../src/synth_setter/configs/dataset.yaml) and the
[`experiment/generate_dataset` configs](../../src/synth_setter/configs/experiment/generate_dataset/).
The command writes and uploads `input_spec.json` before rendering shards.

To create a dynamically claimed shard queue, set the queue flag on the initial run:

```bash
DATASET_CONFIG='generate_dataset/CONFIG_NAME'
synth-setter-generate-dataset "experiment=${DATASET_CONFIG}" use_shard_queue=true
```

## Join an existing shard queue

Use the already-uploaded spec URI. This command renders in-process and joins dynamic claiming
only when that frozen spec has `use_shard_queue: true`:

```bash
SPEC_URI='r2://BUCKET/data/TASK_NAME/RUN_ID/input_spec.json'
synth-setter-generate-dataset-from-spec-uri "$SPEC_URI"
```

Launch the same command on each additional worker; the claims table assigns work. Do not pass
the dataset root here: this entrypoint requires the `input_spec.json` URI. The authoritative
settings are frozen in that spec, originally composed from
[`dataset.yaml`](../../src/synth_setter/configs/dataset.yaml).

## Finalize a dataset

Pass the **dataset root**, not a split `.lance` URI:

```bash
DATASET_ROOT_URI='r2://BUCKET/data/TASK_NAME/RUN_ID/'
synth-setter-finalize-dataset "dataset_root_uri=${DATASET_ROOT_URI}"
```

Finalization commits the staged fragments into split datasets and writes `dataset.complete`
last. Re-running an already completed root exits cleanly. See
[`finalize_dataset.yaml`](../../src/synth_setter/configs/finalize_dataset.yaml).

## Add embeddings

Pass exactly one finalized **split Lance URI**. Repeat the command for other splits when they
also need the columns.

Generic example:

```bash
SPLIT_LANCE_URI='r2://BUCKET/data/TASK_NAME/RUN_ID/train.lance'
synth-setter-add-embeddings \
  "lance_uri=${SPLIT_LANCE_URI}" \
  'embeddings=[clap,m2l]'
```

pyFDN temporal-sketch example:

```bash
SPLIT_LANCE_URI='r2://BUCKET/data/TASK_NAME/RUN_ID/train.lance'
synth-setter-add-embeddings \
  "lance_uri=${SPLIT_LANCE_URI}" \
  'embeddings=[pyfdn_sketch]' \
  device=cpu \
  num_workers=4
```

The selected columns must not already exist. Registry keys, defaults, batching, indexing, and
resume-cache settings are authoritative in
[`add_embeddings.yaml`](../../src/synth_setter/configs/add_embeddings.yaml) and the
`EMBEDDING_REGISTRY` used by its entrypoint.

## Train

Select a checked-in training experiment. To consume a remote finalized dataset, pass its
**root URI**; the datamodule checks `dataset.complete` and hydrates the required split columns.

```bash
DATASET_ROOT_URI='r2://BUCKET/data/TASK_NAME/RUN_ID/'
synth-setter-train \
  experiment=surge/flow_simple \
  "datamodule.download_dataset_root_uri=${DATASET_ROOT_URI}"
```

Use a compatible experiment for the dataset and its columns; for example,
`experiment=pyfdn/flow_sketch` expects pyFDN sketch conditioning. See
[`train.yaml`](../../src/synth_setter/configs/train.yaml), the
[training experiments](../../src/synth_setter/configs/experiment/), and the selected
[datamodule config](../../src/synth_setter/configs/datamodule/).

For a controlled endpoint-loss A/B run, keep the finalized dataset, one-hot parameter schema,
and seed identical. Enable seeded evaluation so validation comparisons reuse local noise. All four
combinations require endpoint parameterization:

```bash
synth-setter-train experiment=surge/flow_simple seed=12345 model.seeded_evaluation=true model.parameterization=endpoint model.endpoint_loss=mse model.endpoint_time_weighting=uniform
synth-setter-train experiment=surge/flow_simple seed=12345 model.seeded_evaluation=true model.parameterization=endpoint model.endpoint_loss=mse model.endpoint_time_weighting=flowmol3
synth-setter-train experiment=surge/flow_simple seed=12345 model.seeded_evaluation=true model.parameterization=endpoint model.endpoint_loss=mixed model.endpoint_time_weighting=uniform
synth-setter-train experiment=surge/flow_simple seed=12345 model.seeded_evaluation=true model.parameterization=endpoint model.endpoint_loss=mixed model.endpoint_time_weighting=flowmol3
```

These commands define comparable configurations; they do not establish a measured quality
improvement for either objective or weighting.

## Create a synth-parameter W&B workspace

Create a shared workspace whose regex-backed panels discover each synth's parameter names:

```bash
synth-setter-create-wandb-parameter-workspace \
  --entity WANDB_ENTITY \
  --project synth-setter
```

The command prints the saved workspace URL. Its run set has no synth-name filter, so the same
panels cover Surge, pyFDN, TorchSynth, OB-Xf, Faust, Cardinal, and KR-106 runs. It creates a new
saved view each time; retain the printed URL instead of rerunning it for the same project.

## Launch with SkyPilot

Before either RunPod recipe, run the required balance preflight. It fails open when the balance
cannot be queried and stops when the known balance is insufficient:

```bash
uv run python -c "from synth_setter.pipeline.skypilot_launch import _check_runpod_balance; _check_runpod_balance(); print('balance preflight passed')"
```

Dataset generation has an integrated SkyPilot path. It materializes and uploads the spec before
dispatching workers:

```bash
DATASET_CONFIG='generate_dataset/CONFIG_NAME'
IMAGE_TAG='IMAGE_TAG'
JOB_NAME='DATASET_JOB_NAME'
synth-setter-generate-dataset \
  "experiment=${DATASET_CONFIG}" \
  skypilot_launch/compute=runpod/smoke \
  "skypilot_launch.worker_image_tag=${IMAGE_TAG}" \
  "skypilot_launch.job_name=${JOB_NAME}"
```

For an arbitrary worker command, use the generic launcher:

```bash
IMAGE_TAG='IMAGE_TAG'
JOB_NAME='TRAINING_JOB_NAME'
synth-setter-skypilot-launch \
  skypilot_launch/compute=runpod/training \
  "skypilot_launch.worker_image_tag=${IMAGE_TAG}" \
  "skypilot_launch.job_name=${JOB_NAME}" \
  'skypilot_launch.cmd="exec synth-setter-train experiment=torchsynth/flow_audio_same"'
```

The launcher syncs the worker checkout before running the command. Escape worker-side shell
expansions as `\${NAME}` inside `skypilot_launch.cmd`. Compute choices and launcher fields are
authoritative in [`skypilot_launch/default.yaml`](../../src/synth_setter/configs/skypilot_launch/default.yaml)
and the [`skypilot_launch/compute` options](../../src/synth_setter/configs/skypilot_launch/compute/).

## Status and inspection

Detached SkyPilot launches print the exact job-specific log and cancel commands. With an
explicit job name, inspect the managed job using SkyPilot's native CLI:

```bash
JOB_NAME='JOB_NAME'
sky jobs queue
sky jobs logs --name "$JOB_NAME"
```

The launcher has no separate synth-setter status subcommand. Its detached/tail behavior and job
name are controlled by
[`skypilot_launch/default.yaml`](../../src/synth_setter/configs/skypilot_launch/default.yaml).

For dataset objects, `rclone` uses `r2:BUCKET/...`, not the application's `r2://BUCKET/...`
URI spelling:

```bash
RCLONE_DATASET_ROOT='r2:BUCKET/data/TASK_NAME/RUN_ID/'
rclone lsf --checksum --recursive "$RCLONE_DATASET_ROOT"
rclone lsjson --checksum "${RCLONE_DATASET_ROOT}dataset.complete"
```

The second command succeeds only when the marker written last by finalization exists. Object
layout is determined by the selected
[dataset experiment](../../src/synth_setter/configs/experiment/generate_dataset/) and
[`r2/default.yaml`](../../src/synth_setter/configs/r2/default.yaml).
