# Getting Started

This guide walks you through setting up synth-setter from scratch, running the
test suite, training your first model, and configuring the external dependencies
needed for the full data pipeline.

______________________________________________________________________

## 1. Prerequisites

- **Python 3.10+** (check with `python --version`)
- **Git** (with `--recurse-submodules` support)
- **make** (ships with macOS/Linux; on Windows use WSL)
- **A CUDA GPU** is recommended for training. CPU and MPS (Apple Silicon) trainers
  are available but significantly slower.

______________________________________________________________________

## 2. Installation

### 2a. Clone the repository

```bash
git clone --recurse-submodules https://github.com/tinaudio/synth-setter.git
cd synth-setter
```

If you already cloned without `--recurse-submodules`, initialize the submodules:

```bash
git submodule update --init
```

### 2b. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 2c. Install dependencies

The project uses [uv](https://docs.astral.sh/uv/getting-started/installation/)
for fast dependency resolution:

```bash
pip install uv
uv pip install -r requirements.txt -e .
```

Or with plain pip:

```bash
pip install -r requirements.txt -e .
```

### 2d. Install pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

The hooks run Ruff (linting + formatting), pyright (type checking), mdformat,
codespell, and several other checks automatically on each commit.

### 2e. Verify the installation

```bash
make test
```

This runs the quick test suite (excluding slow tests and tests that require a
VST plugin). All tests should pass. If you see import errors, double-check that
the virtual environment is active and dependencies installed correctly.

______________________________________________________________________

## 3. k-osc Quickstart (No External Dependencies)

The k-osc task is a synthetic benchmark where the model learns to predict
parameters of a sum-of-sinusoids signal. It generates data on the fly, so you
do not need any external datasets, VST plugins, or cloud storage.

### 3a. Train a model

```bash
python src/train.py +experiment=kosc/ffn_mse trainer.max_epochs=5
```

This runs a feed-forward network with MSE loss on the k-osc task for 5 epochs.
You should see Lightning's progress bar with decreasing loss values.

**What happens:**

- Hydra composes the config from `configs/train.yaml` + the experiment override
- Lightning sets up the data module, model, callbacks, and trainer
- Checkpoints are saved under `logs/train/runs/<timestamp>/checkpoints/`
- If W&B is configured (see [section 4c](#4c-weights--biases-wb)), metrics are
  logged to your dashboard

### 3b. Available k-osc experiments

The `configs/experiment/kosc/` directory contains several variants:

| Config | Description |
| --- | --- |
| `kosc/base` | Base config (used by other variants) |
| `kosc/ffn_mse` | Feed-forward network, MSE loss |
| `kosc/ffn_chamfer` | Feed-forward network, Chamfer loss |
| `kosc/flow` | Flow matching model |
| `kosc/flow_asym` | Flow matching, asymmetric |
| `kosc/flowmlp` | Flow MLP variant |

Run any of them with:

```bash
python src/train.py +experiment=kosc/<variant>
```

______________________________________________________________________

## 4. External Dependencies

The sections below cover dependencies needed for the full workflow: generating
audio datasets from VST plugins, syncing data with cloud storage, and tracking
experiments. **None of these are needed for the k-osc quickstart above.**

### 4a. Surge XT (VST Plugin)

[Surge XT](https://surge-synthesizer.github.io/) is the open-source synthesizer
used for audio dataset generation. The data pipeline renders audio by
programmatically driving this plugin.

**Install:**

1. Download the installer from [surge-synthesizer.github.io](https://surge-synthesizer.github.io/)
1. Run the installer and follow the prompts
1. Note the installation path (the test suite needs to find the plugin binary)

**Verify:**

```bash
pytest -m requires_vst -v
```

If the plugin is found, VST-dependent tests will run. If not, they are
automatically skipped (they are excluded from `make test`).

### 4b. rclone + Cloudflare R2

[rclone](https://rclone.org/) is used for all interactions with Cloudflare R2
object storage, where pipeline data (shards, specs, metadata) is stored. All
rclone operations in this project use `--checksum` for integrity.

**Install rclone:**

```bash
# macOS
brew install rclone

# Linux
curl https://rclone.org/install.sh | sudo bash

# Or see https://rclone.org/install/
```

**Configure the R2 remote:**

You need R2 credentials (access key ID, secret access key, and endpoint URL)
from a project maintainer or your Cloudflare dashboard.

```bash
rclone config
```

Follow the prompts to create a new remote named `r2` with provider
`Cloudflare R2` (or `S3` with the R2 endpoint). Alternatively, set these
environment variables in your `.env` file:

```
R2_ACCESS_KEY_ID=<your-access-key>
R2_SECRET_ACCESS_KEY=<your-secret-key>
R2_ENDPOINT=<your-r2-endpoint-url>
R2_BUCKET=<bucket-name>
```

**Verify:**

```bash
rclone ls r2:<bucket-name>/ --max-depth 1
```

You should see top-level directories like `data/` and `metadata/`.

### 4c. Weights & Biases (W&B)

[Weights & Biases](https://wandb.ai/) is used for experiment tracking, metric
logging, and model checkpoint storage. The integration is handled through
Lightning's `WandbLogger` -- there are no direct `wandb.init()` calls in the
codebase.

**Setup:**

1. Create an account at [wandb.ai](https://wandb.ai/)
1. Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize)
1. Log in:

```bash
wandb login
```

Or set the environment variable in your `.env`:

```
WANDB_API_KEY=<your-api-key>
```

**Optional overrides** (defaults are usually fine):

```
WANDB_ENTITY=tinaudio        # W&B team name
WANDB_PROJECT=synth-setter   # W&B project name
```

To train **without W&B** (e.g., for local experimentation), override the logger:

```bash
python src/train.py +experiment=kosc/ffn_mse logger=csv
```

This logs metrics to CSV files instead.

For full details, see [docs/reference/wandb-integration.md](reference/wandb-integration.md).

### 4d. RunPod (Optional -- Distributed Generation)

[RunPod](https://www.runpod.io/) is used for distributed dataset generation --
spinning up multiple GPU workers to render audio in parallel. **You do not need
RunPod for local development or training.**

If you are working on the data pipeline and need to run distributed generation:

1. Create a RunPod account at [runpod.io](https://www.runpod.io/)
1. Generate an API key from the RunPod dashboard
1. Set the environment variable:

```
RUNPOD_API_KEY=<your-api-key>
```

______________________________________________________________________

## 5. Hydra Configuration System

synth-setter uses [Hydra](https://hydra.cc/) for configuration management. The
config is composed from multiple YAML files that layer on top of each other.

### 5a. Config structure

```
configs/
  train.yaml          # Top-level training defaults
  eval.yaml           # Top-level evaluation defaults
  data/               # Data module configs (kosc, ksin, surge, ...)
  model/              # Model configs (ffn, flow, flowmlp, ...)
  trainer/            # Trainer configs (gpu, cpu, mps, ddp, ...)
  logger/             # Logger configs (wandb, csv, tensorboard, ...)
  callbacks/          # Callback configs
  experiment/         # Experiment configs (compose data + model + overrides)
    kosc/             # k-osc experiments
    surge/            # Surge XT experiments
  dataset/            # Pipeline dataset configs
```

### 5b. Common overrides

Override any config value from the command line:

```bash
# Change batch size
python src/train.py +experiment=kosc/ffn_mse data.batch_size=32

# Change learning rate
python src/train.py +experiment=kosc/ffn_mse model.lr=1e-4

# Use CPU trainer instead of GPU
python src/train.py +experiment=kosc/ffn_mse trainer=cpu

# Use TensorBoard logger instead of W&B
python src/train.py +experiment=kosc/ffn_mse logger=tensorboard

# Limit training epochs
python src/train.py +experiment=kosc/ffn_mse trainer.max_epochs=10

# Run in debug mode (1 batch per epoch, no logging)
python src/train.py +experiment=kosc/ffn_mse debug=default
```

For the full configuration reference, see
[docs/reference/configuration-reference.md](reference/configuration-reference.md).

______________________________________________________________________

## 6. Evaluation

After training, evaluate the model on the test set:

```bash
python src/eval.py
```

By default, evaluation uses the config and checkpoint from the most recent
training run. You can point it at a specific checkpoint:

```bash
python src/eval.py ckpt_path=/path/to/checkpoint.ckpt
```

______________________________________________________________________

## 7. Docker Workflow

A Dockerfile is provided for reproducible environments (training, CI, cloud
deployment). The image bakes in the source code, dependencies, Surge XT, and
rclone/R2 configuration.

**Build the image:**

```bash
make docker-build-dev-snapshot \
  GIT_REF=$(git rev-parse HEAD) \
  GIT_PAT=<your-github-pat> \
  R2_ACCESS_KEY_ID=<key> \
  R2_SECRET_ACCESS_KEY=<secret> \
  R2_ENDPOINT=<endpoint> \
  R2_BUCKET=<bucket>
```

See `make help` for the full list of Docker-related variables and targets. The
`GIT_REF` argument controls which commit is baked into the image (use a full
SHA for reproducibility).

For full Docker documentation, see
[docs/reference/docker.md](reference/docker.md).

______________________________________________________________________

## 8. Troubleshooting

### Tests fail with import errors

Make sure your virtual environment is active and dependencies are installed:

```bash
source .venv/bin/activate
uv pip install -r requirements.txt -e .
```

### `make format` fails on first run

Pre-commit downloads hook environments on first execution. If it fails, try:

```bash
pre-commit clean
pre-commit install
make format
```

### CUDA out of memory

Reduce the batch size:

```bash
python src/train.py +experiment=kosc/ffn_mse data.batch_size=8
```

Or switch to CPU for debugging:

```bash
python src/train.py +experiment=kosc/ffn_mse trainer=cpu
```

### W&B login issues

If `wandb login` does not persist, set the API key as an environment variable:

```bash
export WANDB_API_KEY=<your-api-key>
```

Or add it to your `.env` file (never commit this file).

### VST tests are skipped

This is expected. VST tests require Surge XT to be installed (see
[section 4a](#4a-surge-xt-vst-plugin)). They are excluded from `make test` by
default.

______________________________________________________________________

## 9. What to Try Next

- **Experiment configs:** Browse `configs/experiment/` for pre-configured
  experiments across different models and datasets.
- **Data generation:** See `pipeline/entrypoints/generate_dataset.py` for the
  dataset generation entry point.
- **Design docs:** Read `docs/design/data-pipeline.md` for the data pipeline
  architecture and `docs/design/training-pipeline.md` for the training pipeline.
- **Configuration reference:**
  [docs/reference/configuration-reference.md](reference/configuration-reference.md)
  covers all config layers in detail.
- **Available make targets:** Run `make help` to see all commands.
