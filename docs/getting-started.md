# Getting Started

This guide walks you through setting up synth-setter from scratch, running the
test suite, training your first model, and configuring the external dependencies
needed for the full data pipeline.

______________________________________________________________________

## 1. Prerequisites

- **Linux or macOS** — Windows is not supported (see the project README).
- **Python 3.10+** (check with `python --version`)
- **Git**
- **make** (ships with macOS/Linux)
- **A CUDA GPU** is recommended for training. CPU and MPS (Apple Silicon) trainers
  are available but significantly slower.

______________________________________________________________________

## 2. Installation

### 2a. Clone the repository

```bash
git clone https://github.com/tinaudio/synth-setter.git
cd synth-setter
```

### 2b. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
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

### 2f. Alternative: GitHub Codespaces

Instead of setting up locally, you can open the repo in a GitHub Codespace. The
Codespace uses the same Docker image (`tinaudio/synth-setter:dev-snapshot`) we run on
RunPod, so VST-dependent tests, `generate_dataset` → R2 uploads, and CPU
training all work identically — no local Surge XT, rclone, or R2 setup needed.
GPU training still runs on RunPod.

**Prerequisites:**

Configure these as Codespaces user/org secrets so they're forwarded into
the container at runtime:

- `RCLONE_CONFIG_R2_ACCESS_KEY_ID`, `RCLONE_CONFIG_R2_SECRET_ACCESS_KEY`,
  `RCLONE_CONFIG_R2_ENDPOINT` — for R2 uploads/downloads via rclone
- `WANDB_API_KEY` — for W&B logging

The image itself is public and pulls anonymously; no Docker Hub
credentials are required.

**Open a Codespace:**

1. On GitHub, click **Code → Codespaces → Create codespace on main**.
2. First start takes ~5 min (image pull). Subsequent starts are fast.
3. `.devcontainer/post-create.sh` configures git safety settings,
   optionally authenticates with `RESTRICTED_AGENT_GIT_PAT`, and installs
   pre-commit hooks. If invoked as root (Codespaces default, or opt-in
   `DEVCONTAINER_USER=root` locally), it drops to the `dev` user first so
   workspace mutations under `.git/` land with dev ownership. Then the
   terminal is ready.

**Verify:**

```bash
make test
python -c "import torch; print(torch.cuda.is_available())"   # False (CPU)
rclone lsd r2:intermediate-data
```

**Fall back to RunPod for** full GPU training (CUDA kernels, large batch
sizes) and multi-hour runs.

### 2g. Alternative: Local Dev Container

If you want the same image Codespaces uses (VST plugins, rclone, Python
deps) but prefer to work on your own machine, open the dev container
locally **on the main working tree** and create git worktrees *inside* the
container.

**Prerequisites:**

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) and
  either the VS Code
  [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
  or the [`devcontainer` CLI](https://github.com/devcontainers/cli).
- R2 + W&B credentials (see [§4b](#4b-rclone--cloudflare-r2) and
  [§4c](#4c-weights--biases-wb)). The checked-in dev container configs
  do **not** automatically load `.env` — after opening the container,
  source the vars inside the container shell (`set -a && source .env && set +a`)
  or set them via Codespaces secrets / Dev Container environment settings
  so rclone and W&B can access them.
- Apple Silicon: set `DOCKER_DEFAULT_PLATFORM=linux/amd64` (the image is
  amd64-only).

**Open the container on main:**

```bash
cd /path/to/synth-setter
devcontainer up --workspace-folder .
# or in VS Code: "Dev Containers: Reopen in Container"
```

**Create worktrees from inside the container:**

```bash
# Inside the container, starting from the workspace root on main:
git worktree add .claude/worktrees/my-feature -b feat/my-feature
cd .claude/worktrees/my-feature
```

This is the supported local pattern. Mounting a worktree directly from the
host does not work — the worktree's `.git` file points to
`<repo>/.git/worktrees/<name>/` on the host, which is outside the container's
bind mount, so git submodule/hook operations fail to resolve their gitdir.

**Caveats:**

- `git worktree list` inside the container marks host-created worktrees as
  `prunable` (their host paths don't resolve inside the mount). Do **not**
  run `git worktree prune` inside the container — it will drop registry
  entries for worktrees that are still valid on the host.
- `git submodule update` for the private `tinaudio/skills` submodule needs
  GitHub credentials available to git inside the container. VS Code's git
  credential helper usually forwards these automatically. For the CLI, run
  `gh auth login && gh auth setup-git` inside the container, or configure
  git's credential helper with a PAT. Exporting `GITHUB_TOKEN` alone is not
  sufficient — git will not use it without a credential helper configured.

______________________________________________________________________

## 3. k-osc Quickstart (No External Dependencies)

The k-osc task is a synthetic benchmark where the model learns to predict
parameters of a sum-of-sinusoids signal. It generates data on the fly, so you
do not need any external datasets, VST plugins, or cloud storage.

### 3a. Train a model

```bash
python src/train.py experiment=kosc/ffn_mse trainer.max_steps=5000 trainer.min_steps=null
```

> **No CUDA GPU?** The default trainer is `gpu` (CUDA). On CPU-only machines use
> `trainer=cpu`; on Apple Silicon use `trainer=mps`:
>
> ```bash
> python src/train.py experiment=kosc/ffn_mse trainer=cpu trainer.max_steps=5000 trainer.min_steps=null
> ```

This runs a feed-forward network with MSE loss on the k-osc task for 5,000
training steps. The `trainer.min_steps=null` override is needed because the
default trainer config sets `min_steps: 400_000`, which would otherwise prevent
the run from stopping at 5,000 steps. You should see Lightning's progress bar
with decreasing loss values.

**What happens:**

- Hydra composes the config from `configs/train.yaml` + the experiment override
- Lightning sets up the data module, model, callbacks, and trainer
- Checkpoints are saved under `logs/{task_name}/{experiment_name}/{run_name}-{timestamp}/checkpoints/` (for this command: `logs/train/kosc/ffn_mse-<timestamp>/checkpoints/`)
- If W&B is configured (see [section 4c](#4c-weights--biases-wb)), metrics are
  logged to your dashboard

### 3b. Available k-osc experiments

The `configs/experiment/kosc/` directory contains several variants:

| Config             | Description                          |
| ------------------ | ------------------------------------ |
| `kosc/base`        | Base config (used by other variants) |
| `kosc/ffn_mse`     | Feed-forward network, MSE loss       |
| `kosc/ffn_chamfer` | Feed-forward network, Chamfer loss   |
| `kosc/flow`        | Flow matching model                  |
| `kosc/flow_asym`   | Flow matching, asymmetric            |
| `kosc/flowmlp`     | Flow MLP variant                     |

Run any of them with:

```bash
python src/train.py experiment=kosc/<variant>
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
2. Run the installer and follow the prompts
3. Note the installation path (the test suite needs to find the plugin binary)

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
environment variables in your `.env` file so rclone can auto-configure
the `r2` remote — and so `docker run --env-file .env` will work out of
the box for the synth-setter image. This is the canonical `.env`
template:

```
# --- rclone (R2) remote definition: type/provider are constants ---
RCLONE_CONFIG_R2_TYPE=s3
RCLONE_CONFIG_R2_PROVIDER=Cloudflare
# --- R2 credentials (secrets) ---
RCLONE_CONFIG_R2_ACCESS_KEY_ID=<your-access-key>
RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=<your-secret-key>
RCLONE_CONFIG_R2_ENDPOINT=<your-r2-endpoint-url>
# --- Target bucket name (read by pipeline entrypoints) ---
R2_BUCKET=<bucket-name>
# --- W&B logging ---
WANDB_API_KEY=<your-wandb-api-key>
```

rclone's native env-var auto-config synthesizes the `r2` remote in-memory
from the 5 `RCLONE_CONFIG_R2_*` vars each time you invoke `rclone` (locally
or inside the container). No `rclone.conf` file is written. See
[docs/reference/docker.md § Runtime environment variables](reference/docker.md#runtime-environment-variables)
for the canonical enumeration of every var the image expects at runtime.

The Docker build itself requires no credentials or secrets: the repo is
public, so source is fetched anonymously at build time.

**Verify:**

```bash
rclone lsd r2:<bucket-name>/
```

You should see top-level directories like `data/`, `train/`, and `eval/`.

### 4c. Weights & Biases (W&B)

[Weights & Biases](https://wandb.ai/) is used for experiment tracking, metric
logging, and model checkpoint storage. The integration is handled through
Lightning's `WandbLogger` -- there are no direct `wandb.init()` calls in the
codebase.

**Setup:**

1. Create an account at [wandb.ai](https://wandb.ai/)
2. Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize)
3. Log in:

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
python src/train.py experiment=kosc/ffn_mse logger=csv
```

This logs metrics to CSV files instead.

For full details, see [docs/reference/wandb-integration.md](reference/wandb-integration.md).

### 4d. RunPod (Optional -- Distributed Generation)

[RunPod](https://www.runpod.io/) is used for distributed dataset generation --
spinning up multiple GPU workers to render audio in parallel. **You do not need
RunPod for local development or training.**

If you are working on the data pipeline and need to run distributed generation:

1. Create a RunPod account at [runpod.io](https://www.runpod.io/)
2. Generate an API key from the RunPod dashboard
3. Set the environment variable:

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
python src/train.py experiment=kosc/ffn_mse data.batch_size=32

# Change learning rate
python src/train.py experiment=kosc/ffn_mse model.optimizer.lr=1e-4

# Use CPU trainer instead of GPU
python src/train.py experiment=kosc/ffn_mse trainer=cpu

# Use TensorBoard logger instead of W&B
python src/train.py experiment=kosc/ffn_mse logger=tensorboard

# Limit training steps
python src/train.py experiment=kosc/ffn_mse trainer.max_steps=10000

# Run in debug mode (1 batch per epoch, no logging)
python src/train.py experiment=kosc/ffn_mse debug=default
```

For the full configuration reference, see
[docs/reference/configuration-reference.md](reference/configuration-reference.md).

______________________________________________________________________

## 6. Evaluation

After training, evaluate the model on the test set. You must provide the
checkpoint path (`ckpt_path` is required):

```bash
python src/eval.py ckpt_path=/path/to/checkpoint.ckpt
```

Use the checkpoint saved during training (see the checkpoint path in
[section 3a](#3a-train-a-model)).

______________________________________________________________________

## 7. Docker Workflow

A Dockerfile is provided for reproducible environments (training, CI, cloud
deployment). The image bakes in the source code, dependencies, and Surge XT.
No credentials — R2, W&B, or otherwise — are baked in.

**Build the image:**

The build takes no credentials at all. The repo is public, so source is
fetched anonymously.

> **Note:** The image is public and ships no baked credentials. R2 + W&B
> creds and the target R2 bucket name flow in at runtime via
> `docker run --env-file .env` — see
> [docs/reference/docker.md § Runtime secrets](reference/docker.md#runtime-secrets).

```bash
make docker-build-dev-snapshot \
  GIT_REF=$(git rev-parse HEAD) \
  DOCKER_BUILD_FLAGS=--load
```

The only required input is `GIT_REF` — the commit to bake into the image.
All R2 + W&B credentials and `R2_BUCKET` are supplied to `docker run` from
`.env` at runtime (see [section 4b](#4b-rclone--cloudflare-r2)).

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
python src/train.py experiment=kosc/ffn_mse data.batch_size=8
```

Or switch to CPU for debugging:

```bash
python src/train.py experiment=kosc/ffn_mse trainer=cpu
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
