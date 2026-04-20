# Getting Started

This guide walks you through setting up synth-setter from scratch, running the
test suite, training your first model, and configuring the external dependencies
needed for the full data pipeline.

______________________________________________________________________

## 1. Prerequisites

- **Linux (x86_64) or macOS** — Windows is not supported (see the project README).
- **Git**, **curl**, **make** — standard on most macOS and Linux developer
  machines, but not guaranteed on minimal/server images. Install via your
  package manager (`apt`, `brew`, etc.) if missing.
- **A CUDA GPU** is recommended for training. CPU and MPS (Apple Silicon) trainers
  are available but significantly slower.

`make install` installs [uv](https://docs.astral.sh/uv/) and a managed
Python 3.10 interpreter for you — you do not need to install Python
yourself. If you prefer to manage the interpreter and venv manually, see
[Appendix A](#appendix-a-manual-environment-setup).

______________________________________________________________________

## 2. Installation

### 2a. Clone the repository

```bash
git clone https://github.com/tinaudio/synth-setter.git
cd synth-setter
```

### 2b. Install

`make install` is the canonical end-to-end install. It:

1. Installs [uv](https://docs.astral.sh/uv/) if it is not already on your PATH.
2. Creates `.venv/` using a managed Python 3.10 interpreter (downloaded by uv
   if you do not have one locally). The venv prompt label is `synth-setter`.
3. Installs everything in `requirements.txt` plus the project itself in
   editable mode (`pip install -e .`).
4. Registers the pre-commit hooks — **unless** `git config core.hooksPath`
   is set (as it is in the dev container, where hooks are managed by the
   image). In that case `make install` prints a skip note and leaves the
   configured hooks path untouched; run `.venv/bin/pre-commit install`
   manually if you want to override.

```bash
make install
```

Re-running `make install` is safe: it reuses `.venv/` if it already exists and
is Python 3.10, and refreshes the installed packages. If `.venv/` exists with a
different Python version, `make install` errors and asks you to remove it
first.

The pre-commit hooks run Ruff (linting + formatting), pyright (type checking),
mdformat, codespell, and several other checks automatically on each commit.

> **Prefer pip or conda?** See
> [Appendix A](#appendix-a-manual-environment-setup) for a
> walkthrough using your own Python interpreter and environment tooling.

### 2c. Activate the venv

```bash
source .venv/bin/activate
```

Your prompt should change to `(synth-setter)`. All subsequent commands in this
guide assume the venv is active.

### 2d. Install the Surge XT VST3

The test suite and data pipeline need the [Surge XT](https://surge-synthesizer.github.io/)
VST3 at `plugins/Surge XT.vst3`. `make install-surge-xt` downloads the pinned
release directly from GitHub:

```bash
make install-surge-xt
```

This downloads the `pluginsonly` archive for your platform (Linux x86_64 or
macOS universal) from the [Surge XT 1.3.4 release](https://github.com/surge-synthesizer/releases-xt/releases/tag/1.3.4),
verifies its md5 checksum, and extracts `Surge XT.vst3` into `plugins/`. The
archive is cached at `~/.cache/synth-setter/surge-xt-1.3.4/`, so re-runs that
have to re-extract (e.g. after `rm -rf plugins/`) skip the download. If
`plugins/Surge XT.vst3` already exists, the target is a no-op — remove it
first to reinstall.

> **Already have Surge XT installed system-wide?** Skip
> `make install-surge-xt` and symlink your existing install into `plugins/`:
>
> ```bash
> # Linux
> ln -s "/usr/lib/vst3/Surge XT.vst3" "plugins/Surge XT.vst3"
>
> # macOS
> ln -s "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3" "plugins/Surge XT.vst3"
> ```
>
> The project used to ship a `make link-plugins` wrapper for this; it was
> removed in favour of this one-line symlink so the discovery path stays
> explicit and there's only one way to populate `plugins/`.
>
> **On arm64 Linux?** The official Surge XT release only ships an x86_64
> Linux build. Install via your package manager (`apt install surge-xt`) or
> build from source, then use the manual symlink above.

> **Pointing the VST tests at a non-default install:** `pytest -m requires_vst`
> resolves the plugin at `plugins/Surge XT.vst3` by default. If your install
> lives elsewhere, set `SYNTH_SETTER_PLUGIN_PATH` to the absolute path of the
> `.vst3` bundle before invoking pytest.

### 2e. Export environment variables

The project reads R2 credentials, W&B keys, and other config from a `.env` file.
After creating your `.env` (see [section 4b](#4b-rclone--cloudflare-r2) for the
template), export the variables into your shell:

```bash
set -a && source .env && set +a
```

> Environment variable management is being consolidated under
> [#563](https://github.com/tinaudio/synth-setter/issues/563).

### 2f. Verify the installation

```bash
make test
```

This runs the quick test suite (excluding slow tests and tests that require a
VST plugin). All tests should pass. If you see import errors, double-check that
the virtual environment is active and dependencies installed correctly.

> **Prefer a container- or VM-based setup?** GitHub Codespaces, the local
> dev container, and the Tart macOS VM all come with Python, Surge XT,
> and rclone pre-installed. See
> [Appendix B: Container-based setup](#appendix-b-container-based-setup).

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
- Metrics are logged locally to CSV + TensorBoard by default. If you opt into
  W&B (see [section 4c](#4c-weights--biases-wb)), metrics also go to your W&B
  dashboard

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

Installation is covered in [section 2d](#2d-install-the-surge-xt-vst3) —
`make install-surge-xt` is the canonical path; a manual symlink from a
system-wide install is an alternative.

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

### 4c. Weights & Biases (W&B) — opt-in

[Weights & Biases](https://wandb.ai/) provides experiment tracking, metric
logging, and model checkpoint storage. **It is opt-in.** The default training
logger (`configs/logger/many_loggers.yaml`) composes CSV + TensorBoard, so
`python src/train.py ...` works out of the box with no W&B account and no
`wandb login` prompt. The integration is handled through Lightning's
`WandbLogger` -- there are no direct `wandb.init()` calls in the codebase.

**Disabled (default):** do nothing. Fresh runs log to CSV + TensorBoard only.
No W&B account is required.

**Enabled — per-run override (recommended):**

1. Create an account at [wandb.ai](https://wandb.ai/).

2. Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize).

3. Log in once, or set `WANDB_API_KEY` in your `.env`:

   ```bash
   wandb login
   ```

   ```
   WANDB_API_KEY=<your-api-key>
   ```

4. Pass `logger=wandb` on the command line:

   ```bash
   python src/train.py experiment=kosc/ffn_mse logger=wandb
   ```

   This replaces the default logger composition with W&B only. To log to
   **both** CSV/TensorBoard **and** W&B, uncomment the `- wandb` line in
   `configs/logger/many_loggers.yaml`.

**Enabled — as the default:** edit `configs/logger/many_loggers.yaml` and
uncomment `- wandb`, or change the `logger:` default in `configs/train.yaml`
to `wandb`.

**Optional entity / project overrides:**

```
# WANDB_ENTITY=<your-wandb-team>   # leave unset to use your W&B default entity
WANDB_PROJECT=synth-setter         # W&B project name (default: synth-setter)
```

`WANDB_ENTITY` is unset by default — W&B falls back to the user's default
entity. Set it only if you want to push runs to a specific team.

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

# Enable W&B logging (opt-in; default logger is CSV + TensorBoard)
python src/train.py experiment=kosc/ffn_mse logger=wandb

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

______________________________________________________________________

## Appendix A: Manual environment setup

`make install` is the canonical path for most users — it installs uv, a
managed Python 3.10 interpreter, the venv, dependencies, and pre-commit.
This appendix is for users who want to manage Python and the environment
themselves (pip, conda, pyenv, system Python, etc.).

**Requirement:** Python 3.10 or newer (the project declares
`requires-python = ">=3.10"` in `pyproject.toml`; `pip` enforces this).

### A.1. Plain pip + venv

```bash
# Use any Python 3.10+ interpreter
python3.10 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
pre-commit install
```

Drop `-e` on the second line for a non-editable install.

### A.2. conda

```bash
conda create -n synth-setter python=3.10
conda activate synth-setter

pip install -r requirements.txt
pip install -e .
pre-commit install
```

`requirements.txt` contains pip-only packages (torch, lightning, hydra-core,
etc.), so we install them with pip inside the conda environment rather than
through conda-forge.

### A.3. uv pip without `make install`

If you want to drive uv directly (e.g., to point at a specific interpreter
you manage yourself):

```bash
uv venv --python 3.10 --prompt synth-setter .venv
source .venv/bin/activate
uv pip install -r requirements.txt -e .
pre-commit install
```

This is what `make install` does under the hood.

### A.4. GPU vs CPU PyTorch

`requirements.txt` pins `torch>=2.0.0` without fixing the CPU/CUDA build.
After installing requirements, override with the wheel you want from the
[PyTorch install matrix](https://pytorch.org/get-started/locally/):

```bash
# Example: CUDA 12.1 wheel
pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

`make install` inherits the same CPU/CUDA choice — it does not pick a wheel
for you.

______________________________________________________________________

## Appendix B: Container-based setup

The canonical local flow in [section 2](#2-installation) works on any
POSIX machine. If you'd rather skip installing Surge XT, rclone, and
Python deps yourself, the project ships three container/VM images with
all of these pre-installed.

### B.1. GitHub Codespaces

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

### B.2. Local dev container

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
  [§4c](#4c-weights--biases-wb)). Local dev containers load `.env`
  automatically via `--env-file .env` (`.devcontainer/initialize.sh`
  creates an empty one if missing), so rclone and W&B pick up creds
  on container start. **Codespaces** does not have a host `.env` —
  forward R2/W&B vars via Codespaces user/org secrets instead.
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
bind mount, so git hook operations (and anything else needing the gitdir)
fail to resolve. `.devcontainer/initialize.sh` detects this case on the host
and aborts the build with a clear error before the container is created, so
the failure surfaces immediately rather than partway through `post-create`.

**Caveats:**

- `git worktree list` inside the container marks host-created worktrees as
  `prunable` (their host paths don't resolve inside the mount). Do **not**
  run `git worktree prune` inside the container — it will drop registry
  entries for worktrees that are still valid on the host.
- `plugins/` inside the container is an anonymous Docker volume seeded
  from the base image, which ships `plugins/Surge XT.vst3` as a symlink
  to `/usr/lib/vst3/Surge XT.vst3`. Without this overlay, the host
  workspace bind mount would shadow the baked symlink and break
  VST-dependent tests. Host edits under `plugins/` are not visible
  inside the container; container edits under `plugins/` are not
  visible on the host.

### B.3. macOS VM (Tart)

If you want full dev parity on Apple Silicon inside a throwaway, mostly
reproducible VM — Python 3.10 venv, Surge XT (native .vst3 via cask), Claude
Code installed, auto-activated venv — pull the prebuilt Tart image published
at `registry-1.docker.io/tinaudio/synth-setter-macos`. Rebuilds from the template are not
fully pinned: Homebrew formulas/casks may resolve to newer versions over time,
even if you pin the base image digest and git SHA.

**Prerequisites:**

- Apple Silicon Mac (M1 or later)
- [Homebrew](https://brew.sh/)

**Pull and run the prebuilt image (recommended):**

```bash
brew install cirruslabs/cli/tart
tart clone registry-1.docker.io/tinaudio/synth-setter-macos:latest synth-setter-macos
tart run synth-setter-macos                       # opens a GUI window
ssh admin@$(tart ip synth-setter-macos)           # password: admin
```

> **Security note:** the VM inherits the cirruslabs base image's well-known
> `admin`/`admin` credentials. Treat it as a local-only dev VM. On a shared or
> untrusted network, change the password in the GUI on first boot, or add an
> SSH key to `~admin/.ssh/authorized_keys` and disable `PasswordAuthentication`
> in `/etc/ssh/sshd_config` before exposing port 22.

The image ships with the repo cloned at `~/synth-setter`, a venv with all
`requirements.txt` deps (CPU torch wheels — Tart VMs have no GPU), Surge XT
at `/Library/Audio/Plug-Ins/VST3/Surge XT.vst3`, and
`source ~/synth-setter/.venv/bin/activate` appended to `~/.zshrc` so every
interactive shell has the venv active from login.

Credentials for Claude Code, `gh`, R2, and W&B are **not** baked in — log in
on first boot.

**Build the image yourself (advanced):**

If you need a custom build (pinned repo ref, different torch backend, pinned
base image, updated `uv`, updated Surge XT, etc.), the Packer template at
[`tart/macos.pkr.hcl`](../tart/macos.pkr.hcl) builds the same image locally.
See the bottom of the file for the full publishing workflow to Docker Hub.
The template's `variable` blocks are the authoritative source for supported
overrides. User-overridable packer vars: `synth_setter_git_ref` (default
`main`), `torch_backend` (default `cpu`), `python_version` (default `3.10`),
`vm_name` (default `synth-setter-macos`), `base_image_digest`, `uv_version`,
and `surge_xt_version`.

```bash
brew install cirruslabs/cli/tart packer
packer init tart/macos.pkr.hcl
packer build -var "synth_setter_git_ref=$(git rev-parse HEAD)" tart/macos.pkr.hcl
```
