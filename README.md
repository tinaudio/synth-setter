<div align="center">

# Audio synthesizer inversion in symmetric parameter spaces with approximately equivariant flow matching

This repository accompanies a submission to ISMIR 2025. A full README explaining how to use this code will be provided before the conference. In the meantime, audio examples are available at the [online supplement](https://benhayes.net/synth-perm/).

If you would like to explore the source code, you may find the below helpful:

</div>

```
src/models/components/transformer.py       <- DiT and AST implementations
src/models/components/residual_mlp.py      <- Residual MLP implementations
src/models/components/cnn.py               <- CNN encoder implementations
src/models/components/vae.py               <- VAE+RealNVP baseline implementation
src/models/*_module.py                     <- LightningModule implementations, containing training logic
src/data/vst/*                             <- Dataset generation
src/data/vst/surge_xt_param_spec.py        <- Specification of Surge XT dataset sampling distributions
src/data/ot.py                             <- Optimal transport minibatch coupling
src/data/kosc_datamodule.py                <- Implementation of k-osc task
configs/experiment/kosc                    <- k-osc experiment configs
configs/experiment/surge                   <- Surge XT experiment configs
```

...existing code...

## Setup

1. Install requirements:

   ```bash
   # [OPTIONAL] create conda environment
   conda update --name base conda
   conda env create -f environment.yaml
   conda activate myenv

   # install torch stack and app dependencies separately
   python -m pip install --upgrade pip
   python -m pip install -r requirements-torch.txt
   python -m pip install -r requirements-app.txt

   # backward-compatible one-liner
   python -m pip install -r requirements.txt
   ```

2. Configure Weights & Biases (optional but recommended):

   ```bash
   wandb login
   ```

   - Adjust project/team defaults in [configs/logger/wandb.yaml](configs/logger/wandb.yaml).
   - You can also set `WANDB_ENTITY`, `WANDB_PROJECT`, or run with `logger=wandb`.

## Tests

Run the fast test suite:

```bash
pytest -k "not slow"
```

Run the full suite:

```bash
pytest
```

(You can also use [Makefile](Makefile) targets like `make test` or `make test-full`.)

## Docker

Two Docker workflows are available: a **production build** (self-contained) and a **dev build** (mount local code).
Both use the same Dockerfile (`docker/ubuntu22_04/Dockerfile`) with different build targets.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with BuildKit enabled (default on Docker Desktop ≥ 23.0).
- A GitHub personal access token (`GIT_PAT`) with `repo` read access, because this repository is private.
  Create one at <https://github.com/settings/tokens>.

### Production build (bakes source at a specific commit)

Builds a fully self-contained image you can run anywhere (CI, cloud, vast.ai) without mounting anything.
The repo source is downloaded as a tarball at the exact commit you specify.

```bash
# Build — downloads repo at GIT_REF, builds Surge XT, installs all Python deps
make docker-build-dev-snapshot GIT_REF=<commit-sha> GIT_PAT=<github-pat>

# Run
docker run --rm -it tinaudio/perm:...-dev-snapshot-<commit-sha> python src/train.py
```

> **Tip:** Use a full commit SHA for `GIT_REF` for deterministic, reproducible builds.
> Tags or branch names also work but may resolve to different commits over time.

### Dev build (fast local iteration)

Builds a reusable "environment" image with Surge XT and all Python deps pre-installed,
but **no source code baked in**. You mount your local working tree at runtime so edits
are reflected immediately.

```bash
# Build once (rebuild only when deps or Surge version change)
make docker-build-dev-live GIT_PAT=<github-pat> GIT_REF=<commit-sha>

# Run with your local code mounted (repeat as often as you like)
make docker-run-dev
```

The image is tagged `tinaudio/perm:dev` (latest) and
`tinaudio/perm:...-dev-live-<short-sha>` (pinned to the commit whose dep manifests were used).

Your local code is mounted into the container at `/home/build/synth-permutations`.
To use a GPU, add `--gpus all` to the `docker run` command (or override `DOCKER_BUILD_FLAGS`).

### Makefile variable reference

| Variable                | Default                                                       | Description                                                    |
| ----------------------- | ------------------------------------------------------------- | -------------------------------------------------------------- |
| `GIT_REF`               | _(required)_                                                  | Commit SHA / tag / branch to clone or bake into the image      |
| `GIT_PAT`               | _(required)_                                                  | GitHub PAT with repo read access (passed as a BuildKit secret) |
| `DOCKER_FILE`           | `docker/ubuntu22_04/Dockerfile`                               | Path to the Dockerfile                                         |
| `DOCKER_IMAGE`          | `tinaudio/perm`                                               | Image name                                                     |
| `DOCKER_BASE_IMAGE`     | `vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py310` | Base Docker image                                              |
| `DOCKER_BUILD_MODE`     | `prebuilt`                                                    | `source` (build Surge from source) or `prebuilt` (use .deb)    |
| `DOCKER_TARGETPLATFORM` | `linux/amd64`                                                 | `linux/amd64` or `linux/arm64`                                 |
| `DOCKER_TORCH_IDX`      | `https://download.pytorch.org/whl/cu128`                      | PyTorch wheel index URL                                        |
| `DOCKER_BUILD_FLAGS`    | _(empty)_                                                     | Extra flags passed verbatim to `docker build`                  |
| `USE_CLOUD_BUILDER`     | `false`                                                       | Set to `1` to use the remote cloud builder and push the result |

### Advanced examples

```bash
# Use the remote cloud builder and push the result
make docker-build-dev-snapshot GIT_REF=abc123 GIT_PAT=ghp_xxxx USE_CLOUD_BUILDER=1

# Use a different base image (e.g. plain ubuntu for CPU-only builds)
make docker-build-dev-snapshot GIT_REF=abc123 GIT_PAT=ghp_xxxx \
  DOCKER_BASE_IMAGE=ubuntu:22.04

# Build for arm64
make docker-build-dev-snapshot GIT_REF=abc123 GIT_PAT=ghp_xxxx \
  DOCKER_TARGETPLATFORM=linux/arm64

# Use CPU-only PyTorch wheels (smaller image, no CUDA)
make docker-build-dev-live GIT_REF=abc123 GIT_PAT=ghp_xxxx \
  DOCKER_TORCH_IDX=https://download.pytorch.org/whl/cpu

# Pass extra flags to docker build (e.g. pin a different Surge ref)
make docker-build-dev-snapshot GIT_REF=abc123 GIT_PAT=ghp_xxxx \
  DOCKER_BUILD_FLAGS="--build-arg SURGE_GIT_REF=deadbeef"
```
