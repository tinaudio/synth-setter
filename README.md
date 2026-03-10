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

## R2 upload (non-Docker)

When running `generate_shards.py` outside Docker (e.g. on your local machine),
you need [rclone](https://rclone.org/) installed and configured to upload
shards to Cloudflare R2.

1. **Install rclone:**

   ```bash
   # macOS
   brew install rclone

   # Linux (Debian/Ubuntu)
   sudo apt install rclone
   ```

2. **Configure rclone for R2:**

   ```bash
   # Load your R2 credentials into the shell
   set -a && source .env && set +a

   # Run the setup script (creates an rclone remote named "r2")
   bash scripts/setup-rclone.sh
   ```

   This reads `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, and `R2_ENDPOINT`
   from the environment (see [.env.example](.env.example) for how to obtain these).

3. **Verify the connection:**

   ```bash
   rclone lsd r2:$R2_BUCKET
   ```

4. **Generate shards with upload:**

   ```bash
   python scripts/generate_shards.py \
     --num-shards 12 --shard-size 10000 \
     --output-dir data/surge_simple --param-spec surge_simple \
     --r2-bucket "$R2_BUCKET" --r2-prefix "runs/my-run"
   ```

   Use `--local` to skip R2 upload entirely (no rclone needed):

   ```bash
   python scripts/generate_shards.py \
     --num-shards 12 --shard-size 10000 \
     --output-dir data/surge_simple --param-spec surge_simple \
     --local
   ```

> **Note:** Docker images have rclone pre-configured at build time — this
> setup is only needed for local (non-Docker) workflows.

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
- Cloudflare R2 API credentials (`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT`, `R2_BUCKET`)
  for dataset upload/download. See `.env.example` for details on how to obtain these.

> **⚠️ Security: All images contain baked R2 credentials**
>
> Cloudflare R2 credentials are injected at build time using BuildKit `--secret` (so they **do not**
> appear in image layers or `docker history`), but the resulting rclone config file **is** stored inside
> the image filesystem at `/root/.config/rclone/rclone.conf`.
>
> - **Push ALL images to a PRIVATE registry only.** Anyone who can pull the image has read/write
>   access to the configured R2 bucket.
> - **Rotate R2 API tokens after each build campaign.** Tokens cannot be revoked per-image once baked in.
> - Use a **bucket-scoped token** (Object Read & Write for one bucket only, not account-level).
>
> Future hardening steps (tracked as TODOs, not yet implemented):
>
> - Makefile `docker-push` guard requiring explicit `CONFIRM_PRIVATE=yes`
> - `LABEL security.contains-baked-credentials="true"` for Trivy/Docker Scout scanning
> - CI registry enforcement (verify `ghcr.io` package is private before push)

### Production build (bakes source at a specific commit)

Builds a fully self-contained image you can run anywhere (CI, cloud, vast.ai) without mounting anything.
The repo source is downloaded as a tarball at the exact commit you specify. R2 credentials are baked in
so the container can generate and upload datasets without any extra configuration.

```bash
# Build — includes Surge XT, Python deps, and baked R2 credentials
make docker-build-dev-snapshot GIT_REF=<commit-sha> GIT_PAT=<github-pat> \
  R2_ACCESS_KEY_ID=<key> R2_SECRET_ACCESS_KEY=<secret> \
  R2_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com \
  R2_BUCKET=my-bucket

# Push to your private registry
docker push tinaudio/perm:...-dev-snapshot-<commit-sha>
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

### Dataset generation and training on vast.ai

> **See [docs/pipeline.md](docs/pipeline.md)** for a full architecture diagram, file inventory,
> plug-and-play command examples, and metadata traceability documentation.

The entrypoint script (`scripts/docker_entrypoint.sh`) dispatches on the `MODE` environment variable:

| `MODE`               | What happens                                                                                                                                                          |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `generate` (default) | Generates Surge XT dataset splits, computes normalization stats, writes `metadata.json`, uploads to R2. Exits cleanly by default; set `IDLE_AFTER=1` to stay in bash. |
| `train`              | Downloads a dataset from R2, then runs `python src/train.py`. Exits cleanly by default; set `IDLE_AFTER=1` to stay in bash.                                           |
| `shell`              | Drops directly to bash (for debugging / manual runs).                                                                                                                 |

```bash
# --- GENERATE DATASET (vast.ai instance A) ---
docker run --rm --gpus all --init \
  -e MODE=generate \
  -e TRAIN_SAMPLES=50000 \
  -e VAL_SAMPLES=5000 \
  -e TEST_SAMPLES=5000 \
  tinaudio/perm:<base-image-tag>-dev-snapshot-<commit-sha>
# Uploads to r2:<bucket>/runs/surge_simple/<sha>/
# Add -e IDLE_AFTER=1 to stay in bash for inspection after completion

# --- TRAIN (vast.ai instance B, same image) ---
docker run --rm --gpus all --init \
  -e MODE=train \
  -e R2_DATASET_PATH=runs/surge_simple/<commit-sha> \
  -e TRAIN_ARGS="experiment=surge/flow_simple" \
  tinaudio/perm:<base-image-tag>-dev-snapshot-<commit-sha>

# --- OR: use Makefile helpers (interactive, GPU) ---
make docker-run-generate TRAIN_SAMPLES=10000
make docker-run-train R2_DATASET_PATH=runs/surge_simple/<commit-sha>

# --- Run a specific pinned image tag instead of :dev ---
make docker-run-generate IMAGE_TAG=<base-image-tag>-dev-snapshot-<commit-sha> TRAIN_SAMPLES=10000
```

Dataset files are uploaded to `r2:<bucket>/runs/<param_spec>/<git_sha>/` and a `metadata.json`
records the git SHA, param spec, sample counts, generation parameters, and code provenance
(`git_ref_source`, `git_dirty`) for traceability.

### Makefile variable reference

| Variable                | Default                                                       | Description                                                                                    |
| ----------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `GIT_REF`               | _(required)_                                                  | Commit SHA / tag / branch to clone or bake into the image                                      |
| `GIT_PAT`               | _(required)_                                                  | GitHub PAT with repo read access (passed as a BuildKit secret)                                 |
| `R2_ACCESS_KEY_ID`      | _(required for R2)_                                           | Cloudflare R2 API token access key (BuildKit secret)                                           |
| `R2_SECRET_ACCESS_KEY`  | _(required for R2)_                                           | Cloudflare R2 API token secret (BuildKit secret)                                               |
| `R2_ENDPOINT`           | _(required for R2)_                                           | R2 S3 endpoint URL (BuildKit secret)                                                           |
| `R2_BUCKET`             | _(required for R2)_                                           | R2 bucket name (baked as ENV var)                                                              |
| `DOCKER_FILE`           | `docker/ubuntu22_04/Dockerfile`                               | Path to the Dockerfile                                                                         |
| `DOCKER_IMAGE`          | `tinaudio/perm`                                               | Image name                                                                                     |
| `DOCKER_BASE_IMAGE`     | `vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py310` | Base Docker image                                                                              |
| `DOCKER_BUILD_MODE`     | `prebuilt`                                                    | `source` (build Surge from source) or `prebuilt` (use .deb)                                    |
| `DOCKER_TARGETPLATFORM` | `linux/amd64`                                                 | `linux/amd64` or `linux/arm64`                                                                 |
| `DOCKER_TORCH_IDX`      | `https://download.pytorch.org/whl/cu128`                      | PyTorch wheel index URL                                                                        |
| `DOCKER_BUILD_FLAGS`    | _(empty)_                                                     | Extra flags passed verbatim to `docker build`                                                  |
| `USE_CLOUD_BUILDER`     | `false`                                                       | Set to `1` to use the remote cloud builder and push the result                                 |
| `TRAIN_SAMPLES`         | `10000`                                                       | Train split size for `docker-run-generate` / `docker-ci-generate`                              |
| `VAL_SAMPLES`           | `1000`                                                        | Val split size for `docker-run-generate` / `docker-ci-generate`                                |
| `TEST_SAMPLES`          | `1000`                                                        | Test split size for `docker-run-generate` / `docker-ci-generate`                               |
| `R2_DATASET_PATH`       | _(required for train)_                                        | R2 path to download for `docker-run-train` / `docker-ci-train`                                 |
| `IMAGE_TAG`             | `dev`                                                         | Image tag for all `docker-run-*` and `docker-ci-*` targets. Override to run a pinned snapshot. |
| `IDLE_AFTER`            | `0`                                                           | Set to `1` to drop to bash after generate/train completes (interactive targets only).          |

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
make docker-build-dev-snapshot GIT_REF=<commit-sha> GIT_PAT=<github-pat> \
  DOCKER_BUILD_FLAGS="--build-arg SURGE_GIT_REF=<surge-commit-sha>"
```
