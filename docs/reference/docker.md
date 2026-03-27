# Docker Reference

> **Last verified:** 2026-03-27

Practical guide for building, running, and debugging synth-setter Docker images.
For the image target contract, entrypoint modes, and env var spec, see
`docs/reference/docker-spec.md`.

______________________________________________________________________

## 1. Setup

### Prerequisites

- Docker with BuildKit (Docker Desktop 23+ or `DOCKER_BUILDKIT=1`)
- Secrets in `.env`: `GIT_PAT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`,
  `R2_ENDPOINT`, `R2_BUCKET`, `WANDB_API_KEY`

```bash
# Source credentials
set -a && source .env && set +a
```

### First build (dev-live)

The dev-live image has Surge XT + Python deps but no baked source code.
Mount your local working tree at runtime.

```bash
make docker-build-dev-live \
  GIT_PAT="$GIT_PAT" \
  DOCKER_BUILD_FLAGS="--load"
```

Rebuild only when Python deps or Surge version change.

______________________________________________________________________

## 2. Building Images

### Make targets

| Target         | Command                          | Source code            | Use case  |
| -------------- | -------------------------------- | ---------------------- | --------- |
| `dev-live`     | `make docker-build-dev-live`     | Volume-mounted         | Local dev |
| `dev-snapshot` | `make docker-build-dev-snapshot` | Git clone at `GIT_REF` | CI, cloud |

Both targets require `GIT_PAT`. `dev-snapshot` additionally requires `GIT_REF`:

```bash
# dev-snapshot — self-contained image at a specific commit
make docker-build-dev-snapshot \
  GIT_PAT="$GIT_PAT" \
  GIT_REF="$(git rev-parse HEAD)" \
  DOCKER_BUILD_FLAGS="--load"
```

### Build variables

| Variable                | Default                         | Purpose                                |
| ----------------------- | ------------------------------- | -------------------------------------- |
| `DOCKER_FILE`           | `docker/ubuntu22_04/Dockerfile` | Dockerfile path                        |
| `DOCKER_IMAGE`          | `synth-setter`                  | Image name (local builds)              |
| `DOCKER_BUILD_MODE`     | `prebuilt`                      | Surge install: `source` or `prebuilt`  |
| `DOCKER_TARGETPLATFORM` | `linux/amd64`                   | Target platform                        |
| `DOCKER_TORCH_IDX`      | CUDA 12.8 wheels                | PyTorch wheel index URL                |
| `DOCKER_BUILD_FLAGS`    | *(empty)*                       | Extra flags (e.g., `--load`, `--push`) |

### BuildKit secrets

Credentials are injected via BuildKit secret mounts — they never appear in
image layers or `docker history`. See `docs/reference/rclone.md` § Docker for
the R2 credential flow.

| Secret                 | Source env var         | Purpose                              |
| ---------------------- | ---------------------- | ------------------------------------ |
| `git_pat`              | `GIT_PAT`              | GitHub API access for source tarball |
| `r2_access_key_id`     | `R2_ACCESS_KEY_ID`     | R2 rclone config                     |
| `r2_secret_access_key` | `R2_SECRET_ACCESS_KEY` | R2 rclone config                     |
| `r2_endpoint`          | `R2_ENDPOINT`          | R2 rclone config                     |
| `wandb_api_key`        | `WANDB_API_KEY`        | W&B netrc                            |

> **Security:** R2 and W&B credentials are baked into the image filesystem
> (rclone.conf, .netrc). Push only to private registries.

### Image config (CI)

For CI builds, image parameters are defined in YAML config files under
`configs/image/` and validated by `scripts/image_config.py` — a Pydantic
`BaseModel` with `strict=True` and `extra="forbid"`. The config loader rejects
unknown keys, invalid types, and malformed values at load time.

```yaml
# configs/image/dev-snapshot.yaml
dockerfile: docker/ubuntu22_04/Dockerfile
image: tinaudio/perm
base_image: "ubuntu@sha256:3ba65aa..."
base_image_tag: ubuntu22_04
build_mode: prebuilt
target_platform: linux/amd64
torch_index_url: "https://download.pytorch.org/whl/cu128"
```

Runtime inputs (`github_sha`, `issue_number`) are provided by the caller, not
stored in the YAML. The schema is defined in `scripts/image_config.py` and
tested in `tests/scripts/test_image_config.py` (18 tests covering validation,
defaults, and drift detection against the real YAML).

______________________________________________________________________

## 3. Running Containers

The entrypoint (`scripts/docker_entrypoint.sh`) dispatches on the `MODE`
environment variable. MODE is required — the container errors if unset.

### MODE=idle — debug shell

```bash
# Start in background
docker run -d --name debug -e MODE=idle synth-setter:dev-live

# Attach
docker exec -it debug bash

# Clean up
docker stop debug && docker rm debug
```

### MODE=passthrough — run a command

```bash
# Run a one-off command
docker run --rm -e MODE=passthrough synth-setter:dev-snapshot \
  python -c "import torch; print(torch.cuda.is_available())"

# No-op (exits 0) — useful for CI health checks
docker run --rm -e MODE=passthrough synth-setter:dev-snapshot
```

### Volume mounting (dev-live)

dev-live has no baked source — mount your working tree:

```bash
docker run --rm \
  -e MODE=passthrough \
  -v "$(pwd):/workspace" \
  -w /workspace \
  synth-setter:dev-live \
  python -m pytest tests/ -m "not slow"
```

### Headless VST

VST3 plugins require an X11 display. Use the headless bootstrap script:

```bash
docker run --rm \
  -e MODE=passthrough \
  synth-setter:dev-snapshot \
  scripts/run-linux-vst-headless.sh python -c "
    from pedalboard import VST3Plugin
    p = VST3Plugin('/usr/lib/vst3/Surge XT.vst3')
    print(f'Surge XT loaded, {len(p.parameters)} parameters')
  "
```

______________________________________________________________________

## 4. CI Workflow

### Overview

The GHA workflow `.github/workflows/docker-build-validation.yml` builds a
dev-snapshot image, pushes to Docker Hub, and runs smoke tests.

### Steps

1. Checkout at specified git ref
2. **Load image config** — runs `image_config.py` to load and validate
   `configs/image/dev-snapshot.yaml` via Pydantic. Exports field values as step
   outputs for subsequent steps. If the YAML violates the schema, the workflow
   fails here before any build starts.
3. Set up Docker Buildx
4. Log in to Docker Hub (`docker/login-action`)
5. Generate tags and labels (`docker/metadata-action`)
6. Build and push (`docker/build-push-action`)
7. Smoke tests against the SHA-pinned tag (`dev-snapshot-<full-sha>`)

### Tags

| Tag                                | Mutable? | Purpose                           |
| ---------------------------------- | -------- | --------------------------------- |
| `tinaudio/perm:dev-snapshot`       | Yes      | Latest dev-snapshot (convenience) |
| `tinaudio/perm:dev-snapshot-<sha>` | No       | Immutable, used for smoke tests   |

Smoke tests pull the SHA-pinned tag to avoid race conditions with concurrent
workflow runs.

### Manual trigger

```bash
gh workflow run docker-build-validation.yml --ref main
```

### Required secrets

| Secret                 | Purpose                              |
| ---------------------- | ------------------------------------ |
| `GIT_PAT`              | GitHub API access for source tarball |
| `DOCKERHUB_USERNAME`   | Docker Hub login                     |
| `DOCKERHUB_TOKEN`      | Docker Hub access token              |
| `R2_ACCESS_KEY_ID`     | R2 credentials (baked via BuildKit)  |
| `R2_SECRET_ACCESS_KEY` | R2 credentials                       |
| `R2_ENDPOINT`          | R2 endpoint                          |
| `WANDB_API_KEY`        | W&B auth                             |

______________________________________________________________________

## 5. Debugging

### Shell into a running container

```bash
# If the container is already running
docker exec -it <container> bash

# Start a fresh debug session
docker run --rm -it -e MODE=idle synth-setter:dev-snapshot
# Then from another terminal:
docker exec -it <container> bash
```

### OOM during builds

The multi-stage Dockerfile build can exceed 7 GiB RAM. If the build is killed
with no output:

- **Local:** Increase Docker Desktop memory allocation (16 GiB recommended)
- **GitHub Actions:** Use `ubuntu-latest-4core` (16 GiB) or larger runner

### VST fails to load

Headless X11 issues — check:

1. Is Xvfb running? `scripts/run-linux-vst-headless.sh` starts it automatically
2. Missing libraries: `ldd /usr/lib/vst3/Surge\ XT.vst3/Contents/*/libSurge\ XT.so`
3. `LIBGL_ALWAYS_SOFTWARE=1` is set (no GPU in CI)

### BuildKit cache

The GHA workflow uses GitHub Actions cache (`type=gha`). To clear locally:

```bash
docker buildx prune
```

### Entrypoint errors

| Error              | Cause                | Fix                                         |
| ------------------ | -------------------- | ------------------------------------------- |
| `MODE is required` | MODE env var not set | Add `-e MODE=idle` or `-e MODE=passthrough` |
| `unknown MODE 'X'` | Typo in MODE value   | Use `idle` or `passthrough`                 |

______________________________________________________________________

## 6. Cross-references

- `docs/reference/docker-spec.md` — image target contract, entrypoint spec, env vars
- `docs/reference/rclone.md` — R2 setup, Docker credential baking
- `docs/reference/wandb-integration.md` — W&B logging and auth
- `docs/design/data-pipeline.md` — pipeline architecture, worker provisioning
- `scripts/image_config.py` — image config schema (Pydantic model)
- `tests/scripts/test_image_config.py` — config validation tests
