# Docker Reference

> **Last verified:** 2026-03-27

How to build, run, and debug Docker images for the synth-setter training
pipeline. Intended for developers working locally or in CI environments.

For the image target contract, entrypoint modes, and env var spec, see
[docker-spec.md](docker-spec.md).

______________________________________________________________________

## 1. Setup

### Prerequisites

- Docker with [BuildKit](https://docs.docker.com/build/buildkit/) enabled
  (Docker Desktop 23+ or `DOCKER_BUILDKIT=1`). BuildKit adds secret mounts
  and multi-stage caching — both used heavily in this project.
- Secrets in `.env`: `GIT_PAT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`,
  `R2_ENDPOINT`, `R2_BUCKET`, `WANDB_API_KEY`

```bash
# Source credentials into current shell
set -a && source .env && set +a
```

### BuildKit secrets

Credentials are injected at build time via `--mount=type=secret`. The secret
data is available only during the `RUN` instruction and never appears in
`docker history`. However, the build writes some secrets into config files
that persist in the final image:

| Secret                 | Source env var         | Persisted to                       | Purpose                              |
| ---------------------- | ---------------------- | ---------------------------------- | ------------------------------------ |
| `git_pat`              | `GIT_PAT`              | *(not persisted)*                  | GitHub API access for source tarball |
| `r2_access_key_id`     | `R2_ACCESS_KEY_ID`     | `/root/.config/rclone/rclone.conf` | R2 rclone config                     |
| `r2_secret_access_key` | `R2_SECRET_ACCESS_KEY` | `/root/.config/rclone/rclone.conf` | R2 rclone config                     |
| `wandb_api_key`        | `WANDB_API_KEY`        | `/root/.netrc`                     | W&B auth                             |

> **Note:** `R2_ENDPOINT` and `R2_BUCKET` are passed as `--build-arg`, not as
> BuildKit secrets. `R2_ENDPOINT` is written into `rclone.conf` by the
> `r2-config-base` stage; `R2_BUCKET` is set as an `ENV` in the image. Both
> appear in `docker history`. See the build-arg table below.

> [!WARNING]
> R2 and W&B credentials persist in the final image filesystem (`rclone.conf`,
> `.netrc`). The secrets are injected securely at build time (never in
> `docker history`), but the resulting config files **are baked into the image**.
> Push only to private registries. Rotate R2 tokens after each build campaign.

The rclone reference doc is planned ([#310](https://github.com/tinaudio/synth-setter/issues/310)).

### First build (dev-live)

The dev-live image has Surge XT + Python deps but no baked source code.
Mount your local working tree at runtime.

```bash
make docker-build-dev-live \
  GIT_PAT="$GIT_PAT" \
  DOCKER_BUILD_FLAGS="--load"
  # --load: imports the built image into your local Docker daemon
  # --push: pushes directly to a registry (for CI/multi-platform)
```

### Smoke test

After building, verify the image works. dev-live uses a fallback entrypoint
(not `docker_entrypoint.sh`), so override it with `--entrypoint bash`:

```bash
docker run --rm --entrypoint bash synth-setter:dev-live \
  -c "python -c \"import torch; print('torch', torch.__version__)\""
```

Rebuild only when Python deps or Surge version change.

______________________________________________________________________

## 2. Building Images

### Make targets

| Target         | Source code                  | Typical use           |
| -------------- | ---------------------------- | --------------------- |
| `dev-live`     | Volume-mounted               | Local development     |
| `dev-snapshot` | Git clone at `GIT_REF`       | CI, cloud, evaluation |
| `prod`         | Baked at `GIT_REF` (tarball) | Production            |

> **Note:** The Makefile only has `docker-build-dev-live` and
> `docker-build-dev-snapshot` targets. To build the `prod` target, invoke
> `docker buildx build` directly with `--build-arg IMAGE=prod` — the Makefile's
> `docker-build-*` targets override the `IMAGE` build arg after `DOCKER_BUILD_FLAGS`.

All targets require `GIT_PAT`. `dev-snapshot` and `prod` additionally require `GIT_REF`:

```bash
# dev-snapshot — self-contained image at a specific commit
make docker-build-dev-snapshot \
  GIT_PAT="$GIT_PAT" \
  GIT_REF="$(git rev-parse HEAD)" \
  DOCKER_BUILD_FLAGS="--load"
```

### Build variables

| Variable                | Default                         | Purpose                                          | Override via |
| ----------------------- | ------------------------------- | ------------------------------------------------ | ------------ |
| `DOCKER_FILE`           | `docker/ubuntu22_04/Dockerfile` | Dockerfile path                                  | CLI only     |
| `DOCKER_IMAGE`          | `synth-setter`                  | Image name (local builds)                        | CLI only     |
| `DOCKER_BUILD_MODE`     | `prebuilt`                      | Surge install: `source` or `prebuilt` (see note) | CLI only     |
| `DOCKER_TARGETPLATFORM` | `linux/amd64`                   | Target platform                                  | CLI only     |
| `DOCKER_TORCH_IDX`      | CUDA 12.8 wheels                | PyTorch wheel index URL                          | CLI or YAML  |
| `DOCKER_BUILD_FLAGS`    | *(empty)*                       | `--load` (local) or `--push` (remote)            | CLI only     |

`DOCKER_TORCH_IDX` can also be set via `torch_index_url` in the image config
YAML (see Image config below). CLI takes precedence.

> **BUILD_MODE default divergence:** The Makefile defaults `DOCKER_BUILD_MODE`
> to `prebuilt`, which is passed as `--build-arg BUILD_MODE`. The Dockerfile's
> own `ARG BUILD_MODE=source` default only applies when the arg is not provided
> at all. In practice, builds through `make` always use `prebuilt`.

### Image config (CI)

For CI builds, image parameters are defined in YAML config files under
`configs/image/` and validated by
[image_config.py](../../pipeline/schemas/image_config.py) — a Pydantic `BaseModel`
with `strict=True` and `extra="forbid"`. The config loader rejects unknown
keys, invalid types, and malformed values at load time.

```yaml
# configs/image/dev-snapshot.yaml
dockerfile: docker/ubuntu22_04/Dockerfile
image: tinaudio/perm
base_image: "ubuntu@sha256:3ba65aa..."
base_image_tag: ubuntu22_04
build_mode: prebuilt
target_platform: linux/amd64
torch_index_url: "https://download.pytorch.org/whl/cu128"
r2_endpoint: "https://efb9275d571811db929e83eb710b74a7.r2.cloudflarestorage.com"
r2_bucket: "intermediate-data"
```

Runtime inputs (`github_sha`, `issue_number`) are provided by the caller, not
stored in the YAML. The schema is tested in
[test_image_config.py](../../tests/pipeline/test_schemas/test_image_config.py) (22 tests
covering validation, defaults, and drift detection against the real YAML).

______________________________________________________________________

## 3. Running Containers

### Entrypoint differences

Only `prod` and `dev-snapshot` include `docker_entrypoint.sh` with MODE
dispatch. `dev-live` has a fallback entrypoint that errors unless the repo
is mounted at `/home/build/synth-setter` — use `--entrypoint bash` to bypass
it for ad-hoc commands.

| Target         | Entrypoint                | MODE support | Typical invocation           |
| -------------- | ------------------------- | ------------ | ---------------------------- |
| `dev-snapshot` | `docker_entrypoint.sh`    | Yes          | `-e MODE=idle`               |
| `prod`         | `docker_entrypoint.sh`    | Yes          | `-e MODE=idle`               |
| `dev-live`     | Fallback (requires mount) | No           | `--entrypoint bash` or mount |

For `dev-snapshot` / `prod`, MODE is required — the container exits with an
error if unset. There is no default to avoid silent misconfiguration.

Prefer `docker run --env-file .env` over `set -a && source .env` to avoid
polluting your host shell:

```bash
docker run --rm --env-file .env -e MODE=passthrough synth-setter:dev-snapshot ...
```

### MODE=idle — debug shell

```bash
docker run -d --name debug -e MODE=idle synth-setter:dev-snapshot
docker exec -it debug bash
# Clean up when done
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

### MODE=generate_dataset — VST dataset generation

Generates a VST dataset shard via `generate_vst_dataset.py` under headless X11
(Xvfb). X11 bootstrapping (Xvfb/dbus/xsettings) is handled by
`scripts/run-linux-vst-headless.sh` (invoked from the Docker entrypoint);
the entrypoint module `pipeline/entrypoints/generate_dataset.py` reads env/config
and invokes `generate_vst_dataset.py` with the resolved dataset config.
The container materializes a DatasetPipelineSpec, uploads spec and shard to R2.
`input_spec.json` is written to `RUN_METADATA_DIR`.

| Env var            | Required | Default         | Purpose                                    |
| ------------------ | -------- | --------------- | ------------------------------------------ |
| `DATASET_CONFIG`   | Yes      | —               | Path to dataset config YAML in container   |
| `RUN_METADATA_DIR` | No       | `/run-metadata` | Directory where input_spec.json is written |

```bash
docker run --rm \
  -e MODE=generate_dataset \
  -e DATASET_CONFIG=configs/dataset/ci-smoke-test.yaml \
  -v "$(pwd)/run-metadata:/run-metadata" \
  synth-setter:dev-snapshot
```

### Workflow artifact bundle (generate_dataset)

When the test workflow runs, it uploads an artifact bundle named
`test-run-metadata`. The bundle contains two files:

| File              | Contents                                                           |
| ----------------- | ------------------------------------------------------------------ |
| `input_spec.json` | DatasetPipelineSpec written by the container to `RUN_METADATA_DIR` |
| `generate.log`    | Full container stdout/stderr from generation                       |

**Download:**

```bash
gh run download <run_id> -n test-run-metadata
```

**Inspect:**

```bash
# View the pipeline spec
jq . input_spec.json

# Check how many samples were generated
grep -c "Saving sample" generate.log

# Find the R2 location for this run
jq .r2_prefix input_spec.json
```

**Retention:** 7 days (GitHub Actions default).

### Volume mounting (dev-live)

dev-live has no baked source or `docker_entrypoint.sh`. Mount your working
tree at `/home/build/synth-setter` and override the entrypoint:

```bash
docker run --rm \
  --entrypoint bash \
  -v "$(pwd):/home/build/synth-setter" \
  -w /home/build/synth-setter \
  synth-setter:dev-live \
  -c "python -m pytest tests/ -m 'not slow'"
```

### Headless VST

VST3 plugins (Surge XT) require an X11 display. The headless bootstrap script
starts Xvfb automatically:

```bash
docker run --rm \
  -e MODE=passthrough \
  synth-setter:dev-snapshot \
  scripts/run-linux-vst-headless.sh \
    python -c "
      from pedalboard import VST3Plugin
      p = VST3Plugin('/usr/lib/vst3/Surge XT.vst3')
      print(f'Surge XT loaded, {len(p.parameters)} parameters')
    "
```

______________________________________________________________________

## 4. CI Workflow

The GHA workflow `.github/workflows/docker-build-validation.yml` builds a
dev-snapshot image, pushes to Docker Hub, and runs smoke tests.

### What it does

1. Validates the image config (`configs/image/dev-snapshot.yaml` via Pydantic)
2. Builds the image using Docker Buildx
3. Pushes tagged images to Docker Hub (dispatch/schedule only)
4. Runs smoke tests against the SHA-pinned tag (dispatch/schedule only)

On **pull requests** (Docker-related paths only), the workflow runs steps 1–2
as build validation — no push, no smoke tests.

If the YAML violates the schema, the workflow fails before any build starts.

### Tags

| Tag                                | Mutable? | Purpose                           |
| ---------------------------------- | -------- | --------------------------------- |
| `tinaudio/perm:dev-snapshot`       | Yes      | Latest dev-snapshot (convenience) |
| `tinaudio/perm:dev-snapshot-<sha>` | No       | Immutable, used for smoke tests   |

Smoke tests pull the SHA-pinned tag to avoid race conditions with concurrent
workflow runs.

Local `make` builds use a different tag format:
`<base_image_tag>-dev-snapshot-<GIT_REF>` (e.g., `ubuntu22_04-dev-snapshot-abc1234`).

### Manual trigger

```bash
# Manually trigger the CI docker build workflow
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
# If a container is already running
docker exec -it <container> bash

# Start a fresh interactive debug session (drops into a shell)
docker run --rm -it -e MODE=idle synth-setter:dev-snapshot bash
```

### OOM during builds

> [!WARNING]
> The multi-stage Dockerfile build can exceed 7 GiB RAM. If the build is
> killed with no output, increase memory allocation.

- **Local:** Docker Desktop settings → 16 GiB recommended
- **GitHub Actions:** Use `ubuntu-latest-4core` (16 GiB) or larger runner

### VST fails to load

Headless X11 issues — check in order:

1. **Xvfb running?** `scripts/run-linux-vst-headless.sh` starts it automatically
2. **Missing libraries?** `ldd /usr/lib/vst3/Surge\ XT.vst3/Contents/*/libSurge\ XT.so`
3. **Software rendering?** Verify `LIBGL_ALWAYS_SOFTWARE=1` is set (no GPU in CI)

### BuildKit cache

The GHA workflow uses Docker Hub registry cache (`type=registry`). Cache layers
are stored as `tinaudio/perm:buildcache`.

To clear local builder cache:

```bash
docker buildx prune
```

To clear the remote registry cache, delete the `buildcache` tag from Docker Hub
(Settings → Tags → `buildcache` → Delete).

### Entrypoint errors

| Error              | Cause                | Fix                                                                      |
| ------------------ | -------------------- | ------------------------------------------------------------------------ |
| `MODE is required` | MODE env var not set | Add `-e MODE=idle`, `-e MODE=passthrough`, or `-e MODE=generate_dataset` |
| `unknown MODE 'X'` | Typo in MODE value   | Use `idle`, `passthrough`, or `generate_dataset`                         |

______________________________________________________________________

## 6. Scoped and planned modes

### Scoped — validated on experiment branch, pending port to main

- **MODE=generate-shards** ([#407](https://github.com/tinaudio/synth-setter/issues/407)) — multi-shard parallel VST
  dataset generation with R2 upload. Replaces `generate_dataset` (which becomes
  deprecated, [#411](https://github.com/tinaudio/synth-setter/issues/411)).
  See [data-pipeline.md](../design/data-pipeline.md) § Generate stage.
- **MODE=finalize-shards** ([#408](https://github.com/tinaudio/synth-setter/issues/408)) — download shards from R2,
  reshard into train/val/test, compute normalization stats, upload.
- **MODE=train** ([#409](https://github.com/tinaudio/synth-setter/issues/409)) — download dataset from R2, run
  `src/train.py` via Hydra, upload checkpoints. Currently handled manually.

### Planned — not yet implemented

- **MODE=eval** ([#410](https://github.com/tinaudio/synth-setter/issues/410)) — download checkpoint + dataset,
  run evaluation, upload results. Counterpart to `MODE=train`.

### Other future work

- **dev-live MODE support** — currently dev-live uses a fallback entrypoint
  that requires a volume mount. A future change would add `docker_entrypoint.sh`
  to dev-live so it supports MODE dispatch like dev-snapshot/prod.

______________________________________________________________________

## 7. Cross-references

- [docker-spec.md](docker-spec.md) — image target contract, entrypoint spec, env vars
- rclone.md (planned — [#310](https://github.com/tinaudio/synth-setter/issues/310)) — R2 setup, Docker credential baking
- [wandb-integration.md](wandb-integration.md) — W&B logging and auth
- [data-pipeline.md](../design/data-pipeline.md) — pipeline architecture, worker provisioning
- [image_config.py](../../pipeline/schemas/image_config.py) — image config schema (Pydantic model)
- [test_image_config.py](../../tests/pipeline/test_schemas/test_image_config.py) — config validation tests
