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
  (Docker Desktop 23+ or `DOCKER_BUILDKIT=1`). BuildKit provides multi-stage
  caching used heavily in this project.
- Build-time secrets: none. The repo is public, source is fetched anonymously.
- Runtime env vars: see § Runtime environment variables below for the full
  enumeration. At minimum, a `.env` file containing:
  - `RCLONE_CONFIG_R2_TYPE=s3`, `RCLONE_CONFIG_R2_PROVIDER=Cloudflare` (constants)
  - `RCLONE_CONFIG_R2_ACCESS_KEY_ID`, `RCLONE_CONFIG_R2_SECRET_ACCESS_KEY`,
    `RCLONE_CONFIG_R2_ENDPOINT` (R2 credentials)
  - `WANDB_API_KEY` (W&B credential)

The target R2 bucket is **not** an env var — it is a required field on
`DatasetConfig` / `DatasetPipelineSpec` and flows into the container via
the materialized spec passed to `generate_dataset --spec`.

```bash
# Source credentials into current shell
set -a && source .env && set +a
```

### Runtime environment variables

The image contains no baked credentials and is safe to publish on public
registries. All credentials flow in at runtime via environment variables;
dispatch and dataset-run configuration flow via CLI args (subcommand +
`--spec`). This is the **single source of truth** for what the image
expects at `docker run` time.

| Env var                              | Consumer  | Required for       | Notes                                       |
| ------------------------------------ | --------- | ------------------ | ------------------------------------------- |
| `RCLONE_CONFIG_R2_TYPE`              | rclone    | any rclone R2 op   | Constant: `s3`; from `.env` or `-e`         |
| `RCLONE_CONFIG_R2_PROVIDER`          | rclone    | any rclone R2 op   | Constant: `Cloudflare`; from `.env` or `-e` |
| `RCLONE_CONFIG_R2_ACCESS_KEY_ID`     | rclone    | any rclone R2 op   | **Secret**; from `.env`                     |
| `RCLONE_CONFIG_R2_SECRET_ACCESS_KEY` | rclone    | any rclone R2 op   | **Secret**; from `.env`                     |
| `RCLONE_CONFIG_R2_ENDPOINT`          | rclone    | any rclone R2 op   | **Secret**; from `.env`                     |
| `WANDB_API_KEY`                      | wandb SDK | any W&B-logging op | **Secret**; from `.env`                     |

rclone's native env-var config automatically builds the `r2` remote
inside the container from the `RCLONE_CONFIG_R2_*` variables — no
`rclone.conf` file is read or written. The bucket name is **not** part
of the rclone remote config: it lives in `DatasetPipelineSpec.r2_bucket`
and `generate_dataset.py` interpolates it into upload paths
(`r2:${spec.r2_bucket}/...`).

The build uses **no** BuildKit secrets. The repository is public, so
source fetches (both the tarball and the in-image git clone) happen
anonymously. There is no `GIT_PAT` in the build pipeline.

The rclone reference doc is planned ([#310](https://github.com/tinaudio/synth-setter/issues/310)).

### First build (dev-snapshot)

The dev-snapshot image has Surge XT + Python deps + source code baked at a
specific git ref.

```bash
make docker-build-dev-snapshot \
  GIT_REF="$(git rev-parse HEAD)" \
  DOCKER_BUILD_FLAGS="--load"
  # --load: imports the built image into your local Docker daemon
  # --push: pushes directly to a registry (for CI/multi-platform)
```

### Smoke test

After building, verify the image works:

```bash
docker run --rm synth-setter:dev-snapshot \
  passthrough python -c "import torch; print('torch', torch.__version__)"
```

______________________________________________________________________

## 2. Building Images

### Make targets

| Target               | Source code            | Typical use                                   |
| -------------------- | ---------------------- | --------------------------------------------- |
| `dev-snapshot`       | Git clone at `GIT_REF` | CI, cloud, evaluation                         |
| `devcontainer-tools` | Git clone at `GIT_REF` | Dev container base (CLI tools + non-root dev) |

Set `GIT_REF` for reproducible builds (defaults to `main` if omitted):

```bash
# dev-snapshot — self-contained image at a specific commit
make docker-build-dev-snapshot \
  GIT_REF="$(git rev-parse HEAD)" \
  DOCKER_BUILD_FLAGS="--load"

# devcontainer-tools — dev-base + gh, jq, Node.js, Claude Code, dev user
make docker-build-devcontainer-tools \
  GIT_REF="$(git rev-parse HEAD)" \
  DOCKER_BUILD_FLAGS="--load"
```

The `devcontainer-tools` stage is a sibling of `dev-snapshot` — both stages
build `FROM dev-base`, the shared parent that holds Surge XT, the venv, and
the synth-setter source. `devcontainer-tools` adds CLI tooling (`gh`, `jq`),
Node.js + `@anthropic-ai/claude-code` installed system-wide, a non-root
`dev` user, and a `/commandhistory` directory (owned by `dev`) that
`.devcontainer/{cpu,gpu}/devcontainer.json` mounts as a named volume so bash
history survives container rebuilds. The same devcontainer configs also
overlay `/home/build/synth-setter/plugins` with an anonymous volume so the
baked `plugins/Surge XT.vst3` symlink survives the workspace bind mount —
without it, the host's gitignored `plugins/` would shadow the baked file and
VST-dependent tests would fail. `.devcontainer/Dockerfile` consumes the
stage via `FROM tinaudio/synth-setter:devcontainer-tools`.

### Build variables

| Variable                | Default                         | Purpose                                          | Override via |
| ----------------------- | ------------------------------- | ------------------------------------------------ | ------------ |
| `DOCKER_FILE`           | `docker/ubuntu22_04/Dockerfile` | Dockerfile path                                  | CLI only     |
| `DOCKER_IMAGE`          | `synth-setter`                  | Image name (local builds)                        | CLI only     |
| `DOCKER_BUILD_MODE`     | `prebuilt`                      | Surge install: `source` or `prebuilt` (see note) | CLI only     |
| `DOCKER_TARGETPLATFORM` | `linux/amd64`                   | Target platform                                  | CLI only     |
| `DOCKER_TORCH_BACKEND`  | `cu128`                         | PyTorch backend (e.g. cu128, cpu)                | CLI or YAML  |
| `DOCKER_BUILD_FLAGS`    | *(empty)*                       | `--load` (local) or `--push` (remote)            | CLI only     |

`DOCKER_TORCH_BACKEND` can also be set via `torch_backend` in the image config
YAML (see Image config below). CLI takes precedence.

> **BUILD_MODE default divergence:** The Makefile defaults `DOCKER_BUILD_MODE`
> to `prebuilt`, which is passed as `--build-arg BUILD_MODE`. The Dockerfile's
> own `ARG BUILD_MODE=source` default only applies when the arg is not provided
> at all. In practice, builds through `make` default to `prebuilt` (override with
> `DOCKER_BUILD_MODE=source`).

### Image config (CI)

For CI builds, image parameters are defined in YAML config files under
`configs/image/` and validated by
[image_config.py](../../pipeline/schemas/image_config.py) — a Pydantic `BaseModel`
with `strict=True` and `extra="forbid"`. The config loader rejects unknown
keys, invalid types, and malformed values at load time.

```yaml
# configs/image/dev-snapshot.yaml
dockerfile: docker/ubuntu22_04/Dockerfile
image: tinaudio/synth-setter
base_image: "ubuntu@sha256:3ba65aa..."
base_image_tag: ubuntu22_04
build_mode: prebuilt
target_platform: linux/amd64
torch_backend: "cu128"
r2_bucket: "intermediate-data"
```

Runtime inputs (`github_sha`, `issue_number`) are provided by the caller, not
stored in the YAML. The schema is tested in
[test_image_config.py](../../tests/pipeline/test_schemas/test_image_config.py) (22 tests
covering validation, defaults, and drift detection against the real YAML).

______________________________________________________________________

## 3. Running Containers

### Entrypoint

`dev-snapshot` runs `python docker_entrypoint.py` as its `ENTRYPOINT`: a
click group with five subcommands (`idle`, `passthrough`,
`generate_dataset`, `render_eval`, `train`). A subcommand is required —
the container fails loudly if invoked with none. See
[docker-spec.md](docker-spec.md) for the full table.

Prefer `docker run --env-file .env` over `set -a && source .env` to avoid
polluting your host shell:

```bash
docker run --rm --env-file .env synth-setter:dev-snapshot passthrough ...
```

### `idle` — debug shell

```bash
docker run -d --name debug synth-setter:dev-snapshot idle
docker exec -it debug bash
# Clean up when done
docker stop debug && docker rm debug
```

### `passthrough` — run a command

```bash
# Run a one-off command (no creds needed — just a torch import)
docker run --rm synth-setter:dev-snapshot \
  passthrough python -c "import torch; print(torch.cuda.is_available())"
```

> **Note:** add `--env-file .env` to any passthrough invocation that needs
> R2 (`rclone` operations) or W&B logging. `passthrough` with no trailing
> argv fails loudly (non-zero exit) — there is no silent-no-op mode.

### `generate_dataset` — VST dataset generation

Generates a VST dataset shard via `generate_vst_dataset.py` under
headless X11 (Xvfb). The click entrypoint itself is X11-agnostic; the
headless bootstrap (`scripts/run-linux-vst-headless.sh`) is applied
inside `pipeline.entrypoints.generate_dataset.run()` at the
audio-rendering boundary, wrapping only the generator subprocess — so
`idle` and `passthrough` don't pay the Xvfb startup cost.

Pass the materialized spec via `--spec <path>`. All dataset-run
configuration, including the target R2 bucket, lives in that spec
(`DatasetPipelineSpec.r2_bucket`).

**Required env vars:** See § Runtime environment variables above. For
this subcommand you need the 5 `RCLONE_CONFIG_R2_*` vars (for rclone
auth) and `WANDB_API_KEY` (if W&B logging is enabled in the dataset
config).

```bash
docker run --rm \
  --env-file .env \
  -v "$(pwd)/run-metadata:/run-metadata" \
  synth-setter:dev-snapshot \
  generate_dataset --spec /run-metadata/input_spec.json
```

The example assumes your `.env` already contains the 5 `RCLONE_CONFIG_R2_*`
vars plus `WANDB_API_KEY`. If you prefer to keep the
`TYPE`/`PROVIDER` constants out of `.env`, add them inline:
`-e RCLONE_CONFIG_R2_TYPE=s3 -e RCLONE_CONFIG_R2_PROVIDER=Cloudflare`.

### Workflow artifact bundle (generate_dataset)

When the test workflow runs, it uploads an artifact bundle named
`test-run-metadata`. The bundle contains two files:

| File              | Contents                                                                         |
| ----------------- | -------------------------------------------------------------------------------- |
| `input_spec.json` | DatasetPipelineSpec written by the workflow to the bind-mounted run-metadata dir |
| `generate.log`    | Full container stdout/stderr from generation                                     |

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

### Headless VST

VST3 plugins (Surge XT) require an X11 display. For `generate_dataset`,
X11 is bootstrapped automatically around the generator subprocess inside
`run()`. For ad-hoc VST work through `passthrough`, prepend the headless
wrapper to your command:

```bash
docker run --rm synth-setter:dev-snapshot \
  passthrough scripts/run-linux-vst-headless.sh \
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

| Tag                                              | Mutable? | Purpose                                                     |
| ------------------------------------------------ | -------- | ----------------------------------------------------------- |
| `tinaudio/synth-setter:latest`                   | Yes      | Convenience pointer to the most recent default-branch build |
| `tinaudio/synth-setter:dev-snapshot`             | Yes      | Latest dev-snapshot (convenience)                           |
| `tinaudio/synth-setter:dev-snapshot-<sha>`       | No       | Immutable, used for smoke tests                             |
| `tinaudio/synth-setter:devcontainer-tools`       | Yes      | Latest devcontainer-tools (consumed by `.devcontainer/`)    |
| `tinaudio/synth-setter:devcontainer-tools-<sha>` | No       | Immutable, pinnable from `.devcontainer/Dockerfile`         |

Mutable tags (`latest`, `dev-snapshot`) are only published on dispatch/schedule
runs — not on pull-request build validations. `latest` is additionally gated
to schedule runs or to `workflow_dispatch` with `git_ref=main`, so dispatching
from main with a non-main `git_ref` does not overwrite `latest`.

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

| Secret                 | Purpose                                              |
| ---------------------- | ---------------------------------------------------- |
| `DOCKERHUB_USERNAME`   | Docker Hub login (push-only; pulls are anonymous)    |
| `DOCKERHUB_TOKEN`      | Docker Hub access token (push-only)                  |
| `R2_ACCESS_KEY_ID`     | R2 credentials (runtime; passed via `docker run -e`) |
| `R2_SECRET_ACCESS_KEY` | R2 credentials                                       |
| `R2_ENDPOINT`          | R2 endpoint (runtime)                                |
| `WANDB_API_KEY`        | W&B auth (runtime)                                   |

______________________________________________________________________

## 5. Debugging

### Shell into a running container

```bash
# If a container is already running
docker exec -it <container> bash

# Start a fresh interactive debug session (drops into a shell)
docker run --rm -it synth-setter:dev-snapshot passthrough bash
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
are stored as `tinaudio/synth-setter:buildcache`.

To clear local builder cache:

```bash
docker buildx prune
```

To clear the remote registry cache, delete the `buildcache` tag from Docker Hub
(Settings → Tags → `buildcache` → Delete).

### Entrypoint errors

| Error                                    | Cause                               | Fix                                                                          |
| ---------------------------------------- | ----------------------------------- | ---------------------------------------------------------------------------- |
| `Missing subcommand`                     | Ran the image with no subcommand    | Append one of: `idle`, `passthrough <cmd>`, `generate_dataset --spec <path>` |
| `No such command 'X'`                    | Typo in subcommand name             | Use one of `idle`, `passthrough`, `generate_dataset`, `render_eval`, `train` |
| `passthrough requires a command to exec` | Ran `passthrough` with no argv      | Append the command and its args after `passthrough`                          |
| `Unable to read spec at ...`             | `--spec` path is missing/unreadable | Confirm the path exists inside the container (bind mount + filename)         |
| `Invalid spec at ...`                    | Spec JSON fails pydantic validation | Re-materialize the spec; see `pipeline.ci.materialize_spec`                  |

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

______________________________________________________________________

## 7. Cross-references

- [docker-spec.md](docker-spec.md) — image target contract, entrypoint spec, env vars
- rclone.md (planned — [#310](https://github.com/tinaudio/synth-setter/issues/310)) — R2 setup, Docker credential baking
- [wandb-integration.md](wandb-integration.md) — W&B logging and auth
- [data-pipeline.md](../design/data-pipeline.md) — pipeline architecture, worker provisioning
- [image_config.py](../../pipeline/schemas/image_config.py) — image config schema (Pydantic model)
- [test_image_config.py](../../tests/pipeline/test_schemas/test_image_config.py) — config validation tests
