# Docker Reference

> **Last verified:** 2026-06-02

How to build, run, and debug Docker images for the synth-setter training
pipeline. Intended for developers working locally or in CI environments.

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
`DatasetSpec.r2.bucket` and flows into the container via the materialized
spec consumed by the dataset-generation entrypoint.

```bash
# Source credentials into current shell
set -a && source .env && set +a
```

### Runtime environment variables

The image contains no baked credentials and is safe to publish on public
registries. All credentials flow in at runtime via environment variables;
dispatch and dataset-run configuration flow via CLI args. This table
enumerates the credentials and required overrides callers **must** supply
at `docker run` time — the **single source of truth** for that contract.
`SYNTH_SETTER_PLUGIN_PATH` is baked at `/usr/lib/vst3/Surge XT.vst3` and
may be overridden via `-e`.

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
of the rclone remote config: it lives in `DatasetSpec.r2.bucket` and
`generate_dataset.py` interpolates it into upload paths via
`spec.r2.rclone_prefix()` (`r2:${spec.r2.bucket}/${spec.r2.prefix}`).

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
  python -c "import torch; print('torch', torch.__version__)"
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

# devcontainer-tools — dev-base + CLI tools, Node.js + Claude Code/Codex/Antigravity, zellij, dev user
# (see the "devcontainer-tools" stage in docker/ubuntu22_04/Dockerfile)
make docker-build-devcontainer-tools \
  GIT_REF="$(git rev-parse HEAD)" \
  DOCKER_BUILD_FLAGS="--load"
```

The `devcontainer-tools` stage is a sibling of `dev-snapshot` — both stages
build `FROM dev-base`, the shared parent that holds Surge XT, the venv, and
the synth-setter source. `devcontainer-tools` adds interactive CLI tooling
(see the stage's `apt-get install` list and the GitHub CLI install block),
Node.js + `@anthropic-ai/claude-code` installed system-wide, the OpenAI
`@openai/codex` CLI installed for the `dev` user via a per-user npm prefix
(`~/.npm-global`, on PATH) so later `npm install -g` runs avoid EACCES on the
root-owned global tree, the Google Antigravity (`agy`) CLI installed by its
upstream `install.sh` into `~/.local/bin` (also on PATH), the zellij
terminal multiplexer (pinned upstream musl binary, SHA256-verified, in
`/usr/local/bin`), a non-root
`dev` user, chowns the baked uv venv at `/venv/main` to `dev` so
`uv pip install` and editable installs work without sudo, and adds a
`/commandhistory` directory (owned by `dev`) that
`.devcontainer/{cpu,gpu}/devcontainer.json` mounts as a named volume so bash
history survives container rebuilds. The same configs also mount the
`synth-setter-tmux-resurrect` and `synth-setter-tmux-resurrect-root` named
volumes at `/home/dev/.local/share/tmux/resurrect` and
`/root/.local/share/tmux/resurrect` so tmux sessions saved by
tmux-continuum (configured in `.devcontainer/tmux.conf`) survive container
rebuilds for both `DEVCONTAINER_USER=dev` (the default) and opt-in
`DEVCONTAINER_USER=root` sessions. Restore is opt-in (`@continuum-restore off`):
a rebuilt container starts clean, and the saved session is brought back on
demand with `prefix + Ctrl-r`. The VS Code terminal defaults to the
`zellij` profile (tmux stays selectable); the `synth-setter-zellij-cache` and
`synth-setter-zellij-cache-root` named volumes at `/home/dev/.cache/zellij`
and `/root/.cache/zellij` persist zellij's serialized (resurrectable) sessions
across rebuilds for the same two users. The same devcontainer configs also
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
`src/synth_setter/configs/image/` and validated by
[image_config.py](../../src/synth_setter/pipeline/schemas/image_config.py) — a Pydantic `BaseModel`
with `strict=True` and `extra="forbid"`. The config loader rejects unknown
keys, invalid types, and malformed values at load time.

```yaml
# src/synth_setter/configs/image/dev-snapshot.yaml
dockerfile: docker/ubuntu22_04/Dockerfile
image: tinaudio/synth-setter
base_image: "ubuntu@sha256:3ba65aa..."
base_image_tag: ubuntu22_04
build_mode: prebuilt
target_platform: linux/amd64
torch_backend: "cu128"
```

Runtime inputs (`github_sha`, `issue_number`) are provided by the caller, not
stored in the YAML. The schema is tested in
[test_image_config.py](../../tests/pipeline/schemas/test_image_config.py) — covers
validation, defaults, and drift detection against the real YAML.

______________________________________________________________________

## 3. Running Containers

### Entrypoint

`dev-snapshot` has **no baked `ENTRYPOINT`**, and the default `CMD` is
`/bin/bash`. Callers run the console scripts shipped by the installed
`synth_setter` package directly:

```bash
docker run --rm --env-file .env synth-setter:dev-snapshot \
  synth-setter-generate-dataset experiment=<name>
```

Available console scripts (declared in `pyproject.toml`'s
`[project.scripts]`): `synth-setter-train`, `synth-setter-eval`,
`synth-setter-generate-dataset`, `synth-setter-generate-dataset-from-hydra`,
`synth-setter-spec-uri`.

Prefer `docker run --env-file .env` over `set -a && source .env` to avoid
polluting your host shell.

### Debug shell

```bash
docker run --rm -it synth-setter:dev-snapshot bash
```

### Running ad-hoc commands

`docker run` with trailing argv executes the argv inside the container.
Add `--env-file .env` for any invocation that needs R2 (`rclone` operations)
or W&B logging.

```bash
docker run --rm synth-setter:dev-snapshot \
  python -c "import torch; print(torch.cuda.is_available())"
```

### `generate_dataset` — VST dataset generation

Generates one or more VST dataset shards (looping over `spec.shards`) via
`generate_vst_dataset.py` under headless X11 (Xvfb). The headless bootstrap
(`src/synth_setter/scripts/run-linux-vst-headless.sh`) is applied inside
`synth_setter.cli.generate_dataset.generate()` at the audio-rendering boundary,
wrapping only the generator subprocess.

**Required env vars:** See § Runtime environment variables above. For
dataset generation you need the 5 `RCLONE_CONFIG_R2_*` vars (for rclone
auth) and `WANDB_API_KEY` (if W&B logging is enabled in the dataset
config).

```bash
docker run --rm \
  --env-file .env \
  synth-setter:dev-snapshot \
  synth-setter-generate-dataset experiment=generate_dataset/smoke-shard
```

The example assumes your `.env` already contains the 5 `RCLONE_CONFIG_R2_*`
vars plus `WANDB_API_KEY`. If you prefer to keep the
`TYPE`/`PROVIDER` constants out of `.env`, add them inline:
`-e RCLONE_CONFIG_R2_TYPE=s3 -e RCLONE_CONFIG_R2_PROVIDER=Cloudflare`.

### Workflow artifact bundle (generate_dataset)

When the test workflow runs, it uploads one artifact bundle per provider:
`test-run-metadata-runpod` and `test-run-metadata-oci`. Each bundle
contains two files:

| File              | Contents                                                                 |
| ----------------- | ------------------------------------------------------------------------ |
| `input_spec.json` | DatasetSpec written by the workflow to the bind-mounted run-metadata dir |
| `generate.log`    | Full container stdout/stderr from generation                             |

**Download:**

```bash
# Per-provider:
gh run download <run_id> -n test-run-metadata-runpod
gh run download <run_id> -n test-run-metadata-oci
# Or grab everything for this run:
gh run download <run_id>
```

**Inspect:**

```bash
# View the pipeline spec
jq . input_spec.json

# Check how many samples were generated
grep -c "Saving sample" generate.log

# Find the R2 location for this run
jq -r .r2.prefix input_spec.json
```

**Retention:** 7 days (GitHub Actions default).

### Headless VST

VST3 plugins (Surge XT) require an X11 display. For dataset generation,
X11 is bootstrapped automatically around the generator subprocess inside
`generate()`. For ad-hoc VST work, prepend the headless wrapper to your command:

```bash
docker run --rm synth-setter:dev-snapshot \
  src/synth_setter/scripts/run-linux-vst-headless.sh \
    python -c "
      from pedalboard import VST3Plugin
      p = VST3Plugin('/usr/lib/vst3/Surge XT.vst3')
      print(f'Surge XT loaded, {len(p.parameters)} parameters')
    "
```

______________________________________________________________________

## 4. CI Workflow

The GHA workflow `.github/workflows/docker-build-validation.yml` builds a
dev-snapshot image, pushes to Docker Hub (and mirrors to
`ghcr.io/tinaudio/synth-setter` as a rate-limit-free pull fallback — see
#1254), and runs smoke tests.

### What it does

1. Validates the image config (`src/synth_setter/configs/image/dev-snapshot.yaml` via Pydantic)
2. Builds the image using Docker Buildx
3. Pushes tagged images to Docker Hub and ghcr.io (dispatch/push-to-main only)
4. Runs smoke tests against the SHA-pinned tag (dispatch/push-to-main only)

On **pull requests** (Docker-related paths only), the workflow runs steps 1–2
as build validation — no push, no smoke tests.

If the YAML violates the schema, the workflow fails before any build starts.

### Tags

| Tag                                              | Mutable? | Purpose                                                                          |
| ------------------------------------------------ | -------- | -------------------------------------------------------------------------------- |
| `tinaudio/synth-setter:latest`                   | Yes      | Convenience pointer to the most recent default-branch build                      |
| `tinaudio/synth-setter:dev-snapshot`             | Yes      | Latest dev-snapshot from main (gated like `latest`)                              |
| `tinaudio/synth-setter:dev-snapshot-<branch>`    | Yes      | Per-branch floating tag for feature-branch dispatches (slug = branch, `/` → `-`) |
| `tinaudio/synth-setter:dev-snapshot-<sha>`       | No       | Immutable, used for smoke tests                                                  |
| `tinaudio/synth-setter:devcontainer-tools`       | Yes      | Latest devcontainer-tools (consumed by `.devcontainer/`)                         |
| `tinaudio/synth-setter:devcontainer-tools-<sha>` | No       | Immutable, pinnable from `.devcontainer/Dockerfile`                              |

Every tag above is also published to `ghcr.io/tinaudio/synth-setter:<same-tag>`
as a Docker Hub pull mirror.

Both `latest` and `dev-snapshot` are gated to runs that represent the main
branch — push-to-main runs, dispatches with `git_ref` in `{main, refs/heads/main, refs/remotes/origin/main}`, and dispatches with a 40-char SHA that resolves
to the current `origin/main` HEAD (so a deliberate "rebuild main at this
exact commit" still advances the floating tags). Feature-branch dispatches
publish to `dev-snapshot-<branch>` instead of overwriting `dev-snapshot`.
This matters because other workflows (`test-skypilot-debug`,
`test-dataset-generation`) consume `dev-snapshot` by default — diverting
feature-branch builds to a per-branch tag prevents in-flight feature work
from silently changing what those workflows run against.

The branch slug is derived from `git_ref` after stripping well-known ref
prefixes (`refs/heads/`, `refs/remotes/origin/`), so `feat/foo`,
`refs/heads/feat/foo`, and `refs/remotes/origin/feat/foo` all publish to
the same `dev-snapshot-feat-foo` tag. Git tag dispatches (either
`refs/tags/<tag>` or a bare ref name that exists as a tag on origin) skip
the per-branch tag entirely — the immutable `dev-snapshot-<sha>` tag is
the only stable handle for tag builds.

Known limitation: branch names whose slugs collide after normalization
(e.g., `feat/foo` and `feat-foo`, or two branches sharing their first 100
chars after slugging) share the same `dev-snapshot-<slug>` tag and can
overwrite each other. Branches whose names are already Docker-tag-safe and
≤100 chars are unaffected.

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

| Secret                               | Purpose                                              |
| ------------------------------------ | ---------------------------------------------------- |
| `DOCKERHUB_USERNAME`                 | Docker Hub login (push-only; pulls are anonymous)    |
| `DOCKERHUB_TOKEN`                    | Docker Hub access token (push-only)                  |
| `RCLONE_CONFIG_R2_ACCESS_KEY_ID`     | R2 credentials (runtime; passed via `docker run -e`) |
| `RCLONE_CONFIG_R2_SECRET_ACCESS_KEY` | R2 credentials                                       |
| `RCLONE_CONFIG_R2_ENDPOINT`          | R2 endpoint (runtime)                                |
| `WANDB_API_KEY`                      | W&B auth (runtime)                                   |

The GHCR mirror push auths via the built-in `GITHUB_TOKEN` and needs the job's
`permissions: packages: write` block — no extra secret to provision.

______________________________________________________________________

## 5. Debugging

### Shell into a running container

```bash
# If a container is already running
docker exec -it <container> bash

# Start a fresh interactive debug session (drops into a shell)
docker run --rm -it synth-setter:dev-snapshot bash
```

### OOM during builds

> [!WARNING]
> The multi-stage Dockerfile build can exceed 7 GiB RAM. If the build is
> killed with no output, increase memory allocation.

- **Local:** Docker Desktop settings → 16 GiB recommended
- **GitHub Actions:** Use `ubuntu-latest-4core` (16 GiB) or larger runner

### VST fails to load

Headless X11 issues — check in order:

1. **Xvfb running?** `src/synth_setter/scripts/run-linux-vst-headless.sh` starts it automatically
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

______________________________________________________________________

## 6. Cross-references

- rclone.md (planned — [#310](https://github.com/tinaudio/synth-setter/issues/310)) — R2 setup, Docker credential baking
- [wandb-integration.md](wandb-integration.md) — W&B logging and auth
- [data-pipeline.md](../design/data-pipeline.md) — pipeline architecture, worker provisioning
- [image_config.py](../../src/synth_setter/pipeline/schemas/image_config.py) — image config schema (Pydantic model)
- [test_image_config.py](../../tests/pipeline/schemas/test_image_config.py) — config validation tests
