# Docker Reference Doc Design

## Purpose

Practical usage guide for building, running, and debugging Docker images.
Complements `docs/reference/docker-spec.md` (contract/spec) — this doc covers
the "how", the spec covers the "what".

Follows the pattern established by `docs/reference/rclone.md`: tables for
config, bash code blocks for commands, troubleshooting section, ~200-300 lines.

## Output

`docs/reference/docker.md`

## Sections

### 1. Setup

- Prerequisites: Docker with BuildKit, secrets in `.env`
- First build walkthrough (dev-live target)

### 2. Building Images

- `make docker-build-dev-live` — local dev (volume-mounted source)
- `make docker-build-dev-snapshot` — self-contained CI image
- Build ARGs table (from `configs/image/dev-snapshot.yaml` and Makefile)
- BuildKit secrets: how GIT_PAT, R2, W&B are injected

### 3. Running Containers

- MODE=idle — attach bash to debug
- MODE=passthrough — CI smoke tests, ad-hoc commands
- Volume mounting for dev-live
- Cross-ref to docker-spec.md for planned modes

### 4. CI Workflow

- GHA `docker-build-validation.yml` overview
- `image_config.py` config loading (Pydantic-validated)
- DockerHub push flow (docker/metadata-action + docker/build-push-action)
- Manual trigger: `gh workflow run`

### 5. Debugging

- Shell into running container
- Headless VST issues (`scripts/run-linux-vst-headless.sh`)
- OOM during builds (runner sizing)
- BuildKit cache management

### 6. Cross-references

- `docs/reference/docker-spec.md` — image target contract, entrypoint modes
- `docs/reference/rclone.md` — R2 setup (Docker section)
- `docs/reference/wandb-integration.md` — W&B logging
- `docs/design/data-pipeline.md` — pipeline architecture context

## Source Material

| Source                                          | What to extract                   |
| ----------------------------------------------- | --------------------------------- |
| `Makefile` (lines 53-139)                       | Build targets, variables, secrets |
| `docker/ubuntu22_04/Dockerfile`                 | Stage architecture, build ARGs    |
| `scripts/docker_entrypoint.sh`                  | MODE dispatch, current behavior   |
| `scripts/run-linux-vst-headless.sh`             | Headless X11 bootstrap            |
| `configs/image/dev-snapshot.yaml`               | Image config values               |
| `scripts/image_config.py`                       | Config schema, loader API         |
| `.github/workflows/docker-build-validation.yml` | CI workflow steps                 |

## Scope Boundaries

- Document main branch behavior only (not experiment branch modes)
- Reference docker-spec.md for planned/future behavior
- No architecture selection guidance (amd64-only for now)
- No security hardening section (covered adequately in rclone.md and docker-spec.md)
