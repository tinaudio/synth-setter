# Docker Specification Reference

> **Code version**: `2a3a020` (2026-03-25, `main`)
> **Tracking**: #265, #272, #273, #287, #288

______________________________________________________________________

## 1. Entrypoint MODE Dispatch

The entrypoint (`scripts/docker_entrypoint.sh`) dispatches on the `MODE` env var.
MODE is required -- container errors if unset.

| MODE          | Args    | Behavior              | Use case                                       |
| ------------- | ------- | --------------------- | ---------------------------------------------- |
| `idle`        | ignored | `exec sleep infinity` | Attach bash to debug container                 |
| `passthrough` | given   | `exec "$@"`           | CI smoke tests, ad-hoc commands, training/eval |
| `passthrough` | none    | exit 0                | CI steps that just need success                |
| *(unset)*     | any     | error                 | Footgun prevention                             |
| *(unknown)*   | any     | error                 | Typo prevention                                |

Future modes: `pipeline-worker` (see `docs/design/data-pipeline-implementation-plan.md`).

______________________________________________________________________

## 2. Image Targets

`docker/ubuntu22_04/Dockerfile` defines three targets via `--target`:

| Target         | Entrypoint                      | Source code                  | Use case       |
| -------------- | ------------------------------- | ---------------------------- | -------------- |
| `prod`         | `docker_entrypoint.sh`          | Baked at `GIT_REF` (tarball) | Production     |
| `dev-snapshot` | `docker_entrypoint.sh`          | Git clone at `GIT_REF`       | CI, cloud runs |
| `dev-live`     | fallback (errors without mount) | Volume-mounted               | Local dev      |

All targets inherit from `r2-config-base`, which bakes rclone R2 credentials and W&B auth into the image.

______________________________________________________________________

## 3. Environment Variables

### Build ARGs

| ARG                          | Default        | Purpose                                                   |
| ---------------------------- | -------------- | --------------------------------------------------------- |
| `IMAGE`                      | `dev-snapshot` | Selects final target (`prod`, `dev-snapshot`, `dev-live`) |
| `SYNTH_PERMUTATIONS_GIT_REF` | `main`         | Git ref for source code                                   |
| `SURGE_GIT_REF`              | *(pinned SHA)* | Surge XT release commit                                   |
| `BUILD_MODE`                 | `source`       | `source` or `prebuilt` (Surge install method)             |
| `R2_BUCKET`                  | *(empty)*      | Cloudflare R2 bucket name                                 |
| `TORCH_INDEX_URL`            | *(required)*   | PyTorch wheel index URL                                   |

### Baked ENV vars (available at runtime)

| Variable                     | Set in targets             | Value                                |
| ---------------------------- | -------------------------- | ------------------------------------ |
| `SYNTH_PERMUTATIONS_GIT_REF` | `prod`, `dev-snapshot`     | The git ref the image was built from |
| `R2_BUCKET`                  | all (via `r2-config-base`) | Cloudflare R2 bucket name            |
| `VIRTUAL_ENV`                | all                        | `/venv/main`                         |
| `PATH`                       | all                        | `$VIRTUAL_ENV/bin:$PATH`             |

______________________________________________________________________

## 4. Known Design Issues

| #   | Issue                                                      | Impact                            | Tracking |
| --- | ---------------------------------------------------------- | --------------------------------- | -------- |
| 1   | CI workflows use `--entrypoint bash`, bypassing entrypoint | Setup logic skipped in CI         | #287     |
| 2   | BATS entrypoint tests not in CI                            | Entrypoint regressions undetected | #288     |

______________________________________________________________________

## 5. Cross-references

- `docs/design/storage-provenance-spec.md` -- R2 paths, W&B artifacts, secrets
- `docs/design/data-pipeline-implementation-plan.md` -- future `MODE=pipeline-worker`
- `docs/reference/wandb-integration.md` -- W&B logging reference
