# Docker Specification Reference

> **Status**: Spec — describes target behavior; see § Current vs. Planned for delta from `main`
> **Tracking**: #265, #272, #273, #287, #288

______________________________________________________________________

## Current vs. Planned

The entrypoint on `main` today (`scripts/docker_entrypoint.sh`) is a passthrough stub:
`exec "$@"` if args are given, error if not. It has no MODE dispatch.

Everything in this spec that differs from that behavior — MODE dispatch, `idle` mode,
`passthrough` with no args exiting 0 — is **planned work** tracked in #265.
The spec documents the target contract, not the current implementation.

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

All targets inherit from `r2-config-base`. R2 credentials are baked only when BuildKit secrets are provided at build time (placeholder rclone config otherwise). W&B auth is not baked — `WANDB_API_KEY` is required at runtime.

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
