# Docker Specification Reference

> **Status**: Spec — describes target behavior; see § Current vs. Planned for delta from `main`
> **Tracking**: #265, #272, #273, #287, #288

______________________________________________________________________

## Current vs. Planned

MODE dispatch is fully implemented in `scripts/docker_entrypoint.sh` on `main`.
MODE dispatch was implemented as part of [#265](https://github.com/tinaudio/synth-setter/issues/265).
The entrypoint supports three modes (`idle`, `passthrough`, `generate_dataset`),
exits with an error if MODE is unset or unknown, and exits 0 for `passthrough`
with no arguments. The spec below matches the current implementation.

______________________________________________________________________

## 1. Entrypoint MODE Dispatch

The entrypoint (`scripts/docker_entrypoint.sh`) dispatches on the `MODE` env var.
MODE is required -- container errors if unset.

| MODE               | Args    | Behavior                                                                                   | Use case                                       |
| ------------------ | ------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------- |
| `idle`             | ignored | `exec sleep infinity`                                                                      | Attach bash to debug container                 |
| `passthrough`      | given   | `exec "$@"`                                                                                | CI smoke tests, ad-hoc commands, training/eval |
| `passthrough`      | none    | exit 0                                                                                     | CI steps that just need success                |
| `generate_dataset` | none    | Runs VST dataset generation via `pipeline.entrypoints.generate_dataset` under headless X11 | CI dataset generation workflow                 |
| *(unset)*          | any     | error                                                                                      | Footgun prevention                             |
| *(unknown)*        | any     | error                                                                                      | Typo prevention                                |

`generate_dataset` uses env vars instead of CLI args — see § MODE=generate_dataset env vars below.

> **Note:** `generate_dataset` is the current single-shard MVP. It will be deprecated when `generate-shards` lands on main ([#411](https://github.com/tinaudio/synth-setter/issues/411)).

### Exit codes

| Condition                  | Exit code |
| -------------------------- | --------- |
| Unset or empty MODE        | 1         |
| Unknown MODE value         | 1         |
| `passthrough` with no args | 0         |

### Next modes

| MODE              | Status                     | Description                                                                  | Tracking                                                    |
| ----------------- | -------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `generate-shards` | Scoped (experiment branch) | Multi-shard parallel generation with R2 upload. Replaces `generate_dataset`. | [#407](https://github.com/tinaudio/synth-setter/issues/407) |
| `finalize-shards` | Scoped (experiment branch) | Reshard into train/val/test, compute stats, upload to R2.                    | [#408](https://github.com/tinaudio/synth-setter/issues/408) |
| `train`           | Scoped (experiment branch) | Download dataset from R2, run training, upload checkpoints.                  | [#409](https://github.com/tinaudio/synth-setter/issues/409) |
| `eval`            | Planned                    | Download checkpoint + dataset, run eval, upload results.                     | [#410](https://github.com/tinaudio/synth-setter/issues/410) |

______________________________________________________________________

## 2. Image Targets

`docker/ubuntu22_04/Dockerfile` defines two consumable targets via `--target`:

| Target               | Entrypoint             | Source code            | Use case                                          |
| -------------------- | ---------------------- | ---------------------- | ------------------------------------------------- |
| `dev-snapshot`       | `docker_entrypoint.sh` | Git clone at `GIT_REF` | CI, cloud runs                                    |
| `devcontainer-tools` | *(inherits)*           | Git clone at `GIT_REF` | Dev container base (CLI tools + non-root `dev`)   |

The `dev-snapshot` target inherits directly from
`builder-install-synth-setter-deps`. It contains no baked credentials
and no baked runtime configuration. R2 credentials, the W&B API key,
and the target R2 bucket name are all provided at runtime via env vars
(see `docs/reference/docker.md` § Runtime secrets).

The `devcontainer-tools` target extends `dev-base` with `gh`, `jq`, Node.js,
`@anthropic-ai/claude-code` (installed system-wide), a non-root `dev` user,
and a `/commandhistory` directory for persisted bash history. It is consumed
by `.devcontainer/Dockerfile` as the base for local and Codespaces dev
containers.

## 3. Environment Variables

### Build ARGs

| ARG                          | Default        | Purpose                                       |
| ---------------------------- | -------------- | --------------------------------------------- |
| `IMAGE`                      | `dev-snapshot` | Selects final target (`dev-snapshot`)         |
| `SYNTH_PERMUTATIONS_GIT_REF` | `main`         | Git ref for source code                       |
| `SURGE_GIT_REF`              | *(pinned SHA)* | Surge XT release commit                       |
| `BUILD_MODE`                 | `source`       | `source` or `prebuilt` (Surge install method) |
| `TORCH_BACKEND`              | `cu128`        | PyTorch backend for uv (e.g. cu128, cpu)      |

### Baked ENV vars (available at runtime)

| Variable                     | Set in targets | Value                                |
| ---------------------------- | -------------- | ------------------------------------ |
| `SYNTH_PERMUTATIONS_GIT_REF` | `dev-snapshot` | The git ref the image was built from |
| `VIRTUAL_ENV`                | `dev-snapshot` | `/venv/main`                         |
| `PATH`                       | `dev-snapshot` | `$VIRTUAL_ENV/bin:$PATH`             |

### Runtime env vars (full enumeration)

This table is canonical: every env var the image consumes at runtime is
listed here. Identical to `docs/reference/docker.md` § Runtime environment
variables — kept in sync as the single source of truth.

| Env var                              | Consumer              | Required for                  | Notes                                       |
| ------------------------------------ | --------------------- | ----------------------------- | ------------------------------------------- |
| `MODE`                               | entrypoint dispatcher | all modes                     | Set via `-e MODE=<mode>`; required          |
| `DATASET_CONFIG`                     | `generate_dataset`    | `MODE=generate_dataset`       | Path to dataset config YAML (in image)      |
| `RUN_METADATA_DIR`                   | `generate_dataset`    | `MODE=generate_dataset` (opt) | Defaults to `/run-metadata`                 |
| `R2_BUCKET`                          | `generate_dataset`    | any rclone-to-R2 upload       | Bucket name (non-secret); from `.env`       |
| `RCLONE_CONFIG_R2_TYPE`              | rclone                | any rclone R2 op              | Constant: `s3`; from `.env` or `-e`         |
| `RCLONE_CONFIG_R2_PROVIDER`          | rclone                | any rclone R2 op              | Constant: `Cloudflare`; from `.env` or `-e` |
| `RCLONE_CONFIG_R2_ACCESS_KEY_ID`     | rclone                | any rclone R2 op              | **Secret**; from `.env`                     |
| `RCLONE_CONFIG_R2_SECRET_ACCESS_KEY` | rclone                | any rclone R2 op              | **Secret**; from `.env`                     |
| `RCLONE_CONFIG_R2_ENDPOINT`          | rclone                | any rclone R2 op              | **Secret**; from `.env`                     |
| `WANDB_API_KEY`                      | wandb SDK             | any W&B-logging op            | **Secret**; from `.env`                     |

For `MODE=generate_dataset` specifically, the container materializes a
DatasetPipelineSpec, uploads spec and shard to R2. `input_spec.json` is
written to `RUN_METADATA_DIR`. The entrypoint generates `shard_size`
samples (one shard per invocation). Multi-shard generation
(`num_shards > 1`) raises `NotImplementedError`.

rclone's native env-var config synthesizes the `r2` remote in-memory from
the 5 `RCLONE_CONFIG_R2_*` vars — no `rclone.conf` file is read or written.
`R2_BUCKET` is separate from the rclone remote config: it is a bucket-name
argument interpolated into upload paths by `generate_dataset.py`
(`r2:${R2_BUCKET}/...`).

______________________________________________________________________

## 4. Known Design Issues

| #   | Issue                                                      | Impact                            | Tracking |
| --- | ---------------------------------------------------------- | --------------------------------- | -------- |
| 1   | CI workflows use `--entrypoint bash`, bypassing entrypoint | Setup logic skipped in CI         | #287     |
| 2   | BATS entrypoint tests not in CI                            | Entrypoint regressions undetected | #288     |

______________________________________________________________________

## 5. Cross-references

- `docs/design/storage-provenance-spec.md` -- R2 paths, W&B artifacts, secrets
- `docs/design/data-pipeline-implementation-plan.md` -- `MODE=generate-shards` ([#407](https://github.com/tinaudio/synth-setter/issues/407))
- `docs/reference/wandb-integration.md` -- W&B logging reference
