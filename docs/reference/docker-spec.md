# Docker Specification Reference

> **Status**: Reference — describes the live `ENTRYPOINT` in `dev-snapshot`.
> **Tracking**: #265, #272, #273

______________________________________________________________________

Naming convention: **CLI is snake_case throughout.** Subcommands (`idle`,
`passthrough`, `generate_dataset`, `render_eval`, `train`) and CLI flags
(`--spec`) use snake_case. No kebab-case.

______________________________________________________________________

## 1. Entrypoint — click group with per-mode spec

`src/synth_setter/tools/docker_entrypoint.py` is the image's live `ENTRYPOINT`: a click
group with five subcommands. Each spec-taking subcommand deserializes its
`--spec` into a mode-specific pydantic model at the container boundary
(parse-don't-validate), then hands off to the downstream.

| Subcommand         | Args                     | Behavior                                                                                                                                          |
| ------------------ | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `idle`             | none                     | `exec sleep infinity`                                                                                                                             |
| `passthrough`      | trailing ARGV (required) | `exec ARGV`; errors on empty                                                                                                                      |
| `generate_dataset` | `--spec PATH`            | Parse PATH as `DatasetSpec`, call `synth_setter.cli.generate_dataset.run(spec)`, then `os._exit(0)` (defensive #735 workaround — bypasses atexit) |
| `render_eval`      | `--spec PATH`            | `click.ClickException` — tracked in [#410](https://github.com/tinaudio/synth-setter/issues/410)                                                   |
| `train`            | `--spec PATH`            | `click.ClickException` — tracked in [#409](https://github.com/tinaudio/synth-setter/issues/409)                                                   |

`generate_dataset` does **not** consume any env vars for its dispatch
inputs. All dataset-run configuration — including the R2 bucket
(`DatasetSpec.r2.bucket`) — flows in through the materialized
spec at `--spec`.

> **Note:** `generate_dataset` is the current MVP (sequential multi-shard, single-worker). It will be deprecated when `generate-shards` lands on main ([#411](https://github.com/tinaudio/synth-setter/issues/411)).

### Headless X11

`generate_dataset` invokes `generate_vst_dataset.py` wrapped in
`docker/ubuntu22_04/run-linux-vst-headless.sh` from inside `run()` (the
audio-rendering layer). The click CLI itself does not start Xvfb —
`idle` and `passthrough` don't pay the bootstrap cost. Callers that
need X11 via `passthrough` (notebook execution, spec materialization
that imports pedalboard) should prepend `docker/ubuntu22_04/run-linux-vst-headless.sh`
to their command.

### Exit codes

| Condition                                   | Exit code                                                                                                                                    |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| No subcommand                               | non-zero                                                                                                                                     |
| Unknown subcommand                          | non-zero                                                                                                                                     |
| `passthrough` with no args                  | non-zero                                                                                                                                     |
| `passthrough` exec failure (missing binary) | non-zero                                                                                                                                     |
| `generate_dataset` missing/unreadable spec  | non-zero                                                                                                                                     |
| `generate_dataset` invalid spec             | non-zero                                                                                                                                     |
| `generate_dataset` success                  | 0 (via `os._exit(0)` — atexit handlers do **not** run; defensive workaround for [#735](https://github.com/tinaudio/synth-setter/issues/735)) |

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

| Target               | Entrypoint                    | Source code            | Use case                                        |
| -------------------- | ----------------------------- | ---------------------- | ----------------------------------------------- |
| `dev-snapshot`       | `python docker_entrypoint.py` | Git clone at `GIT_REF` | CI, cloud runs                                  |
| `devcontainer-tools` | *(inherits)*                  | Git clone at `GIT_REF` | Dev container base (CLI tools + non-root `dev`) |

The `dev-snapshot` target inherits directly from
`builder-install-synth-setter-deps`. It contains no baked credentials
and no baked runtime configuration. R2 credentials and the W&B API key
are provided at runtime via env vars (see `docs/reference/docker.md`
§ Runtime env vars). The target R2 bucket is carried in the
DatasetSpec supplied to `generate_dataset --spec`, not an env var.

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

Set by the Dockerfile and inherited by every published target. Callers may
override any of these at `docker run` time with `-e`.

| Variable                     | Set in targets                       | Value                                                                                             |
| ---------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `SYNTH_PERMUTATIONS_GIT_REF` | `dev-snapshot`                       | The git ref the image was built from                                                              |
| `VIRTUAL_ENV`                | `dev-snapshot`                       | `/venv/main`                                                                                      |
| `PATH`                       | `dev-snapshot`                       | `$VIRTUAL_ENV/bin:$PATH`                                                                          |
| `SYNTH_SETTER_PLUGIN_PATH`   | `dev-snapshot`, `devcontainer-tools` | Absolute path to the baked Surge XT VST3 (`python-base` stage in `docker/ubuntu22_04/Dockerfile`) |

### Runtime env vars — credentials & required overrides

This table enumerates every env var callers **must** supply at `docker run`
time (credentials) plus any var the image expects from outside but does not
bake a default for. Baked defaults that callers may override are listed
under § Baked ENV vars above. Kept in sync with the matching table in
`docs/reference/docker.md`.

| Env var                              | Consumer  | Required for       | Notes                                       |
| ------------------------------------ | --------- | ------------------ | ------------------------------------------- |
| `RCLONE_CONFIG_R2_TYPE`              | rclone    | any rclone R2 op   | Constant: `s3`; from `.env` or `-e`         |
| `RCLONE_CONFIG_R2_PROVIDER`          | rclone    | any rclone R2 op   | Constant: `Cloudflare`; from `.env` or `-e` |
| `RCLONE_CONFIG_R2_ACCESS_KEY_ID`     | rclone    | any rclone R2 op   | **Secret**; from `.env`                     |
| `RCLONE_CONFIG_R2_SECRET_ACCESS_KEY` | rclone    | any rclone R2 op   | **Secret**; from `.env`                     |
| `RCLONE_CONFIG_R2_ENDPOINT`          | rclone    | any rclone R2 op   | **Secret**; from `.env`                     |
| `WANDB_API_KEY`                      | wandb SDK | any W&B-logging op | **Secret**; from `.env`                     |

Dispatch and dataset-run configuration flow via CLI, not env vars: the
subcommand (`generate_dataset`, `idle`, `passthrough`, …) is a positional
arg; the pipeline spec — including the R2 bucket — is read from the JSON
file passed via `--spec`. `input_spec.json` is written by the caller (the
`synth_setter.pipeline.ci.materialize_spec` bootstrap step in CI) to a bind-mounted
directory. Multi-shard generation runs sequentially on a single worker;
distributed parallelism is tracked in [#407](https://github.com/tinaudio/synth-setter/issues/407).

rclone's native env-var config synthesizes the `r2` remote in-memory from
the 5 `RCLONE_CONFIG_R2_*` vars — no `rclone.conf` file is read or written.
The bucket name is **not** part of the rclone remote config: it comes
from `DatasetSpec.r2.bucket` and is interpolated into upload
paths by `generate_dataset.py` (`r2:${spec.r2.bucket}/${spec.r2.prefix}...`,
produced by `spec.r2.rclone_prefix()`).

______________________________________________________________________

## 4. Cross-references

- `docs/design/storage-provenance-spec.md` -- R2 paths, W&B artifacts, secrets
- `docs/design/data-pipeline-implementation-plan.md` -- `MODE=generate-shards` ([#407](https://github.com/tinaudio/synth-setter/issues/407))
- `docs/reference/wandb-integration.md` -- W&B logging reference
