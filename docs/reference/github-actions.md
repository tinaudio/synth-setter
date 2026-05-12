# GitHub Actions Reference

Project-specific knowledge for the workflows in [.github/workflows/](../../.github/workflows/). This doc documents **what the YAML can't tell you** — intent, secret purposes, cross-workflow dependencies, and non-obvious gotchas. For literal triggers, runners, and steps, read the YAML.

All workflows run on GitHub-hosted runners. `test-gpu` uses the `gpu-x64` larger runner (GitHub-hosted, GPU-equipped); everything else uses standard labels (`ubuntu-latest`, `ubuntu-latest-4core`, `ubuntu-22.04`, `macos-latest`).

For GitHub Actions concepts, see [GitHub's docs](https://docs.github.com/en/actions).

## Workflow catalog

### CI & quality

| Workflow                  | Purpose                                                                                                                 | Gotcha                                                                                                                                                                                                                                                                        |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test`                    | Runs non-slow pytest on Ubuntu + macOS across Python 3.10/3.11, plus a coverage job.                                    |                                                                                                                                                                                                                                                                               |
| `test-gpu`                | Runs GPU-marked pytest on the `gpu-x64` GitHub-hosted GPU runner.                                                       | Pins `torch<2.7.0` for CUDA 12.8 compatibility. See [GPU runner torch pin](#gpu-runner-torch-pin).                                                                                                                                                                            |
| `test-mps`                | Runs MPS-marked pytest on a `macos-latest` (Apple Silicon) runner.                                                      | Pre-submit signal for slow Surge XT tests on PRs touching `src/` or `configs/`; the standard ubuntu PR runner OOMs the PyTorch CPU forward.                                                                                                                                   |
| `cpu-slow`                | Runs `slow`-marked pytest (excluding `gpu`, `mps`, and `requires_vst`) on a larger ubuntu runner, post-merge on `main`. | Skips docs-only merges (`paths-ignore`). Concurrency-grouped to queue overlapping runs. See [Concurrency](#concurrency). VST-marked tests run separately in `test-vst-slow.yml`. Auto-files a `ci-automation` Bug to `@ktinubu` on post-merge failure (with dedupe by title). |
| `code-quality-pr`         | Runs pre-commit hooks on files changed in the PR.                                                                       |                                                                                                                                                                                                                                                                               |
| `code-quality-main`       | Runs pre-commit hooks on all files after merge to main.                                                                 | Skips `no-commit-to-branch` hook (would reject main commits).                                                                                                                                                                                                                 |
| `pr-metadata-gate`        | Enforces that every PR links a taxonomy-compliant issue (type, label, milestone, Epic lineage).                         | Walks issue parent chain up to 4 levels; falls back to Epic check if GraphQL parent field unavailable.                                                                                                                                                                        |
| `bats-tests`              | Runs BATS tests against shell scripts under `scripts/` and `tests/`.                                                    |                                                                                                                                                                                                                                                                               |
| `docker-build-validation` | Builds the dev-snapshot Docker image, optionally pushes to Docker Hub, runs smoke tests.                                | Image is public and ships no credentials; R2/W&B creds flow in at runtime. See [Public image, runtime secrets](#public-image-runtime-secrets).                                                                                                                                |

### Pipeline

| Workflow                    | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Gotcha                                                                                                                                                                                                                 |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `generate-dataset-shards`   | Reusable + dispatchable workflow: invokes `src.pipeline.skypilot_launch` for one provider (skypilot-local kind / runpod / oci); the launcher uploads the materialized spec to R2 and the workflow emits a `spec_uri` output that downstream callers (`validate-dataset-shards`) consume. A run-metadata artifact is also uploaded for debugging. Consumed by `test-dataset-generation`; also fireable directly via `gh workflow run` or the Actions UI (autogenerates a manual cluster name when `cluster_name` is left blank). | Mounts PR code as a volume into the container for runpod/oci rows. See [Mount-as-volume pattern](#mount-as-volume-pattern).                                                                                            |
| `validate-dataset-shards`   | Reusable + dispatchable workflow: structural spec validation (`src.pipeline.ci.validate_spec`) + per-shard HDF5 validation (`src.pipeline.ci.validate_shard`) against shards downloaded from R2. Consumes the `spec_uri` (`r2://`) workflow output emitted by `generate-dataset-shards` (or any caller that uploaded a spec to a known r2:// URI). All rclone copies happen inside `src.pipeline.r2_io`. Also dispatchable directly with a hand-supplied `spec_uri`.                                                            |                                                                                                                                                                                                                        |
| `test-dataset-generation`   | Dynamic matrix over the data-pipeline smoke. Same-repo PR events run **`local`-only** (in-runner container, no cloud creds — only `R2_*`); fork PRs run an empty matrix (no secrets); `workflow_dispatch` selects via `inputs.provider` (`all` = `runpod` + `oci`; or `local` / `runpod` / `oci`). Validates the resulting spec and shard per provider.                                                                                                                                                                         | Only `runpod` / `oci` matrix cells provision billable capacity (a RunPod pod and/or OCI VM) and need `RUNPOD_API_KEY` + the six `OCI_*` secrets; `local` cells run inside the dev-snapshot container on the GH runner. |
| `test-skypilot-debug`       | `workflow_dispatch`-only canary matrix for the SkyPilot+RunPod path. Each variant adds one production-relevant capability (file mounts, headless wrapper, rclone, Python interpreter, launcher CLI, etc.) so a regression's failure pattern points at the responsible layer. See the workflow header for the authoritative variant list.                                                                                                                                                                                        | Each dispatch spawns one billable RunPod pod per matrix variant. No `push` trigger by design.                                                                                                                          |
| `test-skypilot-local`       | Smoke for SkyPilot's local-Kubernetes (`sky local up`) backend on a GH runner: spins up a kind cluster, registers it as the SkyPilot `kubernetes` cloud, and launches `local-debug-rclone-template.yaml` to verify R2 env-var forwarding via `task.update_envs`. Triggers on `workflow_dispatch` and on `pull_request` events that touch the workflow file or the local-debug template.                                                                                                                                         | No billable cloud capacity; only `R2_*` secrets needed. ~5–10 min per dispatch. PR runs from forks are skipped via job-level `if:` since they can't read the R2 secrets.                                               |
| `spec-materialization`      | Reusable workflow: materializes and structurally validates a DatasetSpec in Docker.                                                                                                                                                                                                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                        |
| `test-spec-materialization` | Exercises `spec-materialization` and validates test-specific config values.                                                                                                                                                                                                                                                                                                                                                                                                                                                     |                                                                                                                                                                                                                        |
| `flush-investigation`       | Runs the `flush-investigation.ipynb` notebook in Docker, uploads rendered HTML + audio as an artifact (90-day retention).                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                                                                                                                                                                        |

### Release & versioning

| Workflow          | Purpose                                                                                                           | Gotcha                                                                                                                                       |
| ----------------- | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `release`         | Semantic release on push to main: bumps version from conventional commits, tags, publishes release and changelog. | Concurrency group serializes releases (`cancel-in-progress: false`); skips commits titled `chore(release)`. See [Concurrency](#concurrency). |
| `release-drafter` | Maintains a rolling draft release from merged PR labels.                                                          |                                                                                                                                              |

### Scheduled

| Workflow  | Purpose                                                                        | Gotcha |
| --------- | ------------------------------------------------------------------------------ | ------ |
| `nightly` | Runs the full pytest suite (including `slow`-marked tests) daily at 06:00 UTC. |        |

### Housekeeping & automation

| Workflow          | Purpose                                                                                                               | Gotcha                                                                                                                                 |
| ----------------- | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `auto-approve`    | Auto-approves PRs once all CI checks pass, Copilot has reviewed, threads are resolved, and the author is allowlisted. | Deduplicates check-runs by name (re-runs share a name but have distinct IDs). See [Check-run deduplication](#check-run-deduplication). |
| `stale`           | Labels issues/PRs inactive for 120 days as stale. Never auto-closes (`days-before-close: -1`).                        |                                                                                                                                        |
| `snooze-issue`    | Lets an issue comment snooze the issue for N days.                                                                    |                                                                                                                                        |
| `unsnooze-issues` | Daily job that unsnoozes issues whose snooze window has elapsed.                                                      |                                                                                                                                        |

## Dependency map

**Reusable workflow calls (`workflow_call`):**

- `test-spec-materialization` calls `spec-materialization`
- `test-dataset-generation` calls `generate-dataset-shards` (per provider for `skypilot-local` / `runpod` / `oci`) and `validate-dataset-shards` (per provider for the validate matrix). The `local` cell stays inline — it doesn't go through the reusable; it runs `src.pipeline.ci.materialize_spec` + `scripts/docker_entrypoint.py generate_dataset` inside the dev-snapshot image on the GH runner.

**Workflow-run triggers (`workflow_run`):**

- `auto-approve` triggers on completion of: `Tests`, `Code Quality PR`

**Check-run triggers (`check_run`):**

- `auto-approve` triggers on completion of any check-run on a PR commit (covers all CI workflows beyond the two listed under `workflow_run`, including ones added later). Self-loop on its own `Auto-approve status` check-run is suppressed by a job-level `if:` guard.

**Scheduled triggers (`schedule`):**

- `auto-approve` runs hourly (`0 * * * *`) as a safety net. Catches blockers that clear without firing any workflow event — most notably, Copilot review threads being resolved (GitHub does not expose `pull_request_review_thread` as a workflow trigger). On a scheduled run the `discover` job paginates the open-PR list and enumerates every eligible PR; the `auto-approve` matrix re-evaluates each.

**Artifact chains (`upload-artifact` → `download-artifact`):**

- `test-dataset-generation` still writes a per-provider `test-run-metadata-<provider>` artifact for failure debugging, but the `validate-dataset-shards` jobs no longer consume it — they read the spec from R2 at `r2://<bucket>/skypilot-launcher-specs/<cluster_name>.json` (the `spec_uri` output of `generate-dataset-shards`, or the same path written by the inline `local` cell's explicit upload step).
- `spec-materialization` uploads spec → `test-spec-materialization` downloads and validates
- `flush-investigation` uploads notebook HTML (terminal; not consumed by another workflow)

## Secrets & variables

All secrets are repo-scoped (no workflow uses an `environment:` block). No custom variables (`${{ vars.* }}`) are in use.

| Name                                      | Used by                                                                                                                                                             | Purpose                                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ANTHROPIC_API_KEY`                       | (currently unused)                                                                                                                                                  | Previously consumed by `claude-review`, which was removed. Secret is kept registered for possible future revival.                                                                                                                                                                                                                                                       |
| `APPROVAL_BOT_APP_ID`                     | `auto-approve`, `release`                                                                                                                                           | GitHub App ID for the approval-bot (issues approval reviews; writes release commits past branch protection).                                                                                                                                                                                                                                                            |
| `APPROVAL_BOT_PRIVATE_KEY`                | `auto-approve`, `release`                                                                                                                                           | GitHub App private key paired with `APPROVAL_BOT_APP_ID`.                                                                                                                                                                                                                                                                                                               |
| `DOCKERHUB_USERNAME`                      | `docker-build-validation`                                                                                                                                           | Docker Hub login for pushing the public image (pulls are anonymous).                                                                                                                                                                                                                                                                                                    |
| `DOCKERHUB_TOKEN`                         | same as above                                                                                                                                                       | Docker Hub token paired with `DOCKERHUB_USERNAME`.                                                                                                                                                                                                                                                                                                                      |
| `RCLONE_CONFIG_R2_ACCESS_KEY_ID`          | `generate-dataset-shards`, `validate-dataset-shards`, `spec-materialization`, `test-dataset-generation`, `test-skypilot-debug`, `test-skypilot-local`, `check-auth` | Cloudflare R2 credentials passed as runtime env vars to `docker run` (rclone reads `RCLONE_CONFIG_R2_*` directly).                                                                                                                                                                                                                                                      |
| `RCLONE_CONFIG_R2_SECRET_ACCESS_KEY`      | same as above                                                                                                                                                       | Paired with `RCLONE_CONFIG_R2_ACCESS_KEY_ID`.                                                                                                                                                                                                                                                                                                                           |
| `RCLONE_CONFIG_R2_ENDPOINT`               | `generate-dataset-shards`, `validate-dataset-shards`, `spec-materialization`, `test-dataset-generation`, `test-skypilot-debug`, `test-skypilot-local`, `check-auth` | R2 endpoint URL (runtime).                                                                                                                                                                                                                                                                                                                                              |
| `R2_ACCOUNT_ID`                           | `test-dataset-generation`, `test-skypilot-debug`, `check-auth`                                                                                                      | Cloudflare account ID; written to `~/.cloudflare/accountid` by `scripts/skypilot_write_provider_creds.sh` for SkyPilot's R2 storage adaptor (#749). Required by every launcher invocation (`skypilot-local`, `runpod`, `oci`). No rclone-prefixed alias — must be set under this name.                                                                                  |
| `RUNPOD_API_KEY`                          | `test-dataset-generation` (`runpod` cells only), `test-skypilot-debug`                                                                                              | RunPod API token; written to `~/.runpod/config.toml` by `scripts/skypilot_write_provider_creds.sh` so SkyPilot can provision pods on demand. Skipped when `SKYPILOT_API_SERVER_ENDPOINT` is set — the remote API server holds provider creds (#785).                                                                                                                    |
| `SKYPILOT_API_SERVER_ENDPOINT` (optional) | `test-dataset-generation`, `test-skypilot-debug`                                                                                                                    | URL of a remote SkyPilot API server with HTTP basic auth embedded. When set, all `sky.*` calls dispatch to that server (which has its own provider creds), so the local cred bootstrap (`scripts/skypilot_write_provider_creds.sh`) is skipped — see [`docs/operations/skypilot-remote-server-playbook.md`](../operations/skypilot-remote-server-playbook.md) and #785. |
| `OCI_USER_OCID`                           | `test-dataset-generation` (`oci` cells only)                                                                                                                        | OCI user OCID (Identity → Domains → Users); written to `~/.oci/config`.                                                                                                                                                                                                                                                                                                 |
| `OCI_TENANCY_OCID`                        | `test-dataset-generation` (`oci` cells only)                                                                                                                        | OCI tenancy OCID (root account identifier); written to `~/.oci/config`.                                                                                                                                                                                                                                                                                                 |
| `OCI_FINGERPRINT`                         | `test-dataset-generation` (`oci` cells only)                                                                                                                        | API signing key fingerprint paired with `OCI_API_KEY_PEM`; written to `~/.oci/config`.                                                                                                                                                                                                                                                                                  |
| `OCI_REGION`                              | `test-dataset-generation` (`oci` cells only)                                                                                                                        | OCI region identifier (e.g. `us-ashburn-1`); written to `~/.oci/config`.                                                                                                                                                                                                                                                                                                |
| `OCI_API_KEY_PEM`                         | `test-dataset-generation` (`oci` cells only)                                                                                                                        | Full PEM of the API signing private key; written to `~/.oci/oci_api_key.pem`.                                                                                                                                                                                                                                                                                           |
| `WANDB_API_KEY`                           | `generate-dataset-shards`                                                                                                                                           | W&B credentials forwarded to the worker by the launcher (`runpod` / `oci` rows; the validate workflow doesn't need it).                                                                                                                                                                                                                                                 |

## Common operations

**Manually trigger a workflow:**

```
gh workflow run <workflow-name>.yml
```

Or use the Actions tab → select workflow → *Run workflow*. Workflows with `workflow_dispatch:` inputs accept `-f name=value`.

**Re-run a failed job:**

```
gh run rerun <run-id> --failed
```

Or use the Actions tab UI.

**Skip CI:** No supported `[skip ci]` convention is configured. Whether CI runs is controlled by each workflow's triggers and `paths:` filters, so doc-only changes may not run the full matrix.

**Add a new workflow:** Copy the closest existing workflow as a template (e.g. [bats-tests.yml](../../.github/workflows/bats-tests.yml) for a simple CI job). Both `.yml` and `.yaml` extensions are present in the repo; either works. After writing, invoke the `gha-workflow-validator` skill and run `make format` to validate.

## Known gotchas

### Concurrency

`release` and `cpu-slow` use concurrency groups (both `cancel-in-progress: false`). Runs **queue** rather than cancel — back-to-back pushes to main produce sequential releases and sequential slow-test runs, not coalesced ones. No other workflow uses concurrency, so multiple pushes can run multiple CI matrices simultaneously.

### GPU runner torch pin

`test-gpu` runs on `gpu-x64`, a GitHub-hosted GPU larger runner (NVIDIA driver 12080 / CUDA 12.8). It pins `torch<2.7.0` via a constraint file passed to `uv pip install --constraint`, because torch 2.7+ requires CUDA 13.x. The pin is applied at install time so `requirements-torch.txt` doesn't need to change. If the runner's driver is upgraded to CUDA 13.x, drop the pin.

### Public image, runtime secrets

`docker-build-validation` publishes `tinaudio/synth-setter` as a public image. The image contains no baked credentials and the build uses no BuildKit secrets — the public repo is fetched anonymously. R2 + W&B credentials and the target R2 bucket name are passed in at runtime as env vars (`RCLONE_CONFIG_R2_*`, `WANDB_API_KEY`, `R2_BUCKET`). Pipeline workflows (`generate-dataset-shards`, `spec-materialization`, `test-dataset-generation`) pull the image anonymously and pipe credentials via `docker run --env-file` or `-e`.

### Mount-as-volume pattern

`generate-dataset-shards` does not bake the PR's code into the Docker image. Instead, it pulls a pre-built image from Docker Hub (which contains the Surge XT environment + dependencies) and mounts the PR's checkout as a volume at runtime. This lets CI test branch code against the published image without rebuilding. Implication: changes to the Dockerfile itself are not exercised by this workflow — `docker-build-validation` covers that.

### Check-run deduplication

`auto-approve` reads check-run statuses via the Checks API. When a job is re-run, GitHub preserves the superseded check-run and adds a new one with the same name but a newer ID. The workflow deduplicates with `group_by(.name) | map(sort_by(.id) | last)` — without this, stale failing check-runs would block approval indefinitely. Keep this in mind when adding new gating checks.

### Approval bot token

`release` writes commits (version bumps) and tags to main, which is branch-protected. It uses the approval-bot App token (`APPROVAL_BOT_APP_ID` + `APPROVAL_BOT_PRIVATE_KEY`) to bypass protection rules that block `GITHUB_TOKEN`. If the App is rotated or revoked, releases will silently stop publishing.

### PR metadata gate epic traversal

`pr-metadata-gate` walks the issue parent chain (sub-issue hierarchy) up to 4 levels looking for an Epic. If GitHub's GraphQL `parent` field is unavailable for the auth token, it falls back to checking whether the issue itself is an Epic. Orphan issues — those not under any Epic — fail the gate.

## Keeping this doc in sync

Update this doc when:

- A workflow is added or removed → add/remove the catalog row and any dependency-map entries.
- A custom secret or variable is added, removed, or repurposed → update the secrets table.
- A workflow's *purpose* changes (not its triggers/runner/steps) → update the catalog row.
- A non-obvious failure mode is discovered → add a gotcha.

Trigger, runner, and step changes do **not** require doc updates — those live in the YAML.

**Audit commands** (run periodically to catch drift):

```bash
# Every workflow file should appear as a catalog row
ls .github/workflows/*.y*ml

# Every custom secret should appear in the secrets table
grep -rho 'secrets\.[A-Z_0-9]\+' .github/workflows | grep -v GITHUB_TOKEN | sort -u

# Every custom variable should appear (currently none)
grep -rho 'vars\.[A-Z_0-9]\+' .github/workflows | sort -u

# All distinct runner labels
grep -h 'runs-on:' .github/workflows/*
```
