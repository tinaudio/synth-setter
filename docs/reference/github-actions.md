# GitHub Actions Reference

Project-specific knowledge for the workflows in [.github/workflows/](../../.github/workflows/). This doc documents **what the YAML can't tell you** — intent, secret purposes, cross-workflow dependencies, and non-obvious gotchas. For literal triggers, runners, and steps, read the YAML.

All workflows run on GitHub-hosted runners. `test-expensive` uses the `gpu-x64` larger runner (GitHub-hosted, GPU-equipped); everything else uses standard labels (`ubuntu-latest`, `ubuntu-latest-4core`, `ubuntu-22.04`, `macos-latest`).

For GitHub Actions concepts, see [GitHub's docs](https://docs.github.com/en/actions).

## Workflow catalog

### CI & quality

| Workflow                  | Purpose                                                                                         | Gotcha                                                                                                                         |
| ------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `test`                    | Runs non-slow pytest on Ubuntu + macOS across Python 3.10/3.11, plus a coverage job.            | Static MNIST cache key (`mnist-dataset-v1`); macOS excludes `test_mnist_datamodule`. See [Caching](#caching).                  |
| `test-expensive`          | Runs GPU-marked pytest on the `gpu-x64` GitHub-hosted GPU runner.                               | Pins `torch<2.7.0` for CUDA 12.8 compatibility. See [GPU runner torch pin](#gpu-runner-torch-pin).                             |
| `code-quality-pr`         | Runs pre-commit hooks on files changed in the PR.                                               |                                                                                                                                |
| `code-quality-main`       | Runs pre-commit hooks on all files after merge to main.                                         | Skips `no-commit-to-branch` hook (would reject main commits).                                                                  |
| `pr-metadata-gate`        | Enforces that every PR links a taxonomy-compliant issue (type, label, milestone, Epic lineage). | Walks issue parent chain up to 4 levels; falls back to Epic check if GraphQL parent field unavailable.                         |
| `bats-tests`              | Runs BATS tests against shell scripts under `scripts/` and `tests/`.                            |                                                                                                                                |
| `docker-build-validation` | Builds the dev-snapshot Docker image, optionally pushes to Docker Hub, runs smoke tests.        | Aborts if repo is public (baked-in credentials would leak). See [Docker build visibility gate](#docker-build-visibility-gate). |

### Pipeline

| Workflow                    | Purpose                                                                                                                   | Gotcha                                                                                                  |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `dataset-generation`        | Reusable workflow: materializes a DatasetPipelineSpec, generates a shard in Docker, uploads spec and shard to R2.         | Mounts PR code as a volume into the container. See [Mount-as-volume pattern](#mount-as-volume-pattern). |
| `test-dataset-generation`   | Exercises `dataset-generation` with the CI smoke-test config and validates the resulting shard.                           |                                                                                                         |
| `spec-materialization`      | Reusable workflow: materializes and structurally validates a DatasetPipelineSpec in Docker.                               |                                                                                                         |
| `test-spec-materialization` | Exercises `spec-materialization` and validates test-specific config values.                                               |                                                                                                         |
| `flush-investigation`       | Runs the `flush-investigation.ipynb` notebook in Docker, uploads rendered HTML + audio as an artifact (90-day retention). |                                                                                                         |

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
| `claude-review`   | Invokes Claude Code to review a PR and post inline comments when the `needs-claude-review` label is applied.          | Gated to non-fork PRs.                                                                                                                 |
| `stale`           | Labels issues/PRs inactive for 120 days as stale. Never auto-closes (`days-before-close: -1`).                        |                                                                                                                                        |
| `snooze-issue`    | Lets an issue comment snooze the issue for N days.                                                                    |                                                                                                                                        |
| `unsnooze-issues` | Daily job that unsnoozes issues whose snooze window has elapsed.                                                      |                                                                                                                                        |

## Dependency map

**Reusable workflow calls (`workflow_call`):**

- `test-dataset-generation` calls `dataset-generation`
- `test-spec-materialization` calls `spec-materialization`

**Workflow-run triggers (`workflow_run`):**

- `auto-approve` triggers on completion of: `Tests`, `Code Quality PR`, `Claude Code Review`

**Artifact chains (`upload-artifact` → `download-artifact`):**

- `dataset-generation` uploads spec + shard → `test-dataset-generation` downloads and validates
- `spec-materialization` uploads spec → `test-spec-materialization` downloads and validates
- `flush-investigation` uploads notebook HTML (terminal; not consumed by another workflow)

## Secrets & variables

All secrets are repo-scoped (no workflow uses an `environment:` block). No custom variables (`${{ vars.* }}`) are in use.

| Name                       | Used by                                                                                            | Purpose                                                                                                      |
| -------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `ANTHROPIC_API_KEY`        | `claude-review`                                                                                    | Auth for Claude Code to post PR review comments.                                                             |
| `APPROVAL_BOT_APP_ID`      | `auto-approve`, `release`                                                                          | GitHub App ID for the approval-bot (issues approval reviews; writes release commits past branch protection). |
| `APPROVAL_BOT_PRIVATE_KEY` | `auto-approve`, `release`                                                                          | GitHub App private key paired with `APPROVAL_BOT_APP_ID`.                                                    |
| `DOCKERHUB_USERNAME`       | `dataset-generation`, `spec-materialization`, `test-dataset-generation`, `docker-build-validation` | Docker Hub login for pulling/pushing pipeline images.                                                        |
| `DOCKERHUB_TOKEN`          | same as above                                                                                      | Docker Hub token paired with `DOCKERHUB_USERNAME`.                                                           |
| `GIT_PAT`                  | `docker-build-validation`, `flush-investigation`                                                   | PAT baked into images for private-repo access at container runtime.                                          |
| `R2_ACCESS_KEY_ID`         | `docker-build-validation`                                                                          | Cloudflare R2 credentials baked into image for smoke tests.                                                  |
| `R2_SECRET_ACCESS_KEY`     | `docker-build-validation`                                                                          | Paired with `R2_ACCESS_KEY_ID`.                                                                              |
| `WANDB_API_KEY`            | `docker-build-validation`                                                                          | W&B credentials baked into image for smoke tests.                                                            |

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

Only `release` uses a concurrency group (`release-${{ github.ref }}`, `cancel-in-progress: false`). Release runs **queue** rather than cancel — two pushes to main in quick succession produce two sequential releases, not one. No other workflow uses concurrency, so multiple pushes can run multiple CI matrices simultaneously.

### Caching

`test` caches the MNIST dataset under the static key `mnist-dataset-v1` (identical across all three jobs). The key is **not** derived from a lockfile or dataset hash — bump the `v1` suffix by hand if the MNIST source changes. Because the key is stable, dependency upgrades don't invalidate this cache.

### GPU runner torch pin

`test-expensive` runs on `gpu-x64`, a GitHub-hosted GPU larger runner (NVIDIA driver 12080 / CUDA 12.8). It pins `torch<2.7.0` via a constraint file passed to `uv pip install --constraint`, because torch 2.7+ requires CUDA 13.x. The pin is applied at install time so `requirements-torch.txt` doesn't need to change. If the runner's driver is upgraded to CUDA 13.x, drop the pin.

### Docker build visibility gate

`docker-build-validation` aborts immediately if the GitHub repo is public. The image bakes in `GIT_PAT`, R2 credentials, and `WANDB_API_KEY`, so a public repo would leak them via image registry metadata. If the repo is ever made public, this workflow (and the private-registry push) need redesigning.

### Mount-as-volume pattern

`dataset-generation` does not bake the PR's code into the Docker image. Instead, it pulls a pre-built image from Docker Hub (which contains the Surge XT environment + dependencies) and mounts the PR's checkout as a volume at runtime. This lets CI test branch code against the published image without rebuilding. Implication: changes to the Dockerfile itself are not exercised by this workflow — `docker-build-validation` covers that.

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
