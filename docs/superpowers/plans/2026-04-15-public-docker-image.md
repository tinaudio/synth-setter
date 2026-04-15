# Public Docker Image Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the Docker image from `tinaudio/perm` → `tinaudio/synth-setter`, strip all baked credentials (R2, W&B) so the image is safe to publish on the public Docker Hub registry, and wire runtime secret piping into every consumer.

**Architecture:** The `r2-config-base` Dockerfile stage (which writes `/root/.config/rclone/rclone.conf`) and the W&B `~/.netrc` bake block are deleted. `R2_BUCKET` remains as a build ARG (non-sensitive). Callers pass R2 creds and W&B API key at runtime via `docker run --env-file .env` or `-e RCLONE_CONFIG_R2_*=...`. rclone's native env-var config auto-builds the `r2` remote inside the container — no entrypoint translation layer needed. GitHub Actions workflows translate `secrets.R2_ACCESS_KEY_ID` (unchanged) into `RCLONE_CONFIG_R2_ACCESS_KEY_ID` when invoking `docker run`, so no GitHub Secrets get renamed.

**Tech Stack:** Docker / BuildKit / Docker Hub, GitHub Actions, rclone, Cloudflare R2, Pydantic image-config schema, Makefile, Ubuntu 22.04 base image.

**Scope links:** Closes #564. Part of #563 (Phase 5: Public release) and #264 (Epic: end-to-end MVP pipeline).

______________________________________________________________________

## File Structure

Files that change together ship in the same commit. Six logical commits:

1. **Image build strip** — `docker/ubuntu22_04/Dockerfile`, `configs/image/dev-snapshot.yaml`, `Makefile`, test fixtures.
2. **Workflows** — all four GHA workflows that build/pull the image.
3. **Devcontainer** — `.devcontainer/Dockerfile`, `.devcontainer/post-create.sh`.
4. **Reference docs** — `docs/reference/docker.md`, `docker-spec.md`, `github-actions.md`, `wandb-integration.md`.
5. **User-facing docs** — `docs/getting-started.md`, `docs/operations/credential-rotation-guide.md`.
6. **Design docs + plan** — `docs/design/skypilot-compute-integration.md`, `docs/design/data-pipeline-implementation-plan.md`, and this plan file itself.

______________________________________________________________________

## Task 1: Rename image in config YAML + test fixtures

**Files:**

- Modify: `configs/image/dev-snapshot.yaml:7`

- Modify: `tests/pipeline/test_schemas/test_image_config.py:22,229,281`

- Modify: `tests/pipeline/test_ci/test_load_image_config.py:24,35,247,281`

- [ ] **Step 1: Update `configs/image/dev-snapshot.yaml`**

```yaml
# configs/image/dev-snapshot.yaml:7
image: tinaudio/synth-setter
```

- [ ] **Step 2: Update test fixtures (replace_all)**

In both test files, replace every literal `tinaudio/perm` with `tinaudio/synth-setter`. Use `Edit` with `replace_all=true`.

- [ ] **Step 3: Run affected tests**

```bash
cd /tmp/claude/synth-setter-public-docker
make test  # or: pytest tests/pipeline/test_schemas/test_image_config.py tests/pipeline/test_ci/test_load_image_config.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Stage for commit (do not commit yet — commit after Task 3)**

______________________________________________________________________

## Task 2: Strip baked R2 + W&B secrets from Dockerfile

**Files:**

- Modify: `docker/ubuntu22_04/Dockerfile:340-393`

- [ ] **Step 1: Read the current `r2-config-base` and W&B stages**

The current layout:

- Lines ~340–373: `r2-config-base` stage that mounts `r2_access_key_id`, `r2_secret_access_key`, `r2_endpoint` BuildKit secrets and writes `/root/.config/rclone/rclone.conf`.

- Lines ~375–393: `RUN --mount=type=secret,id=wandb_api_key` block that writes `/root/.netrc`.

- Line 401: `FROM r2-config-base AS dev-snapshot` — the dev-snapshot stage inherits from the baked stage.

- [ ] **Step 2: Replace the `r2-config-base` stage with a plain rename**

Replace the entire `r2-config-base` + W&B blocks (lines ~340–393) with a single renamed stage that only sets `R2_BUCKET` as a non-sensitive env var. The comment block must make clear the image is safe to publish.

```dockerfile
# ==========================================
# Runtime config: bake only non-sensitive R2 bucket name. Callers provide
# R2 credentials and W&B API key at runtime via env vars (see
# docs/reference/docker.md § Runtime secrets).
#
# This image contains NO baked credentials and is safe to publish on
# public registries.
# ==========================================
FROM builder-install-synth-setter-deps AS runtime-base
ARG R2_BUCKET
ENV R2_BUCKET=${R2_BUCKET}
```

- [ ] **Step 3: Point `dev-snapshot` at `runtime-base`**

```dockerfile
# docker/ubuntu22_04/Dockerfile:401 (was: FROM r2-config-base AS dev-snapshot)
FROM runtime-base AS dev-snapshot
```

- [ ] **Step 4: Verify no other stage references `r2-config-base`**

```bash
grep -n "r2-config-base\|rclone.conf\|wandb_api_key\|r2_access_key_id\|r2_secret_access_key\|r2_endpoint" docker/ubuntu22_04/Dockerfile
```

Expected: no matches.

- [ ] **Step 5: Build the image locally to verify the Dockerfile parses**

```bash
DOCKER_BUILDKIT=1 docker buildx build \
  -f docker/ubuntu22_04/Dockerfile \
  --target dev-snapshot \
  --build-arg BUILD_MODE=prebuilt \
  --build-arg R2_BUCKET=intermediate-data \
  --build-arg SYNTH_PERMUTATIONS_GIT_REF=$(git rev-parse HEAD) \
  --secret id=git_pat,env=GIT_PAT \
  --platform linux/amd64 \
  -t synth-setter:test \
  --load \
  .
```

(Requires `GIT_PAT` in shell env. Skip this step if not building locally; CI will catch syntax errors.)

- [ ] **Step 6: Confirm the built image has no baked creds**

```bash
docker run --rm --entrypoint cat synth-setter:test /root/.config/rclone/rclone.conf 2>&1 | head
# Expected: "No such file or directory" OR "credentials not configured" placeholder only.
docker run --rm --entrypoint cat synth-setter:test /root/.netrc 2>&1 | head
# Expected: "No such file or directory".
```

______________________________________________________________________

## Task 3: Strip R2/W&B vars from Makefile + commit build changes

**Files:**

- Modify: `Makefile:78,96-106,114-116`

- [ ] **Step 1: Delete the BuildKit secrets block**

Remove lines ~96–106 (the R2/W&B vars and `DOCKER_SECRETS`) and drop `$(DOCKER_SECRETS)` from the `docker-build-dev-snapshot` target at line ~115.

Target end state (show the exact diff):

```makefile
# BEFORE (lines 96-106):
# R2 / rclone configuration — passed as BuildKit secrets + build-arg.
R2_ACCESS_KEY_ID     ?=
R2_SECRET_ACCESS_KEY ?=
R2_ENDPOINT          ?=
WANDB_API_KEY        ?=

DOCKER_SECRETS = \
	--secret id=r2_access_key_id,env=R2_ACCESS_KEY_ID \
	--secret id=r2_secret_access_key,env=R2_SECRET_ACCESS_KEY \
	--secret id=r2_endpoint,env=R2_ENDPOINT \
	--secret id=wandb_api_key,env=WANDB_API_KEY

# AFTER:
# (block deleted entirely — no runtime secrets are baked into the image.)
```

```makefile
# BEFORE (in docker-build-dev-snapshot target, line ~115):
		--secret id=git_pat,env=GIT_PAT \
		$(DOCKER_SECRETS) \
		--build-arg R2_BUCKET=$(R2_BUCKET) \

# AFTER:
		--secret id=git_pat,env=GIT_PAT \
		--build-arg R2_BUCKET=$(R2_BUCKET) \
```

- [ ] **Step 2: Update the `DOCKER_IMAGE` comment at line 78**

```makefile
#   DOCKER_IMAGE        Image name                  (default: synth-setter)
```

(was: `(default: tinaudio/perm)`)

- [ ] **Step 3: Run `make format` and `make test`**

```bash
make format
make test
```

Expected: formatters pass, tests pass.

- [ ] **Step 4: Commit Tasks 1–3 together**

```bash
git add docker/ubuntu22_04/Dockerfile configs/image/dev-snapshot.yaml Makefile \
  tests/pipeline/test_schemas/test_image_config.py \
  tests/pipeline/test_ci/test_load_image_config.py
git commit -m "build(docker): rename to tinaudio/synth-setter and strip baked secrets

Delete the r2-config-base Dockerfile stage and W&B netrc bake block so
the image contains no embedded credentials. R2_BUCKET remains as a
non-sensitive build arg. Callers now provide R2 credentials and
WANDB_API_KEY at runtime via env vars.

Refs #564"
```

______________________________________________________________________

## Task 4: Update `docker-build-validation.yml`

**Files:**

- Modify: `.github/workflows/docker-build-validation.yml`

- [ ] **Step 1: Delete the `visibility-check` job**

Remove the entire `visibility-check:` job (lines ~44–68) and the `needs: visibility-check` on `docker-build`.

- [ ] **Step 2: Drop R2/W&B BuildKit secrets from the build step**

In the `Build and push Docker image` step (~L142–166), replace the `secrets:` block:

```yaml
# BEFORE:
          secrets: |
            git_pat=${{ secrets.GIT_PAT }}
            r2_access_key_id=${{ secrets.R2_ACCESS_KEY_ID }}
            r2_secret_access_key=${{ secrets.R2_SECRET_ACCESS_KEY }}
            r2_endpoint=${{ secrets.R2_ENDPOINT }}
            wandb_api_key=${{ secrets.WANDB_API_KEY }}

# AFTER:
          secrets: |
            git_pat=${{ secrets.GIT_PAT }}
```

- [ ] **Step 3: Update cache refs**

```yaml
# BEFORE (L165-166):
          cache-from: type=registry,ref=tinaudio/perm:buildcache
          cache-to: ${{ github.event_name != 'pull_request' && 'type=registry,ref=tinaudio/perm:buildcache,mode=max' || '' }}

# AFTER:
          cache-from: type=registry,ref=tinaudio/synth-setter:buildcache
          cache-to: ${{ github.event_name != 'pull_request' && 'type=registry,ref=tinaudio/synth-setter:buildcache,mode=max' || '' }}
```

- [ ] **Step 4: Add stable `latest` tag for public releases**

In the `Docker metadata` step (~L133–140), extend the tag list:

```yaml
          tags: |
            type=raw,value=dev-snapshot
            type=raw,value=dev-snapshot-${{ steps.source.outputs.sha }}
            type=raw,value=latest,enable={{is_default_branch}}
```

- [ ] **Step 5: Update smoke-test comments + run lines if any still reference baked creds**

Check that the smoke tests (~L168–183) don't assume baked creds. They shouldn't — they just import torch and load the VST.

- [ ] **Step 6: Validate the workflow YAML**

```bash
# Run actionlint or gha-workflow-validator skill
actionlint .github/workflows/docker-build-validation.yml
```

Expected: clean.

______________________________________________________________________

## Task 5: Update `dataset-generation.yml` with runtime env-var piping

**Files:**

- Modify: `.github/workflows/dataset-generation.yml`

- [ ] **Step 1: Rename image references**

Replace `tinaudio/perm` with `tinaudio/synth-setter` (L44, L60) using `Edit` with `replace_all=true`.

- [ ] **Step 2: Rewrite the `docker run` to pipe R2 + W&B env vars**

Current (L56–71):

```yaml
          docker run --rm \
            -v ${{ github.workspace }}:/home/build/synth-setter \
            -v "${METADATA_DIR}:/run-metadata" \
            --entrypoint bash \
            tinaudio/synth-setter:${{ inputs.image_tag }} \
            -c "..."
```

New (replace the `docker run` line and add the `-e` flags, keep everything inside `-c "..."` unchanged):

```yaml
          docker run --rm \
            -v ${{ github.workspace }}:/home/build/synth-setter \
            -v "${METADATA_DIR}:/run-metadata" \
            -e RCLONE_CONFIG_R2_TYPE=s3 \
            -e RCLONE_CONFIG_R2_PROVIDER=Cloudflare \
            -e RCLONE_CONFIG_R2_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID}" \
            -e RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY}" \
            -e RCLONE_CONFIG_R2_ENDPOINT="${R2_ENDPOINT}" \
            -e WANDB_API_KEY="${WANDB_API_KEY}" \
            --entrypoint bash \
            tinaudio/synth-setter:${{ inputs.image_tag }} \
            -c "..."
```

- [ ] **Step 3: Pull `R2_*`/`WANDB_API_KEY` from `secrets` via `env:`**

Add an `env:` block to the `Generate dataset` step:

```yaml
      - name: Generate dataset
        env:
          R2_ACCESS_KEY_ID: ${{ secrets.R2_ACCESS_KEY_ID }}
          R2_SECRET_ACCESS_KEY: ${{ secrets.R2_SECRET_ACCESS_KEY }}
          R2_ENDPOINT: ${{ secrets.R2_ENDPOINT }}
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          set -o pipefail
          ...
```

- [ ] **Step 4: Drop the `Login to Docker Hub` step if pull-only**

Since the image is public, delete the `docker/login-action` step at the top of the `generate-dataset` job. Keep `docker pull` — it works anonymously on public images.

- [ ] **Step 5: Validate**

```bash
actionlint .github/workflows/dataset-generation.yml
```

______________________________________________________________________

## Task 6: Update `spec-materialization.yml`

**Files:**

- Modify: `.github/workflows/spec-materialization.yml`

- [ ] **Step 1: Rename image references**

Replace `tinaudio/perm` with `tinaudio/synth-setter` (L35, L47) using `Edit` with `replace_all=true`.

- [ ] **Step 2: Pipe runtime R2 env vars to `docker run`**

The `Materialize spec in Docker` step (~L37–60) runs `pipeline.ci.materialize_spec`, which uploads the spec to R2. Add the same `env:` block as Task 5 Step 3, and add the same `-e RCLONE_CONFIG_R2_*` flags to the `docker run` invocation.

- [ ] **Step 3: Drop `Login to Docker Hub`**

Same as Task 5 Step 4 — image is public now.

- [ ] **Step 4: Validate**

```bash
actionlint .github/workflows/spec-materialization.yml
```

______________________________________________________________________

## Task 7: Update `test-dataset-generation.yml`

**Files:**

- Modify: `.github/workflows/test-dataset-generation.yml`

- [ ] **Step 1: Rename image reference at L86**

```yaml
          IMAGE: tinaudio/synth-setter:${{ inputs.docker_image_tag || 'dev-snapshot' }}
```

- [ ] **Step 2: Rewrite `validate-shard` job's `docker run` to pipe rclone env vars**

The `Download shard from R2 and validate` step runs:

```yaml
          docker run --rm \
            -e MODE=passthrough \
            -v /tmp/shard-download:/download \
            "$IMAGE" \
            rclone copy --checksum \
              "r2:${R2_BUCKET}/${R2_PREFIX}${SHARD_FILE}" \
              /download/
```

This depends on baked `rclone.conf`. Replace with env-var piping:

```yaml
          docker run --rm \
            -e MODE=passthrough \
            -e RCLONE_CONFIG_R2_TYPE=s3 \
            -e RCLONE_CONFIG_R2_PROVIDER=Cloudflare \
            -e RCLONE_CONFIG_R2_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID}" \
            -e RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY}" \
            -e RCLONE_CONFIG_R2_ENDPOINT="${R2_ENDPOINT}" \
            -v /tmp/shard-download:/download \
            "$IMAGE" \
            rclone copy --checksum \
              "r2:${R2_BUCKET}/${R2_PREFIX}${SHARD_FILE}" \
              /download/
```

And add `env:` block to the step pulling secrets:

```yaml
      - name: Download shard from R2 and validate
        env:
          R2_ACCESS_KEY_ID: ${{ secrets.R2_ACCESS_KEY_ID }}
          R2_SECRET_ACCESS_KEY: ${{ secrets.R2_SECRET_ACCESS_KEY }}
          R2_ENDPOINT: ${{ secrets.R2_ENDPOINT }}
        run: |
          ...
```

- [ ] **Step 3: Drop `Login to Docker Hub` from `validate-shard`**

Image is public.

- [ ] **Step 4: Validate**

```bash
actionlint .github/workflows/test-dataset-generation.yml
```

- [ ] **Step 5: Commit Tasks 4–7 together**

```bash
git add .github/workflows/docker-build-validation.yml \
  .github/workflows/dataset-generation.yml \
  .github/workflows/spec-materialization.yml \
  .github/workflows/test-dataset-generation.yml
git commit -m "ci: switch workflows to public synth-setter image with runtime secrets

Drop BuildKit R2/W&B secret mounts from docker-build-validation, remove
the private-registry visibility gate, add the 'latest' tag on main, and
switch cache refs to tinaudio/synth-setter:buildcache.

Dataset and spec workflows now pipe R2 credentials into docker run via
RCLONE_CONFIG_R2_* env vars. Docker Hub login is dropped from pull-only
steps since the image is public.

Refs #564"
```

______________________________________________________________________

## Task 8: Update devcontainer

**Files:**

- Modify: `.devcontainer/Dockerfile:1,14-29`

- Modify: `.devcontainer/post-create.sh:4`

- [ ] **Step 1: Rename `FROM` line**

```dockerfile
# .devcontainer/Dockerfile:1
FROM tinaudio/synth-setter:dev-snapshot
```

- [ ] **Step 2: Delete the `/root/.config/rclone` + `/root/.netrc` copy block (lines 14–29)**

Everything from `# R2 (rclone) and W&B credentials are baked into /root...` through the end of the `RUN mkdir -p /home/$USERNAME/.config ...` block. The dev user no longer has baked creds to copy.

- [ ] **Step 3: Update the comment in `.devcontainer/post-create.sh` at L4**

```bash
# (tinaudio/synth-setter:dev-snapshot) already ships all deps, Surge XT, xvfb,
# and rclone — but NOT credentials. R2 and W&B creds must be provided
# at runtime via Codespaces secrets or a mounted .env file.
```

- [ ] **Step 4: Commit**

```bash
git add .devcontainer/Dockerfile .devcontainer/post-create.sh
git commit -m "build(devcontainer): use public synth-setter image, drop baked-cred copy

The base image no longer ships credentials, so the /root -> /home/dev
copy block is dead code. Document that R2/W&B creds must come from
runtime env vars (Codespaces secrets or mounted .env).

Refs #564"
```

______________________________________________________________________

## Task 9: Rewrite reference docs

**Files:**

- Modify: `docs/reference/docker.md`

- Modify: `docs/reference/docker-spec.md`

- Modify: `docs/reference/github-actions.md`

- Modify: `docs/reference/wandb-integration.md`

- [ ] **Step 1: `docs/reference/docker.md`**

Apply the following edits:

- Line 20–21: `.env` secrets list — keep the names but note they're now runtime-only, not baked.

- Lines 28–52 (§1 BuildKit secrets table + warnings): replace with a single "Runtime secrets" section explaining that creds flow in via `docker run --env-file .env` and that `RCLONE_CONFIG_R2_*` auto-configures the `r2` remote. Delete the "R2 and W&B credentials persist in the final image" warning and the "Docker Hub repository must remain private" invariant.

- Line 55, 139, 284, 285, 346: `tinaudio/perm` → `tinaudio/synth-setter`. Use `replace_all=true`.

- Lines 280–286 (Tags table): update image name in the tags column.

- Lines 300–310 (§4 Required secrets): change "baked via BuildKit" to "passed at runtime to workflows that need R2/W&B access".

- Lines 166–168 (run examples): add `--env-file .env` to example `docker run` commands.

- [ ] **Step 2: `docs/reference/docker-spec.md`**

- Line 63: replace the sentence "R2 credentials are baked only when BuildKit secrets are provided..." with: "R2 credentials and the W&B API key are provided at runtime via env vars (see `docs/reference/docker.md` § Runtime secrets). The image itself contains no baked credentials."

- §3 tables (L69–86): no structural changes unless you renamed `SYNTH_PERMUTATIONS_GIT_REF`.

- [ ] **Step 3: `docs/reference/github-actions.md`**

Lines 84–86: rewrite the three rows:

```markdown
| `R2_ACCESS_KEY_ID`     | `dataset-generation`, `spec-materialization`, `test-dataset-generation` | Cloudflare R2 credentials passed as runtime env vars to docker run. |
| `R2_SECRET_ACCESS_KEY` | (same)                                                                  | Paired with `R2_ACCESS_KEY_ID`.                                     |
| `WANDB_API_KEY`        | `dataset-generation`, `spec-materialization`                            | W&B credentials passed as runtime env var to docker run.            |
```

Also: `GIT_PAT` still baked into images? No — only used by `docker-build-validation` at build time for the source tarball. Leave that row alone.

- [ ] **Step 4: `docs/reference/wandb-integration.md`**

Line 164: this is historical (mentions the old `benhayes/synth-permutations` W&B entity). Leave as-is. Grep for any other "baked" W&B language and update.

```bash
grep -n "baked\|netrc" docs/reference/wandb-integration.md
```

If any match: rewrite to runtime.

- [ ] **Step 5: Commit reference docs**

```bash
git add docs/reference/docker.md docs/reference/docker-spec.md \
  docs/reference/github-actions.md docs/reference/wandb-integration.md
git commit -m "docs(reference): document runtime secret piping for public image

Update docker.md, docker-spec.md, and github-actions.md to reflect that
R2 and W&B credentials are no longer baked into the image; they flow in
at runtime via env vars. Remove the 'must remain private' invariant.

Refs #564"
```

______________________________________________________________________

## Task 10: Rewrite user-facing docs

**Files:**

- Modify: `docs/getting-started.md`

- Modify: `docs/operations/credential-rotation-guide.md`

- [ ] **Step 1: `docs/getting-started.md` image rename + sections**

- Line 75: `tinaudio/perm:dev-snapshot` → `tinaudio/synth-setter:dev-snapshot`.

- §2f (Codespaces, L72–103): drop the "one-time, org admin" note about private registry secrets (`*_CONTAINER_REGISTRY_*`). Image is public now; Codespaces pulls anonymously.

- §2g (Local Dev Container, L105–145): rewrite "baked R2/W&B credentials" → "R2/W&B credentials hydrated at runtime from `.env`". Drop the `docker login` requirement.

- §4b (rclone / Cloudflare R2, L240–297): this section stays but move it earlier in the doc. It's now the canonical setup path.

- §7 (Docker Workflow, L422–459): rewrite the "Warning: baked credentials, do not push to public registries" block. The new story: "The image is safe to publish. R2 + W&B creds must be provided at runtime via `--env-file .env`."

- L450–452: update required-env list — `GIT_PAT` remains for builds; `R2_*` and `WANDB_API_KEY` move to runtime.

- [ ] **Step 2: `docs/operations/credential-rotation-guide.md`**

- Lines 25–27 (inventory table): drop `Docker image (rclone.conf)` from the "Storage Locations" column for R2 rows.

- Line 28 (W&B row): drop `Docker image (~/.netrc)` from "Storage Locations".

- R2 rotation procedure (L42–78): delete step 7 "Rebuild Docker images (existing images contain old baked credentials)".

- W&B rotation procedure (L99–125): delete step 6 "Rebuild Docker images (old images contain the previous key in ~/.netrc)".

- R2 endpoint section (L82–97): drop "Baked into Docker images" from storage.

- Summary checklist at L298: drop "rebuild Docker images" from the R2 row.

- [ ] **Step 3: Commit user docs**

```bash
git add docs/getting-started.md docs/operations/credential-rotation-guide.md
git commit -m "docs: document public image and runtime credential flow

Update getting-started and credential-rotation to reflect that the
public image does not bake R2/W&B credentials. Rotation no longer
requires rebuilding images.

Refs #564"
```

______________________________________________________________________

## Task 11: Update design docs + add this plan

**Files:**

- Modify: `docs/design/skypilot-compute-integration.md:124`

- Modify: `docs/design/data-pipeline-implementation-plan.md:134`

- Create: `docs/superpowers/plans/2026-04-15-public-docker-image.md` (this file — already created at start of worktree)

- [ ] **Step 1: Update SkyPilot design doc L124**

```yaml
  image_id: docker:tinaudio/synth-setter:<git-sha>
```

- [ ] **Step 2: Update data-pipeline implementation plan L134**

```bash
  --workers 10 --backend runpod --image tinaudio/synth-setter:dev-snapshot-abc1234
```

- [ ] **Step 3: Commit design docs + plan**

```bash
git add docs/design/skypilot-compute-integration.md \
  docs/design/data-pipeline-implementation-plan.md \
  docs/superpowers/plans/2026-04-15-public-docker-image.md
git commit -m "docs(design): rename tinaudio/perm refs and add migration plan

Refs #564"
```

______________________________________________________________________

## Task 12: Final sweep + verification

- [ ] **Step 1: Grep for remaining `tinaudio/perm` refs**

```bash
cd /tmp/claude/synth-setter-public-docker
rg 'tinaudio/perm' --type-not md || echo "CLEAN (code)"
rg 'tinaudio/perm' docs/ -l
```

Expected: zero code matches. Any doc matches must be historical (CHANGELOG, org-migration-checklist) — leave those.

- [ ] **Step 2: Grep for stale baked-cred language**

```bash
rg -i 'baked (in|into) .* image|private registry|must remain private|rclone\.conf.*baked|netrc.*baked' \
  docs/ .github/ docker/ Makefile README.md CONTRIBUTING.md \
  || echo "CLEAN"
```

Expected: zero matches. Flag any remaining and fix.

- [ ] **Step 3: Run full test + format suite**

```bash
make format
make test
```

Expected: all pass.

- [ ] **Step 4: Validate all workflows with actionlint**

```bash
actionlint .github/workflows/docker-build-validation.yml \
  .github/workflows/dataset-generation.yml \
  .github/workflows/spec-materialization.yml \
  .github/workflows/test-dataset-generation.yml
```

Expected: zero errors.

- [ ] **Step 5: Push the branch and open PR**

```bash
git push -u origin build/public-docker-image
gh pr create --title "build(docker): publish public tinaudio/synth-setter image" \
  --body "$(cat <<'PREOF'
## Summary

Renames the Docker image from `tinaudio/perm` → `tinaudio/synth-setter`, strips all baked credentials (R2 rclone config, W&B netrc), and pipes runtime secrets through consumer workflows. The image is now safe to publish on the public Docker Hub registry.

**Plan / spec:** [`docs/superpowers/plans/2026-04-15-public-docker-image.md`](docs/superpowers/plans/2026-04-15-public-docker-image.md) (also posted as a comment on #564).

## What changed

- **Dockerfile:** Deleted the `r2-config-base` stage and W&B netrc bake. `dev-snapshot` now inherits from `runtime-base`, which only sets the non-sensitive `R2_BUCKET` env var.
- **Makefile:** Dropped `R2_*` / `WANDB_API_KEY` BuildKit secret plumbing.
- **Workflows:** `docker-build-validation` no longer gates on Docker Hub visibility; no longer passes R2/W&B BuildKit secrets; adds `latest` tag on main. `dataset-generation`, `spec-materialization`, and `test-dataset-generation` now pipe `RCLONE_CONFIG_R2_*` env vars into `docker run` and drop the Docker Hub login step (image is public).
- **Devcontainer:** Rebased on the public image; dropped the dead `/root/.config/rclone` + `/root/.netrc` copy block.
- **Docs:** `docker.md`, `docker-spec.md`, `github-actions.md`, `getting-started.md`, `credential-rotation-guide.md`, and two design docs updated. `§ Runtime secrets` story consolidated in `docker.md`.

## Test plan

- [ ] `make format` clean
- [ ] `make test` passes locally
- [ ] `actionlint` clean on all four modified workflows
- [ ] `docker-build-validation` workflow run succeeds on this PR (PR path → build-only validation)
- [ ] Manual dispatch of `docker-build-validation` post-merge pushes `tinaudio/synth-setter:dev-snapshot` + `latest` to Docker Hub
- [ ] `test-dataset-generation` smoke test passes against the new image
- [ ] `docker run --env-file .env tinaudio/synth-setter:latest rclone lsd r2:intermediate-data` works with runtime creds
- [ ] `docker run --rm --entrypoint cat tinaudio/synth-setter:latest /root/.config/rclone/rclone.conf` returns "no such file" (no baked creds)

Closes #564
Part of #563
Part of #264
PREOF
)"
```

- [ ] **Step 6: Post the plan spec as a comment on #564**

```bash
gh issue comment 564 --repo tinaudio/synth-setter --body "$(cat docs/superpowers/plans/2026-04-15-public-docker-image.md)"
```

- [ ] **Step 7: Link the PR in the issue comment thread**

The GitHub `Closes #564` in the PR body will auto-link. No further action needed.

______________________________________________________________________

## Self-Review Checklist

- [x] Every file from the research list (27 files) has a task.
- [x] No `TBD`/`TODO` placeholders.
- [x] Every code change step shows the exact before/after.
- [x] Test fixtures updated alongside the image name.
- [x] Runtime secret story documented in exactly one canonical place (`docs/reference/docker.md` § Runtime secrets).
- [x] No GitHub Secrets renamed — workflows translate `R2_*` → `RCLONE_CONFIG_R2_*` at docker-run time.
- [x] PR references plan doc in-tree AND plan is posted as a comment on #564.
- [x] Smoke-test verification includes both positive (rclone works with runtime creds) and negative (no baked creds in image) checks.
