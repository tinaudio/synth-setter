# Credential Rotation Guide

## Overview

This runbook documents how to rotate every credential used by the synth-setter
project. Rotation is required when:

- The repository transitions from private to public (pre-public security audit).
- A credential is suspected of being exposed (CI logs, local `.env` files on
  contributor machines, Docker images containing baked secrets).
- A team member with access leaves the project.
- A credential reaches its scheduled expiration or age threshold.
- A dependency or upstream provider reports a breach.

All credentials listed below are stored as **GitHub Actions repository secrets**
unless otherwise noted. Some are also present in local developer `.env` files
and/or baked into Docker images at build time.

______________________________________________________________________

## Credential Inventory

| Credential                 | Where Used                                     | Storage Locations                                                          |
| -------------------------- | ---------------------------------------------- | -------------------------------------------------------------------------- |
| `R2_ACCESS_KEY_ID`         | Docker builds, pipeline workers, rclone        | GitHub Secrets, `.env`, Docker image (`rclone.conf`)                       |
| `R2_SECRET_ACCESS_KEY`     | Docker builds, pipeline workers, rclone        | GitHub Secrets, `.env`, Docker image (`rclone.conf`)                       |
| `R2_ENDPOINT`              | Docker builds (build-arg, not secret)          | Image config YAML (`configs/image/`), `.env`, Docker image (`rclone.conf`) |
| `WANDB_API_KEY`            | Training, evaluation, promotion, Docker images | GitHub Secrets, `.env`, Docker image (`~/.netrc`)                          |
| `GIT_PAT`                  | Docker builds, CI workflows                    | GitHub Secrets, `.env`                                                     |
| `GITHUB_TOKEN`             | CI workflows (automatic)                       | Automatic per workflow run                                                 |
| `RUNPOD_API_KEY`           | Pipeline orchestration                         | GitHub Secrets, `.env`                                                     |
| `DOCKERHUB_USERNAME`       | CI image push workflows                        | GitHub Secrets                                                             |
| `DOCKERHUB_TOKEN`          | CI image push workflows                        | GitHub Secrets                                                             |
| `APPROVAL_BOT_APP_ID`      | Auto-approve workflow, release workflow        | GitHub Secrets                                                             |
| `APPROVAL_BOT_PRIVATE_KEY` | Auto-approve workflow, release workflow        | GitHub Secrets                                                             |
| `ANTHROPIC_API_KEY`        | Claude review workflow                         | GitHub Secrets                                                             |

______________________________________________________________________

## Rotation Procedures

### Cloudflare R2 (`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`)

**What:** S3-compatible API token for reading/writing pipeline data in Cloudflare R2.

**Where stored:**

- GitHub Secrets: `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`
- Local `.env` files on developer machines
- Baked into Docker images via BuildKit secrets (written to `rclone.conf`)

**Rotation steps:**

1. Log in to the [Cloudflare dashboard](https://dash.cloudflare.com/).
2. Navigate to **R2 > Overview > Manage R2 API Tokens**.
3. Click **Create API token**. Scope it to the `intermediate-data` bucket with
   read/write permissions.
4. Copy the new Access Key ID and Secret Access Key.
5. Update GitHub Secrets:
   - Go to the repo **Settings > Secrets and variables > Actions**.
   - Update `R2_ACCESS_KEY_ID` with the new access key ID.
   - Update `R2_SECRET_ACCESS_KEY` with the new secret access key.
6. Update local `.env` files on all developer machines.
7. Rebuild Docker images (existing images contain old baked credentials):
   ```bash
   make docker-build-dev-snapshot GIT_REF=<sha> GIT_PAT=<token>
   ```
8. Revoke the old API token in the Cloudflare dashboard.

**Verification:**

```bash
# Test rclone access with new credentials
rclone ls r2:intermediate-data/ --max-depth 1 --checksum
```

Confirm that a CI workflow using R2 (e.g., `test-dataset-generation.yml`) passes.

______________________________________________________________________

### R2 Endpoint (`R2_ENDPOINT`)

**What:** The Cloudflare R2 S3-compatible endpoint URL. This is not a secret
(it is a well-known Cloudflare URL), but it is listed here for completeness
since it is passed as a Docker build-arg and stored alongside R2 credentials.

**Where stored:**

- Image config YAML (`configs/image/dev-snapshot.yaml`) — read by CI via
  `pipeline.ci.load_image_config` and passed as a Docker `--build-arg`
- Local `.env` files
- Baked into Docker images (written to `rclone.conf`)

**Rotation:** Only changes if the Cloudflare account ID changes. Update the
value in the image config YAML and `.env` files if the account is migrated.

______________________________________________________________________

### Weights & Biases (`WANDB_API_KEY`)

**What:** API key for logging experiments to Weights & Biases.

**Where stored:**

- GitHub Secrets: `WANDB_API_KEY`
- Local `.env` files
- Baked into Docker images (written to `~/.netrc`)

**Rotation steps:**

1. Log in to [wandb.ai](https://wandb.ai/).
2. Navigate to **Settings > API keys**.
3. Click **Regenerate** to create a new key. (This immediately invalidates the
   old key.)
4. Update GitHub Secrets:
   - Update `WANDB_API_KEY` with the new key.
5. Update local `.env` files on all developer machines.
6. Rebuild Docker images (old images contain the previous key in `~/.netrc`).

**Verification:**

```bash
# Test W&B authentication
WANDB_API_KEY=wapi_xxxxxxxxxxxx wandb login --verify
```

______________________________________________________________________

### GitHub PAT (`GIT_PAT`)

**What:** Fine-grained personal access token used in Docker builds and CI
workflows for accessing the repository source tarball.

**Where stored:**

- GitHub Secrets: `GIT_PAT`
- Local `.env` files

**Rotation steps:**

1. Go to **GitHub > Settings > Developer settings > Fine-grained personal access
   tokens**.
2. Click **Generate new token** (or **Regenerate** on the existing token).
3. Set the required repository access and permissions (repository contents: read).
4. Copy the new token.
5. Update GitHub Secrets:
   - Update `GIT_PAT` with the new token value.
6. Update local `.env` files on developer machines that use Docker builds.
7. If the old token was not regenerated (i.e., you created a new one), delete the
   old token.

**Verification:**

```bash
# Test Docker build with new PAT
make docker-build-dev-snapshot GIT_REF=main GIT_PAT=github_pat_xxxxxxxxxxxx
```

Confirm the `docker-build-validation.yml` workflow passes.

______________________________________________________________________

### GitHub Token (`GITHUB_TOKEN`)

**What:** Automatic token provided by GitHub Actions to each workflow run. Scoped
to the repository with permissions defined in the workflow file.

**Rotation:** This token is **not manually rotated**. It is generated
automatically per workflow run by GitHub. No action is required.

If you need to change its permissions, update the `permissions:` block in the
relevant workflow YAML files.

______________________________________________________________________

### RunPod (`RUNPOD_API_KEY`)

**What:** API key for managing GPU workers on RunPod (worker submission, CRUD).

**Where stored:**

- GitHub Secrets: `RUNPOD_API_KEY`
- Local `.env` files

**Rotation steps:**

1. Log in to [runpod.io](https://www.runpod.io/).
2. Navigate to **Settings > API Keys**.
3. Click **Create API Key**.
4. Copy the new key.
5. Update GitHub Secrets:
   - Update `RUNPOD_API_KEY` with the new key.
6. Update local `.env` files on developer machines.
7. Delete the old API key in the RunPod dashboard.

**Verification:**

```bash
# Test RunPod API access (list pods)
curl -s -H "Authorization: Bearer rpk_xxxxxxxxxxxx" \
  https://api.runpod.io/v2/pods | python3 -m json.tool
```

______________________________________________________________________

### Docker Hub (`DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`)

**What:** Docker Hub credentials for pushing images from CI workflows.

**Where stored:**

- GitHub Secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`

**Rotation steps:**

1. Log in to [Docker Hub](https://hub.docker.com/).
2. Navigate to **Account Settings > Security > Access Tokens**.
3. Click **New Access Token** with read/write permissions.
4. Copy the new token.
5. Update GitHub Secrets:
   - Update `DOCKERHUB_TOKEN` with the new token.
   - Update `DOCKERHUB_USERNAME` if the account changes.
6. Revoke the old access token in Docker Hub.

**Verification:**

```bash
# Test Docker Hub login
echo "dkr_pat_xxxxxxxxxxxx" | docker login -u <username> --password-stdin
```

Confirm that a CI workflow that pushes images (e.g., `docker-build-validation.yml`)
passes.

______________________________________________________________________

### Approval Bot (`APPROVAL_BOT_APP_ID`, `APPROVAL_BOT_PRIVATE_KEY`)

**What:** GitHub App credentials for the auto-approve and release workflows.

**Where stored:**

- GitHub Secrets: `APPROVAL_BOT_APP_ID`, `APPROVAL_BOT_PRIVATE_KEY`

**Rotation steps:**

1. Go to **GitHub > Settings > Developer settings > GitHub Apps**.
2. Select the approval bot app.
3. Under **General > Private keys**, click **Generate a private key**.
4. Download the new `.pem` file.
5. Update GitHub Secrets:
   - Update `APPROVAL_BOT_PRIVATE_KEY` with the contents of the new `.pem` file.
   - `APPROVAL_BOT_APP_ID` does not change unless the app is recreated.
6. Delete the old private key from the GitHub App settings.

**Verification:**

Trigger the `auto-approve.yml` workflow (e.g., by opening a qualifying PR) and
confirm it succeeds.

______________________________________________________________________

### Anthropic (`ANTHROPIC_API_KEY`)

**What:** API key for the Claude review workflow in CI.

**Where stored:**

- GitHub Secrets: `ANTHROPIC_API_KEY`

**Rotation steps:**

1. Log in to the [Anthropic Console](https://console.anthropic.com/).
2. Navigate to **API Keys**.
3. Click **Create Key**.
4. Copy the new key.
5. Update GitHub Secrets:
   - Update `ANTHROPIC_API_KEY` with the new key.
6. Revoke the old key in the Anthropic Console.

**Verification:**

Trigger the `claude-review.yml` workflow (e.g., by opening a PR) and confirm the
review step completes without authentication errors.

______________________________________________________________________

## Rotation Checklist

Use this checklist when performing a full credential rotation (e.g., pre-public
audit):

- [ ] **Cloudflare R2:** Create new API token, update `R2_ACCESS_KEY_ID` and
  `R2_SECRET_ACCESS_KEY` in GitHub Secrets and `.env`, rebuild Docker images,
  revoke old token
- [ ] **W&B:** Regenerate API key, update `WANDB_API_KEY` in GitHub Secrets and
  `.env`, rebuild Docker images
- [ ] **GitHub PAT:** Generate new fine-grained token, update `GIT_PAT` in
  GitHub Secrets and `.env`, delete old token
- [ ] **RunPod:** Create new API key, update `RUNPOD_API_KEY` in GitHub Secrets
  and `.env`, delete old key
- [ ] **Docker Hub:** Create new access token, update `DOCKERHUB_TOKEN` (and
  `DOCKERHUB_USERNAME` if needed) in GitHub Secrets, revoke old token
- [ ] **Approval Bot:** Generate new private key, update
  `APPROVAL_BOT_PRIVATE_KEY` in GitHub Secrets, delete old key
- [ ] **Anthropic:** Create new API key, update `ANTHROPIC_API_KEY` in GitHub
  Secrets, revoke old key
- [ ] **Docker images:** Rebuild all Docker images to replace baked credentials
  (R2 in `rclone.conf`, W&B in `~/.netrc`)
- [ ] **Old Docker images:** Delete or de-list any published images that contain
  old baked credentials
- [ ] **CI verification:** Run a full CI pass to confirm all workflows succeed
  with new credentials
- [ ] **Local `.env` files:** Notify all developers to update their local `.env`
  files

______________________________________________________________________

## Emergency Rotation Procedure

Use this procedure when a credential is known or suspected to be compromised and
must be revoked immediately.

### 1. Revoke first, update second

The priority is to **invalidate the compromised credential immediately**, even
before replacements are in place. This will temporarily break CI and local
workflows.

1. Identify the compromised credential from the inventory table above.
2. Revoke/delete/regenerate it at its source (Cloudflare, W&B, GitHub, RunPod,
   Docker Hub, Anthropic). For credentials that support regeneration (e.g., W&B),
   regenerating simultaneously revokes the old key.
3. If the credential is baked into Docker images (R2, W&B), note that the old
   credential in those images is now invalid. Running containers using those
   images will fail on the next API call.

### 2. Issue replacement credentials

1. Create the new credential at its source.
2. Update GitHub Secrets immediately.
3. Rebuild Docker images if the credential was baked in.

### 3. Notify the team

1. Post in the team channel that an emergency rotation was performed.
2. Specify which credential was rotated.
3. Ask developers to update their local `.env` files.
4. Note any running workloads (RunPod workers, training runs) that may be
   affected.

### 4. Post-incident

1. Determine how the credential was exposed.
2. Check CI logs for accidental credential printing (look for `echo` or debug
   statements that may have leaked values).
3. Review Docker build logs for credential exposure.
4. File an incident report if the exposure was confirmed.

______________________________________________________________________

## Notes

- **R2 endpoint and bucket name** are not secrets. `R2_ENDPOINT` is a
  well-known Cloudflare URL. The bucket name is `intermediate-data`.
- **`GITHUB_TOKEN`** is automatically provisioned per workflow run and does not
  require manual rotation.
- **Docker images with baked credentials** are a secondary exposure surface.
  After rotating R2 or W&B credentials, always rebuild images and ensure old
  images are removed from Docker Hub.
- Never store credentials in code, config files checked into git, or CI logs.
  Use `--secret` (BuildKit) or environment variables.
