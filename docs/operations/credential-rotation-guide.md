# Credential Rotation Guide

## Overview

This runbook documents how to rotate every credential used by the synth-setter
project. Rotation is required when:

- The repository transitions from private to public (pre-public security audit).
- A credential is suspected of being exposed (CI logs, local `.env` files on
  contributor machines).
- A team member with access leaves the project.
- A credential reaches its scheduled expiration or age threshold.
- A dependency or upstream provider reports a breach.

All credentials listed below are stored as **GitHub Actions repository secrets**
unless otherwise noted. Some are also present in local developer `.env` files.
Runtime R2 and W&B credentials are supplied to Docker containers at run time
via `docker run --env-file .env`; they are not baked into images.

______________________________________________________________________

## Credential Inventory

| Credential                 | Where Used                                        | Storage Locations          |
| -------------------------- | ------------------------------------------------- | -------------------------- |
| `R2_ACCESS_KEY_ID`         | Pipeline workers, rclone (runtime env var)        | GitHub Secrets, `.env`     |
| `R2_SECRET_ACCESS_KEY`     | Pipeline workers, rclone (runtime env var)        | GitHub Secrets, `.env`     |
| `R2_ENDPOINT`              | Pipeline workers, rclone (runtime env var)        | GitHub Secrets, `.env`     |
| `WANDB_API_KEY`            | Training, evaluation, promotion (runtime env var) | GitHub Secrets, `.env`     |
| `GITHUB_TOKEN`             | CI workflows (automatic)                          | Automatic per workflow run |
| `RUNPOD_API_KEY`           | Pipeline orchestration                            | GitHub Secrets, `.env`     |
| `DOCKERHUB_USERNAME`       | CI image push workflows                           | GitHub Secrets             |
| `DOCKERHUB_TOKEN`          | CI image push workflows                           | GitHub Secrets             |
| `APPROVAL_BOT_APP_ID`      | Auto-approve workflow, release workflow           | GitHub Secrets             |
| `APPROVAL_BOT_PRIVATE_KEY` | Auto-approve workflow, release workflow           | GitHub Secrets             |
| `ANTHROPIC_API_KEY`        | Claude review workflow                            | GitHub Secrets             |

______________________________________________________________________

## Rotation Procedures

### Cloudflare R2 (`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`)

**What:** S3-compatible API token for reading/writing pipeline data in Cloudflare R2.

**Where stored:**

- GitHub Secrets: `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`
- Local `.env` files on developer machines

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
7. Revoke the old API token in the Cloudflare dashboard.

**Verification:**

```bash
# Test rclone access with new credentials
rclone ls r2:intermediate-data/ --max-depth 1 --checksum
```

Confirm that a CI workflow using R2 (e.g., `test-dataset-generation.yml`) passes.

______________________________________________________________________

### R2 Endpoint (`R2_ENDPOINT`)

**What:** The Cloudflare R2 S3-compatible endpoint URL. Contains the
permanent Cloudflare account ID — treated as a secret to avoid exposing
the account ID in git history.

**Where stored:**

- GitHub Secrets (`R2_ENDPOINT`) — forwarded to jobs and containers at runtime
- Local `.env` files

**Rotation:** Only changes if the Cloudflare account ID changes. Update the
GitHub Secret and `.env` files if the account is migrated.

______________________________________________________________________

### Weights & Biases (`WANDB_API_KEY`)

**What:** API key for logging experiments to Weights & Biases.

**Where stored:**

- GitHub Secrets: `WANDB_API_KEY`
- Local `.env` files

**Rotation steps:**

1. Log in to [wandb.ai](https://wandb.ai/).
2. Navigate to **Settings > API keys**.
3. Click **Regenerate** to create a new key. (This immediately invalidates the
   old key.)
4. Update GitHub Secrets:
   - Update `WANDB_API_KEY` with the new key.
5. Update local `.env` files on all developer machines.

**Verification:**

```bash
# Test W&B authentication (assumes WANDB_API_KEY is already exported, e.g. from .env)
wandb login --verify
```

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
curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
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
# Test Docker Hub login using environment variables
echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
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

<!-- TODO: stale rotation smoke test; claude-review.yml was removed (see git log), find a replacement verification before the next rotation -->

**What:** API key that was consumed by the `claude-review.yml` workflow (removed from the repo). Currently unused, but the secret is kept registered for a possible future revival of Claude-powered CI review.

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

No automated smoke test is currently available — the previous procedure
(triggering `claude-review.yml` via the `needs-claude-review` label) no
longer works since the workflow was removed. See the TODO above; a
replacement verification is needed before the next rotation.

______________________________________________________________________

## Rotation Checklist

Use this checklist when performing a full credential rotation (e.g., pre-public
audit):

- [ ] **Cloudflare R2:** Create new API token, update `R2_ACCESS_KEY_ID` and
  `R2_SECRET_ACCESS_KEY` in GitHub Secrets and `.env`,
  revoke old token
- [ ] **W&B:** Regenerate API key, update `WANDB_API_KEY` in GitHub Secrets and
  `.env`
- [ ] **RunPod:** Create new API key, update `RUNPOD_API_KEY` in GitHub Secrets
  and `.env`, delete old key
- [ ] **Docker Hub:** Create new access token, update `DOCKERHUB_TOKEN` (and
  `DOCKERHUB_USERNAME` if needed) in GitHub Secrets, revoke old token
- [ ] **Approval Bot:** Generate new private key, update
  `APPROVAL_BOT_PRIVATE_KEY` in GitHub Secrets, delete old key
- [ ] **Anthropic:** Create new API key, update `ANTHROPIC_API_KEY` in GitHub
  Secrets, revoke old key
- [ ] **CI verification:** Run a full CI pass to confirm all workflows succeed
  with new credentials
- [ ] **Local `.env` files:** Notify all developers to update their local `.env`
  files

______________________________________________________________________

## Emergency Rotation Procedure

Use this procedure when a credential is known or suspected to be compromised and
must be revoked immediately.

### 1. Revoke first, update second

The priority is to **invalidate the compromised credential**, even
before replacements are in place. This will temporarily break CI and local
workflows.

1. Identify the compromised credential from the inventory table above.
2. Revoke/delete/regenerate it at its source (Cloudflare, W&B, GitHub, RunPod,
   Docker Hub, Anthropic). For credentials that support regeneration (e.g., W&B),
   regenerating simultaneously revokes the old key.
3. Running containers that loaded the old credential via `--env-file .env` will
   fail on the next API call. Restart them with an updated `.env` once the
   replacement is issued.

### 2. Issue replacement credentials

1. Create the new credential at its source.
2. Update GitHub Secrets and local `.env` files.
3. Restart any running containers so they pick up the new `.env`.

### 3. Create a github issue assigned to ktinubu@ documenting that rotation took place.

______________________________________________________________________

## Notes

- **`GITHUB_TOKEN`** is automatically provisioned per workflow run and does not
  require manual rotation.
- **Runtime credentials** (R2, W&B) are supplied to Docker containers via
  `docker run --env-file .env`. Rotating them does not require rebuilding
  images — update `.env` (and GitHub Secrets for CI) and restart containers.
- Never store credentials in code, config files checked into git, or CI logs.
  Use environment variables at runtime. The Docker build uses no BuildKit
  secrets — the repository is public and source is fetched anonymously.
