#!/usr/bin/env bash
# =============================================================================
# Configure rclone for Cloudflare R2.
#
# Reads R2 credentials from the environment and creates an rclone remote
# named "r2". If the remote already exists it is overwritten.
#
# Prerequisites:
#   - rclone installed:  brew install rclone  (macOS)  /  apt install rclone  (Linux)
#   - R2 credentials exported in your shell. The easiest way:
#       set -a && source .env && set +a
#
# Required environment variables:
#   R2_ACCESS_KEY_ID       Cloudflare R2 API token access key
#   R2_SECRET_ACCESS_KEY   Cloudflare R2 API token secret
#   R2_ENDPOINT            R2 S3-compatible endpoint URL
#
# Usage:
#   set -a && source .env && set +a
#   bash scripts/setup-rclone.sh
# =============================================================================
set -euo pipefail

# --- Verify rclone is installed ---
if ! command -v rclone &>/dev/null; then
  echo "ERROR: rclone is not installed." >&2
  echo "" >&2
  echo "Install it with:" >&2
  echo "  macOS:  brew install rclone" >&2
  echo "  Linux:  sudo apt install rclone  (or https://rclone.org/install/)" >&2
  exit 1
fi

# --- Verify required environment variables ---
missing=()
for var in R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_ENDPOINT; do
  if [ -z "${!var:-}" ]; then
    missing+=("$var")
  fi
done

if [ ${#missing[@]} -gt 0 ]; then
  echo "ERROR: Missing required environment variables: ${missing[*]}" >&2
  echo "" >&2
  echo "Source your .env first:" >&2
  echo "  set -a && source .env && set +a" >&2
  exit 1
fi

# --- Create (or update) the rclone remote ---
rclone config create r2 s3 \
  provider Cloudflare \
  access_key_id "$R2_ACCESS_KEY_ID" \
  secret_access_key "$R2_SECRET_ACCESS_KEY" \
  endpoint "$R2_ENDPOINT" \
  no_check_bucket true

echo ""
echo "rclone remote 'r2' configured successfully."
echo "Verify with:  rclone lsd r2:${R2_BUCKET:-<your-bucket>}"
