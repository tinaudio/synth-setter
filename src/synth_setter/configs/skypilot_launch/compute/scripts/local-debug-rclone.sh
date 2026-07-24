#!/usr/bin/env bash
set -euo pipefail
echo "skypilot-local-debug worker ready (host: $(hostname))"
echo "[debug] R2 endpoint: ${RCLONE_CONFIG_R2_ENDPOINT}"
echo "[debug] R2 bucket:   ${R2_BUCKET}"
echo "[debug] R2 prefix:   ${R2_DEBUG_PREFIX}"

printf "%s\n" "$(hostname) $(date -u +%FT%TZ)" > /tmp/skypilot-local-debug-sentinel.txt
rclone copy \
  -v \
  --checksum \
  --contimeout=30s \
  --timeout=300s \
  --retries=3 \
  /tmp/skypilot-local-debug-sentinel.txt \
  "r2:${R2_BUCKET}/${R2_DEBUG_PREFIX}"
rclone lsf "r2:${R2_BUCKET}/${R2_DEBUG_PREFIX}"
echo "skypilot-local-debug variant=rclone done (host: $(hostname))"
