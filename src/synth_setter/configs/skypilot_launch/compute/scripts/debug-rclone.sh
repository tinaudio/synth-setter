#!/usr/bin/env bash
set -euo pipefail
printf "%s\n" "$(hostname) $(date -u +%FT%TZ)" > /tmp/skypilot-debug-rclone-sentinel.txt
rclone copy \
  -v \
  --checksum \
  --contimeout=30s \
  --timeout=300s \
  --retries=3 \
  /tmp/skypilot-debug-rclone-sentinel.txt \
  "r2:${R2_BUCKET}/${R2_DEBUG_PREFIX}"
echo "skypilot-debug variant=rclone done (host: $(hostname))"
