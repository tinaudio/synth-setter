#!/usr/bin/env bash
echo "test from headless-rclone variant" > /tmp/headless-rclone-test.txt
bash src/synth_setter/scripts/run-linux-vst-headless.sh rclone copy --checksum \
  /tmp/headless-rclone-test.txt \
  "r2:${R2_BUCKET}/${R2_DEBUG_PREFIX}headless-rclone-test.txt"
echo "skypilot-debug headless-rclone job done (host: $(hostname))"
