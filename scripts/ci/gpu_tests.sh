#!/bin/bash
# Run the GPU test lane and return partial coverage through R2.
set -euo pipefail

: "${COVERAGE_KEY:?run-scoped R2 key to publish coverage.xml under}"
: "${R2_BUCKET:?R2 bucket receiving the coverage artifact}"

# The pinned mount avoids apt rclone's first-write failure; see #749.
chmod u+x /tmp/synth-setter-tools/rclone
export PATH="/tmp/synth-setter-tools:${PATH}"
rclone version

# Upload coverage without hiding test failures.
# Globals: COVERAGE_KEY, R2_BUCKET; Arguments: none; Returns: does not return.
upload_coverage() {
  local rc=$?
  trap - EXIT
  if [[ -s coverage.xml ]]; then
    rclone copyto coverage.xml "r2:${R2_BUCKET}/${COVERAGE_KEY}" --checksum || rc=$?
  fi
  exit "${rc}"
}
trap upload_coverage EXIT

nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
python -c 'import torch; assert torch.cuda.is_available(), "CUDA not available"; assert torch.cuda.device_count() > 0; print("cuda:", torch.cuda.is_available(), "count:", torch.cuda.device_count())'

# Loading the plugin before pytest separates a broken VST host from a failing test.
src/synth_setter/scripts/run-linux-vst-headless.sh python -X faulthandler -c 'from synth_setter.data.vst.core import load_plugin; plugin = load_plugin("/usr/lib/vst3/Surge XT.vst3"); assert plugin is not None; print("Surge XT loaded")'

src/synth_setter/scripts/run-linux-vst-headless.sh pytest -vv -s -m gpu --cov=src --cov-branch --cov-report=xml --cov-report=term
[[ -s coverage.xml ]]
