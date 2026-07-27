#!/bin/bash
# Run the GPU test lane and return partial coverage through R2.

# Upload coverage without hiding test failures; arguments: none.
# Globals: COVERAGE_KEY, R2_BUCKET; Outputs: rclone stdout/stderr; Returns: exits.
upload_coverage() {
  local rc=$?
  trap - EXIT
  if [[ -s coverage.xml ]]; then
    rclone copyto coverage.xml "r2:${R2_BUCKET}/${COVERAGE_KEY}" --checksum || rc=$?
  fi
  exit "${rc}"
}

# Run CUDA, VST, and GPU pytest checks; arguments: none.
# Globals: COVERAGE_KEY, PATH, R2_BUCKET; Outputs: diagnostics; Returns: test status.
main() {
  set -euo pipefail

  : "${COVERAGE_KEY:?run-scoped R2 key to publish coverage.xml under}"
  : "${R2_BUCKET:?R2 bucket receiving the coverage artifact}"

  # The pinned mount avoids apt rclone's first-write failure; see #749.
  chmod u+x /tmp/synth-setter-tools/rclone
  export PATH="/tmp/synth-setter-tools:${PATH}"
  rclone version

  trap upload_coverage EXIT

  nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
  python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA not available"
assert torch.cuda.device_count() > 0
print("cuda:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
PY

  # Loading the plugin before pytest separates a broken VST host from a failing test.
  src/synth_setter/scripts/run-linux-vst-headless.sh \
    python -X faulthandler - <<'PY'
from synth_setter.data.vst.core import load_plugin

plugin = load_plugin("/usr/lib/vst3/Surge XT.vst3")
assert plugin is not None
print("Surge XT loaded")
PY

  src/synth_setter/scripts/run-linux-vst-headless.sh \
    pytest -vv -s -m gpu \
    --cov=src --cov-branch --cov-report=xml --cov-report=term
  [[ -s coverage.xml ]]
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
