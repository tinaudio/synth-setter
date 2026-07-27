#!/usr/bin/env bash
# Worker body for the GPU test lane (src/synth_setter/configs/launch/gpu-tests-runpod.yaml).
#
# Runs in the dev-snapshot image after sync_worker_checkout.sh pins the checkout
# to WORKER_GIT_REF, so the image supplies CUDA and Surge XT while this script
# comes from the dispatched commit.
set -euo pipefail

: "${COVERAGE_KEY:?run-scoped R2 key to publish coverage.xml under}"
: "${R2_BUCKET:?R2 bucket receiving the coverage artifact}"

# Ahead of the trap so the upload can never fall back to the image's apt rclone,
# which fails the first R2 write of every process. file_mounts drop the exec bit.
chmod u+x /tmp/synth-setter-tools/rclone
export PATH="/tmp/synth-setter-tools:$PATH"
rclone version

# Coverage ships from an EXIT trap so a pytest failure still returns partial
# results before the managed job tears the pod down.
upload_coverage() {
  rc=$?
  trap - EXIT
  if [[ -s coverage.xml ]]; then
    rclone copyto coverage.xml "r2:${R2_BUCKET}/${COVERAGE_KEY}" --checksum || rc=$?
  fi
  exit "$rc"
}
trap upload_coverage EXIT

nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
python -c 'import torch; assert torch.cuda.is_available(), "CUDA not available"; assert torch.cuda.device_count() > 0; print("cuda:", torch.cuda.is_available(), "count:", torch.cuda.device_count())'

# Loading the plugin before pytest separates a broken VST host from a failing test.
src/synth_setter/scripts/run-linux-vst-headless.sh python -X faulthandler -c 'from synth_setter.data.vst.core import load_plugin; plugin = load_plugin("/usr/lib/vst3/Surge XT.vst3"); assert plugin is not None; print("Surge XT loaded")'

src/synth_setter/scripts/run-linux-vst-headless.sh pytest -vv -s -m gpu --cov=src --cov-branch --cov-report=xml --cov-report=term
test -s coverage.xml
