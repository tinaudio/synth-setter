#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
python "$here/baseline_app.py" \
  "host=${INJECTED_HOST_NAME:-localhost}" \
  "task_id=${SGE_TASK_ID:-0}"
