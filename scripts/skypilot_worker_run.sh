#!/usr/bin/env bash
# SkyPilot worker run-block — shared by RunPod + OCI compute templates.
# Workaround for #735 — see configs/compute/runpod-template.yaml.
set -euo pipefail
cd /home/build/synth-setter
if [[ -n "${WORKER_GIT_REF:-}" ]]; then
  echo "Syncing worker checkout to git ref: $WORKER_GIT_REF"
  if ! [[ "$WORKER_GIT_REF" =~ ^[0-9a-f]{7,40}$ ]]; then
    echo "ERROR: WORKER_GIT_REF must be a 7-40 char hex git SHA, got: $WORKER_GIT_REF" >&2
    exit 1
  fi
  git config --global --add safe.directory /home/build/synth-setter
  git fetch --depth=1 origin -- "$WORKER_GIT_REF"
  git checkout FETCH_HEAD
  echo "Worker now at: $(git rev-parse HEAD)"
fi
python - <<'PY'
import os
from pipeline.entrypoints.generate_dataset import load_spec_from_uri, run
run(load_spec_from_uri(os.environ["WORKER_SPEC_URI"]))
print("forcing interpreter exit (#735 workaround) — bypassing atexit", flush=True)
os._exit(0)
PY
