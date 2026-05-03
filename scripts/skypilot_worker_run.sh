#!/usr/bin/env bash
# Worker python entrypoint shared by RunPod + OCI compute templates.
# Caller (template) is responsible for cd into the synth-setter checkout
# and (optionally) syncing it to WORKER_GIT_REF before invoking this.
# Workaround for #735.
set -euo pipefail
python - <<'PY'
import os
from pipeline.entrypoints.generate_dataset import load_spec_from_uri, run
run(load_spec_from_uri(os.environ["WORKER_SPEC_URI"]))
print("forcing interpreter exit (#735 workaround) — bypassing atexit", flush=True)
os._exit(0)
PY
