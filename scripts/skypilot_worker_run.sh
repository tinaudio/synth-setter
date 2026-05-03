#!/usr/bin/env bash
# Worker python entrypoint shared by RunPod + OCI compute templates.
# os._exit(0) workaround for #735 — see configs/compute/runpod-template.yaml.
# Templates do `cd /home/build/synth-setter` + WORKER_GIT_REF checkout
# before invoking this script (the checkout is what makes this file
# exist on a not-yet-rebuilt dev-snapshot image).
set -euo pipefail
python - <<'PY'
import os
from pipeline.entrypoints.generate_dataset import load_spec_from_uri, run
run(load_spec_from_uri(os.environ["WORKER_SPEC_URI"]))
print("forcing interpreter exit (#735 workaround) — bypassing atexit", flush=True)
os._exit(0)
PY
