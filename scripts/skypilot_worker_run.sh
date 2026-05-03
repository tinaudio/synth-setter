#!/usr/bin/env bash
# SkyPilot worker run-block — shared by RunPod + OCI compute templates.
# Sole reusable entry point for the worker side: optional checkout to
# WORKER_GIT_REF, then the python invocation with the #735 os._exit(0)
# workaround. Templates bootstrap us via `bash <(git show <ref>:scripts/skypilot_worker_run.sh)`
# until the dev-snapshot image gets rebuilt with this script baked in.
set -euo pipefail
cd /home/build/synth-setter
if [[ -n "${WORKER_GIT_REF:-}" ]]; then
  echo "Syncing worker checkout to git ref: $WORKER_GIT_REF"
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
