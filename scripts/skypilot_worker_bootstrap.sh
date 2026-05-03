#!/usr/bin/env bash
# SkyPilot worker bootstrap — shared by RunPod + OCI compute templates.
# Caller (template) must set $WORKER_GIT_REF to the PR head SHA when
# bypassing the dev-snapshot image-bake lag (PR CI), or leave it empty
# to run the baked code as-is (operators).
#
# After bootstrap, exec scripts/skypilot_worker_run.sh which owns the
# python invocation + #735 os._exit(0) workaround.
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
exec bash scripts/skypilot_worker_run.sh
