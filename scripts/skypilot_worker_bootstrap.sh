#!/usr/bin/env bash
# SkyPilot worker bootstrap — shared by RunPod + OCI compute templates.
# Caller (template) must set $WORKER_GIT_REF to the PR head SHA when
# bypassing the dev-snapshot image-bake lag (PR CI), or leave it empty
# to run the baked code as-is (operators).
#
# After bootstrap, exec scripts/skypilot_worker_run.sh which owns the
# python invocation + #735 os._exit(0) workaround.
set -euo pipefail

# Self-locate the synth-setter checkout from this script's path. Avoids
# hardcoding /home/build/synth-setter in two places (here + Dockerfile WORKDIR);
# if the image's WORKDIR ever moves, the script still works as long as it
# lives at <repo-root>/scripts/.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [[ -n "${WORKER_GIT_REF:-}" ]]; then
  echo "Syncing worker checkout to git ref: $WORKER_GIT_REF"
  if ! [[ "$WORKER_GIT_REF" =~ ^[0-9a-f]{7,40}$ ]]; then
    echo "ERROR: WORKER_GIT_REF must be a 7-40 char hex git SHA, got: $WORKER_GIT_REF" >&2
    exit 1
  fi
  # Idempotent: only add safe.directory if not already present, so repeated
  # worker starts on a long-lived host don't bloat ~/.gitconfig.
  git config --global --get-all safe.directory | grep -qxF "$REPO_DIR" \
    || git config --global --add safe.directory "$REPO_DIR"
  git fetch --depth=1 origin -- "$WORKER_GIT_REF"
  git checkout FETCH_HEAD
  echo "Worker now at: $(git rev-parse HEAD)"
fi
exec bash "$REPO_DIR/scripts/skypilot_worker_run.sh"
