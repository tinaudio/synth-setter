#!/usr/bin/env bash
# Sync the worker's checkout to $WORKER_GIT_REF if set; otherwise no-op.
#
# Used by RunPod and OCI compute templates to bypass dev-snapshot image-bake
# lag in PR CI: the launcher passes the PR head SHA via WORKER_GIT_REF and the
# worker fetches+checks out that ref over the image's pre-baked checkout.
#
# Caller is responsible for `cd`-ing into the synth-setter checkout. The SHA
# format is validated host-side by skypilot_launch.resolve_worker_env.
set -euo pipefail

if [[ -z "${WORKER_GIT_REF:-}" ]]; then
  exit 0
fi

echo "Syncing worker checkout to git ref: $WORKER_GIT_REF"
repo_dir="$(pwd)"
if ! git config --global --get-all safe.directory | grep -Fxq -- "$repo_dir"; then
  git config --global --add safe.directory "$repo_dir"
fi
git fetch --depth=1 origin -- "$WORKER_GIT_REF"
git checkout FETCH_HEAD
echo "Worker now at: $(git rev-parse HEAD)"
uv pip install -r requirements.txt
