#!/usr/bin/env bash
# Ensure the worker runtime, then sync to $WORKER_GIT_REF when set.
#
# Used by RunPod and OCI compute templates to bypass dev-snapshot image-bake
# lag in PR CI: the launcher passes the PR head SHA via WORKER_GIT_REF and the
# worker fetches+checks out that ref over the image's pre-baked checkout.
#
# Caller is responsible for `cd`-ing into the synth-setter checkout. The SHA
# format is validated host-side by skypilot_launch.resolve_worker_env.
set -euo pipefail

python_ready=false
if [[ "${1:-}" == "--python-ready" ]]; then
  python_ready=true
  shift
fi
if (( $# > 0 )); then
  echo "Usage: $0 [--python-ready]" >&2
  exit 2
fi

if [[ "$python_ready" == "false" ]]; then
  # shellcheck source=scripts/ensure_worker_python.sh
  source scripts/ensure_worker_python.sh
fi

if [[ -z "${WORKER_GIT_REF:-}" ]]; then
  if [[ "$python_ready" == "false" &&
    "${SYNTH_SETTER_WORKER_PYTHON_RECREATED:-0}" == "1" ]]; then
    uv pip install --group runtime -e .
  fi
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
# shellcheck source=scripts/ensure_worker_python.sh
source scripts/ensure_worker_python.sh
# Heavy runtime lives in PEP 735 groups since #1139; `--group runtime` pulls the
# full stack (torch + everything). This activates neither the cpu nor cu128
# extra that [tool.uv.sources] keys torch's index routing on, so torch resolves
# from PyPI on this worker.
uv pip install --group runtime -e .
