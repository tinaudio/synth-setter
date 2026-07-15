#!/bin/bash
set -euo pipefail

#######################################
# Ensure a worker venv uses the canonical Python, recreating it when stale.
# Globals:
#   PATH: Modified to prioritize the worker venv.
#   SYNTH_SETTER_WORKER_PYTHON_RECREATED: Reports whether dependencies need reinstalling.
#   VIRTUAL_ENV: Validated and set to the worker venv.
# Arguments:
#   $1: Worker venv path; defaults to /venv/main.
# Outputs:
#   Writes recreation status to stdout and invalid-path errors to stderr.
# Returns:
#   0 on success, 2 for an unexpected VIRTUAL_ENV, or a dependency error.
#######################################
ensure_worker_python() {
  local worker_venv="${1:-/venv/main}"
  local worker_python="$worker_venv/bin/python"
  export SYNTH_SETTER_WORKER_PYTHON_RECREATED=0

  if [[ "$worker_venv" != /* || "$worker_venv" == "/" ||
    "$worker_venv" == *"/../"* || "$worker_venv" == *"/.." ]]; then
    echo "ERROR: worker venv path must be absolute and normalized (got $worker_venv)" >&2
    return 2
  fi

  if [[ -n "${VIRTUAL_ENV:-}" && "$VIRTUAL_ENV" != "$worker_venv" ]]; then
    echo "ERROR: worker VIRTUAL_ENV must be $worker_venv (got $VIRTUAL_ENV)" >&2
    return 2
  fi

  if [[ ! -x "$worker_python" ]] ||
    ! "$worker_python" -c 'import sys; raise SystemExit(sys.version_info[:3] != (3, 12, 13))'; then
    echo "Recreating $worker_venv with Python 3.12.13"
    rm -rf -- "$worker_venv"
    uv venv --python 3.12.13 "$worker_venv"
    export SYNTH_SETTER_WORKER_PYTHON_RECREATED=1
  fi

  export VIRTUAL_ENV="$worker_venv"
  export PATH="$worker_venv/bin:$PATH"
}

ensure_worker_python "${1:-}"
