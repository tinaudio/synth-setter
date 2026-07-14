#!/usr/bin/env bash
set -euo pipefail

ensure_worker_python() {
  local worker_venv="/venv/main"
  local worker_python="$worker_venv/bin/python"

  if [[ -n "${VIRTUAL_ENV:-}" && "$VIRTUAL_ENV" != "$worker_venv" ]]; then
    echo "ERROR: worker VIRTUAL_ENV must be $worker_venv (got $VIRTUAL_ENV)" >&2
    return 2
  fi

  if [[ -x "$worker_python" ]] &&
    "$worker_python" -c 'import sys; raise SystemExit(sys.version_info[:3] != (3, 12, 13))'; then
    export VIRTUAL_ENV="$worker_venv"
    export PATH="$worker_venv/bin:$PATH"
    return
  fi

  echo "Recreating $worker_venv with Python 3.12.13"
  rm -rf -- "$worker_venv"
  uv venv --python 3.12.13 "$worker_venv"
  export VIRTUAL_ENV="$worker_venv"
  export PATH="$worker_venv/bin:$PATH"
}

ensure_worker_python
