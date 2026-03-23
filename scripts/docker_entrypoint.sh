#!/usr/bin/env bash
# Minimal Docker entrypoint — passthrough to the container command.
# The full-featured entrypoint (MODE dispatch, R2 upload, etc.) lives on the
# experiment branch. This stub exists so the Dockerfile COPY succeeds and
# the prod/dev-snapshot targets build without error.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "Error: no command provided to docker_entrypoint." >&2
  echo "Usage: docker run <image> <command> [args...]" >&2
  exit 1
fi

exec "$@"
