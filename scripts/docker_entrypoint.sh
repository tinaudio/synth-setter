#!/usr/bin/env bash
# Minimal Docker entrypoint — passthrough to the container command.
# The full-featured entrypoint (MODE dispatch, R2 upload, etc.) lives on the
# experiment branch. This stub exists so the Dockerfile COPY succeeds and
# the prod/dev-snapshot targets build without error.
set -euo pipefail
exec "$@"
