#!/usr/bin/env bash
# Sync the worker's /home/build/synth-setter checkout to $WORKER_GIT_REF.
# Shared by RunPod + OCI compute templates.
#
# WORKER_GIT_REF format is validated by the caller (templates) before we
# even reach this script — by then it's already used in the curl URL the
# templates use to bootstrap us into the not-yet-rebuilt dev-snapshot
# image.
set -euo pipefail
cd /home/build/synth-setter
echo "Syncing worker checkout to git ref: $WORKER_GIT_REF"
git config --global --add safe.directory /home/build/synth-setter
git fetch --depth=1 origin -- "$WORKER_GIT_REF"
git checkout FETCH_HEAD
echo "Worker now at: $(git rev-parse HEAD)"
