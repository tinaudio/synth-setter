#!/usr/bin/env bash
# Launches the worker container ourselves (OCI backend can't ingest a docker
# image_id). --privileged + --ulimit nofile is the expedient hammer for the
# Xvfb/dbus stack — minimal-cap follow-up: #776. No bind-mount: image is
# source of truth; the substituted ${WORKER_CMD} runs sync_worker_checkout.sh
# inside the container for the PR-CI bake-lag bypass.
set -euo pipefail
: "${WORKER_IMAGE:?WORKER_IMAGE must be set by the launcher; refusing to docker run an unset image}"
echo "skypilot-run-cwd: $(pwd) host: $(hostname)"
sudo -E docker run --rm \
  --privileged \
  --ulimit nofile=65536:65536 \
  -e RCLONE_CONFIG_R2_TYPE \
  -e RCLONE_CONFIG_R2_PROVIDER \
  -e RCLONE_CONFIG_R2_ACCESS_KEY_ID \
  -e RCLONE_CONFIG_R2_SECRET_ACCESS_KEY \
  -e RCLONE_CONFIG_R2_ENDPOINT \
  -e WORKER_GIT_REF \
  -e SYNTH_SETTER_WORKER_RANK \
  -e SYNTH_SETTER_NUM_WORKERS \
  "$WORKER_IMAGE" \
  bash -c "${WORKER_CMD}"
