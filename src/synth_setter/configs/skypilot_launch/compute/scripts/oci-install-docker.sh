#!/usr/bin/env bash
# Installs docker.io on the stock OCI Ubuntu VM (no docker preinstalled in
# skypilot:cpu-ubuntu-2204) and pre-pulls the worker image so run: stays fast
# and pull errors surface in setup: logs.
#
# Lock-waiting is two-stage because `cloud-init status --wait` returns "done"
# before apt-daily / unattended-upgrades release /var/lib/dpkg/lock-frontend
# on stock OCI Ubuntu images — see #776 follow-up.
set -euo pipefail
: "${WORKER_IMAGE:?WORKER_IMAGE must be set by the launcher (--worker-image-tag); refusing to docker pull an unset image}"
if command -v cloud-init >/dev/null 2>&1; then
  echo "waiting for cloud-init to finish (cap 5min)..."
  sudo timeout 300 cloud-init status --wait || true
fi
echo "waiting for apt/dpkg locks to release (cap 10min)..."
deadline=$((SECONDS + 600))
while sudo fuser /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock /var/lib/dpkg/lock >/dev/null 2>&1; do
  if (( SECONDS > deadline )); then
    echo "apt/dpkg locks still held after 10min; aborting" >&2
    exit 1
  fi
  sleep 5
done
if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get -o DPkg::Lock::Timeout=300 update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300 install -y -qq docker.io
fi
sudo systemctl enable --now docker
sudo -E docker pull "$WORKER_IMAGE"
echo "synth-setter worker ready (host: $(hostname), docker: $(sudo -E docker --version))"
