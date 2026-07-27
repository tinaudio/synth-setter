#!/usr/bin/env bash
# Operator SSH access (#2297): the launcher forwards the operator's public
# keys base64-encoded; decoding them in setup makes the pod reachable from
# the launching machine without RunPod account-key or provisioning-key access.
set -euo pipefail
if [ -n "${OPERATOR_SSH_PUBKEYS_B64:-}" ]; then
  mkdir -p ~/.ssh
  printf '%s' "${OPERATOR_SSH_PUBKEYS_B64}" | base64 -d >> ~/.ssh/authorized_keys
  chmod 600 ~/.ssh/authorized_keys
fi
echo "synth-setter worker ready (host: $(hostname))"
