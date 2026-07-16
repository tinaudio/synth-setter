#!/bin/bash
# Atomically merge the CI Docker data root into a daemon configuration.

set -euo pipefail

daemon_json="${1:?usage: merge-daemon-config.sh DAEMON_JSON}"
daemon_tmp="$(mktemp "${daemon_json}.XXXXXX")"
trap 'rm -f "${daemon_tmp}"' EXIT

if [[ -f "${daemon_json}" ]]; then
  jq '. + {"data-root": "/mnt/docker"}' "${daemon_json}" >"${daemon_tmp}"
else
  jq -n '{"data-root": "/mnt/docker"}' >"${daemon_tmp}"
fi
mv "${daemon_tmp}" "${daemon_json}"
