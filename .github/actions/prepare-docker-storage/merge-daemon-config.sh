#!/bin/bash
# Atomically merge the CI Docker data root into a daemon configuration.

set -euo pipefail

daemon_json="${1:?usage: merge-daemon-config.sh DAEMON_JSON DATA_ROOT}"
data_root="${2:?usage: merge-daemon-config.sh DAEMON_JSON DATA_ROOT}"
daemon_tmp="$(mktemp "${daemon_json}.XXXXXX")"
trap 'rm -f "${daemon_tmp}"' EXIT

if [[ -f "${daemon_json}" ]]; then
  jq --arg data_root "${data_root}" '. + {"data-root": $data_root}' "${daemon_json}" >"${daemon_tmp}"
else
  jq -n --arg data_root "${data_root}" '{"data-root": $data_root}' >"${daemon_tmp}"
fi
mv "${daemon_tmp}" "${daemon_json}"
