#!/bin/bash
# Submit every runnable 1M-step 440k conditioning arm as a managed RunPod job.

set -euo pipefail

readonly LAUNCH_CONFIG="src/synth_setter/configs/launch/train-runpod-flow-simple-440k-1m.yaml"
readonly -a ARMS=(
  clap
  clap_online
  log_mel
  m2l
  matpac_plus
  mel
  same_l
  same_l_online
  same_s
  same_s_online
  ssondo
)

usage() {
  echo "Usage: $0 [--execute]" >&2
}

main() {
  local execute=false
  local repo_root
  local status=0

  if (( $# > 1 )); then
    usage
    return 2
  fi
  if (( $# == 1 )); then
    if [[ "$1" != "--execute" ]]; then
      usage
      return 2
    fi
    execute=true
  fi

  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
  cd "${repo_root}"

  unset SKYPILOT_API_SERVER_ENDPOINT
  unset SKYPILOT_API_SERVER_KEY
  unset SKYPILOT_API_SERVER_TOKEN

  local arm
  local experiment
  for arm in "${ARMS[@]}"; do
    experiment="surge/flow_simple_440k_1m_${arm}"
    if [[ "${execute}" == false ]]; then
      echo "DRY RUN: ${experiment}"
      continue
    fi

    if ! uv run synth-setter-skypilot-launch \
      --extra-env EXPERIMENT "${experiment}" \
      "${LAUNCH_CONFIG}"; then
      echo "Failed to submit ${experiment}" >&2
      status=1
    fi
  done

  return "${status}"
}

main "$@"
