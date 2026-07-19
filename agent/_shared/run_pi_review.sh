#!/bin/bash
# Launch the shared Pi-native full-review harness from Claude Code or Codex.
set -euo pipefail

readonly PI_REVIEW_MODEL="gpt-5.6-terra"
readonly PI_REVIEW_PROVIDER="openai-codex"
readonly PI_REVIEW_THINKING="medium"

usage() {
  echo \
    "usage: $0 <repo-review-full|repo-review-full-no-comments> [--target N]" \
    >&2
}

#######################################
# Validate a host request, then replace it with the shared Pi process.
# Arguments:
#   repo-review-full or repo-review-full-no-comments, optionally --target N.
# Outputs:
#   Writes usage or recursion diagnostics to stderr.
# Returns:
#   2 for invalid input; otherwise Pi's exit status through exec.
#######################################
main() {
  if [[ "${SYNTH_SETTER_PI_REVIEW:-}" == "1" ]]; then
    echo "run_pi_review.sh cannot be nested inside its Pi review session" >&2
    return 2
  fi
  if (( $# != 1 && $# != 3 )); then
    usage
    return 2
  fi

  local skill="${1}"
  case "${skill}" in
    repo-review-full | repo-review-full-no-comments) ;;
    *)
      usage
      return 2
      ;;
  esac

  local target_instruction="Resolve the target from the current branch."
  if (( $# == 3 )); then
    if [[ "${2}" != "--target" || ! "${3}" =~ ^[1-9][0-9]*$ ]]; then
      usage
      return 2
    fi
    target_instruction="Review PR #${3}."
  fi

  local prompt
  prompt="Execute ${skill} using its Pi-native execution path. ${target_instruction} \
The launcher set SYNTH_SETTER_PI_REVIEW=1; execute the skill in this session \
and do not invoke run_pi_review.sh again. Follow the skill exactly, use the \
absolute PI_REVIEW_AFTERCARE_MANIFEST path for any deferred-pass handoff, and \
return only the specified foreground deliverable."

  export SYNTH_SETTER_PI_REVIEW=1
  local aftercare_manifest review_root run_id transcript
  review_root="$(pwd)/.agent-reviews"
  run_id="$(date -u +%Y%m%dT%H%M%SZ).$$"
  transcript="${review_root}/pi-review-host.${run_id}.jsonl"
  aftercare_manifest="${review_root}/pi-review-aftercare.${run_id}.json"
  export PI_REVIEW_AFTERCARE_MANIFEST="${aftercare_manifest}"
  umask 077
  mkdir -p "${review_root}"
  echo "Live Pi transcript: ${transcript}" >&2
  local final_output
  if ! final_output="$(
    pi \
      -p \
      --approve \
      --mode json \
      --provider "${PI_REVIEW_PROVIDER}" \
      --model "${PI_REVIEW_MODEL}" \
      --thinking "${PI_REVIEW_THINKING}" \
      --no-session \
      "${prompt}" \
      | ./.venv/bin/python agent/_shared/pi_review_routing.py stream-host \
        --transcript "${transcript}"
  )"; then
    echo "Pi review host failed; inspect live transcript: ${transcript}" >&2
    return 1
  fi
  if [[ -s "${PI_REVIEW_AFTERCARE_MANIFEST}" ]]; then
    local aftercare_pid
    if aftercare_pid="$(
      ./.venv/bin/python agent/_shared/run_pi_review_aftercare.py \
        "${PI_REVIEW_AFTERCARE_MANIFEST}"
    )"; then
      echo \
        "Deferred Pi review aftercare: ${PI_REVIEW_AFTERCARE_MANIFEST} (PID ${aftercare_pid})" \
        >&2
    else
      echo \
        "Deferred Pi review aftercare failed to launch: ${PI_REVIEW_AFTERCARE_MANIFEST}" \
        >&2
    fi
  fi
  printf '%s\n' "${final_output}"
}

main "$@"
