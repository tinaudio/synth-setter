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

# Validate the host request before replacing it with the shared Pi process.
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
    if [[ "${2}" != "--target" || ! "${3}" =~ ^[0-9]+$ ]]; then
      usage
      return 2
    fi
    target_instruction="Review PR #${3}."
  fi

  local prompt
  prompt="Execute ${skill} using its Pi-native execution path. ${target_instruction} \
The launcher set SYNTH_SETTER_PI_REVIEW=1; execute the skill in this session \
and do not invoke run_pi_review.sh again. Follow the skill exactly and return \
only its specified deliverable."

  export SYNTH_SETTER_PI_REVIEW=1
  exec pi \
    -p \
    --approve \
    --provider "${PI_REVIEW_PROVIDER}" \
    --model "${PI_REVIEW_MODEL}" \
    --thinking "${PI_REVIEW_THINKING}" \
    --no-session \
    "${prompt}"
}

main "$@"
