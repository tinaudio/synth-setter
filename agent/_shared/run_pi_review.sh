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

resolve_review_python() {
  if [[ -x ./.venv/bin/python ]]; then
    printf '%s\n' ./.venv/bin/python
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  echo "run_pi_review.sh requires either ./.venv/bin/python or python3 on PATH" >&2
  return 1
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

  local review_python
  review_python="$(resolve_review_python)"
  export PI_REVIEW_PYTHON="${review_python}"

  local target_instruction="Resolve the target from the current branch."
  if (( $# == 3 )); then
    if [[ "${2}" != "--target" || ! "${3}" =~ ^[1-9][0-9]*$ ]]; then
      usage
      return 2
    fi
    target_instruction="Review PR #${3}."
  fi

  if [[ "${skill}" == "repo-review-full-no-comments" && $# == 1 ]]; then
    local branch open_pr_number
    branch="$(git branch --show-current)"
    if ! open_pr_number="$(
      gh pr list --state open --head "${branch}" --limit 2 --json number --jq '.[].number'
    )"; then
      echo "Unable to resolve whether the current branch has an open PR." >&2
      return 2
    fi
    if [[ -n "${open_pr_number}" ]]; then
      if [[ ! "${open_pr_number}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Open PR lookup returned an ambiguous result." >&2
        return 2
      fi
      target_instruction="Review PR #${open_pr_number}."
    else
      local attempt claim_output limit
      if claim_output="$(
        "${review_python}" agent/_shared/review_sentinel.py claim "${branch}"
      )"; then
        read -r attempt limit <<<"${claim_output}"
        echo "Pre-PR sentinel review attempt ${attempt}/${limit}." >&2
      else
        local claim_status=$?
        if (( claim_status == 3 )); then
          limit="${claim_output}"
          echo "Pre-PR sentinel review limit reached after ${limit} attempts." >&2
          echo \
            "Refusing another repo-review-full-no-comments run. Open the PR and continue with /repo-review-full so the public GitHub review bot can review subsequent changes." \
            >&2
          return 2
        fi
        echo "Unable to claim a pre-PR sentinel review attempt." >&2
        return 2
      fi
    fi
  fi

  local prompt
  prompt="Execute ${skill} using its Pi-native execution path. ${target_instruction} \
The launcher set SYNTH_SETTER_PI_REVIEW=1; execute the skill in this session \
and do not invoke run_pi_review.sh again. Follow the skill exactly, use the \
absolute PI_REVIEW_FOLLOW_UP_MANIFEST path for any deferred-pass handoff, and \
return only the specified foreground deliverable."

  export SYNTH_SETTER_PI_REVIEW=1
  local follow_up_manifest review_root run_id transcript
  review_root="$(pwd)/.agent-reviews"
  run_id="$(date -u +%Y%m%dT%H%M%SZ).$$"
  transcript="${review_root}/pi-review-host.${run_id}.jsonl"
  follow_up_manifest="${review_root}/pi-review-follow-up.${run_id}.json"
  export PI_REVIEW_FOLLOW_UP_MANIFEST="${follow_up_manifest}"
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
      | "${review_python}" agent/_shared/pi_review_routing.py stream-host \
        --transcript "${transcript}"
  )"; then
    echo "Pi review host failed; inspect live transcript: ${transcript}" >&2
    return 1
  fi
  if [[ -s "${PI_REVIEW_FOLLOW_UP_MANIFEST}" ]]; then
    if [[ "${CI:-}" == "true" ]]; then
      if ! "${review_python}" agent/_shared/run_pi_review_follow_up.py \
        --supervise "${PI_REVIEW_FOLLOW_UP_MANIFEST}"; then
        echo \
          "Synchronous Pi review follow-up failed: ${PI_REVIEW_FOLLOW_UP_MANIFEST}" \
          >&2
        return 1
      fi
      echo \
        "Synchronous Pi review follow-up completed: ${PI_REVIEW_FOLLOW_UP_MANIFEST}" \
        >&2
    else
      local follow_up_pid
      if follow_up_pid="$(
        "${review_python}" agent/_shared/run_pi_review_follow_up.py \
          "${PI_REVIEW_FOLLOW_UP_MANIFEST}"
      )"; then
        echo \
          "Deferred Pi review follow-up: ${PI_REVIEW_FOLLOW_UP_MANIFEST} (PID ${follow_up_pid})" \
          >&2
      else
        echo \
          "Deferred Pi review follow-up failed to launch: ${PI_REVIEW_FOLLOW_UP_MANIFEST}" \
          >&2
      fi
    fi
  fi
  printf '%s\n' "${final_output}"
}

main "$@"
