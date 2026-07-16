#!/bin/bash
# Launch a project-pinned Codex PR-review role.
set -euo pipefail

readonly ORCHESTRATOR_TIMEOUT_S=900
readonly TERMINATION_GRACE_S=2
readonly WORKER_TIMEOUT_S=600

# Terminate a review process group after its deadline, escalating to SIGKILL.
# Arguments:
#   Process group ID, timeout seconds, and timeout marker path.
# Returns:
#   Zero after cancellation, target exit, or forced termination.
terminate_after() {
  local target_pgid="$1"
  local timeout_s="$2"
  local timeout_marker="$3"
  local sleeper_pid=""

  trap '
    if [[ -n "${sleeper_pid}" ]]; then
      kill "${sleeper_pid}" 2>/dev/null || true
      wait "${sleeper_pid}" 2>/dev/null || true
    fi
    exit 0
  ' TERM
  sleep "${timeout_s}" &
  sleeper_pid=$!
  wait "${sleeper_pid}" || return 0
  sleeper_pid=""

  printf 'timed out\n' >"${timeout_marker}"
  kill -TERM -- "-${target_pgid}" 2>/dev/null || return 0
  sleep "${TERMINATION_GRACE_S}" &
  sleeper_pid=$!
  wait "${sleeper_pid}" || return 0
  kill -KILL -- "-${target_pgid}" 2>/dev/null || true
}

repo_root="$(git rev-parse --show-toplevel)"
launcher="${repo_root}/agent/_shared/run_codex_review_agent.py"
resolved="$(uv run --no-project --script "${launcher}" "$@")"
dry_run_filter='if (.dry_run | type) == "boolean" then
  (.dry_run | tostring)
else
  error("invalid dry_run")
end'
dry_run="$(jq -er "${dry_run_filter}" <<<"${resolved}")"

if [[ ${dry_run} == true ]]; then
  printf '%s\n' "${resolved}"
  exit 0
fi

command_filter='.command | if type == "array"
  and length > 0
  and all(.[]; type == "string")
then
  .[]
else
  error("invalid command")
end'
command_lines="$(jq -er "${command_filter}" <<<"${resolved}")"
# `read` (not `mapfile`) so macOS bash 3.2 works. Each line is one argv element.
command=()
while IFS= read -r line; do
  command+=("${line}")
done <<<"${command_lines}"
prompt_filter='if (.prompt | type) == "string" then
  .prompt
else
  error("invalid prompt")
end'
prompt="$(jq -er "${prompt_filter}" <<<"${resolved}")"
default_timeout_s="${WORKER_TIMEOUT_S}"
if [[ "${1:-}" == "pr-review-orchestrator" ]]; then
  default_timeout_s="${ORCHESTRATOR_TIMEOUT_S}"
fi
timeout_s="${CODEX_REVIEW_TIMEOUT:-${default_timeout_s}}"
if [[ ! "${timeout_s}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CODEX_REVIEW_TIMEOUT must be a positive integer, got: ${timeout_s}" >&2
  exit 1
fi
# codex exec appends piped stdin to the prompt and blocks until EOF (v0.144.4).
event_file="$(mktemp)"
log_file="$(mktemp)"
timeout_marker="$(mktemp)"
trap 'rm -f "${event_file}" "${log_file}" "${timeout_marker}"' EXIT
set -m
"${command[@]}" --json "${prompt}" \
  </dev/null >"${event_file}" 2>"${log_file}" &
run_pid=$!
set +m
terminate_after "${run_pid}" "${timeout_s}" "${timeout_marker}" &
watchdog_pid=$!
status=0
wait "${run_pid}" || status=$?
kill "${watchdog_pid}" >/dev/null 2>&1 || true
wait "${watchdog_pid}" 2>/dev/null || true
if [[ "${status}" -ne 0 ]]; then
  cat "${log_file}" >&2
  if [[ -s "${timeout_marker}" ]]; then
    echo "codex exec timed out after ${timeout_s}s" >&2
  fi
  exit 1
fi
report_filter='[
  .[]
  | select(.type == "item.completed")
  | select(.item.type == "agent_message")
  | .item.text
] | last // error("missing final agent message")'
if ! report="$(jq -ers "${report_filter}" "${event_file}")"; then
  cat "${log_file}" >&2
  exit 1
fi
printf '%s' "${report}"
