#!/bin/bash
# Launch a project-pinned Codex PR-review role.
set -euo pipefail

readonly ORCHESTRATOR_TIMEOUT_S=900
readonly WORKER_TIMEOUT_S=600

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
event_file="$(mktemp)"
log_file="$(mktemp)"
trap 'rm -f "${event_file}" "${log_file}"' EXIT
default_timeout_s="${WORKER_TIMEOUT_S}"
if [[ "${1:-}" == "pr-review-orchestrator" ]]; then
  default_timeout_s="${ORCHESTRATOR_TIMEOUT_S}"
fi
timeout_s="${CODEX_REVIEW_TIMEOUT:-${default_timeout_s}}"
# codex exec appends piped stdin to the prompt and blocks until EOF (v0.144.4).
"${command[@]}" --json "${prompt}" \
  </dev/null >"${event_file}" 2>"${log_file}" &
run_pid=$!
# macOS ships no coreutils `timeout`; a watchdog subshell bounds hung runs.
(sleep "${timeout_s}" && kill "${run_pid}") >/dev/null 2>&1 &
watchdog_pid=$!
status=0
wait "${run_pid}" || status=$?
kill "${watchdog_pid}" >/dev/null 2>&1 || true
wait "${watchdog_pid}" 2>/dev/null || true
if [[ "${status}" -ne 0 ]]; then
  cat "${log_file}" >&2
  if [[ "${status}" -eq 143 ]]; then
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
